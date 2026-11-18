from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def pack_observations_padded(
    X: np.ndarray,
    obs_group_ids: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Pack contiguous observation-grouped candidate rows (N_rows, F) into padded
    observation tensors (N_obs, max_k, F) for listwise set-aware training & inference.
    """
    change_pts = np.flatnonzero(np.diff(obs_group_ids)) + 1
    starts = np.concatenate([[0], change_pts])
    counts = np.diff(np.append(starts, len(obs_group_ids)))
    n_obs = len(starts)
    max_k = int(counts.max()) if n_obs > 0 else 1
    n_feat = X.shape[1]

    row_obs_idx = np.repeat(np.arange(n_obs), counts)
    row_slot_idx = np.arange(len(obs_group_ids)) - np.repeat(starts, counts)

    X_pad = np.zeros((n_obs, max_k, n_feat), dtype=np.float32)
    valid_mask = np.zeros((n_obs, max_k), dtype=bool)

    X_pad[row_obs_idx, row_slot_idx] = X.astype(np.float32)
    valid_mask[row_obs_idx, row_slot_idx] = True

    y_pad = None
    if y is not None:
        y_pad = np.zeros((n_obs, max_k), dtype=np.float32)
        y_pad[row_obs_idx, row_slot_idx] = y.astype(np.float32)

    return X_pad, valid_mask, row_obs_idx, row_slot_idx, y_pad


class CandidateSetResNet(nn.Module):
    """
    Set-Aware Residual Neural Network for candidate road-segment scoring.
    Each candidate representation is enriched with masked mean/max context across
    all competing candidates in the same GPS observation before scoring.
    """

    def __init__(self, in_features: int = 28, hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Combines candidate embedding with observation-level mean & max competition context
        self.set_context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, F) candidate feature tensor
            valid_mask: (B, K) boolean mask of valid candidates per observation
        Returns:
            logits: (B, K) raw candidate logits
        """
        h = self.input_proj(x)
        h = F.silu(h + self.res_block1(h))

        mask_f = valid_mask.unsqueeze(-1).float()  # (B, K, 1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_ctx = (h * mask_f).sum(dim=1, keepdim=True) / denom  # (B, 1, D)

        h_masked_max = h.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        max_ctx = h_masked_max.max(dim=1, keepdim=True).values  # (B, 1, D)
        max_ctx = torch.where(valid_mask.any(dim=1, keepdim=True).unsqueeze(-1), max_ctx, torch.zeros_like(max_ctx))

        mean_exp = mean_ctx.expand_as(h)
        max_exp = max_ctx.expand_as(h)
        h_ctx = torch.cat([h, h - mean_exp, max_exp], dim=-1)
        h = h + self.set_context_proj(h_ctx)

        logits = self.head(h).squeeze(-1)
        return logits


class PyTorchCandidateSetWrapper:
    """
    Scikit-learn-compatible wrapper for CandidateSetResNet trained with a hybrid
    Listwise Observation Softmax Cross-Entropy + Binary Calibration objective.
    """

    def __init__(
        self,
        in_features: int = 28,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        epochs: int = 16,
        listwise_weight: float = 0.70,
        prob_blend_alpha: float = 0.50,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.listwise_weight = listwise_weight
        self.prob_blend_alpha = prob_blend_alpha
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CandidateSetResNet] = None
        self.history_: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": []}

    @staticmethod
    def _infer_obs_groups_from_features(X: np.ndarray) -> np.ndarray:
        """
        Reconstruct contiguous observation IDs from standardized features when obs_group_ids
        is not explicitly passed (using local drops in feature 22: dist_rank).
        """
        ranks = X[:, 22]
        starts = np.concatenate([[True], np.diff(ranks) < -1e-4])
        return np.cumsum(starts) - 1

    def _compute_hybrid_loss(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        y_pad: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Masked pointwise BCE loss for probability calibration
        bce_all = F.binary_cross_entropy_with_logits(logits, y_pad, reduction="none")
        mask_f = valid_mask.float()
        bce_loss = (bce_all * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        # 2. Listwise Softmax Cross-Entropy over valid candidates of each observation
        has_pos = (y_pad * mask_f).sum(dim=1) > 0.5
        if has_pos.any():
            masked_logits = logits[has_pos].masked_fill(~valid_mask[has_pos], -1e9)
            targets = torch.argmax(y_pad[has_pos], dim=1)
            ce_loss = F.cross_entropy(masked_logits, targets)
        else:
            ce_loss = bce_loss

        return self.listwise_weight * ce_loss + (1.0 - self.listwise_weight) * bce_loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        obs_group_ids: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        val_obs_group_ids: Optional[np.ndarray] = None,
    ) -> "PyTorchCandidateSetWrapper":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if obs_group_ids is None:
            obs_group_ids = self._infer_obs_groups_from_features(X)

        X_pad, valid_mask, _, _, y_pad = pack_observations_padded(X, obs_group_ids, y)

        train_ds = TensorDataset(
            torch.from_numpy(X_pad),
            torch.from_numpy(valid_mask),
            torch.from_numpy(y_pad),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_tensors = None
        if X_val is not None and y_val is not None:
            if val_obs_group_ids is None:
                val_obs_group_ids = self._infer_obs_groups_from_features(X_val)
            Xv_pad, mv_pad, _, _, yv_pad = pack_observations_padded(X_val, val_obs_group_ids, y_val)
            val_tensors = (Xv_pad, mv_pad, yv_pad)

        self.model = CandidateSetResNet(self.in_features, self.hidden_dim, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1))

        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}

        for ep in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            total_obs = 0

            for bx, bmask, by in train_loader:
                bx = bx.to(self.device)
                bmask = bmask.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                logits = self.model(bx, bmask)
                loss = self._compute_hybrid_loss(logits, bmask, by)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * len(bx)
                total_obs += len(bx)

            scheduler.step()
            train_loss = running_loss / max(total_obs, 1)
            self.history_["epoch"].append(ep)
            self.history_["train_loss"].append(train_loss)

            if val_tensors is not None:
                self.model.eval()
                Xv_pad, mv_pad, yv_pad = val_tensors
                val_running = 0.0
                val_count = 0
                eval_bs = self.batch_size * 2
                with torch.no_grad():
                    for s in range(0, len(Xv_pad), eval_bs):
                        e = min(s + eval_bs, len(Xv_pad))
                        bx = torch.from_numpy(Xv_pad[s:e]).to(self.device)
                        bm = torch.from_numpy(mv_pad[s:e]).to(self.device)
                        by = torch.from_numpy(yv_pad[s:e]).to(self.device)
                        logits = self.model(bx, bm)
                        vloss = self._compute_hybrid_loss(logits, bm, by)
                        val_running += float(vloss.item()) * (e - s)
                        val_count += (e - s)
                self.history_["val_loss"].append(val_running / max(val_count, 1))

        return self

    def predict_proba(self, X: np.ndarray, obs_group_ids: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.model is not None, "PyTorchCandidateSetWrapper must be fitted before predict_proba."
        if obs_group_ids is None:
            obs_group_ids = self._infer_obs_groups_from_features(X)

        X_pad, valid_mask, row_obs_idx, row_slot_idx, _ = pack_observations_padded(X, obs_group_ids, None)
        n_obs = len(X_pad)
        max_k = X_pad.shape[1]

        self.model.eval()
        probs_pad = np.zeros((n_obs, max_k), dtype=np.float32)
        eval_bs = self.batch_size * 2

        with torch.no_grad():
            for s in range(0, n_obs, eval_bs):
                e = min(s + eval_bs, n_obs)
                bx = torch.from_numpy(X_pad[s:e]).to(self.device)
                bm = torch.from_numpy(valid_mask[s:e]).to(self.device)
                logits = self.model(bx, bm)
                # Blend calibrated sigmoid and within-observation softmax probabilities
                sig_p = torch.sigmoid(logits)
                soft_p = torch.softmax(logits.masked_fill(~bm, -1e9), dim=1)
                alpha = float(self.prob_blend_alpha)
                blend_p = (1.0 - alpha) * sig_p + alpha * soft_p
                probs_pad[s:e] = blend_p.cpu().numpy()

        pos_probs = probs_pad[row_obs_idx, row_slot_idx]
        return np.column_stack([1.0 - pos_probs, pos_probs])


@dataclass
class ModelSpec:
    name: str
    slug: str
    family: str
    uses_scaled_features: bool
    key_config: str
    loss_objective: str


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="Logistic Regression",
        slug="logistic_regression",
        family="Linear & Probabilistic",
        uses_scaled_features=True,
        key_config="L2 penalty, C=1.0, lbfgs, max_iter=500",
        loss_objective="Binary Cross-Entropy (Log Loss) with L2 regularization",
    ),
    ModelSpec(
        name="Extra Trees",
        slug="extra_trees",
        family="Randomized Ensemble",
        uses_scaled_features=False,
        key_config="n_estimators=120, max_depth=20, min_samples_leaf=10",
        loss_objective="Extremely Randomized Gini Impurity smoothing over GPS noise",
    ),
    ModelSpec(
        name="Random Forest",
        slug="random_forest",
        family="Bagging Ensemble",
        uses_scaled_features=False,
        key_config="n_estimators=120, max_depth=18, min_samples_leaf=12",
        loss_objective="Bootstrap-aggregated Gini Impurity reduction",
    ),
    ModelSpec(
        name="HistGradientBoosting",
        slug="hist_gbdt",
        family="Histogram GBDT",
        uses_scaled_features=False,
        key_config="max_iter=300, lr=0.05, max_leaf_nodes=63, l2_regularization=1.0",
        loss_objective="Quantile-binned Binary Log Loss with L2 leaf penalty",
    ),
    ModelSpec(
        name="LightGBM",
        slug="lightgbm",
        family="Leaf-Wise Boosting",
        uses_scaled_features=False,
        key_config="n_estimators=350, lr=0.05, num_leaves=63, min_child_samples=40",
        loss_objective="Leaf-wise Gradient Boosted Binary Log Loss",
    ),
    ModelSpec(
        name="XGBoost",
        slug="xgboost",
        family="Depth-Wise Boosting",
        uses_scaled_features=False,
        key_config="n_estimators=350, lr=0.05, max_depth=7, subsample=0.8, reg_lambda=2.0",
        loss_objective="Regularized Depth-wise Binary Logistic Loss (hist)",
    ),
    ModelSpec(
        name="CandidateSetResNet",
        slug="candidate_set_resnet",
        family="Set-Aware Neural Network",
        uses_scaled_features=True,
        key_config="ResNet(28->96) + Candidate-Set Mean/Max Context, AdamW, CosineLR",
        loss_objective="Hybrid Listwise Observation Softmax Cross-Entropy + Calibrated BCE",
    ),
]


def build_estimator(slug: str, num_features: int = 28, epochs: int = 16, seed: int = 42):
    """Instantiate an untuned baseline estimator for the redesigned Stage 4 lineup."""
    if slug == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    if slug == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        )
    if slug == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=12,
            random_state=seed,
            n_jobs=-1,
        )
    if slug == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed,
        )
    if slug == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if slug == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if slug == "candidate_set_resnet":
        return PyTorchCandidateSetWrapper(
            in_features=num_features,
            hidden_dim=96,
            dropout=0.15,
            lr=1.5e-3,
            weight_decay=1e-4,
            batch_size=2048,
            epochs=epochs,
            listwise_weight=0.70,
            random_state=seed,
        )
    raise ValueError(f"Unknown model slug: {slug}")