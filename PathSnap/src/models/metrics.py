from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def load_trip_arc_topology(split_dir: Path, trip_ids: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load (from_node, to_node) integer arrays for each trip to evaluate topological Route Continuity.
    """
    topo: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tid in trip_ids:
        trip_dir = split_dir / tid
        arcs_path = trip_dir / f"{tid}.arcs"
        if not arcs_path.exists():
            for f in trip_dir.iterdir():
                if "arc" in f.name.lower():
                    arcs_path = f
                    break
        arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"])
        topo[tid] = (
            arcs_df["from_node"].to_numpy(dtype=np.int64),
            arcs_df["to_node"].to_numpy(dtype=np.int64),
        )
    return topo


def compute_route_continuity(
    pred_arcs: np.ndarray,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
) -> float:
    """
    Compute fraction of consecutive observation transitions (t-1 -> t) where the predicted
    arc either stays on the same arc or connects topologically via to_node[prev] == from_node[curr].
    """
    if len(pred_arcs) <= 1:
        return 1.0
    prev_a = pred_arcs[:-1]
    curr_a = pred_arcs[1:]
    same_arc = prev_a == curr_a
    connected_arc = to_nodes[prev_a] == from_nodes[curr_a]
    return float(np.mean(same_arc | connected_arc))


def evaluate_trips_predictions(
    trips: List[Dict],
    trip_probs: List[np.ndarray],
    topology_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    use_valid_mask: bool = False,
) -> Dict:
    """
    Compute candidate-level and observation-level evaluation metrics across a list of trips.
    When use_valid_mask=True (used on training CV folds), unannotated partial-route outliers
    (>75m from .route) are excluded from CV metric aggregation; on test partitions, use_valid_mask=False
    evaluates 100% of observations.
    """
    all_y_true: List[np.ndarray] = []
    all_y_prob: List[np.ndarray] = []

    total_obs = 0
    top1_hits = 0
    trip_obs_macro_f1s: List[float] = []
    trip_continuities: List[float] = []
    obs_nlls: List[float] = []

    for trip, probs in zip(trips, trip_probs):
        tid = trip["trip_id"]
        y_true = trip["y"]
        obs_idx = trip["obs_idx"]
        cand_arcs = trip["cand_arc_id"]
        gt_arcs = trip["obs_gt_arc_id"]
        T = int(trip["num_observations"])

        obs_valid = (
            trip["obs_valid_mask"]
            if (use_valid_mask and "obs_valid_mask" in trip)
            else np.ones(T, dtype=bool)
        )
        row_valid = (
            trip["train_row_mask"]
            if (use_valid_mask and "train_row_mask" in trip)
            else np.ones(len(y_true), dtype=bool)
        )

        probs_clipped = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
        all_y_true.append(y_true[row_valid])
        all_y_prob.append(probs_clipped[row_valid])

        change_points = np.flatnonzero(np.diff(obs_idx)) + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [len(obs_idx)]])

        pred_obs_arcs = np.full(T, -1, dtype=np.int64)

        for s, e in zip(starts, ends):
            t = int(obs_idx[s])
            seg_probs = probs_clipped[s:e]
            seg_arcs = cand_arcs[s:e]
            best_local = int(np.argmax(seg_probs))
            pred_arc = int(seg_arcs[best_local])
            pred_obs_arcs[t] = pred_arc

            if not obs_valid[t]:
                continue

            gt_arc = int(gt_arcs[t])
            if pred_arc == gt_arc:
                top1_hits += 1
            total_obs += 1

            match_mask = seg_arcs == gt_arc
            if np.any(match_mask):
                norm_p = float(seg_probs[match_mask][0] / np.sum(seg_probs))
                obs_nlls.append(-np.log(max(norm_p, 1e-7)))

        eval_mask = (pred_obs_arcs >= 0) & obs_valid
        if np.any(eval_mask):
            trip_f1 = f1_score(
                gt_arcs[eval_mask],
                pred_obs_arcs[eval_mask],
                average="macro",
                zero_division=0,
            )
            trip_obs_macro_f1s.append(float(trip_f1))

            from_nodes, to_nodes = topology_map[tid]
            cont = compute_route_continuity(pred_obs_arcs[eval_mask], from_nodes, to_nodes)
            trip_continuities.append(cont)

    y_true_cat = np.concatenate(all_y_true)
    y_prob_cat = np.concatenate(all_y_prob)
    y_pred_bin = (y_prob_cat >= 0.5).astype(np.int8)

    pair_acc = float(accuracy_score(y_true_cat, y_pred_bin))
    pair_prec = float(precision_score(y_true_cat, y_pred_bin, zero_division=0))
    pair_rec = float(recall_score(y_true_cat, y_pred_bin, zero_division=0))
    pair_macro_f1 = float(f1_score(y_true_cat, y_pred_bin, average="macro", zero_division=0))
    pair_logloss = float(log_loss(y_true_cat, y_prob_cat, labels=[0, 1]))
    cm = confusion_matrix(y_true_cat, y_pred_bin, labels=[0, 1])

    obs_top1_acc = float(top1_hits / max(total_obs, 1))
    obs_macro_f1 = float(np.mean(trip_obs_macro_f1s)) if trip_obs_macro_f1s else 0.0
    obs_logloss = float(np.mean(obs_nlls)) if obs_nlls else float("nan")
    route_continuity = float(np.mean(trip_continuities)) if trip_continuities else 0.0

    return {
        "obs_top1_acc": obs_top1_acc,
        "obs_macro_f1": obs_macro_f1,
        "obs_logloss": obs_logloss,
        "route_continuity": route_continuity,
        "pair_accuracy": pair_acc,
        "pair_precision": pair_prec,
        "pair_recall": pair_rec,
        "pair_macro_f1": pair_macro_f1,
        "pair_logloss": pair_logloss,
        "confusion_matrix": cm.tolist(),
    }