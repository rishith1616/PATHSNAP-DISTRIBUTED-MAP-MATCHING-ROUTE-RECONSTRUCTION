import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_extractor import FEATURE_COLUMNS
from src.models.metrics import evaluate_trips_predictions, load_trip_arc_topology
from src.models.model_zoo import MODEL_SPECS, PyTorchCandidateSetWrapper, build_estimator

RANDOM_SEED = 42
N_CV_FOLDS = 5


def get_trip_X(trip: Dict, use_scaled: bool) -> np.ndarray:
    return trip["X_scaled"] if use_scaled else trip["X"]


def stack_valid_training_rows(
    trips: List[Dict],
    use_scaled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack valid on-route candidate rows across a list of training trips and assign
    globally unique observation IDs for listwise set-aware training.
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    obs_uid_list: List[np.ndarray] = []
    uid_offset = 0

    for t in trips:
        mask = t["train_row_mask"]
        X_t = get_trip_X(t, use_scaled)[mask]
        y_t = t["y"][mask]
        obs_t = t["obs_idx"][mask].astype(np.int64) + uid_offset
        if len(obs_t) > 0:
            uid_offset = int(obs_t.max()) + 1
            X_list.append(X_t)
            y_list.append(y_t)
            obs_uid_list.append(obs_t)

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(obs_uid_list)


def predict_trip_list(model, trips: List[Dict], use_scaled: bool) -> List[np.ndarray]:
    probs_list: List[np.ndarray] = []
    is_set_net = isinstance(model, PyTorchCandidateSetWrapper)
    for t in trips:
        X_t = get_trip_X(t, use_scaled)
        if is_set_net:
            p_pos = model.predict_proba(X_t, obs_group_ids=t["obs_idx"])[:, 1]
        else:
            p_pos = model.predict_proba(X_t)[:, 1]
        probs_list.append(p_pos)
    return probs_list


def generate_stage4_visualizations(
    summary_df: pd.DataFrame,
    detailed_records: Dict[str, Dict],
    nn_hist_df: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    vis_dir: Path,
) -> None:
    """Generate multi-model comparison charts, confusion matrices, feature importances, and neural loss curve."""
    models = summary_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    # Plot 1: Multi-Model Performance Comparison (CV vs Student-Test)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "PathSnap Stage 4 (Redesigned Lineup + 28 Spatiotemporal Features): Model Benchmark",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].bar(x - width / 2, summary_df["CV_Top1_Acc"] * 100, width, label="5-Fold CV (Train)", color="#2b5c8f")
    axes[0, 0].bar(
        x + width / 2, summary_df["Student_Test_Top1_Acc"] * 100, width, label="Student-Test (100%)", color="#1b9e3e"
    )
    axes[0, 0].set_title("Observation-Level Top-1 Accuracy (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=22, ha="right")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(x - width / 2, summary_df["CV_Obs_Macro_F1"], width, label="5-Fold CV (Train)", color="#2b5c8f")
    axes[0, 1].bar(
        x + width / 2, summary_df["Student_Test_Obs_Macro_F1"], width, label="Student-Test (100%)", color="#d95f02"
    )
    axes[0, 1].set_title("Observation-Level Macro F1-Score")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=22, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(x - width / 2, summary_df["CV_Obs_LogLoss"], width, label="5-Fold CV Obs NLL", color="#7570b3")
    axes[1, 0].bar(
        x + width / 2, summary_df["Student_Test_Obs_LogLoss"], width, label="Student-Test Obs NLL", color="#e7298a"
    )
    axes[1, 0].set_title("Observation-Level Emission Log Loss (Lower = Better HMM Emissions)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=22, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(
        x, summary_df["Student_Test_Route_Continuity"] * 100, width=0.5, color="#17becf", label="Route Continuity (%)"
    )
    axes[1, 1].set_title("Classifier-Only Route Continuity on Student-Test (%)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=22, ha="right")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / "stage4_model_comparison.png", dpi=180)
    plt.close(fig)

    # Plot 2: Confusion Matrices on Student-Test (Top 4 Models)
    top_slugs = summary_df["Slug"].tolist()[:4]
    top_names = summary_df["Model"].tolist()[:4]
    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(16, 4.2))
    fig_cm.suptitle("Student-Test Candidate Confusion Matrices (Top 4 Models)", fontsize=12.5, fontweight="bold")
    for ax, slug, mname in zip(axes_cm, top_slugs, top_names):
        cm = np.array(detailed_records[slug]["student_test_eval"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(mname, fontsize=10.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(vis_dir / "stage4_confusion_matrices.png", dpi=180)
    plt.close(fig_cm)

    # Plot 3: Top 15 Feature Importances (Tree/Boosting Ensembles)
    if not feat_imp_df.empty:
        top_f = feat_imp_df.head(15).iloc[::-1]
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        ax_fi.barh(top_f["feature"], top_f["mean_importance"], color="#2b5c8f")
        ax_fi.set_title("Top 15 Most Informative Engineered Features (Mean Normalized Tree Gain)")
        ax_fi.set_xlabel("Normalized Feature Importance")
        ax_fi.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / "stage4_feature_importance.png", dpi=180)
        plt.close(fig_fi)

    # Plot 4: Authentic Epoch-Wise Loss Curve for CandidateSetResNet
    if not nn_hist_df.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(
            nn_hist_df["epoch"],
            nn_hist_df["train_loss"],
            marker="o",
            label="Training Hybrid Listwise+BCE Loss (72 Train Trips)",
            color="#2b5c8f",
        )
        ax2.plot(
            nn_hist_df["epoch"],
            nn_hist_df["val_loss"],
            marker="s",
            label="Validation Hybrid Listwise+BCE Loss (8 Held-Out Trips)",
            color="#d95f02",
        )
        ax2.set_title("CandidateSetResNet — Actual Epoch-Wise Training & Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Hybrid Listwise + Calibration Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / "stage4_candidate_set_resnet_loss_curve.png", dpi=180)
        plt.close(fig2)


def main() -> None:
    processed_dir = ROOT / "data" / "processed"
    logs_dir = ROOT / "logs"
    vis_dir = ROOT / "visualizations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("PathSnap Stage 4 (Redesigned): 7 Models across 6 Families on 28 Features")
    print("Validation: 5-Fold GroupKFold (by trip_id) on 80 Train Trips + 10 Student-Test")
    print("=" * 84)

    train_trips: List[Dict] = joblib.load(processed_dir / "train_features.joblib")
    student_trips: List[Dict] = joblib.load(processed_dir / "student_test_features.joblib")

    train_topo = load_trip_arc_topology(
        ROOT / "data" / "raw" / "train", [t["trip_id"] for t in train_trips]
    )
    student_topo = load_trip_arc_topology(
        ROOT / "data" / "raw" / "student_test", [t["trip_id"] for t in student_trips]
    )

    num_features = train_trips[0]["X"].shape[1]
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    trip_indices = np.arange(len(train_trips))
    trip_groups = np.array([t["trip_id"] for t in train_trips])
    cv_splits = list(gkf.split(trip_indices, groups=trip_groups))

    summary_rows: List[Dict] = []
    detailed_records: Dict[str, Dict] = {}
    nn_hist_df = pd.DataFrame()
    importances_dict: Dict[str, np.ndarray] = {}

    for spec in MODEL_SPECS:
        print(f"\n>>> Experimenting with: {spec.name} [{spec.family}]")
        use_scaled = spec.uses_scaled_features
        is_set_net = spec.slug == "candidate_set_resnet"

        # 1. 5-Fold Trip-Grouped Cross-Validation on 80 Training Trips
        fold_metrics_list: List[Dict] = []
        cv_t0 = time.perf_counter()

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
            tr_trips = [train_trips[i] for i in tr_idx]
            va_trips = [train_trips[i] for i in va_idx]

            X_tr, y_tr, obs_uid_tr = stack_valid_training_rows(tr_trips, use_scaled)

            est = build_estimator(spec.slug, num_features=num_features, epochs=12, seed=RANDOM_SEED + fold_idx)
            if is_set_net:
                est.fit(X_tr, y_tr, obs_group_ids=obs_uid_tr)
            else:
                est.fit(X_tr, y_tr)

            va_probs = predict_trip_list(est, va_trips, use_scaled)
            va_metrics = evaluate_trips_predictions(va_trips, va_probs, train_topo, use_valid_mask=True)
            fold_metrics_list.append(va_metrics)

        cv_elapsed = time.perf_counter() - cv_t0

        cv_top1_mean = float(np.mean([m["obs_top1_acc"] for m in fold_metrics_list]))
        cv_top1_std = float(np.std([m["obs_top1_acc"] for m in fold_metrics_list]))
        cv_obs_f1_mean = float(np.mean([m["obs_macro_f1"] for m in fold_metrics_list]))
        cv_obs_f1_std = float(np.std([m["obs_macro_f1"] for m in fold_metrics_list]))
        cv_obs_ll_mean = float(np.mean([m["obs_logloss"] for m in fold_metrics_list]))
        cv_pair_f1_mean = float(np.mean([m["pair_macro_f1"] for m in fold_metrics_list]))
        cv_pair_ll_mean = float(np.mean([m["pair_logloss"] for m in fold_metrics_list]))
        cv_cont_mean = float(np.mean([m["route_continuity"] for m in fold_metrics_list]))

        print(
            f"    [5-Fold CV] Top-1 Acc: {cv_top1_mean*100:.2f}% (±{cv_top1_std*100:.2f}%) | "
            f"Obs Macro F1: {cv_obs_f1_mean:.4f} | Obs NLL: {cv_obs_ll_mean:.4f} ({cv_elapsed:.1f}s)"
        )

        # 2. Full Fit on Valid Rows of All 80 Training Trips & Evaluate on Train + 100% of 10 Student-Test Trips
        full_est = build_estimator(spec.slug, num_features=num_features, epochs=16, seed=RANDOM_SEED)

        fit_t0 = time.perf_counter()
        if is_set_net:
            hist_est = build_estimator(spec.slug, num_features=num_features, epochs=16, seed=RANDOM_SEED)
            X_h_tr, y_h_tr, uid_h_tr = stack_valid_training_rows(train_trips[:72], use_scaled)
            X_h_va, y_h_va, uid_h_va = stack_valid_training_rows(train_trips[72:], use_scaled)
            hist_est.fit(
                X_h_tr,
                y_h_tr,
                obs_group_ids=uid_h_tr,
                X_val=X_h_va,
                y_val=y_h_va,
                val_obs_group_ids=uid_h_va,
            )
            nn_hist_df = pd.DataFrame(hist_est.history_)
            nn_hist_df.to_csv(logs_dir / "stage4_candidate_set_resnet_history.csv", index=False)

        X_full_tr, y_full_tr, uid_full_tr = stack_valid_training_rows(train_trips, use_scaled)
        if is_set_net:
            full_est.fit(X_full_tr, y_full_tr, obs_group_ids=uid_full_tr)
        else:
            full_est.fit(X_full_tr, y_full_tr)
        train_time_s = time.perf_counter() - fit_t0

        if hasattr(full_est, "feature_importances_"):
            imp = np.asarray(full_est.feature_importances_, dtype=np.float64)
            if np.sum(imp) > 0:
                importances_dict[spec.slug] = imp / np.sum(imp)

        train_probs = predict_trip_list(full_est, train_trips, use_scaled)
        train_eval = evaluate_trips_predictions(train_trips, train_probs, train_topo, use_valid_mask=True)

        # Student-Test benchmark (100% of observations, timed)
        inf_t0 = time.perf_counter()
        student_probs = predict_trip_list(full_est, student_trips, use_scaled)
        student_inf_s = time.perf_counter() - inf_t0
        student_eval = evaluate_trips_predictions(student_trips, student_probs, student_topo, use_valid_mask=False)

        print(
            f"    [Train Fit] Top-1 Acc: {train_eval['obs_top1_acc']*100:.2f}% | "
            f"Obs Macro F1: {train_eval['obs_macro_f1']:.4f} | Fit Time: {train_time_s:.2f}s"
        )
        print(
            f"    [Stu-Test ] Top-1 Acc: {student_eval['obs_top1_acc']*100:.2f}% | "
            f"Obs Macro F1: {student_eval['obs_macro_f1']:.4f} | "
            f"Obs NLL: {student_eval['obs_logloss']:.4f} | "
            f"Pair LogLoss: {student_eval['pair_logloss']:.4f} | "
            f"Continuity: {student_eval['route_continuity']*100:.2f}% | "
            f"Inf Time: {student_inf_s*1000:.1f}ms"
        )

        summary_rows.append(
            {
                "Model": spec.name,
                "Slug": spec.slug,
                "Algorithm_Family": spec.family,
                "Key_Configuration": spec.key_config,
                "Loss_Objective": spec.loss_objective,
                "Train_Top1_Acc": round(train_eval["obs_top1_acc"], 5),
                "Train_Obs_Macro_F1": round(train_eval["obs_macro_f1"], 5),
                "CV_Top1_Acc": round(cv_top1_mean, 5),
                "CV_Top1_Std": round(cv_top1_std, 5),
                "CV_Obs_Macro_F1": round(cv_obs_f1_mean, 5),
                "CV_Obs_Macro_F1_Std": round(cv_obs_f1_std, 5),
                "CV_Obs_LogLoss": round(cv_obs_ll_mean, 5),
                "CV_Pair_Macro_F1": round(cv_pair_f1_mean, 5),
                "CV_Pair_LogLoss": round(cv_pair_ll_mean, 5),
                "CV_Route_Continuity": round(cv_cont_mean, 5),
                "Student_Test_Top1_Acc": round(student_eval["obs_top1_acc"], 5),
                "Student_Test_Obs_Macro_F1": round(student_eval["obs_macro_f1"], 5),
                "Student_Test_Obs_LogLoss": round(student_eval["obs_logloss"], 5),
                "Student_Test_Pair_Accuracy": round(student_eval["pair_accuracy"], 5),
                "Student_Test_Pair_Precision": round(student_eval["pair_precision"], 5),
                "Student_Test_Pair_Recall": round(student_eval["pair_recall"], 5),
                "Student_Test_Pair_Macro_F1": round(student_eval["pair_macro_f1"], 5),
                "Student_Test_Pair_LogLoss": round(student_eval["pair_logloss"], 5),
                "Student_Test_Route_Continuity": round(student_eval["route_continuity"], 5),
                "Full_Train_Time_s": round(train_time_s, 3),
                "Student_Test_Inf_Time_ms": round(student_inf_s * 1000.0, 2),
            }
        )

        detailed_records[spec.slug] = {
            "spec": spec.__dict__,
            "cv_folds": fold_metrics_list,
            "train_eval": train_eval,
            "student_test_eval": student_eval,
            "train_time_s": train_time_s,
            "student_inf_time_ms": student_inf_s * 1000.0,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["CV_Obs_Macro_F1", "CV_Top1_Acc"], ascending=False).reset_index(drop=True)

    comp_csv_path = logs_dir / "stage4_model_comparison.csv"
    summary_df.to_csv(comp_csv_path, index=False)

    details_json_path = logs_dir / "stage4_detailed_results.json"
    with open(details_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_records, f, indent=2)

    feat_imp_df = pd.DataFrame()
    if importances_dict:
        imp_mat = np.vstack(list(importances_dict.values()))
        feat_imp_df = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "mean_importance": np.mean(imp_mat, axis=0)}
        ).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
        feat_imp_df.to_csv(logs_dir / "stage4_feature_importance.csv", index=False)

    generate_stage4_visualizations(summary_df, detailed_records, nn_hist_df, feat_imp_df, vis_dir)

    print("\n" + "=" * 84)
    print("STAGE 4 CONSOLIDATED MODEL COMPARISON (Redesigned Lineup — Ranked by 5-Fold CV)")
    print("=" * 84)
    display_cols = [
        "Model",
        "Algorithm_Family",
        "CV_Top1_Acc",
        "CV_Obs_Macro_F1",
        "Student_Test_Top1_Acc",
        "Student_Test_Obs_Macro_F1",
        "Student_Test_Obs_LogLoss",
        "Student_Test_Route_Continuity",
        "Student_Test_Inf_Time_ms",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\n[SAVED] Model comparison CSV -> {comp_csv_path}")
    print(f"[SAVED] Detailed JSON + CMs  -> {details_json_path}")
    print(f"[SAVED] Stage 4 plots        -> {vis_dir / 'stage4_model_comparison.png'}")


if __name__ == "__main__":
    main()