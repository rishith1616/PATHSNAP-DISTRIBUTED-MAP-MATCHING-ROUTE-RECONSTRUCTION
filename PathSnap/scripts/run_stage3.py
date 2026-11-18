import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_extractor import FEATURE_COLUMNS, extract_trip_candidate_features
from src.preprocessing.spatial_graph import build_trip_data

CANDIDATE_RADIUS_M = 75.0
CANDIDATE_TOP_K = 10


def process_partition(split_dir: Path, desc: str) -> List[Dict]:
    trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    records: List[Dict] = []
    for td in tqdm(trip_dirs, desc=desc):
        trip = build_trip_data(td, include_ground_truth=True)
        feat_dict = extract_trip_candidate_features(
            trip,
            radius_m=CANDIDATE_RADIUS_M,
            top_k=CANDIDATE_TOP_K,
            include_labels=True,
        )
        records.append(feat_dict)
    return records


def save_feature_diagnostic_plot(
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    vis_dir: Path,
) -> Path:
    """Plot positive (true arc) vs. negative (distractor arc) distributions for key engineered features."""
    pos = X_valid[y_valid == 1]
    neg = X_valid[y_valid == 0]

    key_feats = [
        ("perp_dist_m", "Perpendicular Distance to Arc (m)", (0, 60)),
        ("cos_heading_diff", "Smoothed Cosine Heading Alignment", (-1.05, 1.05)),
        ("multi_step_mean_dist_m", "Multi-Step [t-2..t+2] Corridor Dist (m)", (0, 60)),
        ("bidirectional_topo_support", "Bidirectional Graph Reachability Support", (0, 2.1)),
        ("softmax_joint_prob", "Within-Obs Joint Spatial-Topo Softmax Prob", (0, 1.02)),
        ("longitudinal_progress_m", "Signed Forward Progress Along Arc (m)", (-15, 35)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(
        "PathSnap Stage 3: Engineered Spatiotemporal & Topological Feature Separability (Train)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (col_name, title, xlim) in zip(axes.flat, key_feats):
        idx = FEATURE_COLUMNS.index(col_name)
        p_vals = np.clip(pos[:, idx], xlim[0], xlim[1])
        n_vals = np.clip(neg[:, idx], xlim[0], xlim[1])
        ax.hist(n_vals, bins=40, density=True, alpha=0.6, color="#d95f02", label="Negative Candidate (y=0)")
        ax.hist(p_vals, bins=40, density=True, alpha=0.65, color="#1b9e3e", label="True Road Segment (y=1)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    out_path = vis_dir / "stage3_feature_diagnostics.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    train_dir = ROOT / "data" / "raw" / "train"
    student_test_dir = ROOT / "data" / "raw" / "student_test"
    processed_dir = ROOT / "data" / "processed"
    artifacts_dir = ROOT / "artifacts"
    logs_dir = ROOT / "logs"
    vis_dir = ROOT / "visualizations"

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PathSnap Stage 3: Gap-Aware Alignment & 28-Feature Spatiotemporal Extraction")
    print(f"Candidate Config: radius_m={CANDIDATE_RADIUS_M}m, top_k={CANDIDATE_TOP_K}")
    print("Strict Rule: StandardScaler fitted ONLY on valid 80-train rows (eval_test untouched)")
    print("=" * 78)

    t0 = time.perf_counter()
    train_trips = process_partition(train_dir, "Extracting Train Features (80 trips)")
    student_trips = process_partition(student_test_dir, "Extracting Student-Test Features (10 trips)")

    # Fit StandardScaler strictly on valid on-route rows of the 80 training trips
    X_train_valid = np.vstack([t["X"][t["train_row_mask"]] for t in train_trips])
    y_train_valid = np.concatenate([t["y"][t["train_row_mask"]] for t in train_trips])
    X_student_all = np.vstack([t["X"] for t in student_trips])

    assert np.isfinite(X_train_valid).all(), "Non-finite values found in X_train_valid!"
    assert np.isfinite(X_student_all).all(), "Non-finite values found in X_student_all!"

    scaler = StandardScaler()
    scaler.fit(X_train_valid)

    for t in train_trips:
        t["X_scaled"] = scaler.transform(t["X"]).astype(np.float32)

    for t in student_trips:
        t["X_scaled"] = scaler.transform(t["X"]).astype(np.float32)

    scaler_path = artifacts_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    feature_config = {
        "feature_columns": FEATURE_COLUMNS,
        "num_features": len(FEATURE_COLUMNS),
        "candidate_radius_m": CANDIDATE_RADIUS_M,
        "candidate_top_k": CANDIDATE_TOP_K,
        "fallback_nearest": True,
        "partial_route_filter_threshold_m": 75.0,
        "scaler_fitted_on": "train valid on-route rows (80 trips)",
    }
    config_path = artifacts_dir / "feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    train_cache_path = processed_dir / "train_features.joblib"
    student_cache_path = processed_dir / "student_test_features.joblib"
    joblib.dump(train_trips, train_cache_path, compress=3)
    joblib.dump(student_trips, student_cache_path, compress=3)

    # Summary statistics
    train_obs_total = sum(t["num_observations"] for t in train_trips)
    train_obs_valid = sum(int(t["obs_valid_mask"].sum()) for t in train_trips)
    train_pairs_valid = int(len(y_train_valid))
    train_pos_valid = int(y_train_valid.sum())
    train_misses = sum(t["candidate_misses"] for t in train_trips)

    stu_obs_total = sum(t["num_observations"] for t in student_trips)
    stu_pairs = sum(len(t["y"]) for t in student_trips)
    stu_pos = sum(int(t["y"].sum()) for t in student_trips)
    stu_misses = sum(t["candidate_misses"] for t in student_trips)

    feat_stats_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "train_mean": np.mean(X_train_valid, axis=0),
            "train_std": np.std(X_train_valid, axis=0),
            "train_min": np.min(X_train_valid, axis=0),
            "train_median": np.median(X_train_valid, axis=0),
            "train_max": np.max(X_train_valid, axis=0),
        }
    )
    feat_stats_path = logs_dir / "train_feature_statistics.csv"
    feat_stats_df.to_csv(feat_stats_path, index=False)

    diag_plot_path = save_feature_diagnostic_plot(X_train_valid, y_train_valid, vis_dir)

    elapsed = time.perf_counter() - t0
    print(f"\n[OK] Stage 3 completed in {elapsed:.2f}s")
    print("\n--- UPGRADED STAGE 3 PARTITION FEATURE SUMMARY ---")
    print(
        f"  TRAIN (80 trips)        : {train_obs_valid:,} valid on-route obs (of {train_obs_total:,} total) | "
        f"{train_pairs_valid:,} valid candidate pairs | {train_pos_valid:,} positives | "
        f"On-Route Miss Rate: {100.0 * train_misses / max(train_obs_valid, 1):.3f}%"
    )
    print(
        f"  STUDENT-TEST (10 trips) : {stu_obs_total:,} obs | {stu_pairs:,} candidate pairs | "
        f"{stu_pos:,} positives | Miss Rate: {100.0 * stu_misses / stu_obs_total:.3f}%"
    )
    print(f"\n[SAVED] Scaler artifact         -> {scaler_path}")
    print(f"[SAVED] Feature config JSON     -> {config_path}")
    print(f"[SAVED] Train feature cache     -> {train_cache_path}")
    print(f"[SAVED] Student-test cache      -> {student_cache_path}")
    print(f"[SAVED] Feature statistics CSV  -> {feat_stats_path}")
    print(f"[SAVED] Feature diagnostic plot -> {diag_plot_path}")


if __name__ == "__main__":
    main()