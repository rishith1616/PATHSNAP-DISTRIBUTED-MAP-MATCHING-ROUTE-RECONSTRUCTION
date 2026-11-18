import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
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


def main() -> None:
    train_dir = ROOT / "data" / "raw" / "train"
    student_test_dir = ROOT / "data" / "raw" / "student_test"
    processed_dir = ROOT / "data" / "processed"
    artifacts_dir = ROOT / "artifacts"
    logs_dir = ROOT / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PathSnap Stage 3: Feature Engineering & Leakage-Free Preprocessing")
    print(f"Candidate Config: radius_m={CANDIDATE_RADIUS_M}m, top_k={CANDIDATE_TOP_K}")
    print("Strict Rule: StandardScaler fitted ONLY on 80 training trips (eval_test untouched)")
    print("=" * 70)

    t0 = time.perf_counter()
    train_trips = process_partition(train_dir, "Extracting Train Features (80 trips)")
    student_trips = process_partition(student_test_dir, "Extracting Student-Test Features (10 trips)")

    # Fit StandardScaler strictly on the 80 training trips' feature rows
    X_train_all = np.vstack([t["X"] for t in train_trips])
    X_student_all = np.vstack([t["X"] for t in student_trips])

    assert np.isfinite(X_train_all).all(), "Non-finite values found in X_train_all!"
    assert np.isfinite(X_student_all).all(), "Non-finite values found in X_student_all!"

    scaler = StandardScaler()
    scaler.fit(X_train_all)

    # Apply fitted scaler to both train and student_test partitions
    for t in train_trips:
        t["X_scaled"] = scaler.transform(t["X"]).astype(np.float32)

    for t in student_trips:
        t["X_scaled"] = scaler.transform(t["X"]).astype(np.float32)

    # Save scaler & feature config artifacts
    scaler_path = artifacts_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    feature_config = {
        "feature_columns": FEATURE_COLUMNS,
        "num_features": len(FEATURE_COLUMNS),
        "candidate_radius_m": CANDIDATE_RADIUS_M,
        "candidate_top_k": CANDIDATE_TOP_K,
        "fallback_nearest": True,
        "scaler_fitted_on": "train (80 trips)",
    }
    config_path = artifacts_dir / "feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    # Save processed datasets for fast model training & evaluation in Stages 4–6
    train_cache_path = processed_dir / "train_features.joblib"
    student_cache_path = processed_dir / "student_test_features.joblib"
    joblib.dump(train_trips, train_cache_path, compress=3)
    joblib.dump(student_trips, student_cache_path, compress=3)

    # Compute & log dataset feature summary statistics
    train_obs = sum(t["num_observations"] for t in train_trips)
    train_pairs = sum(len(t["y"]) for t in train_trips)
    train_pos = sum(int(t["y"].sum()) for t in train_trips)
    train_misses = sum(t["candidate_misses"] for t in train_trips)

    stu_obs = sum(t["num_observations"] for t in student_trips)
    stu_pairs = sum(len(t["y"]) for t in student_trips)
    stu_pos = sum(int(t["y"].sum()) for t in student_trips)
    stu_misses = sum(t["candidate_misses"] for t in student_trips)

    feat_stats_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "train_mean": np.mean(X_train_all, axis=0),
            "train_std": np.std(X_train_all, axis=0),
            "train_min": np.min(X_train_all, axis=0),
            "train_median": np.median(X_train_all, axis=0),
            "train_max": np.max(X_train_all, axis=0),
        }
    )
    feat_stats_path = logs_dir / "train_feature_statistics.csv"
    feat_stats_df.to_csv(feat_stats_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n[OK] Stage 3 completed in {elapsed:.2f}s")
    print("\n--- PARTITION FEATURE SUMMARY ---")
    print(
        f"  TRAIN (80 trips)        : {train_obs:,} obs | {train_pairs:,} candidate pairs | "
        f"{train_pos:,} positives | Miss Rate: {100.0 * train_misses / train_obs:.3f}%"
    )
    print(
        f"  STUDENT-TEST (10 trips) : {stu_obs:,} obs | {stu_pairs:,} candidate pairs | "
        f"{stu_pos:,} positives | Miss Rate: {100.0 * stu_misses / stu_obs:.3f}%"
    )
    print(f"\n[SAVED] Scaler artifact         -> {scaler_path}")
    print(f"[SAVED] Feature config JSON     -> {config_path}")
    print(f"[SAVED] Train feature cache     -> {train_cache_path}")
    print(f"[SAVED] Student-test cache      -> {student_cache_path}")
    print(f"[SAVED] Feature statistics CSV  -> {feat_stats_path}")


if __name__ == "__main__":
    main()