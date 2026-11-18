import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Ensure PathSnap root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.candidate_generator import evaluate_candidate_recall
from src.preprocessing.spatial_graph import build_trip_data


def main() -> None:
    train_dir = ROOT / "data" / "raw" / "train"
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print("=" * 70)
    print(f"PathSnap Stage 2: Spatial CRS Projection, Graph Build & Candidate Audit")
    print(f"Processing {len(trip_dirs)} TRAINING trips (student_test & eval_test untouched)")
    print("=" * 70)

    t0 = time.perf_counter()
    trips = []
    align_dists = []
    utm_zones = set()

    for td in tqdm(trip_dirs, desc="Loading & Aligning Train Trips"):
        trip = build_trip_data(td, include_ground_truth=True)
        trips.append(trip)
        utm_zones.add(trip.epsg)
        if trip.alignment_mean_dist_m is not None:
            align_dists.append(trip.alignment_mean_dist_m)

    load_elapsed = time.perf_counter() - t0
    print(f"\n[OK] Loaded & projected {len(trips)} trips in {load_elapsed:.2f}s")
    print(f"[INFO] Distinct local UTM EPSG zones detected : {len(utm_zones)} -> {sorted(utm_zones)}")
    print(f"[INFO] Median GT alignment GPS-to-arc distance: {np.median(align_dists):.2f} m")
    print(f"[INFO] 95th pct GT alignment GPS-to-arc dist  : {np.percentile(align_dists, 95):.2f} m")

    print("\nRunning Candidate Miss Rate benchmark across (Radius R, Top-K)...")
    t1 = time.perf_counter()
    recall_df = evaluate_candidate_recall(
        trips,
        radii=[25.0, 50.0, 75.0, 100.0],
        top_k_list=[5, 10, 15],
    )
    bench_elapsed = time.perf_counter() - t1

    out_csv = logs_dir / "candidate_miss_rate_benchmark.csv"
    recall_df.to_csv(out_csv, index=False)

    print(f"\n--- CANDIDATE MISS RATE BENCHMARK (Completed in {bench_elapsed:.2f}s) ---")
    print(recall_df.to_string(index=False))
    print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()