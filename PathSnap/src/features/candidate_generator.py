import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.preprocessing.spatial_graph import TripData, point_to_segments_distance_and_pos


class SpatialCandidateIndex:
    """
    Exact spatial segment index using cKDTree over densified arc sample points
    followed by exact vectorized point-to-segment distance computation.
    """

    def __init__(self, arcs_df: pd.DataFrame, sample_step_m: float = 15.0):
        self.arcs_df = arcs_df
        self.sample_step_m = sample_step_m
        self.x1 = arcs_df["x1"].to_numpy(dtype=np.float64)
        self.y1 = arcs_df["y1"].to_numpy(dtype=np.float64)
        self.x2 = arcs_df["x2"].to_numpy(dtype=np.float64)
        self.y2 = arcs_df["y2"].to_numpy(dtype=np.float64)
        self.lengths = arcs_df["length_m"].to_numpy(dtype=np.float64)

        pts_list: List[np.ndarray] = []
        arc_id_list: List[np.ndarray] = []

        for aid in range(len(arcs_df)):
            n_samples = max(2, int(math.ceil(self.lengths[aid] / sample_step_m)) + 1)
            alphas = np.linspace(0.0, 1.0, n_samples)
            sx = self.x1[aid] + alphas * (self.x2[aid] - self.x1[aid])
            sy = self.y1[aid] + alphas * (self.y2[aid] - self.y1[aid])
            pts_list.append(np.column_stack([sx, sy]))
            arc_id_list.append(np.full(n_samples, aid, dtype=np.int64))

        self.sample_pts = np.vstack(pts_list)
        self.sample_arc_ids = np.concatenate(arc_id_list)
        self.tree = cKDTree(self.sample_pts)

    def query_observation(
        self,
        px: float,
        py: float,
        radius_m: float = 75.0,
        top_k: int = 10,
        fallback_nearest: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return up to top_k candidate arcs within radius_m sorted by exact distance.
        If no arc lies within radius_m and fallback_nearest=True, returns the nearest top_k arcs.
        Returns: (cand_arc_ids, dist_m, t_proj, signed_cross_m)
        """
        search_r = math.hypot(radius_m, 0.5 * self.sample_step_m)
        pt_indices = self.tree.query_ball_point([px, py], r=search_r)

        if len(pt_indices) > 0:
            unique_arcs = np.unique(self.sample_arc_ids[pt_indices])
            dists, t_projs, signed_cross = point_to_segments_distance_and_pos(
                px, py,
                self.x1[unique_arcs], self.y1[unique_arcs],
                self.x2[unique_arcs], self.y2[unique_arcs],
            )
            within_mask = dists <= radius_m
            if np.any(within_mask):
                unique_arcs = unique_arcs[within_mask]
                dists = dists[within_mask]
                t_projs = t_projs[within_mask]
                signed_cross = signed_cross[within_mask]

                order = np.argsort(dists)[:top_k]
                return unique_arcs[order], dists[order], t_projs[order], signed_cross[order]

        if not fallback_nearest:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        # Fallback for extreme GPS dropouts outside radius_m
        k_query = min(len(self.sample_pts), max(top_k * 6, 30))
        _, nn_indices = self.tree.query([px, py], k=k_query)
        unique_arcs = np.unique(self.sample_arc_ids[np.atleast_1d(nn_indices)])
        dists, t_projs, signed_cross = point_to_segments_distance_and_pos(
            px, py,
            self.x1[unique_arcs], self.y1[unique_arcs],
            self.x2[unique_arcs], self.y2[unique_arcs],
        )
        order = np.argsort(dists)[:top_k]
        return unique_arcs[order], dists[order], t_projs[order], signed_cross[order]


def evaluate_candidate_recall(
    trips: List[TripData],
    radii: List[float] = [25.0, 50.0, 75.0, 100.0],
    top_k_list: List[int] = [5, 10, 15],
) -> pd.DataFrame:
    """
    Benchmark Candidate Miss Rate (1 - Recall@K,R) across search radii R and candidate caps K
    on the training trips to justify candidate-generation hyperparameters.
    """
    records: List[Dict] = []

    # Build spatial index once per trip and query up to max(R) and max(K)
    max_r = max(radii)
    max_k = max(top_k_list)

    per_obs_results: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for trip in trips:
        idx = SpatialCandidateIndex(trip.arcs_df)
        gx = trip.track_df["x"].to_numpy()
        gy = trip.track_df["y"].to_numpy()
        gt = trip.gt_arc_ids
        for t in range(len(gx)):
            c_arcs, c_dists, _, _ = idx.query_observation(
                gx[t], gy[t], radius_m=max_r, top_k=max_k, fallback_nearest=False
            )
            per_obs_results.append((int(gt[t]), c_arcs, c_dists))

    total_obs = len(per_obs_results)
    for r in radii:
        for k in top_k_list:
            misses = 0
            total_cands = 0
            for gt_arc, c_arcs, c_dists in per_obs_results:
                valid = c_arcs[c_dists <= r][:k]
                total_cands += len(valid)
                if gt_arc not in valid:
                    misses += 1
            miss_rate = 100.0 * misses / max(total_obs, 1)
            records.append(
                {
                    "radius_m": r,
                    "top_k": k,
                    "total_observations": total_obs,
                    "candidate_misses": misses,
                    "candidate_miss_rate_pct": round(miss_rate, 3),
                    "candidate_recall_pct": round(100.0 - miss_rate, 3),
                    "avg_candidates_per_obs": round(total_cands / max(total_obs, 1), 2),
                }
            )

    return pd.DataFrame(records)