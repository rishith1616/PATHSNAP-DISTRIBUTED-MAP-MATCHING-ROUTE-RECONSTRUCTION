import math
from typing import Dict, List

import numpy as np

from src.features.candidate_generator import SpatialCandidateIndex
from src.preprocessing.spatial_graph import TripData

# 18 spatial, kinematic, topological, and relative candidate features
FEATURE_COLUMNS: List[str] = [
    "perp_dist_m",
    "abs_cross_track_m",
    "signed_cross_track_m",
    "t_proj",
    "is_interior_proj",
    "heading_diff_rad",
    "cos_heading_diff",
    "effective_heading_penalty",
    "speed_mps",
    "dt_s",
    "arc_length_m",
    "log_arc_length",
    "from_node_in_degree",
    "to_node_out_degree",
    "dist_rank",
    "dist_diff_from_best_m",
    "dist_ratio_to_best",
    "num_candidates",
]


def wrap_angle_diff(angle_a: np.ndarray, angle_b: np.ndarray) -> np.ndarray:
    """Compute shortest unsigned angular difference in [0, pi] radians."""
    return np.abs((angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi)


def extract_trip_candidate_features(
    trip: TripData,
    radius_m: float = 75.0,
    top_k: int = 10,
    include_labels: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate candidate arcs and extract the 18-dimensional feature matrix for every
    (observation, candidate_arc) pair in a single trip.

    Returns a dictionary containing:
      - trip_id: str
      - num_observations: int T
      - X: float32 array of shape (N_pairs, len(FEATURE_COLUMNS))
      - obs_idx: int32 array mapping each candidate row to its GPS observation index [0..T-1]
      - cand_arc_id: int64 array of candidate arc_ids
      - obs_gt_arc_id: int64 array of length T (ground truth arc per observation, if include_labels=True)
      - y: int8 binary target array of shape (N_pairs,) (1 if cand_arc_id == gt_arc_id, else 0)
      - candidate_misses: int count of observations where gt_arc_id is absent from candidate set
    """
    spatial_index = SpatialCandidateIndex(trip.arcs_df)

    gx = trip.track_df["x"].to_numpy(dtype=np.float64)
    gy = trip.track_df["y"].to_numpy(dtype=np.float64)
    dt = np.clip(trip.track_df["dt"].to_numpy(dtype=np.float64), 0.0, 60.0)
    speed = np.clip(trip.track_df["speed_mps"].to_numpy(dtype=np.float64), 0.0, 50.0)
    gps_heading = trip.track_df["gps_heading_rad"].to_numpy(dtype=np.float64)
    valid_heading = trip.track_df["valid_heading"].to_numpy(dtype=np.float64)

    arc_lengths = np.maximum(trip.arcs_df["length_m"].to_numpy(dtype=np.float64), 0.0)
    arc_bearings = trip.arcs_df["bearing_rad"].to_numpy(dtype=np.float64)
    from_nodes = trip.arcs_df["from_node"].to_numpy(dtype=np.int64)
    to_nodes = trip.arcs_df["to_node"].to_numpy(dtype=np.int64)

    # Precompute node in-degrees and out-degrees from the directed road graph
    max_node_id = max(
        len(trip.nodes_xy),
        int(from_nodes.max()) + 1 if len(from_nodes) > 0 else 0,
        int(to_nodes.max()) + 1 if len(to_nodes) > 0 else 0,
    )
    in_deg = np.zeros(max_node_id, dtype=np.float64)
    out_deg = np.zeros(max_node_id, dtype=np.float64)
    for u, deg in trip.graph.out_degree():
        out_deg[int(u)] = float(deg)
    for v, deg in trip.graph.in_degree():
        in_deg[int(v)] = float(deg)

    T = len(gx)
    X_chunks: List[np.ndarray] = []
    obs_idx_chunks: List[np.ndarray] = []
    cand_arc_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    candidate_misses = 0
    gt_arcs = trip.gt_arc_ids if (include_labels and trip.gt_arc_ids is not None) else None

    for t in range(T):
        c_arcs, c_dists, c_tproj, c_cross = spatial_index.query_observation(
            gx[t], gy[t], radius_m=radius_m, top_k=top_k, fallback_nearest=True
        )
        K_curr = len(c_arcs)
        if K_curr == 0:
            continue

        # Kinematic & directional alignment
        if valid_heading[t] > 0.5:
            h_diff = wrap_angle_diff(gps_heading[t], arc_bearings[c_arcs])
            cos_h = np.cos(h_diff)
            eff_h_pen = 1.0 - cos_h
        else:
            # Neutral values when vehicle is stationary or after a large time gap
            h_diff = np.full(K_curr, 0.5 * math.pi, dtype=np.float64)
            cos_h = np.zeros(K_curr, dtype=np.float64)
            eff_h_pen = np.zeros(K_curr, dtype=np.float64)

        c_len = arc_lengths[c_arcs]
        best_dist = float(c_dists[0])
        ranks = np.arange(K_curr, dtype=np.float64)

        feat_block = np.column_stack(
            [
                c_dists,                                            # 0: perp_dist_m
                np.abs(c_cross),                                    # 1: abs_cross_track_m
                c_cross,                                            # 2: signed_cross_track_m
                c_tproj,                                            # 3: t_proj
                ((c_tproj > 0.0) & (c_tproj < 1.0)).astype(float),  # 4: is_interior_proj
                h_diff,                                             # 5: heading_diff_rad
                cos_h,                                              # 6: cos_heading_diff
                eff_h_pen,                                          # 7: effective_heading_penalty
                np.full(K_curr, speed[t]),                          # 8: speed_mps
                np.full(K_curr, dt[t]),                             # 9: dt_s
                c_len,                                              # 10: arc_length_m
                np.log1p(c_len),                                    # 11: log_arc_length
                in_deg[from_nodes[c_arcs]],                         # 12: from_node_in_degree
                out_deg[to_nodes[c_arcs]],                          # 13: to_node_out_degree
                ranks,                                              # 14: dist_rank
                c_dists - best_dist,                                # 15: dist_diff_from_best_m
                (c_dists + 1.0) / (best_dist + 1.0),                # 16: dist_ratio_to_best
                np.full(K_curr, float(K_curr)),                     # 17: num_candidates
            ]
        ).astype(np.float32)

        feat_block = np.nan_to_num(feat_block, nan=0.0, posinf=0.0, neginf=0.0)

        X_chunks.append(feat_block)
        obs_idx_chunks.append(np.full(K_curr, t, dtype=np.int32))
        cand_arc_chunks.append(c_arcs.astype(np.int64))

        if gt_arcs is not None:
            gt_id = int(gt_arcs[t])
            is_match = (c_arcs == gt_id).astype(np.int8)
            if not np.any(is_match):
                candidate_misses += 1
            y_chunks.append(is_match)

    result: Dict[str, np.ndarray] = {
        "trip_id": trip.trip_id,
        "num_observations": T,
        "X": np.vstack(X_chunks),
        "obs_idx": np.concatenate(obs_idx_chunks),
        "cand_arc_id": np.concatenate(cand_arc_chunks),
    }

    if gt_arcs is not None:
        result["obs_gt_arc_id"] = gt_arcs.astype(np.int64)
        result["y"] = np.concatenate(y_chunks)
        result["candidate_misses"] = candidate_misses

    return result