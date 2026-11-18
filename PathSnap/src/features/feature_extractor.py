import math
from typing import Dict, List, Set

import numpy as np

from src.features.candidate_generator import SpatialCandidateIndex
from src.preprocessing.spatial_graph import TripData, point_to_segments_distance_and_pos

# 28 spatiotemporal, smoothed kinematic, multi-step alignment, topological, and relative features
FEATURE_COLUMNS: List[str] = [
    # Spatial Geometry (6)
    "perp_dist_m",
    "abs_cross_track_m",
    "signed_cross_track_m",
    "t_proj",
    "is_interior_proj",
    "dist_to_nearest_endpoint_m",
    # Smoothed Kinematic & Directional Alignment (7)
    "heading_diff_rad",
    "cos_heading_diff",
    "sin_heading_diff",
    "effective_heading_penalty",
    "smooth_speed_mps",
    "dt_s",
    "curvature_rad",
    # Multi-Step Trajectory-to-Arc Consistency (4)
    "prev_ping_arc_dist_m",
    "next_ping_arc_dist_m",
    "multi_step_mean_dist_m",
    "longitudinal_progress_m",
    # Road Graph Topology & Bidirectional Reachability (5)
    "arc_length_m",
    "log_arc_length",
    "from_node_in_degree",
    "to_node_out_degree",
    "bidirectional_topo_support",
    # Within-Observation Relative Competition (6)
    "dist_rank",
    "dist_diff_from_best_m",
    "dist_ratio_to_best",
    "softmax_spatial_prob",
    "softmax_joint_prob",
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
    Generate candidate arcs and extract the 28-dimensional spatiotemporal & topological
    feature matrix for every (observation, candidate_arc) pair in a single trip.
    """
    spatial_index = SpatialCandidateIndex(trip.arcs_df)

    gx = trip.track_df["x"].to_numpy(dtype=np.float64)
    gy = trip.track_df["y"].to_numpy(dtype=np.float64)
    dt = np.clip(trip.track_df["dt"].to_numpy(dtype=np.float64), 0.0, 60.0)
    raw_dt = trip.track_df["dt"].to_numpy(dtype=np.float64)
    speed = np.clip(trip.track_df["speed_mps"].to_numpy(dtype=np.float64), 0.0, 50.0)
    gps_heading = trip.track_df["gps_heading_rad"].to_numpy(dtype=np.float64)
    valid_heading = trip.track_df["valid_heading"].to_numpy(dtype=np.float64)
    curvature = trip.track_df["curvature_rad"].to_numpy(dtype=np.float64)

    arc_x1 = trip.arcs_df["x1"].to_numpy(dtype=np.float64)
    arc_y1 = trip.arcs_df["y1"].to_numpy(dtype=np.float64)
    arc_x2 = trip.arcs_df["x2"].to_numpy(dtype=np.float64)
    arc_y2 = trip.arcs_df["y2"].to_numpy(dtype=np.float64)
    arc_lengths = np.maximum(trip.arcs_df["length_m"].to_numpy(dtype=np.float64), 0.0)
    arc_bearings = trip.arcs_df["bearing_rad"].to_numpy(dtype=np.float64)
    from_nodes = trip.arcs_df["from_node"].to_numpy(dtype=np.int64)
    to_nodes = trip.arcs_df["to_node"].to_numpy(dtype=np.int64)

    # Precompute node in-degrees and out-degrees
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

    # Pass 1: Query candidates for all T observations first so frame t can access
    # 1-step bidirectional graph reachability from t-1 and t+1
    all_c_arcs: List[np.ndarray] = []
    all_c_dists: List[np.ndarray] = []
    all_c_tproj: List[np.ndarray] = []
    all_c_cross: List[np.ndarray] = []

    for t in range(T):
        c_arcs, c_dists, c_tproj, c_cross = spatial_index.query_observation(
            gx[t], gy[t], radius_m=radius_m, top_k=top_k, fallback_nearest=True
        )
        all_c_arcs.append(c_arcs)
        all_c_dists.append(c_dists)
        all_c_tproj.append(c_tproj)
        all_c_cross.append(c_cross)

    X_chunks: List[np.ndarray] = []
    obs_idx_chunks: List[np.ndarray] = []
    cand_arc_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    train_row_mask_chunks: List[np.ndarray] = []

    candidate_misses = 0
    gt_arcs = trip.gt_arc_ids if (include_labels and trip.gt_arc_ids is not None) else None
    valid_obs_mask = (
        trip.valid_train_mask
        if (include_labels and trip.valid_train_mask is not None)
        else np.ones(T, dtype=bool)
    )

    # Pass 2: Extract 28 spatiotemporal, multi-step, and graph-topological features
    for t in range(T):
        c_arcs = all_c_arcs[t]
        c_dists = all_c_dists[t]
        c_tproj = all_c_tproj[t]
        c_cross = all_c_cross[t]
        K_curr = len(c_arcs)
        if K_curr == 0:
            continue

        cx1 = arc_x1[c_arcs]
        cy1 = arc_y1[c_arcs]
        cx2 = arc_x2[c_arcs]
        cy2 = arc_y2[c_arcs]
        c_len = arc_lengths[c_arcs]
        c_bear = arc_bearings[c_arcs]
        c_u = from_nodes[c_arcs]
        c_v = to_nodes[c_arcs]

        # 1. Distance to nearest endpoint of candidate segment
        dist_to_end = np.minimum(c_tproj, 1.0 - c_tproj) * c_len

        # 2. Smoothed directional alignment
        if valid_heading[t] > 0.5:
            h_diff = wrap_angle_diff(gps_heading[t], c_bear)
            cos_h = np.cos(h_diff)
            sin_h = np.abs(np.sin(h_diff))
            eff_h_pen = 1.0 - cos_h
        else:
            h_diff = np.full(K_curr, 0.5 * math.pi, dtype=np.float64)
            cos_h = np.zeros(K_curr, dtype=np.float64)
            sin_h = np.zeros(K_curr, dtype=np.float64)
            eff_h_pen = np.zeros(K_curr, dtype=np.float64)

        # 3. Multi-step trajectory-to-arc consistency & longitudinal forward progression
        has_prev = (t > 0) and (raw_dt[t] <= 5.0)
        has_next = (t + 1 < T) and (raw_dt[t + 1] <= 5.0)

        if has_prev:
            prev_dist, _, _ = point_to_segments_distance_and_pos(gx[t - 1], gy[t - 1], cx1, cy1, cx2, cy2)
            # Signed forward displacement along candidate arc's unit direction vector
            step_dx = gx[t] - gx[t - 1]
            step_dy = gy[t] - gy[t - 1]
            long_prog = step_dx * np.sin(c_bear) + step_dy * np.cos(c_bear)
        else:
            prev_dist = c_dists.copy()
            long_prog = np.zeros(K_curr, dtype=np.float64)

        if has_next:
            next_dist, _, _ = point_to_segments_distance_and_pos(gx[t + 1], gy[t + 1], cx1, cy1, cx2, cy2)
        else:
            next_dist = c_dists.copy()

        # Multi-step window [t-2 .. t+2] mean perpendicular distance to infinite line of candidate arc
        # (measures whether the local 5-point road corridor is collinear with this arc)
        window_pts_x = [gx[t]]
        window_pts_y = [gy[t]]
        if has_prev:
            window_pts_x.append(gx[t - 1])
            window_pts_y.append(gy[t - 1])
            if (t > 1) and (raw_dt[t - 1] <= 5.0):
                window_pts_x.append(gx[t - 2])
                window_pts_y.append(gy[t - 2])
        if has_next:
            window_pts_x.append(gx[t + 1])
            window_pts_y.append(gy[t + 1])
            if (t + 2 < T) and (raw_dt[t + 2] <= 5.0):
                window_pts_x.append(gx[t + 2])
                window_pts_y.append(gy[t + 2])

        wx = np.array(window_pts_x, dtype=np.float64)[:, None]
        wy = np.array(window_pts_y, dtype=np.float64)[:, None]
        adx = (cx2 - cx1)[None, :]
        ady = (cy2 - cy1)[None, :]
        alen = np.maximum(c_len[None, :], 1e-6)
        line_cross = np.abs((wx - cx1[None, :]) * ady - (wy - cy1[None, :]) * adx) / alen
        multi_step_mean_dist = np.mean(line_cross, axis=0)

        # 4. Bidirectional 1-step & 2-step road-graph reachability from t-1 and t+1 candidates
        topo_support = np.zeros(K_curr, dtype=np.float64)

        if has_prev and len(all_c_arcs[t - 1]) > 0:
            p_arcs = all_c_arcs[t - 1]
            p_weights = np.exp(-0.5 * (all_c_dists[t - 1] / 12.0) ** 2)
            p_to_nodes = to_nodes[p_arcs]
            # 2-hop reachable nodes from prev candidates
            two_hop_from_prev: Set[int] = set()
            for pv in p_to_nodes:
                if int(pv) in trip.graph:
                    for nxt_n in trip.graph.successors(int(pv)):
                        two_hop_from_prev.add(int(nxt_n))

            for k_i in range(K_curr):
                aid_k = int(c_arcs[k_i])
                u_k = int(c_u[k_i])
                same_mask = p_arcs == aid_k
                conn1_mask = p_to_nodes == u_k
                if np.any(same_mask):
                    topo_support[k_i] += float(np.max(p_weights[same_mask]))
                elif np.any(conn1_mask):
                    topo_support[k_i] += 0.95 * float(np.max(p_weights[conn1_mask]))
                elif u_k in two_hop_from_prev:
                    topo_support[k_i] += 0.70

        if has_next and len(all_c_arcs[t + 1]) > 0:
            n_arcs = all_c_arcs[t + 1]
            n_weights = np.exp(-0.5 * (all_c_dists[t + 1] / 12.0) ** 2)
            n_from_nodes = from_nodes[n_arcs]
            two_hop_to_next: Set[int] = set()
            for nu in n_from_nodes:
                if int(nu) in trip.graph:
                    for prv_n in trip.graph.predecessors(int(nu)):
                        two_hop_to_next.add(int(prv_n))

            for k_i in range(K_curr):
                aid_k = int(c_arcs[k_i])
                v_k = int(c_v[k_i])
                same_mask = n_arcs == aid_k
                conn1_mask = n_from_nodes == v_k
                if np.any(same_mask):
                    topo_support[k_i] += float(np.max(n_weights[same_mask]))
                elif np.any(conn1_mask):
                    topo_support[k_i] += 0.95 * float(np.max(n_weights[conn1_mask]))
                elif v_k in two_hop_to_next:
                    topo_support[k_i] += 0.70

        # 5. Within-observation normalized spatial & joint spatial-directional probabilities
        best_dist = float(c_dists[0])
        ranks = np.arange(K_curr, dtype=np.float64)

        log_spat = -0.5 * (c_dists / 10.0) ** 2
        log_spat -= np.max(log_spat)
        exp_spat = np.exp(log_spat)
        softmax_spat = exp_spat / np.sum(exp_spat)

        # Combine spatial proximity + smoothed heading + topological support
        head_bonus = 1.5 * cos_h if valid_heading[t] > 0.5 else 0.0
        log_joint = -0.5 * (c_dists / 10.0) ** 2 + head_bonus + 1.2 * topo_support
        log_joint -= np.max(log_joint)
        exp_joint = np.exp(log_joint)
        softmax_joint = exp_joint / np.sum(exp_joint)

        feat_block = np.column_stack(
            [
                c_dists,                                            # 0: perp_dist_m
                np.abs(c_cross),                                    # 1: abs_cross_track_m
                c_cross,                                            # 2: signed_cross_track_m
                c_tproj,                                            # 3: t_proj
                ((c_tproj > 0.0) & (c_tproj < 1.0)).astype(float),  # 4: is_interior_proj
                dist_to_end,                                        # 5: dist_to_nearest_endpoint_m
                h_diff,                                             # 6: heading_diff_rad
                cos_h,                                              # 7: cos_heading_diff
                sin_h,                                              # 8: sin_heading_diff
                eff_h_pen,                                          # 9: effective_heading_penalty
                np.full(K_curr, speed[t]),                          # 10: smooth_speed_mps
                np.full(K_curr, dt[t]),                             # 11: dt_s
                np.full(K_curr, curvature[t]),                      # 12: curvature_rad
                prev_dist,                                          # 13: prev_ping_arc_dist_m
                next_dist,                                          # 14: next_ping_arc_dist_m
                multi_step_mean_dist,                               # 15: multi_step_mean_dist_m
                long_prog,                                          # 16: longitudinal_progress_m
                c_len,                                              # 17: arc_length_m
                np.log1p(c_len),                                    # 18: log_arc_length
                in_deg[c_u],                                        # 19: from_node_in_degree
                out_deg[c_v],                                       # 20: to_node_out_degree
                topo_support,                                       # 21: bidirectional_topo_support
                ranks,                                              # 22: dist_rank
                c_dists - best_dist,                                # 23: dist_diff_from_best_m
                (c_dists + 1.0) / (best_dist + 1.0),                # 24: dist_ratio_to_best
                softmax_spat,                                       # 25: softmax_spatial_prob
                softmax_joint,                                      # 26: softmax_joint_prob
                np.full(K_curr, float(K_curr)),                     # 27: num_candidates
            ]
        ).astype(np.float32)

        feat_block = np.nan_to_num(feat_block, nan=0.0, posinf=0.0, neginf=0.0)

        X_chunks.append(feat_block)
        obs_idx_chunks.append(np.full(K_curr, t, dtype=np.int32))
        cand_arc_chunks.append(c_arcs.astype(np.int64))

        if gt_arcs is not None:
            gt_id = int(gt_arcs[t])
            is_match = (c_arcs == gt_id).astype(np.int8)
            if valid_obs_mask[t] and not np.any(is_match):
                candidate_misses += 1
            y_chunks.append(is_match)
            train_row_mask_chunks.append(np.full(K_curr, valid_obs_mask[t], dtype=bool))

    result: Dict[str, np.ndarray] = {
        "trip_id": trip.trip_id,
        "num_observations": T,
        "X": np.vstack(X_chunks),
        "obs_idx": np.concatenate(obs_idx_chunks),
        "cand_arc_id": np.concatenate(cand_arc_chunks),
    }

    if gt_arcs is not None:
        result["obs_gt_arc_id"] = gt_arcs.astype(np.int64)
        result["obs_valid_mask"] = valid_obs_mask
        result["y"] = np.concatenate(y_chunks)
        result["train_row_mask"] = np.concatenate(train_row_mask_chunks)
        result["candidate_misses"] = candidate_misses

    return result