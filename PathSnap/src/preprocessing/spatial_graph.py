import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer


def infer_trip_utm_epsg(lons: np.ndarray, lats: np.ndarray) -> int:
    """Compute the local WGS84 UTM EPSG code for a trip from its mean lon/lat."""
    mean_lon = float(np.nanmean(lons))
    mean_lat = float(np.nanmean(lats))
    zone = int(math.floor((mean_lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 + zone) if mean_lat >= 0.0 else (32700 + zone)


@dataclass
class TripData:
    trip_id: str
    epsg: int
    track_df: pd.DataFrame          # columns: lon, lat, timestamp, x, y, dt, speed_mps, gps_heading_rad, valid_heading
    nodes_xy: np.ndarray            # shape (N_nodes, 2) in local UTM meters
    arcs_df: pd.DataFrame           # columns: arc_id, from_node, to_node, x1, y1, x2, y2, length_m, bearing_rad
    graph: nx.DiGraph               # directed road network graph with edge attribute arc_id & length_m
    route_arcs: Optional[np.ndarray]        # ordered sequence of traversed arc_ids from .route
    gt_arc_ids: Optional[np.ndarray]        # aligned ground-truth arc_id per GPS observation (len = N_gps)
    alignment_mean_dist_m: Optional[float]  # diagnostic mean distance of GPS pings to aligned ground-truth arcs


def load_trip_files(trip_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load <trip_id>.track, <trip_id>.nodes, <trip_id>.arcs, and optional <trip_id>.route."""
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    # Fallback discovery if prefix differs
    if not track_path.exists():
        for f in trip_dir.iterdir():
            nl = f.name.lower()
            if "track" in nl:
                track_path = f
            elif "node" in nl:
                nodes_path = f
            elif "arc" in nl:
                arcs_path = f
            elif "route" in nl:
                route_path = f

    track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"])

    route_df = None
    if route_path.exists() and route_path.stat().st_size > 0:
        route_df = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])

    return track_df, nodes_df, arcs_df, route_df


def point_to_segments_distance_and_pos(
    px: float,
    py: float,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized exact Euclidean distance from point (px, py) to line segments (x1,y1)->(x2,y2).
    Returns:
      dist_m: perpendicular/endpoint distance in meters, shape (M,)
      t_proj: normalized longitudinal position along segment clipped to [0, 1], shape (M,)
      signed_cross_m: signed lateral offset (left/right of directed arc) in meters, shape (M,)
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    # Handle degenerate zero-length segments safely
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    t_raw = ((px - x1) * dx + (py - y1) * dy) / safe_len_sq
    t_proj = np.where(seg_len_sq > 1e-12, np.clip(t_raw, 0.0, 1.0), 0.0)

    proj_x = x1 + t_proj * dx
    proj_y = y1 + t_proj * dy
    dist_m = np.hypot(px - proj_x, py - proj_y)

    # Signed cross-track offset: positive = left of arc direction, negative = right
    seg_len = np.sqrt(safe_len_sq)
    signed_cross_m = np.where(
        seg_len_sq > 1e-12,
        ((px - x1) * dy - (py - y1) * dx) / seg_len,
        0.0,
    )
    return dist_m, t_proj, signed_cross_m


def align_gps_to_route_dp(
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    route_arcs: np.ndarray,
    arcs_df: pd.DataFrame,
    max_step_jump: int = 8,
    step_penalty: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    Monotonically align T GPS observations onto an ordered sequence of M traversed route arcs
    using dynamic programming so each GPS point gets its exact ground-truth arc_id.
    """
    T = len(gps_x)
    M = len(route_arcs)
    if M == 0:
        return np.full(T, -1, dtype=np.int64), float("nan")
    if M == 1:
        x1 = np.array([arcs_df.iloc[route_arcs[0]]["x1"]])
        y1 = np.array([arcs_df.iloc[route_arcs[0]]["y1"]])
        x2 = np.array([arcs_df.iloc[route_arcs[0]]["x2"]])
        y2 = np.array([arcs_df.iloc[route_arcs[0]]["y2"]])
        dists = [point_to_segments_distance_and_pos(gps_x[t], gps_y[t], x1, y1, x2, y2)[0][0] for t in range(T)]
        return np.full(T, route_arcs[0], dtype=np.int64), float(np.mean(dists))

    r_x1 = arcs_df["x1"].to_numpy()[route_arcs]
    r_y1 = arcs_df["y1"].to_numpy()[route_arcs]
    r_x2 = arcs_df["x2"].to_numpy()[route_arcs]
    r_y2 = arcs_df["y2"].to_numpy()[route_arcs]

    # Cost matrix C[t, j] = distance (meters) from GPS ping t to route step j
    cost_mat = np.empty((T, M), dtype=np.float64)
    for t in range(T):
        cost_mat[t], _, _ = point_to_segments_distance_and_pos(
            gps_x[t], gps_y[t], r_x1, r_y1, r_x2, r_y2
        )

    INF = 1e18
    dp = np.full((T, M), INF, dtype=np.float64)
    backptr = np.zeros((T, M), dtype=np.int32)

    # Allow starting on early route segments (in case initial GPS pings start slightly downstream)
    init_window = min(M, 5)
    dp[0, :init_window] = cost_mat[0, :init_window]

    for t in range(1, T):
        for j in range(M):
            prev_start = max(0, j - max_step_jump)
            prev_costs = dp[t - 1, prev_start : j + 1]
            # Small penalty for jumping multiple arcs in 1 step to keep alignment smooth
            jumps = np.arange(j - prev_start, -1, -1, dtype=np.float64)
            total_prev = prev_costs + step_penalty * jumps
            best_offset = int(np.argmin(total_prev))
            dp[t, j] = cost_mat[t, j] + total_prev[best_offset]
            backptr[t, j] = prev_start + best_offset

    # End near the tail of the reference route
    end_start = max(0, M - 5)
    last_j = end_start + int(np.argmin(dp[T - 1, end_start:]))

    aligned_steps = np.empty(T, dtype=np.int32)
    aligned_steps[T - 1] = last_j
    for t in range(T - 1, 0, -1):
        aligned_steps[t - 1] = backptr[t, aligned_steps[t]]

    gt_arc_ids = route_arcs[aligned_steps].astype(np.int64)
    mean_dist = float(np.mean(cost_mat[np.arange(T), aligned_steps]))
    return gt_arc_ids, mean_dist


def build_trip_data(trip_dir: Path, include_ground_truth: bool = True) -> TripData:
    """
    Load raw files for a single trip, project coordinates to the trip's local UTM CRS,
    compute trajectory kinematics and directed road graph, and optionally align ground truth.
    """
    trip_id = trip_dir.name
    track_df, nodes_df, arcs_df, route_df = load_trip_files(trip_dir)

    # 1. Local UTM projection
    epsg = infer_trip_utm_epsg(track_df["lon"].to_numpy(), track_df["lat"].to_numpy())
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    gx, gy = transformer.transform(track_df["lon"].to_numpy(), track_df["lat"].to_numpy())
    track_df["x"] = gx
    track_df["y"] = gy

    # 2. Trajectory kinematics (dt, speed, heading)
    dt = track_df["timestamp"].diff().fillna(1.0).to_numpy(dtype=np.float64)
    dt_safe = np.where(dt > 0.0, dt, 1.0)
    dx_gps = np.diff(gx, prepend=gx[0])
    dy_gps = np.diff(gy, prepend=gy[0])
    step_dist = np.hypot(dx_gps, dy_gps)

    # Use forward difference for index 0 so initial heading is defined if vehicle moves
    if len(gx) > 1:
        dx_gps[0] = gx[1] - gx[0]
        dy_gps[0] = gy[1] - gy[0]
        step_dist[0] = math.hypot(dx_gps[0], dy_gps[0])

    speed_mps = step_dist / dt_safe
    # Bearing in radians: atan2(dx, dy) -> 0 = North, pi/2 = East (matches compass convention)
    gps_heading = np.arctan2(dx_gps, dy_gps)
    # Heading is reliable when vehicle moved >= 1.0 m and timestamp gap <= 10 s
    valid_heading = (step_dist >= 1.0) & (dt <= 10.0)

    track_df["dt"] = dt
    track_df["step_dist_m"] = step_dist
    track_df["speed_mps"] = speed_mps
    track_df["gps_heading_rad"] = gps_heading
    track_df["valid_heading"] = valid_heading.astype(np.float64)

    # 3. Project road network nodes and enrich arcs table
    nx_arr, ny_arr = transformer.transform(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy())
    nodes_xy = np.column_stack([nx_arr, ny_arr])

    from_idx = arcs_df["from_node"].to_numpy(dtype=np.int64)
    to_idx = arcs_df["to_node"].to_numpy(dtype=np.int64)

    x1 = nodes_xy[from_idx, 0]
    y1 = nodes_xy[from_idx, 1]
    x2 = nodes_xy[to_idx, 0]
    y2 = nodes_xy[to_idx, 1]
    arc_dx = x2 - x1
    arc_dy = y2 - y1
    length_m = np.hypot(arc_dx, arc_dy)
    bearing_rad = np.arctan2(arc_dx, arc_dy)

    arcs_df["arc_id"] = np.arange(len(arcs_df), dtype=np.int64)
    arcs_df["x1"] = x1
    arcs_df["y1"] = y1
    arcs_df["x2"] = x2
    arcs_df["y2"] = y2
    arcs_df["length_m"] = length_m
    arcs_df["bearing_rad"] = bearing_rad

    # 4. Build NetworkX DiGraph for topological queries & shortest-path transitions
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes_xy)))
    for aid, u, v, w in zip(arcs_df["arc_id"], from_idx, to_idx, length_m):
        # If parallel edges exist between (u, v), keep the shorter one for routing, store all arc_ids
        if G.has_edge(int(u), int(v)):
            G[int(u)][int(v)]["arc_ids"].append(int(aid))
            if float(w) < G[int(u)][int(v)]["weight"]:
                G[int(u)][int(v)]["weight"] = float(w)
                G[int(u)][int(v)]["primary_arc_id"] = int(aid)
        else:
            G.add_edge(
                int(u),
                int(v),
                weight=float(w),
                primary_arc_id=int(aid),
                arc_ids=[int(aid)],
            )

    # 5. Optional ground-truth route alignment (strictly disabled during blind inference)
    route_arcs = None
    gt_arc_ids = None
    mean_align_dist = None
    if include_ground_truth and route_df is not None:
        route_arcs = route_df["arc_id"].to_numpy(dtype=np.int64)
        gt_arc_ids, mean_align_dist = align_gps_to_route_dp(
            gx, gy, route_arcs, arcs_df
        )

    return TripData(
        trip_id=trip_id,
        epsg=epsg,
        track_df=track_df,
        nodes_xy=nodes_xy,
        arcs_df=arcs_df,
        graph=G,
        route_arcs=route_arcs,
        gt_arc_ids=gt_arc_ids,
        alignment_mean_dist_m=mean_align_dist,
    )