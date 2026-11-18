import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer


def discover_trip_files(trip_dir: Path) -> Dict[str, Optional[Path]]:
    """Locate .track, .nodes, .arcs, and .route files inside a trip directory."""
    discovered: Dict[str, Optional[Path]] = {
        "track": None,
        "nodes": None,
        "arcs": None,
        "route": None,
    }
    for f in trip_dir.iterdir():
        if not f.is_file():
            continue
        nl = f.name.lower()
        if "track" in nl:
            discovered["track"] = f
        elif "node" in nl:
            discovered["nodes"] = f
        elif "arc" in nl:
            discovered["arcs"] = f
        elif "route" in nl:
            discovered["route"] = f
    return discovered


def infer_trip_utm_epsg(lons: np.ndarray, lats: np.ndarray) -> Tuple[int, str]:
    """Compute local WGS84 UTM EPSG code from a trip's centroid longitude and latitude."""
    mean_lon = float(np.nanmean(lons))
    mean_lat = float(np.nanmean(lats))
    zone = max(1, min(60, int(math.floor((mean_lon + 180.0) / 6.0)) + 1))
    epsg = (32600 + zone) if mean_lat >= 0.0 else (32700 + zone)
    hemi = "N" if mean_lat >= 0.0 else "S"
    return epsg, f"UTM {zone}{hemi} (EPSG:{epsg})"


def min_dist_to_route_arcs(
    gx: np.ndarray,
    gy: np.ndarray,
    rx1: np.ndarray,
    ry1: np.ndarray,
    rx2: np.ndarray,
    ry2: np.ndarray,
) -> np.ndarray:
    """
    Compute unbiased minimum Euclidean distance (meters) from every GPS observation
    to any segment in the trip's reference .route sequence.
    """
    if len(rx1) == 0:
        return np.full(len(gx), np.nan, dtype=np.float64)

    dx = (rx2 - rx1)[None, :]
    dy = (ry2 - ry1)[None, :]
    seg_len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)

    px = gx[:, None]
    py = gy[:, None]
    t_raw = ((px - rx1[None, :]) * dx + (py - ry1[None, :]) * dy) / safe_len_sq
    t_proj = np.where(seg_len_sq > 1e-12, np.clip(t_raw, 0.0, 1.0), 0.0)
    proj_x = rx1[None, :] + t_proj * dx
    proj_y = ry1[None, :] + t_proj * dy
    dist_mat = np.hypot(px - proj_x, py - proj_y)
    return np.min(dist_mat, axis=1)


def generate_diagnostic_folium_map(
    partition: str,
    trip_id: str,
    label_tag: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame],
    output_html: Path,
) -> None:
    """Generate an interactive Folium map using CartoDB Positron tiles (no 403 local file blocks)."""
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri WorldStreetMap",
        )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri WorldImagery",
        name="Satellite Imagery",
        ).add_to(m)
    
    pad = 0.004
    min_lon, max_lon = track_df["lon"].min() - pad, track_df["lon"].max() + pad
    min_lat, max_lat = track_df["lat"].min() - pad, track_df["lat"].max() + pad

    node_lat = nodes_df["lat"].to_numpy()
    node_lon = nodes_df["lon"].to_numpy()
    n_nodes = len(nodes_df)

    # Layer 1: Local Road Network Geometry
    road_group = folium.FeatureGroup(name="Road Network (Local Arcs)", show=True)
    drawn_arcs = 0
    for arc_idx, row in arcs_df.iterrows():
        u, v = int(row["from_node"]), int(row["to_node"])
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            lat1, lon1 = float(node_lat[u]), float(node_lon[u])
            lat2, lon2 = float(node_lat[v]), float(node_lon[v])
            if (min_lon <= lon1 <= max_lon and min_lat <= lat1 <= max_lat) or (
                min_lon <= lon2 <= max_lon and min_lat <= lat2 <= max_lat
            ):
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color="#7f8c8d",
                    weight=2,
                    opacity=0.45,
                    tooltip=f"arc_id: {arc_idx} ({u}->{v})",
                ).add_to(road_group)
                drawn_arcs += 1
                if drawn_arcs >= 4500:
                    break
    road_group.add_to(m)

    # Layer 2: Reference Route (.route)
    if route_df is not None and len(route_df) > 0:
        route_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        n_arcs = len(arcs_df)
        for seq_idx, arc_id in enumerate(route_df["arc_id"].tolist()):
            if 0 <= arc_id < n_arcs:
                u = int(arcs_df.iloc[arc_id]["from_node"])
                v = int(arcs_df.iloc[arc_id]["to_node"])
                if 0 <= u < n_nodes and 0 <= v < n_nodes:
                    folium.PolyLine(
                        locations=[[float(node_lat[u]), float(node_lon[u])], [float(node_lat[v]), float(node_lon[v])]],
                        color="#1b9e3e",
                        weight=5,
                        opacity=0.85,
                        tooltip=f"Route Step {seq_idx} | arc_id: {arc_id}",
                    ).add_to(route_group)
        route_group.add_to(m)

    # Layer 3: Raw GPS Observations & Time-Gap Jumps (>5s highlighted in red dashed lines)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    gap_group = folium.FeatureGroup(name="GPS Time Gaps (>5s)", show=True)

    step = max(1, len(track_df) // 2500)
    for idx in range(0, len(track_df), step):
        row = track_df.iloc[idx]
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=3,
            color="#0055ff",
            fill=True,
            fill_color="#0055ff",
            fill_opacity=0.8,
            tooltip=f"Obs {idx} | t={row['timestamp']}s",
        ).add_to(gps_group)

    dt_vals = track_df["timestamp"].diff().fillna(1.0).to_numpy()
    gap_indices = np.flatnonzero(dt_vals > 5.0)
    for g_idx in gap_indices:
        p1 = track_df.iloc[g_idx - 1]
        p2 = track_df.iloc[g_idx]
        folium.PolyLine(
            locations=[[float(p1["lat"]), float(p1["lon"])], [float(p2["lat"]), float(p2["lon"])]],
            color="#e31a1c",
            weight=3,
            dash_array="6, 6",
            opacity=0.9,
            tooltip=f"GPS Gap: Δt={dt_vals[g_idx]:.1f}s (Obs {g_idx-1} -> {g_idx})",
        ).add_to(gap_group)

    folium.Marker(
        location=[float(track_df.iloc[0]["lat"]), float(track_df.iloc[0]["lon"])],
        popup=f"{trip_id} START (t={track_df.iloc[0]['timestamp']}s)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(gps_group)
    folium.Marker(
        location=[float(track_df.iloc[-1]["lat"]), float(track_df.iloc[-1]["lon"])],
        popup=f"{trip_id} END (t={track_df.iloc[-1]['timestamp']}s)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(gps_group)

    gps_group.add_to(m)
    gap_group.add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background-color: white; padding: 10px 14px; border: 2px solid #444;
                border-radius: 6px; font-size: 13px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
      <b>PathSnap Stage 2 EDA — {partition}/{trip_id} ({label_tag})</b><br>
      <span style="color:#0055ff;">&#9679;</span> Raw GPS Observations ({len(track_df)} pts)<br>
      <span style="color:#1b9e3e;">&#9644;</span> Reference Route (.route)<br>
      <span style="color:#e31a1c;">&#9476;</span> GPS Time Gaps &gt;5s ({len(gap_indices)} gaps)<br>
      <span style="color:#7f8c8d;">&#9644;</span> Local Road Network
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))


def inspect_partition(split_name: str, split_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Perform deep exploratory analysis on a permitted partition (train or student_test)."""
    trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    rows: List[Dict] = []
    all_dt: List[np.ndarray] = []
    all_speeds: List[np.ndarray] = []
    all_route_offsets: List[np.ndarray] = []
    all_arc_lengths: List[np.ndarray] = []
    norm_trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
    cached_trips: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]] = {}

    for td in trip_dirs:
        tid = td.name
        files = discover_trip_files(td)

        track_sz = files["track"].stat().st_size if files["track"] else 0
        nodes_sz = files["nodes"].stat().st_size if files["nodes"] else 0
        arcs_sz = files["arcs"].stat().st_size if files["arcs"] else 0
        route_sz = files["route"].stat().st_size if files["route"] else 0

        track_df = pd.read_csv(files["track"], sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
        nodes_df = pd.read_csv(files["nodes"], sep=r"\s+", header=None, names=["lon", "lat"])
        arcs_df = pd.read_csv(files["arcs"], sep=r"\s+", header=None, names=["from_node", "to_node"])
        route_df = (
            pd.read_csv(files["route"], sep=r"\s+", header=None, names=["arc_id"])
            if files["route"] is not None and route_sz > 0
            else None
        )

        # Data quality & timestamp checks
        missing_vals = int(track_df.isna().sum().sum() + nodes_df.isna().sum().sum() + arcs_df.isna().sum().sum())
        invalid_coords = int(
            ((track_df["lon"].abs() > 180) | (track_df["lat"].abs() > 90)).sum()
            + ((track_df["lon"] == 0) & (track_df["lat"] == 0)).sum()
        )
        dt = track_df["timestamp"].diff().dropna().to_numpy(dtype=np.float64)
        all_dt.append(dt)

        non_mono = int(np.sum(dt < 0))
        dup_ts = int(np.sum(dt == 0))
        n_1s = int(np.sum(dt == 1.0))
        n_2_5s = int(np.sum((dt > 1.0) & (dt <= 5.0)))
        n_6_30s = int(np.sum((dt > 5.0) & (dt <= 30.0)))
        n_gt_30s = int(np.sum(dt > 30.0))
        max_gap = float(np.max(dt)) if len(dt) > 0 else 0.0

        # Project to trip's local UTM CRS for accurate physical measurements
        epsg, utm_label = infer_trip_utm_epsg(track_df["lon"].to_numpy(), track_df["lat"].to_numpy())
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        gx, gy = transformer.transform(track_df["lon"].to_numpy(), track_df["lat"].to_numpy())
        nx_arr, ny_arr = transformer.transform(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy())

        # Origin-centered trajectory shape (km) for trajectory pattern visualization
        norm_trajectories.append(((gx - gx[0]) / 1000.0, (gy - gy[0]) / 1000.0))

        step_dist = np.hypot(np.diff(gx), np.diff(gy))
        traj_len_km = float(np.sum(step_dist) / 1000.0)
        dt_safe = np.where(dt > 0, dt, 1.0)
        speeds = step_dist / dt_safe
        all_speeds.append(speeds)
        speed_spikes_gt_45mps = int(np.sum(speeds > 45.0))

        bbox_w_km = float((np.max(gx) - np.min(gx)) / 1000.0)
        bbox_h_km = float((np.max(gy) - np.min(gy)) / 1000.0)

        # Road network topology & arc lengths
        u_idx = arcs_df["from_node"].to_numpy(dtype=np.int64)
        v_idx = arcs_df["to_node"].to_numpy(dtype=np.int64)
        oob_arcs = int(np.sum((u_idx < 0) | (u_idx >= len(nodes_df)) | (v_idx < 0) | (v_idx >= len(nodes_df))))
        parallel_arcs = int(len(arcs_df) - len(arcs_df.drop_duplicates(subset=["from_node", "to_node"])))

        arc_lens = np.hypot(nx_arr[v_idx] - nx_arr[u_idx], ny_arr[v_idx] - ny_arr[u_idx])
        all_arc_lengths.append(arc_lens)
        zero_len_arcs = int(np.sum(arc_lens < 1e-3))

        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes_df)))
        G.add_edges_from(zip(u_idx.tolist(), v_idx.tolist()))
        isolated_nodes = nx.number_of_isolates(G)
        weak_cc = nx.number_weakly_connected_components(G)
        strong_cc = nx.number_strongly_connected_components(G)

        # Reference route structure & unbiased GPS-to-route error
        n_route_arcs = len(route_df) if route_df is not None else 0
        unique_route_arcs = int(route_df["arc_id"].nunique()) if route_df is not None else 0
        has_revisited_arcs = unique_route_arcs < n_route_arcs
        route_len_km = 0.0
        route_discontinuities = 0
        route_offset_median_m = np.nan
        route_offset_p95_m = np.nan
        pct_gps_gt_75m_from_route = np.nan

        if route_df is not None and n_route_arcs > 0:
            r_ids = route_df["arc_id"].to_numpy(dtype=np.int64)
            route_len_km = float(np.sum(arc_lens[r_ids]) / 1000.0)

            for i in range(len(r_ids) - 1):
                a1, a2 = int(r_ids[i]), int(r_ids[i + 1])
                if a1 != a2 and int(v_idx[a1]) != int(u_idx[a2]):
                    route_discontinuities += 1

            offsets = min_dist_to_route_arcs(
                gx,
                gy,
                nx_arr[u_idx[r_ids]],
                ny_arr[u_idx[r_ids]],
                nx_arr[v_idx[r_ids]],
                ny_arr[v_idx[r_ids]],
            )
            all_route_offsets.append(offsets)
            route_offset_median_m = float(np.median(offsets))
            route_offset_p95_m = float(np.percentile(offsets, 95))
            pct_gps_gt_75m_from_route = float(100.0 * np.mean(offsets > 75.0))

        rows.append(
            {
                "partition": split_name,
                "trip_id": tid,
                "centroid_lon": round(float(track_df["lon"].mean()), 5),
                "centroid_lat": round(float(track_df["lat"].mean()), 5),
                "utm_epsg": epsg,
                "utm_zone": utm_label,
                "track_kb": round(track_sz / 1024.0, 1),
                "nodes_kb": round(nodes_sz / 1024.0, 1),
                "arcs_kb": round(arcs_sz / 1024.0, 1),
                "route_kb": round(route_sz / 1024.0, 1),
                "n_gps": len(track_df),
                "n_nodes": len(nodes_df),
                "n_arcs": len(arcs_df),
                "n_route_arcs": n_route_arcs,
                "unique_route_arcs": unique_route_arcs,
                "has_revisited_arcs": has_revisited_arcs,
                "missing_vals": missing_vals,
                "invalid_coords": invalid_coords,
                "non_mono_ts": non_mono,
                "dup_ts": dup_ts,
                "steps_1s": n_1s,
                "gaps_2_5s": n_2_5s,
                "gaps_6_30s": n_6_30s,
                "gaps_gt_30s": n_gt_30s,
                "max_gap_s": max_gap,
                "traj_len_km": round(traj_len_km, 3),
                "route_len_km": round(route_len_km, 3),
                "bbox_w_km": round(bbox_w_km, 3),
                "bbox_h_km": round(bbox_h_km, 3),
                "speed_spikes_gt_45mps": speed_spikes_gt_45mps,
                "oob_arcs": oob_arcs,
                "parallel_arcs": parallel_arcs,
                "zero_len_arcs": zero_len_arcs,
                "isolated_nodes": isolated_nodes,
                "weak_components": weak_cc,
                "strong_components": strong_cc,
                "route_discontinuities": route_discontinuities,
                "route_offset_median_m": round(route_offset_median_m, 2),
                "route_offset_p95_m": round(route_offset_p95_m, 2),
                "pct_gps_gt_75m_from_route": round(pct_gps_gt_75m_from_route, 3),
            }
        )
        cached_trips[tid] = (track_df, nodes_df, arcs_df, route_df)

    inv_df = pd.DataFrame(rows)
    stats_bundle = {
        "dt": np.concatenate(all_dt),
        "speeds": np.concatenate(all_speeds),
        "route_offsets": np.concatenate(all_route_offsets) if all_route_offsets else np.array([]),
        "arc_lengths": np.concatenate(all_arc_lengths),
        "norm_trajectories": norm_trajectories,
        "cached_trips": cached_trips,
    }
    return inv_df, stats_bundle


def run_stage2_eda() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    logs_dir = root / "logs"
    vis_dir = root / "visualizations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Count partitions (strictly do NOT read eval_test files!)
    eval_trips = [d for d in (root / "data" / "raw" / "eval_test").iterdir() if d.is_dir()]

    print("=" * 78)
    print("PathSnap Stage 2: Deep Dataset Exploration, Anomaly Audit & Visualisations")
    print(f"Partition Inventory -> train: 80 | student_test: 10 | eval_test: {len(eval_trips)} (UNTOUCHED)")
    print("=" * 78)

    train_df, train_stats = inspect_partition("train", root / "data" / "raw" / "train")
    stu_df, stu_stats = inspect_partition("student_test", root / "data" / "raw" / "student_test")

    full_inv = pd.concat([train_df, stu_df], ignore_index=True)
    inv_csv = logs_dir / "stage2_dataset_inventory.csv"
    full_inv.to_csv(inv_csv, index=False)

    anomalous_train = train_df[
        (train_df["gaps_gt_30s"] > 0)
        | (train_df["route_discontinuities"] > 0)
        | (train_df["pct_gps_gt_75m_from_route"] > 5.0)
    ].sort_values(by=["pct_gps_gt_75m_from_route", "gaps_gt_30s"], ascending=False)

    print("\n--- TRAIN (80 TRIPS) vs. STUDENT-TEST (10 TRIPS) DIAGNOSTIC COMPARISON ---")
    for name, df, st in [("TRAIN (80)", train_df, train_stats), ("STUDENT_TEST (10)", stu_df, stu_stats)]:
        off = st["route_offsets"]
        dt = st["dt"]
        print(f"\n[{name}]")
        print(
            f"  Total GPS Observations      : {int(df['n_gps'].sum()):,} "
            f"(min={df['n_gps'].min()}, med={df['n_gps'].median():.0f}, max={df['n_gps'].max()})"
        )
        print(
            f"  Exact 1s Steps / Gaps >5s   : {100.0*np.mean(dt==1.0):.2f}% 1s | "
            f"{int(np.sum(dt>5.0))} gaps >5s | max_gap={np.max(dt):.1f}s"
        )
        print(
            f"  Road Network Size (Median)  : {df['n_nodes'].median():.0f} nodes | "
            f"{df['n_arcs'].median():.0f} arcs | parallel_arcs_total={int(df['parallel_arcs'].sum())}"
        )
        print(
            f"  Trips with Revisited Arcs   : {int(df['has_revisited_arcs'].sum())} / {len(df)} trips (loops/round-trips)"
        )
        print(
            f"  Route Discontinuities Total : {int(df['route_discontinuities'].sum())} across "
            f"{int((df['route_discontinuities']>0).sum())} trips"
        )
        print(
            f"  Unbiased GPS-to-Route Error : median={np.median(off):.2f}m | "
            f"p95={np.percentile(off, 95):.2f}m | >75m: {100.0*np.mean(off>75.0):.3f}%"
        )

    print(f"\n--- TOP ANOMALOUS TRAINING TRIPS ({len(anomalous_train)} trips flagged) ---")
    if not anomalous_train.empty:
        cols = [
            "trip_id",
            "n_gps",
            "gaps_6_30s",
            "gaps_gt_30s",
            "max_gap_s",
            "route_discontinuities",
            "route_offset_median_m",
            "pct_gps_gt_75m_from_route",
        ]
        print(anomalous_train[cols].head(12).to_string(index=False))

    summary_payload = {
        "partitions": {"train": 80, "student_test": 10, "eval_test_untouched": len(eval_trips)},
        "train_metrics": {
            "total_gps": int(train_df["n_gps"].sum()),
            "pct_1s_steps": round(float(100.0 * np.mean(train_stats["dt"] == 1.0)), 3),
            "unbiased_gps_to_route_median_m": round(float(np.median(train_stats["route_offsets"])), 3),
            "unbiased_gps_to_route_p95_m": round(float(np.percentile(train_stats["route_offsets"], 95)), 3),
            "pct_gps_gt_75m_from_route": round(float(100.0 * np.mean(train_stats["route_offsets"] > 75.0)), 3),
            "trips_with_revisited_arcs": int(train_df["has_revisited_arcs"].sum()),
            "anomalous_trips_count": int(len(anomalous_train)),
        },
        "student_test_metrics": {
            "total_gps": int(stu_df["n_gps"].sum()),
            "pct_1s_steps": round(float(100.0 * np.mean(stu_stats["dt"] == 1.0)), 3),
            "unbiased_gps_to_route_median_m": round(float(np.median(stu_stats["route_offsets"])), 3),
            "unbiased_gps_to_route_p95_m": round(float(np.percentile(stu_stats["route_offsets"], 95)), 3),
            "pct_gps_gt_75m_from_route": round(float(100.0 * np.mean(stu_stats["route_offsets"] > 75.0)), 3),
            "trips_with_revisited_arcs": int(stu_df["has_revisited_arcs"].sum()),
        },
        "data_quality_findings_and_decisions": [
            "1. Global Multi-Continent Span: 25+ UTM zones detected; dynamic per-trip UTM CRS projection is mandatory.",
            "2. Path-Level Reference Routes & Loops: .route lists traversed arc_ids sequentially (not 1-to-1 per GPS ping) and contains revisited arcs in multiple trips.",
            "3. Multi-Minute/Multi-Hour GPS Dropouts: Training trips contain >30s gaps and >75m GPS-to-route dropouts; Stage 3 DP ground-truth aligner must scale allowed arc jumps dynamically with delta_t and displacement rather than capping at 8 arcs.",
            "4. 1 Hz GPS Jitter (~4-6m median): Single-second consecutive headings are noisy; Stage 3 must engineer multi-second smoothed bearings, multi-step trajectory consistency, and 1-step graph reachability features.",
        ],
    }
    summary_json = logs_dir / "stage2_eda_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Plot 1: Geographical Coordinate Distributions & Trajectory Patterns
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9.5))
    fig1.suptitle("PathSnap Stage 2 EDA: Geographical Extent & Trajectory Patterns", fontsize=13, fontweight="bold")

    axes1[0, 0].scatter(train_df["centroid_lon"], train_df["centroid_lat"], c="#2b5c8f", alpha=0.8, s=45, label="Train (80)")
    axes1[0, 0].scatter(stu_df["centroid_lon"], stu_df["centroid_lat"], c="#1b9e3e", marker="^", s=75, label="Student-Test (10)")
    axes1[0, 0].set_title("Global Trip Centroid Coordinates (Lon / Lat)")
    axes1[0, 0].set_xlabel("Longitude (°)")
    axes1[0, 0].set_ylabel("Latitude (°)")
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].legend()

    for nx_km, ny_km in train_stats["norm_trajectories"][:40]:
        axes1[0, 1].plot(nx_km, ny_km, alpha=0.35, linewidth=1.0, color="#2b5c8f")
    for nx_km, ny_km in stu_stats["norm_trajectories"]:
        axes1[0, 1].plot(nx_km, ny_km, alpha=0.75, linewidth=1.4, color="#1b9e3e")
    axes1[0, 1].set_title("Origin-Aligned Trajectory Shapes (km from Start)")
    axes1[0, 1].set_xlabel("East Offset ΔX (km)")
    axes1[0, 1].set_ylabel("North Offset ΔY (km)")
    axes1[0, 1].grid(True, alpha=0.3)

    axes1[1, 0].hist(train_df["n_gps"], bins=20, alpha=0.75, label="Train (80)", color="#2b5c8f", edgecolor="white")
    axes1[1, 0].hist(stu_df["n_gps"], bins=10, alpha=0.75, label="Student-Test (10)", color="#1b9e3e", edgecolor="white")
    axes1[1, 0].set_title("GPS Observations per Trip")
    axes1[1, 0].set_xlabel("GPS Observation Count")
    axes1[1, 0].set_ylabel("Trips")
    axes1[1, 0].legend()

    axes1[1, 1].scatter(train_df["traj_len_km"], train_df["route_len_km"], alpha=0.75, color="#2b5c8f", label="Train")
    axes1[1, 1].scatter(stu_df["traj_len_km"], stu_df["route_len_km"], alpha=0.9, color="#1b9e3e", marker="^", s=65, label="Student-Test")
    max_km = max(train_df["traj_len_km"].max(), train_df["route_len_km"].max())
    axes1[1, 1].plot([0, max_km], [0, max_km], "k--", alpha=0.5, label="y = x (Ideal)")
    axes1[1, 1].set_title("Raw GPS Trajectory Length vs. Reference Route Length (km)")
    axes1[1, 1].set_xlabel("Raw GPS Cumulative Distance (km)")
    axes1[1, 1].set_ylabel("Reference Route Length (km)")
    axes1[1, 1].legend()

    plt.tight_layout()
    plot1_path = vis_dir / "stage2_geo_and_trajectories.png"
    plt.savefig(plot1_path, dpi=180)
    plt.close(fig1)

    # Plot 2: Road Network Connectivity, Sampling Gaps & Unbiased GPS-to-Route Error
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9.5))
    fig2.suptitle("PathSnap Stage 2 EDA: Network Connectivity, Gaps & GPS-to-Route Error", fontsize=13, fontweight="bold")

    axes2[0, 0].scatter(train_df["n_nodes"], train_df["n_arcs"], alpha=0.75, color="#2b5c8f", label="Train")
    axes2[0, 0].scatter(stu_df["n_nodes"], stu_df["n_arcs"], alpha=0.9, color="#1b9e3e", marker="^", s=65, label="Student-Test")
    axes2[0, 0].set_title("Road Network Graph Size (Nodes vs. Directed Arcs)")
    axes2[0, 0].set_xlabel("Nodes")
    axes2[0, 0].set_ylabel("Arcs")
    axes2[0, 0].legend()

    axes2[0, 1].hist(train_df["weak_components"], bins=20, color="#e7298a", edgecolor="white")
    axes2[0, 1].set_title("Weakly Connected Components per Trip Road Graph")
    axes2[0, 1].set_xlabel("Weakly Connected Components")
    axes2[0, 1].set_ylabel("Trip Count")

    dt_clipped = np.clip(train_stats["dt"], 0, 20)
    axes2[1, 0].hist(dt_clipped, bins=20, color="#d95f02", edgecolor="white")
    axes2[1, 0].set_yscale("log")
    axes2[1, 0].set_title("Sampling Interval Δt Distribution (Log Scale, Capped at 20s)")
    axes2[1, 0].set_xlabel("Δt (seconds)")
    axes2[1, 0].set_ylabel("Step Count (log)")

    off_train_clip = np.clip(train_stats["route_offsets"], 0, 100)
    off_stu_clip = np.clip(stu_stats["route_offsets"], 0, 100)
    axes2[1, 1].hist(off_train_clip, bins=40, density=True, alpha=0.65, color="#2b5c8f", label="Train (80)")
    axes2[1, 1].hist(off_stu_clip, bins=40, density=True, alpha=0.65, color="#1b9e3e", label="Student-Test (10)")
    axes2[1, 1].axvline(75.0, color="red", linestyle="--", linewidth=1.5, label="75m Candidate Radius")
    axes2[1, 1].set_title("Unbiased GPS-to-Reference-Route Distance Error (m)")
    axes2[1, 1].set_xlabel("Distance from GPS Ping to Nearest .route Arc (m)")
    axes2[1, 1].set_ylabel("Density")
    axes2[1, 1].legend()

    plt.tight_layout()
    plot2_path = vis_dir / "stage2_network_and_quality.png"
    plt.savefig(plot2_path, dpi=180)
    plt.close(fig2)

    # Interactive Folium Maps: 1 Clean Reference Trip + 1 Anomalous Gap Trip
    clean_candidates = train_df[
        (train_df["gaps_gt_30s"] == 0)
        & (train_df["route_discontinuities"] == 0)
        & (train_df["n_gps"].between(500, 2000))
    ]
    clean_tid = str(clean_candidates.iloc[0]["trip_id"]) if not clean_candidates.empty else str(train_df.iloc[0]["trip_id"])
    t_df, n_df, a_df, r_df = train_stats["cached_trips"][clean_tid]
    clean_map_path = vis_dir / f"stage2_map_clean_{clean_tid}.html"
    generate_diagnostic_folium_map("train", clean_tid, "Clean Reference Trip", t_df, n_df, a_df, r_df, clean_map_path)

    if not anomalous_train.empty:
        anom_tid = str(anomalous_train.iloc[0]["trip_id"])
        at_df, an_df, aa_df, ar_df = train_stats["cached_trips"][anom_tid]
        anom_map_path = vis_dir / f"stage2_map_anomaly_{anom_tid}.html"
        generate_diagnostic_folium_map("train", anom_tid, "Anomalous / Gap Trip", at_df, an_df, aa_df, ar_df, anom_map_path)
    else:
        anom_map_path = None

    print(f"\n[SAVED] Full trip inventory CSV      -> {inv_csv}")
    print(f"[SAVED] Stage 2 EDA summary JSON     -> {summary_json}")
    print(f"[SAVED] Geo & trajectory plots       -> {plot1_path}")
    print(f"[SAVED] Network & noise plots        -> {plot2_path}")
    print(f"[SAVED] Clean trip Folium map        -> {clean_map_path}")
    if anom_map_path:
        print(f"[SAVED] Anomalous trip Folium map    -> {anom_map_path}")


if __name__ == "__main__":
    run_stage2_eda()