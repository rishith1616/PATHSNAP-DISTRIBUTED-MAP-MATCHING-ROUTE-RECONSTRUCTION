import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def discover_trip_files(trip_dir: Path) -> Dict[str, Optional[Path]]:
    """Locate track, nodes, arcs, and route files inside a trip directory regardless of exact prefix/extension."""
    files = [f for f in trip_dir.iterdir() if f.is_file()]
    discovered: Dict[str, Optional[Path]] = {
        "track": None,
        "nodes": None,
        "arcs": None,
        "route": None,
    }
    for f in files:
        name_lower = f.name.lower()
        if "track" in name_lower:
            discovered["track"] = f
        elif "node" in name_lower:
            discovered["nodes"] = f
        elif "arc" in name_lower:
            discovered["arcs"] = f
        elif "route" in name_lower:
            discovered["route"] = f
    return discovered


def infer_utm_epsg(mean_lon: float, mean_lat: float) -> Tuple[int, str]:
    """Compute the appropriate WGS84 UTM EPSG code from mean longitude and latitude."""
    zone = int(math.floor((mean_lon + 180.0) / 6.0)) + 1
    if mean_lat >= 0:
        epsg = 32600 + zone
        hemi = "N"
    else:
        epsg = 32700 + zone
        hemi = "S"
    return epsg, f"WGS 84 / UTM zone {zone}{hemi} (EPSG:{epsg})"


def safe_read_whitespace_table(filepath: Optional[Path]) -> Optional[pd.DataFrame]:
    """Read an unheaded whitespace-separated file safely."""
    if filepath is None or not filepath.exists() or filepath.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    except Exception:
        return None


def generate_sample_folium_map(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_df: Optional[pd.DataFrame],
    output_html: Path,
) -> None:
    """Generate an interactive Folium map comparing road network, raw GPS, and reference route."""
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Padded bounding box around the GPS trajectory to keep road rendering fast
    pad = 0.005
    min_lon, max_lon = track_df["lon"].min() - pad, track_df["lon"].max() + pad
    min_lat, max_lat = track_df["lat"].min() - pad, track_df["lat"].max() + pad

    # Layer 1: Road Network Geometry
    road_group = folium.FeatureGroup(name="Road Network (Local Arcs)", show=True)
    node_coords = nodes_df[["lat", "lon"]].to_numpy()
    n_nodes = len(node_coords)

    for arc_idx, row in arcs_df.iterrows():
        u, v = int(row["from_node"]), int(row["to_node"])
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            lat1, lon1 = node_coords[u]
            lat2, lon2 = node_coords[v]
            if (
                (min_lon <= lon1 <= max_lon and min_lat <= lat1 <= max_lat)
                or (min_lon <= lon2 <= max_lon and min_lat <= lat2 <= max_lat)
            ):
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color="#7f8c8d",
                    weight=2,
                    opacity=0.55,
                    tooltip=f"arc_id: {arc_idx} ({u} -> {v})",
                ).add_to(road_group)
    road_group.add_to(m)

    # Layer 2: Reference Route (Ground Truth)
    if route_df is not None and len(route_df) > 0:
        route_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        n_arcs = len(arcs_df)
        for seq_idx, arc_id in enumerate(route_df["arc_id"].tolist()):
            if 0 <= arc_id < n_arcs:
                u = int(arcs_df.iloc[arc_id]["from_node"])
                v = int(arcs_df.iloc[arc_id]["to_node"])
                if 0 <= u < n_nodes and 0 <= v < n_nodes:
                    lat1, lon1 = node_coords[u]
                    lat2, lon2 = node_coords[v]
                    folium.PolyLine(
                        locations=[[lat1, lon1], [lat2, lon2]],
                        color="#1b9e3e",
                        weight=5,
                        opacity=0.85,
                        tooltip=f"Ref Step {seq_idx} | arc_id: {arc_id}",
                    ).add_to(route_group)
        route_group.add_to(m)

    # Layer 3: Raw GPS Observations
    gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    for idx, row in track_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#0055ff",
            fill=True,
            fill_color="#0055ff",
            fill_opacity=0.85,
            tooltip=f"Obs {idx} | t={row['timestamp']}s | ({row['lon']:.6f}, {row['lat']:.6f})",
        ).add_to(gps_group)

    # Start and End markers
    folium.Marker(
        location=[track_df.iloc[0]["lat"], track_df.iloc[0]["lon"]],
        popup=f"{trip_id} START (t={track_df.iloc[0]['timestamp']}s)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(gps_group)
    folium.Marker(
        location=[track_df.iloc[-1]["lat"], track_df.iloc[-1]["lon"]],
        popup=f"{trip_id} END (t={track_df.iloc[-1]['timestamp']}s)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(gps_group)
    gps_group.add_to(m)

    # Legend overlay
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background-color: white; padding: 10px 14px; border: 2px solid #444;
                border-radius: 6px; font-size: 13px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
      <b>PathSnap EDA — {trip_id} (Train Partition)</b><br>
      <span style="color:#0055ff;">&#9679;</span> Raw GPS Pings ({len(track_df)} pts)<br>
      <span style="color:#1b9e3e;">&#9644;</span> Reference Route<br>
      <span style="color:#7f8c8d;">&#9644;</span> Local Road Network Arcs
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))


def run_eda() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    train_dir = root / "data" / "raw" / "train"
    logs_dir = root / "logs"
    vis_dir = root / "visualizations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print("=" * 70)
    print(f"PathSnap Stage 1 EDA — Partition: TRAIN ({len(trip_dirs)} trips)")
    print("=" * 70)

    if not trip_dirs:
        print("[ERROR] No trip directories found inside data/raw/train/.")
        return

    # Print sample filenames from first 2 trips to confirm naming convention
    for sample_dir in trip_dirs[:2]:
        names = sorted([f.name for f in sample_dir.iterdir() if f.is_file()])
        print(f"[SCHEMA CHECK] {sample_dir.name}/ -> {names}")

    inventory_rows: List[Dict] = []
    all_dt: List[float] = []
    all_lons: List[float] = []
    all_lats: List[float] = []
    sample_trip_payload = None

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        files = discover_trip_files(trip_dir)

        track_df = safe_read_whitespace_table(files["track"])
        nodes_df = safe_read_whitespace_table(files["nodes"])
        arcs_df = safe_read_whitespace_table(files["arcs"])
        route_df = safe_read_whitespace_table(files["route"])

        row_info: Dict = {
            "trip_id": trip_id,
            "has_track": track_df is not None,
            "has_nodes": nodes_df is not None,
            "has_arcs": arcs_df is not None,
            "has_route": route_df is not None,
            "track_cols": track_df.shape[1] if track_df is not None else 0,
            "nodes_cols": nodes_df.shape[1] if nodes_df is not None else 0,
            "arcs_cols": arcs_df.shape[1] if arcs_df is not None else 0,
            "route_cols": route_df.shape[1] if route_df is not None else 0,
            "n_gps": len(track_df) if track_df is not None else 0,
            "n_nodes": len(nodes_df) if nodes_df is not None else 0,
            "n_arcs": len(arcs_df) if arcs_df is not None else 0,
            "n_route": len(route_df) if route_df is not None else 0,
        }

        if track_df is not None and track_df.shape[1] >= 3:
            track_df.columns = ["lon", "lat", "timestamp"] + [
                f"extra_{i}" for i in range(track_df.shape[1] - 3)
            ]
            invalid_coords = int(
                track_df[["lon", "lat", "timestamp"]].isna().any(axis=1).sum()
                + ((track_df["lon"].abs() > 180) | (track_df["lat"].abs() > 90)).sum()
                + ((track_df["lon"] == 0) & (track_df["lat"] == 0)).sum()
            )
            dt = track_df["timestamp"].diff().dropna()
            non_monotonic = int((dt < 0).sum())
            duplicate_ts = int((dt == 0).sum())
            steps_1s = int((dt == 1).sum())
            max_gap = float(dt.max()) if len(dt) > 0 else 0.0
            gaps_gt_5s = int((dt > 5).sum())

            all_dt.extend(dt.tolist())
            all_lons.extend(track_df["lon"].dropna().tolist())
            all_lats.extend(track_df["lat"].dropna().tolist())

            row_info.update(
                {
                    "invalid_gps_rows": invalid_coords,
                    "non_monotonic_ts": non_monotonic,
                    "duplicate_ts": duplicate_ts,
                    "pct_1s_steps": round(100.0 * steps_1s / max(len(dt), 1), 2),
                    "max_time_gap_s": max_gap,
                    "gaps_gt_5s": gaps_gt_5s,
                    "lon_min": float(track_df["lon"].min()),
                    "lon_max": float(track_df["lon"].max()),
                    "lat_min": float(track_df["lat"].min()),
                    "lat_max": float(track_df["lat"].max()),
                }
            )

        # Graph & Route consistency checks
        if nodes_df is not None and arcs_df is not None and nodes_df.shape[1] >= 2 and arcs_df.shape[1] >= 2:
            nodes_df.columns = ["lon", "lat"] + [f"extra_{i}" for i in range(nodes_df.shape[1] - 2)]
            arcs_df.columns = ["from_node", "to_node"] + [f"extra_{i}" for i in range(arcs_df.shape[1] - 2)]

            G = nx.DiGraph()
            G.add_nodes_from(range(len(nodes_df)))
            valid_arc_mask = (
                (arcs_df["from_node"] >= 0)
                & (arcs_df["from_node"] < len(nodes_df))
                & (arcs_df["to_node"] >= 0)
                & (arcs_df["to_node"] < len(nodes_df))
            )
            out_of_bounds_arcs = int((~valid_arc_mask).sum())
            valid_arcs = arcs_df[valid_arc_mask]
            G.add_edges_from(zip(valid_arcs["from_node"], valid_arcs["to_node"]))

            isolated_nodes = nx.number_of_isolates(G)
            weak_components = nx.number_weakly_connected_components(G)
            strong_components = nx.number_strongly_connected_components(G)

            row_info.update(
                {
                    "out_of_bounds_arcs": out_of_bounds_arcs,
                    "isolated_nodes": isolated_nodes,
                    "weak_components": weak_components,
                    "strong_components": strong_components,
                }
            )

            if route_df is not None and route_df.shape[1] >= 1:
                route_df.columns = ["arc_id"] + [f"extra_{i}" for i in range(route_df.shape[1] - 1)]
                r_ids = route_df["arc_id"].to_numpy()
                oob_route = int(((r_ids < 0) | (r_ids >= len(arcs_df))).sum())

                # Count topological discontinuities between consecutive distinct arcs in route
                discontinuities = 0
                for i in range(len(r_ids) - 1):
                    a1, a2 = int(r_ids[i]), int(r_ids[i + 1])
                    if a1 != a2 and 0 <= a1 < len(arcs_df) and 0 <= a2 < len(arcs_df):
                        if int(arcs_df.iloc[a1]["to_node"]) != int(arcs_df.iloc[a2]["from_node"]):
                            discontinuities += 1

                row_info.update(
                    {
                        "route_equals_gps_len": len(route_df) == len(track_df),
                        "out_of_bounds_route_arcs": oob_route,
                        "route_discontinuities": discontinuities,
                    }
                )

                if sample_trip_payload is None and len(track_df) > 50:
                    sample_trip_payload = (trip_id, track_df.copy(), nodes_df.copy(), arcs_df.copy(), route_df.copy())

        inventory_rows.append(row_info)

    inv_df = pd.DataFrame(inventory_rows)
    inv_csv_path = logs_dir / "eda_trip_inventory.csv"
    inv_df.to_csv(inv_csv_path, index=False)

    mean_lon = float(np.mean(all_lons))
    mean_lat = float(np.mean(all_lats))
    utm_epsg, utm_desc = infer_utm_epsg(mean_lon, mean_lat)
    dt_arr = np.array(all_dt)

    summary = {
        "partition_analyzed": "train",
        "total_trips": int(len(inv_df)),
        "trips_with_route": int(inv_df["has_route"].sum()),
        "route_len_matches_gps_len_count": int(inv_df.get("route_equals_gps_len", pd.Series(dtype=bool)).sum()),
        "total_gps_observations": int(inv_df["n_gps"].sum()),
        "gps_per_trip_min": int(inv_df["n_gps"].min()),
        "gps_per_trip_median": float(inv_df["n_gps"].median()),
        "gps_per_trip_max": int(inv_df["n_gps"].max()),
        "invalid_gps_rows_total": int(inv_df["invalid_gps_rows"].sum()),
        "non_monotonic_ts_total": int(inv_df["non_monotonic_ts"].sum()),
        "duplicate_ts_total": int(inv_df["duplicate_ts"].sum()),
        "pct_steps_exactly_1s": round(float(100.0 * np.mean(dt_arr == 1.0)), 3),
        "max_time_gap_seconds": float(np.max(dt_arr)),
        "total_gaps_gt_5s": int(np.sum(dt_arr > 5.0)),
        "lon_bounds": [round(float(np.min(all_lons)), 6), round(float(np.max(all_lons)), 6)],
        "lat_bounds": [round(float(np.min(all_lats)), 6), round(float(np.max(all_lats)), 6)],
        "centroid_lon_lat": [round(mean_lon, 6), round(mean_lat, 6)],
        "recommended_utm_epsg": utm_epsg,
        "recommended_utm_name": utm_desc,
        "nodes_per_trip_median": float(inv_df["n_nodes"].median()),
        "arcs_per_trip_median": float(inv_df["n_arcs"].median()),
        "isolated_nodes_total": int(inv_df["isolated_nodes"].sum()),
        "weak_components_max": int(inv_df["weak_components"].max()),
        "route_discontinuities_total": int(inv_df.get("route_discontinuities", pd.Series([0])).sum()),
    }

    summary_path = logs_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print formatted summary report
    print("\n--- STAGE 1 EDA SUMMARY (TRAIN PARTITION) ---")
    for k, v in summary.items():
        print(f"  {k:<34}: {v}")

    # Generate 4-panel EDA diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("PathSnap Exploratory Data Analysis (Train Partition — 80 Trips)", fontsize=13, fontweight="bold")

    axes[0, 0].hist(inv_df["n_gps"], bins=20, color="#2b5c8f", edgecolor="white")
    axes[0, 0].set_title("GPS Observations per Trip")
    axes[0, 0].set_xlabel("Number of GPS Pings")
    axes[0, 0].set_ylabel("Trip Count")

    clipped_dt = dt_arr[(dt_arr >= 0) & (dt_arr <= 15)]
    axes[0, 1].hist(clipped_dt, bins=15, color="#1b9e3e", edgecolor="white")
    axes[0, 1].set_title("Sampling Interval Δt Distribution (0–15s)")
    axes[0, 1].set_xlabel("Δt (seconds)")
    axes[0, 1].set_ylabel("Step Count")

    axes[1, 0].scatter(inv_df["n_nodes"], inv_df["n_arcs"], alpha=0.75, color="#d95f02")
    axes[1, 0].set_title("Road Network Size per Trip (Nodes vs. Arcs)")
    axes[1, 0].set_xlabel("Node Count")
    axes[1, 0].set_ylabel("Arc Count")

    axes[1, 1].scatter(all_lons[::10], all_lats[::10], s=2, alpha=0.35, color="#7570b3")
    axes[1, 1].set_title(f"Spatial Coverage (Subsampled 1:10) | {utm_desc}")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")

    plt.tight_layout()
    plot_path = vis_dir / "eda_overview_plots.png"
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)

    # Generate interactive Folium map for the first valid training trip
    if sample_trip_payload is not None:
        s_id, s_track, s_nodes, s_arcs, s_route = sample_trip_payload
        map_path = vis_dir / f"eda_sample_{s_id}.html"
        generate_sample_folium_map(s_id, s_track, s_nodes, s_arcs, s_route, map_path)
        print(f"\n[SAVED] Interactive Folium map -> {map_path}")

    print(f"[SAVED] Trip inventory CSV     -> {inv_csv_path}")
    print(f"[SAVED] EDA summary JSON       -> {summary_path}")
    print(f"[SAVED] EDA overview plots     -> {plot_path}")


if __name__ == "__main__":
    run_eda()