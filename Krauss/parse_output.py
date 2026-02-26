"""
parse_output.py - Compute summary statistics COMPATIBLE with TomTom data

Strategy: aggregate SUMO edges over time (like TomTom does over the month),
then compute distributional statistics across edges/segments.

Both SUMO and TomTom produce:
  - One average speed per road segment
  - We compare the DISTRIBUTION of those speeds
"""
import numpy as np
import xml.etree.ElementTree as ET


def parse_edge_data(filepath):
    """Parse SUMO edgeData XML into dict of dicts per time bin"""
    tree = ET.parse(filepath)
    root = tree.getroot()

    intervals = {}
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        edges = {}
        for edge in interval.findall("edge"):
            edges[edge.get("id")] = {
                "speed": float(edge.get("speed", 0)),
                "density": float(edge.get("density", 0)),
                "waitingTime": float(edge.get("waitingTime", 0)),
                "timeLoss": float(edge.get("timeLoss", 0)),
                "entered": int(edge.get("entered", 0)),
                "occupancy": float(edge.get("occupancy", 0)),
            }
        intervals[t] = edges

    return intervals


def summary_statistics(edge_data_dict):
    """
    Compute summary stats that are COMPARABLE between SUMO and TomTom.

    Approach: aggregate each edge over all time bins (like TomTom aggregates
    over the month), then compute distribution stats across edges.

    Returns fixed-length numpy array.
    """
    time_bins = sorted(edge_data_dict.keys())
    if len(time_bins) == 0:
        return None

    # ============================================================
    # Step 1: Aggregate each edge over time
    # (same as what TomTom does over the month)
    # ============================================================
    edge_stats = {}  # edge_id → {speeds: [], densities: [], ...}

    for t in time_bins:
        edges = edge_data_dict[t]
        for eid, data in edges.items():
            if eid not in edge_stats:
                edge_stats[eid] = {
                    "speeds": [],
                    "densities": [],
                    "waitingTimes": [],
                    "timeLosses": [],
                    "vehicles": 0,
                }
            edge_stats[eid]["speeds"].append(data["speed"])
            edge_stats[eid]["densities"].append(data["density"])
            edge_stats[eid]["waitingTimes"].append(data["waitingTime"])
            edge_stats[eid]["timeLosses"].append(data["timeLoss"])
            edge_stats[eid]["vehicles"] += data["entered"]

    # Compute per-edge averages (like TomTom's averageSpeed per segment)
    edge_avg_speed = []      # mean speed per edge (like TomTom averageSpeed)
    edge_std_speed = []      # speed variability per edge (like TomTom stdSpeed)
    edge_avg_density = []
    edge_total_waiting = []
    edge_total_timeloss = []
    edge_vehicles = []

    for eid, data in edge_stats.items():
        if data["vehicles"] > 0:  # only edges with traffic
            edge_avg_speed.append(np.mean(data["speeds"]))
            edge_std_speed.append(np.std(data["speeds"]))
            edge_avg_density.append(np.mean(data["densities"]))
            edge_total_waiting.append(np.sum(data["waitingTimes"]))
            edge_total_timeloss.append(np.sum(data["timeLosses"]))
            edge_vehicles.append(data["vehicles"])

    edge_avg_speed = np.array(edge_avg_speed)
    edge_std_speed = np.array(edge_std_speed)
    edge_avg_density = np.array(edge_avg_density)
    edge_total_waiting = np.array(edge_total_waiting)
    edge_total_timeloss = np.array(edge_total_timeloss)
    edge_vehicles = np.array(edge_vehicles)

    if len(edge_avg_speed) == 0:
        return None

    # Convert to km/h for TomTom compatibility
    edge_avg_speed_kmh = edge_avg_speed * 3.6
    edge_std_speed_kmh = edge_std_speed * 3.6

    stats = []

    # ============================================================
    # 2. Speed distribution across edges (MATCHES TomTom directly)
    #    TomTom gives: averageSpeed per segment → we compare distributions
    # ============================================================

    # Speed percentiles (19 values — same as TomTom!)
    percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    speed_pcts = np.percentile(edge_avg_speed_kmh, percentiles)
    stats.extend(speed_pcts.tolist())                    # 19 values

    # Basic speed stats
    stats.append(np.mean(edge_avg_speed_kmh))            # 1
    stats.append(np.std(edge_avg_speed_kmh))              # 1
    stats.append(np.median(edge_avg_speed_kmh))           # 1

    # ============================================================
    # 3. Speed standard deviation distribution
    #    TomTom gives: standardDeviationSpeed per segment
    # ============================================================
    std_pcts = np.percentile(edge_std_speed_kmh, [10, 25, 50, 75, 90])
    stats.extend(std_pcts.tolist())                       # 5 values
    stats.append(np.mean(edge_std_speed_kmh))             # 1

    # ============================================================
    # 4. Speed bin fractions (histogram-like)
    #    What fraction of edges fall in each speed range?
    # ============================================================
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100]
    hist, _ = np.histogram(edge_avg_speed_kmh, bins=bins)
    hist_frac = hist / len(edge_avg_speed_kmh)
    stats.extend(hist_frac.tolist())                      # 11 values

    # ============================================================
    # 5. Density statistics (SUMO-only but helps identify tau)
    # ============================================================
    density_pcts = np.percentile(edge_avg_density, [10, 25, 50, 75, 90])
    stats.extend(density_pcts.tolist())                   # 5 values
    stats.append(np.mean(edge_avg_density))               # 1

    # ============================================================
    # 6. Waiting time / time loss (SUMO-only, helps identify tau & sigma)
    # ============================================================
    # Normalize by vehicles to make comparable
    waiting_per_veh = edge_total_waiting / np.maximum(edge_vehicles, 1)
    timeloss_per_veh = edge_total_timeloss / np.maximum(edge_vehicles, 1)

    stats.append(np.mean(waiting_per_veh))                # 1
    stats.append(np.median(waiting_per_veh))              # 1
    stats.append(np.mean(timeloss_per_veh))               # 1
    stats.append(np.median(timeloss_per_veh))             # 1
    stats.append(np.std(timeloss_per_veh))                # 1

    # ============================================================
    # 7. Network-level aggregates
    # ============================================================
    stats.append(float(len(edge_avg_speed)))              # n edges with traffic
    stats.append(float(np.sum(edge_vehicles)))            # total vehicles
    stats.append(float((edge_avg_speed_kmh < 10).mean())) # frac very slow
    stats.append(float((edge_avg_speed_kmh < 20).mean())) # frac slow
    stats.append(float((edge_avg_speed_kmh > 40).mean())) # frac fast

    return np.array(stats, dtype=np.float32)


def summary_statistics_tomtom(tomtom_evening_df):
    """
    Compute the SAME summary statistics from TomTom data.
    Input: DataFrame with columns from tomtom_evening_filtered.csv
    Returns: same-length numpy array as summary_statistics()
    """
    df = tomtom_evening_df.copy()

    # Filter to segments with reliable data
    df = df[df["sampleSize"] >= 30].copy()
    df = df[df["averageSpeed"].notna()].copy()

    speeds_kmh = df["averageSpeed"].values
    std_speeds_kmh = df["standardDeviationSpeed"].fillna(0).values

    if len(speeds_kmh) == 0:
        return None

    stats = []

    # 2. Speed percentiles (same 19 as SUMO)
    percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    speed_pcts = np.percentile(speeds_kmh, percentiles)
    stats.extend(speed_pcts.tolist())

    # Basic speed stats
    stats.append(np.mean(speeds_kmh))
    stats.append(np.std(speeds_kmh))
    stats.append(np.median(speeds_kmh))

    # 3. Speed std distribution
    std_pcts = np.percentile(std_speeds_kmh, [10, 25, 50, 75, 90])
    stats.extend(std_pcts.tolist())
    stats.append(np.mean(std_speeds_kmh))

    # 4. Speed bin fractions
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100]
    hist, _ = np.histogram(speeds_kmh, bins=bins)
    hist_frac = hist / len(speeds_kmh)
    stats.extend(hist_frac.tolist())

    # 5. Density — TomTom doesn't have this, use zeros
    stats.extend([0.0] * 5)  # density percentiles
    stats.append(0.0)         # mean density

    # 6. Waiting/timeloss — TomTom doesn't have this directly
    # Use travelTimeRatio as proxy
    tt_ratio = df["travelTimeRatio"].fillna(1.0).values
    stats.append(np.mean(tt_ratio))        # proxy for waiting
    stats.append(np.median(tt_ratio))
    stats.append(np.mean(tt_ratio - 1.0))  # excess time = proxy for timeloss
    stats.append(np.median(tt_ratio - 1.0))
    stats.append(np.std(tt_ratio))

    # 7. Network-level
    stats.append(float(len(speeds_kmh)))
    stats.append(0.0)  # total vehicles not available
    stats.append(float((speeds_kmh < 10).mean()))
    stats.append(float((speeds_kmh < 20).mean()))
    stats.append(float((speeds_kmh > 40).mean()))

    return np.array(stats, dtype=np.float32)