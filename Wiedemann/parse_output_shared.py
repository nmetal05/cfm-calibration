"""
parse_output_shared.py — FINAL VERSION

Summary statistics for speed + congestion, compatible between SUMO and TomTom.

Congestion = travelTimeRatio, where:
  TomTom: uses their measured free-flow times (travelTimeRatio field directly)
  SUMO:   uses maxSpeed per edge as free-flow speed proxy
"""
import numpy as np
import xml.etree.ElementTree as ET


def parse_edge_data(filepath):
    """Parse SUMO edgeData XML"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    intervals = {}
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        edges = {}
        for edge in interval.findall("edge"):
            edges[edge.get("id")] = {
                "speed": float(edge.get("speed", 0)),
                "traveltime": float(edge.get("traveltime", 0)),
                "entered": int(edge.get("entered", 0)),
                "density": float(edge.get("density", 0)),
            }
        intervals[t] = edges
    return intervals


def get_edge_max_speeds(net_file):
    """
    Get max allowed speed per edge from network file.
    This is the ACTUAL free-flow speed (not speed limit sign).
    In SUMO, lane speed = the max speed vehicles can go on that lane.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    edge_speeds = {}
    edge_lengths = {}
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid is None or eid.startswith(":"):
            continue

        max_speed = 0
        total_length = 0
        n_lanes = 0
        for lane in edge.findall("lane"):
            speed = float(lane.get("speed", 0))
            length = float(lane.get("length", 0))
            if speed > max_speed:
                max_speed = speed
            total_length = length  # all lanes same length
            n_lanes += 1

        if max_speed > 0:
            edge_speeds[eid] = max_speed
            edge_lengths[eid] = total_length

    return edge_speeds, edge_lengths


def summary_statistics_sumo(edge_data_dict, edge_max_speeds=None, edge_lengths=None):
    """
    SUMO -> summary stats (speed + congestion)
    
    Congestion: computed as actual_speed / max_speed ratio per edge,
    then inverted to get travel time ratio (max_speed / actual_speed)
    """
    time_bins = sorted(edge_data_dict.keys())
    if len(time_bins) == 0:
        return None

    # Aggregate each edge over time
    edge_data = {}
    for t in time_bins:
        for eid, data in edge_data_dict[t].items():
            if eid not in edge_data:
                edge_data[eid] = {"speeds": [], "vehicles": 0}
            edge_data[eid]["speeds"].append(data["speed"])
            edge_data[eid]["vehicles"] += data["entered"]

    # Per-edge stats
    avg_speeds_kmh = []
    std_speeds_kmh = []
    travel_time_ratios = []

    for eid, data in edge_data.items():
        if data["vehicles"] > 0:
            avg_speed_ms = np.mean(data["speeds"])
            avg_speeds_kmh.append(avg_speed_ms * 3.6)
            std_speeds_kmh.append(np.std(data["speeds"]) * 3.6)

            # Travel time ratio = freeflow_speed / actual_speed
            # freeflow_speed = edge max speed (what you'd drive with no traffic)
            if edge_max_speeds and eid in edge_max_speeds:
                freeflow_speed = edge_max_speeds[eid]  # m/s
                if avg_speed_ms > 0.1:
                    ttr = freeflow_speed / avg_speed_ms
                    ttr = min(ttr, 10.0)  # clamp extremes
                    travel_time_ratios.append(ttr)

    if len(avg_speeds_kmh) == 0:
        return None

    avg_speeds_kmh = np.array(avg_speeds_kmh)
    std_speeds_kmh = np.array(std_speeds_kmh)

    if len(travel_time_ratios) > 0:
        travel_time_ratios = np.array(travel_time_ratios)
    else:
        travel_time_ratios = np.array([1.0])

    return _compute_shared_stats(avg_speeds_kmh, std_speeds_kmh, travel_time_ratios)


def summary_statistics_tomtom(tomtom_df, min_samples=30):
    """
    TomTom -> summary stats (SAME format as SUMO)
    Uses TomTom's own travelTimeRatio (based on their measured free-flow)
    """
    df = tomtom_df[
        (tomtom_df["sampleSize"] >= min_samples) &
        (tomtom_df["averageSpeed"].notna())
    ].copy()

    if len(df) == 0:
        return None

    avg_speeds_kmh = df["averageSpeed"].values
    std_speeds_kmh = df["standardDeviationSpeed"].fillna(0).values

    # Use TomTom's travelTimeRatio directly
    # This is properly computed by TomTom using their actual free-flow measurements
    travel_time_ratios = df["travelTimeRatio"].fillna(1.0).values
    travel_time_ratios = np.clip(travel_time_ratios, 0.1, 10.0)

    return _compute_shared_stats(avg_speeds_kmh, std_speeds_kmh, travel_time_ratios)


def _compute_shared_stats(avg_speeds_kmh, std_speeds_kmh, travel_time_ratios):
    """
    IDENTICAL computation for both SUMO and TomTom.
    
    Block 1: Speed distribution (23 values)
    Block 2: Speed variability (6 values)
    Block 3: Speed histogram (11 values)
    Block 4: Congestion / TTR (12 values)
    Block 5: Speed thresholds (5 values)
    
    Total: 57 values
    """
    stats = []

    # ============================================================
    # BLOCK 1: SPEED DISTRIBUTION (23 values)
    # ============================================================
    percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    stats.extend(np.percentile(avg_speeds_kmh, percentiles).tolist())
    stats.append(np.mean(avg_speeds_kmh))
    stats.append(np.std(avg_speeds_kmh))
    stats.append(np.median(avg_speeds_kmh))
    stats.append(np.percentile(avg_speeds_kmh, 75) - np.percentile(avg_speeds_kmh, 25))

    # ============================================================
    # BLOCK 2: SPEED VARIABILITY (6 values)
    # ============================================================
    stats.extend(np.percentile(std_speeds_kmh, [10, 25, 50, 75, 90]).tolist())
    stats.append(np.mean(std_speeds_kmh))

    # ============================================================
    # BLOCK 3: SPEED HISTOGRAM (11 values)
    # ============================================================
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 120]
    hist, _ = np.histogram(avg_speeds_kmh, bins=bins)
    stats.extend((hist / len(avg_speeds_kmh)).tolist())

    # ============================================================
    # BLOCK 4: CONGESTION — Travel Time Ratio (12 values)
    # Follows TomTom definition:
    #   TTR = actual_travel_time / freeflow_travel_time
    #   congestion_level = (TTR - 1) * 100%
    # ============================================================

    # TTR percentiles
    stats.extend(np.percentile(travel_time_ratios, [10, 25, 50, 75, 90, 95]).tolist())

    # TTR basic stats
    stats.append(np.mean(travel_time_ratios))
    stats.append(np.std(travel_time_ratios))
    stats.append(np.median(travel_time_ratios))

    # Congestion level fractions (following TomTom's definition)
    congestion_pct = (travel_time_ratios - 1.0) * 100
    stats.append(float((congestion_pct > 10).mean()))    # >10% delay
    stats.append(float((congestion_pct > 25).mean()))    # >25% delay
    stats.append(float((congestion_pct > 50).mean()))    # >50% delay

    # ============================================================
    # BLOCK 5: SPEED THRESHOLDS (5 values)
    # ============================================================
    stats.append(float((avg_speeds_kmh < 10).mean()))
    stats.append(float((avg_speeds_kmh < 15).mean()))
    stats.append(float((avg_speeds_kmh < 20).mean()))
    stats.append(float((avg_speeds_kmh > 40).mean()))
    stats.append(float((avg_speeds_kmh > 50).mean()))

    return np.array(stats, dtype=np.float32)