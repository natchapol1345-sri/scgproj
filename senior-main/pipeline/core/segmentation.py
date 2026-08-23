"""Marker-based leg surface segmentation using HSV + Excess Green color detection.

Steps:
    1. Detect marker points via HSV color thresholding + Excess Green Index.
    2. Cluster markers spatially with DBSCAN.
    3. Fit a plane to each marker cluster via SVD to capture tilt/slant.
    4. Cut the point cloud using signed distance to the tilted marker planes.

Axis convention: After pipeline leveling (Stage 3), Z is the vertical axis.
Use ``height_axis="z"`` for post-leveled point clouds.
"""

import numpy as np
from sklearn.cluster import DBSCAN

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def rgb_to_hsv(r, g, b):
    """Vectorized RGB [0-255] -> HSV (H in degrees, S/V in %)."""
    r_n = r.astype(np.float32) / 255.0
    g_n = g.astype(np.float32) / 255.0
    b_n = b.astype(np.float32) / 255.0

    mx = np.maximum(np.maximum(r_n, g_n), b_n)
    mn = np.minimum(np.minimum(r_n, g_n), b_n)
    delta = mx - mn

    h = np.zeros_like(mx)
    mask_r = (mx == r_n) & (delta > 1e-9)
    mask_g = (mx == g_n) & (delta > 1e-9)
    mask_b = (mx == b_n) & (delta > 1e-9)

    h[mask_r] = (60.0 * ((g_n[mask_r] - b_n[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60.0 * ((b_n[mask_g] - r_n[mask_g]) / delta[mask_g]) + 120) % 360
    h[mask_b] = (60.0 * ((r_n[mask_b] - g_n[mask_b]) / delta[mask_b]) + 240) % 360

    s = np.zeros_like(mx)
    s[mx > 1e-9] = (delta[mx > 1e-9] / mx[mx > 1e-9]) * 100.0

    v = mx * 100.0
    return h, s, v


def detect_markers(colors_uint8):
    """Detect marker points using HSV + Excess Green Index.

    Rules:
        (Saturation > 15%) AND (Hue > 60)   -- HSV mask
        (2*G - R - B) > 10                  -- ExG mask

    Args:
        colors_uint8: (N, 3) numpy array of RGB colors [0-255].

    Returns:
        marker_mask: boolean (N,) array, True where markers are detected.
        stats: dict with diagnostic counts and colour statistics.
    """
    r = colors_uint8[:, 0].astype(np.int32)
    g = colors_uint8[:, 1].astype(np.int32)
    b = colors_uint8[:, 2].astype(np.int32)

    h, s, v = rgb_to_hsv(r, g, b)

    hsv_mask = (s > 15) & (h > 60)

    exg = 2 * g - r - b
    exg_mask = exg > 10

    marker_mask = hsv_mask | exg_mask

    stats = {
        "n_total": len(colors_uint8),
        "n_hsv": int(hsv_mask.sum()),
        "n_exg": int(exg_mask.sum()),
        "n_markers": int(marker_mask.sum()),
        "hsv_only": int((hsv_mask & ~exg_mask).sum()),
        "exg_only": int((exg_mask & ~hsv_mask).sum()),
        "both": int((hsv_mask & exg_mask).sum()),
    }

    if marker_mask.sum() > 0:
        stats["h_mean"] = float(h[hsv_mask].mean()) if hsv_mask.sum() > 0 else 0
        stats["s_mean"] = float(s[hsv_mask].mean()) if hsv_mask.sum() > 0 else 0
        stats["exg_mean"] = float(exg[exg_mask].mean()) if exg_mask.sum() > 0 else 0
        stats["mr"], stats["mg"], stats["mb"] = (
            float(r[marker_mask].mean()),
            float(g[marker_mask].mean()),
            float(b[marker_mask].mean()),
        )

    return marker_mask, stats


def cluster_markers(coords, eps=0.03, min_samples=10):
    """Spatial DBSCAN clustering of marker points.

    Args:
        coords: (M, 3) numpy array of marker point coordinates.
        eps: DBSCAN epsilon (default 0.03 ~ 3 cm).
        min_samples: Minimum samples per cluster (default 10).

    Returns:
        labels: cluster label for each point (-1 = noise).
        n_clusters: number of valid clusters (excluding noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def compute_cluster_planes(coords, labels, colors_uint8, axis_idx, min_cluster_size=150):
    """Compute per-cluster centroid and SVD-based plane normal.

    For each marker cluster, fits a best-fit plane via Singular Value
    Decomposition.  The normal vector captures the tilt/slant of the
    marker surface.

    Returns list of (cluster_id, centroid, normal, n_points, color_mean)
    sorted by centroid position along the height axis ascending.
    Noise (label == -1) and clusters smaller than min_cluster_size are
    excluded.
    """
    planes = []
    skipped = 0
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        npts = mask.sum()
        if npts < min_cluster_size:
            skipped += 1
            continue
        cluster_coords = coords[mask]
        centroid = cluster_coords.mean(axis=0)

        centered = cluster_coords - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]

        if normal[axis_idx] < 0:
            normal = -normal

        color_mean = colors_uint8[mask].mean(axis=0)
        planes.append((cid, centroid, normal, npts, color_mean))

    if skipped:
        print(f"    (skipped {skipped} cluster(s) with < {min_cluster_size} points)")

    planes.sort(key=lambda x: x[1][axis_idx])
    return planes


def cut_surface_plane(coords, planes, axis_idx, axis_name="Z"):
    """Create keep-mask based on signed distance to tilted marker planes.

    Uses the SVD-fitted plane for each marker cluster so that slanted
    and tilted markers are handled correctly.

    For point P and plane (centroid C, unit normal N):
        signed_distance(P) = dot(P - C, N)

    Normal is oriented so that positive N[axis_idx] points "upward".
    Therefore:
        distance < 0  ->  below the plane
        distance > 0  ->  above the plane

    Case logic:
        0 markers -> keep all points
        1 marker  -> keep points *below* the tilted plane (distance < 0)
        2 markers -> keep points *above* bottom plane (dist > 0) AND
                     *below* top plane (dist < 0)
        3+ markers -> keep points between lowest and highest marker planes

    Returns boolean mask of points to keep.
    """
    n = len(planes)

    if n == 0:
        return np.ones(len(coords), dtype=bool), {"case": 0}

    if n == 1:
        cid, centroid, normal, npts, clr = planes[0]

        heights = coords[:, axis_idx]
        min_h = heights.min()
        max_h = heights.max()
        threshold = min_h + 0.3 * (max_h - min_h)

        if centroid[axis_idx] < threshold:
            return np.ones(len(coords), dtype=bool), {
                "case": 0,
                "reason": "marker_below_30pct",
                "marker_height": float(centroid[axis_idx]),
                "threshold": float(threshold),
            }

        dist = np.dot(coords - centroid, normal)
        keep = dist < 0
        info = {
            "case": 1,
            "centroid": centroid.tolist(),
            "normal": normal.tolist(),
        }

    elif n == 2:
        cid_low, centroid_low, normal_low, npts_low, clr_low = planes[0]
        cid_high, centroid_high, normal_high, npts_high, clr_high = planes[1]

        dist_low = np.dot(coords - centroid_low, normal_low)
        dist_high = np.dot(coords - centroid_high, normal_high)

        keep = (dist_low > 0) & (dist_high < 0)
        info = {
            "case": 2,
            "centroid_low": centroid_low.tolist(),
            "normal_low": normal_low.tolist(),
            "centroid_high": centroid_high.tolist(),
            "normal_high": normal_high.tolist(),
        }

    else:
        cid_low, centroid_low, normal_low, npts_low, clr_low = planes[0]
        cid_high, centroid_high, normal_high, npts_high, clr_high = planes[-1]

        dist_low = np.dot(coords - centroid_low, normal_low)
        dist_high = np.dot(coords - centroid_high, normal_high)

        keep = (dist_low > 0) & (dist_high < 0)
        info = {
            "case": 3,
            "n_markers": n,
            "centroid_low": centroid_low.tolist(),
            "normal_low": normal_low.tolist(),
            "centroid_high": centroid_high.tolist(),
            "normal_high": normal_high.tolist(),
        }

    info["kept"] = int(keep.sum())
    info["total"] = len(coords)
    return keep, info


def segment_point_cloud(pcd, height_axis="z", verbose=True):
    """Run full marker-based leg segmentation on an Open3D point cloud.

    Marker planes are fitted via SVD so that slanted/tilted markers
    are handled by cutting with signed distance to the tilted plane.

    Steps:
        1. HSV + ExG color detection (same universal marker logic).
        2. DBSCAN spatial clustering.
        3. SVD plane fitting per cluster (centroid + normal vector).
        4. Signed-distance cut using the tilted marker planes.

    Args:
        pcd: Open3D PointCloud with colors.
        height_axis: axis used for normal orientation and sorting ("x","y","z").
                     Normal is flipped to point positively along this axis,
                     so that "below plane" = negative signed distance.
                     Default "z" (vertical after pipeline leveling).
        verbose: print progress messages.

    Returns:
        segmented_pcd: Open3D PointCloud containing only the kept points.
        summary: dict with detection/clustering/cut statistics.
    """
    import open3d as o3d

    axis_idx = AXIS_MAP[height_axis.lower()]
    axis_name = height_axis.upper()

    coords = np.asarray(pcd.points, dtype=np.float64)
    if not pcd.has_colors():
        raise ValueError("Point cloud has no color information")

    colors_float = np.asarray(pcd.colors, dtype=np.float64)
    colors = np.clip(colors_float * 255.0, 0, 255).astype(np.uint8)

    n_total = len(coords)

    # --- Step 1: Color Detection ---
    if verbose:
        print(f"  Points: {n_total:,}  "
              f"BBox Z[{coords[:,2].min():.3f},{coords[:,2].max():.3f}]")
        print("  Detecting markers (HSV S>15 & H>60, ExG>10)...")

    marker_mask, stats = detect_markers(colors)

    if verbose:
        print(f"    HSV-only: {stats['n_hsv']:,}  ExG-only: {stats['exg_only']:,}  "
              f"both: {stats['both']:,}  total: {stats['n_markers']:,} "
              f"({stats['n_markers']/n_total*100:.1f}%)")

    summary = {"detection": stats, "n_clusters": 0, "cut": {}}

    # --- Too few markers: keep all ---
    if marker_mask.sum() < 10:
        if verbose:
            print("  Too few marker points (< 10) — keeping all points")
        segmented_pcd = pcd
        summary["cut"] = {"case": 0, "reason": "too_few_markers"}
        summary["n_kept"] = n_total
        return segmented_pcd, summary

    marker_coords = coords[marker_mask]
    marker_colors = colors[marker_mask]

    # --- Step 2: Spatial Clustering ---
    if verbose:
        print(f"  Clustering {len(marker_coords):,} marker points (DBSCAN eps=0.03)...")

    labels, n_clusters = cluster_markers(marker_coords)
    summary["n_clusters"] = n_clusters

    if verbose:
        n_noise = (labels == -1).sum()
        print(f"    {n_clusters} clusters, {n_noise} noise points")

    # --- All noise: keep all ---
    valid = labels != -1
    if not np.any(valid):
        if verbose:
            print("  All markers classified as noise — keeping all points")
        segmented_pcd = pcd
        summary["cut"] = {"case": 0, "reason": "all_noise"}
        summary["n_kept"] = n_total
        return segmented_pcd, summary

    # --- Step 3: SVD plane fitting per cluster ---
    planes = compute_cluster_planes(marker_coords, labels, marker_colors, axis_idx)

    if verbose:
        print(f"  Marker planes (sorted by {axis_name} centroid):")
        for cid, centroid, normal, npts, color_mean in planes:
            print(f"    cluster {cid}: centroid=({centroid[0]:.4f},{centroid[1]:.4f},"
                  f"{centroid[2]:.4f})  normal=({normal[0]:+.4f},{normal[1]:+.4f},"
                  f"{normal[2]:+.4f})  {npts} pts  "
                  f"RGB=({color_mean[0]:.0f},{color_mean[1]:.0f},{color_mean[2]:.0f})")

    # --- Step 4: Cut using signed distance to tilted planes ---
    if verbose:
        print("  Cutting surface via plane signed-distance...")

    keep_mask, cut_info = cut_surface_plane(coords, planes, axis_idx, axis_name)
    summary["cut"] = cut_info

    if verbose:
        kept = cut_info["kept"]
        print(f"    Kept {kept:,} / {n_total:,} ({kept/n_total*100:.1f}%) "
              f"({cut_info.get('case', '?')} marker(s))")

    summary["n_kept"] = cut_info["kept"]

    filtered_coords = coords[keep_mask]
    filtered_colors = colors[keep_mask]

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
    segmented_pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64) / 255.0)

    return segmented_pcd, summary
