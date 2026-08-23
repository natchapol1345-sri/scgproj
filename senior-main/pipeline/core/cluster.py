"""DBSCAN clustering + box/obj detection."""
import numpy as np


def compute_eps(pcd, factor=4.0):
    """Estimate DBSCAN epsilon from average nearest-neighbor distance."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"Avg NN distance: {avg_dist:.6f}")
    return avg_dist * factor


def get_cluster_info(cluster):
    """Return extent, density, and maximum dimension of one cluster."""
    bbox = cluster.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    volume = np.prod(extent) if np.all(extent > 0) else 1e-6
    density = len(cluster.points) / volume
    max_dim = max(extent)
    return extent, density, max_dim


def aruco_cubeness(cluster):
    """Score 0..1: how cube-like the cluster bbox is (ArUco = 14cm cube → ~1.0)."""
    extent = cluster.get_axis_aligned_bounding_box().get_extent()
    mx = float(np.max(extent))
    if mx <= 1e-9:
        return 0.0
    return float(np.min(extent) / mx)


def aruco_bw_ratio(cluster):
    """Score 0..1: fraction of points near-black or near-white (ArUco signature)."""
    if not cluster.has_colors():
        return 0.0
    colors = np.asarray(cluster.colors)
    if len(colors) == 0:
        return 0.0
    brightness = colors.mean(axis=1)
    frac_black = float((brightness < 0.20).mean())
    frac_white = float((brightness > 0.80).mean())
    return frac_black + frac_white


def detect_top_k_objects(pcd, k=2, visualize=False):
    """Detect top-k clusters, then identify which is the box and which is the other object.

    Returns: (box_cluster, obj_cluster)
        box_cluster  — the dominant non-marker cluster (the target box)
        obj_cluster  — the other cluster (ArUco marker or secondary object)
        Returns (pcd, None) if only 1 cluster found.
    """
    del visualize  # currently unused; reserved for future debug viewer
    print("Running DBSCAN...")
    eps = compute_eps(pcd)
    print(f"Adaptive eps: {eps:.5f}")

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
    valid = labels >= 0

    if not np.any(valid):
        print("No clusters found → using whole cloud as single object")
        return (pcd, None)

    unique_ids = np.unique(labels[valid])
    print(f"Total clusters found: {len(unique_ids)}")

    min_points = 0.01 * len(pcd.points)

    clusters = []
    for cid in unique_ids:
        idx = np.where(labels == cid)[0]
        cluster = pcd.select_by_index(idx)
        extent, density, max_dim = get_cluster_info(cluster)
        npts = len(cluster.points)

        if npts < min_points:
            continue

        norm_pts = npts / 1000.0
        norm_density = min(density, 5_000_000) / 1_000_000.0
        score = norm_pts * 0.5 + norm_density * 0.3 - max_dim * 0.2
        clusters.append((cluster, score, npts, extent, density, max_dim))

    if len(clusters) == 0:
        print("No valid clusters → using whole cloud")
        return (pcd, None)

    clusters.sort(key=lambda x: x[1], reverse=True)
    top = clusters[:min(k, len(clusters))]

    if len(top) < 2:
        print("Only 1 cluster → returning as box, no obj")
        return (top[0][0], None)

    # Box detection: which cluster has highest cubeness (most box-shaped)
    # ArUco check: if one cluster is a strong marker, the other is the box
    print("\n  Box detection (cubeness = min_extent / max_extent):")
    info = []
    for idx, (cluster, score, npts, extent, density, max_dim) in enumerate(top):
        cube = aruco_cubeness(cluster)
        bw = aruco_bw_ratio(cluster)
        aruco = cube * 0.3 + bw * 0.7
        ext_str = f"({extent[0]:.4f},{extent[1]:.4f},{extent[2]:.4f})"
        print(f"    cluster #{idx+1}: {npts:,} pts, cubeness={cube:.4f}, aruco={aruco:.3f} {ext_str}")
        info.append((idx, cluster, score, npts, cube, bw, aruco))

    aruco_scores = [x[6] for x in info]
    has_marker = any(a > 0.7 for a in aruco_scores)

    if has_marker:
        marker_idx = int(np.argmax(aruco_scores))
        box_idx = 1 - marker_idx
        print(f"    ArUco marker detected at cluster #{marker_idx+1} → other is box")
    else:
        cube_scores = [x[4] for x in info]
        box_idx = int(np.argmax(cube_scores))
        print(f"    No strong ArUco → highest cubeness = box")

    obj_idx = 1 - box_idx

    for i, (idx, cluster, score, npts, cube, bw, aruco) in enumerate(info):
        if i == box_idx:
            print(f"    → cluster #{idx+1}: {npts:,} pts — BOX (box.ply)")
        else:
            print(f"    → cluster #{idx+1}: {npts:,} pts — OBJ (obj.ply)")

    return (top[box_idx][0], top[obj_idx][0])
