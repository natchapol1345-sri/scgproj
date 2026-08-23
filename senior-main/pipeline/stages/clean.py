"""Stage 3 — Clean the point cloud, extract objects, and optionally segment legs.

Steps:
    1. RANSAC-based leveling so the ground plane is horizontal (Z up).
    2. Adaptive statistical-outlier removal.
    3. Voxel downsampling if very dense.
    4. Smart laser-cut floor removal (only if a horizontal bottom plane exists).
    5. DBSCAN clustering + box detection (cubeness) → box.ply + obj.ply.
    6. (Optional) Marker-based leg surface segmentation on obj.ply.
"""
import os

import numpy as np
import open3d as o3d

from pipeline.core.plane import (
    auto_ransac_threshold,
    detect_plane_ransac_deterministic,
    get_rotation_to_z_axis,
)
from pipeline.core.cluster import compute_eps, detect_top_k_objects
from pipeline.core.segmentation import (
    segment_point_cloud,
    detect_markers,
    cluster_markers,
)
from pipeline.core.fill import cap_point_cloud_bottom


def _segment_leg(object_paths, output_dir, height_axis="z", seed=42):
    """Apply marker-based leg segmentation to the obj.ply in object_paths.

    Args:
        object_paths: List of PLY paths from the clean stage (box.ply, obj.ply).
        output_dir: Base output directory for the pipeline.
        height_axis: Axis for height-based cut ("z" by default, vertical after leveling).
        seed: Random seed (unused, kept for API consistency).

    Returns:
        new_paths: List of PLY paths with obj.ply replaced by segmented_obj.ply.
                   If no obj.ply is found or segmentation produces too few points,
                   returns the original list unchanged.
    """
    print()
    print("-" * 40)
    print("Leg segmentation (Stage 3)")
    print("-" * 40)

    seg_dir = os.path.join(output_dir, "segmented")
    os.makedirs(seg_dir, exist_ok=True)

    new_paths = list(object_paths)
    obj_ply = None
    obj_idx = None

    for i, p in enumerate(object_paths):
        if os.path.basename(p).lower().startswith("obj"):
            obj_ply = p
            obj_idx = i
            break

    if obj_ply is None:
        print("  No obj.ply found in clean outputs — skipping segmentation")
        return new_paths

    print(f"  Input: {obj_ply}")

    try:
        pcd = o3d.io.read_point_cloud(obj_ply)
        if len(pcd.points) == 0:
            print("  Empty point cloud — skipping")
            return new_paths

        segmented_pcd, summary = segment_point_cloud(pcd, height_axis=height_axis, verbose=True)

    except Exception as e:
        print(f"  ERROR during segmentation: {e}")
        return new_paths

    cut_info = summary.get("cut", {})
    if cut_info.get("case") == 0:
        reason = cut_info.get("reason", "no_markers")
        print(f"  No cut applied ({reason}) — keeping original obj.ply for reconstruction")
        return new_paths

    n_kept = summary.get("n_kept", 0)
    if n_kept < 100:
        print(f"  Only {n_kept} points after cut (< 100) — keeping original obj.ply")
        return new_paths

    seg_path = os.path.join(seg_dir, "segmented_obj.ply")
    o3d.io.write_point_cloud(seg_path, segmented_pcd, write_ascii=False)

    seg_pts = np.asarray(segmented_pcd.points)
    print(f"  Saved: {seg_path} ({n_kept:,} pts) "
          f"X[{seg_pts[:,0].min():.3f},{seg_pts[:,0].max():.3f}] "
          f"Y[{seg_pts[:,1].min():.3f},{seg_pts[:,1].max():.3f}] "
          f"Z[{seg_pts[:,2].min():.3f},{seg_pts[:,2].max():.3f}]")

    new_paths[obj_idx] = seg_path
    return new_paths


def clean_and_extract_objects(
    input_path,
    output_folder="output_objects",
    k=2,
    visualize=False,
    skip_plane=False,
    merge_clusters=False,
    seed=42,
    ransac_factor=3,
    fill_enabled=True,
):
    os.makedirs(output_folder, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")

    initial_count = len(pcd.points)
    print(f"Loaded point cloud: {initial_count:,} points")

    # ---- STEP 1: LEVELING (FLIP) FIRST ----
    if not skip_plane:
        try:
            distance_threshold = auto_ransac_threshold(pcd, base_factor=ransac_factor)
            plane_model, _ = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=1000, seed=seed
            )

            normal = plane_model[:3]
            print("Leveling scan based on detected plane...")
            R = get_rotation_to_z_axis(normal)
            pcd.rotate(R, center=(0, 0, 0))

            # --- SMART UPSIDE-DOWN CHECK ---
            _, inliers_rot = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=500, seed=seed
            )
            pts_rot = np.asarray(pcd.points)
            plane_z = np.mean(pts_rot[inliers_rot, 2])

            mask = np.ones(len(pts_rot), dtype=bool)
            mask[inliers_rot] = False
            non_plane_z = pts_rot[mask, 2]

            if len(non_plane_z) > 0 and np.mean(non_plane_z) < plane_z:
                print("-> Detected upside-down orientation! Flipping 180 degrees...")
                flip_R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
                pcd.rotate(flip_R, center=(0, 0, 0))

        except Exception as e:
            print(f"Leveling failed: {e}")

    # ---- STEP 2: GENTLER OUTLIER REMOVAL ----
    if initial_count < 50000:
        std_ratio = 3.5
        nb_neighbors = 10
    elif initial_count < 200000:
        std_ratio = 3.0
        nb_neighbors = 15
    else:
        std_ratio = 2.5
        nb_neighbors = 20

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Outlier removal: {initial_count:,} → {len(pcd.points):,} (std_ratio={std_ratio})")

    # ---- STEP 3: ADAPTIVE DOWNSAMPLING ----
    if len(pcd.points) > 100000:
        voxel_size = 0.002
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled to {len(pcd.points):,} (voxel={voxel_size})")

    # ---- STEP 4: SMART LASER CUT PLANE REMOVAL ----
    if not skip_plane:
        try:
            distance_threshold = auto_ransac_threshold(pcd, base_factor=ransac_factor)
            distance_threshold = min(distance_threshold, 0.015)

            plane_model, inliers = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=1000, seed=seed
            )

            pts = np.asarray(pcd.points)
            plane_ratio = len(inliers) / len(pts)

            # --- AUTO-PLANE GEOMETRY CHECK ---
            normal = plane_model[:3]

            # 1. Horizontal? Dot product with Z-axis ≈ ±1
            is_horizontal = abs(np.dot(normal, [0, 0, 1])) > 0.85

            # 2. At the bottom of the scene?
            z_vals = pts[:, 2]
            z_min, z_max = np.min(z_vals), np.max(z_vals)
            plane_z = np.median(pts[inliers, 2])
            is_at_bottom = (plane_z - z_min) < 0.25 * (z_max - z_min)

            if is_horizontal and is_at_bottom and plane_ratio > 0.05:
                margin = 0.008
                keep_mask = pts[:, 2] > (plane_z + margin)
                pcd = pcd.select_by_index(np.where(keep_mask)[0])
                print(f"-> Smart Laser-cut floor at Z={plane_z:.4f} "
                      f"({plane_ratio*100:.0f}% of points removed)")
            else:
                reason = []
                if not is_horizontal:
                    reason.append("not horizontal")
                if not is_at_bottom:
                    reason.append("not at the bottom")
                if plane_ratio <= 0.05:
                    reason.append("too small (<5%)")
                print(f"Plane skipped: {', '.join(reason)} ({plane_ratio*100:.0f}% of points)")

        except Exception as e:
            print(f"Plane removal skipped (detection failed): {e}")

    # ---- STEP 5: DETECT OBJECTS ----
    if merge_clusters:
        print("Merge mode: filtering noise, merging all clusters")
        eps = compute_eps(pcd)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
        noise_mask = labels < 0
        if np.any(~noise_mask):
            pcd = pcd.select_by_index(np.where(~noise_mask)[0])
            print(f"  Removed {noise_mask.sum():,} noise points, kept {len(pcd.points):,}")
        box_cluster = pcd
        obj_cluster = None
    else:
        box_cluster, obj_cluster = detect_top_k_objects(pcd, k=k, visualize=visualize)

    # ---- STEP 6: FILL BOTTOM CAP ----
    if fill_enabled and not skip_plane and box_cluster is not None:
        box_cluster = cap_point_cloud_bottom(box_cluster, alpha=0.0)
    if fill_enabled and not skip_plane and obj_cluster is not None:
        obj_cluster = cap_point_cloud_bottom(obj_cluster, alpha=2.0)

    # ---- STEP 7: SAVE box.ply + obj.ply ----
    output_paths = []

    if box_cluster is not None:
        path = os.path.join(output_folder, "box.ply")
        o3d.io.write_point_cloud(path, box_cluster)
        output_paths.append(path)
        pts = np.asarray(box_cluster.points)
        print(f"Saved: {path} ({len(pts):,} pts) "
              f"X[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] "
              f"Y[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] "
              f"Z[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    if obj_cluster is not None:
        path = os.path.join(output_folder, "obj.ply")
        o3d.io.write_point_cloud(path, obj_cluster)
        output_paths.append(path)
        pts = np.asarray(obj_cluster.points)
        print(f"Saved: {path} ({len(pts):,} pts) "
              f"X[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] "
              f"Y[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] "
              f"Z[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    if not output_paths:
        print("WARNING: No clusters saved!")

    return output_paths


def _count_markers_on_ply(ply_path, axis_idx=2):
    """Quick marker-cluster count on raw PLY before cleaning.

    Returns number of valid marker clusters (>=150 pts), or 0 if
    detection fails or finds too few markers.
    """
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) == 0:
            return 0
        coords = np.asarray(pcd.points, dtype=np.float64)
        if not pcd.has_colors():
            return 0
        colors_float = np.asarray(pcd.colors, dtype=np.float64)
        colors = np.clip(colors_float * 255.0, 0, 255).astype(np.uint8)
        marker_mask, _stats = detect_markers(colors)
        if marker_mask.sum() < 10:
            return 0
        marker_coords = coords[marker_mask]
        labels, n_clusters = cluster_markers(marker_coords)
        valid = labels != -1
        if not np.any(valid):
            return 0
        valid_clusters = 0
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            if (labels == cid).sum() >= 150:
                valid_clusters += 1
        return valid_clusters
    except Exception:
        return 0


def clean_and_extract(ply_path, output_dir, num_objects=2, seed=42,
                      segment_leg=False, segment_height_axis="z",
                      fill_enabled=True):
    """Pipeline wrapper around clean_and_extract_objects.

    Optionally runs marker-based leg segmentation on obj.ply after cleaning.
    Returns the list of per-object PLY paths, or None on failure.
    """
    print()
    print("=" * 60)
    print("STAGE 3: Cleaning point cloud and extracting objects")
    print("=" * 60)

    clean_output_dir = os.path.join(output_dir, "clean_objects")
    os.makedirs(clean_output_dir, exist_ok=True)

    marker_count = _count_markers_on_ply(ply_path) if segment_leg else 0
    fill_skip_2_marker = segment_leg and marker_count == 2
    fill_this_run = fill_enabled and not fill_skip_2_marker
    if not fill_this_run:
        if not fill_enabled:
            reason = "disabled by --no-fill"
        else:
            reason = f"skipped (2-marker cut: {marker_count} markers detected)"
        print(f"  Bottom fill: {reason}")

    try:
        object_paths = clean_and_extract_objects(
            input_path=ply_path,
            output_folder=clean_output_dir,
            k=num_objects,
            visualize=False,
            seed=seed,
            fill_enabled=fill_this_run,
        )
        print(f"  Extracted {len(object_paths)} objects:")
        for p in object_paths:
            print(f"    → {p}")

        if object_paths and segment_leg:
            object_paths = _segment_leg(
                object_paths, output_dir,
                height_axis=segment_height_axis, seed=seed)

        return object_paths
    except Exception as e:
        print(f"  ERROR during cleaning: {e}")
        return None