"""Stage 2 — Filter predictions and export point cloud as PLY."""
import os
import numpy as np
import trimesh

from pipeline.core.filters import adaptive_confidence_filter, remove_spatial_outliers


def export_ply(predictions, output_dir, args):
    """Filter and export point cloud as PLY with adaptive confidence and outlier removal."""
    print()
    print("=" * 60)
    print("STAGE 2: Exporting PLY point cloud")
    print("=" * 60)

    if args.prediction_mode == "pointmap":
        world_points = predictions["world_points"]         # (S, H, W, 3)
        conf = predictions["world_points_conf"]            # (S, H, W)
        print("  Mode: pointmap regression")
    else:
        world_points = predictions["world_points_from_depth"]
        conf = predictions["depth_conf"]
        print("  Mode: depth-based unprojection")

    # Images: (S, 3, H, W) → (S, H, W, 3)
    imgs_np = predictions["images"]
    colors = imgs_np.transpose(0, 2, 3, 1)

    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    threshold_val = adaptive_confidence_filter(conf_flat, args.conf_thres)
    mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

    if args.mask_black_bg:
        brightness = colors_flat.astype(np.float32).mean(axis=1)
        mask &= brightness > 15.0
    if args.mask_white_bg:
        brightness = colors_flat.astype(np.float32).mean(axis=1)
        mask &= brightness < 240.0

    points_out = points_flat[mask]
    colors_out = colors_flat[mask]
    conf_out = conf_flat[mask]

    print(f"  After filtering: {points_flat.shape[0]:,} → {points_out.shape[0]:,}")

    points_out, colors_out, _conf_out = remove_spatial_outliers(points_out, colors_out, conf_out)

    print(f"  Final point count: {points_out.shape[0]:,}")

    ply_path = os.path.join(output_dir, "points.ply")
    pc = trimesh.PointCloud(points_out, colors=colors_out)
    pc.export(ply_path)
    print(f"  Exported: {ply_path}")

    return ply_path
