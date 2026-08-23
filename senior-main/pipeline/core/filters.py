"""Confidence-based and spatial filters for raw point clouds."""
import numpy as np


def adaptive_confidence_filter(conf_flat, target_percentile):
    """Smart confidence filtering that adapts to the data distribution.

    Problem: some datasets (e.g. vase) have 85% of points at conf=1.0,
    so a 50th percentile filter keeps tons of garbage low-confidence points.
    Solution: use the HIGHER of percentile-based and absolute thresholds.
    """
    percentile_val = np.percentile(conf_flat, target_percentile)

    conf_min = conf_flat.min()
    frac_at_min = (conf_flat <= conf_min + 0.01).mean()

    if frac_at_min > 0.5:
        # Most points are at minimum confidence → raise the bar.
        # Keep only truly confident points (top of the distribution).
        if (conf_flat > conf_min + 0.01).any():
            abs_threshold = np.percentile(conf_flat[conf_flat > conf_min + 0.01], 25)
        else:
            abs_threshold = conf_min + 0.1
        threshold_val = max(percentile_val, abs_threshold)
        print(f"  Adaptive filter: {frac_at_min*100:.0f}% of points at min conf={conf_min:.2f}")
        print(f"    Percentile threshold: {percentile_val:.4f}")
        print(f"    Absolute threshold:   {abs_threshold:.4f}")
        print(f"    Using: {threshold_val:.4f}")
    else:
        threshold_val = percentile_val
        print(f"  Confidence threshold: {threshold_val:.4f} "
              f"({target_percentile:.0f}th percentile)")

    return threshold_val


def remove_spatial_outliers(points, colors, conf, k=20, std_ratio=2.5):
    """Remove spatial outliers using Open3D statistical outlier removal."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        inlier_idx = np.asarray(inlier_idx)
        removed = len(points) - len(inlier_idx)
        if removed > 0:
            print(f"  Spatial outlier removal: {len(points):,} → "
                  f"{len(inlier_idx):,} (removed {removed:,})")
        return points[inlier_idx], colors[inlier_idx], conf[inlier_idx]
    except Exception:
        return points, colors, conf
