import numpy as np
import open3d as o3d
import alphashape
from shapely.geometry import Point


def cap_point_cloud_bottom(pcd, alpha=2.0, z_offset=0.0, slice_thickness=0.01):
    """
    Creates a synthetic flat bottom cap by matching the exact point density
    of the original scan, preventing the reconstruction filter from deleting it.
    """
    print("  -> Extracting coordinates...")
    points = np.asarray(pcd.points)

    if len(pcd.normals) == 0:
        print("  -> Normals not found. Estimating outward normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    print("  -> Analyzing scan density...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)

    point_spacing = np.clip(avg_spacing, 0.0005, 0.005)
    print(f"  -> Calculated exact point spacing from scan: {point_spacing:.5f}")

    z_min = np.min(points[:, 2])
    target_z = z_min + z_offset

    print(f"  -> Extracting bottom cross-section (thickness: {slice_thickness} units)...")
    bottom_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_min + slice_thickness)
    bottom_points = points[bottom_mask]

    if len(bottom_points) < 10:
        print("  -> Warning: Too few points in slice. Using full object footprint instead.")
        points_2d = points[:, :2]
        bottom_mask = np.ones(len(points), dtype=bool)
    else:
        points_2d = bottom_points[:, :2]

    print("  -> Calculating boundary hull...")
    try:
        hull_polygon = alphashape.alphashape(points_2d, alpha)

        if hull_polygon.geom_type == 'MultiPolygon':
            hull_polygon = max(hull_polygon.geoms, key=lambda a: a.area)
    except Exception:
        print(f"  -> Warning: Hull generation failed with alpha={alpha}. Defaulting to convex hull.")
        hull_polygon = alphashape.alphashape(points_2d, 0.0)

    print("  -> Generating uniform point grid inside the outline...")
    min_x, min_y, max_x, max_y = hull_polygon.bounds
    x_grid = np.arange(min_x, max_x, point_spacing)
    y_grid = np.arange(min_y, max_y, point_spacing)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    synthetic_points_2d = [pt for pt in grid_points_2d if hull_polygon.contains(Point(pt))]

    synthetic_points_2d = np.array(synthetic_points_2d)
    num_synthetic_points = len(synthetic_points_2d)
    print(f"  -> Generated {num_synthetic_points} synthetic points for the cap.")

    z_array = np.full((num_synthetic_points, 1), target_z)
    synthetic_points_3d = np.hstack((synthetic_points_2d, z_array))

    synthetic_normals = np.tile([0.0, 0.0, -1.0], (num_synthetic_points, 1))

    cap_pcd = o3d.geometry.PointCloud()
    cap_pcd.points = o3d.utility.Vector3dVector(synthetic_points_3d)
    cap_pcd.normals = o3d.utility.Vector3dVector(synthetic_normals)

    if pcd.has_colors():
        print("  -> Preserving colors and painting the cap...")
        colors = np.asarray(pcd.colors)
        bottom_colors = colors[bottom_mask]
        if len(bottom_colors) > 0:
            avg_color = np.mean(bottom_colors, axis=0)
        else:
            avg_color = [0.5, 0.5, 0.5]
        synthetic_colors = np.tile(avg_color, (num_synthetic_points, 1))
        cap_pcd.colors = o3d.utility.Vector3dVector(synthetic_colors)

    merged_pcd = pcd + cap_pcd
    return merged_pcd
