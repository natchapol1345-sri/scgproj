#!/usr/bin/env python3
"""Standalone real-world volume computation from a mesh + reference cube PLY.

Not part of the main pipeline (see pipeline.stages.volume for that). Useful for
ad-hoc measurements when you already have a reconstructed mesh and a reference
point cloud.
"""
import numpy as np
import open3d as o3d


def compute_volume_from_mesh(
    mesh_path: str,
    reference_ply_path: str,
    real_cube_size_m: float = 0.14,
):
    """Compute scaled real-world volume from STL/PLY mesh.

    Args:
        mesh_path: Path to reconstructed mesh (STL/PLY).
        reference_ply_path: Path to cleaned point cloud used as size reference.
        real_cube_size_m: Real cube side length in meters.

    Returns:
        dict: volume metrics and debug information.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or unreadable: {mesh_path}")
    print("Loaded mesh")

    if not mesh.is_watertight():
        print("Not watertight; using convex hull")
        pcd_temp = mesh.sample_points_uniformly(number_of_points=50000)
        mesh, _ = pcd_temp.compute_convex_hull()

    raw_volume = mesh.get_volume()
    print(f"Raw volume (mesh units cubed): {raw_volume}")

    cube_pcd = o3d.io.read_point_cloud(reference_ply_path)
    if len(cube_pcd.points) == 0:
        raise ValueError(f"Reference point cloud is empty or unreadable: {reference_ply_path}")
    print("Loaded cube reference")

    bbox = cube_pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    estimated_side = np.median(extent)
    if estimated_side <= 0:
        raise ValueError("Estimated reference side length is non-positive.")

    print("Cube size in mesh units:", extent)
    print(f"Estimated cube side (mesh units): {estimated_side}")

    scale_factor = real_cube_size_m / estimated_side
    print(f"Scale factor: {scale_factor}")

    mesh.scale(scale_factor, center=mesh.get_center())
    print("Applied scale to mesh")

    real_volume_m3 = mesh.get_volume()
    real_volume_cm3 = real_volume_m3 * 1e6

    print(f"Real volume: {real_volume_m3:.6f} m^3")
    print(f"Real volume: {real_volume_cm3:.2f} cm^3")
    print("Watertight:", mesh.is_watertight())

    return {
        "raw_volume_units3": float(raw_volume),
        "estimated_side_units": float(estimated_side),
        "scale_factor": float(scale_factor),
        "real_volume_m3": float(real_volume_m3),
        "real_volume_cm3": float(real_volume_cm3),
        "is_watertight": bool(mesh.is_watertight()),
    }


if __name__ == "__main__":
    default_mesh_path = "output_ply/mesh_input.ply"
    default_reference = "clean_input_ply/clean_input.ply"
    result = compute_volume_from_mesh(default_mesh_path, default_reference,
                                      real_cube_size_m=0.14)
    print("Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
