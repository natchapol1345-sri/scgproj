#!/usr/bin/env python3
"""Subprocess worker: Poisson reconstruction only.

Produces a non-watertight triangle mesh from a point cloud.
Runs in a clean process to avoid memory conflicts with PyTorch/CUDA.

Usage:
    python recons_worker.py input_pcd.ply output.ply
"""
import sys
import os
import traceback
import numpy as np
import open3d as o3d


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    seed = None
    for i, arg in enumerate(sys.argv):
        if arg == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            break
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    pcd = o3d.io.read_point_cloud(input_path)
    num_points = len(pcd.points)
    print(f"Points: {num_points:,}")

    # Downsample to <165k for Poisson stability (depth 9 limit)
    max_points = 165000
    if num_points > max_points:
        bbox_extent = pcd.get_axis_aligned_bounding_box().get_max_extent()
        voxel_size = bbox_extent / 350
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        attempts = 0
        while len(pcd.points) > max_points and attempts < 10:
            voxel_size *= 1.15
            pcd = o3d.io.read_point_cloud(input_path)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            attempts += 1
        print(f"Downsampled: {num_points:,} -> {len(pcd.points):,}")

    # Normals
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = max(avg_dist * 4.0, 0.005)
    max_nn = min(max(30, int(len(pcd.points) * 0.01)), 100)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # Orient normals — tangent plane with progressive fallback
    oriented = False
    for k in [min(len(pcd.points) // 100, 50), 20, 10]:
        if k < 5:
            break
        try:
            pcd.orient_normals_consistent_tangent_plane(k)
            print(f"Normals: tangent_plane(k={k})")
            oriented = True
            break
        except Exception:
            continue
    if not oriented:
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.orient_normals_towards_camera_location(centroid)
        print("Normals: camera_location")

    # Poisson — try multiple depths in a sub-subprocess so crashes don't kill
    # the worker. Write the prepared pcd to a temp file.
    import tempfile
    mesh, densities = None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        pcd_path = os.path.join(tmpdir, "pcd_with_normals.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        for d in range(9, 6, -1):
            mesh_out = os.path.join(tmpdir, f"mesh_d{d}.ply")
            dens_out = os.path.join(tmpdir, f"dens_d{d}.npy")
            poisson_script = f"""
import sys, numpy as np, open3d as o3d
pcd = o3d.io.read_point_cloud({pcd_path!r})
try:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth={d})
    densities = np.asarray(densities)
    if len(mesh.triangles) == 0:
        sys.exit(2)
    o3d.io.write_triangle_mesh({mesh_out!r}, mesh)
    np.save({dens_out!r}, densities)
    sys.exit(0)
except Exception as e:
    print(f'failed: {{e}}', file=sys.stderr)
    sys.exit(1)
"""
            import subprocess
            r = subprocess.run([sys.executable, "-c", poisson_script],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.exists(mesh_out):
                mesh = o3d.io.read_triangle_mesh(mesh_out)
                densities = np.load(dens_out)
                print(f"Poisson: depth={d}, {len(mesh.vertices):,} verts, "
                      f"{len(mesh.triangles):,} faces")
                break
            err = r.stderr.strip()[:100] if r.stderr.strip() else f"exit {r.returncode}"
            print(f"Poisson depth={d} failed: {err}")

    if mesh is None or len(mesh.triangles) == 0:
        print("ERROR: Poisson failed at all depths")
        sys.exit(1)

    # Density filter (5% — removes far-field artifacts, good for both watertight and view)
    if densities is not None and len(densities) > 0:
        threshold = np.quantile(densities, 0.05)
        before = len(mesh.vertices)
        mesh.remove_vertices_by_mask(densities < threshold)
        print(f"Density filter (5%): {before:,} -> {len(mesh.vertices):,}")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(f"Cleanup: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")

    # Keep only the largest connected component
    # Gives MeshFix a single surface with clean boundaries to close
    # (like object 1 naturally has, but object 0 needs this explicit step)
    try:
        cluster_ids, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        cluster_ids = np.asarray(cluster_ids)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) > 1:
            largest = int(np.argmax(cluster_n_tris))
            remove_mask = cluster_ids != largest
            before = len(mesh.triangles)
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            print(f"Largest component: kept {len(mesh.triangles):,}/{before:,} faces "
                  f"({len(mesh.triangles)/before*100:.1f}%), "
                  f"{len(cluster_n_tris)} components total")
    except Exception as e:
        print(f"Component filtering skipped: {e}")

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Watertight: {mesh.is_watertight()}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
