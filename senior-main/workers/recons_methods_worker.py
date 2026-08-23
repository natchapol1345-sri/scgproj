#!/usr/bin/env python3
"""Subprocess worker: multiple reconstruction methods for determinism testing.

Usage:
    python recons_methods_worker.py input.ply output.ply --method METHOD [--seed SEED]
"""
import os
import sys
import time
import traceback
import numpy as np
import open3d as o3d


def _load_and_prep(input_path, seed=None):
    """Load pcd, downsample if huge, estimate+orient normals. Returns (pcd, num_points)."""
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    pcd = o3d.io.read_point_cloud(input_path)
    num_points = len(pcd.points)
    print(f"Points: {num_points:,}")

    if num_points > 90000:
        bbox_extent = pcd.get_axis_aligned_bounding_box().get_max_extent()
        voxel_size = bbox_extent / 350
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        attempts = 0
        while len(pcd.points) > 90000 and attempts < 10:
            voxel_size *= 1.15
            pcd = o3d.io.read_point_cloud(input_path)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            attempts += 1
        print(f"Downsampled: {num_points:,} -> {len(pcd.points):,}")

    _dbg_t = time.time()
    distances = pcd.compute_nearest_neighbor_distance()
    print(f"[DBG-prep] nn_distance: {time.time() - _dbg_t:.2f}s")
    avg_dist = np.mean(distances)
    radius = max(avg_dist * 4.0, 0.005)
    max_nn = min(max(30, int(len(pcd.points) * 0.01)), 100)
    _dbg_t = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    print(f"[DBG-prep] estimate_normals(max_nn={max_nn}): {time.time() - _dbg_t:.2f}s")

    oriented = False
    for k in [min(len(pcd.points) // 100, 15), 10]:
        if k < 5:
            break
        try:
            _dbg_t = time.time()
            pcd.orient_normals_consistent_tangent_plane(k)
            print(f"[DBG-prep] orient_tangent_plane(k={k}): {time.time() - _dbg_t:.2f}s")
            print(f"Normals: tangent_plane(k={k})")
            oriented = True
            break
        except Exception:
            continue
    if not oriented:
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.orient_normals_towards_camera_location(centroid)
        print("Normals: camera_location")

    return pcd


def _post_process(mesh, densities=None):
    """Cleanup + density filter + largest component. Returns clean mesh."""
    if densities is not None and len(densities) > 0:
        threshold = np.quantile(densities, 0.05)
        before = len(mesh.vertices)
        try:
            mesh.remove_vertices_by_mask(densities < threshold)
        except Exception:
            pass
        if len(mesh.vertices) != before:
            print(f"Density filter (5%): {before:,} -> {len(mesh.vertices):,}")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"Cleanup: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")

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

    return mesh


def method_poisson(pcd, output_path, seed=None):
    import tempfile
    import subprocess as sp

    mesh, densities = None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        pcd_path = os.path.join(tmpdir, "pcd_with_normals.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        for d in range(9, 6, -1):
            mesh_out = os.path.join(tmpdir, f"mesh_d{d}.ply")
            dens_out = os.path.join(tmpdir, f"dens_d{d}.npy")

            seed_lines = ""
            if seed is not None:
                seed_lines = f"""
import random
random.seed({seed})
np.random.seed({seed})
o3d.utility.random.seed({seed})
os.environ.setdefault('PYTHONHASHSEED', '0')
"""
            script = f"""
import sys, os, numpy as np, open3d as o3d
{seed_lines}
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
            _dbg_t = time.time()
            r = sp.run([sys.executable, "-c", script],
                        capture_output=True, text=True, timeout=300)
            print(f"[DBG-poisson] depth={d}: {time.time() - _dbg_t:.2f}s rc={r.returncode}")
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

    mesh = _post_process(mesh, densities)
    return mesh


def method_ball_pivot(pcd, output_path, seed=None):
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    pts = np.asarray(pcd.points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radii = [
        avg_dist * 1.5,
        avg_dist * 2.0,
        avg_dist * 3.0,
        avg_dist * 5.0,
    ]

    mesh = None
    for radius in radii:
        if radius < 0.0001:
            continue
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
            )
            if len(mesh.triangles) > 0:
                print(f"Ball Pivoting: radius={radius:.5f}, "
                      f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")
                break
        except Exception as e:
            print(f"Ball Pivoting radius={radius:.5f} failed: {e}")

    if mesh is None or len(mesh.triangles) == 0:
        print("ERROR: Ball Pivoting failed at all radii")
        sys.exit(1)

    mesh = _post_process(mesh)
    return mesh


def method_alpha_shape(pcd, output_path, seed=None):
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    pts = np.asarray(pcd.points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    alphas = []
    for mul in [0.5, 1.0, 2.0, 4.0, 8.0]:
        a = avg_dist * mul
        if a > 0.00001:
            alphas.append(a)

    mesh = None
    for alpha in alphas:
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            if len(mesh.triangles) > 0:
                print(f"Alpha Shape: alpha={alpha:.5f}, "
                      f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")
                break
        except Exception as e:
            print(f"Alpha Shape alpha={alpha:.5f} failed: {e}")

    if mesh is None or len(mesh.triangles) == 0:
        print("ERROR: Alpha Shape failed at all alphas")
        sys.exit(1)

    mesh = _post_process(mesh)
    return mesh


def method_poisson_omp1(pcd, output_path, seed=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return method_poisson(pcd, output_path, seed=seed)


METHODS = {
    "poisson": method_poisson,
    "ball_pivot": method_ball_pivot,
    "alpha_shape": method_alpha_shape,
    "poisson_omp1": method_poisson_omp1,
}


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    method_name = "poisson"
    seed = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--method" and i + 1 < len(sys.argv):
            method_name = sys.argv[i + 1]
            i += 2
        elif arg == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    if method_name not in METHODS:
        print(f"ERROR: unknown method '{method_name}'. Choices: {list(METHODS)}")
        sys.exit(1)

    print(f"Method: {method_name}")
    if seed is not None:
        print(f"Seed: {seed}")

    _dbg_t = time.time()
    pcd = _load_and_prep(input_path, seed=None)  # seed worker-level rng
    print(f"[DBG-worker] load_and_prep: {time.time() - _dbg_t:.2f}s")

    _dbg_t = time.time()
    mesh = METHODS[method_name](pcd, output_path, seed=seed)
    print(f"[DBG-worker] method {method_name}: {time.time() - _dbg_t:.2f}s")

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Watertight: {mesh.is_watertight()}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
