#!/usr/bin/env python3
"""Subprocess worker for watertight mesh repair.

Strategy:
    1. PyMeshFix.repair(joincomp=False, remove_smallest_components=False)
       — shape-preserving hole-fill + manifold repair (deterministic).
    2. If still not watertight, run Open3D fill_holes on the result.
    3. Color transfer from original mesh by KDTree nearest-neighbor.

Usage:
    python meshfix_worker.py input.ply output.ply
    python meshfix_worker.py input.ply output.ply color_source.ply
"""
import os
import sys
import traceback
import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "0")


def _pymeshfix_repair(vertices, faces):
    """Shape-preserving hole-fill via PyMeshFix. Existing geometry untouched.
    Returns (verts, faces)."""
    import pymeshfix
    mf = pymeshfix.MeshFix(np.ascontiguousarray(vertices, dtype=np.float64),
                           np.ascontiguousarray(faces, dtype=np.int32))
    # fill_holes() only — never drops faces; preserves shape and retention.
    mf.fill_holes()
    return np.asarray(mf.points, dtype=np.float64), np.asarray(mf.faces, dtype=np.int64)


def _o3d_fill_holes(vertices, faces, hole_size=1e9):
    """Fill remaining holes via Open3D tensor mesh. Deterministic."""
    import open3d as o3d
    tm = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.int32)),
    )
    filled = tm.fill_holes(hole_size=float(hole_size))
    v = filled.vertex.positions.numpy().astype(np.float64)
    f = filled.triangle.indices.numpy().astype(np.int64)
    return v, f


def _is_watertight(vertices, faces):
    import trimesh
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return bool(m.is_watertight)


def _merge_duplicates(vertices, faces, tol=1e-8):
    """Weld coincident vertices so re-loaders see one watertight mesh."""
    del tol
    import trimesh
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=8)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    color_source_path = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        import trimesh
        src_mesh = trimesh.load(input_path, process=False)

        in_verts = np.asarray(src_mesh.vertices, dtype=np.float64)
        in_faces = np.asarray(src_mesh.faces, dtype=np.int64)
        original_faces = len(in_faces)
        print(f"Input: {len(in_verts)} vertices, {original_faces} faces")
        print(f"Watertight (input): {_is_watertight(in_verts, in_faces)}")

        # Stage A: PyMeshFix fill_holes (preserves existing geometry).
        v, f = _pymeshfix_repair(in_verts, in_faces)
        wt = _is_watertight(v, f)
        print(f"After PyMeshFix: {len(v)} vertices, {len(f)} faces, watertight={wt}")

        # Stage B: residual hole-fill via Open3D for any leftover gaps.
        if not wt:
            v2, f2 = _o3d_fill_holes(v, f)
            wt2 = _is_watertight(v2, f2)
            print(f"After O3D fill: {len(v2)} vertices, {len(f2)} faces, watertight={wt2}")
            if wt2 or len(f2) > len(f):
                v, f, wt = v2, f2, wt2

        retention = len(f) / original_faces * 100 if original_faces > 0 else 100.0
        print(f"Output: {len(v)} vertices, {len(f)} faces")
        print(f"Retention: {retention:.1f}%")
        print(f"Watertight (output): {wt}")

        out_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

        # Color transfer
        if color_source_path:
            from scipy.spatial import cKDTree
            cs = trimesh.load(color_source_path, process=False)
            src_colors = cs.visual.vertex_colors if hasattr(cs.visual, "vertex_colors") else None
            if src_colors is not None and len(src_colors) > 0:
                tree = cKDTree(np.asarray(cs.vertices, dtype=np.float64))
                dists, idx = tree.query(out_mesh.vertices, k=1)
                out_mesh.visual.vertex_colors = src_colors[idx]
                print(f"Colors: transferred (mean_dist={dists.mean():.6f})")

        out_mesh.export(output_path)
        print("OK" if wt else "OK_NOT_WATERTIGHT")
        sys.exit(0 if wt else 2)

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
