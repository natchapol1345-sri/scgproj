#!/usr/bin/env python3
"""Standalone volume computation — run directly, no pipeline needed.

Usage:
    python volume.py                          # use defaults below
    python volume.py --obj output/mesh/segmented_obj.stl
    python volume.py --obj path/to/leg.stl --box path/to/box.stl --ref-size 14.0
    python volume.py --voxel-res 200          # voxel grid resolution (default: 150)
    python volume.py --auto-res               # auto-tune voxel resolution until converged
    python volume.py --list-meshes            # show all meshes in output/mesh/

Volume method priority (per mesh):
    1. watertight        — exact signed volume, best accuracy
    2. warp_floodfill    — GPU surface mark + CPU flood-fill, ~40x faster than trimesh
    3. trimesh_voxel     — CPU fallback if warp unavailable
    4. convex_hull       — last resort, overestimates concave shapes
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import trimesh
from scipy import ndimage

# warp optional — GPU voxelization
try:
    import warp as wp
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config — edit these defaults or override via CLI flags
# ---------------------------------------------------------------------------

DEFAULT_OBJ_MESH  = "output/mesh/segmented_obj.stl"
DEFAULT_BOX_MESH  = "output/mesh/box.stl"
DEFAULT_REF_SIZE  = 14.0   # real-world side length of reference cube, in cm
DEFAULT_VOXEL_RES = 150    # voxel grid steps along longest axis

# ---------------------------------------------------------------------------
# Warp GPU kernel (defined at module level — warp requires file scope)
# ---------------------------------------------------------------------------

if _WARP_AVAILABLE:
    wp.init()

    @wp.kernel
    def _warp_mark_surface(
        mesh_id:   wp.uint64,
        surface:   wp.array3d(dtype=wp.int32),
        b_min:     wp.vec3,
        pitch:     float,
        threshold: float,
    ):
        i, j, k = wp.tid()
        cx = b_min[0] + (float(i) + 0.5) * pitch
        cy = b_min[1] + (float(j) + 0.5) * pitch
        cz = b_min[2] + (float(k) + 0.5) * pitch
        q = wp.mesh_query_point_sign_normal(mesh_id, wp.vec3(cx, cy, cz), threshold)
        if q.result:
            surface[i, j, k] = 1


# ---------------------------------------------------------------------------
# Volume methods
# ---------------------------------------------------------------------------

def _volume_voxel_warp(mesh: trimesh.Trimesh, resolution: int,
                        device: str = "cuda:0") -> float:
    """GPU surface-mark (warp BVH) + CPU flood-fill (scipy). ~40x faster than trimesh."""
    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    wp_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=wp.int32, device=device),
    )

    b_min  = verts.min(axis=0)
    b_max  = verts.max(axis=0)
    pitch  = float((b_max - b_min).max()) / resolution
    nx = max(1, int(np.ceil((b_max[0] - b_min[0]) / pitch)))
    ny = max(1, int(np.ceil((b_max[1] - b_min[1]) / pitch)))
    nz = max(1, int(np.ceil((b_max[2] - b_min[2]) / pitch)))

    threshold   = pitch * (3.0 ** 0.5) / 2.0   # voxel half-diagonal
    surface_gpu = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    b_min_wp    = wp.vec3(float(b_min[0]), float(b_min[1]), float(b_min[2]))

    wp.launch(_warp_mark_surface, dim=(nx, ny, nz),
              inputs=[wp_mesh.id, surface_gpu, b_min_wp, pitch, threshold],
              device=device)
    wp.synchronize()

    surface_np = surface_gpu.numpy().astype(bool)

    struct         = ndimage.generate_binary_structure(3, 1)
    padded         = np.pad(surface_np, 1, constant_values=False)
    labeled, _     = ndimage.label(~padded, structure=struct)
    exterior_label = int(labeled[0, 0, 0])
    interior       = (~padded) & (labeled != exterior_label)
    interior       = interior[1:-1, 1:-1, 1:-1]

    return float((interior.sum() + surface_np.sum()) * pitch**3)


def _volume_voxel(mesh: trimesh.Trimesh, resolution: int) -> float:
    """Ray-cast voxelization volume. Works on concave and non-manifold meshes."""
    pitch = float(mesh.extents.max()) / resolution
    vox = mesh.voxelized(pitch=pitch)
    filled = vox.fill()          # flood-fill interior
    print(type(filled))
    return float(filled.volume)  # n_voxels * pitch³


def _export_voxel_meshes(mesh: trimesh.Trimesh, resolution: int,
                         export_dir: str, label: str) -> None:
    """Export filled voxel grid as smooth (marching_cubes) PLY."""
    os.makedirs(export_dir, exist_ok=True)
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").strip("_")
    pitch = float(mesh.extents.max()) / resolution
    filled = mesh.voxelized(pitch=pitch).fill()

    mc_path = os.path.join(export_dir, f"{safe}_smooth.ply")
    filled.marching_cubes.export(mc_path)
    print(f"  voxel export → {mc_path}  (smooth)")


def _volume_convex_hull(mesh: trimesh.Trimesh) -> float:
    return float(abs(mesh.convex_hull.volume))


def auto_tune_voxel_res(
    mesh: trimesh.Trimesh,
    start: int = 50,
    stop: int = 500,
    step: int = 50,
    tol: float = 0.01,
    use_warp: bool = False,
) -> tuple[int, float]:
    """Increase resolution until volume converges. Returns (best_res, volume)."""
    backend = "warp+floodfill" if use_warp else "trimesh"
    print(f"  {'res':>6}  {'pitch':>10}  {'volume':>14}  {'Δ%':>8}  [{backend}]")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}")

    prev_vol = None
    best_res = start

    for res in range(start, stop + 1, step):
        pitch = float(mesh.extents.max()) / res
        try:
            if use_warp:
                vol = _volume_voxel_warp(mesh, res)
            else:
                vol = float(mesh.voxelized(pitch=pitch).fill().volume)
        except Exception as e:
            print(f"  {res:>6}  {pitch:>10.5f}  {'ERROR':>14}  {'—':>8}  (skipped: {e})")
            continue

        if prev_vol is not None and prev_vol > 0:
            delta_pct = abs(vol - prev_vol) / prev_vol * 100
            print(f"  {res:>6}  {pitch:>10.5f}  {vol:>14.6f}  {delta_pct:>7.2f}%")
            if delta_pct < tol * 100:
                best_res = res
                print(f"\n  converged at res={res}  (Δ < {tol*100:.1f}%)")
                return best_res, vol
        else:
            print(f"  {res:>6}  {pitch:>10.5f}  {vol:>14.6f}  {'—':>8}")

        prev_vol = vol
        best_res = res

    print(f"\n  [warn] did not converge by res={stop}. using res={best_res}.")
    return best_res, prev_vol


def _measure_volume(mesh: trimesh.Trimesh, voxel_res: int,
                    auto_res: bool = False,
                    export_voxel_dir: str | None = None,
                    label: str = "mesh") -> tuple[float, str]:
    """Return (volume_mesh_units3, method_label) using best available method."""
    if mesh.is_watertight:
        return float(abs(mesh.volume)), "watertight"

    # GPU path (warp BVH + scipy flood-fill)
    if _WARP_AVAILABLE:
        try:
            if auto_res:
                best_res, vol = auto_tune_voxel_res(mesh, use_warp=True)
                method = f"warp+floodfill (auto res={best_res})"
            else:
                best_res = voxel_res
                vol = _volume_voxel_warp(mesh, voxel_res)
                method = f"warp+floodfill (res={voxel_res})"
            if vol > 0:
                if export_voxel_dir:
                    _export_voxel_meshes(mesh, best_res, export_voxel_dir, label)
                return vol, method
        except Exception as e:
            print(f"  [warn] warp failed: {e} — falling back to trimesh")

    try:
        if auto_res:
            best_res, vol = auto_tune_voxel_res(mesh, use_warp=False)
            method = f"voxel (auto res={best_res})"
        else:
            best_res = voxel_res
            vol = _volume_voxel(mesh, voxel_res)
            method = f"voxel (res={voxel_res})"
        if vol > 0:
            if export_voxel_dir:
                _export_voxel_meshes(mesh, best_res, export_voxel_dir, label)
            return vol, method
    except Exception as e:
        print(f"  [warn] voxel failed: {e}")

    try:
        return _volume_convex_hull(mesh), "convex_hull (fallback — overestimates concave)"
    except Exception as e:
        sys.exit(f"[ERROR] all volume methods failed: {e}")


# ---------------------------------------------------------------------------
# Mesh loader
# ---------------------------------------------------------------------------

def load_mesh_info(path: str, label: str, voxel_res: int,
                   auto_res: bool = False,
                   export_voxel_dir: str | None = None) -> dict:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] file not found: {path}")

    mesh = trimesh.load(path, force="mesh", process=False)
    if len(mesh.vertices) == 0:
        sys.exit(f"[ERROR] mesh is empty: {path}")

    # STL stores raw per-triangle vertices (no shared indices). Merge to restore
    # connectivity so watertight check works correctly.
    mesh.merge_vertices()

    extents = mesh.bounds[1] - mesh.bounds[0]

    if auto_res:
        print(f"\n  auto-tuning res for {label}:")
    volume, method = _measure_volume(mesh, voxel_res, auto_res=auto_res,
                                     export_voxel_dir=export_voxel_dir, label=label)

    print(f"  {label}: {method}")

    return {
        "label":    label,
        "file":     os.path.basename(path),
        "method":   method,
        "volume":   volume,
        "ext_x":    float(extents[0]),
        "ext_y":    float(extents[1]),
        "ext_z":    float(extents[2]),
        "bbox_vol": float(np.prod(extents)),
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_volumes(obj_path: str, box_path: str,
                    ref_size_cm: float, voxel_res: int,
                    auto_res: bool = False,
                    export_voxel_dir: str | None = None) -> pd.DataFrame:
    """Compute scaled real-world volumes. Returns result DataFrame."""
    print("\n── Loading meshes ──")
    obj = load_mesh_info(obj_path, label="object",          voxel_res=voxel_res,
                         auto_res=auto_res, export_voxel_dir=export_voxel_dir)
    box = load_mesh_info(box_path, label="reference_box",   voxel_res=voxel_res,
                         auto_res=auto_res, export_voxel_dir=export_voxel_dir)

    df = pd.DataFrame([obj, box])

    print("\n── Raw mesh measurements (mesh units) ──")
    print(df[["label", "file", "method", "volume", "ext_x", "ext_y", "ext_z", "bbox_vol"]]
          .to_string(index=False))

    # Scale: k converts mesh-unit³ → cm³
    real_ref_vol = ref_size_cm ** 3          # e.g. 14³ = 2744 cm³
    k            = real_ref_vol / box["volume"]
    linear_scale = k ** (1.0 / 3.0)         # cm per mesh-unit (for extent display)

    print(f"\n── Scale factor ──")
    print(f"  ref mesh vol  = {box['volume']:.6f} mesh-units³")
    print(f"  real ref vol  = {ref_size_cm}³ = {real_ref_vol:.2f} cm³")
    print(f"  k             = {real_ref_vol:.2f} / {box['volume']:.6f} = {k:.6f} cm³/unit³")
    print(f"  linear_scale  = k^(1/3) = {linear_scale:.6f} cm/unit")

    df["real_vol_cm3"] = df["volume"] * k
    df["real_vol_L"]   = df["real_vol_cm3"] / 1000.0
    df["size_x_cm"]    = df["ext_x"] * linear_scale
    df["size_y_cm"]    = df["ext_y"] * linear_scale
    df["size_z_cm"]    = df["ext_z"] * linear_scale

    result_cols = ["label", "file", "size_x_cm", "size_y_cm", "size_z_cm",
                   "real_vol_cm3", "real_vol_L", "method"]
    print(f"\n── Real-world dimensions & volumes ──")
    print(df[result_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj",       default=DEFAULT_OBJ_MESH,
                   help=f"object mesh path  (default: {DEFAULT_OBJ_MESH})")
    p.add_argument("--box",       default=DEFAULT_BOX_MESH,
                   help=f"reference box mesh path  (default: {DEFAULT_BOX_MESH})")
    p.add_argument("--ref-size",  type=float, default=DEFAULT_REF_SIZE,
                   help=f"real side-length of reference cube in cm  (default: {DEFAULT_REF_SIZE})")
    p.add_argument("--voxel-res", type=int, default=DEFAULT_VOXEL_RES,
                   help=f"voxel grid resolution along longest axis  (default: {DEFAULT_VOXEL_RES})")
    p.add_argument("--auto-res", action="store_true",
                   help="auto-tune voxel resolution until volume converges (overrides --voxel-res)")
    p.add_argument("--export-voxel", nargs="?", const="output/mesh/voxel", default=None,
                   metavar="DIR",
                   help="export filled voxel mesh (blocky + smooth) to DIR  (default: output/mesh/voxel)")
    p.add_argument("--list-meshes", action="store_true",
                   help="list all meshes in output/mesh/ and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_meshes:
        mesh_dir = "output/mesh"
        if not os.path.isdir(mesh_dir):
            sys.exit(f"[ERROR] {mesh_dir} not found")
        files = sorted(f for f in os.listdir(mesh_dir)
                       if f.endswith((".ply", ".stl")))
        print(f"\nMeshes in {mesh_dir}/:")
        for f in files:
            size = os.path.getsize(os.path.join(mesh_dir, f))
            print(f"  {f:<40}  {size/1024:>8.1f} KB")
        return

    print(f"  obj         : {args.obj}")
    print(f"  box         : {args.box}")
    print(f"  ref-size    : {args.ref_size} cm")
    print(f"  voxel-res   : {'auto' if args.auto_res else args.voxel_res}")
    if args.export_voxel:
        print(f"  export-voxel: {args.export_voxel}")

    compute_volumes(args.obj, args.box, args.ref_size, args.voxel_res,
                    auto_res=args.auto_res, export_voxel_dir=args.export_voxel)


if __name__ == "__main__":
    main()
