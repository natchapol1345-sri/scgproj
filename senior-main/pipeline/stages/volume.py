"""Stage 7 — Compute real-world volumes using the ArUco reference (box).

Scale derivation:
    k            = real_ref_vol_cm3 / ref_mesh_vol          # cm³ per mesh-unit³
    linear_scale = k ** (1/3)                               # cm per mesh-unit
    real_vol     = mesh_vol * k
    ext_cm       = mesh_extents * linear_scale

Volume method priority (per mesh):
    1. watertight        — exact signed volume
    2. warp+floodfill    — GPU BVH surface mark + CPU flood-fill (~40x faster than trimesh)
    3. trimesh voxel     — CPU fallback
    4. convex_hull       — last resort, overestimates concave shapes

The reference is a 14 × 14 × 14 cm ArUco cube identified by 'box' in its filename.
"""
import os
import numpy as np
import pandas as pd
import trimesh
from scipy import ndimage

from pipeline.config import REFERENCE_REAL_SIZE_CM

DEFAULT_VOXEL_RES = 150

# warp optional — GPU voxelization
try:
    import warp as wp
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Warp GPU kernel (must be at module level — warp requires file scope)
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
    """GPU surface-mark (warp BVH) + CPU flood-fill (scipy)."""
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

    threshold   = pitch * (3.0 ** 0.5) / 2.0
    surface_gpu = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    b_min_wp    = wp.vec3(float(b_min[0]), float(b_min[1]), float(b_min[2]))

    wp.launch(_warp_mark_surface, dim=(nx, ny, nz),
              inputs=[wp_mesh.id, surface_gpu, b_min_wp, pitch, threshold],
              device=device)
    wp.synchronize()

    surface_np     = surface_gpu.numpy().astype(bool)
    struct         = ndimage.generate_binary_structure(3, 1)
    padded         = np.pad(surface_np, 1, constant_values=False)
    labeled, _     = ndimage.label(~padded, structure=struct)
    exterior_label = int(labeled[0, 0, 0])
    interior       = (~padded) & (labeled != exterior_label)
    interior       = interior[1:-1, 1:-1, 1:-1]

    return float((interior.sum() + surface_np.sum()) * pitch**3)


def _volume_voxel(mesh: trimesh.Trimesh, resolution: int) -> float:
    pitch = float(mesh.extents.max()) / resolution
    return float(mesh.voxelized(pitch=pitch).fill().volume)


def _volume_convex_hull(mesh: trimesh.Trimesh) -> float:
    return float(abs(mesh.convex_hull.volume))


def auto_tune_voxel_res(
    mesh: trimesh.Trimesh,
    start: int = 50,
    stop: int = 300,
    step: int = 50,
    tol: float = 0.015,
    use_warp: bool = False,
) -> tuple[int, float]:
    """Increase resolution until volume converges. Returns (best_res, volume)."""
    backend = "warp+floodfill" if use_warp else "trimesh"
    print(f"  {'res':>6}  {'pitch':>10}  {'volume':>14}  {'Δ%':>8}  [{backend}]")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}")

    prev_vol = None
    best_res = start
    best_vol = None

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
                best_vol = vol
                print(f"\n  converged at res={res}  (Δ < {tol*100:.1f}%)")
                return best_res, best_vol
        else:
            print(f"  {res:>6}  {pitch:>10.5f}  {vol:>14.6f}  {'—':>8}")

        prev_vol = vol
        best_res = res
        best_vol = vol

    print(f"\n  [warn] did not converge by res={stop}. using res={best_res}.")
    return best_res, best_vol


def _measure_volume(mesh: trimesh.Trimesh,
                    voxel_res: int = DEFAULT_VOXEL_RES,
                    auto_res: bool = False) -> tuple[float, str]:
    """Return (volume_mesh_units3, method_label) using best available method."""
    if mesh.is_watertight:
        return float(abs(mesh.volume)), "watertight"

    if _WARP_AVAILABLE:
        try:
            if auto_res:
                best_res, vol = auto_tune_voxel_res(mesh, use_warp=True)
                method = f"warp+floodfill (auto res={best_res})"
            else:
                vol = _volume_voxel_warp(mesh, voxel_res)
                method = f"warp+floodfill (res={voxel_res})"
            if vol > 0:
                return vol, method
        except Exception as e:
            print(f"  [warn] warp failed: {e} — falling back to trimesh")

    try:
        if auto_res:
            best_res, vol = auto_tune_voxel_res(mesh, use_warp=False)
            method = f"trimesh voxel (auto res={best_res})"
        else:
            vol = _volume_voxel(mesh, voxel_res)
            method = f"trimesh voxel (res={voxel_res})"
        if vol > 0:
            return vol, method
    except Exception as e:
        print(f"  [warn] trimesh voxel failed: {e}")

    try:
        return _volume_convex_hull(mesh), "convex_hull (fallback — overestimates concave)"
    except Exception as e:
        print(f"  [ERROR] all volume methods failed: {e}")
        return 0.0, "failed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ref_mesh(path: str) -> bool:
    name = os.path.splitext(os.path.basename(path).lower())[0]
    return "box" in name


def _load_mesh_info(path: str, voxel_res: int, auto_res: bool = False) -> dict | None:
    if not os.path.exists(path):
        print(f"  skipping (not found): {path}")
        return None

    mesh = trimesh.load(path, force="mesh", process=False)
    if len(mesh.vertices) == 0:
        print(f"  skipping (empty): {path}")
        return None

    mesh.merge_vertices()
    extents        = mesh.bounds[1] - mesh.bounds[0]
    volume, method = _measure_volume(mesh, voxel_res, auto_res=auto_res)

    return {
        "name":     os.path.basename(path),
        "is_ref":   _is_ref_mesh(path),
        "volume":   volume,
        "method":   method,
        "ext_x":    float(extents[0]),
        "ext_y":    float(extents[1]),
        "ext_z":    float(extents[2]),
        "bbox_vol": float(extents[0] * extents[1] * extents[2]),
    }


# ---------------------------------------------------------------------------
# Main stage entry-point
# ---------------------------------------------------------------------------

def compute_volumes(object_mesh_paths: list[str],
                    voxel_res: int = DEFAULT_VOXEL_RES,
                    auto_res: bool = True):
    """Compute real-world volume of each object using ArUco box for scale."""
    res_label = "auto" if auto_res else str(voxel_res)
    print()
    print("=" * 60)
    print(f"STAGE 7: real-world volumes  "
          f"(ref = {REFERENCE_REAL_SIZE_CM} cm ArUco cube  |  voxel_res={res_label})")
    print("=" * 60)

    rows = [_load_mesh_info(p, voxel_res, auto_res=auto_res) for p in object_mesh_paths]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("  No meshes loaded.")
        return

    df = pd.DataFrame(rows)

    raw_cols = ["name", "method", "volume", "bbox_vol", "ext_x", "ext_y", "ext_z"]
    print("\n  Raw mesh measurements (mesh units):")
    print(df[raw_cols].to_string(index=False))

    ref_rows = df[df["is_ref"]]
    if ref_rows.empty:
        print("\n  No box (reference) mesh found — cannot scale. Aborting.")
        return

    ref = ref_rows.iloc[0]
    if ref["volume"] <= 0:
        print(f"\n  Reference mesh '{ref['name']}' has zero volume. Aborting.")
        return

    real_ref_vol = REFERENCE_REAL_SIZE_CM ** 3
    k            = real_ref_vol / ref["volume"]
    linear_scale = k ** (1.0 / 3.0)

    print(f"\n  Scale factor:")
    print(f"    ref mesh vol  = {ref['volume']:.6f} mesh-units³")
    print(f"    real ref vol  = {REFERENCE_REAL_SIZE_CM}³ = {real_ref_vol:.2f} cm³")
    print(f"    k             = {real_ref_vol:.2f} / {ref['volume']:.6f} = {k:.6f} cm³/unit³")
    print(f"    linear_scale  = k^(1/3) = {linear_scale:.6f} cm/unit")

    df["real_vol_cm3"] = df["volume"] * k
    df["real_vol_L"]   = df["real_vol_cm3"] / 1000.0
    df["size_x_cm"]    = df["ext_x"] * linear_scale
    df["size_y_cm"]    = df["ext_y"] * linear_scale
    df["size_z_cm"]    = df["ext_z"] * linear_scale

    result_cols = ["name", "size_x_cm", "size_y_cm", "size_z_cm",
                   "real_vol_cm3", "real_vol_L", "method"]
    print("\n  Real-world dimensions and volumes:")
    print(df[result_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return df
