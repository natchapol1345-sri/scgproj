"""Stage 5 — Make each reconstructed mesh watertight via PyMeshFix worker."""
import os
import shutil
import subprocess
import sys

import open3d as o3d

from pipeline.core.mesh import merge_meshes, verify_watertight, clean_merged_scene


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_MESHFIX_WORKER = os.path.join(_PROJECT_ROOT, "workers", "meshfix_worker.py")


def _wt_name(recon_path):
    """Derive output name from recon path: box_recon.ply → box, obj_recon.ply → obj."""
    base = os.path.basename(recon_path)
    name = os.path.splitext(base)[0]
    if name.endswith("_recon"):
        name = name[:-len("_recon")]
    return name


def make_watertight_meshes(recon_paths, output_folder="output_mesh", base_name="scene"):
    """Fill each reconstructed mesh to be watertight (PyMeshFix + color transfer).

    Returns (scene_ply, scene_stl, watertight_mesh_paths).
    Files saved as box.ply / obj.ply (watertight) based on input names.
    """
    os.makedirs(output_folder, exist_ok=True)

    meshes = []
    watertight_paths = []

    for i, recon_path in enumerate(recon_paths):
        print(f"\nRepairing: {recon_path}")

        name = _wt_name(recon_path)
        wt_ply = os.path.join(output_folder, f"{name}.ply")
        wt_stl = os.path.join(output_folder, f"{name}.stl")

        # Remove stale files from previous runs so existence == success.
        for p in (wt_ply, wt_stl):
            if os.path.exists(p):
                os.remove(p)

        # meshfix_worker.py takes (input, output, color_source).
        # Retry up to 3 times for transient subprocess failures (PyMeshFix
        # C++ side has historically had random "double free" crashes).
        # Exit codes: 0 = watertight OK, 2 = file written but not watertight,
        # 1 = hard failure.
        success = False
        last_err = ""
        for attempt in range(3):
            result = subprocess.run(
                [sys.executable, _MESHFIX_WORKER, recon_path, wt_ply, recon_path],
                capture_output=True, text=True, timeout=600,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    print(f"  {line}")

            if result.returncode in (0, 2) and os.path.exists(wt_ply):
                success = True
                break

            last_err = result.stderr.strip() or f"exit code {result.returncode}"
            if attempt < 2:
                print(f"  MeshFix attempt {attempt+1} failed ({last_err[:100]}), retrying...")

        if not success:
            print(f"  ERROR: MeshFix failed after 3 attempts: {last_err[:500]}")
            shutil.copy2(recon_path, wt_ply)
            print("  Fallback: saved recon mesh as output (NOT watertight)")

        mesh = o3d.io.read_triangle_mesh(wt_ply)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(wt_stl, mesh)

        is_wt = verify_watertight(wt_ply)
        size_mb = os.path.getsize(wt_ply) / (1024 * 1024)
        wt_status = "watertight" if is_wt else "NOT watertight (recon fallback)"
        print(f"  {name}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.triangles):,} faces, {wt_status} ({size_mb:.1f} MB)")

        watertight_paths.append(wt_ply)
        meshes.append(mesh)

    if len(meshes) == 0:
        raise ValueError("No watertight meshes produced")

    print("\nMerging watertight meshes...")
    final_mesh = merge_meshes(meshes)
    final_mesh = clean_merged_scene(final_mesh)
    print(f"Scene triangles: {len(final_mesh.triangles):,}")

    scene_ply = os.path.join(output_folder, f"{base_name}.ply")
    scene_stl = os.path.join(output_folder, f"{base_name}.stl")
    o3d.io.write_triangle_mesh(scene_ply, final_mesh)
    o3d.io.write_triangle_mesh(scene_stl, final_mesh)

    # scene_colour.ply — merge per-object watertight meshes preserving vertex
    # colors (o3d merge path drops them). process=False keeps PyMeshFix seam
    # duplicates intact so each object stays watertight after concatenation.
    import trimesh
    tm_objs = [trimesh.load(p, process=False) for p in watertight_paths]
    scene_colour = trimesh.util.concatenate(tm_objs)
    scene_colour_ply = os.path.join(output_folder, f"{base_name}_colour.ply")
    scene_colour_stl = os.path.join(output_folder, f"{base_name}_colour.stl")
    scene_colour.export(scene_colour_ply)
    scene_colour.export(scene_colour_stl)  # STL is geometry-only; colors dropped
    cs_size_mb = os.path.getsize(scene_colour_ply) / (1024 * 1024)
    print(f"Scene colour: {len(scene_colour.vertices):,} verts, "
          f"{len(scene_colour.faces):,} faces ({cs_size_mb:.1f} MB) "
          f"→ {scene_colour_ply} + {scene_colour_stl}")

    return scene_ply, scene_stl, watertight_paths


def watertight_stage(recon_paths, output_dir):
    """Pipeline wrapper. Returns (scene_watertight_path, watertight_mesh_paths)."""
    print()
    print("=" * 60)
    print("STAGE 5: Making meshes watertight (PyMeshFix)")
    print("=" * 60)

    mesh_output_dir = os.path.join(output_dir, "mesh")

    try:
        scene_wt, _, wt_paths = make_watertight_meshes(
            recon_paths=recon_paths,
            output_folder=mesh_output_dir,
            base_name="scene",
        )
        print(f"  Scene watertight mesh: {scene_wt}")
        for p in wt_paths:
            print(f"  Object watertight mesh: {p}")
        return scene_wt, wt_paths
    except Exception as e:
        print(f"  ERROR during watertight repair: {e}")
        return None, []
