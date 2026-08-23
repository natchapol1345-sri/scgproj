"""Mesh-level utilities shared by reconstruct/watertight stages."""
import numpy as np
import open3d as o3d


def merge_meshes(mesh_list):
    """Merge multiple meshes into one triangle mesh."""
    merged = o3d.geometry.TriangleMesh()
    vertex_offset = 0
    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        merged.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged.triangles.extend(o3d.utility.Vector3iVector(triangles + vertex_offset))
        vertex_offset += len(vertices)
    merged.compute_vertex_normals()
    merged.compute_triangle_normals()
    return merged


def verify_watertight(mesh_path):
    """Verify watertightness using trimesh.

    process=False — PyMeshFix's fill_holes leaves a few intentional duplicate
    seam vertices that trimesh's default merge would collapse, breaking
    edge-manifoldness. The on-disk mesh has 2 faces per edge by index.
    """
    import trimesh
    t = trimesh.load(mesh_path, process=False)
    return t.is_watertight


def clean_merged_scene(mesh):
    """Standard cleanup applied to merged scene meshes."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh
