#!/usr/bin/env python3
"""View .ply and .stl files. Usage: python viewer.py <file> [file2 ...]"""
import sys
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not installed. Run: pip install open3d")

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly not installed. Run: pip install plotly")


def load(path: Path):
    ext = path.suffix.lower()
    if ext == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_vertices():
            raise ValueError(f"Empty/invalid STL: {path}")
        mesh.compute_vertex_normals()
        return mesh
    if ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            return mesh
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_points():
            raise ValueError(f"Empty/invalid PLY: {path}")
        return pcd
    raise ValueError(f"Unsupported extension: {ext}")


def geom_to_trace(geom, name="geom"):
    if isinstance(geom, o3d.geometry.TriangleMesh):
        v = np.asarray(geom.vertices)
        f = np.asarray(geom.triangles)
        vertexcolor = None
        if geom.has_vertex_colors():
            vc = (np.asarray(geom.vertex_colors) * 255).astype(int)
            vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in vc]
        return go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            vertexcolor=vertexcolor,
            color="lightblue" if vertexcolor is None else None,
            opacity=1.0,
            name=name,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
        )
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
        colors = None
        if geom.has_colors():
            c = (np.asarray(geom.colors) * 255).astype(int)
            colors = [f"rgb({r},{g},{b})" for r, g, b in c]
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors or "blue"),
            name=name,
        )
    raise ValueError(f"Unknown geometry type: {type(geom)}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python viewer.py <file.ply|file.stl> [...]")

    traces = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"skip (not found): {p}", file=sys.stderr)
            continue
        try:
            geom = load(p)
            print(f"loaded: {p}")
            traces.append(geom_to_trace(geom, name=p.name))
        except Exception as e:
            print(f"skip ({e}): {p}", file=sys.stderr)

    if not traces:
        sys.exit("Nothing to display.")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="PLY/STL Viewer",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    fig.show()


if __name__ == "__main__":
    main()
