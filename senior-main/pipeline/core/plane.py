"""Deterministic RANSAC plane detection and orientation helpers."""
import numpy as np
import torch


def auto_ransac_threshold(pcd, base_factor=3):
    """Estimate RANSAC distance_threshold from point cloud density."""
    distances = pcd.compute_nearest_neighbor_distance()
    median_nn = np.median(distances)
    threshold = median_nn * base_factor
    print(f"Auto RANSAC threshold: {threshold:.6f} (median nn: {median_nn:.6f} x {base_factor})")
    return threshold


def _sample_triplets(n, m, gen, device):
    """Sample m triplets of distinct point indices. Returns (tri (m,3), valid (m,))."""
    tri = torch.randint(0, n, (m, 3), generator=gen, device=device)
    for _ in range(128):
        bad = ((tri[:, 0] == tri[:, 1]) |
               (tri[:, 1] == tri[:, 2]) |
               (tri[:, 0] == tri[:, 2]))
        nbad = int(bad.sum())
        if nbad == 0:
            break
        idx = bad.nonzero(as_tuple=True)[0]
        tri[idx] = torch.randint(0, n, (nbad, 3), generator=gen, device=device)
    valid = ~((tri[:, 0] == tri[:, 1]) |
              (tri[:, 1] == tri[:, 2]) |
              (tri[:, 0] == tri[:, 2]))
    return tri, valid


def detect_plane_ransac_deterministic(pcd, distance_threshold=0.015, num_iterations=1000, seed=42):
    """GPU-batched deterministic RANSAC plane detection via torch (CUDA, CPU fallback).

    All hypotheses are evaluated as one batched matmul instead of a Python loop.
    Determinism is preserved through a seeded torch.Generator.
    """
    pts_np = np.asarray(pcd.points)
    n = len(pts_np)
    if n < 3:
        raise ValueError("Need at least 3 points for plane detection")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RANSAC: {num_iterations} iters, {n:,} pts, device={device}")

    pts = torch.as_tensor(pts_np, dtype=torch.float32, device=device)  # (N, 3)
    gen = torch.Generator(device=device).manual_seed(seed)

    m = num_iterations
    tri, valid = _sample_triplets(n, m, gen, device)

    p1 = pts[tri[:, 0]]                                # (M, 3)
    p2 = pts[tri[:, 1]]
    p3 = pts[tri[:, 2]]
    normals = torch.linalg.cross(p2 - p1, p3 - p1)     # (M, 3)
    norm_len = normals.norm(dim=1, keepdim=True)       # (M, 1)
    valid = valid & (norm_len.squeeze(1) > 1e-12)
    normals = normals / norm_len.clamp_min(1e-12)
    d = -(normals * p1).sum(dim=1)                     # (M,)

    # Inlier counts per hypothesis: |pts @ normals.T + d| <= threshold.
    # Chunk over hypotheses to bound the (N, chunk) intermediate (~200 MB).
    counts = torch.empty(m, dtype=torch.int64, device=device)
    chunk = min(m, max(1, 50_000_000 // n))
    for a in range(0, m, chunk):
        b = min(a + chunk, m)
        dist = (pts @ normals[a:b].t() + d[a:b]).abs()  # (N, b-a)
        counts[a:b] = (dist <= distance_threshold).sum(dim=0)

    counts[~valid] = -1
    best_m = int(counts.argmax())
    if counts[best_m].item() < 0:
        raise ValueError("Could not detect any plane")

    nrm = normals[best_m]
    dd = d[best_m]
    dist_best = (pts @ nrm + dd).abs()                  # (N,)
    inlier_mask = dist_best <= distance_threshold

    best_plane = np.array([nrm[0].item(), nrm[1].item(), nrm[2].item(), dd.item()])
    best_inliers = inlier_mask.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
    return best_plane, best_inliers


def get_rotation_to_z_axis(normal):
    """Calculates a rotation matrix to align the given normal with [0, 0, 1]."""
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)

    if s < 1e-8:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
    return r


def relocate_to_origin(pcd):
    """Moves the point cloud so XY is centered at 0 and Z-min is at 0."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    tx = -(min_bound[0] + max_bound[0]) / 2
    ty = -(min_bound[1] + max_bound[1]) / 2
    tz = -min_bound[2]

    pcd.translate((tx, ty, tz))
    return pcd
