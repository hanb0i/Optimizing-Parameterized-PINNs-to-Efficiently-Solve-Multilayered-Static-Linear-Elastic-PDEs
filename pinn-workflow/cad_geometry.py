from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import struct

import numpy as np


@dataclass(frozen=True)
class STLBounds:
    min_xyz: np.ndarray  # (3,)
    max_xyz: np.ndarray  # (3,)

    @property
    def size(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz


def _read_stl_triangles(stl_path: str | Path) -> np.ndarray:
    """
    Minimal STL reader (ASCII or binary) that returns triangles as (T, 3, 3) float64.

    This avoids extra dependencies (and avoids the common `stl` vs `numpy-stl` import clash).
    """
    stl_path = Path(stl_path)
    data = stl_path.read_bytes()
    size = len(data)
    if size < 84:
        raise ValueError(f"Invalid STL (too small): {stl_path}")

    # Heuristic: if size matches 84 + 50*N for the uint32 N at byte 80, treat as binary.
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * tri_count
    is_binary = expected == size

    if is_binary:
        # Parse triangles
        # Each record: normal(12) + v0(12) + v1(12) + v2(12) + attr(2)
        tri_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v0", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        records = np.frombuffer(data, dtype=tri_dtype, count=tri_count, offset=84)
        vectors = np.stack([records["v0"], records["v1"], records["v2"]], axis=1)
        return vectors.astype(np.float64, copy=False)

    # ASCII fallback: parse `vertex` lines
    # NOTE: This will not handle every malformed STL, but covers typical CAD exports.
    text = data.decode("utf-8", errors="ignore")
    vertices = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("vertex "):
            continue
        parts = s.split()
        if len(parts) != 4:
            continue
        try:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue

    if len(vertices) % 3 != 0 or len(vertices) == 0:
        raise ValueError(f"Failed to parse STL vertices: {stl_path}")

    vectors = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    return vectors


def stl_bounds(stl_path: str | Path) -> STLBounds:
    vectors = _read_stl_triangles(stl_path)
    min_xyz = vectors.reshape(-1, 3).min(axis=0)
    max_xyz = vectors.reshape(-1, 3).max(axis=0)
    return STLBounds(min_xyz=min_xyz, max_xyz=max_xyz)


def affine_map_points(
    points: np.ndarray,
    src_bounds: STLBounds,
    dst_min_xyz: Tuple[float, float, float],
    dst_max_xyz: Tuple[float, float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    src_min = src_bounds.min_xyz
    src_size = np.maximum(src_bounds.size, 1e-12)
    dst_min = np.asarray(dst_min_xyz, dtype=np.float64)
    dst_max = np.asarray(dst_max_xyz, dtype=np.float64)
    dst_size = np.maximum(dst_max - dst_min, 1e-12)
    return (points - src_min) / src_size * dst_size + dst_min


def sample_uniform_box(n: int, min_xyz: Tuple[float, float, float], max_xyz: Tuple[float, float, float]) -> np.ndarray:
    mn = np.asarray(min_xyz, dtype=np.float64)
    mx = np.asarray(max_xyz, dtype=np.float64)
    u = np.random.rand(n, 3)
    return mn + u * (mx - mn)


def sample_uniform_rect_on_plane(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z: float,
) -> np.ndarray:
    x = np.random.rand(n) * (x_max - x_min) + x_min
    y = np.random.rand(n) * (y_max - y_min) + y_min
    z_arr = np.full(n, z, dtype=np.float64)
    return np.stack([x, y, z_arr], axis=1)
