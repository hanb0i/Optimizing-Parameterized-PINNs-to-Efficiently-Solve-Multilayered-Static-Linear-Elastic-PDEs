#!/usr/bin/env python3
"""
Generate simple ASCII STL primitives used by this project.

Creates:
  - unit_plate_1x1x0p1.stl : a 1 x 1 x 0.1 rectangular prism
  - sphere.stl             : a sphere centered in [0,1]^3 by default

STL stores geometry only (no material properties).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Tri = Tuple[Vec3, Vec3, Vec3]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: Vec3) -> Vec3:
    n = _norm(v)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _tri_normal(tri: Tri) -> Vec3:
    a, b, c = tri
    ab = _sub(b, a)
    ac = _sub(c, a)
    return _unit(_cross(ab, ac))


def write_ascii_stl(path: Path, name: str, triangles: Iterable[Tri]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            nx, ny, nz = _tri_normal(tri)
            f.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
            f.write("    outer loop\n")
            for (x, y, z) in tri:
                f.write(f"      vertex {x:.8e} {y:.8e} {z:.8e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def make_unit_plate(Lx: float, Ly: float, H: float) -> List[Tri]:
    # 8 corners (axis-aligned box)
    p000 = (0.0, 0.0, 0.0)
    p100 = (Lx, 0.0, 0.0)
    p010 = (0.0, Ly, 0.0)
    p110 = (Lx, Ly, 0.0)
    p001 = (0.0, 0.0, H)
    p101 = (Lx, 0.0, H)
    p011 = (0.0, Ly, H)
    p111 = (Lx, Ly, H)

    tris: List[Tri] = []

    # bottom (z=0)
    tris += [(p000, p110, p100), (p000, p010, p110)]
    # top (z=H)
    tris += [(p001, p101, p111), (p001, p111, p011)]
    # x=0
    tris += [(p000, p001, p011), (p000, p011, p010)]
    # x=Lx
    tris += [(p100, p110, p111), (p100, p111, p101)]
    # y=0
    tris += [(p000, p100, p101), (p000, p101, p001)]
    # y=Ly
    tris += [(p010, p011, p111), (p010, p111, p110)]

    return tris


def make_uv_sphere(center: Vec3, radius: float, n_lat: int, n_lon: int) -> List[Tri]:
    n_lat = max(3, int(n_lat))
    n_lon = max(6, int(n_lon))

    cx, cy, cz = center
    tris: List[Tri] = []

    def p(theta: float, phi: float) -> Vec3:
        # theta: [0, pi] (polar), phi: [0, 2pi) (azimuth)
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        return (cx + x, cy + y, cz + z)

    # Build quads between latitude rings and split into triangles.
    for i in range(n_lat - 1):
        theta0 = math.pi * (i / (n_lat - 1))
        theta1 = math.pi * ((i + 1) / (n_lat - 1))
        for j in range(n_lon):
            phi0 = 2.0 * math.pi * (j / n_lon)
            phi1 = 2.0 * math.pi * ((j + 1) / n_lon)

            v00 = p(theta0, phi0)
            v01 = p(theta0, phi1)
            v10 = p(theta1, phi0)
            v11 = p(theta1, phi1)

            # Skip degenerate triangles at the poles by collapsing correctly.
            if i == 0:
                tris.append((v00, v10, v11))
            elif i == n_lat - 2:
                tris.append((v00, v10, v01))
            else:
                tris.append((v00, v10, v11))
                tris.append((v00, v11, v01))

    return tris


def _parse_vec3(values: Sequence[str]) -> Vec3:
    if len(values) != 3:
        raise ValueError("Expected 3 values for a vector.")
    return (float(values[0]), float(values[1]), float(values[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate STL primitives (plate + sphere).")
    parser.add_argument("--outdir", default=str(Path(__file__).parent), help="Output directory for STL files.")

    parser.add_argument("--plate", action="store_true", help="Generate the unit plate STL.")
    parser.add_argument("--plate-lx", type=float, default=1.0)
    parser.add_argument("--plate-ly", type=float, default=1.0)
    parser.add_argument("--plate-h", type=float, default=0.1)

    parser.add_argument("--sphere", action="store_true", help="Generate the sphere STL.")
    parser.add_argument("--sphere-center", nargs=3, default=("0.5", "0.5", "0.5"))
    parser.add_argument("--sphere-radius", type=float, default=0.5)
    parser.add_argument("--sphere-lat", type=int, default=30, help="Latitude resolution (>=3).")
    parser.add_argument("--sphere-lon", type=int, default=60, help="Longitude resolution (>=6).")

    args = parser.parse_args()
    outdir = Path(args.outdir)

    if not args.plate and not args.sphere:
        args.plate = True
        args.sphere = True

    if args.plate:
        tris = make_unit_plate(args.plate_lx, args.plate_ly, args.plate_h)
        out = outdir / "unit_plate_1x1x0p1.stl"
        write_ascii_stl(out, "unit_plate_1x1x0p1", tris)
        print(f"Wrote {out}")

    if args.sphere:
        center = _parse_vec3(args.sphere_center)
        tris = make_uv_sphere(center=center, radius=float(args.sphere_radius), n_lat=args.sphere_lat, n_lon=args.sphere_lon)
        out = outdir / "sphere.stl"
        write_ascii_stl(out, "sphere", tris)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

