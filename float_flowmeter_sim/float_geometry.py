"""
Float geometry for the axisymmetric rotameter study.

Axis on r=0; dimensions in mm.   Domain layout (axial z):
   [ inlet run z_in ] [ float 0..Ltot mapped to z_in.. ] [ wake ]
We place the float nose at z = z_in.  Local body coord  zb = z - z_in.

Float (body of revolution), local zb in [0, Ltot]:
  Head: zb in [0, Lh].
     blunt/cup : cylinder radius R
     rounded   : ogive  R*sqrt(1 - ((Lh-zb)/Lh)^2)
  Tail: zb in [Lh, Ltot]: cone from (Lh,R) to (Ltot, rtail)
  Solid = { r <= profile(zb) }.
  cup6/cup2: remove solid (carve fluid pit) where zb < depth and r < rp.

Returns a boolean solid-cell mask on the cell-centre grid (z_c, r_c).
"""
import numpy as np

# canonical dimensions (mm)
R = 6.0          # head radius (Ø12)
Lh = 8.0         # head length
Ltot = 30.0      # total float length
rtail = 0.5      # tail tip radius
rp = 4.0         # pit radius (Ø8)
GAP = 2.5        # annular gap nose-to-bore
R_OUT = R + GAP  # bore radius = 8.5 mm

Z_IN = 10.0      # inlet run before nose
Z_WAKE = 30.0    # wake after tail
LZ = Z_IN + Ltot + Z_WAKE     # total domain length = 70 mm

# --- redesigned float: blunt face Ø12 + short metering land + streamlined cone tail ---
LAND_RD = 3.0    # cylindrical metering land length
LTOT_RD = 26.0   # redesigned float total length
RTAIL_RD = 1.5   # redesigned tail tip radius (Ø3)


def profile_radius(zb, variant):
    """Body radius vs local axial coord zb (array), 0 outside [0,Ltot]."""
    Ro = np.zeros_like(zb)
    head = (zb >= 0) & (zb < Lh)
    tail = (zb >= Lh) & (zb <= Ltot)
    if variant == "rounded":
        Ro[head] = R * np.sqrt(np.clip(1 - ((Lh - zb[head]) / Lh) ** 2, 0, 1))
    else:
        Ro[head] = R
    Ro[tail] = R - (R - rtail) * (zb[tail] - Lh) / (Ltot - Lh)
    return Ro


def solid_mask(z_c, r_c, variant, depth=None):
    """
    Boolean (nz, nr) solid mask.  variant in {blunt, cup6, cup2, rounded}.
    cup depth defaults: cup6->6, cup2->2.
    """
    Z, Rg = np.meshgrid(z_c, r_c, indexing="ij")
    zb = Z - Z_IN
    if variant == "redesign":
        Ro = np.zeros_like(zb)
        head = (zb >= 0) & (zb < LAND_RD)
        tail = (zb >= LAND_RD) & (zb <= LTOT_RD)
        Ro[head] = R
        Ro[tail] = R - (R - RTAIL_RD) * (zb[tail] - LAND_RD) / (LTOT_RD - LAND_RD)
        return (zb >= 0) & (zb <= LTOT_RD) & (Rg <= Ro)
    Ro = profile_radius(zb, "rounded" if variant == "rounded" else "blunt")
    env = (zb >= 0) & (zb <= Ltot) & (Rg <= Ro)
    if variant in ("cup6", "cup2"):
        d = depth if depth is not None else (6.0 if variant == "cup6" else 2.0)
        pit = (zb >= 0) & (zb < d) & (Rg < rp)
        env = env & ~pit
    return env


VARIANTS = ["cup6", "cup2", "blunt", "rounded"]


if __name__ == "__main__":
    # quick sanity: cell counts & pit resolution at 8 cells/mm
    cpm = 8
    dz = dr = 1.0 / cpm
    nz = int(round(LZ / dz)); nr = int(round(R_OUT / dr))
    z_c = (np.arange(nz) + 0.5) * dz
    r_c = (np.arange(nr) + 0.5) * dr
    print(f"domain Lz={LZ} R_out={R_OUT}  nz={nz} nr={nr}  cells/mm={cpm}")
    print(f"pit diameter Ø{2*rp} -> {int(2*rp*cpm)} cells across (need >=8x ... actually radius {int(rp*cpm)} cells)")
    for v in VARIANTS:
        m = solid_mask(z_c, r_c, v)
        # front-face area cells, solid volume (approx)
        vol = np.sum(m * r_c[None, :]) * 2 * np.pi * dz * dr
        print(f"  {v:8s}: solid cells={m.sum():6d}  approx solid vol={vol:7.2f} mm^3")
