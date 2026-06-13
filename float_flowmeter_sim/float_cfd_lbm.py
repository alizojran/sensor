"""
2D CFD of the sharp-edged viscosity-immune float (meridional-plane representation)
using the Lattice Boltzmann Method (D2Q9, BGK single-relaxation-time).

Flow goes left -> right through the bore; the float sits on the centreline, leaving
an annular gap to the (bounce-back) bore walls top and bottom. The goal is to show
the flow TOPOLOGY: acceleration through the gap, a FIXED separation point at the
sharp metering edge, and the recirculation wake behind the float.

NOTE: planar (not axisymmetric) — qualitative flow field, not a quantitative
axisymmetric solution. Laminar regime (Re ~ 150), which is exactly the
low-Re / viscous-fluid regime where float shape matters most.

Usage:  python3 float_cfd_lbm.py [maxIter]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------- domain / grid -----------------------------------------
maxIter = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
s       = 8.0                       # cells per mm
nx, ny  = 540, 144
cy      = 72                        # centreline row
wall    = 8                         # bore-wall thickness (cells), top & bottom
x0      = 80                        # float nose x-position (cells)
Redge   = 6.0                       # float edge radius [mm]
uLB     = 0.05                      # inlet lattice velocity
Re      = 150.0                     # Reynolds number on float edge diameter
nulb    = uLB * (2 * Redge * s) / Re
omega   = 1.0 / (3.0 * nulb + 0.5)

# ----------------------------- float radius profile  R_outer(axial mm) ---------------
def R_outer(a):
    a = np.asarray(a, float)
    r = np.zeros_like(a)
    nose = a < 6.0
    edge = (a >= 6.0) & (a < 6.4)
    body = (a >= 6.4) & (a < 24.0)
    tail = (a >= 24.0) & (a <= 30.0)
    r[nose] = 0.5 + (6.0 - 0.5) * (a[nose] / 6.0)          # nose cone 0.5 -> 6.0
    r[edge] = 6.0                                          # sharp metering edge (max dia.)
    r[body] = 4.5                                          # body (step 6.0->4.5 = sharp edge)
    r[tail] = 4.5 * (1 - (a[tail] - 24) / 6.0) + 0.5 * ((a[tail] - 24) / 6.0)  # streamlined tail
    return r

# ----------------------------- build obstacle mask -----------------------------------
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")   # (nx,ny)
amm  = (X - x0) / s                                               # axial position [mm]
infloat = (amm >= 0) & (amm <= 30)
rprof   = R_outer(np.clip(amm, 0, 30)) * s                        # local radius [cells]
float_mask = infloat & (np.abs(Y - cy) <= rprof)
walls = (Y < wall) | (Y >= ny - wall)
obstacle = float_mask | walls

# ----------------------------- D2Q9 ---------------------------------------------------
v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
i1 = np.array([6, 7, 8])     # v_x < 0  (unknown on right outflow)
i2 = np.array([3, 4, 5])     # v_x = 0
i3 = np.array([0, 1, 2])     # v_x > 0  (unknown on left inflow)
opp = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])


vf = v.astype(float)


def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(vf, u, axes=([1], [0]))      # (9,nx,ny)
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)                 # (nx,ny)
    return rho[None] * t[:, None, None] * (1 + cu + 0.5 * cu * cu - usqr[None])


# inlet velocity (uniform), with tiny perturbation to break symmetry
vel = np.zeros((2, nx, ny))
vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]

fin = equilibrium(np.ones((nx, ny)), vel)

# ----------------------------- main loop ---------------------------------------------
print(f"grid {nx}x{ny}, Re={Re}, omega={omega:.3f}, nulb={nulb:.4f}, steps={maxIter}")
for it in range(maxIter):
    fin[i1, -1, :] = fin[i1, -2, :]                       # outflow (right)
    rho = np.sum(fin, axis=0)
    u = np.tensordot(vf.T, fin, axes=([1], [0])) / rho[None]
    u[:, 0, :] = vel[:, 0, :]                             # inlet velocity (left)
    rho[0, :] = (1 / (1 - u[0, 0, :])) * (np.sum(fin[i2, 0, :], 0)
                                          + 2 * np.sum(fin[i1, 0, :], 0))
    feq = equilibrium(rho, u)
    fin[i3, 0, :] = feq[i3, 0, :] + fin[i1[::-1], 0, :] - feq[i1[::-1], 0, :]  # Zou/He inlet
    fout = fin - omega * (fin - feq)                      # BGK collision
    fout[:, obstacle] = fin[opp][:, obstacle]             # bounce-back on solid
    for i in range(9):                                    # streaming
        fin[i] = np.roll(np.roll(fout[i], v[i, 0], axis=0), v[i, 1], axis=1)

    if it % 3000 == 0:
        if not np.isfinite(rho).all():
            print(f"  diverged at it={it}"); break
        print(f"  it={it:6d}  max|u|/U={np.nanmax(np.sqrt(u[0]**2+u[1]**2))/uLB:.2f}")

# ----------------------------- post-process ------------------------------------------
rho = np.sum(fin, axis=0)
u = np.zeros((2, nx, ny))
for i in range(9):
    u[0] += v[i, 0] * fin[i]; u[1] += v[i, 1] * fin[i]
u /= rho
speed = np.sqrt(u[0] ** 2 + u[1] ** 2) / uLB
ux, uy = u[0].copy(), u[1].copy()
vort = (np.gradient(uy, axis=0) - np.gradient(ux, axis=1)) / uLB
press = (rho - 1.0) / 3.0                                  # gauge pressure (lattice)

for f in (speed, vort, press):
    f[obstacle] = np.nan

np.savez(__file__.rsplit("/", 1)[0] + "/cfd_field.npz",
         ux=ux, uy=uy, speed=speed, vort=vort, press=press,
         obstacle=obstacle, float_mask=float_mask, cy=cy, x0=x0, s=s, uLB=uLB)
print("solver done -> cfd_field.npz")
