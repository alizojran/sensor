"""
True 3D CFD (D3Q19 LBM, BGK) of the float in a round bore -- no symmetry assumption.

Runs ONE (geometry, Re) per invocation and saves drag + mid-plane slices:
    python3 float_cfd3d.py <geom> <Re> [steps]
    geom in {sphere, cup6, cup2, blunt, rounded}

'sphere' is a validation case: its drag Cd is compared to the Schiller-Naumann
standard drag curve  Cd = 24/Re (1 + 0.15 Re^0.687).

Drag = pressure (form) drag on the body in the flow (x) direction, time-averaged.
"""
import sys
import numpy as np

geom = sys.argv[1] if len(sys.argv) > 1 else "sphere"
Re = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
maxIter = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
warm = maxIter // 2

# ----------------------------- grid [cells] ------------------------------------------
s = 3.0                                   # cells per mm
nx, ny, nz = 200, 56, 56
cyc, czc = ny // 2, nz // 2
x0 = 30                                   # float nose / sphere front
R = 6.0                                   # float head radius (Ø12) [mm]
Lh, Ltot, rtail = 8.0, 30.0, 0.5
rp = 4.0                                  # pit radius (Ø8)
bore_r = (R + 7.0 / s) * s                # bore radius [cells] (gap ~7 cells)
uLB = 0.05

# ----------------------------- geometry mask -----------------------------------------
xx = np.arange(nx)[:, None, None]
yy = np.arange(ny)[None, :, None]
zz = np.arange(nz)[None, None, :]
rad = np.sqrt((yy - cyc) ** 2 + (zz - czc) ** 2)          # radial distance [cells]
amm = (xx - x0) / s                                        # axial position [mm]
walls = rad >= bore_r                                      # bore wall (+ outside)


def head_profile(rounded):
    Ro = np.zeros_like(amm, dtype=float)
    head = (amm >= 0) & (amm < Lh)
    tail = (amm >= Lh) & (amm <= Ltot)
    Ro = np.where(head, (R * np.sqrt(np.clip(1 - ((Lh - amm) / Lh) ** 2, 0, 1)) if rounded else R), Ro)
    Ro = np.where(tail, R - (R - rtail) * (amm - Lh) / (Ltot - Lh), Ro)
    return Ro


def build():
    if geom == "sphere":
        Rs = 18.0
        xc = x0 + Rs
        body = np.sqrt((xx - xc) ** 2 + (yy - cyc) ** 2 + (zz - czc) ** 2) <= Rs
        return body, np.pi * Rs ** 2, Rs
    dp = {"cup6": 6.0, "cup2": 2.0, "blunt": 0.0, "rounded": 0.0}[geom]
    rounded = geom == "rounded"
    Ro = head_profile(rounded) * s
    env = (amm >= 0) & (amm <= Ltot) & (rad <= Ro)
    if dp > 0:
        pit = (amm >= 0) & (amm < dp) & (rad < rp * s)
        env = env & ~pit
    return env, np.pi * (R * s) ** 2, R * s


body, A_front, Lref = build()
obstacle = body | walls
fluid = ~obstacle

# ----------------------------- D3Q19 -------------------------------------------------
e = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
              [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
w = np.array([1/3] + [1/18] * 6 + [1/36] * 12)
opp = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
ef = e.astype(float)
nulb = uLB * (2 * Lref) / Re                       # Re on body diameter
omega = 1.0 / (3.0 * nulb + 0.5)


def equilibrium(rho, u):
    eu = np.tensordot(ef, u, axes=([1], [0]))      # (19,nx,ny,nz)
    usq = 1.5 * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    return w[:, None, None, None] * rho[None] * (1 + 3 * eu + 4.5 * eu ** 2 - usq[None])


U = np.zeros((3, nx, ny, nz)); U[0] = uLB
feq_in = equilibrium(np.ones((nx, ny, nz)), U)[:, 0, :, :].copy()
fin = equilibrium(np.ones((nx, ny, nz)), U)

ff = fluid & np.roll(obstacle, -1, axis=0)         # front faces (+x neighbour solid)
fb = fluid & np.roll(obstacle, +1, axis=0)         # back faces

print(f"geom={geom} Re={Re} grid={nx}x{ny}x{nz} omega={omega:.3f} bore_r={bore_r:.1f} "
      f"solid={obstacle.sum()} steps={maxIter}", flush=True)

Fs = []
for it in range(maxIter):
    rho = fin.sum(0)
    u = np.tensordot(ef.T, fin, axes=([1], [0])) / rho[None]
    feq = equilibrium(rho, u)
    fout = fin - omega * (fin - feq)
    fout[:, obstacle] = fin[opp][:, obstacle]
    for i in range(19):
        fin[i] = np.roll(fout[i], tuple(e[i]), axis=(0, 1, 2))
    fin[:, 0, :, :] = feq_in                        # equilibrium velocity inlet
    fin[:, -1, :, :] = fin[:, -2, :, :]             # outflow
    if it >= warm:
        p = (rho - 1.0) / 3.0
        Fs.append(float(p[ff].sum() - p[fb].sum()))
    if it % 1000 == 0:
        mx = np.nanmax(np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)) / uLB
        print(f"  it={it} max|u|/U={mx:.2f}", flush=True)
        if not np.isfinite(mx):
            print("  diverged"); sys.exit(1)

F = np.array(Fs)
Fbar = F.mean()
Cd = Fbar / (0.5 * uLB ** 2 * A_front)
puls = F.std() / abs(Fbar) * 100
rho = fin.sum(0)
u = np.tensordot(ef.T, fin, axes=([1], [0])) / rho[None]
spd = np.sqrt((u ** 2).sum(0)) / uLB
spd[obstacle] = np.nan
mid_xy = spd[:, :, czc]          # meridional slice
cross = spd[x0 + int(3 * s), :, :]    # cross-section just behind the head
note = ""
if geom == "sphere":
    Cd_sn = 24 / Re * (1 + 0.15 * Re ** 0.687)
    note = f" | Schiller-Naumann Cd={Cd_sn:.3f} (ratio {Cd/Cd_sn:.2f})"
print(f"RESULT {geom} Re={Re}: Cd={Cd:.3f} puls={puls:.1f}%{note}", flush=True)
np.savez(f"{__file__.rsplit('/',1)[0]}/cfd3d_{geom}_{int(Re)}.npz",
         geom=geom, Re=Re, Cd=Cd, puls=puls, Fbar=Fbar, omega=omega,
         mid_xy=mid_xy, cross=cross, obstacle_mid=obstacle[:, :, czc],
         obstacle_cross=obstacle[x0 + int(3 * s), :, :], s=s, x0=x0, cyc=cyc)
print("saved", flush=True)
