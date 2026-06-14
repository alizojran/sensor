"""
A/B/C study of the float HEAD shape (same Ø12 max-dia + same cone tail, only the
front differs). Flow left->right; float centred; bore walls top & bottom.

  A  cup     : blunt head with a forward cylindrical pit  Ø8 x 2 deep  (the design)
  B  blunt   : blunt flat-faced head, no pit
  C  rounded : rounded (ogive) nose, no sharp rim          (Re-sensitive control)

These are BLUFF floats held up by DRAG, so we measure the drag via the far-field
pressure drop  dPdrag = p(upstream) - p(downstream)  ~ form drag.  With the inlet
velocity fixed, the drag coefficient  Cd_d  is proportional to dPdrag, so its
variation with Reynolds number is the viscosity-(in)sensitivity we care about.

Outputs cup_study.npz (fields at one Re + dPdrag sweep + fitted coefficients).
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

# ----------------------------- grid / geometry [mm,cells] ----------------------------
s = 7.0
nx, ny = 480, 128
cy, wall = 64, 7
bore_r = cy - wall                       # 57
x0 = 70
R = 6.0                                   # head radius (Ø12)
Lh = 5.0                                  # blunt head length
Ltot = 26.0                               # tail tip
rtail = 0.5
rp, dp = 4.0, 2.0                         # pit radius (Ø8) / depth
Dchar = 2 * R * s                         # float diameter [cells]

X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
amm = (X - x0) / s
rr = np.abs(Y - cy) / s                   # radial position [mm]


def envelope(rounded=False):
    Ro = np.zeros_like(amm)
    head = (amm >= 0) & (amm < Lh)
    tail = (amm >= Lh) & (amm <= Ltot)
    if rounded:
        Ro[head] = R * np.sqrt(np.clip(1 - ((Lh - amm[head]) / Lh) ** 2, 0, 1))
    else:
        Ro[head] = R
    Ro[tail] = R - (R - rtail) * (amm[tail] - Lh) / (Ltot - Lh)
    return (amm >= 0) & (amm <= Ltot) & (rr <= Ro)


def mask_cup():
    env = envelope(False)
    pit = (amm >= 0) & (amm < dp) & (rr < rp)
    return env & ~pit


def mask_blunt():
    return envelope(False)


def mask_rounded():
    return envelope(True)


GEOMS = {"cup": mask_cup, "blunt": mask_blunt, "rounded": mask_rounded}

# ----------------------------- D2Q9 LBM ----------------------------------------------
v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
vf = v.astype(float)
t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
i1 = np.array([6, 7, 8]); i2 = np.array([3, 4, 5]); i3 = np.array([0, 1, 2])
opp = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])


def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(vf, u, axes=([1], [0]))
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * t[:, None, None] * (1 + cu + 0.5 * cu * cu - usqr[None])


A_front = 2 * R * s                       # projected frontal height [cells]


def run(maskfun, Re, uLB, maxIter=24000, minIter=4000, check=400, tol=2.0e-3,
        want_field=False, tag=""):
    obstacle = maskfun()
    fluid = ~obstacle
    # fluid cells whose +x / -x neighbour is solid  -> front / back faces of the body
    face_front = fluid & np.roll(obstacle, -1, axis=0)   # pressure here pushes body +x
    face_back = fluid & np.roll(obstacle, +1, axis=0)     # pressure here pushes body -x
    nulb = uLB * Dchar / Re
    omega = 1.0 / (3.0 * nulb + 0.5)
    vel = np.zeros((2, nx, ny))
    vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]
    fin = equilibrium(np.ones((nx, ny)), vel)
    prev, hits = None, 0

    def drag(rho):
        p = (rho - 1.0) / 3.0                  # form (pressure) drag on the body, x-dir
        return float(p[face_front].sum() - p[face_back].sum())

    for it in range(maxIter):
        fin[i1, -1, :] = fin[i1, -2, :]
        rho = np.sum(fin, axis=0)
        u = np.tensordot(vf.T, fin, axes=([1], [0])) / rho[None]
        u[:, 0, :] = vel[:, 0, :]
        rho[0, :] = (1 / (1 - u[0, 0, :])) * (np.sum(fin[i2, 0, :], 0) + 2 * np.sum(fin[i1, 0, :], 0))
        feq = equilibrium(rho, u)
        fin[i3, 0, :] = feq[i3, 0, :] + fin[i1[::-1], 0, :] - feq[i1[::-1], 0, :]
        fout = fin - omega * (fin - feq)
        fout[:, obstacle] = fin[opp][:, obstacle]
        for i in range(9):
            fin[i] = np.roll(np.roll(fout[i], v[i, 0], axis=0), v[i, 1], axis=1)
        if it >= minIter and it % check == 0:
            F = drag(rho)
            if not np.isfinite(F):
                print(f"  [{tag}] diverged it={it}"); return None
            if prev is not None and abs(F - prev) <= tol * abs(F):
                hits += 1
                if hits >= 2:
                    break
            else:
                hits = 0
            prev = F
    rho = np.sum(fin, axis=0)
    u = np.tensordot(vf.T, fin, axes=([1], [0])) / rho[None]
    F = drag(rho)
    Cd_d = F / (0.5 * uLB ** 2 * A_front)          # drag coefficient (V=uLB fixed)
    res = dict(Re=Re, uLB=uLB, Fdrag=F, Cd_d=Cd_d, omega=float(omega), steps=it + 1)
    if want_field:
        res.update(ux=u[0].copy(), uy=u[1].copy(), obstacle=obstacle)
    print(f"  [{tag}] Re={Re:5.0f} steps={it+1:6d} omega={omega:.3f} Fdrag={F:.4e} Cd_d={Cd_d:.3f}")
    return res


if __name__ == "__main__":
    test = "test" in sys.argv
    Re_field, uLB_field = 150.0, 0.05
    Re_sweep, uLB_sweep = [60.0, 150.0, 300.0], 0.03
    kw = dict(maxIter=900, minIter=300, check=200) if test else {}
    if test:
        Re_sweep = [150.0]

    fields, sweep, betas, Cinf = {}, {}, {}, {}
    print("=== field runs (Re=%.0f) ===" % Re_field)
    for g, mf in GEOMS.items():
        fields[g] = run(mf, Re_field, uLB_field, want_field=True, tag="field-" + g, **kw)

    print("=== drag sweep ===")
    for g, mf in GEOMS.items():
        sweep[g] = {"Re": [], "Cd_d": []}
        for Re in Re_sweep:
            r = run(mf, Re, uLB_sweep, tag=g, **kw)
            if r:
                sweep[g]["Re"].append(Re); sweep[g]["Cd_d"].append(r["Cd_d"])

    def model(Re, a, b):
        return a * (1 + b / np.sqrt(Re))                 # drag rises at low Re

    for g in GEOMS:
        Re = np.array(sweep[g]["Re"]); cd = np.array(sweep[g]["Cd_d"])
        if len(Re) >= 2:
            try:
                popt, _ = curve_fit(model, Re, cd, p0=[cd.min(), 1.0], maxfev=20000)
            except Exception:
                popt = [cd.mean(), 0.0]
            Cinf[g], betas[g] = float(popt[0]), float(popt[1])
            print(f"  fit {g}: Cd_inf={popt[0]:.3f}  beta={popt[1]:.3f}")
        else:
            Cinf[g], betas[g] = float(cd[0]) if len(cd) else float("nan"), float("nan")

    out = __file__.rsplit("/", 1)[0] + "/cup_study.npz"
    np.savez(out,
             **{f"{g}_ux": fields[g]["ux"] for g in GEOMS},
             **{f"{g}_uy": fields[g]["uy"] for g in GEOMS},
             **{f"{g}_obst": fields[g]["obstacle"] for g in GEOMS},
             **{f"sw_{g}_Re": sweep[g]["Re"] for g in GEOMS},
             **{f"sw_{g}_Cd": sweep[g]["Cd_d"] for g in GEOMS},
             **{f"beta_{g}": betas[g] for g in GEOMS},
             **{f"Cinf_{g}": Cinf[g] for g in GEOMS},
             s=s, nx=nx, ny=ny, cy=cy, x0=x0, Re_field=Re_field, uLB_field=uLB_field)
    print("saved:", out)
    print("BETAS:", betas)
