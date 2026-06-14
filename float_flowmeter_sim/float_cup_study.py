"""
A/B/C study of the float HEAD shape (same Ø12 max-dia + same cone tail, only the
front differs). Flow left->right; float centred; bore walls top & bottom.

  A  cup     : blunt head with a forward cylindrical pit  Ø8 x 2 deep  (the design)
  B  blunt   : blunt flat-faced head, no pit
  C  rounded : rounded (ogive) nose, no sharp rim          (Re-sensitive control)

These bluff floats are held up by DRAG, so we measure the pressure (form) drag on
the body in the flow direction.  The forward cavity makes the cup flow UNSTEADY,
so the drag and the velocity field are TIME-AVERAGED over the last `avg` steps
(for the steady blunt/rounded cases this just equals the steady value).
With the inlet velocity fixed, Cd_d is proportional to the drag, so its variation
with Reynolds number is the viscosity-(in)sensitivity we care about.

Outputs cup_study.npz (time-averaged fields at one Re + drag sweep + fits).
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

# ----------------------------- grid / geometry [mm,cells] ----------------------------
s = 7.0
nx, ny = 480, 128
cy, wall = 64, 7
bore_r = cy - wall
x0 = 70
R, Lh, Ltot, rtail = 6.0, 5.0, 26.0, 0.5
rp, dp = 4.0, 2.0
Dchar = 2 * R * s
A_front = 2 * R * s

X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
amm = (X - x0) / s
rr = np.abs(Y - cy) / s


def envelope(rounded=False):
    Ro = np.zeros_like(amm)
    head = (amm >= 0) & (amm < Lh)
    tail = (amm >= Lh) & (amm <= Ltot)
    Ro[head] = R * np.sqrt(np.clip(1 - ((Lh - amm[head]) / Lh) ** 2, 0, 1)) if rounded else R
    Ro[tail] = R - (R - rtail) * (amm[tail] - Lh) / (Ltot - Lh)
    return (amm >= 0) & (amm <= Ltot) & (rr <= Ro)


def mask_cup():
    return envelope(False) & ~((amm >= 0) & (amm < dp) & (rr < rp))


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


def run(maskfun, Re, uLB, maxIter=14000, avg=6000, want_field=False, tag=""):
    obstacle = maskfun()
    face_front = (~obstacle) & np.roll(obstacle, -1, axis=0)
    face_back = (~obstacle) & np.roll(obstacle, +1, axis=0)
    nulb = uLB * Dchar / Re
    omega = 1.0 / (3.0 * nulb + 0.5)
    vel = np.zeros((2, nx, ny))
    vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]
    fin = equilibrium(np.ones((nx, ny)), vel)
    warm = maxIter - avg
    Fsum, Nf = 0.0, 0
    uxs = np.zeros((nx, ny)); uys = np.zeros((nx, ny))
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
        if it >= warm:
            p = (rho - 1.0) / 3.0
            F = float(p[face_front].sum() - p[face_back].sum())
            if not np.isfinite(F):
                print(f"  [{tag}] diverged it={it}"); return None
            Fsum += F; Nf += 1
            if want_field:
                uxs += u[0]; uys += u[1]
    Fbar = Fsum / Nf
    Cd_d = Fbar / (0.5 * uLB ** 2 * A_front)
    res = dict(Re=Re, uLB=uLB, Fdrag=Fbar, Cd_d=Cd_d, omega=float(omega))
    if want_field:
        res.update(ux=uxs / Nf, uy=uys / Nf, obstacle=obstacle)
    print(f"  [{tag}] Re={Re:5.0f} omega={omega:.3f} <Fdrag>={Fbar:.4e} Cd_d={Cd_d:.3f}")
    return res


if __name__ == "__main__":
    test = "test" in sys.argv
    Re_field = 100.0
    Re_sweep, uLB = [50.0, 100.0, 200.0], 0.04
    kw = dict(maxIter=700, avg=300) if test else {}
    if test:
        Re_sweep = [100.0]

    sweep, fields, betas, Cinf = {}, {}, {}, {}
    print("=== drag sweep (time-averaged) ===")
    for g, mf in GEOMS.items():
        sweep[g] = {"Re": [], "Cd_d": []}
        for Re in Re_sweep:
            r = run(mf, Re, uLB, want_field=(Re == Re_field), tag=g, **kw)
            if r:
                sweep[g]["Re"].append(Re); sweep[g]["Cd_d"].append(r["Cd_d"])
                if Re == Re_field:
                    fields[g] = r

    def model(Re, a, b):
        return a * (1 + b / np.sqrt(Re))

    for g in GEOMS:
        Re = np.array(sweep[g]["Re"]); cd = np.array(sweep[g]["Cd_d"])
        if len(Re) >= 2:
            try:
                popt, _ = curve_fit(model, Re, cd, p0=[cd.mean(), 1.0], maxfev=20000)
            except Exception:
                popt = [cd.mean(), 0.0]
            Cinf[g], betas[g] = float(popt[0]), float(popt[1])
            print(f"  fit {g}: Cd_inf={popt[0]:.3f} beta={popt[1]:.3f}")
        else:
            Cinf[g] = float(cd[0]) if len(cd) else float("nan"); betas[g] = float("nan")

    out = __file__.rsplit("/", 1)[0] + "/cup_study.npz"
    np.savez(out,
             **{f"{g}_ux": fields[g]["ux"] for g in GEOMS},
             **{f"{g}_uy": fields[g]["uy"] for g in GEOMS},
             **{f"{g}_obst": fields[g]["obstacle"] for g in GEOMS},
             **{f"sw_{g}_Re": sweep[g]["Re"] for g in GEOMS},
             **{f"sw_{g}_Cd": sweep[g]["Cd_d"] for g in GEOMS},
             **{f"beta_{g}": betas[g] for g in GEOMS},
             **{f"Cinf_{g}": Cinf[g] for g in GEOMS},
             s=s, nx=nx, ny=ny, cy=cy, x0=x0, Re_field=Re_field, uLB_field=uLB)
    print("saved:", out)
    print("BETAS:", betas, "CINF:", Cinf)
