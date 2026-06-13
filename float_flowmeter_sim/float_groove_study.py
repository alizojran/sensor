"""
A/B study: smooth sharp-edged cone  vs  cone with a sharp-edged annular GROOVE.

Heavy compute (run in background). Produces groove_study.npz containing:
  * CFD flow fields for both geometries at one Re (for the flow-field figure)
  * a small Reynolds sweep -> pressure drop -> discharge-coefficient droop Cd(Re)
    and a fitted  Cd = Cdinf*(1 - beta/sqrt(Re))  for each geometry.

Cd extraction (fixed inlet velocity => fixed throat flow Q, fixed throat area A):
        Q = Cd * A * sqrt(2*dP/rho)   ->   Cd  proportional to  1/sqrt(dP)
so the SHAPE of Cd(Re) (its low-Re droop = viscosity sensitivity) follows from dP(Re).
dP is measured locally: p(just upstream of nose) - p(at the metering-edge throat).
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

# ----------------------------- grid / geometry ---------------------------------------
s   = 7.0                  # cells per mm
nx, ny = 480, 128
cy  = 64
wall = 7
x0  = 70                   # nose x (cells)
bore_r = cy - wall         # 57 cells
Redge_mm = 6.0
Redge_c  = Redge_mm * s    # 42 cells
Dchar = 2 * Redge_c        # characteristic length (edge diameter, cells)

X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")


def R_smooth(a):
    a = np.asarray(a, float); r = np.zeros_like(a)
    m1 = a < 6.0; m2 = (a >= 6.0) & (a < 6.4); m3 = (a >= 6.4) & (a < 24.0)
    m4 = (a >= 24.0) & (a <= 30.0)
    r[m1] = 0.5 + 5.5 * (a[m1] / 6.0)
    r[m2] = 6.0
    r[m3] = 4.5
    r[m4] = 4.5 * (1 - (a[m4] - 24) / 6.0) + 0.5 * ((a[m4] - 24) / 6.0)
    return r


GROOVE_A = (2.8, 3.6)                       # groove axial extent [mm]
GROOVE_BOTTOM = (0.5 + 5.5 * (2.8 / 6.0)) - 0.7   # recessed flat-slot radius [mm]


def R_groove(a):
    a = np.asarray(a, float)
    r = R_smooth(a).copy()
    g = (a >= GROOVE_A[0]) & (a <= GROOVE_A[1])
    r[g] = GROOVE_BOTTOM                     # sharp-edged annular slot in the front cone
    return r


def build_mask(Rfunc):
    amm = (X - x0) / s
    infl = (amm >= 0) & (amm <= 30)
    rprof = Rfunc(np.clip(amm, 0, 30)) * s
    fmask = infl & (np.abs(Y - cy) <= rprof)
    walls = (Y < wall) | (Y >= ny - wall)
    return (fmask | walls), fmask


# ----------------------------- D2Q9 LBM ----------------------------------------------
v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
vf = v.astype(float)
t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
i1 = np.array([6, 7, 8]); i2 = np.array([3, 4, 5]); i3 = np.array([0, 1, 2])
opp = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])

x_up = int(round(x0 - 5.0 * s))             # upstream pressure plane
x_th = int(round(x0 + 6.0 * s))             # throat (metering-edge) plane


def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(vf, u, axes=([1], [0]))
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * t[:, None, None] * (1 + cu + 0.5 * cu * cu - usqr[None])


def run_lbm(Rfunc, Re, uLB, maxIter=22000, minIter=4000, check=400, tol=2.5e-3,
            want_field=False, tag=""):
    obstacle, fmask = build_mask(Rfunc)
    fluid = ~obstacle
    nulb = uLB * Dchar / Re
    omega = 1.0 / (3.0 * nulb + 0.5)
    vel = np.zeros((2, nx, ny))
    vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]
    fin = equilibrium(np.ones((nx, ny)), vel)

    up_col = fluid[x_up, :]
    th_col = fluid[x_th, :]
    dp_prev, hits = None, 0
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
            p = (rho - 1.0) / 3.0
            dp = p[x_up, up_col].mean() - p[x_th, th_col].mean()
            if not np.isfinite(dp):
                print(f"  [{tag}] diverged it={it}"); return None
            if dp_prev is not None and abs(dp - dp_prev) <= tol * abs(dp):
                hits += 1
                if hits >= 2:
                    break
            else:
                hits = 0
            dp_prev = dp

    rho = np.sum(fin, axis=0)
    u = np.tensordot(vf.T, fin, axes=([1], [0])) / rho[None]
    p = (rho - 1.0) / 3.0
    dp = p[x_up, up_col].mean() - p[x_th, th_col].mean()
    # throat gap Reynolds number (consistent with the meter's gap-Re)
    gapcells = bore_r - Redge_c
    Vth = u[0, x_th, th_col].mean()
    gapRe = Vth * (2 * gapcells) / nulb
    res = dict(dp=float(dp), gapRe=float(gapRe), Vth=float(Vth), omega=float(omega),
               steps=it + 1, Re=Re, uLB=uLB)
    if want_field:
        res["ux"] = u[0].copy(); res["uy"] = u[1].copy()
        res["obstacle"] = obstacle; res["fmask"] = fmask
    print(f"  [{tag}] Re={Re:5.0f} uLB={uLB} steps={it+1:6d} omega={omega:.3f} "
          f"dP={dp:.3e} gapRe={gapRe:7.1f}")
    return res


# ----------------------------- driver ------------------------------------------------
if __name__ == "__main__":
    test = "test" in sys.argv
    Re_field = 120.0
    uLB_field = 0.05
    Re_sweep = [45.0, 90.0, 180.0]
    uLB_sweep = 0.03
    if test:
        Re_field = 60.0; Re_sweep = [60.0]
        kw = dict(maxIter=900, minIter=300, check=200)
    else:
        kw = {}

    print("=== flow-field runs (Re=%.0f) ===" % Re_field)
    F_sm = run_lbm(R_smooth, Re_field, uLB_field, want_field=True, tag="field-smooth", **kw)
    F_gr = run_lbm(R_groove, Re_field, uLB_field, want_field=True, tag="field-groove", **kw)

    print("=== Cd(Re) sweep ===")
    sweep = {"smooth": {"gapRe": [], "dp": []}, "groove": {"gapRe": [], "dp": []}}
    for Re in Re_sweep:
        for name, Rf in (("smooth", R_smooth), ("groove", R_groove)):
            r = run_lbm(Rf, Re, uLB_sweep, tag=name, **kw)
            if r:
                sweep[name]["gapRe"].append(r["gapRe"]); sweep[name]["dp"].append(r["dp"])

    # fit Cd ~ 1/sqrt(dp) = C*(1 - beta/sqrt(Re))
    def model(Re, C, beta):
        return C * (1 - beta / np.sqrt(Re))

    betas = {}
    for name in ("smooth", "groove"):
        Re = np.array(sweep[name]["gapRe"]); dp = np.array(sweep[name]["dp"])
        y = 1.0 / np.sqrt(dp)
        if len(Re) >= 2:
            p0 = [y.max(), 1.0]
            try:
                popt, _ = curve_fit(model, Re, y, p0=p0, maxfev=20000)
            except Exception:
                popt = p0
            betas[name] = float(popt[1])
            print(f"  fit {name}: C={popt[0]:.4f}  beta={popt[1]:.4f}")
        else:
            betas[name] = float("nan")

    out = __file__.rsplit("/", 1)[0] + "/groove_study.npz"
    np.savez(out,
             sm_ux=F_sm["ux"], sm_uy=F_sm["uy"], sm_obst=F_sm["obstacle"], sm_fmask=F_sm["fmask"],
             gr_ux=F_gr["ux"], gr_uy=F_gr["uy"], gr_obst=F_gr["obstacle"], gr_fmask=F_gr["fmask"],
             sw_sm_Re=sweep["smooth"]["gapRe"], sw_sm_dp=sweep["smooth"]["dp"],
             sw_gr_Re=sweep["groove"]["gapRe"], sw_gr_dp=sweep["groove"]["dp"],
             beta_smooth=betas["smooth"], beta_groove=betas["groove"],
             s=s, nx=nx, ny=ny, cy=cy, x0=x0, bore_r=bore_r, Redge_c=Redge_c,
             Re_field=Re_field, uLB_field=uLB_field)
    print("saved:", out)
    print("BETAS:", betas)
