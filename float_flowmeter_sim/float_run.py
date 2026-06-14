"""
Runner: march one float geometry at one Re to (quasi-)steady state and return
time-averaged drag + fields.
"""
import time
import numpy as np
from float_axisym_solver import AxisymSolver
import float_geometry as G
from float_drag import drag_cv, drag_surface


def run_case(variant, Re, U_in=0.5, cpm=8, max_steps=8000, avg_steps=3000,
             tol=2e-6, verbose=True, want_field=False):
    """
    variant in G.VARIANTS.  Re based on float DIAMETER 2R and U_in.
    Returns dict with drag (time-averaged), Cd, fluctuation RMS%, etc.
    """
    dz = dr = 1.0 / cpm
    nz = int(round(G.LZ / dz))
    nr = int(round(G.R_OUT / dr))
    z_c = (np.arange(nz) + 0.5) * dz
    r_c = (np.arange(nr) + 0.5) * dr
    solid = G.solid_mask(z_c, r_c, variant)

    nu = U_in * (2 * G.R) / Re
    sol = AxisymSolver(Lz=G.LZ, R_out=G.R_OUT, dz=dz, dr=dr, nu=nu,
                       U_in=U_in, solid_cell=solid)
    # Conservative FIXED dt: account for gap acceleration. The annular gap shrinks
    # the flow area by ~ (R_out^2)/(R_out^2 - R^2); peak velocity can be several x
    # U_in. Use a velocity estimate of ~6*U_in for the advective CFL plus the
    # diffusive limit, and take a safety factor.
    area_ratio = G.R_OUT ** 2 / (G.R_OUT ** 2 - G.R ** 2)   # ~ inlet/gap area
    # peak gap velocity ~ U_in*area_ratio*(few); empirically ~3*U_in here.
    u_est = max(U_in * 4.0 * area_ratio, U_in)
    dt_adv = 0.35 * min(dz, dr) / u_est
    dt_diff = 0.20 * min(dz, dr) ** 2 / nu
    dt = min(dt_adv, dt_diff)
    # inlet ramp duration (steps) to avoid startup shock
    ramp = 400

    # CV planes: upstream of nose, downstream of tail (in w-face indices)
    i1 = int(round((G.Z_IN - 3.0) / dz))          # 3 mm before nose
    i2 = int(round((G.Z_IN + G.Ltot + 8.0) / dz))  # 8 mm after tail
    i1 = max(2, i1); i2 = min(nz - 2, i2)

    t0 = time.time()
    F_series = []
    Fs_series = []
    w_prev = sol.w.copy()
    warm = max_steps - avg_steps
    converged_step = None
    field_accum_w = np.zeros_like(sol.w)
    field_accum_v = np.zeros_like(sol.v)
    field_accum_p = np.zeros_like(sol.p)
    nfield = 0

    for n in range(max_steps):
        # smooth inlet ramp 0 -> U_in over `ramp` steps (cosine)
        if n < ramp:
            sol.U_now = U_in * 0.5 * (1 - np.cos(np.pi * n / ramp))
        else:
            sol.U_now = U_in
        sol.step(dt)
        if not np.isfinite(sol.w).all():
            print(f"  [{variant} Re={Re}] DIVERGED at step {n}")
            return None
        if n % 100 == 0 and n > 0:
            dw = np.abs(sol.w - w_prev).max() / (U_in * dt * 100)
            w_prev = sol.w.copy()
            if converged_step is None and dw < tol:
                converged_step = n
        # collect drag time series in the averaging window
        if n >= warm:
            Fcv, _ = drag_cv(sol, i1, i2)
            F_series.append(Fcv)
            if want_field:
                field_accum_w += sol.w
                field_accum_v += sol.v
                field_accum_p += sol.p
                nfield += 1

    F = np.array(F_series)
    Fmean = float(F.mean())
    Frms = float(F.std())
    rms_pct = 100.0 * Frms / abs(Fmean) if Fmean != 0 else float("nan")
    # drag-plateau convergence: relative drift between 1st and 2nd half of window
    half = len(F) // 2
    drift_pct = 100.0 * abs(F[half:].mean() - F[:half].mean()) / abs(Fmean) if Fmean != 0 else float("nan")
    # surface-method cross check on final field
    Fsurf, surf_bd = drag_surface(sol)

    Cd = Fmean / (0.5 * sol.rho * U_in ** 2 * np.pi * G.R ** 2)
    Cd_surf = Fsurf / (0.5 * sol.rho * U_in ** 2 * np.pi * G.R ** 2)

    runtime = time.time() - t0
    if verbose:
        print(f"  [{variant:8s} Re={Re:4.0f}] Cd_cv={Cd:6.3f} Cd_surf={Cd_surf:6.3f}"
              f"  rms={rms_pct:4.2f}%  drift={drift_pct:.2f}%  maxdiv={sol.max_divergence():.1e}"
              f"  ({runtime:.0f}s)")
    out = dict(variant=variant, Re=Re, U_in=U_in, Fmean=Fmean, Frms=Frms,
               rms_pct=rms_pct, drift_pct=drift_pct, Cd=Cd, Cd_surf=Cd_surf,
               Fsurf=Fsurf, converged_step=converged_step, runtime=runtime,
               maxdiv=float(sol.max_divergence()), nu=nu,
               surf_breakdown=surf_bd)
    if want_field:
        out["w"] = field_accum_w / max(nfield, 1)
        out["v"] = field_accum_v / max(nfield, 1)
        out["p"] = field_accum_p / max(nfield, 1)
        out["solid"] = sol.solid
        out["z_c"] = z_c
        out["r_c"] = r_c
        out["r_v"] = sol.r_v
        out["z_w"] = sol.z_w
    return out


if __name__ == "__main__":
    import sys
    # quick single-case test
    v = sys.argv[1] if len(sys.argv) > 1 else "blunt"
    Re = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    r = run_case(v, Re, max_steps=steps, avg_steps=min(2000, steps // 2))
    if r:
        print("  breakdown:", {k: round(val, 4) for k, val in r["surf_breakdown"].items()})
