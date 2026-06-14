"""
VALIDATION GATE 1: Hagen-Poiseuille pipe flow.
Straight pipe radius R_pipe, no float, uniform inlet U. March to steady state.
Developed profile must match  w(r) = 2 U (1 - (r/R)^2).
"""
import time
import numpy as np
from float_axisym_solver import AxisymSolver


def run_poiseuille(R_pipe=1.0, Lz=20.0, dr=0.05, dz=0.2, Re=50.0, U=1.0,
                   max_steps=40000, tol=1e-6, verbose=True):
    nu = U * (2 * R_pipe) / Re      # Re on diameter
    sol = AxisymSolver(Lz=Lz, R_out=R_pipe, dz=dz, dr=dr, nu=nu, U_in=U)
    t0 = time.time()
    dt = sol.cfl_dt(safety=0.4)
    if verbose:
        print(f"  grid nz={sol.nz} nr={sol.nr}  nu={nu:.4g}  dt={dt:.4g}")
    w_prev = sol.w.copy()
    for n in range(max_steps):
        sol.step(dt)
        if n % 200 == 0 and n > 0:
            dw = np.abs(sol.w - w_prev).max() / (U * dt * 200)  # approx d/dt
            w_prev = sol.w.copy()
            if verbose and n % 2000 == 0:
                print(f"    step {n:6d}  dwdt~{dw:.3e}  maxdiv={sol.max_divergence():.2e}")
            if dw < tol:
                if verbose:
                    print(f"    converged at step {n} (dwdt={dw:.2e})")
                break
    # outlet plane (last interior w face)
    r_c, w_out = sol.w_centre_profile(sol.nz - 1)
    w_an = 2 * U * (1 - (r_c / R_pipe) ** 2)
    w_an = np.clip(w_an, 0, None)
    err = np.abs(w_out - w_an).max() / (2 * U)        # normalized by U_max(analytic)=2U
    # centreline/mean ratio: centreline = w at r=dr/2 (axis), mean = flux/area
    # flux = 2*pi * integral w r dr ; area-mean over pipe
    flux = 2 * np.pi * np.sum(w_out * r_c * dr)
    area = np.pi * R_pipe ** 2
    w_mean = flux / area
    w_centre = w_out[0]
    ratio = w_centre / w_mean
    dt_run = time.time() - t0
    return dict(r_c=r_c, w_out=w_out, w_an=w_an, err=err, ratio=ratio,
                w_mean=w_mean, U=U, R_pipe=R_pipe, nz=sol.nz, nr=sol.nr,
                steps=n, runtime=dt_run, sol=sol)


if __name__ == "__main__":
    print("=== Poiseuille validation (Re=50) ===")
    res = run_poiseuille()
    print(f"  max|w_sim - w_analytic|/U_max = {res['err']*100:.2f} %")
    print(f"  centreline/mean ratio = {res['ratio']:.3f}  (analytic 2.000)")
    print(f"  mean velocity = {res['w_mean']:.4f}  (should be ~U=1.0)")
    print(f"  runtime {res['runtime']:.1f}s, steps {res['steps']}")
    ok = (res['err'] < 0.05) and (abs(res['ratio'] - 2.0) < 0.1)
    print("  PASS" if ok else "  *** FAIL ***")
