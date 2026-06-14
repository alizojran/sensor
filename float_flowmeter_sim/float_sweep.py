"""
Main sweep: for each geometry {cup6, cup2, blunt, rounded} sweep Re in {50,100,200,400}.
Compute time-averaged drag, Cd, fit Cd(Re)=Cd_inf*(1+beta/sqrt(Re)).
Save axisym_results.npz (Poiseuille validation + sweep + fits + cup6 field).
"""
import time
import numpy as np
from scipy.optimize import curve_fit

import float_geometry as G
from float_run import run_case
from test_poiseuille import run_poiseuille

RE_LIST = [50.0, 100.0, 200.0, 400.0]
RE_FIELD = 200.0     # which Re to save the cup6 field at
U_IN = 0.5
CPM = 8
MAX_STEPS = 8000
AVG_STEPS = 3000


def cd_model(Re, Cd_inf, beta):
    return Cd_inf * (1 + beta / np.sqrt(Re))


def main():
    t_all = time.time()

    # ---------------- Validation gate ----------------
    print("=== GATE 1: Poiseuille validation (Re=100) ===")
    pois = run_poiseuille(Re=100.0, verbose=False, max_steps=60000)
    print(f"  max err = {pois['err']*100:.2f} %   centreline/mean = {pois['ratio']:.3f}")
    gate_ok = (pois['err'] < 0.05) and (abs(pois['ratio'] - 2.0) < 0.1)
    print("  GATE:", "PASS" if gate_ok else "FAIL")
    if not gate_ok:
        print("  *** Validation failed; aborting float sweep. ***")
        return

    # ---------------- Float sweep ----------------
    print("\n=== Float drag sweep (axisymmetric) ===")
    results = {v: {"Re": [], "Cd": [], "Cd_surf": [], "rms": [], "drift": [],
                   "Fmean": []} for v in G.VARIANTS}
    field = None
    for v in G.VARIANTS:
        for Re in RE_LIST:
            want = (v == "cup6" and Re == RE_FIELD)
            r = run_case(v, Re, U_in=U_IN, cpm=CPM, max_steps=MAX_STEPS,
                         avg_steps=AVG_STEPS, want_field=want)
            if r is None:
                print(f"  [{v} Re={Re}] FAILED/diverged - skipped")
                continue
            results[v]["Re"].append(Re)
            results[v]["Cd"].append(r["Cd"])
            results[v]["Cd_surf"].append(r["Cd_surf"])
            results[v]["rms"].append(r["rms_pct"])
            results[v]["drift"].append(r["drift_pct"])
            results[v]["Fmean"].append(r["Fmean"])
            if want:
                field = r

    # ---------------- Fit Cd(Re) ----------------
    print("\n=== Cd(Re) = Cd_inf*(1 + beta/sqrt(Re)) fits ===")
    fits = {}
    for v in G.VARIANTS:
        Re = np.array(results[v]["Re"]); cd = np.array(results[v]["Cd"])
        if len(Re) >= 2:
            try:
                popt, _ = curve_fit(cd_model, Re, cd, p0=[cd.min(), 1.0], maxfev=40000)
                Cd_inf, beta = float(popt[0]), float(popt[1])
            except Exception as e:
                print(f"  fit {v} failed: {e}")
                Cd_inf, beta = float(cd.mean()), float("nan")
        else:
            Cd_inf, beta = (float(cd[0]) if len(cd) else float("nan")), float("nan")
        fits[v] = (Cd_inf, beta)
        rmsmax = max(results[v]["rms"]) if results[v]["rms"] else float("nan")
        print(f"  {v:8s}: Cd_inf={Cd_inf:6.3f}  beta={beta:6.3f}  (max drag rms {rmsmax:.2f}%)")

    # ---------------- Save ----------------
    save = dict(
        # Poiseuille
        pois_r=pois["r_c"], pois_w=pois["w_out"], pois_wan=pois["w_an"],
        pois_err=pois["err"], pois_ratio=pois["ratio"], pois_U=pois["U"],
        pois_R=pois["R_pipe"],
        # sweep meta
        RE_LIST=np.array(RE_LIST), U_IN=U_IN, CPM=CPM, RE_FIELD=RE_FIELD,
        VARIANTS=np.array(G.VARIANTS),
    )
    for v in G.VARIANTS:
        save[f"{v}_Re"] = np.array(results[v]["Re"])
        save[f"{v}_Cd"] = np.array(results[v]["Cd"])
        save[f"{v}_Cd_surf"] = np.array(results[v]["Cd_surf"])
        save[f"{v}_rms"] = np.array(results[v]["rms"])
        save[f"{v}_drift"] = np.array(results[v]["drift"])
        save[f"{v}_Fmean"] = np.array(results[v]["Fmean"])
        save[f"{v}_Cdinf"] = fits[v][0]
        save[f"{v}_beta"] = fits[v][1]
    if field is not None:
        save["field_w"] = field["w"]
        save["field_v"] = field["v"]
        save["field_p"] = field["p"]
        save["field_solid"] = field["solid"]
        save["field_z_c"] = field["z_c"]
        save["field_r_c"] = field["r_c"]
        save["field_z_w"] = field["z_w"]
        save["field_r_v"] = field["r_v"]
        save["field_variant"] = "cup6"
        save["field_Re"] = RE_FIELD

    out = "/home/user/sensor/float_flowmeter_sim/axisym_results.npz"
    np.savez(out, **save)
    print(f"\nsaved {out}")
    print(f"total wall time {(time.time()-t_all)/60:.1f} min")
    return save


if __name__ == "__main__":
    main()
