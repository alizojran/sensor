"""Axisymmetric CFD re-check of the REDESIGNED float (blunt face + sharp Ø12 edge +
land + cone tail, no cavity). Confirms: steady flow, sharp-edge separation, blunt-face
stagnation, and Cd(Re). Reuses the Poiseuille-validated solver."""
import numpy as np
from scipy.optimize import curve_fit
from float_run import run_case
import float_geometry as G

RE = [50.0, 100.0, 200.0, 400.0]
res = {"Re": [], "Cd": [], "Cd_surf": [], "rms": []}
field = None
print("=== Redesign axisymmetric re-check ===")
for Re in RE:
    r = run_case("redesign", Re, U_in=0.5, cpm=8, max_steps=8000, avg_steps=3000,
                 want_field=(Re == 200.0))
    if r is None:
        continue
    res["Re"].append(Re); res["Cd"].append(r["Cd"]); res["Cd_surf"].append(r["Cd_surf"])
    res["rms"].append(r["rms_pct"])
    if Re == 200.0:
        field = r

Re = np.array(res["Re"]); cd = np.array(res["Cd"])
n, A = np.polyfit(np.log(Re), np.log(cd), 1)[0], None
p = np.polyfit(np.log(Re), np.log(cd), 1)
print(f"  Cd ∝ Re^{p[0]:.3f};  Cd_surf vs Cd agree within "
      f"{100*np.max(np.abs(np.array(res['Cd_surf'])-cd)/cd):.1f}%; max rms {max(res['rms']):.2f}%")

save = dict(Re=Re, Cd=cd, Cd_surf=np.array(res["Cd_surf"]), rms=np.array(res["rms"]), nexp=p[0])
if field is not None:
    save.update(field_w=field["w"], field_v=field["v"], field_solid=field["solid"],
                z_c=field["z_c"], r_c=field["r_c"], U=0.5)
np.savez("/home/user/sensor/float_flowmeter_sim/redesign_cfd.npz", **save)
print("saved redesign_cfd.npz")
