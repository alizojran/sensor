"""
Variable-area (spring-loaded piston / float) flow-meter accuracy simulation.

Models an ifm-SBZ224-style "mechatronic" flow meter: a spring-loaded float (piston)
sits in a tapered bore. Flow lifts the float against a spring; a magnet in the float
reports its position, which is converted to flow rate.

The simulation answers: how does the FLOAT SHAPE affect measurement accuracy across
the 1...50 l/min range and for different fluids (water vs. coolant/glycol)?

Physics
-------
Force balance on the float (vertical, buoyancy-corrected gravity folded into the spring
preload):
        dP * A_f = F0 + k*h                       (1)
    => dP(h)     = (F0 + k*h) / A_f

Flow through the annular metering gap (orifice / Bernoulli):
        Q = Cd(Re) * A_a(h) * sqrt(2*dP(h)/rho)   (2)

with
    A_a(h)  annular area between float edge (dia. Df) and the tapered bore,
    gap(h)  = g0 + a*h           (radial gap grows linearly as the float rises),
    Re      = rho * v * Dh / mu  (gap Reynolds number, v = Q/A_a, Dh = 2*gap).

The discharge coefficient Cd depends on Reynolds number, and HOW MUCH it depends on Re
is set by the float's shape:
        Cd(Re) = Cd_inf * (1 - beta / sqrt(Re))
    * sharp-edged "viscosity-immune" float -> small beta -> Cd nearly constant
    * rounded "ball" float                 -> large beta -> Cd droops at low Re

Calibration model
-----------------
Real SB meters expose a single "calibration coefficient" (校准系数). We model that as a
single-point SPAN calibration: the electronics assume a CONSTANT Cd, chosen so the meter
reads exactly right at full scale for the calibration fluid. Any change of Cd away from
that value (because Re changes along the range, or the fluid viscosity changes) shows up
directly as reading error. That residual is exactly what the float shape controls.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# 1. FLOAT / METER GEOMETRY  (the "design")
# --------------------------------------------------------------------------------------
Df      = 12.0e-3          # float metering-edge diameter            [m]
H       = 20.0e-3          # float travel (stroke), h in [0, H]       [m]
g0      = 0.20e-3          # radial gap at h=0 (zero-flow seat lift)  [m]
gap_fs  = 3.20e-3          # radial gap at full stroke                [m]
a       = (gap_fs - g0) / H                                          # taper slope [-]

A_f     = np.pi / 4.0 * Df**2          # float reference (pressure) area [m^2]

# Spring (linear) expressed via the pressure drop it sustains at the two stroke ends.
dP0     = 5.0e3           # dP at h=0     [Pa]   (preload)
dP_fs   = 31.0e3          # dP at h=H     [Pa]
F0      = A_f * dP0                      # spring preload force  [N]
k       = A_f * (dP_fs - dP0) / H        # spring rate           [N/m]


def gap(h):                              # radial gap [m]
    return g0 + a * h


def A_annulus(h):                        # exact annular flow area [m^2]
    Dbore = Df + 2.0 * gap(h)
    return np.pi / 4.0 * (Dbore**2 - Df**2)


def dP(h):                               # pressure drop from spring force balance [Pa]
    return (F0 + k * h) / A_f


# --------------------------------------------------------------------------------------
# 2. FLOAT-SHAPE DISCHARGE-COEFFICIENT MODELS  Cd(Re)
# --------------------------------------------------------------------------------------
Cd_inf = 0.72
SHAPES = {
    "Sharp-edged (viscosity-immune)": 0.20,   # beta  -> flat Cd
    "Ball (viscosity-sensitive)":     1.00,   # beta  -> Cd droops at low Re
}


def Cd(Re, beta):
    Re = np.maximum(Re, 20.0)            # keep model in its valid laminar-ish window
    return np.clip(Cd_inf * (1.0 - beta / np.sqrt(Re)), 0.30, Cd_inf)


# --------------------------------------------------------------------------------------
# 3. FLUIDS  (rho [kg/m^3], mu [Pa.s])
# --------------------------------------------------------------------------------------
FLUIDS = {
    "Water 20C":        dict(rho=998.0,  mu=1.00e-3),
    "Coolant/glycol":   dict(rho=1040.0, mu=5.00e-3),   # ~30% glycol mix, warm
    "Glycol cold(0C)":  dict(rho=1080.0, mu=1.50e-2),   # ~50% glycol, cold-start (table only)
}


# --------------------------------------------------------------------------------------
# 4. SOLVE true flow Q(h) for a fluid+shape (Cd <-> Q are coupled via Re; fixed point)
# --------------------------------------------------------------------------------------
def true_flow(h, rho, mu, beta):
    Aa  = A_annulus(h)
    K   = Aa * np.sqrt(2.0 * dP(h) / rho)        # Q = Cd * K
    c   = 2.0 * rho * gap(h) / (mu * Aa)         # Re = c * Q
    Q   = 0.7 * K                                 # initial guess
    for _ in range(60):
        Q = Cd(c * Q, beta) * K
    Re  = c * Q
    return Q, Re


M3S_TO_LPM = 60.0 * 1000.0                        # m^3/s -> l/min


def simulate(rho, mu, beta):
    """Return (flow_lpm, Re, Cd_true, error_pct) sampled along the stroke."""
    h    = np.linspace(1e-4, H, 600)              # avoid exactly h=0
    Q, Re = true_flow(h, rho, mu, beta)
    Cdt  = Cd(Re, beta)
    # single-point SPAN calibration: assume constant Cd = Cd at full stroke for THIS fluid
    Cd_cal = Cdt[-1]
    # meter reports Q_ind(h) = Cd_cal * K(h) (density known); error vs true at same h
    err  = (Cd_cal / Cdt - 1.0) * 100.0           # % of reading
    return Q * M3S_TO_LPM, Re, Cdt, err


# --------------------------------------------------------------------------------------
# 5. RUN + REPORT
# --------------------------------------------------------------------------------------
def lin_interp(xq, x, y):
    order = np.argsort(x)
    return np.interp(xq, x[order], y[order])


NOMINAL = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50], dtype=float)

print("=" * 78)
print("VARIABLE-AREA FLOW METER  (spring-loaded float)  -- design summary")
print("=" * 78)
print(f"  Float edge diameter Df     : {Df*1e3:6.2f} mm")
print(f"  Stroke (travel) H          : {H*1e3:6.2f} mm")
print(f"  Radial gap  g0 -> gap_fs   : {g0*1e3:5.2f} -> {gap_fs*1e3:.2f} mm  (taper a={a:.3f})")
print(f"  Spring: preload F0={F0:.3f} N , rate k={k:.1f} N/m")
print(f"  Pressure drop dP: {dP0/1e3:.1f} -> {dP_fs/1e3:.1f} kPa over the stroke")
# verify the geometry actually spans ~1..50 l/min on water
qg, _ = true_flow(np.array([1e-4, H]), 998.0, 1.0e-3, 0.20)
print(f"  -> water flow span         : {qg[0]*M3S_TO_LPM:5.2f} ... {qg[1]*M3S_TO_LPM:5.1f} l/min")

for fluid, p in FLUIDS.items():
    print("\n" + "-" * 78)
    print(f"FLUID: {fluid}   (rho={p['rho']} kg/m3, mu={p['mu']*1e3:.1f} mPa.s)")
    print("-" * 78)
    header = f"{'Q [l/min]':>10} {'Re':>8} |"
    for s in SHAPES:
        header += f" {s.split()[0]+' Cd':>16} {'err%':>7} |"
    print(header)
    data = {s: simulate(p["rho"], p["mu"], SHAPES[s]) for s in SHAPES}
    for q in NOMINAL:
        row = f"{q:10.1f}"
        # Re from the first shape (nearly shape-independent) for display
        flow0, Re0, _, _ = data[list(SHAPES)[0]]
        row += f" {lin_interp(q, flow0, Re0):8.0f} |"
        for s in SHAPES:
            flow, Re, Cdt, err = data[s]
            row += f" {lin_interp(q, flow, Cdt):16.4f} {lin_interp(q, flow, err):7.2f} |"
        print(row)

# --------------------------------------------------------------------------------------
# 6. PLOTS
# --------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) Cd(Re) -- the intrinsic float-shape property
Re_axis = np.logspace(1.5, 4.8, 400)
for s, beta in SHAPES.items():
    ax1.semilogx(Re_axis, Cd(Re_axis, beta), lw=2, label=s)
ax1.set_xlabel("Reynolds number  Re  (gap)")
ax1.set_ylabel("Discharge coefficient  Cd")
ax1.set_title("(a) Float-shape signature: Cd vs Re")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=9)

# (b) accuracy vs flow, water + coolant, both shapes
styles = {"Water 20C": "-", "Coolant/glycol": "--"}
colors = {"Sharp-edged (viscosity-immune)": "tab:green",
          "Ball (viscosity-sensitive)":     "tab:red"}
for fluid in ("Water 20C", "Coolant/glycol"):
    p = FLUIDS[fluid]
    for s, beta in SHAPES.items():
        flow, Re, Cdt, err = simulate(p["rho"], p["mu"], beta)
        ax2.plot(flow, err, styles[fluid], color=colors[s], lw=2,
                 label=f"{s.split()[0]} / {fluid}")
# datasheet accuracy band: +/-(4% of reading + 1% of FS=50) -> in % of reading
qb = np.linspace(1, 50, 200)
band = 4.0 + (0.01 * 50.0) / qb * 100.0
ax2.fill_between(qb, -band, band, color="gray", alpha=0.12,
                 label="datasheet band  ±(4%MW+1%MEW)")
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xlabel("True flow rate  [l/min]")
ax2.set_ylabel("Reading error  [% of reading]")
ax2.set_title("(b) Accuracy across the range (single-point span cal.)")
ax2.set_ylim(-3, 12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, ncol=1)

fig.suptitle("Variable-area float flow meter — how float shape drives accuracy",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = __file__.rsplit("/", 1)[0] + "/float_flowmeter_accuracy.png"
fig.savefig(out, dpi=130)
print(f"\nSaved plot -> {out}")
