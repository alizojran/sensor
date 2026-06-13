"""Plots for the smooth-vs-grooved A/B study (reads groove_study.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/groove_study.npz")

s, x0, cy = float(D["s"]), float(D["x0"]), float(D["cy"])
nx, ny = int(D["nx"]), int(D["ny"])
uLB = float(D["uLB_field"])
bmin, bgr = float(D["beta_smooth"]), float(D["beta_groove"])
x_mm = (np.arange(nx) - x0) / s
y_mm = (np.arange(ny) - cy) / s

# ============================ Figure 1: flow-field A/B ===============================
def field(ux, uy, obst, fmask):
    sp = np.sqrt(ux ** 2 + uy ** 2) / uLB
    sp = sp.T.copy(); ux2, uy2 = ux.T.copy(), uy.T.copy()
    solid = obst.T
    sp[solid] = np.nan; ux2[solid] = 0; uy2[solid] = 0
    return sp, ux2, uy2, fmask.T, solid


SM = field(D["sm_ux"], D["sm_uy"], D["sm_obst"], D["sm_fmask"])
GR = field(D["gr_ux"], D["gr_uy"], D["gr_obst"], D["gr_fmask"])


def solid_overlay(ax, fmaskT, solidT):
    rgba = np.zeros((ny, nx, 4))
    rgba[fmaskT] = [0.35, 0.42, 0.55, 1.0]
    rgba[solidT & ~fmaskT] = [0.2, 0.2, 0.2, 1.0]
    ax.imshow(rgba, origin="lower", extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
              aspect="equal", zorder=5)


fig1, axes = plt.subplots(2, 2, figsize=(15, 8),
                          gridspec_kw=dict(width_ratios=[1.7, 1.0]))
rows = [("光面锐边锥", SM), ("前锥带锐边环槽", GR)]
for r, (name, (sp, ux2, uy2, fmT, soT)) in enumerate(rows):
    # full view
    ax = axes[r, 0]
    pc = ax.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(sp), cmap="turbo",
                       shading="auto", vmin=0, vmax=4.0)
    ax.streamplot(x_mm, y_mm, ux2, uy2, color="0.12", density=1.3, linewidth=0.5,
                  arrowsize=0.6, broken_streamlines=False)
    solid_overlay(ax, fmT, soT)
    ax.set_title(f"{name} — 速度场与流线", fontproperties=zh, fontsize=12)
    ax.set_ylabel("r [mm]", fontproperties=zh)
    ax.set_xlim(-8, 32); ax.set_ylim(y_mm[0], y_mm[-1])
    fig1.colorbar(pc, ax=ax, pad=0.01, fraction=0.025).set_label("|u|/U_in", fontproperties=zh)
    # zoom on front cone / groove
    az = axes[r, 1]
    pc = az.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(sp), cmap="turbo",
                       shading="auto", vmin=0, vmax=4.0)
    az.streamplot(x_mm, y_mm, ux2, uy2, color="k", density=3.0, linewidth=0.55,
                  arrowsize=0.7, broken_streamlines=False)
    solid_overlay(az, fmT, soT)
    az.set_title(f"{name} — 前锥放大", fontproperties=zh, fontsize=12)
    az.set_xlim(-1, 8); az.set_ylim(1, 9)
    if r == 1:
        az.annotate("槽内驻涡\n(固定分离线)", (3.1, 3.0), (4.5, 6.5),
                    fontproperties=zh, fontsize=10, color="w",
                    arrowprops=dict(arrowstyle="->", color="w", lw=1.4))
    else:
        az.annotate("锥面边界层\n随 Re 变化", (3.0, 4.0), (4.3, 7.0),
                    fontproperties=zh, fontsize=10, color="w",
                    arrowprops=dict(arrowstyle="->", color="w", lw=1.4))
for ax in axes[1, :]:
    ax.set_xlabel("x [mm] (流动 →)", fontproperties=zh)
fig1.suptitle("CFD A/B:光面锥 vs 前锥锐边环槽(LBM, Re≈%.0f,子午面 2D)" % float(D["Re_field"]),
              fontproperties=zh, fontsize=14, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(HERE + "/float_groove_cfd.png", dpi=140)
print("saved float_groove_cfd.png")

# ============================ Figure 2: Cd(Re) + accuracy ============================
# --- accuracy model (same framework as float_flowmeter_sim.py) ---
M = 6e4
Df, H, g0, gap_fs = 12e-3, 20e-3, 0.2e-3, 3.2e-3
a_t = (gap_fs - g0) / H
A_f = np.pi / 4 * Df ** 2
dP0, dP_fs = 5e3, 31e3
F0, kspr = A_f * dP0, A_f * (dP_fs - dP0) / H
Cdinf = 0.72


def gp(h): return g0 + a_t * h
def Aa(h):
    Db = Df + 2 * gp(h); return np.pi / 4 * (Db ** 2 - Df ** 2)
def dPh(h): return (F0 + kspr * h) / A_f
def Cdf(Re, beta):
    Re = np.maximum(Re, 20.0); return np.clip(Cdinf * (1 - beta / np.sqrt(Re)), 0.3, Cdinf)


def simulate(rho, mu, beta):
    h = np.linspace(1e-4, H, 600)
    A = Aa(h); K = A * np.sqrt(2 * dPh(h) / rho); c = 2 * rho * gp(h) / (mu * A)
    Q = 0.7 * K
    for _ in range(80):
        Q = Cdf(c * Q, beta) * K
    Re = c * Q
    err = (Cdf(Re[-1], beta) / Cdf(Re, beta) - 1) * 100
    return Q * M, err


fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# Left: Cd/Cdinf vs gap-Re from CFD (points) + fit
ReA = np.logspace(1.5, 4.6, 300)
for name, beta, Re_c, dp_c, col in [
        ("光面锥", bmin, D["sw_sm_Re"], D["sw_sm_dp"], "tab:red"),
        ("带环槽", bgr, D["sw_gr_Re"], D["sw_gr_dp"], "tab:green")]:
    Re_c = np.asarray(Re_c); dp_c = np.asarray(dp_c)
    y = 1 / np.sqrt(dp_c); y = y / y.max()                       # CFD points (normalized)
    axL.plot(Re_c, y, "o", color=col, ms=8)
    axL.plot(ReA, (1 - beta / np.sqrt(ReA)) / (1 - beta / np.sqrt(ReA[-1])),
             "-", color=col, lw=2, label=f"{name}  (β={beta:.2f})")
axL.set_xscale("log")
axL.set_xlabel("环隙雷诺数 Re", fontproperties=zh)
axL.set_ylabel("Cd / Cd∞ (归一化)", fontproperties=zh)
axL.set_title("(左) CFD 测得的 Cd 下垂:槽越平→粘度免疫越好", fontproperties=zh, fontsize=12)
axL.grid(True, which="both", alpha=0.3); axL.legend(prop=zh)

# Right: reading error vs flow, water + coolant, both geometries
sty = {"水 20°C": ("-", dict(rho=998., mu=1e-3)),
       "冷却液/乙二醇": ("--", dict(rho=1040., mu=5e-3))}
col = {"光面锥": "tab:red", "带环槽": "tab:green"}
for fl, (ls, p) in sty.items():
    for name, beta in [("光面锥", bmin), ("带环槽", bgr)]:
        flow, err = simulate(p["rho"], p["mu"], beta)
        axR.plot(flow, err, ls, color=col[name], lw=2, label=f"{name} / {fl}")
qb = np.linspace(1, 50, 200)
axR.fill_between(qb, -(4 + 50 / qb), (4 + 50 / qb), color="gray", alpha=0.12)
axR.axhline(0, color="k", lw=0.8)
axR.set_xlabel("流量 [l/min]", fontproperties=zh)
axR.set_ylabel("读数误差 [% of reading]", fontproperties=zh)
axR.set_title("(右) 精度对比(单点 span 标定)", fontproperties=zh, fontsize=12)
axR.set_ylim(-2, 10); axR.grid(True, alpha=0.3); axR.legend(prop=zh, fontsize=9)

fig2.suptitle("环形锐边槽的收益:Cd 更平(粘度免疫)→ 低流量/高粘度精度更好",
              fontproperties=zh, fontsize=14, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(HERE + "/float_groove_accuracy.png", dpi=140)
print("saved float_groove_accuracy.png")
print(f"beta_smooth={bmin:.3f}  beta_groove={bgr:.3f}")
