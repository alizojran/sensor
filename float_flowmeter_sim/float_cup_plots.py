"""Plots for the head-shape A/B/C study (reads cup_study.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/cup_study.npz")

s, x0, cy = float(D["s"]), float(D["x0"]), float(D["cy"])
nx, ny = int(D["nx"]), int(D["ny"])
uLB = float(D["uLB_field"])
x_mm = (np.arange(nx) - x0) / s
y_mm = (np.arange(ny) - cy) / s
NAMES = {"cup": "钝头+前向凹坑(杯)", "blunt": "钝头(无坑)", "rounded": "圆头"}
ORDER = ["cup", "blunt", "rounded"]
COL = {"cup": "tab:green", "blunt": "tab:orange", "rounded": "tab:red"}

# ============================ Figure 1: flow fields ==================================
fig1, axes = plt.subplots(3, 1, figsize=(12, 9.6))
for ax, g in zip(axes, ORDER):
    ux, uy, ob = D[f"{g}_ux"], D[f"{g}_uy"], D[f"{g}_obst"]
    sp = np.sqrt(ux ** 2 + uy ** 2).T / uLB
    sp[ob.T] = np.nan
    ux2, uy2 = ux.T.copy(), uy.T.copy(); ux2[ob.T] = 0; uy2[ob.T] = 0
    pc = ax.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(sp), cmap="turbo",
                       shading="auto", vmin=0, vmax=4.0)
    ax.streamplot(x_mm, y_mm, ux2, uy2, color="0.12", density=1.5, linewidth=0.5,
                  arrowsize=0.6, broken_streamlines=False)
    rgba = np.zeros((ny, nx, 4)); rgba[ob.T] = [0.32, 0.39, 0.5, 1]
    ax.imshow(rgba, origin="lower", extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
              aspect="equal", zorder=5)
    fig1.colorbar(pc, ax=ax, pad=0.01, fraction=0.02).set_label("|u|/U_in", fontproperties=zh)
    ax.set_title(NAMES[g], fontproperties=zh, fontsize=12)
    ax.set_ylabel("r [mm]", fontproperties=zh)
    ax.set_xlim(-8, 30); ax.set_ylim(y_mm[0], y_mm[-1])
axes[0].annotate("坑内驻涡", (1.5, 0), (-6, 9), fontproperties=zh, color="w", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="w"))
axes[-1].set_xlabel("x [mm] (流动 →)", fontproperties=zh)
fig1.suptitle("CFD 流场对比(LBM, Re≈%.0f):杯 / 钝 / 圆 头" % float(D["Re_field"]),
              fontproperties=zh, fontsize=14, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(HERE + "/float_cup_cfd.png", dpi=140)
print("saved float_cup_cfd.png")

# ============================ Figure 2: Cd(Re) + accuracy ============================
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
ReA = np.logspace(1.6, 4.8, 300)
Dd = 0.012


def cd_fit(Re, g):
    return float(D[f"Cinf_{g}"]) * (1 + float(D[f"beta_{g}"]) / np.sqrt(Re))


# Left: Cd_d(Re) normalized to its own high-Re value -> shows the droop/slope
for g in ORDER:
    Re_c = np.asarray(D[f"sw_{g}_Re"], float); cd = np.asarray(D[f"sw_{g}_Cd"], float)
    norm = cd_fit(ReA[-1], g)
    axL.plot(Re_c, cd / norm, "o", color=COL[g], ms=9)
    axL.plot(ReA, cd_fit(ReA, g) / norm, "-", color=COL[g], lw=2,
             label=f"{NAMES[g]}  (β={float(D[f'beta_{g}']):.2f})")
axL.set_xscale("log")
axL.set_xlabel("浮子雷诺数 Re", fontproperties=zh)
axL.set_ylabel("阻力系数 Cd / Cd∞ (归一化)", fontproperties=zh)
axL.set_title("(左) CFD 测得阻力系数随 Re 的变化(越平→越粘度免疫)", fontproperties=zh, fontsize=11.5)
axL.grid(True, which="both", alpha=0.3); axL.legend(prop=zh, fontsize=9)

# Right: reading error vs flow (drag-balanced float: err = sqrt(Cd(Re)/Cd_cal) - 1)
def Re_of_Q(Q_lpm, rho, mu):
    Q = Q_lpm / 6e4
    return 4 * rho * Q / (np.pi * mu * Dd)


fluids = {"水 20°C": ("-", 998., 1e-3), "冷却液/乙二醇": (":", 1040., 5e-3)}
Q = np.linspace(1, 50, 200)
for fl, (ls, rho, mu) in fluids.items():
    Re_cal = Re_of_Q(50.0, 998., 1e-3)              # span-calibrated on water FS
    for g in ORDER:
        Re = Re_of_Q(Q, rho, mu)
        err = (np.sqrt(cd_fit(Re, g) / cd_fit(Re_cal, g)) - 1) * 100
        axR.plot(Q, err, ls, color=COL[g], lw=2, label=f"{NAMES[g]} / {fl}")
axR.fill_between(Q, -(4 + 50 / Q), (4 + 50 / Q), color="gray", alpha=0.12)
axR.axhline(0, color="k", lw=0.8)
axR.set_xlabel("流量 [l/min]", fontproperties=zh)
axR.set_ylabel("读数误差 [% of reading]", fontproperties=zh)
axR.set_title("(右) 精度对比(阻力平衡浮子,单点 span 标定)", fontproperties=zh, fontsize=11.5)
axR.grid(True, alpha=0.3); axR.legend(prop=zh, fontsize=7.5, ncol=1)

fig2.suptitle("头部形状对比:阻力系数 Re 稳定性 → 精度(水/乙二醇)",
              fontproperties=zh, fontsize=13.5, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(HERE + "/float_cup_accuracy.png", dpi=140)
print("saved float_cup_accuracy.png")
for g in ORDER:
    print(f"{g}: Cinf={float(D[f'Cinf_{g}']):.3f} beta={float(D[f'beta_{g}']):.3f} "
          f"Cd_sweep={np.asarray(D[f'sw_{g}_Cd'])}")
