"""Plot the LBM flow field around the sharp-edged float (reads cfd_field.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

D = np.load(__file__.rsplit("/", 1)[0] + "/cfd_field.npz")
ux, uy = D["ux"], D["uy"]
speed, vort, press = D["speed"], D["vort"], D["press"]
obst, fmask = D["obstacle"], D["float_mask"]
cy, x0, s, uLB = float(D["cy"]), float(D["x0"]), float(D["s"]), float(D["uLB"])
nx, ny = ux.shape

# physical-ish axes in mm (relative to float nose / centreline)
x_mm = (np.arange(nx) - x0) / s
y_mm = (np.arange(ny) - cy) / s
Xn, Yn = np.meshgrid(x_mm, y_mm)            # (ny,nx)

speedT = speed.T
vortT = vort.T
uxT, uyT = ux.T.copy(), uy.T.copy()
solidT = obst.T
uxT[solidT] = 0.0
uyT[solidT] = 0.0

# quantitative read-outs
gap_peak = np.nanmax(speed)
dp = np.nanmean(press.T[:, 5]) - np.nanmean(press.T[:, -6])
print(f"peak gap speed = {gap_peak:.2f} x U_in ;  dp(in-out) ~ {dp:.4e} (lattice)")


def overlay_solid(ax):
    rgba = np.zeros((ny, nx, 4))
    rgba[fmask.T] = [0.35, 0.42, 0.55, 1.0]   # float -> steel blue-grey
    walls = solidT & ~fmask.T
    rgba[walls] = [0.2, 0.2, 0.2, 1.0]         # bore walls -> dark
    ax.imshow(rgba, origin="lower", extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
              aspect="equal", zorder=5)


def ann(ax, text, xy, xytext, color="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10,
                color=color, ha="center", zorder=7,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))


fig, axes = plt.subplots(3, 1, figsize=(14, 12.4))

# ---- (a) speed + streamlines ----
ax = axes[0]
pc = ax.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(speedT), cmap="turbo",
                   shading="auto", vmin=0, vmax=4.0)
ax.streamplot(x_mm, y_mm, uxT, uyT, color="0.12", density=1.4, linewidth=0.5,
              arrowsize=0.6, broken_streamlines=False)
overlay_solid(ax)
cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025, extend="max")
cb.set_label("速度 |u| / U_in", fontproperties=zh)
ann(ax, f"高速射流(可变环隙)\n峰值≈{gap_peak:.1f}×U_in", (6.5, 6.6), (16, 12.5), "w")
ann(ax, "锐边:分离点固定", (6.0, -6.0), (18, -13), "yellow")
ann(ax, "回流区 / 尾涡", (29, 3.0), (42, 11), "w")
ann(ax, "驻点", (0.2, 0.0), (-7, 9), "w")
ax.set_title("(a) 速度场与流线 — 流体经环隙加速,在锐边分离,尾部形成回流",
             fontproperties=zh, fontsize=12)
ax.set_ylabel("径向 r [mm]", fontproperties=zh)
ax.set_xlim(x_mm[0], x_mm[-1]); ax.set_ylim(y_mm[0], y_mm[-1])

# ---- (b) vorticity ----
ax = axes[1]
vmax = np.nanpercentile(np.abs(vortT), 99)
pc = ax.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(vortT), cmap="RdBu_r",
                   shading="auto", vmin=-vmax, vmax=vmax)
overlay_solid(ax)
cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025)
cb.set_label("涡量 ω", fontproperties=zh)
ann(ax, "锐边剪切层\n(分离线被钉死在边缘)", (6.0, 6.0), (20, 12), "k")
ax.set_title("(b) 涡量场 — 剪切层从锐边规则脱落:分离点不随 Re/粘度漂移(viscosity-immune)",
             fontproperties=zh, fontsize=12)
ax.set_ylabel("径向 r [mm]", fontproperties=zh)
ax.set_xlim(x_mm[0], x_mm[-1]); ax.set_ylim(y_mm[0], y_mm[-1])

# ---- (c) pressure ----
ax = axes[2]
pT = press.T
pmax = np.nanpercentile(np.abs(pT - np.nanmedian(pT)), 99)
pmed = np.nanmedian(pT)
pc = ax.pcolormesh(x_mm, y_mm, np.ma.masked_invalid(pT), cmap="coolwarm",
                   shading="auto", vmin=pmed - pmax, vmax=pmed + pmax)
overlay_solid(ax)
cb = fig.colorbar(pc, ax=ax, pad=0.01, fraction=0.025)
cb.set_label("压力 p (规一化)", fontproperties=zh)
ann(ax, "迎流面高压", (-1.5, 0.0), (-8, 10), "k")
ann(ax, "环隙/尾部低压", (10, 5.0), (24, 12), "k")
ax.set_title("(c) 压力场 — 迎流面高压、环隙低压;前后压差 ΔP 正是顶起浮子的力 (ΔP·A_f = 弹簧力)",
             fontproperties=zh, fontsize=12)
ax.set_xlabel("轴向 x [mm]  (流动 →)", fontproperties=zh)
ax.set_ylabel("径向 r [mm]", fontproperties=zh)
ax.set_xlim(x_mm[0], x_mm[-1]); ax.set_ylim(y_mm[0], y_mm[-1])

fig.suptitle("锐边 viscosity-immune 浮子 CFD 流场(LBM, Re≈150,子午面 2D 近似)",
             fontproperties=zh, fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = __file__.rsplit("/", 1)[0] + "/float_cfd_flowfield.png"
fig.savefig(out, dpi=140)
print("saved:", out)
