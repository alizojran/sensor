"""Plot axisymmetric results (reads axisym_results.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/axisym_results.npz", allow_pickle=True)

NAME = {"cup6": "深腔 Ø8×6 (d/Dp=0.75)", "cup2": "浅腔 Ø8×2 (d/Dp=0.25)",
        "blunt": "钝头(无坑)", "rounded": "圆头"}
COL = {"cup6": "tab:green", "cup2": "tab:red", "blunt": "tab:orange", "rounded": "tab:blue"}
ORDER = ["cup6", "cup2", "blunt", "rounded"]

fig = plt.figure(figsize=(16.5, 5.0))
axA = fig.add_subplot(1, 3, 1)
axB = fig.add_subplot(1, 3, 2)
axC = fig.add_subplot(1, 3, 3)

# ---- (A) Poiseuille validation ----
r = D["pois_r"]; w = D["pois_w"]; wan = D["pois_wan"]
axA.plot(wan, r, "-", color="k", lw=2, label="解析抛物线")
axA.plot(w, r, "o", color="tab:red", ms=4, label="轴对称求解器")
axA.set_xlabel("轴向速度 w", fontproperties=zh); axA.set_ylabel("半径 r", fontproperties=zh)
axA.set_title(f"(A) 验证:管流 Poiseuille\n最大误差 {float(D['pois_err'])*100:.2f}%,中心/平均={float(D['pois_ratio']):.3f}",
              fontproperties=zh, fontsize=11)
axA.grid(True, alpha=0.3); axA.legend(prop=zh)

# ---- (B) Cd vs Re + power-law sensitivity (refit from raw points) ----
ReA = np.logspace(np.log10(45), np.log10(440), 200)


def powfit(Re, cd):
    p = np.polyfit(np.log(Re), np.log(cd), 1)
    return p[0], float(np.exp(p[1]))           # exponent n, prefactor A


for v in ORDER:
    Re = np.asarray(D[f"{v}_Re"], float); cd = np.asarray(D[f"{v}_Cd"], float)
    if len(Re) < 2:
        continue
    n, A = powfit(Re, cd)
    axB.plot(Re, cd, "o", color=COL[v], ms=8, zorder=4)
    axB.plot(ReA, A * ReA ** n, "-", color=COL[v], lw=2,
             label=f"{NAME[v]}  Cd∝Re^{n:.2f}")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("浮子雷诺数 Re", fontproperties=zh)
axB.set_ylabel("阻力系数 Cd", fontproperties=zh)
axB.set_title("(B) Cd(Re):cup6≈cup2≈blunt(凹坑无影响);圆头较低", fontproperties=zh, fontsize=11)
axB.grid(True, which="both", alpha=0.3); axB.legend(prop=zh, fontsize=8)

# ---- (C) cup6 axisymmetric flow field (mirrored) ----
if "field_w" in D.files:
    z_c = D["field_z_c"]; r_c = D["field_r_c"]
    nz, nr = len(z_c), len(r_c)

    def to_c(arr):
        a = np.asarray(arr, float)
        if a.shape[0] == nz + 1:
            a = 0.5 * (a[:-1, :] + a[1:, :])
        if a.shape[1] == nr + 1:
            a = 0.5 * (a[:, :-1] + a[:, 1:])
        return a

    wc = to_c(D["field_w"]); vc = to_c(D["field_v"]); solid = np.asarray(D["field_solid"])
    U = float(D["U_IN"])
    spd = np.sqrt(wc ** 2 + vc ** 2) / U
    spd = np.ma.masked_where(solid, spd)
    r_full = np.concatenate([-r_c[::-1], r_c])
    spd_full = np.ma.concatenate([spd[:, ::-1], spd], axis=1)
    W = np.concatenate([wc[:, ::-1], wc], axis=1)
    Vr = np.concatenate([-vc[:, ::-1], vc], axis=1)
    solid_full = np.concatenate([solid[:, ::-1], solid], axis=1)
    pc = axC.pcolormesh(z_c, r_full, spd_full.T, cmap="turbo", shading="auto",
                        vmin=0, vmax=np.nanpercentile(spd_full.compressed(), 99))
    Wp = np.where(solid_full, 0, W).T; Vp = np.where(solid_full, 0, Vr).T
    axC.streamplot(z_c, r_full, Wp, Vp, color="0.1", density=1.6, linewidth=0.5, arrowsize=0.6)
    rgba = np.zeros((len(r_full), nz, 4)); rgba[solid_full.T] = [0.3, 0.37, 0.48, 1]
    axC.imshow(rgba, origin="lower", extent=[z_c[0], z_c[-1], r_full[0], r_full[-1]],
               aspect="auto", zorder=5)
    fig.colorbar(pc, ax=axC, pad=0.01, fraction=0.046).set_label("|u|/U_in", fontproperties=zh)
    axC.set_xlim(z_c[0], z_c[-1]); axC.set_ylim(r_full[0], r_full[-1])
    axC.set_xlabel("z [mm] (流动 →)", fontproperties=zh); axC.set_ylabel("r [mm]", fontproperties=zh)
    axC.set_title(f"(C) cup6 轴对称流场 (Re={int(float(D['field_Re']))})", fontproperties=zh, fontsize=11)

fig.suptitle("轴对称 CFD(已用管流解析解验证):头部形状对阻力 Cd(Re) 与粘度免疫(β)的影响",
             fontproperties=zh, fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE + "/float_axisym.png", dpi=140)
print("saved float_axisym.png")
for v in ORDER:
    if len(np.asarray(D[f"{v}_Re"])):
        print(f"{v:8s}: Cd∞={float(D[f'{v}_Cdinf']):6.2f} β={float(D[f'{v}_beta']):6.3f} "
              f"Cd={np.asarray(D[f'{v}_Cd'])}  rms%={np.asarray(D[f'{v}_rms'])}")
