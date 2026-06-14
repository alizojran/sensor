"""Verification figure for the redesigned float (reads redesign_cfd.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/redesign_cfd.npz")

Re = D["Re"]; Cd = D["Cd"]; Cds = D["Cd_surf"]; rms = D["rms"]; nexp = float(D["nexp"])
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

axL.plot(Re, Cd, "o-", color="tab:green", lw=2, ms=9, label="Cd (控制体)")
axL.plot(Re, Cds, "s--", color="tab:olive", lw=1.6, ms=7, label="Cd (表面积分)")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("浮子雷诺数 Re", fontproperties=zh); axL.set_ylabel("阻力系数 Cd", fontproperties=zh)
axL.set_title(f"(左) 新外形 Cd(Re):Cd∝Re^{nexp:.2f};两法吻合;流动定常(rms≤{rms.max():.2f}%)",
              fontproperties=zh, fontsize=10.5)
axL.grid(True, which="both", alpha=0.3); axL.legend(prop=zh)

if "field_w" in D.files:
    z_c = D["z_c"]; r_c = D["r_c"]; nz, nr = len(z_c), len(r_c)

    def to_c(a):
        a = np.asarray(a, float)
        if a.shape[0] == nz + 1:
            a = 0.5 * (a[:-1, :] + a[1:, :])
        if a.shape[1] == nr + 1:
            a = 0.5 * (a[:, :-1] + a[:, 1:])
        return a

    wc, vc = to_c(D["field_w"]), to_c(D["field_v"]); solid = np.asarray(D["field_solid"]); U = float(D["U"])
    spd = np.ma.masked_where(solid, np.sqrt(wc ** 2 + vc ** 2) / U)
    r_full = np.concatenate([-r_c[::-1], r_c])
    spd_full = np.ma.concatenate([spd[:, ::-1], spd], axis=1)
    W = np.concatenate([wc[:, ::-1], wc], axis=1); Vr = np.concatenate([-vc[:, ::-1], vc], axis=1)
    sf = np.concatenate([solid[:, ::-1], solid], axis=1)
    pc = axR.pcolormesh(z_c, r_full, spd_full.T, cmap="turbo", shading="auto",
                        vmin=0, vmax=np.nanpercentile(spd_full.compressed(), 99))
    axR.streamplot(z_c, r_full, np.where(sf, 0, W).T, np.where(sf, 0, Vr).T,
                   color="0.1", density=1.6, linewidth=0.5, arrowsize=0.6)
    rgba = np.zeros((len(r_full), nz, 4)); rgba[sf.T] = [0.3, 0.37, 0.48, 1]
    axR.imshow(rgba, origin="lower", extent=[z_c[0], z_c[-1], r_full[0], r_full[-1]],
               aspect="auto", zorder=5)
    fig.colorbar(pc, ax=axR, pad=0.01, fraction=0.046).set_label("|u|/U_in", fontproperties=zh)
    axR.set_xlim(z_c[0] + 4, z_c[-1] - 6)
    z0 = z_c[0] + 10.0
    axR.annotate("钝面驻点(高压)", (z0 + 0.3, 0), (z0 - 5, 6.5), fontproperties=zh, color="w",
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="w"))
    axR.annotate("锐边分离(钉死)", (z0 + 0.3, 5.8), (z0 + 6, 8.5), fontproperties=zh, color="yellow",
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="yellow"))
    axR.set_xlabel("z [mm] (流动 →)", fontproperties=zh); axR.set_ylabel("r [mm]", fontproperties=zh)
    axR.set_title("(右) 新外形轴对称流场(Re=200):钝面+锐边按预期工作", fontproperties=zh, fontsize=10.5)

fig.suptitle("CFD 复核:重新设计的浮子(钝面 + 锐利计量边,无凹坑)按预期工作",
             fontproperties=zh, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE + "/float_redesign_cfd.png", dpi=140)
print("saved float_redesign_cfd.png")
