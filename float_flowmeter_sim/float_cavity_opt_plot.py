"""Plot cavity sizing trade-off (reads cavity_opt.npz)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/cavity_opt.npz")

d_dp = np.asarray(D["d_dp"]); d_Cd = np.asarray(D["d_Cd"]); d_puls = np.asarray(D["d_puls"])
a_Dp = np.asarray(D["a_Dp"]); a_Cd = np.asarray(D["a_Cd"]); a_puls = np.asarray(D["a_puls"])
Dp_depth = 8.0
d_ratio = d_dp / Dp_depth
a_ratio = 3.0 / a_Dp

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- (A) depth sweep: Cd & pulsation vs depth ----
axA.axvspan(5.2, 6.6, color="green", alpha=0.10)
axA.text(5.9, 250, "推荐\nd/Dp≥0.75", fontproperties=zh, color="green", ha="center", fontsize=10)
axA.axvspan(1.6, 4.2, color="red", alpha=0.07)
axA.text(2.9, 250, "浅腔拍打\n(避开)", fontproperties=zh, color="red", ha="center", fontsize=10)
l1, = axA.plot(d_dp, d_puls, "o-", color="tab:red", lw=2, ms=8, label="脉动 RMS%")
axA.set_xlabel("凹坑深度 d [mm]  (Ø8)", fontproperties=zh)
axA.set_ylabel("脉动 RMS [% of 平均阻力]", fontproperties=zh, color="tab:red")
axA.tick_params(axis="y", labelcolor="tab:red")
axA.set_ylim(0, 270)
for x, r in zip(d_dp, d_ratio):
    axA.annotate(f"{r:.2f}", (x, 8), fontproperties=zh, fontsize=8, color="0.4", ha="center")
axA.text(0.98, 0.02, "数字 = d/Dp", transform=axA.transAxes, fontproperties=zh,
         fontsize=8, color="0.4", ha="right")
axt = axA.twinx()
l2, = axt.plot(d_dp, d_Cd, "s--", color="tab:blue", lw=2, ms=7, label="阻力 Cd")
axt.set_ylabel("阻力系数 Cd", fontproperties=zh, color="tab:blue")
axt.tick_params(axis="y", labelcolor="tab:blue"); axt.set_ylim(0, 20)
axA.set_title("(A) 深度扫描:脉动随深度的变化(阻力近乎不变)", fontproperties=zh, fontsize=12)
axA.legend(handles=[l1, l2], prop=zh, loc="center right")

# ---- (B) pulsation collapses on d/Dp (depth + diameter sweeps) ----
axB.axvspan(0.6, 1.0, color="green", alpha=0.10)
axB.axvspan(0.2, 0.55, color="red", alpha=0.07)
axB.semilogy(d_ratio, d_puls, "o", color="tab:red", ms=10, label="深度扫描(Ø8)")
axB.semilogy(a_ratio, a_puls, "s", color="tab:purple", ms=10, label="直径扫描(d=3)")
axB.set_xlabel("深径比  d / Dp", fontproperties=zh)
axB.set_ylabel("脉动 RMS%  (对数)", fontproperties=zh)
axB.set_title("(B) 脉动由 d/Dp 主导:两组扫描基本重合", fontproperties=zh, fontsize=12)
axB.text(0.8, 200, "稳定驻涡区", fontproperties=zh, color="green", ha="center", fontsize=10)
axB.text(0.37, 12, "浅腔拍打区", fontproperties=zh, color="red", ha="center", fontsize=10)
axB.grid(True, which="both", alpha=0.3); axB.legend(prop=zh, loc="lower left")

fig.suptitle("凹坑尺寸选取:深径比 d/Dp 是关键 — 取 d/Dp≥0.75(Ø8→深≈6mm)以兼顾高阻力与低脉动",
             fontproperties=zh, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE + "/float_cavity_opt.png", dpi=140)
print("saved float_cavity_opt.png")
