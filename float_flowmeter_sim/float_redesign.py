"""
Evidence-based redesign of the spring-loaded variable-area (mechatronic) float meter.
Left  : redesigned FLOAT cross-section (dimensions + why, tied to the CFD findings).
Right : assembly (float in tapered bore: seat/check-valve, spring, guide rod, magnet
        position sensor, flow path).  Flow is upward.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- redesigned float profile (axis vertical = flow up; r radial) -------
RB = 1.6          # guide-bore radius (Ø3.2)
RE = 6.0          # metering edge radius (Ø12)
LAND = 3.0        # cylindrical metering land height
ZT = 26.0         # float top
# right-half material polygon (between guide bore and outer profile), bottom->top
right = [(RB, 0), (RE, 0), (RE, LAND), (RB, ZT - 1.5), (RB, 0)]
floatpoly = [(x, y) for (x, y) in right] + [(-x, y) for (x, y) in right[::-1]]


def draw_float(ax, z0=0.0, fc="#c9d2de", hatch="////", lw=1.8, mag=True):
    ax.add_patch(Polygon([(x, y + z0) for x, y in floatpoly], closed=True,
                         facecolor=fc, edgecolor="k", hatch=hatch, lw=lw, zorder=3))
    if mag:
        for sx in (1, -1):
            ax.add_patch(Rectangle((sx * 2.6 if sx > 0 else -4.4, 7 + z0), 1.8, 9,
                                   facecolor="#c44", edgecolor="k", lw=0.6, zorder=4))


# ====================================================================================
fig, (axF, axA) = plt.subplots(1, 2, figsize=(15, 9),
                               gridspec_kw=dict(width_ratios=[1, 1]))

# -------------------- (A) FLOAT DETAIL ----------------------------------------------
draw_float(axF)
axF.plot([0, 0], [-2, ZT + 2], "k-.", lw=0.7)
# incoming flow (from below)
for xx in (-3, 0, 3):
    axF.add_patch(FancyArrowPatch((xx, -6), (xx, -2.5), arrowstyle="-|>",
                                  mutation_scale=15, color="#1f77b4", lw=2))
axF.text(0, -7.4, "来流 ↑", fontproperties=zh, color="#1f77b4", ha="center", fontsize=11)


def call(ax, t, xy, xytext, c="k"):
    ax.annotate(t, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.2), zorder=6)


call(axF, "钝平迎流面 + 锐利计量边 Ø12\n(钝面→阻力高/灵敏;锐边→分离钉死=粘度免疫)",
     (RE, 0.1), (16, -4), "tab:red")
call(axF, "短计量柱段 3mm\n(与锥孔定义环隙,线性化)", (RE, LAND), (16, 6))
call(axF, "流线锥尾(尾流稳定、导向)", (3.2, 16), (15, 16))
call(axF, "中心导杆孔 Ø3.2\n(定心/防转)", (RB, 21), (-13, 22))
call(axF, "环形磁铁(位置检测)", (3.4, 11), (-15, 11), "#c44")
axF.text(0, ZT + 5.5, "去掉前向凹坑 / 锥面槽\n(CFD 验证:对计量无影响,从简)",
         fontproperties=zh, fontsize=10, color="0.35", ha="center",
         bbox=dict(boxstyle="round", fc="#eee", alpha=0.8))
# dims
axF.annotate("", (-RE, -1.3), (RE, -1.3), arrowprops=dict(arrowstyle="<->", color="b"))
axF.text(0, -2.2, "Ø12", fontproperties=zh, color="b", ha="center", fontsize=9)
axF.annotate("", (8.5, 0), (8.5, ZT), arrowprops=dict(arrowstyle="<->", color="b"))
axF.text(9, ZT/2, "≈26", fontproperties=zh, color="b", rotation=90, va="center", fontsize=9)
axF.set_title("(A) 重新设计的浮子(去凹坑/去槽,锐边计量)", fontproperties=zh, fontsize=13)
axF.set_xlim(-20, 22); axF.set_ylim(-9, 34); axF.set_aspect("equal"); axF.axis("off")

# -------------------- (B) ASSEMBLY (float in tapered bore) ---------------------------
HB, HT = -10, 60                      # housing bottom/top
WALL = 14
def bore_r(z):                         # tapered bore widens upward
    return RE + 0.6 + np.maximum(0, np.asarray(z, float) - 12) * 0.13
zb = np.linspace(12, HT, 50)
axA.add_patch(Rectangle((-WALL, HB), WALL - (RE + 0.6), HT - HB, facecolor="#ececec",
                        edgecolor="k", lw=1.3))
axA.add_patch(Rectangle((RE + 0.6, HB), WALL - (RE + 0.6), HT - HB, facecolor="#ececec",
                        edgecolor="k", lw=1.3))
axA.plot(bore_r(zb), zb, "k-", lw=1.2); axA.plot(-bore_r(zb), zb, "k-", lw=1.2)
axA.plot([RE + 0.6, RE + 0.6], [HB, 12], "k-", lw=1.2)
axA.plot([-(RE + 0.6), -(RE + 0.6)], [HB, 12], "k-", lw=1.2)
# seat (check valve)
axA.plot([RE + 0.6, 3], [12, 7], "k-", lw=2.3); axA.plot([-(RE + 0.6), -3], [12, 7], "k-", lw=2.3)
axA.text(10.5, 8, "阀座\n(止回)", fontproperties=zh, fontsize=9, va="center")
# float at mid stroke
draw_float(axA, z0=14, fc="#aebacb", hatch="")
# guide rod
axA.add_patch(Rectangle((-1.4, HB + 3), 2.8, HT - HB - 8, facecolor="#888", ec="k", lw=0.5, zorder=2))
axA.text(-3.2, HT - 4, "导杆", fontproperties=zh, fontsize=9, ha="center")
# spring above float
sy = np.linspace(14 + ZT + 1, HT - 4, 200)
axA.plot(3.0 * np.sin(2 * np.pi * 7 * (sy - sy[0]) / (sy[-1] - sy[0])), sy, color="#555", lw=1.5, zorder=4)
axA.text(5, (sy[0] + sy[-1]) / 2, "弹簧\n(复位/定向无关)", fontproperties=zh, fontsize=9, va="center")
# flow
for (x, ya, yb) in [(0, HB + 1, 6), (RE + 2.0, 16, 26), (-(RE + 2.0), 16, 26), (0, HT - 9, HT - 1)]:
    axA.add_patch(FancyArrowPatch((x, ya), (x, yb), arrowstyle="-|>", mutation_scale=15,
                                  color="#1f77b4", lw=2.2, zorder=6))
axA.text(0, HB - 1.5, "流入 (G½)", fontproperties=zh, fontsize=9.5, ha="center", color="#1f77b4")
axA.text(0, HT + 1.5, "流出 (G½)", fontproperties=zh, fontsize=9.5, ha="center", color="#1f77b4")
# sensor
axA.add_patch(Rectangle((WALL + 0.5, 24), 5.5, 7, facecolor="#ffe08a", ec="k", lw=1, zorder=6))
axA.annotate("", (WALL + 0.5, 27), (3.4, 27), arrowprops=dict(arrowstyle="->", color="#c44", lw=1.2, ls="--"))
axA.text(WALL + 3.2, 33, "磁致伸缩/AMR\n位置传感器", fontproperties=zh, fontsize=8.5, ha="center")
axA.text(WALL + 3.2, 22.5, "位置 ∝ 流量\n→4–20mA / IO-Link", fontproperties=zh, fontsize=8, ha="center", va="top")
axA.text(0, HB - 5, "竖直/任意方向(弹簧主导)", fontproperties=zh, fontsize=9, ha="center", color="0.4")
axA.set_title("(B) 装配:锥孔 + 阀座 + 弹簧 + 导杆 + 磁位置读出", fontproperties=zh, fontsize=13)
axA.set_xlim(-20, 24); axA.set_ylim(HB - 7, HT + 6); axA.set_aspect("equal"); axA.axis("off")

fig.suptitle("重新设计:弹簧活塞式变截面流量计(证据驱动 —— 来自验证级轴对称 CFD)",
             fontproperties=zh, fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_redesign.png", dpi=145)
print("saved float_redesign.png")
