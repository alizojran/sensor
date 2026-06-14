"""Mating manufacturing drawing of the HOUSING / tapered bore (mates with the float).
Cross-section + bore tolerances + the float<->bore fit table. Flow left->right."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

R_HOUS = 14.0
L = 40.0
R_PORT, R_ZERO, R_FULL = 5.0, 6.2, 9.2
BORE = [(0, R_PORT), (8, R_PORT), (12, R_ZERO), (32, R_FULL), (L, R_FULL)]

fig, ax = plt.subplots(figsize=(15, 8))
# wall material (between bore and outer), upper + lower
up = BORE + [(L, R_HOUS), (0, R_HOUS)]
lo = [(x, -y) for x, y in up]
for poly in (up, lo):
    ax.add_patch(Polygon(poly, closed=True, facecolor="#e8ebf0", edgecolor="k",
                         hatch="////", lw=1.7, zorder=2))
ax.plot([-4, L + 4], [0, 0], "k-.", lw=0.7)
# flow
ax.annotate("", (-2, 0), (-5.5, 0), arrowprops=dict(arrowstyle="<|-", color="#1f77b4", lw=2.5))
ax.text(-5.5, 1.2, "流入", fontproperties=zh, color="#1f77b4", fontsize=10)
ax.text(L + 1, 1.2, "流出", fontproperties=zh, color="#1f77b4", fontsize=10)


def dh(x1, x2, y, t, c="b"):
    ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=c, lw=1))
    ax.text((x1 + x2) / 2, y + 0.4, t, fontproperties=zh, color=c, fontsize=8.5, ha="center")


def dv(x, y1, y2, t, side=1):
    ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="<->", color="b", lw=1))
    ax.text(x + 0.5 * side, (y1 + y2) / 2, t, fontproperties=zh, color="b", fontsize=8.3,
            va="center", ha="left" if side > 0 else "right")


def lead(t, xy, xytext, c="k"):
    ax.annotate(t, xy=xy, xytext=xytext, fontproperties=zh, fontsize=9, color=c, ha="center",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.1), zorder=6)


dv(-2.5, -R_HOUS, R_HOUS, "Ø28 外形", side=-1)
dv(12.0, -R_ZERO, R_ZERO, "Ø12.40 H7\n(零位锥孔)", side=1)
dv(L - 0.4, -R_FULL, R_FULL, "Ø18.4\n(满量程)", side=1)
dv(2.0, -R_PORT, R_PORT, "G1/2", side=1)
dh(0, L, -R_HOUS - 2.2, "L 40")
dh(12, 32, R_FULL + 5.5, "计量行程 20 (锥度 0.15 mm/mm 单边)")
lead("锥形阀座(止回)\n浮子零流量落座", (11, R_ZERO - 0.4), (4, R_HOUS + 3), "tab:red")
lead("渐扩锥孔(线性化)", (24, R_FULL - 0.5), (24, R_HOUS + 3))
lead("进/出口 G1/2 内螺纹", (3, R_PORT), (3, -R_HOUS - 6))

fit = ("配合(与浮子):\n"
       "• 计量:壳体锥孔零位 Ø12.40 H7  ×  浮子边 Ø12 -0/-0.02\n"
       "   → 零位半径环隙 0.20(0.20…0.23);满量程 3.2 由锥度给出\n"
       "• 导向:壳体/上下蜘蛛 导杆孔 Ø3.0 H7  ×  导杆 Ø3.0 g6(滑动)\n"
       "   导杆 × 浮子中心孔 Ø3.0 H7 → 浮子在导杆上自由滑动")
ax.text(-5, -R_HOUS - 4.5, fit, fontproperties=zh, fontsize=8.4, ha="left", va="top",
        bbox=dict(boxstyle="round", fc="#eef5ff", ec="tab:blue"))

notes = ("技术要求:\n"
         "1. 材料 316L;接液 O 形圈 FKM;耐压 200 bar;IP67。\n"
         "2. 锥孔表面 Ra0.4;锥孔相对进出口/导杆孔 同轴度 ≤0.03。\n"
         "3. 零位锥孔 Ø12.40 是计量基准;锥度沿轴线性渐扩。\n"
         "4. 外壁非磁性(316L)以便磁致伸缩/AMR 在外读浮子位置。")
ax.text(20, -R_HOUS - 4.5, notes, fontproperties=zh, fontsize=8.5, ha="left", va="top",
        bbox=dict(boxstyle="round", fc="#fffdf0", ec="0.5"))

ax.set_title("壳体 / 锥孔 配合制造图(与浮子配作)  316L  比例示意",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-8, 46); ax.set_ylim(-30, 22); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/housing_mfg_drawing.png", dpi=150)
print("saved housing_mfg_drawing.png")
