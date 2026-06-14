"""Manufacturing drawing of the redesigned float: dimensions, tolerances, GD&T,
the metering-gap tolerance chain, guide-rod clearance, and the sharp-edge spec."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# float profile, horizontal: blunt face at x=0, tail to the right. r in mm.
RB, RE, LAND, LTOT, RT = 1.6, 6.0, 3.0, 26.0, 1.5
right = [(0, RB), (0, RE), (LAND, RE), (LTOT, RT), (LTOT, RB)]   # upper, between bore & outer
upper = right
lower = [(x, -r) for (x, r) in right[::-1]]
mat_up = upper + [(LTOT, RB), (0, RB)]
mat_lo = [(x, -r) for (x, r) in mat_up]

fig, ax = plt.subplots(figsize=(15, 8))

# material (hatched), upper & lower halves (annulus between guide bore and profile)
for poly in (mat_up, mat_lo):
    ax.add_patch(Polygon(poly, closed=True, facecolor="#e8ebf0", edgecolor="k",
                         hatch="////", lw=1.8, zorder=2))
# guide bore (through hole) center band
ax.add_patch(Rectangle((0, -RB), LTOT, 2 * RB, facecolor="white", edgecolor="k",
                       lw=1.0, ls=(0, (6, 3)), zorder=3))
# magnet pockets
for s in (1, -1):
    ax.add_patch(Rectangle((6, s * 2.5 if s > 0 else -4.5), 9, 2.0,
                           facecolor="#e9c0c0", edgecolor="0.3", lw=0.8, zorder=4))
ax.text(10.5, 0, "磁铁腔", fontproperties=zh, fontsize=8, ha="center", va="center", zorder=5)
ax.plot([-3, LTOT + 4], [0, 0], "k-.", lw=0.7, zorder=1)


def dh(x1, x2, y, t, c="b", dy=0.5):
    ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=c, lw=1))
    ax.text((x1 + x2) / 2, y + dy, t, fontproperties=zh, color=c, fontsize=8.5, ha="center")


def dv(x, y1, y2, t, c="b", side=1):
    ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="<->", color=c, lw=1))
    ax.text(x + 0.5 * side, (y1 + y2) / 2, t, fontproperties=zh, color=c, fontsize=8.5,
            va="center", ha="left" if side > 0 else "right")


def lead(t, xy, xytext, c="k"):
    ax.annotate(t, xy=xy, xytext=xytext, fontproperties=zh, fontsize=9, color=c, ha="center",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.1), zorder=6)


# dimensions / tolerances
dv(-2.0, -RE, RE, "Ø12 -0/-0.02\n(计量边, 基准B)", side=-1)
dv(LTOT + 2.2, -RB, RB, "Ø3.2 H7\n(导杆孔=基准A)", side=1)
dv(LTOT + 0.2, -RT, RT, "Ø3", side=1)
dh(0, LAND, RE + 1.2, "3 ±0.05")
dh(0, LTOT, -RE - 2.4, "26 ±0.2")
lead("锐利计量边: R<=0.05, 去毛刺\n(控制分离=粘度免疫, 关键特征)", (0.1, RE), (6, RE + 4.5), "tab:red")
lead("流线锥尾", (20, 3.0), (22, 6.5))
ax.text(0.6, 0, "通孔", fontproperties=zh, fontsize=7.5, ha="center", color="0.4")

# GD&T row
ax.text(-3, RE + 6.2,
        "同轴度: Ø12 计量边 相对 基准A(导杆孔)  跳动 <= 0.02 TIR\n"
        "表面粗糙度: 计量边/柱段 Ra0.8, 其余 Ra1.6",
        fontproperties=zh, fontsize=9, ha="left",
        bbox=dict(boxstyle="round", fc="#f3f3f3", ec="0.5"))

# tolerance-chain box (the key ask)
chain = ("公差链 — 计量环隙 (零位, 半径向):\n"
         "  环隙 = (D_孔 − D_边)/2 = (12.40 − 12.00)/2 = 0.20 mm\n"
         "  · 配合孔 Ø12.40 H7 (+0.027/0),  计量边 Ø12 -0/-0.02\n"
         "    → 直径贡献使环隙 0.20…0.23 (≈ ±0.015)\n"
         "  · 偏心: 导杆配合 H7/g6 → 半径偏心 <= 0.012\n"
         "    → 局部最小环隙 >= 0.19;偏心环隙抬升流量需靠紧配合抑制\n"
         "  · 满量程环隙 3.2 mm 由锥孔渐扩给出 (锥度 0.15 mm/mm)\n"
         "  → 绝对环隙散差由单点[校准系数]吸收;\n"
         "    线性/重复性靠 同轴度<=0.02 + 锐边<=R0.05 保证")
ax.text(-3, -RE - 4.5, chain, fontproperties=zh, fontsize=8.2, ha="left", va="top",
        bbox=dict(boxstyle="round", fc="#eef5ff", ec="tab:blue"))

# notes / title block
notes = ("技术要求:\n"
         "1. 材料 316L 不锈钢;磁铁 NdFeB 全密封于磁铁腔内(激光焊封)。\n"
         "2. 配对件:壳体锥孔 零位 Ø12.40 H7;导杆 Ø3.2 g6(滑动配合)。\n"
         "3. 计量边须锐利无毛刺(R<=0.05);此特征决定粘度免疫,关键尺寸。\n"
         "4. 装配后浮子在锥孔内自由滑动、无卡滞;前置 200 µm 过滤。")
ax.text(15.5, -RE - 4.5, notes, fontproperties=zh, fontsize=8.6, ha="left", va="top",
        bbox=dict(boxstyle="round", fc="#fffdf0", ec="0.5"))

ax.set_title("重新设计浮子 — 制造图(尺寸 / 公差 / 公差链)  316L  比例示意",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-7, 33); ax.set_ylim(-20, 14); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_mfg_drawing.png", dpi=150)
print("saved float_mfg_drawing.png")
