"""Confirmed float outline: blunt head (Ø12) + forward cylindrical pit (Ø8 x 2 deep)
+ cone tail. Clean engineering cross-section. Flow from the left."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# parameters [mm]
R, Lh, Ltot, rp, dp, rtail = 6.0, 5.0, 26.0, 4.0, 2.0, 0.5
NEW = [(0, R), (Lh, R), (Ltot, rtail), (Ltot, -rtail), (Lh, -R), (0, -R),
       (0, -rp), (dp, -rp), (dp, rp), (0, rp)]

fig, ax = plt.subplots(figsize=(12.5, 5.6))

# pit fluid
ax.add_patch(Rectangle((0, -rp), dp, 2 * rp, facecolor="#cfe3f5", edgecolor="none", zorder=1))
# solid (hatched)
ax.add_patch(Polygon(NEW, closed=True, facecolor="#d9dfe8", edgecolor="k",
                     hatch="////", lw=2.0, zorder=2))
ax.plot([-7, 29], [0, 0], "k-.", lw=0.8, zorder=0)
ax.text(28.5, 0.5, "轴线", fontproperties=zh, fontsize=8)

# flow into pit
for yy in (-3, 0, 3):
    ax.add_patch(FancyArrowPatch((-7.5, yy), (-0.6, yy), arrowstyle="-|>",
                                 mutation_scale=15, color="#1f77b4", lw=2.2, zorder=3))
ax.text(-7.8, 4.6, "来流", fontproperties=zh, color="#1f77b4", fontsize=12)


def dim_h(x1, x2, y, txt, tcol="b"):
    ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=tcol, lw=1.1))
    ax.text((x1 + x2) / 2, y + 0.25, txt, fontproperties=zh, color=tcol, fontsize=9, ha="center")


def dim_v(x, y1, y2, txt, side=-1):
    ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="<->", color="b", lw=1.1))
    ax.text(x + 0.35 * side, (y1 + y2) / 2, txt, fontproperties=zh, color="b", fontsize=9,
            ha="right" if side < 0 else "left", va="center")


def co(text, xy, xytext, c="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10.5, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3), zorder=5)


# dimensions
dim_v(-2.6, -R, R, "Ø12")
dim_v(1.0, -rp, rp, "Ø8", side=1)
dim_h(0, dp, rp + 0.9, "2")
dim_h(0, Lh, R + 1.6, "钝头 5")
dim_h(0, Ltot, -R - 2.2, "总长 26")

# callouts
co("杯口锐边 = 最大直径(迎流)", (0, R), (7, 11.5), "tab:red")
co("前向圆柱凹坑(平底)\nØ8 × 深2", (1.2, -3.0), (-3, -10.5), "k")
co("锥尾(向下游收尖)", (18, 2.6), (21, 9), "k")

ax.set_title("确认外形:钝头 Ø12 + 前向圆柱凹坑 Ø8×深2(平底)+ 锥尾",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-10, 31); ax.set_ylim(-13, 13.5)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_shape_final.png", dpi=150)
print("saved float_shape_final.png")
