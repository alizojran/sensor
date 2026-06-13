"""Proposed float v3: BLUNT head with a forward-facing CONCAVE CAVITY (cup) + cone tail.
Flow comes from the left into the cup. The sharp cup rim pins the separation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# Solid cross-section: top rim -> tapered tail -> back -> bottom rim -> cup apex -> close
RIM_X, RIM_R = 2.5, 6.0          # sharp cup rim (max diameter, faces flow)
CUP_APEX = (6.5, 0.0)            # cavity bottom (recessed downstream)
TAIL = (26.0, 0.5)
NEW = [(RIM_X, RIM_R), (TAIL[0], TAIL[1]), (TAIL[0], -TAIL[1]),
       (RIM_X, -RIM_R), CUP_APEX]
# old ghost: the symmetric bi-cone (pointed nose) that was also wrong
OLD = [(0, 0.5), (12, 6.0), (24, 0.5), (12, -6.0), (0, -0.5)]

fig, ax = plt.subplots(figsize=(12, 5.6))
ax.add_patch(Polygon(OLD, closed=True, facecolor="none", edgecolor="0.65", ls="--", lw=1.3))
ax.text(12, -8.2, "上一版(尖锥纺锤,也不对)", fontproperties=zh, color="0.55",
        fontsize=9, ha="center")

# fluid-filled cup (light blue) + recirculation
cup = [(RIM_X, RIM_R), CUP_APEX, (RIM_X, -RIM_R)]
ax.add_patch(Polygon(cup, closed=True, facecolor="#bcd6f0", edgecolor="none", zorder=1))
ax.add_patch(Polygon(NEW, closed=True, facecolor="#5b6e8c", edgecolor="k", lw=1.9, zorder=2))
ax.plot([-5, 27], [0, 0], "k-.", lw=0.7, zorder=0)

# incoming flow into the cup
for yy in (-2.6, 0, 2.6):
    ax.add_patch(FancyArrowPatch((-7, yy), (RIM_X - 0.4, yy * 0.5), arrowstyle="-|>",
                                 mutation_scale=16, color="#1f77b4", lw=2.2, zorder=3))
ax.text(-8.5, 4.2, "来流", fontproperties=zh, color="#1f77b4", fontsize=12)
# recirculation inside cup
ax.annotate("", (4.3, 2.3), (4.3, -2.3),
            arrowprops=dict(arrowstyle="->", color="#163d66", lw=1.6,
                            connectionstyle="arc3,rad=0.6"), zorder=4)
ax.text(4.6, 0, "驻涡", fontproperties=zh, color="#163d66", fontsize=9, zorder=4)


def co(text, xy, xytext, c="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10.5, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3), zorder=5)


co("钝头 + 前向凹坑(杯形)\n杯口迎流", (RIM_X + 0.5, 4.5), (-2, 11), "k")
co("杯口锐边 = 最大直径\n分离点钉死于此\n→ 阻力 Re 无关(粘度免疫)", (RIM_X, RIM_R), (9, 11.5), "tab:red")
co("锥尾", (20, 2.0), (22, 9), "k")

ax.annotate("", (RIM_X, -6.9), (TAIL[0], -6.9), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(14, -7.0, "锥尾(向下游收尖)", fontproperties=zh, color="b", fontsize=9,
        ha="center", va="bottom")

ax.set_title("外形 v3:钝头 + 前向凹坑(杯形,迎流)+ 锥尾 — 这个对吗?",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-10, 30); ax.set_ylim(-12, 13.5)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_shape_proposal3.png", dpi=140)
print("saved float_shape_proposal3.png")
