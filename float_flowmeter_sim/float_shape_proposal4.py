"""Proposed float v4: BLUNT head with a flat-bottomed CYLINDRICAL pit (~2 mm deep)
facing the flow, + cone tail. Flow from the left into the pit."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# --- parameters (mm) ---
R = 6.0          # head radius (Ø12)
Lh = 5.0         # blunt head (cylinder) length
Ltot = 26.0      # tail tip
rp = 4.0         # pit radius (Ø8)  -- assumed, adjustable
dp = 2.0         # pit depth -- given ~2 mm

# solid cross-section (front face with cylindrical pit notch, then head, then cone tail)
NEW = [(0, R), (Lh, R), (Ltot, 0.5), (Ltot, -0.5), (Lh, -R), (0, -R),
       (0, -rp), (dp, -rp), (dp, rp), (0, rp)]

fig, ax = plt.subplots(figsize=(12, 5.8))

# fluid inside the pit
ax.add_patch(Rectangle((0, -rp), dp, 2 * rp, facecolor="#bcd6f0", edgecolor="none", zorder=1))
ax.add_patch(Polygon(NEW, closed=True, facecolor="#5b6e8c", edgecolor="k", lw=1.9, zorder=2))
ax.plot([-6, 28], [0, 0], "k-.", lw=0.7, zorder=0)

# incoming flow into the pit
for yy in (-3, 0, 3):
    ax.add_patch(FancyArrowPatch((-7, yy), (-0.5, yy), arrowstyle="-|>",
                                 mutation_scale=15, color="#1f77b4", lw=2.2, zorder=3))
ax.text(-8.5, 4.5, "来流", fontproperties=zh, color="#1f77b4", fontsize=12)
ax.annotate("", (1.0, 2.2), (1.0, -2.2),
            arrowprops=dict(arrowstyle="->", color="#163d66", lw=1.4,
                            connectionstyle="arc3,rad=0.5"), zorder=4)
ax.text(2.6, 0, "兜住驻涡", fontproperties=zh, color="#163d66", fontsize=8.5, zorder=4)


def co(text, xy, xytext, c="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10.5, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3), zorder=5)


co("钝头", (Lh, R), (Lh + 1, 11), "k")
co("前向圆柱形凹坑(平底)\n杯口迎流", (0.3, -3.5), (-4, -11), "k")
co("杯口锐边 = 最大直径\n分离钉死→阻力Re无关(粘度免疫)", (0, R), (8, 12), "tab:red")
co("锥尾", (20, 2.0), (22, 9), "k")

# dims
ax.annotate("", (0, rp + 0.4), (dp, rp + 0.4), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(dp / 2, rp + 0.7, "坑深≈2", fontproperties=zh, color="b", fontsize=9, ha="center")
ax.annotate("", (dp + 0.5, -rp), (dp + 0.5, rp), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(dp + 0.8, 0, "坑Ø8(可调)", fontproperties=zh, color="b", fontsize=8.5, va="center")
ax.annotate("", (-1.2, -R), (-1.2, R), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(-1.5, 0, "Ø12", fontproperties=zh, color="b", fontsize=9, ha="right", va="center")
ax.annotate("", (0, -8.0), (Ltot, -8.0), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(13, -8.1, "总长(可调)", fontproperties=zh, color="b", fontsize=9, ha="center", va="bottom")

ax.set_title("外形 v4:钝头 + 前向圆柱形凹坑(深≈2)+ 锥尾 — 这个对吗?",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-10, 30); ax.set_ylim(-12.5, 13.5)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_shape_proposal4.png", dpi=140)
print("saved float_shape_proposal4.png")
