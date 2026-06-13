"""Proposed float outline v2: symmetric bi-conical (spindle). Flow from the left."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False


def full_section(upper):
    return upper + [(a, -r) for (a, r) in upper[::-1]]


# NEW: symmetric bi-cone (spindle) -- pointed both ends, sharp max-dia rim in the middle
NEW = full_section([(0, 0.5), (12, 6.0), (24, 0.5)])
# OLD ghost: the asymmetric necked float actually simulated before
OLD = full_section([(0, 0.5), (6, 6.0), (6, 4.5), (24, 4.5), (30, 0.6)])

fig, ax = plt.subplots(figsize=(12, 5.4))
ax.add_patch(Polygon(OLD, closed=True, facecolor="none", edgecolor="0.6", ls="--", lw=1.4))
ax.text(16, -7.6, "旧版(不对称·长主体,你说不对的那版)", fontproperties=zh,
        color="0.55", fontsize=9, ha="center")

ax.add_patch(Polygon(NEW, closed=True, facecolor="#5b6e8c", edgecolor="k", lw=1.8))
ax.plot([-3, 27], [0, 0], "k-.", lw=0.7)

ax.add_patch(FancyArrowPatch((-7, 0), (-2.5, 0), arrowstyle="-|>", mutation_scale=22,
                             color="#1f77b4", lw=3))
ax.text(-8, 1.3, "来流", fontproperties=zh, color="#1f77b4", fontsize=12)


def co(text, xy, xytext, c="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10.5, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3))


co("上游锥头(迎流)", (6, 3.0), (4, 10.5), "k")
co("中部 = 最大直径\n锐利计量边(主控制边)", (12, 6.0), (12, 11.5), "tab:red")
co("下游锥尾(对称)", (18, 3.0), (20, 10.5), "k")
co("槽可选位置(你最初圈的前锥)", (7, 2.2), (7, -10.5), "tab:green")

ax.annotate("", (0, -7.0), (24, -7.0), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(12, -7.0, "对称(前后等长)", fontproperties=zh, color="b", fontsize=9,
        ha="center", va="bottom")
ax.annotate("", (12.7, -6), (12.7, 6), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(13.2, 0, "Ø12 锐边", fontproperties=zh, color="b", fontsize=9, va="center")

ax.set_title("提议外形 v2:对称双锥(纺锤)型 — 两头收尖,中部锐边。这个对吗?",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-9, 31); ax.set_ylim(-12, 13.5)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_shape_proposal2.png", dpi=140)
print("saved float_shape_proposal2.png")
