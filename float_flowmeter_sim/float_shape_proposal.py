"""Quick proposed float outline for confirmation (no CFD). Flow comes from the left."""
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
    """upper = list of (a,r) along axis; return closed full cross-section polygon."""
    low = [(a, -r) for (a, r) in upper[::-1]]
    return upper + low


# ---- proposed NEW shape: bullet / piston ----
new_upper = [(0, 0.6), (9, 6.0), (24, 6.0), (27, 2.0)]
NEW = full_section(new_upper)

# ---- OLD shape (my previous CFD float, necked-down) for contrast ----
old_upper = [(0, 0.5), (6, 6.0), (6, 4.5), (24, 4.5), (30, 0.6)]
OLD = full_section(old_upper)

fig, ax = plt.subplots(figsize=(12, 5.2))

# old ghost
ax.add_patch(Polygon(OLD, closed=True, facecolor="none", edgecolor="0.6",
                     ls="--", lw=1.4))
ax.text(15, -7.4, "旧版(颈缩型,已证明槽无效的那版)", fontproperties=zh,
        color="0.5", fontsize=9, ha="center")

# new shape
ax.add_patch(Polygon(NEW, closed=True, facecolor="#5b6e8c", edgecolor="k", lw=1.8))
ax.plot([-3, 30], [0, 0], "k-.", lw=0.7)

# flow
ax.add_patch(FancyArrowPatch((-7, 0), (-2, 0), arrowstyle="-|>", mutation_scale=22,
                             color="#1f77b4", lw=3))
ax.text(-7.5, 1.2, "来流", fontproperties=zh, color="#1f77b4", fontsize=12)

# feature callouts
def co(text, xy, xytext, c="k"):
    ax.annotate(text, xy=xy, xytext=xytext, fontproperties=zh, fontsize=10.5, color=c,
                ha="center", arrowprops=dict(arrowstyle="->", color=c, lw=1.3))

co("尖锥头(迎流)\n≈ 60-70°", (4.5, 3.0), (1, 10), "k")
co("肩部 = 锐利计量边\n(主控制性边缘)", (9, 6.0), (13, 11), "tab:red")
co("等径主体(活塞)\n沿全长与锥孔成环隙", (17, 6.0), (18, 11), "k")
co("尾部斜切", (26, 4.0), (29, 9), "k")

# key dims
ax.annotate("", (0, -8.6), (27, -8.6), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(13.5, -9.6, "总长(示意,可调)", fontproperties=zh, color="b", fontsize=9, ha="center")
ax.annotate("", (9.6, -6), (9.6, 6), arrowprops=dict(arrowstyle="<->", color="b"))
ax.text(10.2, 0, "Ø12 (最大/锐边)", fontproperties=zh, color="b", fontsize=9, va="center")

ax.set_title("提议的新外形:子弹/活塞型(尖锥迎流 + 肩部锐边 + 等径主体)— 这个对吗?",
             fontproperties=zh, fontsize=13, fontweight="bold")
ax.set_xlim(-9, 33); ax.set_ylim(-11, 13)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(__file__.rsplit("/", 1)[0] + "/float_shape_proposal.png", dpi=140)
print("saved float_shape_proposal.png")
