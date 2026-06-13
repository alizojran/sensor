"""
Engineering cross-section drawing of the sharp-edged, viscosity-immune float
designed for the spring-loaded variable-area flow meter (ifm SBZ224 style).

Left  : float detail (longitudinal section) with dimensions and feature callouts.
Right : the float working inside the tapered bore (seat / spring / magnet / flow).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Arc
from matplotlib import font_manager as fm

# ---- Chinese font ----
zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------------------
# Float outer contour (right half), bottom -> top   [x = radius mm, y = axial mm]
# --------------------------------------------------------------------------------------
R_EDGE = 6.0          # Ø12 metering edge
R_BORE = 1.6          # Ø3.2 central guide bore
R_BODY = 4.5          # guide-body radius
Y_EDGE = 5.2          # height of metering edge (top of 60 deg cone)
Y_BODY = 22.0         # top of cylindrical body
Y_TOP  = 30.0         # tail tip

# right-half material polygon (between bore wall and outer contour)
right = [
    (R_BORE, 0.0), (3.0, 0.0),        # bottom annular face
    (R_EDGE, Y_EDGE),                 # 60 deg sealing/metering cone -> sharp edge
    (R_BODY, Y_EDGE),                 # sharp 90 deg back face (the metering rim)
    (R_BODY, Y_BODY),                 # cylindrical guide body
    (R_BORE, Y_TOP),                  # streamlined tail
    (R_BORE, 0.0),                    # down the guide bore -> close
]
left = [(-x, y) for (x, y) in right]
float_full = right + left[::-1]


def draw_float(ax, y0=0.0, hatch="////", fc="#d9d9d9", alpha=1.0, ec="k", lw=1.6):
    poly = [(x, y + y0) for (x, y) in float_full]
    ax.add_patch(Polygon(poly, closed=True, facecolor=fc, edgecolor=ec,
                         hatch=hatch, lw=lw, alpha=alpha, zorder=3))
    # ring magnet (both sides), embedded in the body
    for sx in (+1, -1):
        ax.add_patch(Rectangle((sx * 2.4 if sx > 0 else -4.0, 8.0 + y0),
                               1.6, 10.0, facecolor="#c44", edgecolor="k",
                               lw=0.8, zorder=4))


def dim_h(ax, x1, x2, y, text, dy=0.0):
    ax.annotate("", (x2, y), (x1, y),
                arrowprops=dict(arrowstyle="<->", color="b", lw=1.2))
    ax.text((x1 + x2) / 2, y + dy, text, ha="center", va="bottom",
            color="b", fontsize=9, fontproperties=zh)


def dim_v(ax, x, y1, y2, text):
    ax.annotate("", (x, y2), (x, y1),
                arrowprops=dict(arrowstyle="<->", color="b", lw=1.2))
    ax.text(x + 0.4, (y1 + y2) / 2, text, ha="left", va="center",
            color="b", fontsize=9, rotation=90, fontproperties=zh)


def callout(ax, xy, xytext, text):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.5, fontproperties=zh,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="k", lw=1.1))


# ======================================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 9.5),
                               gridspec_kw=dict(width_ratios=[1.0, 1.05]))

# --------------------------------------------------------------------------------------
# (A) FLOAT DETAIL
# --------------------------------------------------------------------------------------
draw_float(axA)
axA.plot([0, 0], [-2, Y_TOP + 2], "k-.", lw=0.8, zorder=1)          # centerline
axA.text(0, Y_TOP + 2.3, "回转轴线", ha="center", fontsize=8, fontproperties=zh)

# dimensions
dim_h(axA, -R_EDGE, R_EDGE, Y_EDGE + 6.0, "Ø12  计量边", dy=0.2)
axA.plot([-R_EDGE, -R_EDGE], [Y_EDGE, Y_EDGE + 6.0], "b:", lw=0.7)
axA.plot([R_EDGE, R_EDGE], [Y_EDGE, Y_EDGE + 6.0], "b:", lw=0.7)
dim_h(axA, -R_BORE, R_BORE, -2.4, "Ø3.2 导向孔")
dim_v(axA, R_EDGE + 6.5, 0.0, Y_TOP, "总长 ≈ 30")

# 60 deg cone angle
axA.add_patch(Arc((0, 0), 7.0, 7.0, angle=0, theta1=60, theta2=120,
                  color="b", lw=1.0))
axA.text(0, 4.6, "≈60°", ha="center", color="b", fontsize=9, fontproperties=zh)

# feature callouts
callout(axA, (R_EDGE, Y_EDGE), (R_EDGE + 1.5, Y_EDGE - 3.0),
        "锐利计量边\n(分离点固定→Cd 不随\n 粘度/Re 变化)")
callout(axA, (-R_EDGE, Y_EDGE - 1.2), (-13.0, Y_EDGE + 1.0),
        "60° 锥面\n(零流量落座=止回密封)")
callout(axA, (3.2, 13.0), (R_BODY + 1.2, 14.0), "环形磁铁\n(磁感应位置)")
callout(axA, (-3.0, 26.0), (-13.5, 24.0), "流线型尾部\n(抑制涡激振动)")
callout(axA, (R_BORE, 18.0), (R_BODY + 1.2, 21.5), "中心导杆孔\n(定心/防转)")

axA.set_title("(A) 锐边 viscosity-immune 浮子  —  纵剖面", fontproperties=zh, fontsize=13)
axA.set_xlim(-15, 15)
axA.set_ylim(-5, 35)
axA.set_aspect("equal")
axA.axis("off")

# --------------------------------------------------------------------------------------
# (B) FLOAT IN THE TAPERED BORE (working principle)
# --------------------------------------------------------------------------------------
HOUS_T, HOUS_B = 56.0, -8.0          # housing top / bottom (mm)
WALL = 13.0                          # housing half-width (outer)
y0 = 14.0                            # float lift (mid stroke)

# tapered bore walls: radius grows with height (gap 0.2 -> ~3.2 mm, exaggerated x2 for clarity)
def bore_r(y):
    g = 0.4 + (y - (y0)) * 0.16      # visual gap law
    return R_EDGE + np.clip(g, 0.4, 6.5)

ys = np.linspace(y0 - 2, HOUS_T, 50)
xs = bore_r(ys)
# housing solid (outer rectangle minus bore) drawn as two side bands
axB.add_patch(Rectangle((-WALL, HOUS_B), WALL - (R_EDGE + 0.4), HOUS_T - HOUS_B,
                        facecolor="#eaeaea", edgecolor="k", lw=1.4, zorder=1))
axB.add_patch(Rectangle((R_EDGE + 0.4, HOUS_B), WALL - (R_EDGE + 0.4), HOUS_T - HOUS_B,
                        facecolor="#eaeaea", edgecolor="k", lw=1.4, zorder=1))
# tapered bore inner walls
axB.plot(xs, ys, "k-", lw=1.4, zorder=2)
axB.plot(-xs, ys, "k-", lw=1.4, zorder=2)

# seat ring (conical) at bottom -> check-valve seat
axB.plot([R_EDGE + 0.4, 3.0], [y0 - 2, y0 - 6.5], "k-", lw=2.2, zorder=2)
axB.plot([-(R_EDGE + 0.4), -3.0], [y0 - 2, y0 - 6.5], "k-", lw=2.2, zorder=2)
axB.text(8.5, y0 - 6.5, "阀座\n(止回)", fontproperties=zh, fontsize=9, va="center")

# the float at mid stroke (no dimensions/magnet hatch lighter)
draw_float(axB, y0=y0, hatch="", fc="#cfd8e6")

# guide rod (fixed, through bore) -- behind the float so it only shows through the bore hole
axB.add_patch(Rectangle((-1.4, HOUS_B + 2), 2.8, HOUS_T - HOUS_B - 6,
                        facecolor="#888", edgecolor="k", lw=0.6, zorder=2))
axB.text(-3.0, HOUS_T - 3, "导杆", fontproperties=zh, fontsize=9, ha="center")

# spring above the float (pushes float down; flow pushes up)
sy = np.linspace(y0 + Y_TOP + 1, HOUS_T - 4, 220)
sx = 2.9 * np.sin(2 * np.pi * 7 * (sy - sy[0]) / (sy[-1] - sy[0]))
axB.plot(sx, sy, color="#555", lw=1.6, zorder=4)
axB.text(4.2, (sy[0] + sy[-1]) / 2, "弹簧\n(复位)", fontproperties=zh, fontsize=9, va="center")

# flow arrows (up): inlet, through gap, outlet
for (x, ya, yb) in [(0, HOUS_B + 1, y0 - 7),            # inlet
                    (R_EDGE + 1.6, y0 + 1, y0 + 9),      # through annular gap (right)
                    (-(R_EDGE + 1.6), y0 + 1, y0 + 9),   # gap (left)
                    (0, HOUS_T - 9, HOUS_T - 1)]:        # outlet
    axB.add_patch(FancyArrowPatch((x, ya), (x, yb), arrowstyle="-|>",
                                  mutation_scale=16, color="#1f77b4", lw=2.2, zorder=6))
axB.text(0, HOUS_B - 1.5, "流入", fontproperties=zh, fontsize=10, ha="center", color="#1f77b4")
axB.text(0, HOUS_T + 1.5, "流出", fontproperties=zh, fontsize=10, ha="center", color="#1f77b4")
axB.text(R_EDGE + 3.0, y0 + 7.5, "可变环隙\n(锥孔)", fontproperties=zh, fontsize=8.5,
         color="#1f77b4", va="center")

# magnetic sensor outside the wall + link to magnet
axB.add_patch(Rectangle((WALL + 0.4, y0 + 8), 5.2, 6.0, facecolor="#ffe08a",
                        edgecolor="k", lw=1.0, zorder=6))
axB.annotate("", (WALL + 0.4, y0 + 13.0), (3.2, y0 + 13.0),
             arrowprops=dict(arrowstyle="->", color="#c44", lw=1.3, ls="--"))
axB.text(WALL + 3.0, y0 + 17.5, "磁感应\n位置传感器", fontproperties=zh, fontsize=8.5,
         ha="center", va="bottom")
axB.text(WALL + 3.0, y0 + 4.5, "位置 ∝ 流量\n→4–20 mA / IO-Link",
         fontproperties=zh, fontsize=8, ha="center", va="top")

axB.set_title("(B) 在锥孔中的工作原理", fontproperties=zh, fontsize=13)
axB.set_xlim(-16, 22)
axB.set_ylim(HOUS_B - 5, HOUS_T + 6)
axB.set_aspect("equal")
axB.axis("off")

fig.suptitle("锐边 viscosity-immune 浮子设计(SBZ224 型弹簧活塞变截面流量计)",
             fontproperties=zh, fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = __file__.rsplit("/", 1)[0] + "/float_design_drawing.png"
fig.savefig(out, dpi=150)
print("saved:", out)
