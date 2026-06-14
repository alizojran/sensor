"""Re-import the exported STEP solids (validity check) and render a 3D preview."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import font_manager as fm
import cadquery as cq

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
HERE = os.path.dirname(os.path.abspath(__file__))


def tess(shape, tol=0.3):
    vs, ts = shape.tessellate(tol)
    V = np.array([[v.x, v.y, v.z] for v in vs])
    T = np.array(ts)
    return V, T


def show(ax, V, T, color, title):
    polys = V[T]
    pc = Poly3DCollection(polys, facecolor=color, edgecolor="k", linewidths=0.1, alpha=1.0)
    ax.add_collection3d(pc)
    mins = V.min(0); maxs = V.max(0); ctr = (mins + maxs) / 2; rng = (maxs - mins).max() / 2
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng); ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_title(title, fontproperties=zh, fontsize=12)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1)); ax.axis("off")


# re-import (validity) + tessellate
fb = cq.importers.importStep(os.path.join(HERE, "float_body.step"))
hb = cq.importers.importStep(os.path.join(HERE, "housing_body.step"))
print("re-imported OK: float vol=%.0f  housing vol=%.0f"
      % (fb.val().Volume(), hb.val().Volume()))
# half-cut the housing to reveal the bore
big = cq.Workplane("XY").box(200, 200, 200).translate((0, -100, 0))
hb_cut = hb.cut(big)

Vf, Tf = tess(fb.val())
Vh, Th = tess(hb_cut.val())

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
show(ax1, Vf, Tf, "#9fb0c8", "float_body.step(回转实体,中心 Ø3 导杆孔)")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
show(ax2, Vh, Th, "#cdbfae", "housing_body.step(半剖:锥孔/阀座/进出口)")
fig.suptitle("STEP 实体预览(已重新导入校验有效)", fontproperties=zh, fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "float_cad_preview.png"), dpi=140)
print("saved float_cad_preview.png")
