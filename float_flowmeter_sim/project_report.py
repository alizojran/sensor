"""One-page A4 PDF summary of the whole project: 原理→设计→精度→CFD→重设计."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))

PANELS = [
    ("float_flowmeter_accuracy.png", "① 精度模型:锐边浮子 Cd 平、误差小;球形随粘度漂移大"),
    ("float_cfd_flowfield.png", "② CFD 机理(LBM):环隙射流·锐边分离·前后压差顶起浮子"),
    ("float_axisym.png", "③ 轴对称 CFD(管流验证 0.28%):凹坑无影响,环隙主导"),
    ("float_redesign.png", "④ 重新设计:锐边计量浮子 + 整机(去凹坑/去槽)"),
    ("float_mfg_drawing.png", "⑤ 制造图:公差链(环隙 0.2–3.2)·同轴度·锐边"),
    ("float_redesign_cfd.png", "⑥ CFD 复核:新外形(钝面+锐边)按预期工作"),
]

SUMMARY = (
    "对象:ifm SBZ224 — 弹簧活塞式变截面(机电)流量计,1–50 l/min,水/乙二醇/冷却液,G½,200 bar,IP67,IO-Link。\n"
    "原理:流体推动弹簧加载的浮子,位置∝流量,磁感应非接触读出 →4–20 mA/IO-Link;弹簧主导→安装方向无关。\n"
    "方法链:解析精度模型 → 2D LBM 机理 → 参数/对比 A/B → 轴对称 CFD(经 Hagen–Poiseuille 解析解验证,误差 0.28%)。"
)

FINDINGS = (
    "关键发现(证据驱动):\n"
    "• 计量由【环隙=主节流口】主导;前脸细节(锥面槽、前向凹坑)对计量无影响(轴对称 CFD:cup6≈cup2≈blunt,<0.1%)。\n"
    "• 真正的粘度免疫杠杆 = 【锐利计量边】把分离点钉死;钝迎流面阻力高且更 Re 稳定(blunt>rounded)。\n"
    "• 2D 平面算出的“210% 脉动/需加深凹坑”是错误几何的假象——轴对称(正确几何)流动定常(rms≈0)。\n"
    "重设计要点:钝面+90°锐边(Ø12)+短计量柱段+流线锥尾+中心导杆+内置磁铁;去凹坑/去槽(从简、可靠)。"
)

fig = plt.figure(figsize=(8.27, 11.69))           # A4 portrait
gs = fig.add_gridspec(5, 3, height_ratios=[0.7, 1.0, 1.0, 1.0, 0.9],
                      hspace=0.32, wspace=0.06,
                      left=0.04, right=0.96, top=0.97, bottom=0.03)

# title band
axt = fig.add_subplot(gs[0, :]); axt.axis("off")
axt.text(0.5, 0.86, "弹簧活塞式变截面流量计 — 原理 · 设计 · 精度 · CFD · 重设计(一页总览)",
         fontproperties=zh, fontsize=15, fontweight="bold", ha="center", va="top")
axt.text(0.01, 0.55, SUMMARY, fontproperties=zh, fontsize=8.2, ha="left", va="top",
         bbox=dict(boxstyle="round", fc="#f4f7ff", ec="0.6"))

# 6 image panels in rows 1..3 (2 rows x 3) + use row positions
pos = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
for (fn, cap), (r, c) in zip(PANELS, pos):
    ax = fig.add_subplot(gs[r, c]); ax.axis("off")
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        ax.imshow(mpimg.imread(p));
    else:
        ax.text(0.5, 0.5, fn + "\n(缺)", ha="center", va="center", fontproperties=zh)
    ax.set_title(cap, fontproperties=zh, fontsize=7.3, pad=2)

# row 3 spare? we used rows 1,2 for 6 panels -> rows 3,4 for a big redesign + findings
# big redesign panel across row 3
axb = fig.add_subplot(gs[3, :]); axb.axis("off")
pbig = os.path.join(HERE, "float_redesign.png")
if os.path.exists(pbig):
    axb.imshow(mpimg.imread(pbig))
axb.set_title("重新设计(放大):新浮子外形 + 整机装配", fontproperties=zh, fontsize=8, pad=2)

# findings band
axf = fig.add_subplot(gs[4, :]); axf.axis("off")
axf.text(0.01, 0.95, FINDINGS, fontproperties=zh, fontsize=8.4, ha="left", va="top",
         bbox=dict(boxstyle="round", fc="#fffdf0", ec="0.6"))
axf.text(0.99, 0.02, "数据/脚本/全部图见 float_flowmeter_sim/(README §1–9, REDESIGN.md)",
         fontproperties=zh, fontsize=7, ha="right", va="bottom", color="0.4")

out = os.path.join(HERE, "float_meter_report.pdf")
with PdfPages(out) as pdf:
    pdf.savefig(fig)
fig.savefig(os.path.join(HERE, "float_meter_report.png"), dpi=130)
print("saved", out)
