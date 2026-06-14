"""Detailed multi-page A4 PDF report of the whole project.
原理 → 设计 → 精度 → CFD机理 → A/B修正 → 轴对称定论 → 重设计 → 制造图 → 复核结论."""
import os
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
A4 = (8.27, 11.69)


def img(ax, fn):
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        ax.imshow(mpimg.imread(p))
    else:
        ax.text(0.5, 0.5, fn + " (缺)", ha="center", va="center", fontproperties=zh)
    ax.axis("off")


def footer(fig, n):
    fig.text(0.5, 0.018, f"弹簧活塞式变截面流量计 · 详细报告        — {n} —",
             ha="center", fontsize=7.5, color="0.45", fontproperties=zh)


def content(pdf, n, title, text, imgs):
    """imgs: list of (filename, caption). 1 or 2 images, stacked."""
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=14.5,
             fontweight="bold", fontproperties=zh)
    fig.add_axes([0.06, 0.945, 0.88, 0.001]).axis("off")
    fig.text(0.065, 0.93, text, ha="left", va="top", fontsize=9.4, linespacing=1.55,
             fontproperties=zh)
    nlines = text.count("\n") + 1
    img_top = 0.93 - nlines * 0.0175 - 0.02
    if len(imgs) == 1:
        ax = fig.add_axes([0.06, 0.05, 0.88, img_top - 0.05]); img(ax, imgs[0][0])
        ax.set_title(imgs[0][1], fontproperties=zh, fontsize=8.5, pad=3)
    else:
        h = (img_top - 0.06) / 2
        for k, (fn, cap) in enumerate(imgs[:2]):
            b = 0.06 + (1 - k) * (h + 0.015)
            ax = fig.add_axes([0.07, b, 0.86, h - 0.022]); img(ax, fn)
            ax.set_title(cap, fontproperties=zh, fontsize=8, pad=2)
    footer(fig, n)
    pdf.savefig(fig); plt.close(fig)


# ---------------- cover ----------------
def cover(pdf):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.93, "弹簧活塞式变截面(机电)流量计", ha="center", fontsize=20,
             fontweight="bold", fontproperties=zh)
    fig.text(0.5, 0.89, "原理 · 设计 · 精度 · CFD · 重设计 —— 详细报告", ha="center",
             fontsize=13, fontproperties=zh)
    fig.text(0.5, 0.855, "对象:ifm SBZ224(带止回阀及显示屏的流量计)", ha="center",
             fontsize=10.5, color="0.3", fontproperties=zh)
    obj = (
        "一、对象与原理\n"
        "  ifm SBZ224 是弹簧活塞式变截面(机电)流量计:流体推动一个弹簧加载的浮子,\n"
        "  浮子位置正比于流量,由内置磁铁经外部磁感应非接触读出 → 4–20 mA / IO-Link。\n"
        "  力平衡为 ΔP·A_f = 弹簧力;弹簧主导使其【安装方向无关】。零流量时浮子落座=止回。\n"
        "  量程 1–50 l/min(水),动态 1:50;介质 水/乙二醇/冷却液;G½;-10…100 °C;200 bar;IP67。\n\n"
        "二、方法链(从原理到验证级 CFD)\n"
        "  解析精度模型 → 2D LBM 机理 → 槽/凹坑 A/B 与尺寸研究 → 轴对称 N–S CFD\n"
        "  (经 Hagen–Poiseuille 解析解验证,误差 0.28%)→ 证据驱动重设计 → 制造图 + 复核。"
    )
    fig.text(0.08, 0.80, obj, ha="left", va="top", fontsize=10, linespacing=1.6,
             fontproperties=zh)
    concl = (
        "主要结论(证据驱动)\n"
        "①  计量由【环隙=主节流口】主导;前脸细节(锥面槽、前向凹坑)对计量无影响\n"
        "     —— 轴对称 CFD:cup6 ≈ cup2 ≈ blunt,阻力差 < 0.1%。\n"
        "②  真正的粘度免疫杠杆 = 【锐利计量边】把分离点钉死;钝迎流面阻力高且更 Re 稳定。\n"
        "③  2D 平面算出的“210% 脉动 / 需加深凹坑”是错误几何的假象——轴对称流动定常(rms≈0)。\n"
        "④  重设计:钝面 + 90° 锐边 Ø12 + 短计量柱段 + 流线锥尾 + 中心导杆 + 内置磁铁;\n"
        "     去掉凹坑/锥面槽(从简、可靠);复核确认按预期工作。"
    )
    fig.text(0.08, 0.50, concl, ha="left", va="top", fontsize=9.6, linespacing=1.6,
             fontproperties=zh, bbox=dict(boxstyle="round", fc="#fffdf0", ec="0.5", pad=0.8))
    toc = ("目录\n  1 原理与背景    2 浮子形状要求与精度模型    3 2D CFD 流动机理\n"
           "  4 A/B 研究与修正(槽/凹坑)    5 轴对称验证级 CFD(定论)\n"
           "  6 重新设计(证据驱动)    7 制造图与公差链    8 CFD 复核与总结")
    fig.text(0.08, 0.22, toc, ha="left", va="top", fontsize=9.4, linespacing=1.6,
             fontproperties=zh, bbox=dict(boxstyle="round", fc="#f4f7ff", ec="0.6", pad=0.8))
    fig.text(0.5, 0.05, "代码 / 数据 / 全部图见 float_flowmeter_sim/(README §1–9, REDESIGN.md)",
             ha="center", fontsize=8, color="0.45", fontproperties=zh)
    pdf.savefig(fig); plt.close(fig)


with PdfPages(os.path.join(HERE, "float_meter_report_full.pdf")) as pdf:
    cover(pdf)

    content(pdf, 2, "1  原理与背景(SBZ224)",
        "测量原理:弹簧活塞式变截面(机电)。流体自下而上推动被弹簧顶住的浮子(活塞),\n"
        "流量越大、浮子被顶得越高;弹簧建立力平衡使【位置 ↔ 流量】一一对应。浮子内置磁铁,\n"
        "外壳外的磁感应元件非接触读出其轴向位置,换算为流量 → 4–20 mA / 开关 / 频率 / IO-Link。\n"
        "介质与电子件由壳壁隔开(磁耦合)→ 可耐 200 bar、316L 接液、FKM 密封。\n"
        "零流量时浮子落座于阀座 = 内置止回。下图为浮子纵剖面与在锥孔中的工作原理。",
        [("float_design_drawing.png", "浮子纵剖面(左)与锥孔工作原理(右)")])

    content(pdf, 3, "2  浮子形状设计要求 与 精度模型",
        "形状要求:① 锐利计量边——把分离点钉死在边缘,使 Cd 与 Re/粘度无关(粘度免疫);\n"
        "② 导向/定心(导杆或导筋)防偏心与卡滞;③ 锥面+阀座=止回密封;④ 轴对称磁铁→位置线性;\n"
        "⑤ 流线尾抑制涡激;⑥ 缝隙不过小+200 µm 前置过滤防颗粒;⑦ 配合锥孔做 1:50 线性化。\n"
        "精度模型:Q = Cd·A_a(h)·√(2ΔP(h)/ρ),力平衡 ΔP·A_f = F0 + k·h;单点 span 标定。\n"
        "下图:锐边浮子 Cd 随 Re 平、误差小;球形浮子随粘度漂移大(水 vs 乙二醇)。",
        [("float_flowmeter_accuracy.png", "精度模型:读数误差 vs 流量(锐边 vs 球形;水/乙二醇)")])

    content(pdf, 4, "3  2D CFD 流动机理(格子玻尔兹曼 LBM)",
        "用 D2Q9 LBM 在子午面解流场(教学/机理用)。可见三件事:流体经【可变环隙】加速到\n"
        "约 6.5× 入口速度;在【锐边分离】并在尾部形成回流;迎流面高压、环隙/尾部低压,\n"
        "前后【压差 ΔP】正是顶起浮子的力,与解析力平衡 ΔP·A_f = 弹簧力 对应。\n"
        "注:2D 平面是无限长狭缝近似,几何不等于真实回转体——后续轴对称会纠正其定量结论。",
        [("float_cfd_flowfield.png", "2D LBM 流场:速度+流线 / 涡量 / 压力")])

    content(pdf, 5, "4  A/B 研究与修正(锥面槽 / 前向凹坑)",
        "锥面环形槽:A/B 同条件对比显示两者 ΔP 逐点吻合、拟合 β 不变 → 对计量【无影响】\n"
        "(槽在上游低速区,够不到主节流口)。前向凹坑:2D 一度显示“有益”(阻力高、β 小),\n"
        "且凹坑致【非定常】,据此做了深径比 d/Dp 尺寸研究(2D 给出 d/Dp≥0.75 降脉动)。\n"
        "★ 重要:这些 2D 结论随后被轴对称(正确几何)CFD 推翻——见第 5 节。下面两图为 2D 阶段结果。",
        [("float_cup_accuracy.png", "凹坑/钝/圆头 2D 对比(后被轴对称修正)"),
         ("float_cavity_opt.png", "2D 凹坑尺寸研究:脉动 vs 深径比(后证为 2D 假象)")])

    content(pdf, 6, "5  轴对称验证级 CFD(定论)",
        "全 3D 在 CPU/numpy 不现实(单次~1h、测 β 十几小时)。对回转体的正确且可行做法=轴对称 N–S。\n"
        "自建交错网格分步投影求解器,先用 Hagen–Poiseuille 解析抛物线【验证:最大误差 0.28%、\n"
        "中心/平均=1.993】,再扫 cup6/cup2/blunt/rounded(无散度 1e-15,CV/表面两法一致)。结论:\n"
        "① 流动【定常】(rms≈0)→ 2D 的 210% 脉动是平面假象;② cup6≈cup2≈blunt 阻力差<0.1%\n"
        "→ 前向凹坑【对计量无影响】;③ rounded 阻力明显更低 → 钝 vs 圆 才重要,凹坑不重要;\n"
        "④ Re 50–400 内 Cd∝Re^-0.65、各形状 Re 敏感度相近,无“粘度免疫平台”。",
        [("float_axisym.png", "轴对称 CFD:管流验证(左)/ Cd(Re)(中)/ cup6 流场(右)")])

    content(pdf, 7, "6  重新设计(证据驱动)",
        "依据上述验证结论,把功夫放在真正决定计量的【主节流口】,去掉被证无效的前脸花样。\n"
        "新浮子:钝平迎流面(阻力高/更 Re 稳定)+ 90° 锐利计量边 Ø12(粘度免疫杠杆)+ 短计量柱段\n"
        "(定义环隙、线性化)+ 流线锥尾(尾流稳定/导向)+ 中心导杆孔(定心防转)+ 内置环形磁铁;\n"
        "【不设】前向凹坑、【不设】锥面槽。整机:渐扩锥孔 + 阀座(止回)+ 弹簧(方向无关)+ 导杆 +\n"
        "磁致伸缩/AMR 位置传感 → 4–20 mA/IO-Link。沿用 316L/FKM、200 bar、IP67、G½。",
        [("float_redesign.png", "重新设计:新浮子外形(左)+ 整机装配(右)")])

    content(pdf, 8, "7  制造图与公差链",
        "关键尺寸:计量边 Ø12 -0/-0.02(基准B)、导杆孔 Ø3.2 H7(基准A)、计量柱段 3±0.05、总长 26±0.2。\n"
        "公差链(零位环隙,半径向):环隙=(D孔-D边)/2=(12.40-12.00)/2=0.20;配合孔 Ø12.40 H7 +\n"
        "边 Ø12 -0/-0.02 → 直径贡献 0.20…0.23;导杆 H7/g6 → 偏心 ≤0.012;满量程 3.2 由锥孔渐扩给出。\n"
        "绝对环隙散差由单点【校准系数】吸收;线性/重复性靠【同轴度 ≤0.02 TIR + 锐边 ≤R0.05、去毛刺】。\n"
        "锐边是关键特征(决定粘度免疫);材料 316L,磁铁 NdFeB 全密封;配对锥孔 Ø12.40 H7、导杆 Ø3.2 g6。",
        [("float_mfg_drawing.png", "重新设计浮子 制造图(尺寸/公差/公差链/锐边规格)")])

    content(pdf, 9, "8  CFD 复核 与 总结",
        "复核:用已验证的轴对称求解器复跑【新外形】(blunt 面+锐边+柱段+锥尾,无凹坑)。\n"
        "结果:流动【定常】(rms≤0.05%)、CV/表面两法吻合、Cd∝Re^-0.6;流场确认【钝面驻点(高压)+\n"
        "锐边分离(钉死)】按预期工作 → 新外形成立。\n"
        "总结:① 环隙(主节流)决定计量,前脸细节次要;② 锐利计量边是粘度免疫的真正杠杆;\n"
        "③ 验证级 CFD 纠正了 2D 的多处假象(脉动、凹坑收益);④ 重设计更简单、可靠、可制造。\n"
        "局限:轴对称不含周向失稳;绝对 Cd 受高阻塞影响;凹坑的现实价值多在机械层面(定心/防卡/旋转)。",
        [("float_redesign_cfd.png", "CFD 复核:新外形 Cd(Re)(左)+ 轴对称流场(右)")])

print("saved float_meter_report_full.pdf")
