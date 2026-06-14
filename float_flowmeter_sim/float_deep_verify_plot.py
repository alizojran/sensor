"""Plot deep-cavity verification (reads deep_verify.npz + cup_study.npz refs)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(family="WenQuanYi Zen Hei")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
HERE = __file__.rsplit("/", 1)[0]
D = np.load(HERE + "/deep_verify.npz")
uLB = float(D["uLB"])
Dp = 8.0 * 7.0          # pit diameter in cells (Ø8 * s)

try:
    C = np.load(HERE + "/cup_study.npz")
    ref = {"钝(无坑)": float(C["beta_blunt"]), "圆头": float(C["beta_rounded"])}
except Exception:
    ref = {}

CFG = {"d2": ("Ø8×深2 (d/Dp=0.25, 原)", "tab:red"),
       "d6": ("Ø8×深6 (d/Dp=0.75, 推荐)", "tab:green")}

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.8))

# ---- (A) viscosity immunity: Cd/Cd_inf vs Re ----
ReA = np.logspace(1.6, 4.8, 300)
for k, (lab, col) in CFG.items():
    Re = np.asarray(D[f"{k}_Re"], float); Cd = np.asarray(D[f"{k}_Cd"], float)
    b, a = float(D[f"{k}_beta"]), float(D[f"{k}_Cinf"])
    axA.plot(Re, Cd / a, "o", color=col, ms=9)
    axA.plot(ReA, (1 + b / np.sqrt(ReA)) / (1 + b / np.sqrt(ReA[-1])), "-", color=col, lw=2,
             label=f"{lab}  β={b:.2f}")
for lab, b in ref.items():
    axA.plot(ReA, (1 + b / np.sqrt(ReA)) / (1 + b / np.sqrt(ReA[-1])), "--", lw=1.4,
             color="0.5", label=f"{lab}(参考) β={b:.2f}")
axA.set_xscale("log")
axA.set_xlabel("浮子雷诺数 Re", fontproperties=zh)
axA.set_ylabel("Cd / Cd∞ (归一化)", fontproperties=zh)
axA.set_title("(A) 粘度免疫:Cd 随 Re 的平坦度(越平越好)", fontproperties=zh, fontsize=11.5)
axA.grid(True, which="both", alpha=0.3); axA.legend(prop=zh, fontsize=8.5)

# ---- (B) drag time history ----
for k, (lab, col) in CFG.items():
    F = np.asarray(D[f"{k}_ts"], float)
    rms = F.std() / abs(F.mean()) * 100
    axB.plot(np.arange(len(F)), F / F.mean(), color=col, lw=0.8,
             label=f"{lab}  RMS={rms:.0f}%")
axB.axhline(1, color="k", lw=0.6, ls=":")
axB.set_xlabel("时间步", fontproperties=zh)
axB.set_ylabel("瞬时阻力 / 平均", fontproperties=zh)
axB.set_title("(B) 阻力时程:深腔脉动远小", fontproperties=zh, fontsize=11.5)
axB.legend(prop=zh, fontsize=9, loc="upper right")

# ---- (C) spectrum ----
for k, (lab, col) in CFG.items():
    F = np.asarray(D[f"{k}_ts"], float)
    g = (F - F.mean()) * np.hanning(len(F))
    sp = np.abs(np.fft.rfft(g)); fr = np.fft.rfftfreq(len(F), d=1.0)
    St = fr * Dp / uLB
    axC.plot(St, sp / sp.max() if sp.max() > 0 else sp, color=col, lw=1.2, label=lab)
axC.set_xlim(0, 8)
axC.set_xlabel("Strouhal 数  St = f·Dp/U", fontproperties=zh)
axC.set_ylabel("归一化幅值", fontproperties=zh)
axC.set_title("(C) 脉动频谱:浅腔有强峰,深腔平", fontproperties=zh, fontsize=11.5)
axC.grid(True, alpha=0.3); axC.legend(prop=zh, fontsize=9)

fig.suptitle("深腔(Ø8×深6,d/Dp=0.75)验证:粘度免疫保住 + 脉动大幅降低",
             fontproperties=zh, fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(HERE + "/float_deep_verify.png", dpi=140)
print("saved float_deep_verify.png")
for k in CFG:
    F = np.asarray(D[f"{k}_ts"], float)
    print(f"{k}: beta={float(D[f'{k}_beta']):.3f} Cinf={float(D[f'{k}_Cinf']):.2f} "
          f"RMS={F.std()/abs(F.mean())*100:.1f}%  Cd_sweep={np.asarray(D[f'{k}_Cd'])}")
