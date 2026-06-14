"""
Verify the recommended DEEP cavity (Ø8 x depth 6, d/Dp=0.75) vs the original shallow
one (Ø8 x depth 2, d/Dp=0.25):
  (1) viscosity immunity  -> Re sweep [50,100,200] -> fit Cd = Cd_inf(1+beta/sqrt(Re))
  (2) unsteadiness        -> long drag time series at Re=150 (-> RMS + spectrum)

Same geometry family as the sizing sweep (Lh=8 so depth 6 fits).
"""
import numpy as np
from scipy.optimize import curve_fit

s = 7.0
nx, ny = 480, 128
cy, wall = 64, 7
x0 = 70
R, Lh, Ltot, rtail = 6.0, 8.0, 30.0, 0.5
rp = 4.0                                   # Ø8
Dchar = 2 * R * s
A_front = 2 * R * s

X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
amm = (X - x0) / s
rr = np.abs(Y - cy) / s


def mask(dp):
    head = (amm >= 0) & (amm < Lh)
    tail = (amm >= Lh) & (amm <= Ltot)
    Ro = np.zeros_like(amm)
    Ro[head] = R
    Ro[tail] = R - (R - rtail) * (amm[tail] - Lh) / (Ltot - Lh)
    env = (amm >= 0) & (amm <= Ltot) & (rr <= Ro)
    pit = (amm >= 0) & (amm < dp) & (rr < rp)
    return env & ~pit


v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
vf = v.astype(float)
t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
i1 = np.array([6, 7, 8]); i2 = np.array([3, 4, 5]); i3 = np.array([0, 1, 2])
opp = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])


def equilibrium(rho, u):
    cu = 3.0 * np.tensordot(vf, u, axes=([1], [0]))
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * t[:, None, None] * (1 + cu + 0.5 * cu * cu - usqr[None])


def run(dp, Re, uLB=0.04, maxIter=16000, warm=6000, keep=False, tag=""):
    obstacle = mask(dp)
    ff = (~obstacle) & np.roll(obstacle, -1, axis=0)
    fb = (~obstacle) & np.roll(obstacle, +1, axis=0)
    nulb = uLB * Dchar / Re
    omega = 1.0 / (3.0 * nulb + 0.5)
    vel = np.zeros((2, nx, ny))
    vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]
    fin = equilibrium(np.ones((nx, ny)), vel)
    ser = []
    for it in range(maxIter):
        fin[i1, -1, :] = fin[i1, -2, :]
        rho = np.sum(fin, axis=0)
        u = np.tensordot(vf.T, fin, axes=([1], [0])) / rho[None]
        u[:, 0, :] = vel[:, 0, :]
        rho[0, :] = (1 / (1 - u[0, 0, :])) * (np.sum(fin[i2, 0, :], 0) + 2 * np.sum(fin[i1, 0, :], 0))
        feq = equilibrium(rho, u)
        fin[i3, 0, :] = feq[i3, 0, :] + fin[i1[::-1], 0, :] - feq[i1[::-1], 0, :]
        fout = fin - omega * (fin - feq)
        fout[:, obstacle] = fin[opp][:, obstacle]
        for i in range(9):
            fin[i] = np.roll(np.roll(fout[i], v[i, 0], axis=0), v[i, 1], axis=1)
        if it >= warm:
            p = (rho - 1.0) / 3.0
            ser.append(float(p[ff].sum() - p[fb].sum()))
    F = np.array(ser)
    Cd = F.mean() / (0.5 * uLB ** 2 * A_front)
    puls = F.std() / abs(F.mean()) * 100
    print(f"  [{tag}] d={dp} Re={Re:.0f} Cd={Cd:.2f} puls={puls:.1f}%")
    return (Cd, F) if keep else (Cd, None)


def model(Re, a, b):
    return a * (1 + b / np.sqrt(Re))


if __name__ == "__main__":
    out = {}
    for dp in (2.0, 6.0):
        Res = [50.0, 100.0, 200.0]
        Cds, series150 = [], None
        for Re in Res:
            Cd, F = run(dp, Re, keep=(Re == 150.0), tag=f"d{int(dp)}")
            Cds.append(Cd)
        # dedicated long Re=150 run for the time series / spectrum
        _, series150 = run(dp, 150.0, maxIter=18000, warm=4000, keep=True, tag=f"d{int(dp)}-ts")
        popt, _ = curve_fit(model, np.array(Res), np.array(Cds), p0=[min(Cds), 1.0], maxfev=20000)
        out[f"d{int(dp)}_Re"] = Res
        out[f"d{int(dp)}_Cd"] = Cds
        out[f"d{int(dp)}_beta"] = float(popt[1])
        out[f"d{int(dp)}_Cinf"] = float(popt[0])
        out[f"d{int(dp)}_ts"] = series150
        print(f"  >>> d={dp}: beta={popt[1]:.3f} Cinf={popt[0]:.3f}")
    np.savez(__file__.rsplit("/", 1)[0] + "/deep_verify.npz", uLB=0.04, **out)
    print("saved deep_verify.npz")
