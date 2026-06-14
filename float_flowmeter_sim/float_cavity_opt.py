"""
Cavity (forward pit) sizing study: how do DEPTH and DIAMETER of the front pit
affect (i) mean drag Cd, (ii) flow unsteadiness (drag-fluctuation RMS = pulsation),
and (iii) the oscillation Strouhal number?

Method basis (see chat): forward-cavity flow is governed by the depth/diameter ratio
d/Dp (open-shallow vs deep-trapped-vortex regimes) and the shear-layer/Rossiter
oscillation.  We sweep d and Dp, time-resolve the drag, and report mean + RMS + St.

Grid/geometry as in the cup study, head lengthened (Lh=8) so deep pits fit.
"""
import numpy as np

s = 7.0
nx, ny = 480, 128
cy, wall = 64, 7
x0 = 70
R, Lh, Ltot, rtail = 6.0, 8.0, 30.0, 0.5
Dchar = 2 * R * s
A_front = 2 * R * s

X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
amm = (X - x0) / s
rr = np.abs(Y - cy) / s


def mask(dp, rp):
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


def run(dp, rp, Re=150.0, uLB=0.04, maxIter=16000, warm=6000, tag=""):
    obstacle = mask(dp, rp)
    ff = (~obstacle) & np.roll(obstacle, -1, axis=0)
    fb = (~obstacle) & np.roll(obstacle, +1, axis=0)
    nulb = uLB * Dchar / Re
    omega = 1.0 / (3.0 * nulb + 0.5)
    vel = np.zeros((2, nx, ny))
    vel[0] = uLB * (1 + 1e-4 * np.sin(np.arange(ny) / ny * 2 * np.pi))[None, :]
    fin = equilibrium(np.ones((nx, ny)), vel)
    series = []
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
            series.append(float(p[ff].sum() - p[fb].sum()))
    F = np.array(series)
    if not np.isfinite(F).all():
        print(f"  [{tag}] diverged"); return None
    mean = F.mean()
    Cd = mean / (0.5 * uLB ** 2 * A_front)
    puls = F.std() / abs(mean) * 100.0
    # dominant oscillation frequency via FFT of detrended drag
    g = F - mean
    sp = np.abs(np.fft.rfft(g * np.hanning(len(g))))
    fr = np.fft.rfftfreq(len(g), d=1.0)               # cycles/step
    fpk = fr[1 + np.argmax(sp[1:])] if len(sp) > 2 else 0.0
    St = fpk * (2 * rp * s) / uLB                      # Strouhal on pit diameter
    print(f"  [{tag}] d={dp} Dp={2*rp} d/Dp={dp/(2*rp):.2f}  Cd={Cd:.2f}  puls={puls:.1f}%  St={St:.3f}")
    return dict(dp=dp, rp=rp, Cd=Cd, puls=puls, St=St, mean=mean)


if __name__ == "__main__":
    print("=== DEPTH sweep (Dp=Ø8) ===")
    depth = [run(d, 4.0, tag="depth") for d in (1.0, 2.0, 3.0, 4.0, 6.0)]
    print("=== DIAMETER sweep (d=3) ===")
    dia = [run(3.0, rp, tag="dia") for rp in (3.0, 4.0, 5.0)]
    depth = [r for r in depth if r]; dia = [r for r in dia if r]
    np.savez(__file__.rsplit("/", 1)[0] + "/cavity_opt.npz",
             d_dp=[r["dp"] for r in depth], d_Cd=[r["Cd"] for r in depth],
             d_puls=[r["puls"] for r in depth], d_St=[r["St"] for r in depth],
             a_Dp=[2 * r["rp"] for r in dia], a_Cd=[r["Cd"] for r in dia],
             a_puls=[r["puls"] for r in dia], a_St=[r["St"] for r in dia])
    print("saved cavity_opt.npz")
