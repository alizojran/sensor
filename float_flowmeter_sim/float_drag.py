"""
Axial drag on the immersed float, two independent methods.

(1) Control-volume momentum balance (robust, preferred).
    Take a CV = full annulus cross-section between two axial planes z1 (upstream of
    nose) and z2 (downstream of tail), radius 0..R_out.  Steady axial momentum:

      sum of axial forces on fluid in CV = net axial momentum OUTFLUX

      [p1*A1 - p2*A2]  +  [viscous_z on inlet/outlet (small)]  - F_wall_shear
        - F_float_on_fluid  =  Mdot_out - Mdot_in

    The drag the FLUID exerts on the FLOAT  =  -F_float_on_fluid  (reaction).
    Rearranged, the axial force the float exerts ON the fluid is negative of drag:

      F_drag = (p1*A1 - p2*A2) + (Mom_in - Mom_out) - F_wall_shear

    where Mom = integral over annulus of rho*w^2 dA (2*pi*r dr), p*A = integral p dA,
    F_wall_shear = integral over the outer wall (r=R_out) of  -mu * dw/dr * dz * 2*pi*R
    (the wall pulls fluid back; it is a force on the fluid, so appears with sign so that
     it is subtracted as a loss).  We compute it directly and include it.

(2) Surface integration (form drag + viscous skin drag) over the float boundary.
    form: sum over solid-fluid faces with axial-facing normal of  p * (face area, axial).
    skin: sum over solid-fluid faces with radial/axial tangential of  mu * (shear) * area.
    For a staggered grid this is fiddly; we implement an approximate but consistent
    version and use it as a CROSS-CHECK on method (1).
"""
import numpy as np


def drag_cv(sol, i1, i2):
    """
    Control-volume momentum drag between axial face planes i1 (upstream) and i2
    (downstream).  i1, i2 are w-face indices (0..nz).  Returns F_drag (axial force
    on the float by the fluid), and a breakdown dict.
    rho=sol.rho, mu = rho*nu.
    """
    nz, nr = sol.nz, sol.nr
    dz, dr = sol.dz, sol.dr
    r_c = sol.r_c
    rho, mu = sol.rho, sol.rho * sol.nu
    w, v, p = sol.w, sol.v, sol.p

    dA = 2 * np.pi * r_c * dr          # annular ring area per radial cell (nr,)

    # --- pressure force on the two end planes (use cell-centre p adjacent to the face)
    # plane i1 is a w-face; the cell centre just downstream is cell i1, just up is i1-1.
    p_in = p[i1, :]            # use cell centre at i1 (fluid)
    p_out = p[i2 - 1, :]       # cell centre just upstream of face i2
    F_p_in = np.sum(p_in * dA)
    F_p_out = np.sum(p_out * dA)

    # --- momentum flux:  integral rho w^2 dA  at each plane (w is on the face)
    w_in = w[i1, :]
    w_out = w[i2, :]
    Mom_in = np.sum(rho * w_in ** 2 * dA)
    Mom_out = np.sum(rho * w_out ** 2 * dA)

    # --- wall shear on outer wall r=R_out over z in [i1,i2)
    # dw/dr at wall: w at last centre r_c[nr-1] vs wall w=0 at r=R_out (dist dr/2)
    # tau_wall = mu * dw/dr |_wall ; force on fluid is mu*dw/dr*area (acts +z if w drops)
    z_lo = i1 * dz
    z_hi = i2 * dz
    # axial extent of wall in the CV: cells i1..i2-1
    wcells = w[i1:i2, :]              # (ncv, nr) w on faces; use cell-avg of bounding faces
    w_cell = 0.5 * (w[i1:i2, :] + w[i1 + 1:i2 + 1, :])   # cell-centre w  (ncv, nr)
    dwdr_wall = (0.0 - w_cell[:, -1]) / (dr / 2)         # (ncv,)  (w at wall=0)
    area_wall = 2 * np.pi * sol.R_out * dz               # per axial cell
    F_wall = np.sum(mu * dwdr_wall * area_wall)          # force on fluid from wall (negative)

    # axial momentum balance for fluid in CV (steady):
    #   Mom_out - Mom_in = F_p_in - F_p_out + F_wall + F_float_on_fluid
    # => F_float_on_fluid = (Mom_out - Mom_in) - (F_p_in - F_p_out) - F_wall
    F_float_on_fluid = (Mom_out - Mom_in) - (F_p_in - F_p_out) - F_wall
    F_drag = -F_float_on_fluid       # reaction: fluid on float
    return F_drag, dict(F_p_in=F_p_in, F_p_out=F_p_out, Mom_in=Mom_in,
                        Mom_out=Mom_out, F_wall=F_wall,
                        dP=F_p_in - F_p_out)


def drag_surface(sol):
    """
    Surface-integration drag (form + skin) over the float boundary.
    Cross-check method.  Approximate on the staggered grid.

    Form drag: axial pressure force on axially-facing solid faces.
       A solid cell with a FLUID neighbour in -z (front face) gets +p*area (pushes +z).
       A solid cell with a FLUID neighbour in +z (back face) gets -p*area.
       (pressure acts on the body inward along -normal; net axial = p_front - p_back.)
    Skin drag: viscous shear on the (mostly radial) lateral surface.
       For a face where solid is at larger r and fluid at smaller r (top of body),
       tau = mu * dw/dr; axial force on body = tau * lateral area.
    """
    nz, nr = sol.nz, sol.nr
    dz, dr = sol.dz, sol.dr
    r_c, r_v = sol.r_c, sol.r_v
    rho, mu = sol.rho, sol.rho * sol.nu
    p, w = sol.p, sol.w
    solid = sol.solid

    F_form = 0.0
    F_skin = 0.0

    # ---- FORM drag: axial-facing faces ----
    # front faces: solid[i,j] True and solid[i-1,j] False  -> face at z=z_w[i], normal -z
    # pressure in the fluid cell (i-1,j) pushes the body in +z by p*area
    front = solid & ~np.roll(solid, 1, axis=0)
    front[0, :] = False
    back = solid & ~np.roll(solid, -1, axis=0)
    back[-1, :] = False
    area_ax = 2 * np.pi * r_c * dr                # (nr,) axial face area at radius r_c
    ii, jj = np.where(front)
    for i, j in zip(ii, jj):
        F_form += p[i - 1, j] * area_ax[j]        # fluid pressure pushes +z
    ii, jj = np.where(back)
    for i, j in zip(ii, jj):
        F_form -= p[i + 1, j] * area_ax[j]        # fluid pressure pushes -z

    # ---- SKIN drag: radial-facing faces (lateral surface) ----
    # top lateral: solid[i,j] True, solid[i,j+1] False -> face at r=r_v[j+1], normal +r
    #   shear stress tau_rz = mu*(dw/dr) ; axial drag on body = tau * lateral area
    #   dw/dr across the face ~ (w_fluid - w_solid)/dr ; w_solid=0
    lat = solid & ~np.roll(solid, -1, axis=1)
    lat[:, -1] = False
    ii, jj = np.where(lat)
    for i, j in zip(ii, jj):
        rf = r_v[j + 1]
        # axial velocity in fluid cell just outside: average of w faces of cell (i,j+1)
        w_fluid = 0.5 * (w[i, j + 1] + w[i + 1, j + 1])
        dwdr = (w_fluid - 0.0) / dr
        lat_area = 2 * np.pi * rf * dz
        F_skin += mu * dwdr * lat_area
    # bottom lateral (solid above axis-side fluid): solid[i,j], solid[i,j-1] False
    latb = solid & ~np.roll(solid, 1, axis=1)
    latb[:, 0] = False
    ii, jj = np.where(latb)
    for i, j in zip(ii, jj):
        rf = r_v[j]
        w_fluid = 0.5 * (w[i, j - 1] + w[i + 1, j - 1])
        dwdr = (0.0 - w_fluid) / dr      # solid above fluid: stress sign
        lat_area = 2 * np.pi * rf * dz
        F_skin += -mu * dwdr * lat_area
    return F_form + F_skin, dict(F_form=F_form, F_skin=F_skin)
