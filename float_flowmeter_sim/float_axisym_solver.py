"""
Axisymmetric (r-z) incompressible Navier-Stokes solver.
Pure numpy/scipy.  Staggered MAC grid + Chorin fractional-step projection.

Coordinates
-----------
z = axial (flow direction), r = radial.   w = u_z, v = u_r.
Constant density rho (=1) and kinematic viscosity nu.

Governing eqs
-------------
continuity:  dw/dz + (1/r) d(r v)/dr = 0
z-mom:  dw/dt + w dw/dz + v dw/dr = -(1/rho) dp/dz + nu[ d2w/dz2 + (1/r) d/dr(r dw/dr) ]
r-mom:  dv/dt + w dv/dz + v dv/dr = -(1/rho) dp/dr + nu[ d2v/dz2 + (1/r) d/dr(r dv/dr) - v/r^2 ]

Staggered storage (nz cells axially, nr cells radially; dz, dr spacing)
----------------------------------------------------------------------
  p  : cell centres,  shape (nz, nr).   z_c[i]=(i+.5)dz, r_c[j]=(j+.5)dr  (first centre dr/2, so 1/r safe)
  w  : axial faces,   shape (nz+1, nr). z_w[i]=i*dz,     r_c[j]=(j+.5)dr
  v  : radial faces,  shape (nz, nr+1). z_c[i]=(i+.5)dz, r_v[j]=j*dr      (r_v[0]=0 is the axis)

Boundary conditions
-------------------
  axis r=0      : symmetry  -> v=0 there;  dw/dr=0  (ghost mirror)
  inlet z=0     : uniform w=U_in, v=0
  outlet z=Lz   : zero-gradient (convective-ish) for velocity; p reference (Dirichlet p=0)
  outer r=R_out : bore wall, no-slip  w=v=0
  immersed solid: boolean cell mask; w=v=0 on/inside solid; solid faces are no-flux
                  (Neumann) in the pressure Poisson so the flow goes around it.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class AxisymSolver:
    def __init__(self, Lz, R_out, dz, dr, nu, U_in, solid_cell=None,
                 rho=1.0, p_outlet_dirichlet=True):
        self.Lz, self.R_out = Lz, R_out
        self.dz, self.dr = dz, dr
        self.nu, self.U_in, self.rho = nu, U_in, rho
        self.nz = int(round(Lz / dz))
        self.nr = int(round(R_out / dr))
        nz, nr = self.nz, self.nr

        # coordinate vectors
        self.z_c = (np.arange(nz) + 0.5) * dz          # cell-centre z
        self.r_c = (np.arange(nr) + 0.5) * dr          # cell-centre r  (first = dr/2)
        self.z_w = np.arange(nz + 1) * dz              # w faces (axial)
        self.r_v = np.arange(nr + 1) * dr              # v faces (radial); r_v[0]=0 axis

        # solid cell mask (nz, nr); default = no float
        if solid_cell is None:
            solid_cell = np.zeros((nz, nr), dtype=bool)
        self.solid = solid_cell.astype(bool)
        self.fluid = ~self.solid

        # ---- face solid masks (a face is solid if either neighbouring cell is solid) ----
        # w faces: (nz+1, nr).  interior face i (1..nz-1) borders cells i-1 and i.
        wsolid = np.zeros((nz + 1, nr), dtype=bool)
        wsolid[1:nz, :] = self.solid[:-1, :] | self.solid[1:, :]
        wsolid[0, :] = self.solid[0, :]      # inlet face touching solid (shouldn't happen)
        wsolid[nz, :] = self.solid[-1, :]
        self.wsolid = wsolid
        # v faces: (nz, nr+1).  interior face j (1..nr-1) borders cells j-1 and j.
        vsolid = np.zeros((nz, nr + 1), dtype=bool)
        vsolid[:, 1:nr] = self.solid[:, :-1] | self.solid[:, 1:]
        vsolid[:, 0] = self.solid[:, 0]      # axis face
        vsolid[:, nr] = self.solid[:, -1]    # wall face
        self.vsolid = vsolid

        self.p_outlet_dirichlet = p_outlet_dirichlet
        self._build_poisson()

        # state
        self.w = np.zeros((nz + 1, nr))
        self.v = np.zeros((nz, nr + 1))
        self.p = np.zeros((nz, nr))
        # ramped inlet velocity (set externally before each step for a smooth start)
        self.U_now = U_in
        # init interior axial velocity to U_in in the OPEN (non-solid) part only,
        # and zero behind/inside the body so the first projection isn't a shock.
        self.w[:, :] = 0.0
        self.apply_velocity_bc()

    # ------------------------------------------------------------------ Poisson
    def _build_poisson(self):
        """
        Assemble the axisymmetric pressure-Poisson operator for the divergence
        of the *face* velocity:  Lap(p) = rhs.   Cell-centred, finite volume.

        For cell (i,j) with centre radius r_c[j], face radii r_v[j], r_v[j+1]:
        (1/r d/dr(r dp/dr)) ~ [ r_v[j+1]*(p[i,j+1]-p[i,j]) - r_v[j]*(p[i,j]-p[i,j-1]) ]
                               / (r_c[j] * dr^2)
        d2p/dz2 ~ (p[i+1,j]-2p[i,j]+p[i-1,j])/dz^2
        Neumann (zero normal grad) at: axis, wall, inlet, and every SOLID face.
        Dirichlet p=0 at outlet (last z column) if p_outlet_dirichlet.
        Solid cells: identity row (p=0 there, decoupled).
        """
        nz, nr = self.nz, self.nr
        dz2, dr2 = self.dz ** 2, self.dr ** 2
        r_c, r_v = self.r_c, self.r_v
        N = nz * nr

        def idx(i, j):
            return i * nr + j

        rows, cols, data = [], [], []
        for i in range(nz):
            for j in range(nr):
                k = idx(i, j)
                if self.solid[i, j]:
                    rows.append(k); cols.append(k); data.append(1.0)
                    continue
                diag = 0.0
                # ---- axial (z) neighbours ----
                # minus z face (between i-1 and i): open unless i==0 (inlet, Neumann)
                if i > 0 and not self.wsolid[i, j]:
                    coef = 1.0 / dz2
                    rows.append(k); cols.append(idx(i - 1, j)); data.append(coef)
                    diag -= coef
                # plus z face (between i and i+1)
                if i < nz - 1 and not self.wsolid[i + 1, j]:
                    coef = 1.0 / dz2
                    rows.append(k); cols.append(idx(i + 1, j)); data.append(coef)
                    diag -= coef
                # ---- radial (r) neighbours ----
                # minus r face at r_v[j]: open unless j==0 (axis) or solid
                if j > 0 and not self.vsolid[i, j]:
                    coef = r_v[j] / (r_c[j] * dr2)
                    rows.append(k); cols.append(idx(i, j - 1)); data.append(coef)
                    diag -= coef
                # plus r face at r_v[j+1]: open unless j==nr-1 (wall) or solid
                if j < nr - 1 and not self.vsolid[i, j + 1]:
                    coef = r_v[j + 1] / (r_c[j] * dr2)
                    rows.append(k); cols.append(idx(i, j + 1)); data.append(coef)
                    diag -= coef
                # ---- outlet Dirichlet (p=0): add a ghost with p=0 across the +z face
                if self.p_outlet_dirichlet and i == nz - 1:
                    # outlet face is open: ghost cell value 0 -> contributes -coef to diag
                    coef = 1.0 / dz2
                    diag -= coef
                rows.append(k); cols.append(k); data.append(diag)

        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        # If no Dirichlet anywhere (pure Neumann), pin one fluid cell to remove null space.
        if not self.p_outlet_dirichlet:
            # pin first fluid cell
            ff = np.argwhere(self.fluid.ravel())[0, 0]
            A = A.tolil()
            A.rows[ff] = [ff]; A.data[ff] = [1.0]
            A = A.tocsr()
            self._pin = ff
        else:
            self._pin = None
        self.A = A.tocsc()
        self.lu = spla.splu(self.A)

    # ------------------------------------------------------------------ BCs
    def apply_velocity_bc(self):
        nz, nr = self.nz, self.nr
        # inlet: uniform axial velocity at z=0 face (in fluid columns)
        self.w[0, :] = self.U_now
        self.w[0, self.wsolid[0, :]] = 0.0
        # outlet: zero-gradient axial velocity
        self.w[nz, :] = self.w[nz - 1, :]
        # wall (outer r): w=0 stored via ghost in diffusion; v at wall face =0
        self.v[:, nr] = 0.0
        # axis: v=0 at r=0 face
        self.v[:, 0] = 0.0
        # radial velocity zero-gradient at inlet/outlet handled in advection ghosts
        # enforce solid faces zero
        self.w[self.wsolid] = 0.0
        self.v[self.vsolid] = 0.0

    # ------------------------------------------------------------------ helpers
    def _w_with_ghost_r(self, w):
        """Pad w in r with ghosts: axis mirror (dw/dr=0) at j=-1, wall no-slip ghost at j=nr."""
        nz1, nr = w.shape
        wg = np.empty((nz1, nr + 2))
        wg[:, 1:-1] = w
        wg[:, 0] = w[:, 0]            # axis: dw/dr=0 -> ghost = first interior
        wg[:, -1] = -w[:, -1]         # wall no-slip: w(wall)=0 -> ghost = -interior (centre at r_c[nr-1])
        return wg

    def _v_with_ghost_z(self, v):
        """Pad v in z with ghosts: inlet v=0 (mirror to give 0 at face), outlet zero-grad."""
        nz, nr1 = v.shape
        vg = np.empty((nz + 2, nr1))
        vg[1:-1, :] = v
        vg[0, :] = -v[0, :]          # inlet v=0 at z=0 -> antisymmetric ghost
        vg[-1, :] = v[-1, :]         # outlet zero-gradient
        return vg

    # ------------------------------------------------------------------ one step
    def step(self, dt):
        nz, nr = self.nz, self.nr
        dz, dr, nu = self.dz, self.dr, self.nu
        r_c, r_v = self.r_c, self.r_v
        w, v, rho = self.w, self.v, self.rho
        self.apply_velocity_bc()

        # ============ provisional w* (axial momentum on w-faces) ============
        # interior w faces in z: i = 1..nz-1 ; all j
        wstar = w.copy()
        # ghosts in r for w
        wg = self._w_with_ghost_r(w)           # (nz+1, nr+2)
        # interior indices
        I = slice(1, nz)         # z faces 1..nz-1
        # --- advection (1st-order upwind) ---
        # dw/dz at face i (centre of two half-cells) using w itself
        wc = w[I, :]                                   # (nz-1, nr)
        dwdz_p = (w[2:nz + 1, :] - w[I, :]) / dz       # forward
        dwdz_m = (w[I, :] - w[0:nz - 1, :]) / dz       # backward
        adv_z = np.where(wc >= 0, wc * dwdz_m, wc * dwdz_p)
        # radial velocity interpolated to w-faces (avg of 4 surrounding v)
        # v is (nz, nr+1). at w-face i (between cells i-1,i), r-centre j:
        #   v around = v[i-1,j], v[i-1,j+1], v[i,j], v[i,j+1]
        v_at_w = 0.25 * (v[0:nz - 1, 0:nr] + v[0:nz - 1, 1:nr + 1]
                         + v[1:nz, 0:nr] + v[1:nz, 1:nr + 1])      # (nz-1, nr)
        wgI = wg[I, :]                                  # (nz-1, nr+2)
        dwdr_p = (wgI[:, 2:] - wgI[:, 1:-1]) / dr
        dwdr_m = (wgI[:, 1:-1] - wgI[:, 0:-2]) / dr
        adv_r = np.where(v_at_w >= 0, v_at_w * dwdr_m, v_at_w * dwdr_p)
        # --- diffusion ---
        d2wdz2 = (w[2:nz + 1, :] - 2 * w[I, :] + w[0:nz - 1, :]) / dz ** 2
        # (1/r) d/dr(r dw/dr) at r_c[j]
        rp = r_v[1:nr + 1][None, :]      # outer face radius for cell j -> r_v[j+1]
        rm = r_v[0:nr][None, :]          # inner face radius -> r_v[j]
        flux_p = rp * (wgI[:, 2:] - wgI[:, 1:-1]) / dr
        flux_m = rm * (wgI[:, 1:-1] - wgI[:, 0:-2]) / dr
        lap_r_w = (flux_p - flux_m) / (r_c[None, :] * dr)
        diff_w = nu * (d2wdz2 + lap_r_w)
        wstar[I, :] = w[I, :] + dt * (-adv_z - adv_r + diff_w)

        # ============ provisional v* (radial momentum on v-faces) ============
        vstar = v.copy()
        vg = self._v_with_ghost_z(v)           # (nz+2, nr+1)
        J = slice(1, nr)        # r faces 1..nr-1 (exclude axis j=0 and wall j=nr)
        vc = v[:, J]                                  # (nz, nr-1)
        # axial velocity interpolated to v-faces:
        # w is (nz+1, nr). at v-face (cell i, r-face j between cells j-1,j):
        #   w around = w[i,j-1], w[i,j], w[i+1,j-1], w[i+1,j]
        w_at_v = 0.25 * (w[0:nz, 0:nr - 1] + w[0:nz, 1:nr]
                         + w[1:nz + 1, 0:nr - 1] + w[1:nz + 1, 1:nr])   # (nz, nr-1)
        vgJ = vg[:, J]                                 # (nz+2, nr-1)
        dvdz_p = (vgJ[2:, :] - vgJ[1:-1, :]) / dz
        dvdz_m = (vgJ[1:-1, :] - vgJ[0:-2, :]) / dz
        adv_vz = np.where(w_at_v >= 0, w_at_v * dvdz_m, w_at_v * dvdz_p)
        dvdr_p = (v[:, 2:nr + 1] - v[:, J]) / dr
        dvdr_m = (v[:, J] - v[:, 0:nr - 1]) / dr
        adv_vr = np.where(vc >= 0, vc * dvdr_m, vc * dvdr_p)
        # diffusion of v
        d2vdz2 = (vgJ[2:, :] - 2 * vgJ[1:-1, :] + vgJ[0:-2, :]) / dz ** 2
        rface = r_v[J][None, :]          # radius at v-face j
        # (1/r) d/dr(r dv/dr) - v/r^2 at r_v[j].  Use centred fluxes between centres.
        # flux at r_c around the v-face: r_c[j] is outer centre, r_c[j-1] inner centre.
        rcp = r_c[J][None, :]            # r_c[j]
        rcm = r_c[0:nr - 1][None, :]     # r_c[j-1]
        flux_vp = rcp * (v[:, 2:nr + 1] - v[:, J]) / dr
        flux_vm = rcm * (v[:, J] - v[:, 0:nr - 1]) / dr
        lap_r_v = (flux_vp - flux_vm) / (rface * dr) - v[:, J] / rface ** 2
        diff_v = nu * (d2vdz2 + lap_r_v)
        vstar[:, J] = v[:, J] + dt * (-adv_vz - adv_vr + diff_v)

        # zero the provisional vel on solid/wall/axis faces & re-apply inlet
        wstar[self.wsolid] = 0.0
        vstar[self.vsolid] = 0.0
        wstar[0, :] = self.U_now
        wstar[0, self.wsolid[0, :]] = 0.0
        # convective (zero-gradient) outlet, then RESCALE to enforce global mass
        # conservation: total outflux must equal total influx.  This keeps the
        # pressure-Poisson solvable and stops slow divergence drift.
        wstar[nz, :] = wstar[nz - 1, :]
        dA = 2 * np.pi * r_c * dr
        Qin = np.sum(wstar[0, :] * dA)
        Qout = np.sum(wstar[nz, :] * dA)
        open_out = ~self.wsolid[nz, :]
        Aout = np.sum(dA[open_out])
        if Aout > 0:
            wstar[nz, open_out] += (Qin - Qout) / Aout
        vstar[:, 0] = 0.0
        vstar[:, nr] = 0.0

        # ============ pressure Poisson:  Lap(p) = (rho/dt) div(u*) ============
        # divergence at cell centres (axisymmetric):
        #   dw/dz + (1/r) d(r v)/dr
        div = ((wstar[1:nz + 1, :] - wstar[0:nz, :]) / dz
               + (r_v[1:nr + 1][None, :] * vstar[:, 1:nr + 1]
                  - r_v[0:nr][None, :] * vstar[:, 0:nr]) / (r_c[None, :] * dr))
        rhs = (rho / dt) * div
        rhs[self.solid] = 0.0
        if self._pin is not None:
            rhs.ravel()[self._pin] = 0.0
        p = self.lu.solve(rhs.ravel()).reshape(nz, nr)
        p[self.solid] = 0.0
        self.p = p

        # ============ velocity correction:  u = u* - dt/rho * grad(p) ============
        # w correction on interior z-faces (i=1..nz-1): dp/dz between cells i-1,i
        w_new = wstar.copy()
        gradz = (p[1:nz, :] - p[0:nz - 1, :]) / dz
        w_new[1:nz, :] -= dt / rho * gradz
        # v correction on interior r-faces (j=1..nr-1): dp/dr between cells j-1,j
        v_new = vstar.copy()
        gradr = (p[:, 1:nr] - p[:, 0:nr - 1]) / dr
        v_new[:, 1:nr] -= dt / rho * gradr

        # re-enforce BC/solid
        w_new[self.wsolid] = 0.0
        v_new[self.vsolid] = 0.0
        w_new[0, :] = self.U_now
        w_new[0, self.wsolid[0, :]] = 0.0
        # mass-consistent outlet (same rescale as the provisional step)
        w_new[nz, :] = w_new[nz - 1, :]
        dA = 2 * np.pi * r_c * dr
        Qin = np.sum(w_new[0, :] * dA)
        Qout = np.sum(w_new[nz, :] * dA)
        open_out = ~self.wsolid[nz, :]
        Aout = np.sum(dA[open_out])
        if Aout > 0:
            w_new[nz, open_out] += (Qin - Qout) / Aout
        v_new[:, 0] = 0.0
        v_new[:, nr] = 0.0

        self.w, self.v = w_new, v_new
        return div

    # ------------------------------------------------------------------ diagnostics
    def max_divergence(self, exclude_outlet=True):
        """Max |div| in the interior fluid.  The outlet column is a BC boundary
        (convective + mass rescale), not projected, so it carries a small benign
        residual; exclude it to report the physically meaningful interior value."""
        nz, nr = self.nz, self.nr
        r_c, r_v = self.r_c, self.r_v
        div = ((self.w[1:nz + 1, :] - self.w[0:nz, :]) / self.dz
               + (r_v[1:nr + 1][None, :] * self.v[:, 1:nr + 1]
                  - r_v[0:nr][None, :] * self.v[:, 0:nr]) / (r_c[None, :] * self.dr))
        div[self.solid] = 0.0
        if exclude_outlet:
            div = div[:-1, :]
        return np.abs(div).max()

    def cfl_dt(self, safety=0.4):
        """Pick a stable dt from advective CFL and diffusive limit."""
        umax = max(abs(self.w).max(), abs(self.v).max(), self.U_in, 1e-9)
        dt_adv = safety * min(self.dz, self.dr) / umax
        dt_diff = 0.20 * min(self.dz, self.dr) ** 2 / self.nu
        return min(dt_adv, dt_diff)

    def w_centre_profile(self, i_plane):
        """Axial velocity vs r_c at axial face plane i (returns r_c, w)."""
        return self.r_c.copy(), self.w[i_plane, :].copy()
