"""
CAD export of the redesigned float meter:
  * DXF  : 2D meridional sketch (revolve generatrix) for the float and the housing bore,
           on layers AXIS/PROFILE/BORE/TEXT  (ezdxf).
  * STEP : true 3D solids by revolving the profiles 360 deg  (cadquery / OCCT).
Run:  python3 float_cad_export.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- shared parameters (mm) --------------------------------------------
# Float (body of revolution, axis = X; radial = Y).  Blunt face at x=0 (upstream).
R_EDGE = 6.0        # Ø12 sharp metering edge
LAND = 3.0          # cylindrical metering land length
LTOT = 26.0         # float length
R_TAIL = 2.25       # Ø4.5 tail tip (leaves wall over the guide bore)
R_BORE = 1.5        # Ø3.0 central guide bore
# Float closed generatrix (axial x, radial y), CCW, not touching axis (has bore):
FLOAT = [(0, R_BORE), (0, R_EDGE), (LAND, R_EDGE), (LTOT, R_TAIL), (LTOT, R_BORE)]

# Housing (inline body with tapered internal bore). axis = X.
R_HOUS = 14.0       # outer radius
L_HOUS = 40.0       # length (inlet face .. outlet face)
R_PORT = 5.0        # inlet/outlet port bore (G1/2 core, simplified)
R_ZERO = R_EDGE + 0.2   # 6.2  零位锥孔 Ø12.40 (H7)
R_FULL = R_EDGE + 3.2   # 9.2  满量程锥孔 Ø18.4
# Bore inner profile (axial x, radial y): inlet port -> conical seat -> tapered bore -> outlet
BORE = [(0, R_PORT), (8, R_PORT), (12, R_ZERO), (32, R_FULL), (L_HOUS, R_FULL)]

# ============================ DXF (ezdxf) ===========================================
def write_dxf():
    import ezdxf
    # ---- float profile ----
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    for name, col in (("AXIS", 5), ("PROFILE", 7), ("BORE", 3), ("TEXT", 1)):
        if name not in doc.layers:
            doc.layers.add(name, color=col)
    doc.layers.get("AXIS").dxf.linetype = "CENTER"
    # closed profile polyline
    msp.add_lwpolyline([(x, y) for x, y in FLOAT], close=True,
                       dxfattribs={"layer": "PROFILE"})
    # guide-bore line (already an edge) + axis
    msp.add_line((-4, 0), (LTOT + 4, 0), dxfattribs={"layer": "AXIS"})
    # text notes
    notes = [
        (0, R_EDGE + 1.5, "Ø12 -0/-0.02  锐利计量边 (基准B, R<=0.05 去毛刺)"),
        (LAND + 0.3, R_EDGE + 0.4, "计量柱段 3"),
        (LTOT - 8, R_TAIL + 1.2, "流线锥尾 -> Ø4.5"),
        (1, R_BORE + 0.3, "中心导杆孔 Ø3.0 H7 (基准A)"),
        (2, -1.6, "回转母线 360°  材料 316L  总长 26"),
    ]
    for x, y, t in notes:
        msp.add_text(t, height=0.6, dxfattribs={"layer": "TEXT"}).set_placement((x, y))
    # a couple of linear dims
    dim = msp.add_linear_dim(base=(0, R_EDGE + 4), p1=(0, R_EDGE), p2=(LAND, R_EDGE),
                             dxfattribs={"layer": "TEXT"})
    dim.render()
    out1 = os.path.join(HERE, "float_profile.dxf")
    doc.saveas(out1)

    # ---- housing bore profile ----
    doc2 = ezdxf.new("R2010", setup=True)
    m2 = doc2.modelspace()
    for name, col in (("AXIS", 5), ("BORE", 3), ("OUTER", 7), ("TEXT", 1)):
        if name not in doc2.layers:
            doc2.layers.add(name, color=col)
    m2.add_lwpolyline(BORE, dxfattribs={"layer": "BORE"})
    m2.add_lwpolyline([(0, R_HOUS), (L_HOUS, R_HOUS)], dxfattribs={"layer": "OUTER"})
    m2.add_line((-4, 0), (L_HOUS + 4, 0), dxfattribs={"layer": "AXIS"})
    h_notes = [
        (0.5, R_PORT + 0.3, "进口 G1/2 (简化 Ø10)"),
        (9, R_ZERO + 0.3, "锥形阀座 (止回)"),
        (12, R_ZERO - 1.2, "零位锥孔 Ø12.40 H7"),
        (30, R_FULL + 0.4, "满量程 Ø18.4 (锥度 0.15/mm)"),
        (33, R_PORT, "出口 G1/2"),
    ]
    for x, y, t in h_notes:
        m2.add_text(t, height=0.7, dxfattribs={"layer": "TEXT"}).set_placement((x, y))
    out2 = os.path.join(HERE, "housing_bore_profile.dxf")
    doc2.saveas(out2)
    return out1, out2


# ============================ STEP (cadquery) =======================================
def write_step():
    import cadquery as cq
    # float: revolve closed generatrix about the X axis
    wp = cq.Workplane("XZ").polyline([(x, y) for x, y in FLOAT]).close()
    floatsolid = wp.revolve(360, (0, 0, 0), (1, 0, 0))
    f1 = os.path.join(HERE, "float_body.step")
    cq.exporters.export(floatsolid, f1)

    # housing: closed meridional region between bore (inner) and outer wall, revolved
    hpts = BORE + [(L_HOUS, R_HOUS), (0, R_HOUS)]
    hw = cq.Workplane("XZ").polyline([(x, y) for x, y in hpts]).close()
    housing = hw.revolve(360, (0, 0, 0), (1, 0, 0))
    f2 = os.path.join(HERE, "housing_body.step")
    cq.exporters.export(housing, f2)

    # quick validity check: volumes
    return f1, f2, floatsolid.val().Volume(), housing.val().Volume()


if __name__ == "__main__":
    print("=== parametric sketch (revolve generatrix), axial x / radial y [mm] ===")
    print("FLOAT  :", FLOAT)
    print("HOUSING bore:", BORE, " outer R", R_HOUS, " L", L_HOUS)
    d1, d2 = write_dxf()
    print("DXF  ->", os.path.basename(d1), ",", os.path.basename(d2))
    try:
        s1, s2, vf, vh = write_step()
        print(f"STEP ->", os.path.basename(s1), f"(vol {vf:.0f} mm^3) ,",
              os.path.basename(s2), f"(vol {vh:.0f} mm^3)")
    except Exception as e:
        print("STEP export failed:", repr(e))
