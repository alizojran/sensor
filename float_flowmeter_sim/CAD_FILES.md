# CAD 导出文件(浮子 + 壳体)

由 `float_cad_export.py` 生成(ezdxf 出 DXF,cadquery/OCCT 出 STEP);
`float_cad_preview.py` 重新导入 STEP 校验并出 3D 预览。

## 文件
| 文件 | 内容 | 用途 |
|---|---|---|
| `float_profile.dxf` | 浮子**回转母线**(闭合)+ 轴线 + 图层(AXIS/PROFILE/BORE/TEXT)+ 注释 | 导入 CAD,绕轴 **revolve 360°** 得浮子实体 |
| `housing_bore_profile.dxf` | 壳体**锥孔/阀座**母线 + 外形线 + 注释 | 锥孔/壳体的 2D 草图参数 |
| `float_body.step` | 浮子**3D 实体**(回转,含中心 Ø3 导杆孔),V≈1470 mm³ | 直接导入 CAD/CAM |
| `housing_body.step` | 壳体**3D 实体**(锥孔+阀座+进出口),V≈17707 mm³ | 直接导入 |
| `float_cad_preview.png` | STEP 3D 预览(已重新导入校验) | 查看 |
| `float_mfg_drawing.png` / `housing_mfg_drawing.png` | 浮子 / 壳体**配合制造图**(尺寸·公差·公差链) | 加工/检验 |

## 参数化草图(回转母线;轴向 x / 径向 y,mm)
- **浮子** `FLOAT = [(0,1.5),(0,6.0),(3.0,6.0),(26.0,2.25),(26.0,1.5)]`
  - 计量边 Ø12(r=6)、计量柱段 0–3、锥尾→Ø4.5(r=2.25)、中心导杆孔 Ø3.0(r=1.5)。
- **壳体锥孔** `BORE = [(0,5.0),(8,5.0),(12,6.2),(32,9.2),(40,9.2)]`,外形 R=14、L=40
  - 进/出口 Ø10(G½ 简化)、阀座 8–12、零位锥孔 Ø12.40(r=6.2)、满量程 Ø18.4(r=9.2),锥度 0.15 mm/mm。

## 关键配合
- 计量:壳体零位锥孔 **Ø12.40 H7** × 浮子边 **Ø12 -0/-0.02** → 零位半径环隙 **0.20**(0.20…0.23);满量程 3.2 由锥度给出。
- 导向:导杆 **Ø3.0 g6** × 浮子中心孔 **Ø3.0 H7**(滑动)→ 偏心 ≤0.012。
- 关键特征:计量边 **锐利 R≤0.05、去毛刺**(决定粘度免疫);锥孔同轴度 ≤0.03。
- 材料 316L;FKM O 形圈;磁铁(NdFeB 环)在装配中嵌入并密封(STEP 未含磁铁腔,按 2 件焊接件实现)。

## 重新生成 / 校验
```bash
pip install ezdxf cadquery
python3 float_cad_export.py      # -> *.dxf, *.step
python3 float_cad_preview.py     # 重新导入 STEP 校验 + 3D 预览
python3 housing_mfg_drawing.py   # 壳体配合制造图
```
