import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeCompiler

/-! Generated exact semantic column layout for one terminal-CE claim. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe

open Nightstream.Implementation.R1CS.TerminalCeCompiler

set_option maxHeartbeats 1000000

def layout : Layout where
normBound := 2
expectedPublicWidth := some 257
structureRows := 1
structureColumns := 257
witnessRows := 54
witnessColumns := 5
witnessCols := ((List.range 5).map (fun index => 1 + 54 * index)) ++
    ((List.range 5).map (fun index => 2 + 54 * index)) ++
    ((List.range 5).map (fun index => 3 + 54 * index)) ++
    ((List.range 5).map (fun index => 4 + 54 * index)) ++
    ((List.range 5).map (fun index => 5 + 54 * index)) ++
    ((List.range 5).map (fun index => 6 + 54 * index)) ++
    ((List.range 5).map (fun index => 7 + 54 * index)) ++
    ((List.range 5).map (fun index => 8 + 54 * index)) ++
    ((List.range 5).map (fun index => 9 + 54 * index)) ++
    ((List.range 5).map (fun index => 10 + 54 * index)) ++
    ((List.range 5).map (fun index => 11 + 54 * index)) ++
    ((List.range 5).map (fun index => 12 + 54 * index)) ++
    ((List.range 5).map (fun index => 13 + 54 * index)) ++
    ((List.range 5).map (fun index => 14 + 54 * index)) ++
    ((List.range 5).map (fun index => 15 + 54 * index)) ++
    ((List.range 5).map (fun index => 16 + 54 * index)) ++
    ((List.range 5).map (fun index => 17 + 54 * index)) ++
    ((List.range 5).map (fun index => 18 + 54 * index)) ++
    ((List.range 5).map (fun index => 19 + 54 * index)) ++
    ((List.range 5).map (fun index => 20 + 54 * index)) ++
    ((List.range 5).map (fun index => 21 + 54 * index)) ++
    ((List.range 5).map (fun index => 22 + 54 * index)) ++
    ((List.range 5).map (fun index => 23 + 54 * index)) ++
    ((List.range 5).map (fun index => 24 + 54 * index)) ++
    ((List.range 5).map (fun index => 25 + 54 * index)) ++
    ((List.range 5).map (fun index => 26 + 54 * index)) ++
    ((List.range 5).map (fun index => 27 + 54 * index)) ++
    ((List.range 5).map (fun index => 28 + 54 * index)) ++
    ((List.range 5).map (fun index => 29 + 54 * index)) ++
    ((List.range 5).map (fun index => 30 + 54 * index)) ++
    ((List.range 5).map (fun index => 31 + 54 * index)) ++
    ((List.range 5).map (fun index => 32 + 54 * index)) ++
    ((List.range 5).map (fun index => 33 + 54 * index)) ++
    ((List.range 5).map (fun index => 34 + 54 * index)) ++
    ((List.range 5).map (fun index => 35 + 54 * index)) ++
    ((List.range 5).map (fun index => 36 + 54 * index)) ++
    ((List.range 5).map (fun index => 37 + 54 * index)) ++
    ((List.range 5).map (fun index => 38 + 54 * index)) ++
    ((List.range 5).map (fun index => 39 + 54 * index)) ++
    ((List.range 5).map (fun index => 40 + 54 * index)) ++
    ((List.range 5).map (fun index => 41 + 54 * index)) ++
    ((List.range 5).map (fun index => 42 + 54 * index)) ++
    ((List.range 5).map (fun index => 43 + 54 * index)) ++
    ((List.range 5).map (fun index => 44 + 54 * index)) ++
    ((List.range 5).map (fun index => 45 + 54 * index)) ++
    ((List.range 5).map (fun index => 46 + 54 * index)) ++
    ((List.range 5).map (fun index => 47 + 54 * index)) ++
    ((List.range 5).map (fun index => 48 + 54 * index)) ++
    ((List.range 5).map (fun index => 49 + 54 * index)) ++
    ((List.range 5).map (fun index => 50 + 54 * index)) ++
    ((List.range 5).map (fun index => 51 + 54 * index)) ++
    ((List.range 5).map (fun index => 52 + 54 * index)) ++
    ((List.range 5).map (fun index => 53 + 54 * index)) ++
    ((List.range 5).map (fun index => 54 + 54 * index))
normFirstAllocatedColumn := 1514
commitmentCols := ((List.range 972).map (fun index => 271 + 1 * index))
commitmentD := 54
commitmentKappa := 18
publicCols := ((List.range 6).map (fun index => 1243 + 54 * index)) ++
    List.replicate 251 1513 ++
    ((List.range 5).map (fun index => 1244 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1245 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1246 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1247 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1248 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1249 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1250 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1251 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1252 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1253 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1254 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1255 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1256 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1257 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1258 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1259 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1260 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1261 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1262 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1263 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1264 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1265 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1266 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1267 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1268 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1269 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1270 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1271 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1272 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1273 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1274 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1275 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1276 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1277 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1278 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1279 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1280 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1281 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1282 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1283 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1284 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1285 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1286 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1287 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1288 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1289 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1290 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1291 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1292 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1293 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1294 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1295 + 54 * index)) ++
    List.replicate 252 1513 ++
    ((List.range 5).map (fun index => 1296 + 54 * index)) ++
    List.replicate 252 1513
publicRows := 54
publicWidth := 257
publicInputLen := 257
pointCols := [{ c0 := 1784, c1 := 1786 }]
evaluationCols := [(((List.range 128).map (fun index => 1796 + 1 * index))),
    (((List.range 128).map (fun index => 1924 + 1 * index))),
    (((List.range 128).map (fun index => 2052 + 1 * index)))]
constantTermCols := [{ c0 := 2180, c1 := 2181 }, { c0 := 2182, c1 := 2183 }, { c0 := 2184, c1 := 2185 }]
ncPointCols := [{ c0 := 2186, c1 := 2188 }, { c0 := 2198, c1 := 2200 }, { c0 := 2220, c1 := 2222 }, { c0 := 2262, c1 := 2264 }, { c0 := 2344, c1 := 2346 }, { c0 := 2506, c1 := 2508 }, { c0 := 2828, c1 := 2830 }, { c0 := 3470, c1 := 3472 }, { c0 := 4752, c1 := 4754 }]
ncEvaluationCols := [7324, 7325, 7336, 7337, 7348, 7349, 7360, 7361, 7372, 7373, 7384, 7385, 7396, 7397, 7408, 7409, 7420, 7421, 7432, 7433, 7444, 7445, 7456, 7457, 7468, 7469, 7480, 7481, 7492, 7493, 7504, 7505, 7516, 7517, 7528, 7529, 7540, 7541, 7552, 7553, 7564, 7565, 7576, 7577, 7588, 7589, 7600, 7601, 7612, 7613, 7624, 7625, 7636, 7637, 7648, 7649, 7660, 7661, 7672, 7673, 7684, 7685, 7696, 7697, 7708, 7709, 7720, 7721, 7732, 7733, 7744, 7745, 7756, 7757, 7768, 7769, 7780, 7781, 7792, 7793, 7804, 7805, 7814, 7815, 7824, 7825, 7834, 7835, 7844, 7845, 7854, 7855, 7864, 7865, 7874, 7875, 7884, 7885, 7894, 7895, 7904, 7905, 7914, 7915, 7924, 7925] ++
    ((List.range 22).map (fun index => 7934 + 1 * index))
ncEvaluationLanes := 64

theorem layout_shape : ShapeValid layout where
witnessSize := by native_decide
commitmentSize := by native_decide
publicSize := by native_decide
publicRowsPositive := by native_decide
publicProjectionWithinStructure := by native_decide
publicWidthPinned := by rfl
constantTermSize := by native_decide
evaluationRowsNonempty := by native_decide
evaluationRowsEven := by native_decide
ncEvaluationSize := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe
