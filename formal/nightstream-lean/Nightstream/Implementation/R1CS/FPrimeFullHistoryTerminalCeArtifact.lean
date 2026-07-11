import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCePacked
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeLayout
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap12
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeMap13

import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile

/-! Exact normalized checked program for every direct terminal-CE claim. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

def inputColumns : List Nat :=
    [0] ++
    ((List.range 972).map (fun index => 271 + 1 * index)) ++
    [1513] ++
    ((List.range 5).map (fun index => 1243 + 54 * index)) ++
    ((List.range 5).map (fun index => 1244 + 54 * index)) ++
    ((List.range 5).map (fun index => 1245 + 54 * index)) ++
    ((List.range 5).map (fun index => 1246 + 54 * index)) ++
    ((List.range 5).map (fun index => 1247 + 54 * index)) ++
    ((List.range 5).map (fun index => 1248 + 54 * index)) ++
    ((List.range 5).map (fun index => 1249 + 54 * index)) ++
    ((List.range 5).map (fun index => 1250 + 54 * index)) ++
    ((List.range 5).map (fun index => 1251 + 54 * index)) ++
    ((List.range 5).map (fun index => 1252 + 54 * index)) ++
    ((List.range 5).map (fun index => 1253 + 54 * index)) ++
    ((List.range 5).map (fun index => 1254 + 54 * index)) ++
    ((List.range 5).map (fun index => 1255 + 54 * index)) ++
    ((List.range 5).map (fun index => 1256 + 54 * index)) ++
    ((List.range 5).map (fun index => 1257 + 54 * index)) ++
    ((List.range 5).map (fun index => 1258 + 54 * index)) ++
    ((List.range 5).map (fun index => 1259 + 54 * index)) ++
    ((List.range 5).map (fun index => 1260 + 54 * index)) ++
    ((List.range 5).map (fun index => 1261 + 54 * index)) ++
    ((List.range 5).map (fun index => 1262 + 54 * index)) ++
    ((List.range 5).map (fun index => 1263 + 54 * index)) ++
    ((List.range 5).map (fun index => 1264 + 54 * index)) ++
    ((List.range 5).map (fun index => 1265 + 54 * index)) ++
    ((List.range 5).map (fun index => 1266 + 54 * index)) ++
    ((List.range 5).map (fun index => 1267 + 54 * index)) ++
    ((List.range 5).map (fun index => 1268 + 54 * index)) ++
    ((List.range 5).map (fun index => 1269 + 54 * index)) ++
    ((List.range 5).map (fun index => 1270 + 54 * index)) ++
    ((List.range 5).map (fun index => 1271 + 54 * index)) ++
    ((List.range 5).map (fun index => 1272 + 54 * index)) ++
    ((List.range 5).map (fun index => 1273 + 54 * index)) ++
    ((List.range 5).map (fun index => 1274 + 54 * index)) ++
    ((List.range 5).map (fun index => 1275 + 54 * index)) ++
    ((List.range 5).map (fun index => 1276 + 54 * index)) ++
    ((List.range 5).map (fun index => 1277 + 54 * index)) ++
    ((List.range 5).map (fun index => 1278 + 54 * index)) ++
    ((List.range 5).map (fun index => 1279 + 54 * index)) ++
    ((List.range 5).map (fun index => 1280 + 54 * index)) ++
    ((List.range 5).map (fun index => 1281 + 54 * index)) ++
    ((List.range 5).map (fun index => 1282 + 54 * index)) ++
    ((List.range 5).map (fun index => 1283 + 54 * index)) ++
    ((List.range 5).map (fun index => 1284 + 54 * index)) ++
    ((List.range 5).map (fun index => 1285 + 54 * index)) ++
    ((List.range 5).map (fun index => 1286 + 54 * index)) ++
    ((List.range 5).map (fun index => 1287 + 54 * index)) ++
    ((List.range 5).map (fun index => 1288 + 54 * index)) ++
    ((List.range 5).map (fun index => 1289 + 54 * index)) ++
    ((List.range 5).map (fun index => 1290 + 54 * index)) ++
    ((List.range 5).map (fun index => 1291 + 54 * index)) ++
    ((List.range 5).map (fun index => 1292 + 54 * index)) ++
    ((List.range 5).map (fun index => 1293 + 54 * index)) ++
    ((List.range 5).map (fun index => 1294 + 54 * index)) ++
    ((List.range 5).map (fun index => 1295 + 54 * index)) ++
    ((List.range 5).map (fun index => 1296 + 54 * index)) ++
    ((List.range 390).map (fun index => 1796 + 1 * index)) ++
    [1784, 1786, 2186, 2188, 2198, 2200, 2220, 2222, 2262, 2264, 2344, 2346, 2506, 2508, 2828, 2830, 3470, 3472, 4752, 4754, 7324, 7325, 7336, 7337, 7348, 7349, 7360, 7361, 7372, 7373, 7384, 7385, 7396, 7397, 7408, 7409, 7420, 7421, 7432, 7433, 7444, 7445, 7456, 7457, 7468, 7469, 7480, 7481, 7492, 7493, 7504, 7505, 7516, 7517, 7528, 7529, 7540, 7541, 7552, 7553, 7564, 7565, 7576, 7577, 7588, 7589, 7600, 7601, 7612, 7613, 7624, 7625, 7636, 7637, 7648, 7649, 7660, 7661, 7672, 7673, 7684, 7685, 7696, 7697, 7708, 7709, 7720, 7721, 7732, 7733, 7744, 7745, 7756, 7757, 7768, 7769, 7780, 7781, 7792, 7793, 7804, 7805, 7814, 7815, 7824, 7825, 7834, 7835, 7844, 7845, 7854, 7855, 7864, 7865, 7874, 7875, 7884, 7885, 7894, 7895, 7904, 7905, 7914, 7915, 7924, 7925] ++
    ((List.range 22).map (fun index => 7934 + 1 * index)) ++
    ((List.range 5).map (fun index => 1 + 54 * index)) ++
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

def packedDecode : Option (List Instruction) := Generated.decoded

def instructions : List Instruction := packedDecode.getD []

def rows : List Row := CheckedProgram.rows instructions
def columnMaps : List (List Nat) :=
    [GeneratedMaps.claimMap0,
    GeneratedMaps.claimMap1,
    GeneratedMaps.claimMap2,
    GeneratedMaps.claimMap3,
    GeneratedMaps.claimMap4,
    GeneratedMaps.claimMap5,
    GeneratedMaps.claimMap6,
    GeneratedMaps.claimMap7,
    GeneratedMaps.claimMap8,
    GeneratedMaps.claimMap9,
    GeneratedMaps.claimMap10,
    GeneratedMaps.claimMap11,
    GeneratedMaps.claimMap12,
    GeneratedMaps.claimMap13]
def claimRows : List (List Row) :=
columnMaps.map fun columnMap => rows.map (Relabel.row columnMap)
def terminalCeRows : List Row := claimRows.flatten

def schedule : TerminalCeCompiler.Schedule where
commitmentStart := 0
commitmentEnd := 972
publicInputStart := 972
publicInputEnd := 14850
normStart := 14850
normEnd := 15390
evaluationsStart := 15390
evaluationsEnd := 15784
constantTermStart := 15784
constantTermEnd := 15790
ncChannelStart := 15790
ncChannelEnd := 21542

def program : TerminalCeCompiler.Program where
layout := layout
schedule := schedule
inputColumns := inputColumns
instructions := instructions

theorem packed_decode_ok : packedDecode = some instructions := by native_decide
theorem instructions_length : instructions.length = 21542 := by native_decide
theorem rows_length : rows.length = 21542 := by native_decide
theorem definitions_length : (definitions instructions).length = 5904 := by native_decide
theorem checks_length : (checks instructions).length = 15638 := by native_decide
theorem definitions_canonical :
∀ definition ∈ definitions instructions, definition.Canonical := by native_decide
theorem definitions_wellFormed :
WellFormed inputColumns (definitions instructions) := by native_decide
theorem checks_reference :
ChecksReference (knownAfter inputColumns (definitions instructions)) instructions := by native_decide
theorem column_maps_length : columnMaps.length = 14 := by native_decide
theorem column_maps_one :
∀ columnMap ∈ columnMaps, Relabel.column columnMap 0 = 0 := by native_decide
theorem column_maps_injective :
∀ columnMap ∈ columnMaps, columnMap.Nodup := by native_decide

theorem commitment_checks_match :
LinearOutputs.rows program.commitmentChecks =
checks program.commitmentInstructions := by native_decide
theorem public_program_match :
CheckedProgram.rows program.publicInstructions =
LinearOutputs.rows (TerminalCeCompiler.projectionChecks layout) := by
native_decide
theorem norm_program_match :
program.normInstructionsSlice =
TerminalCeCompiler.normInstructions layout := by native_decide
theorem evaluation_checks_match :
LinearOutputs.rows program.evaluationChecks =
checks program.evaluationInstructions := by native_decide
theorem constant_term_program_match :
CheckedProgram.rows program.constantTermInstructions =
LinearOutputs.rows (TerminalCeCompiler.constantTermChecks layout) := by
native_decide
theorem nc_checks_match :
LinearOutputs.rows program.ncChecks =
checks program.ncInstructions := by native_decide

theorem commitment_check_outputs :
program.commitmentChecks.map LinearOutputs.Check.output =
layout.commitmentCols := by native_decide
theorem evaluation_check_outputs :
program.evaluationChecks.map LinearOutputs.Check.output =
layout.evaluationCols.flatten := by native_decide
theorem nc_check_outputs :
program.ncChecks.map LinearOutputs.Check.output =
layout.ncEvaluationCols := by native_decide
theorem linear_checks_canonical :
LinearOutputs.Canonical program.commitmentChecks ∧
LinearOutputs.Canonical program.evaluationChecks ∧
LinearOutputs.Canonical program.ncChecks := by native_decide
theorem semantic_columns_known :
∀ column ∈ TerminalCeCompiler.semanticColumns layout,
column ∈ knownAfter inputColumns (definitions instructions) := by
native_decide
theorem semantic_columns_input :
∀ column ∈ TerminalCeCompiler.semanticColumns layout,
column ∈ inputColumns := by native_decide
theorem public_rows_reference_input :
∀ row ∈ CheckedProgram.rows program.publicInstructions,
∀ column ∈ rowRefs row, column ∈ inputColumns := by native_decide
theorem constant_term_rows_reference_input :
∀ row ∈ CheckedProgram.rows program.constantTermInstructions,
∀ column ∈ rowRefs row, column ∈ inputColumns := by native_decide
theorem norm_definitions_in_program :
∀ definition ∈ definitions (TerminalCeCompiler.normInstructions layout),
definition ∈ definitions instructions := by native_decide
theorem phase_partition :
instructions = program.commitmentInstructions ++
program.publicInstructions ++ program.normInstructionsSlice ++
program.evaluationInstructions ++ program.constantTermInstructions ++
program.ncInstructions := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe
