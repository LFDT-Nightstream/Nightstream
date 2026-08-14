import Nightstream.Implementation.R1CS.Ownership.Nebula.NebulaProgramBindingArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated exact Poseidon2 sponge trace for the Nebula program binding. -/

namespace Nightstream.Implementation.R1CS.NebulaProgramBindingPoseidon

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

def trace : Trace :=
{ inputColumns := [67, 68, 69, 70, 71, 72, 73, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], zeroColumn := 74, zeroRow := 0, rounds := [
      { kind := .absorb [67, 68, 69, 70], stateBeforeColumns := [74, 74, 74, 74, 74, 74, 74, 74], permutationInputColumns := [75, 76, 77, 78, 74, 74, 74, 74], permutationOutputColumns := [671, 672, 673, 674, 675, 676, 677, 678], definingRows := [1, 2, 3, 4], call := { rowStart := 5, rowEnd := 605, inputColumns := [75, 76, 77, 78, 74, 74, 74, 74], firstAllocatedColumn := 79 } }
    , { kind := .absorb [71, 72, 73, 55], stateBeforeColumns := [671, 672, 673, 674, 675, 676, 677, 678], permutationInputColumns := [679, 680, 681, 682, 675, 676, 677, 678], permutationOutputColumns := [1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282], definingRows := [605, 606, 607, 608], call := { rowStart := 609, rowEnd := 1209, inputColumns := [679, 680, 681, 682, 675, 676, 677, 678], firstAllocatedColumn := 683 } }
    , { kind := .absorb [56, 57, 58, 59], stateBeforeColumns := [1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282], permutationInputColumns := [1283, 1284, 1285, 1286, 1279, 1280, 1281, 1282], permutationOutputColumns := [1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886], definingRows := [1209, 1210, 1211, 1212], call := { rowStart := 1213, rowEnd := 1813, inputColumns := [1283, 1284, 1285, 1286, 1279, 1280, 1281, 1282], firstAllocatedColumn := 1287 } }
    , { kind := .absorb [60, 61, 62, 63], stateBeforeColumns := [1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886], permutationInputColumns := [1887, 1888, 1889, 1890, 1883, 1884, 1885, 1886], permutationOutputColumns := [2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490], definingRows := [1813, 1814, 1815, 1816], call := { rowStart := 1817, rowEnd := 2417, inputColumns := [1887, 1888, 1889, 1890, 1883, 1884, 1885, 1886], firstAllocatedColumn := 1891 } }
    , { kind := .absorb [64, 65, 66], stateBeforeColumns := [2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490], permutationInputColumns := [2491, 2492, 2493, 2486, 2487, 2488, 2489, 2490], permutationOutputColumns := [3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093], definingRows := [2417, 2418, 2419], call := { rowStart := 2420, rowEnd := 3020, inputColumns := [2491, 2492, 2493, 2486, 2487, 2488, 2489, 2490], firstAllocatedColumn := 2494 } }
    , { kind := .pad, stateBeforeColumns := [3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093], permutationInputColumns := [3094, 3087, 3088, 3089, 3090, 3091, 3092, 3093], permutationOutputColumns := [3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694], definingRows := [3020], call := { rowStart := 3021, rowEnd := 3621, inputColumns := [3094, 3087, 3088, 3089, 3090, 3091, 3092, 3093], firstAllocatedColumn := 3095 } }
    ], outputColumns := [3687, 3688, 3689, 3690] }

def inputFieldCount : Nat := 19
def rowStart : Nat := 7
def traceRowCount : Nat := 3621

theorem trace_valid :
trace.Valid trace.rows := by constructor <;> decide
theorem trace_rows_exact :
(NebulaProgramBinding.rows.drop rowStart).take traceRowCount =
trace.rows := by decide
theorem trace_input_layout :
trace.inputColumns = NebulaProgramBinding.tagColumns ++
NebulaProgramBinding.initialSemanticStateColumns ++
NebulaProgramBinding.planDigestColumns ++
NebulaProgramBinding.initialMemoryDigestColumns := by decide
theorem trace_output_layout :
trace.outputColumns = NebulaProgramBinding.computedBindingColumns := by decide

end Nightstream.Implementation.R1CS.NebulaProgramBindingPoseidon
