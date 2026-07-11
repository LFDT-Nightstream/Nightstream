import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseArtifact
import Nightstream.Implementation.R1CS.Poseidon2Sponge

/-! Generated sponge certificates for the exact composed base owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBasePoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def traces : List Trace :=
[
  { inputColumns := [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46], zeroColumn := 47, zeroRow := 12, rounds := [
      { kind := .absorb [36, 37, 38, 39], stateBeforeColumns := [47, 47, 47, 47, 47, 47, 47, 47], permutationInputColumns := [48, 49, 50, 51, 47, 47, 47, 47], permutationOutputColumns := [644, 645, 646, 647, 648, 649, 650, 651], definingRows := [13, 14, 15, 16], call := { rowStart := 17, rowEnd := 617, inputColumns := [48, 49, 50, 51, 47, 47, 47, 47], firstAllocatedColumn := 52 } }
    , { kind := .absorb [40, 41, 42, 43], stateBeforeColumns := [644, 645, 646, 647, 648, 649, 650, 651], permutationInputColumns := [652, 653, 654, 655, 648, 649, 650, 651], permutationOutputColumns := [1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255], definingRows := [617, 618, 619, 620], call := { rowStart := 621, rowEnd := 1221, inputColumns := [652, 653, 654, 655, 648, 649, 650, 651], firstAllocatedColumn := 656 } }
    , { kind := .absorb [44, 45, 46], stateBeforeColumns := [1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255], permutationInputColumns := [1256, 1257, 1258, 1251, 1252, 1253, 1254, 1255], permutationOutputColumns := [1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858], definingRows := [1221, 1222, 1223], call := { rowStart := 1224, rowEnd := 1824, inputColumns := [1256, 1257, 1258, 1251, 1252, 1253, 1254, 1255], firstAllocatedColumn := 1259 } }
    , { kind := .pad, stateBeforeColumns := [1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858], permutationInputColumns := [1859, 1852, 1853, 1854, 1855, 1856, 1857, 1858], permutationOutputColumns := [2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459], definingRows := [1824], call := { rowStart := 1825, rowEnd := 2425, inputColumns := [1859, 1852, 1853, 1854, 1855, 1856, 1857, 1858], firstAllocatedColumn := 1860 } }
    ], outputColumns := [2452, 2453, 2454, 2455] }
  , { inputColumns := [2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 11, 2468, 2452, 2453, 2454, 2455], zeroColumn := 2469, zeroRow := 2434, rounds := [
      { kind := .absorb [2460, 2461, 2462, 2463], stateBeforeColumns := [2469, 2469, 2469, 2469, 2469, 2469, 2469, 2469], permutationInputColumns := [2470, 2471, 2472, 2473, 2469, 2469, 2469, 2469], permutationOutputColumns := [3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073], definingRows := [2435, 2436, 2437, 2438], call := { rowStart := 2439, rowEnd := 3039, inputColumns := [2470, 2471, 2472, 2473, 2469, 2469, 2469, 2469], firstAllocatedColumn := 2474 } }
    , { kind := .absorb [2464, 2465, 2466, 2467], stateBeforeColumns := [3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073], permutationInputColumns := [3074, 3075, 3076, 3077, 3070, 3071, 3072, 3073], permutationOutputColumns := [3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677], definingRows := [3039, 3040, 3041, 3042], call := { rowStart := 3043, rowEnd := 3643, inputColumns := [3074, 3075, 3076, 3077, 3070, 3071, 3072, 3073], firstAllocatedColumn := 3078 } }
    , { kind := .absorb [11, 2468, 2452, 2453], stateBeforeColumns := [3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677], permutationInputColumns := [3678, 3679, 3680, 3681, 3674, 3675, 3676, 3677], permutationOutputColumns := [4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281], definingRows := [3643, 3644, 3645, 3646], call := { rowStart := 3647, rowEnd := 4247, inputColumns := [3678, 3679, 3680, 3681, 3674, 3675, 3676, 3677], firstAllocatedColumn := 3682 } }
    , { kind := .absorb [2454, 2455], stateBeforeColumns := [4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281], permutationInputColumns := [4282, 4283, 4276, 4277, 4278, 4279, 4280, 4281], permutationOutputColumns := [4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883], definingRows := [4247, 4248], call := { rowStart := 4249, rowEnd := 4849, inputColumns := [4282, 4283, 4276, 4277, 4278, 4279, 4280, 4281], firstAllocatedColumn := 4284 } }
    , { kind := .pad, stateBeforeColumns := [4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883], permutationInputColumns := [4884, 4877, 4878, 4879, 4880, 4881, 4882, 4883], permutationOutputColumns := [5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484], definingRows := [4849], call := { rowStart := 4850, rowEnd := 5450, inputColumns := [4884, 4877, 4878, 4879, 4880, 4881, 4882, 4883], firstAllocatedColumn := 4885 } }
    ], outputColumns := [5477, 5478, 5479, 5480] }
  , { inputColumns := [6271, 2, 3, 4, 5, 6, 7, 8, 9, 6272, 6273, 6274, 6275, 6342, 6343, 32, 33, 34, 35, 5939, 5940, 5941, 5942], zeroColumn := 6344, zeroRow := 6533, rounds := [
      { kind := .absorb [6271, 2, 3, 4], stateBeforeColumns := [6344, 6344, 6344, 6344, 6344, 6344, 6344, 6344], permutationInputColumns := [6345, 6346, 6347, 6348, 6344, 6344, 6344, 6344], permutationOutputColumns := [6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948], definingRows := [6534, 6535, 6536, 6537], call := { rowStart := 6538, rowEnd := 7138, inputColumns := [6345, 6346, 6347, 6348, 6344, 6344, 6344, 6344], firstAllocatedColumn := 6349 } }
    , { kind := .absorb [5, 6, 7, 8], stateBeforeColumns := [6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948], permutationInputColumns := [6949, 6950, 6951, 6952, 6945, 6946, 6947, 6948], permutationOutputColumns := [7545, 7546, 7547, 7548, 7549, 7550, 7551, 7552], definingRows := [7138, 7139, 7140, 7141], call := { rowStart := 7142, rowEnd := 7742, inputColumns := [6949, 6950, 6951, 6952, 6945, 6946, 6947, 6948], firstAllocatedColumn := 6953 } }
    , { kind := .absorb [9, 6272, 6273, 6274], stateBeforeColumns := [7545, 7546, 7547, 7548, 7549, 7550, 7551, 7552], permutationInputColumns := [7553, 7554, 7555, 7556, 7549, 7550, 7551, 7552], permutationOutputColumns := [8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156], definingRows := [7742, 7743, 7744, 7745], call := { rowStart := 7746, rowEnd := 8346, inputColumns := [7553, 7554, 7555, 7556, 7549, 7550, 7551, 7552], firstAllocatedColumn := 7557 } }
    , { kind := .absorb [6275, 6342, 6343, 32], stateBeforeColumns := [8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156], permutationInputColumns := [8157, 8158, 8159, 8160, 8153, 8154, 8155, 8156], permutationOutputColumns := [8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760], definingRows := [8346, 8347, 8348, 8349], call := { rowStart := 8350, rowEnd := 8950, inputColumns := [8157, 8158, 8159, 8160, 8153, 8154, 8155, 8156], firstAllocatedColumn := 8161 } }
    , { kind := .absorb [33, 34, 35, 5939], stateBeforeColumns := [8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760], permutationInputColumns := [8761, 8762, 8763, 8764, 8757, 8758, 8759, 8760], permutationOutputColumns := [9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364], definingRows := [8950, 8951, 8952, 8953], call := { rowStart := 8954, rowEnd := 9554, inputColumns := [8761, 8762, 8763, 8764, 8757, 8758, 8759, 8760], firstAllocatedColumn := 8765 } }
    , { kind := .absorb [5940, 5941, 5942], stateBeforeColumns := [9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364], permutationInputColumns := [9365, 9366, 9367, 9360, 9361, 9362, 9363, 9364], permutationOutputColumns := [9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967], definingRows := [9554, 9555, 9556], call := { rowStart := 9557, rowEnd := 10157, inputColumns := [9365, 9366, 9367, 9360, 9361, 9362, 9363, 9364], firstAllocatedColumn := 9368 } }
    , { kind := .pad, stateBeforeColumns := [9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967], permutationInputColumns := [9968, 9961, 9962, 9963, 9964, 9965, 9966, 9967], permutationOutputColumns := [10561, 10562, 10563, 10564, 10565, 10566, 10567, 10568], definingRows := [10157], call := { rowStart := 10158, rowEnd := 10758, inputColumns := [9968, 9961, 9962, 9963, 9964, 9965, 9966, 9967], firstAllocatedColumn := 9969 } }
    ], outputColumns := [10561, 10562, 10563, 10564] }
]

theorem traces_accepted :
traces.all (fun trace => decide (trace.Valid FPrimeFullHistoryBase.rows)) = true := by
native_decide

theorem traces_valid :
∀ trace ∈ traces, trace.Valid FPrimeFullHistoryBase.rows := by
intro trace member
exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)

def xOutTrace : Trace := traces[2]!

theorem xOutTrace_output : xOutTrace.outputColumns = FPrimeFullHistoryBase.xOutColumns := by
native_decide

theorem xOutTrace_valid : xOutTrace.Valid FPrimeFullHistoryBase.rows := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBasePoseidonHashes
