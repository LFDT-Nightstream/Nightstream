import Nightstream.Implementation.R1CS.Ownership.FPrimeBase.FPrimeBaseProgramArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated exact Poseidon2 sponge certificates for the production plain F' base step. -/

namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def traces : List Trace :=
[
  { inputColumns := [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], zeroColumn := 50, zeroRow := 15, rounds := [
      { kind := .absorb [36, 37, 38, 39], stateBeforeColumns := [50, 50, 50, 50, 50, 50, 50, 50], permutationInputColumns := [51, 52, 53, 54, 50, 50, 50, 50], permutationOutputColumns := [647, 648, 649, 650, 651, 652, 653, 654], definingRows := [16, 17, 18, 19], call := { rowStart := 20, rowEnd := 620, inputColumns := [51, 52, 53, 54, 50, 50, 50, 50], firstAllocatedColumn := 55 } }
    , { kind := .absorb [40, 41, 42, 43], stateBeforeColumns := [647, 648, 649, 650, 651, 652, 653, 654], permutationInputColumns := [655, 656, 657, 658, 651, 652, 653, 654], permutationOutputColumns := [1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258], definingRows := [620, 621, 622, 623], call := { rowStart := 624, rowEnd := 1224, inputColumns := [655, 656, 657, 658, 651, 652, 653, 654], firstAllocatedColumn := 659 } }
    , { kind := .absorb [44, 45, 46, 47], stateBeforeColumns := [1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258], permutationInputColumns := [1259, 1260, 1261, 1262, 1255, 1256, 1257, 1258], permutationOutputColumns := [1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862], definingRows := [1224, 1225, 1226, 1227], call := { rowStart := 1228, rowEnd := 1828, inputColumns := [1259, 1260, 1261, 1262, 1255, 1256, 1257, 1258], firstAllocatedColumn := 1263 } }
    , { kind := .absorb [48, 49], stateBeforeColumns := [1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862], permutationInputColumns := [1863, 1864, 1857, 1858, 1859, 1860, 1861, 1862], permutationOutputColumns := [2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464], definingRows := [1828, 1829], call := { rowStart := 1830, rowEnd := 2430, inputColumns := [1863, 1864, 1857, 1858, 1859, 1860, 1861, 1862], firstAllocatedColumn := 1865 } }
    , { kind := .pad, stateBeforeColumns := [2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464], permutationInputColumns := [2465, 2458, 2459, 2460, 2461, 2462, 2463, 2464], permutationOutputColumns := [3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065], definingRows := [2430], call := { rowStart := 2431, rowEnd := 3031, inputColumns := [2465, 2458, 2459, 2460, 2461, 2462, 2463, 2464], firstAllocatedColumn := 2466 } }
    ], outputColumns := [3058, 3059, 3060, 3061] }
  , { inputColumns := [3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 11, 3074, 3058, 3059, 3060, 3061, 3058, 3059, 3060, 3061, 3058, 3059, 3060, 3061], zeroColumn := 3075, zeroRow := 3040, rounds := [
      { kind := .absorb [3066, 3067, 3068, 3069], stateBeforeColumns := [3075, 3075, 3075, 3075, 3075, 3075, 3075, 3075], permutationInputColumns := [3076, 3077, 3078, 3079, 3075, 3075, 3075, 3075], permutationOutputColumns := [3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679], definingRows := [3041, 3042, 3043, 3044], call := { rowStart := 3045, rowEnd := 3645, inputColumns := [3076, 3077, 3078, 3079, 3075, 3075, 3075, 3075], firstAllocatedColumn := 3080 } }
    , { kind := .absorb [3070, 3071, 3072, 3073], stateBeforeColumns := [3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679], permutationInputColumns := [3680, 3681, 3682, 3683, 3676, 3677, 3678, 3679], permutationOutputColumns := [4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283], definingRows := [3645, 3646, 3647, 3648], call := { rowStart := 3649, rowEnd := 4249, inputColumns := [3680, 3681, 3682, 3683, 3676, 3677, 3678, 3679], firstAllocatedColumn := 3684 } }
    , { kind := .absorb [11, 3074, 3058, 3059], stateBeforeColumns := [4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283], permutationInputColumns := [4284, 4285, 4286, 4287, 4280, 4281, 4282, 4283], permutationOutputColumns := [4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887], definingRows := [4249, 4250, 4251, 4252], call := { rowStart := 4253, rowEnd := 4853, inputColumns := [4284, 4285, 4286, 4287, 4280, 4281, 4282, 4283], firstAllocatedColumn := 4288 } }
    , { kind := .absorb [3060, 3061, 3058, 3059], stateBeforeColumns := [4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887], permutationInputColumns := [4888, 4889, 4890, 4891, 4884, 4885, 4886, 4887], permutationOutputColumns := [5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491], definingRows := [4853, 4854, 4855, 4856], call := { rowStart := 4857, rowEnd := 5457, inputColumns := [4888, 4889, 4890, 4891, 4884, 4885, 4886, 4887], firstAllocatedColumn := 4892 } }
    , { kind := .absorb [3060, 3061, 3058, 3059], stateBeforeColumns := [5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491], permutationInputColumns := [5492, 5493, 5494, 5495, 5488, 5489, 5490, 5491], permutationOutputColumns := [6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095], definingRows := [5457, 5458, 5459, 5460], call := { rowStart := 5461, rowEnd := 6061, inputColumns := [5492, 5493, 5494, 5495, 5488, 5489, 5490, 5491], firstAllocatedColumn := 5496 } }
    , { kind := .absorb [3060, 3061], stateBeforeColumns := [6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095], permutationInputColumns := [6096, 6097, 6090, 6091, 6092, 6093, 6094, 6095], permutationOutputColumns := [6690, 6691, 6692, 6693, 6694, 6695, 6696, 6697], definingRows := [6061, 6062], call := { rowStart := 6063, rowEnd := 6663, inputColumns := [6096, 6097, 6090, 6091, 6092, 6093, 6094, 6095], firstAllocatedColumn := 6098 } }
    , { kind := .pad, stateBeforeColumns := [6690, 6691, 6692, 6693, 6694, 6695, 6696, 6697], permutationInputColumns := [6698, 6691, 6692, 6693, 6694, 6695, 6696, 6697], permutationOutputColumns := [7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298], definingRows := [6663], call := { rowStart := 6664, rowEnd := 7264, inputColumns := [6698, 6691, 6692, 6693, 6694, 6695, 6696, 6697], firstAllocatedColumn := 6699 } }
    ], outputColumns := [7291, 7292, 7293, 7294] }
  , { inputColumns := [8085, 2, 3, 4, 5, 6, 7, 8, 9, 8086, 8087, 8088, 8089, 8156, 8157, 32, 33, 34, 35, 7753, 7754, 7755, 7756], zeroColumn := 8158, zeroRow := 8347, rounds := [
      { kind := .absorb [8085, 2, 3, 4], stateBeforeColumns := [8158, 8158, 8158, 8158, 8158, 8158, 8158, 8158], permutationInputColumns := [8159, 8160, 8161, 8162, 8158, 8158, 8158, 8158], permutationOutputColumns := [8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762], definingRows := [8348, 8349, 8350, 8351], call := { rowStart := 8352, rowEnd := 8952, inputColumns := [8159, 8160, 8161, 8162, 8158, 8158, 8158, 8158], firstAllocatedColumn := 8163 } }
    , { kind := .absorb [5, 6, 7, 8], stateBeforeColumns := [8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762], permutationInputColumns := [8763, 8764, 8765, 8766, 8759, 8760, 8761, 8762], permutationOutputColumns := [9359, 9360, 9361, 9362, 9363, 9364, 9365, 9366], definingRows := [8952, 8953, 8954, 8955], call := { rowStart := 8956, rowEnd := 9556, inputColumns := [8763, 8764, 8765, 8766, 8759, 8760, 8761, 8762], firstAllocatedColumn := 8767 } }
    , { kind := .absorb [9, 8086, 8087, 8088], stateBeforeColumns := [9359, 9360, 9361, 9362, 9363, 9364, 9365, 9366], permutationInputColumns := [9367, 9368, 9369, 9370, 9363, 9364, 9365, 9366], permutationOutputColumns := [9963, 9964, 9965, 9966, 9967, 9968, 9969, 9970], definingRows := [9556, 9557, 9558, 9559], call := { rowStart := 9560, rowEnd := 10160, inputColumns := [9367, 9368, 9369, 9370, 9363, 9364, 9365, 9366], firstAllocatedColumn := 9371 } }
    , { kind := .absorb [8089, 8156, 8157, 32], stateBeforeColumns := [9963, 9964, 9965, 9966, 9967, 9968, 9969, 9970], permutationInputColumns := [9971, 9972, 9973, 9974, 9967, 9968, 9969, 9970], permutationOutputColumns := [10567, 10568, 10569, 10570, 10571, 10572, 10573, 10574], definingRows := [10160, 10161, 10162, 10163], call := { rowStart := 10164, rowEnd := 10764, inputColumns := [9971, 9972, 9973, 9974, 9967, 9968, 9969, 9970], firstAllocatedColumn := 9975 } }
    , { kind := .absorb [33, 34, 35, 7753], stateBeforeColumns := [10567, 10568, 10569, 10570, 10571, 10572, 10573, 10574], permutationInputColumns := [10575, 10576, 10577, 10578, 10571, 10572, 10573, 10574], permutationOutputColumns := [11171, 11172, 11173, 11174, 11175, 11176, 11177, 11178], definingRows := [10764, 10765, 10766, 10767], call := { rowStart := 10768, rowEnd := 11368, inputColumns := [10575, 10576, 10577, 10578, 10571, 10572, 10573, 10574], firstAllocatedColumn := 10579 } }
    , { kind := .absorb [7754, 7755, 7756], stateBeforeColumns := [11171, 11172, 11173, 11174, 11175, 11176, 11177, 11178], permutationInputColumns := [11179, 11180, 11181, 11174, 11175, 11176, 11177, 11178], permutationOutputColumns := [11774, 11775, 11776, 11777, 11778, 11779, 11780, 11781], definingRows := [11368, 11369, 11370], call := { rowStart := 11371, rowEnd := 11971, inputColumns := [11179, 11180, 11181, 11174, 11175, 11176, 11177, 11178], firstAllocatedColumn := 11182 } }
    , { kind := .pad, stateBeforeColumns := [11774, 11775, 11776, 11777, 11778, 11779, 11780, 11781], permutationInputColumns := [11782, 11775, 11776, 11777, 11778, 11779, 11780, 11781], permutationOutputColumns := [12375, 12376, 12377, 12378, 12379, 12380, 12381, 12382], definingRows := [11971], call := { rowStart := 11972, rowEnd := 12572, inputColumns := [11782, 11775, 11776, 11777, 11778, 11779, 11780, 11781], firstAllocatedColumn := 11783 } }
    ], outputColumns := [12375, 12376, 12377, 12378] }
]

theorem traces_accepted :
traces.all (fun trace => decide (trace.Valid FPrimeBaseProgram.rows)) = true := by
native_decide

theorem traces_valid :
∀ trace ∈ traces, trace.Valid FPrimeBaseProgram.rows := by
intro trace member
exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)

end Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes
