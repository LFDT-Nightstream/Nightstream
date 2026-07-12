import Nightstream.Implementation.R1CS.Ownership.FPrimeBase.FPrimeBaseProgramArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated exact Poseidon2 sponge certificates for the production plain F' base step. -/

namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes

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
  , { inputColumns := [2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 11, 2468, 2452, 2453, 2454, 2455, 2452, 2453, 2454, 2455, 2452, 2453, 2454, 2455], zeroColumn := 2469, zeroRow := 2434, rounds := [
      { kind := .absorb [2460, 2461, 2462, 2463], stateBeforeColumns := [2469, 2469, 2469, 2469, 2469, 2469, 2469, 2469], permutationInputColumns := [2470, 2471, 2472, 2473, 2469, 2469, 2469, 2469], permutationOutputColumns := [3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073], definingRows := [2435, 2436, 2437, 2438], call := { rowStart := 2439, rowEnd := 3039, inputColumns := [2470, 2471, 2472, 2473, 2469, 2469, 2469, 2469], firstAllocatedColumn := 2474 } }
    , { kind := .absorb [2464, 2465, 2466, 2467], stateBeforeColumns := [3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073], permutationInputColumns := [3074, 3075, 3076, 3077, 3070, 3071, 3072, 3073], permutationOutputColumns := [3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677], definingRows := [3039, 3040, 3041, 3042], call := { rowStart := 3043, rowEnd := 3643, inputColumns := [3074, 3075, 3076, 3077, 3070, 3071, 3072, 3073], firstAllocatedColumn := 3078 } }
    , { kind := .absorb [11, 2468, 2452, 2453], stateBeforeColumns := [3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677], permutationInputColumns := [3678, 3679, 3680, 3681, 3674, 3675, 3676, 3677], permutationOutputColumns := [4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281], definingRows := [3643, 3644, 3645, 3646], call := { rowStart := 3647, rowEnd := 4247, inputColumns := [3678, 3679, 3680, 3681, 3674, 3675, 3676, 3677], firstAllocatedColumn := 3682 } }
    , { kind := .absorb [2454, 2455, 2452, 2453], stateBeforeColumns := [4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281], permutationInputColumns := [4282, 4283, 4284, 4285, 4278, 4279, 4280, 4281], permutationOutputColumns := [4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885], definingRows := [4247, 4248, 4249, 4250], call := { rowStart := 4251, rowEnd := 4851, inputColumns := [4282, 4283, 4284, 4285, 4278, 4279, 4280, 4281], firstAllocatedColumn := 4286 } }
    , { kind := .absorb [2454, 2455, 2452, 2453], stateBeforeColumns := [4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885], permutationInputColumns := [4886, 4887, 4888, 4889, 4882, 4883, 4884, 4885], permutationOutputColumns := [5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489], definingRows := [4851, 4852, 4853, 4854], call := { rowStart := 4855, rowEnd := 5455, inputColumns := [4886, 4887, 4888, 4889, 4882, 4883, 4884, 4885], firstAllocatedColumn := 4890 } }
    , { kind := .absorb [2454, 2455], stateBeforeColumns := [5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489], permutationInputColumns := [5490, 5491, 5484, 5485, 5486, 5487, 5488, 5489], permutationOutputColumns := [6084, 6085, 6086, 6087, 6088, 6089, 6090, 6091], definingRows := [5455, 5456], call := { rowStart := 5457, rowEnd := 6057, inputColumns := [5490, 5491, 5484, 5485, 5486, 5487, 5488, 5489], firstAllocatedColumn := 5492 } }
    , { kind := .pad, stateBeforeColumns := [6084, 6085, 6086, 6087, 6088, 6089, 6090, 6091], permutationInputColumns := [6092, 6085, 6086, 6087, 6088, 6089, 6090, 6091], permutationOutputColumns := [6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692], definingRows := [6057], call := { rowStart := 6058, rowEnd := 6658, inputColumns := [6092, 6085, 6086, 6087, 6088, 6089, 6090, 6091], firstAllocatedColumn := 6093 } }
    ], outputColumns := [6685, 6686, 6687, 6688] }
  , { inputColumns := [7479, 2, 3, 4, 5, 6, 7, 8, 9, 7480, 7481, 7482, 7483, 7550, 7551, 32, 33, 34, 35, 7147, 7148, 7149, 7150], zeroColumn := 7552, zeroRow := 7741, rounds := [
      { kind := .absorb [7479, 2, 3, 4], stateBeforeColumns := [7552, 7552, 7552, 7552, 7552, 7552, 7552, 7552], permutationInputColumns := [7553, 7554, 7555, 7556, 7552, 7552, 7552, 7552], permutationOutputColumns := [8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156], definingRows := [7742, 7743, 7744, 7745], call := { rowStart := 7746, rowEnd := 8346, inputColumns := [7553, 7554, 7555, 7556, 7552, 7552, 7552, 7552], firstAllocatedColumn := 7557 } }
    , { kind := .absorb [5, 6, 7, 8], stateBeforeColumns := [8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156], permutationInputColumns := [8157, 8158, 8159, 8160, 8153, 8154, 8155, 8156], permutationOutputColumns := [8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760], definingRows := [8346, 8347, 8348, 8349], call := { rowStart := 8350, rowEnd := 8950, inputColumns := [8157, 8158, 8159, 8160, 8153, 8154, 8155, 8156], firstAllocatedColumn := 8161 } }
    , { kind := .absorb [9, 7480, 7481, 7482], stateBeforeColumns := [8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760], permutationInputColumns := [8761, 8762, 8763, 8764, 8757, 8758, 8759, 8760], permutationOutputColumns := [9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364], definingRows := [8950, 8951, 8952, 8953], call := { rowStart := 8954, rowEnd := 9554, inputColumns := [8761, 8762, 8763, 8764, 8757, 8758, 8759, 8760], firstAllocatedColumn := 8765 } }
    , { kind := .absorb [7483, 7550, 7551, 32], stateBeforeColumns := [9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364], permutationInputColumns := [9365, 9366, 9367, 9368, 9361, 9362, 9363, 9364], permutationOutputColumns := [9961, 9962, 9963, 9964, 9965, 9966, 9967, 9968], definingRows := [9554, 9555, 9556, 9557], call := { rowStart := 9558, rowEnd := 10158, inputColumns := [9365, 9366, 9367, 9368, 9361, 9362, 9363, 9364], firstAllocatedColumn := 9369 } }
    , { kind := .absorb [33, 34, 35, 7147], stateBeforeColumns := [9961, 9962, 9963, 9964, 9965, 9966, 9967, 9968], permutationInputColumns := [9969, 9970, 9971, 9972, 9965, 9966, 9967, 9968], permutationOutputColumns := [10565, 10566, 10567, 10568, 10569, 10570, 10571, 10572], definingRows := [10158, 10159, 10160, 10161], call := { rowStart := 10162, rowEnd := 10762, inputColumns := [9969, 9970, 9971, 9972, 9965, 9966, 9967, 9968], firstAllocatedColumn := 9973 } }
    , { kind := .absorb [7148, 7149, 7150], stateBeforeColumns := [10565, 10566, 10567, 10568, 10569, 10570, 10571, 10572], permutationInputColumns := [10573, 10574, 10575, 10568, 10569, 10570, 10571, 10572], permutationOutputColumns := [11168, 11169, 11170, 11171, 11172, 11173, 11174, 11175], definingRows := [10762, 10763, 10764], call := { rowStart := 10765, rowEnd := 11365, inputColumns := [10573, 10574, 10575, 10568, 10569, 10570, 10571, 10572], firstAllocatedColumn := 10576 } }
    , { kind := .pad, stateBeforeColumns := [11168, 11169, 11170, 11171, 11172, 11173, 11174, 11175], permutationInputColumns := [11176, 11169, 11170, 11171, 11172, 11173, 11174, 11175], permutationOutputColumns := [11769, 11770, 11771, 11772, 11773, 11774, 11775, 11776], definingRows := [11365], call := { rowStart := 11366, rowEnd := 11966, inputColumns := [11176, 11169, 11170, 11171, 11172, 11173, 11174, 11175], firstAllocatedColumn := 11177 } }
    ], outputColumns := [11769, 11770, 11771, 11772] }
]

theorem traces_accepted :
traces.all (fun trace => decide (trace.Valid FPrimeBaseProgram.rows)) = true := by
native_decide

theorem traces_valid :
∀ trace ∈ traces, trace.Valid FPrimeBaseProgram.rows := by
intro trace member
exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)

end Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes
