import Nightstream.Implementation.R1CS.FPrimeBaseProgramArtifact
import Nightstream.Implementation.R1CS.Poseidon2Call

/-! Generated exact Poseidon2 call-site certificates for the production plain F' base step. -/

namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls

open Nightstream.Implementation.R1CS.Poseidon2Call

set_option maxRecDepth 524288

def calls : List Call :=
[
  { rowStart := 17, rowEnd := 617, inputColumns := [48, 49, 50, 51, 47, 47, 47, 47], firstAllocatedColumn := 52 }
, { rowStart := 621, rowEnd := 1221, inputColumns := [652, 653, 654, 655, 648, 649, 650, 651], firstAllocatedColumn := 656 }
, { rowStart := 1224, rowEnd := 1824, inputColumns := [1256, 1257, 1258, 1251, 1252, 1253, 1254, 1255], firstAllocatedColumn := 1259 }
, { rowStart := 1825, rowEnd := 2425, inputColumns := [1859, 1852, 1853, 1854, 1855, 1856, 1857, 1858], firstAllocatedColumn := 1860 }
, { rowStart := 2439, rowEnd := 3039, inputColumns := [2470, 2471, 2472, 2473, 2469, 2469, 2469, 2469], firstAllocatedColumn := 2474 }
, { rowStart := 3043, rowEnd := 3643, inputColumns := [3074, 3075, 3076, 3077, 3070, 3071, 3072, 3073], firstAllocatedColumn := 3078 }
, { rowStart := 3647, rowEnd := 4247, inputColumns := [3678, 3679, 3680, 3681, 3674, 3675, 3676, 3677], firstAllocatedColumn := 3682 }
, { rowStart := 4251, rowEnd := 4851, inputColumns := [4282, 4283, 4284, 4285, 4278, 4279, 4280, 4281], firstAllocatedColumn := 4286 }
, { rowStart := 4855, rowEnd := 5455, inputColumns := [4886, 4887, 4888, 4889, 4882, 4883, 4884, 4885], firstAllocatedColumn := 4890 }
, { rowStart := 5457, rowEnd := 6057, inputColumns := [5490, 5491, 5484, 5485, 5486, 5487, 5488, 5489], firstAllocatedColumn := 5492 }
, { rowStart := 6058, rowEnd := 6658, inputColumns := [6092, 6085, 6086, 6087, 6088, 6089, 6090, 6091], firstAllocatedColumn := 6093 }
, { rowStart := 7746, rowEnd := 8346, inputColumns := [7553, 7554, 7555, 7556, 7552, 7552, 7552, 7552], firstAllocatedColumn := 7557 }
, { rowStart := 8350, rowEnd := 8950, inputColumns := [8157, 8158, 8159, 8160, 8153, 8154, 8155, 8156], firstAllocatedColumn := 8161 }
, { rowStart := 8954, rowEnd := 9554, inputColumns := [8761, 8762, 8763, 8764, 8757, 8758, 8759, 8760], firstAllocatedColumn := 8765 }
, { rowStart := 9558, rowEnd := 10158, inputColumns := [9365, 9366, 9367, 9368, 9361, 9362, 9363, 9364], firstAllocatedColumn := 9369 }
, { rowStart := 10162, rowEnd := 10762, inputColumns := [9969, 9970, 9971, 9972, 9965, 9966, 9967, 9968], firstAllocatedColumn := 9973 }
, { rowStart := 10765, rowEnd := 11365, inputColumns := [10573, 10574, 10575, 10568, 10569, 10570, 10571, 10572], firstAllocatedColumn := 10576 }
, { rowStart := 11366, rowEnd := 11966, inputColumns := [11176, 11169, 11170, 11171, 11172, 11173, 11174, 11175], firstAllocatedColumn := 11177 }
]

theorem calls_match_exact_ranges :
∀ call ∈ calls, call.Matches FPrimeBaseProgram.rows := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls
