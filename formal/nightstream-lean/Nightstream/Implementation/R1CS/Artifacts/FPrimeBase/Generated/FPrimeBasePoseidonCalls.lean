import Nightstream.Implementation.R1CS.Ownership.FPrimeBase.FPrimeBaseProgramArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Call

/-! Generated exact Poseidon2 call-site certificates for the production plain F' base step. -/

namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls

open Nightstream.Implementation.R1CS.Poseidon2Call

set_option maxRecDepth 524288

def calls : List Call :=
[
  { rowStart := 20, rowEnd := 620, inputColumns := [51, 52, 53, 54, 50, 50, 50, 50], firstAllocatedColumn := 55 }
, { rowStart := 624, rowEnd := 1224, inputColumns := [655, 656, 657, 658, 651, 652, 653, 654], firstAllocatedColumn := 659 }
, { rowStart := 1228, rowEnd := 1828, inputColumns := [1259, 1260, 1261, 1262, 1255, 1256, 1257, 1258], firstAllocatedColumn := 1263 }
, { rowStart := 1830, rowEnd := 2430, inputColumns := [1863, 1864, 1857, 1858, 1859, 1860, 1861, 1862], firstAllocatedColumn := 1865 }
, { rowStart := 2431, rowEnd := 3031, inputColumns := [2465, 2458, 2459, 2460, 2461, 2462, 2463, 2464], firstAllocatedColumn := 2466 }
, { rowStart := 3045, rowEnd := 3645, inputColumns := [3076, 3077, 3078, 3079, 3075, 3075, 3075, 3075], firstAllocatedColumn := 3080 }
, { rowStart := 3649, rowEnd := 4249, inputColumns := [3680, 3681, 3682, 3683, 3676, 3677, 3678, 3679], firstAllocatedColumn := 3684 }
, { rowStart := 4253, rowEnd := 4853, inputColumns := [4284, 4285, 4286, 4287, 4280, 4281, 4282, 4283], firstAllocatedColumn := 4288 }
, { rowStart := 4857, rowEnd := 5457, inputColumns := [4888, 4889, 4890, 4891, 4884, 4885, 4886, 4887], firstAllocatedColumn := 4892 }
, { rowStart := 5461, rowEnd := 6061, inputColumns := [5492, 5493, 5494, 5495, 5488, 5489, 5490, 5491], firstAllocatedColumn := 5496 }
, { rowStart := 6063, rowEnd := 6663, inputColumns := [6096, 6097, 6090, 6091, 6092, 6093, 6094, 6095], firstAllocatedColumn := 6098 }
, { rowStart := 6664, rowEnd := 7264, inputColumns := [6698, 6691, 6692, 6693, 6694, 6695, 6696, 6697], firstAllocatedColumn := 6699 }
, { rowStart := 8352, rowEnd := 8952, inputColumns := [8159, 8160, 8161, 8162, 8158, 8158, 8158, 8158], firstAllocatedColumn := 8163 }
, { rowStart := 8956, rowEnd := 9556, inputColumns := [8763, 8764, 8765, 8766, 8759, 8760, 8761, 8762], firstAllocatedColumn := 8767 }
, { rowStart := 9560, rowEnd := 10160, inputColumns := [9367, 9368, 9369, 9370, 9363, 9364, 9365, 9366], firstAllocatedColumn := 9371 }
, { rowStart := 10164, rowEnd := 10764, inputColumns := [9971, 9972, 9973, 9974, 9967, 9968, 9969, 9970], firstAllocatedColumn := 9975 }
, { rowStart := 10768, rowEnd := 11368, inputColumns := [10575, 10576, 10577, 10578, 10571, 10572, 10573, 10574], firstAllocatedColumn := 10579 }
, { rowStart := 11371, rowEnd := 11971, inputColumns := [11179, 11180, 11181, 11174, 11175, 11176, 11177, 11178], firstAllocatedColumn := 11182 }
, { rowStart := 11972, rowEnd := 12572, inputColumns := [11782, 11775, 11776, 11777, 11778, 11779, 11780, 11781], firstAllocatedColumn := 11783 }
]

theorem calls_match_exact_ranges :
∀ call ∈ calls, call.Matches FPrimeBaseProgram.rows := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls
