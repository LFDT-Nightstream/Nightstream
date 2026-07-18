import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPublicPinsArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated sponge certificate for the exact public-image pins. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def xOutTrace : Trace :=
  { inputColumns := [3491505, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 3491506, 3491507, 3491508, 3491509, 3491510, 3491511, 10864, 10865, 10866, 10867, 3489705, 3489706, 3489707, 3489708], zeroColumn := 3491512, zeroRow := 50, rounds := [
      { kind := .absorb [3491505, 10834, 10835, 10836], stateBeforeColumns := [3491512, 3491512, 3491512, 3491512, 3491512, 3491512, 3491512, 3491512], permutationInputColumns := [3491513, 3491514, 3491515, 3491516, 3491512, 3491512, 3491512, 3491512], permutationOutputColumns := [3492109, 3492110, 3492111, 3492112, 3492113, 3492114, 3492115, 3492116], definingRows := [51, 52, 53, 54], call := { rowStart := 55, rowEnd := 655, inputColumns := [3491513, 3491514, 3491515, 3491516, 3491512, 3491512, 3491512, 3491512], firstAllocatedColumn := 3491517 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [3492109, 3492110, 3492111, 3492112, 3492113, 3492114, 3492115, 3492116], permutationInputColumns := [3492117, 3492118, 3492119, 3492120, 3492113, 3492114, 3492115, 3492116], permutationOutputColumns := [3492713, 3492714, 3492715, 3492716, 3492717, 3492718, 3492719, 3492720], definingRows := [655, 656, 657, 658], call := { rowStart := 659, rowEnd := 1259, inputColumns := [3492117, 3492118, 3492119, 3492120, 3492113, 3492114, 3492115, 3492116], firstAllocatedColumn := 3492121 } }
    , { kind := .absorb [10841, 3491506, 3491507, 3491508], stateBeforeColumns := [3492713, 3492714, 3492715, 3492716, 3492717, 3492718, 3492719, 3492720], permutationInputColumns := [3492721, 3492722, 3492723, 3492724, 3492717, 3492718, 3492719, 3492720], permutationOutputColumns := [3493317, 3493318, 3493319, 3493320, 3493321, 3493322, 3493323, 3493324], definingRows := [1259, 1260, 1261, 1262], call := { rowStart := 1263, rowEnd := 1863, inputColumns := [3492721, 3492722, 3492723, 3492724, 3492717, 3492718, 3492719, 3492720], firstAllocatedColumn := 3492725 } }
    , { kind := .absorb [3491509, 3491510, 3491511, 10864], stateBeforeColumns := [3493317, 3493318, 3493319, 3493320, 3493321, 3493322, 3493323, 3493324], permutationInputColumns := [3493325, 3493326, 3493327, 3493328, 3493321, 3493322, 3493323, 3493324], permutationOutputColumns := [3493921, 3493922, 3493923, 3493924, 3493925, 3493926, 3493927, 3493928], definingRows := [1863, 1864, 1865, 1866], call := { rowStart := 1867, rowEnd := 2467, inputColumns := [3493325, 3493326, 3493327, 3493328, 3493321, 3493322, 3493323, 3493324], firstAllocatedColumn := 3493329 } }
    , { kind := .absorb [10865, 10866, 10867, 3489705], stateBeforeColumns := [3493921, 3493922, 3493923, 3493924, 3493925, 3493926, 3493927, 3493928], permutationInputColumns := [3493929, 3493930, 3493931, 3493932, 3493925, 3493926, 3493927, 3493928], permutationOutputColumns := [3494525, 3494526, 3494527, 3494528, 3494529, 3494530, 3494531, 3494532], definingRows := [2467, 2468, 2469, 2470], call := { rowStart := 2471, rowEnd := 3071, inputColumns := [3493929, 3493930, 3493931, 3493932, 3493925, 3493926, 3493927, 3493928], firstAllocatedColumn := 3493933 } }
    , { kind := .absorb [3489706, 3489707, 3489708], stateBeforeColumns := [3494525, 3494526, 3494527, 3494528, 3494529, 3494530, 3494531, 3494532], permutationInputColumns := [3494533, 3494534, 3494535, 3494528, 3494529, 3494530, 3494531, 3494532], permutationOutputColumns := [3495128, 3495129, 3495130, 3495131, 3495132, 3495133, 3495134, 3495135], definingRows := [3071, 3072, 3073], call := { rowStart := 3074, rowEnd := 3674, inputColumns := [3494533, 3494534, 3494535, 3494528, 3494529, 3494530, 3494531, 3494532], firstAllocatedColumn := 3494536 } }
    , { kind := .pad, stateBeforeColumns := [3495128, 3495129, 3495130, 3495131, 3495132, 3495133, 3495134, 3495135], permutationInputColumns := [3495136, 3495129, 3495130, 3495131, 3495132, 3495133, 3495134, 3495135], permutationOutputColumns := [3495729, 3495730, 3495731, 3495732, 3495733, 3495734, 3495735, 3495736], definingRows := [3674], call := { rowStart := 3675, rowEnd := 4275, inputColumns := [3495136, 3495129, 3495130, 3495131, 3495132, 3495133, 3495134, 3495135], firstAllocatedColumn := 3495137 } }
    ], outputColumns := [3495729, 3495730, 3495731, 3495732] }

theorem xOutTrace_valid :
xOutTrace.Valid FPrimeFullHistoryPublicPins.rows := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsPoseidonHashes
