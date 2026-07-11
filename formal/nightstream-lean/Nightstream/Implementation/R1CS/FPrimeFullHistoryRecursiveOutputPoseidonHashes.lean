import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputArtifact
import Nightstream.Implementation.R1CS.Poseidon2Sponge

/-! Generated sponge certificate for the exact recursive output owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def xOutTrace : Trace :=
  { inputColumns := [924847, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 924848, 924849, 924850, 924851, 924852, 924853, 10864, 10865, 10866, 10867, 924511, 924512, 924513, 924514], zeroColumn := 924854, zeroRow := 11, rounds := [
      { kind := .absorb [924847, 10834, 10835, 10836], stateBeforeColumns := [924854, 924854, 924854, 924854, 924854, 924854, 924854, 924854], permutationInputColumns := [924855, 924856, 924857, 924858, 924854, 924854, 924854, 924854], permutationOutputColumns := [925451, 925452, 925453, 925454, 925455, 925456, 925457, 925458], definingRows := [12, 13, 14, 15], call := { rowStart := 16, rowEnd := 616, inputColumns := [924855, 924856, 924857, 924858, 924854, 924854, 924854, 924854], firstAllocatedColumn := 924859 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [925451, 925452, 925453, 925454, 925455, 925456, 925457, 925458], permutationInputColumns := [925459, 925460, 925461, 925462, 925455, 925456, 925457, 925458], permutationOutputColumns := [926055, 926056, 926057, 926058, 926059, 926060, 926061, 926062], definingRows := [616, 617, 618, 619], call := { rowStart := 620, rowEnd := 1220, inputColumns := [925459, 925460, 925461, 925462, 925455, 925456, 925457, 925458], firstAllocatedColumn := 925463 } }
    , { kind := .absorb [10841, 924848, 924849, 924850], stateBeforeColumns := [926055, 926056, 926057, 926058, 926059, 926060, 926061, 926062], permutationInputColumns := [926063, 926064, 926065, 926066, 926059, 926060, 926061, 926062], permutationOutputColumns := [926659, 926660, 926661, 926662, 926663, 926664, 926665, 926666], definingRows := [1220, 1221, 1222, 1223], call := { rowStart := 1224, rowEnd := 1824, inputColumns := [926063, 926064, 926065, 926066, 926059, 926060, 926061, 926062], firstAllocatedColumn := 926067 } }
    , { kind := .absorb [924851, 924852, 924853, 10864], stateBeforeColumns := [926659, 926660, 926661, 926662, 926663, 926664, 926665, 926666], permutationInputColumns := [926667, 926668, 926669, 926670, 926663, 926664, 926665, 926666], permutationOutputColumns := [927263, 927264, 927265, 927266, 927267, 927268, 927269, 927270], definingRows := [1824, 1825, 1826, 1827], call := { rowStart := 1828, rowEnd := 2428, inputColumns := [926667, 926668, 926669, 926670, 926663, 926664, 926665, 926666], firstAllocatedColumn := 926671 } }
    , { kind := .absorb [10865, 10866, 10867, 924511], stateBeforeColumns := [927263, 927264, 927265, 927266, 927267, 927268, 927269, 927270], permutationInputColumns := [927271, 927272, 927273, 927274, 927267, 927268, 927269, 927270], permutationOutputColumns := [927867, 927868, 927869, 927870, 927871, 927872, 927873, 927874], definingRows := [2428, 2429, 2430, 2431], call := { rowStart := 2432, rowEnd := 3032, inputColumns := [927271, 927272, 927273, 927274, 927267, 927268, 927269, 927270], firstAllocatedColumn := 927275 } }
    , { kind := .absorb [924512, 924513, 924514], stateBeforeColumns := [927867, 927868, 927869, 927870, 927871, 927872, 927873, 927874], permutationInputColumns := [927875, 927876, 927877, 927870, 927871, 927872, 927873, 927874], permutationOutputColumns := [928470, 928471, 928472, 928473, 928474, 928475, 928476, 928477], definingRows := [3032, 3033, 3034], call := { rowStart := 3035, rowEnd := 3635, inputColumns := [927875, 927876, 927877, 927870, 927871, 927872, 927873, 927874], firstAllocatedColumn := 927878 } }
    , { kind := .pad, stateBeforeColumns := [928470, 928471, 928472, 928473, 928474, 928475, 928476, 928477], permutationInputColumns := [928478, 928471, 928472, 928473, 928474, 928475, 928476, 928477], permutationOutputColumns := [929071, 929072, 929073, 929074, 929075, 929076, 929077, 929078], definingRows := [3635], call := { rowStart := 3636, rowEnd := 4236, inputColumns := [928478, 928471, 928472, 928473, 928474, 928475, 928476, 928477], firstAllocatedColumn := 928479 } }
    ], outputColumns := [929071, 929072, 929073, 929074] }

theorem xOutTrace_valid :
xOutTrace.Valid FPrimeFullHistoryRecursiveOutput.rows := by native_decide

theorem xOutTrace_output :
xOutTrace.outputColumns = FPrimeFullHistoryRecursiveOutput.xOutColumns := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes
