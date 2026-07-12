import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPriorLinkArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated sponge certificate for the exact recursive prior-link owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def traces : List Trace :=
[
  { inputColumns := [882911, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 882978, 882979, 883046, 883047, 883114, 883115, 10848, 10849, 10850, 10851, 10856, 10857, 10858, 10859], zeroColumn := 883116, zeroRow := 218, rounds := [
      { kind := .absorb [882911, 10834, 10835, 10836], stateBeforeColumns := [883116, 883116, 883116, 883116, 883116, 883116, 883116, 883116], permutationInputColumns := [883117, 883118, 883119, 883120, 883116, 883116, 883116, 883116], permutationOutputColumns := [883713, 883714, 883715, 883716, 883717, 883718, 883719, 883720], definingRows := [219, 220, 221, 222], call := { rowStart := 223, rowEnd := 823, inputColumns := [883117, 883118, 883119, 883120, 883116, 883116, 883116, 883116], firstAllocatedColumn := 883121 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [883713, 883714, 883715, 883716, 883717, 883718, 883719, 883720], permutationInputColumns := [883721, 883722, 883723, 883724, 883717, 883718, 883719, 883720], permutationOutputColumns := [884317, 884318, 884319, 884320, 884321, 884322, 884323, 884324], definingRows := [823, 824, 825, 826], call := { rowStart := 827, rowEnd := 1427, inputColumns := [883721, 883722, 883723, 883724, 883717, 883718, 883719, 883720], firstAllocatedColumn := 883725 } }
    , { kind := .absorb [10841, 882978, 882979, 883046], stateBeforeColumns := [884317, 884318, 884319, 884320, 884321, 884322, 884323, 884324], permutationInputColumns := [884325, 884326, 884327, 884328, 884321, 884322, 884323, 884324], permutationOutputColumns := [884921, 884922, 884923, 884924, 884925, 884926, 884927, 884928], definingRows := [1427, 1428, 1429, 1430], call := { rowStart := 1431, rowEnd := 2031, inputColumns := [884325, 884326, 884327, 884328, 884321, 884322, 884323, 884324], firstAllocatedColumn := 884329 } }
    , { kind := .absorb [883047, 883114, 883115, 10848], stateBeforeColumns := [884921, 884922, 884923, 884924, 884925, 884926, 884927, 884928], permutationInputColumns := [884929, 884930, 884931, 884932, 884925, 884926, 884927, 884928], permutationOutputColumns := [885525, 885526, 885527, 885528, 885529, 885530, 885531, 885532], definingRows := [2031, 2032, 2033, 2034], call := { rowStart := 2035, rowEnd := 2635, inputColumns := [884929, 884930, 884931, 884932, 884925, 884926, 884927, 884928], firstAllocatedColumn := 884933 } }
    , { kind := .absorb [10849, 10850, 10851, 10856], stateBeforeColumns := [885525, 885526, 885527, 885528, 885529, 885530, 885531, 885532], permutationInputColumns := [885533, 885534, 885535, 885536, 885529, 885530, 885531, 885532], permutationOutputColumns := [886129, 886130, 886131, 886132, 886133, 886134, 886135, 886136], definingRows := [2635, 2636, 2637, 2638], call := { rowStart := 2639, rowEnd := 3239, inputColumns := [885533, 885534, 885535, 885536, 885529, 885530, 885531, 885532], firstAllocatedColumn := 885537 } }
    , { kind := .absorb [10857, 10858, 10859], stateBeforeColumns := [886129, 886130, 886131, 886132, 886133, 886134, 886135, 886136], permutationInputColumns := [886137, 886138, 886139, 886132, 886133, 886134, 886135, 886136], permutationOutputColumns := [886732, 886733, 886734, 886735, 886736, 886737, 886738, 886739], definingRows := [3239, 3240, 3241], call := { rowStart := 3242, rowEnd := 3842, inputColumns := [886137, 886138, 886139, 886132, 886133, 886134, 886135, 886136], firstAllocatedColumn := 886140 } }
    , { kind := .pad, stateBeforeColumns := [886732, 886733, 886734, 886735, 886736, 886737, 886738, 886739], permutationInputColumns := [886740, 886733, 886734, 886735, 886736, 886737, 886738, 886739], permutationOutputColumns := [887333, 887334, 887335, 887336, 887337, 887338, 887339, 887340], definingRows := [3842], call := { rowStart := 3843, rowEnd := 4443, inputColumns := [886740, 886733, 886734, 886735, 886736, 886737, 886738, 886739], firstAllocatedColumn := 886741 } }
    ], outputColumns := [887333, 887334, 887335, 887336] }
]

theorem traces_accepted :
traces.all (fun trace => decide (trace.Valid FPrimeFullHistoryPriorLink.rows)) = true := by
native_decide

theorem traces_valid :
∀ trace ∈ traces, trace.Valid FPrimeFullHistoryPriorLink.rows := by
intro trace member
exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)

def priorXOutTrace : Trace := traces[0]!

theorem priorXOutTrace_output :
priorXOutTrace.outputColumns = FPrimeFullHistoryPriorLink.digestColumns := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes
