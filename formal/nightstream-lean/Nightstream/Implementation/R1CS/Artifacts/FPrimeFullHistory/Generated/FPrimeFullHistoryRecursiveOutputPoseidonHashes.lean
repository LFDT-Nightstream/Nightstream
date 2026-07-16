import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputArtifact
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-! Generated sponge certificate for the exact recursive output owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes

open Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 524288

def xOutTrace : Trace :=
  { inputColumns := [1127804, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 1127805, 1127806, 1127807, 1127808, 1127809, 1127810, 10864, 10865, 10866, 10867, 1127468, 1127469, 1127470, 1127471], zeroColumn := 1127811, zeroRow := 11, rounds := [
      { kind := .absorb [1127804, 10834, 10835, 10836], stateBeforeColumns := [1127811, 1127811, 1127811, 1127811, 1127811, 1127811, 1127811, 1127811], permutationInputColumns := [1127812, 1127813, 1127814, 1127815, 1127811, 1127811, 1127811, 1127811], permutationOutputColumns := [1128408, 1128409, 1128410, 1128411, 1128412, 1128413, 1128414, 1128415], definingRows := [12, 13, 14, 15], call := { rowStart := 16, rowEnd := 616, inputColumns := [1127812, 1127813, 1127814, 1127815, 1127811, 1127811, 1127811, 1127811], firstAllocatedColumn := 1127816 } }
    , { kind := .absorb [10837, 10838, 10839, 10840], stateBeforeColumns := [1128408, 1128409, 1128410, 1128411, 1128412, 1128413, 1128414, 1128415], permutationInputColumns := [1128416, 1128417, 1128418, 1128419, 1128412, 1128413, 1128414, 1128415], permutationOutputColumns := [1129012, 1129013, 1129014, 1129015, 1129016, 1129017, 1129018, 1129019], definingRows := [616, 617, 618, 619], call := { rowStart := 620, rowEnd := 1220, inputColumns := [1128416, 1128417, 1128418, 1128419, 1128412, 1128413, 1128414, 1128415], firstAllocatedColumn := 1128420 } }
    , { kind := .absorb [10841, 1127805, 1127806, 1127807], stateBeforeColumns := [1129012, 1129013, 1129014, 1129015, 1129016, 1129017, 1129018, 1129019], permutationInputColumns := [1129020, 1129021, 1129022, 1129023, 1129016, 1129017, 1129018, 1129019], permutationOutputColumns := [1129616, 1129617, 1129618, 1129619, 1129620, 1129621, 1129622, 1129623], definingRows := [1220, 1221, 1222, 1223], call := { rowStart := 1224, rowEnd := 1824, inputColumns := [1129020, 1129021, 1129022, 1129023, 1129016, 1129017, 1129018, 1129019], firstAllocatedColumn := 1129024 } }
    , { kind := .absorb [1127808, 1127809, 1127810, 10864], stateBeforeColumns := [1129616, 1129617, 1129618, 1129619, 1129620, 1129621, 1129622, 1129623], permutationInputColumns := [1129624, 1129625, 1129626, 1129627, 1129620, 1129621, 1129622, 1129623], permutationOutputColumns := [1130220, 1130221, 1130222, 1130223, 1130224, 1130225, 1130226, 1130227], definingRows := [1824, 1825, 1826, 1827], call := { rowStart := 1828, rowEnd := 2428, inputColumns := [1129624, 1129625, 1129626, 1129627, 1129620, 1129621, 1129622, 1129623], firstAllocatedColumn := 1129628 } }
    , { kind := .absorb [10865, 10866, 10867, 1127468], stateBeforeColumns := [1130220, 1130221, 1130222, 1130223, 1130224, 1130225, 1130226, 1130227], permutationInputColumns := [1130228, 1130229, 1130230, 1130231, 1130224, 1130225, 1130226, 1130227], permutationOutputColumns := [1130824, 1130825, 1130826, 1130827, 1130828, 1130829, 1130830, 1130831], definingRows := [2428, 2429, 2430, 2431], call := { rowStart := 2432, rowEnd := 3032, inputColumns := [1130228, 1130229, 1130230, 1130231, 1130224, 1130225, 1130226, 1130227], firstAllocatedColumn := 1130232 } }
    , { kind := .absorb [1127469, 1127470, 1127471], stateBeforeColumns := [1130824, 1130825, 1130826, 1130827, 1130828, 1130829, 1130830, 1130831], permutationInputColumns := [1130832, 1130833, 1130834, 1130827, 1130828, 1130829, 1130830, 1130831], permutationOutputColumns := [1131427, 1131428, 1131429, 1131430, 1131431, 1131432, 1131433, 1131434], definingRows := [3032, 3033, 3034], call := { rowStart := 3035, rowEnd := 3635, inputColumns := [1130832, 1130833, 1130834, 1130827, 1130828, 1130829, 1130830, 1130831], firstAllocatedColumn := 1130835 } }
    , { kind := .pad, stateBeforeColumns := [1131427, 1131428, 1131429, 1131430, 1131431, 1131432, 1131433, 1131434], permutationInputColumns := [1131435, 1131428, 1131429, 1131430, 1131431, 1131432, 1131433, 1131434], permutationOutputColumns := [1132028, 1132029, 1132030, 1132031, 1132032, 1132033, 1132034, 1132035], definingRows := [3635], call := { rowStart := 3636, rowEnd := 4236, inputColumns := [1131435, 1131428, 1131429, 1131430, 1131431, 1131432, 1131433, 1131434], firstAllocatedColumn := 1131436 } }
    ], outputColumns := [1132028, 1132029, 1132030, 1132031] }

theorem xOutTrace_valid :
xOutTrace.Valid FPrimeFullHistoryRecursiveOutput.rows := by native_decide

theorem xOutTrace_output :
xOutTrace.outputColumns = FPrimeFullHistoryRecursiveOutput.xOutColumns := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes
