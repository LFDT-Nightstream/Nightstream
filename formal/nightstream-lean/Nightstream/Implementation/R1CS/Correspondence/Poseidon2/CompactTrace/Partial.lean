import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial0
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial1
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial2
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial3
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial4
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial5

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

theorem compact_partial_schedule_exact :
    ∀ round : Fin partialRounds, PartialScheduleExactAt round := by
  intro round
  have roundBound : round.val < 22 := by
    simpa only [partialRounds] using round.isLt
  by_cases first : round.val < 4
  · let offset : Fin 4 := ⟨round.val, first⟩
    have same : partialShardIndex0 offset = round := by
      apply Fin.ext
      simp [partialShardIndex0, offset]
    simpa [same] using compact_partial_schedule_exact_0 offset
  · by_cases second : round.val < 8
    · let offset : Fin 4 := ⟨round.val - 4, by omega⟩
      have same : partialShardIndex1 offset = round := by
        apply Fin.ext
        simp [partialShardIndex1, offset]
        omega
      simpa [same] using compact_partial_schedule_exact_1 offset
    · by_cases third : round.val < 12
      · let offset : Fin 4 := ⟨round.val - 8, by omega⟩
        have same : partialShardIndex2 offset = round := by
          apply Fin.ext
          simp [partialShardIndex2, offset]
          omega
        simpa [same] using compact_partial_schedule_exact_2 offset
      · by_cases fourth : round.val < 16
        · let offset : Fin 4 := ⟨round.val - 12, by omega⟩
          have same : partialShardIndex3 offset = round := by
            apply Fin.ext
            simp [partialShardIndex3, offset]
            omega
          simpa [same] using compact_partial_schedule_exact_3 offset
        · by_cases fifth : round.val < 20
          · let offset : Fin 4 := ⟨round.val - 16, by omega⟩
            have same : partialShardIndex4 offset = round := by
              apply Fin.ext
              simp [partialShardIndex4, offset]
              omega
            simpa [same] using compact_partial_schedule_exact_4 offset
          · let offset : Fin 2 := ⟨round.val - 20, by omega⟩
            have same : partialShardIndex5 offset = round := by
              apply Fin.ext
              simp [partialShardIndex5, offset]
              omega
            simpa [same] using compact_partial_schedule_exact_5 offset

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
