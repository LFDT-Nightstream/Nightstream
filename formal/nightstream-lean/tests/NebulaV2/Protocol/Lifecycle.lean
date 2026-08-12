import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2Lifecycle

open Nightstream.Protocol.NebulaV2.Lifecycle

theorem one_claim_base_produces_zero :
    producedAt (baseIndex 1) = some (0 : Fin 1) := by
  simpa using base_produces_first (claimCount := 1) (by decide)

theorem one_claim_terminal_consumes_zero :
    consumedAt (terminalIndex 1) = some (0 : Fin 1) := by
  simpa using terminal_consumes_trailing (claimCount := 1) (by decide)

theorem one_claim_terminal_produces_nothing :
    producedAt (terminalIndex 1) = none :=
  terminal_produces_none 1

def middleOfTwo : InvocationIndex 2 := ⟨1, by decide⟩

theorem middle_consumes_zero_and_produces_one :
    consumedAt middleOfTwo = some (0 : Fin 2) ∧
      producedAt middleOfTwo = some (1 : Fin 2) := by
  simpa [middleOfTwo] using
    recursive_consumes_prior_and_produces_current middleOfTwo
      (by decide) (by decide)

/- Stopping after invocation one in a two-claim chain is invalid. That
invocation must produce C[1], which only invocation two can consume. -/
theorem early_terminal_leaves_a_trailing_claim :
    producedAt middleOfTwo = some (1 : Fin 2) ∧
      consumedAt (terminalIndex 2) = some (1 : Fin 2) := by
  exact ⟨middle_consumes_zero_and_produces_one.2,
    by simpa using terminal_consumes_trailing (claimCount := 2) (by decide)⟩

theorem first_segment_boundary :
    claimSegment (1 * claimsPerSegment - 1) = 0 ∧
      claimStep (1 * claimsPerSegment - 1) = claimsPerSegment - 1 ∧
      claimSegment (1 * claimsPerSegment) = 1 ∧
      claimStep (1 * claimsPerSegment) = 0 := by
  simpa using segment_boundary_locations (segmentIndex := 1) (by decide)

theorem maximum_final_claim_is_last_step_of_segment_63 :
    claimSegment (totalClaims maximumSegments - 1) = 63 ∧
      claimStep (totalClaims maximumSegments - 1) = 1087 := by
  simpa [maximumSegments, claimsPerSegment] using
    final_claim_location (segmentCount := maximumSegments) (by decide)

theorem maximum_counts :
    totalClaims maximumSegments = 69632 ∧
      totalClaims maximumSegments + 1 = 69633 :=
  ⟨maximum_claim_count, maximum_augmented_invocation_count⟩

theorem maximum_schedule_consumes_the_trailing_claim :
    ∃ last : Fin (totalClaims maximumSegments),
      consumedAt (terminalIndex (totalClaims maximumSegments)) = some last ∧
        last.val + 1 = totalClaims maximumSegments :=
  (completeSchedule (claimCount := totalClaims maximumSegments)
    (by decide)).terminalConsumesLast

end tests.NebulaV2Lifecycle
