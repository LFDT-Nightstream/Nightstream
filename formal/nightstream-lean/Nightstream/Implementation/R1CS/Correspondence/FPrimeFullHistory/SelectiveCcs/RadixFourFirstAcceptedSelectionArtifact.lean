import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourFirstAcceptedSelection

/-!
Contract: executable validation of the exact compact first-accepted selection
schedule for the production-width radix-four candidate.

Assurance tier: Rust-conformant for property
`FPRIME-R4-FIRST-ACCEPTED-SCHEDULE`. The Rust drift owner emits the schedule
only after exact local checks of all 36 source rows and compiler-ledger joins.
The substitution theorem is model-level for property
`FPRIME-SELECTION-PRODUCT-SUBSTITUTION`.

Owns: exact dimensions, eight samplers, their bounded expansion into 432
non-overlapping source and emitted intervals, and algebraic substitution for
every generated sampler and output position.

Does not own: final nine-row low-norm gate semantics, sampler one-hotness,
complete PiRLC semantics, whole-relation refinement, or row-removal authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.FirstAcceptedSelection
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourFirstAcceptedSelection

private abbrev candidateRaw := rawCoverage

theorem candidate_profile_exact :
    profileId = "wasm-nebula-radix-four-candidate-v1" := by
  decide

theorem candidate_dimensions_exact :
    candidateRaw.relationRows = 8102331 ∧
    candidateRaw.relationColumns = 12288726 ∧
    candidateRaw.sourceRows = 16407566 ∧
    candidateRaw.sourceColumns = 16237141 := by
  decide

theorem candidate_sampler_count_exact :
    candidateRaw.samplers.length = 8 := by
  decide

theorem candidate_occurrence_count_exact :
    candidateRaw.occurrences.length = 432 := by
  calc
    candidateRaw.occurrences.length =
        candidateRaw.samplers.length * outputCount :=
      RawCoverage.occurrences_length candidateRaw
    _ = 8 * 54 := by simp [candidate_sampler_count_exact, outputCount]
    _ = 432 := by decide

theorem candidate_row_geometry_exact :
    candidateRaw.blockCount = 432 ∧
    candidateRaw.sourceBlockRows = 15552 ∧
    candidateRaw.emittedBlockRows = 3888 ∧
    candidateRaw.sourceBlockRows - candidateRaw.emittedBlockRows = 11664 := by
  decide

theorem candidate_coverage_valid : CoverageValid candidateRaw := by
  native_decide

theorem generated_currentAt_iff_aggregateAt {value : Type}
    [Add value] [OfNat value 0] [Mul value] [OfNat value 1]
    (assignment : Nat → value) (positionValue : Nat → value)
    (sampler : { sampler // sampler ∈ candidateRaw.samplers })
    (position : Fin outputCount) :
    CurrentAt assignment positionValue sampler.1 position.val ↔
      AggregateAt assignment positionValue sampler.1 position.val :=
  currentAt_iff_aggregateAt assignment positionValue sampler.1 position.val

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourFirstAcceptedSelectionArtifact
