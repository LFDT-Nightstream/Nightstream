import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSourceStageCoverage

/-!
Contract: executable validation of the compact source-stage census for the
production-width radix-four candidate.

Assurance tier: Rust-conformant for property
`FPRIME-R4-SOURCE-STAGE-COVERAGE`. The Rust drift owner reads the live
compiler's exclusive physical stages and five source dispositions before it
emits this artifact.

Owns: exact candidate totals, exact owner order, and arithmetic consistency of
all fourteen aggregate records.

Does not own: semantic authority of caller-supplied path labels, individual
decoder rules, arithmetic-family identity, recursive or terminal relation
soundness, constraint necessity, or permission to remove rows or columns.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourSourceStageCoverageArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SourceStageCoverage
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSourceStageCoverage

private abbrev candidateRaw := rawCoverage

theorem candidate_profile_exact :
    profileId = "wasm-nebula-radix-four-candidate-v1" := by
  decide

theorem candidate_census_exact :
    candidateRaw.physicalStages = 6578 ∧
    candidateRaw.unownedEmptyStages = 3 ∧
    candidateRaw.sourceFields = 16181176 ∧
    candidateRaw.direct = 2838233 ∧
    candidateRaw.decompositionAlias = 5006998 ∧
    candidateRaw.equalityAlias = 445 ∧
    candidateRaw.linearDefinition = 86880 ∧
    candidateRaw.traceEliminated = 8248620 ∧
    candidateRaw.allocatedCoordinates = 11502388 := by
  decide

theorem candidate_owner_order_exact :
    candidateRaw.owners.map (fun census => census.owner) = expectedOwners := by
  decide

theorem candidate_pi_rlc_stage_count :
    (candidateRaw.owners[5]!).owner = .piRlc ∧
    (candidateRaw.owners[5]!).stages = 6420 := by
  decide

theorem candidate_coverage_valid : CoverageValid candidateRaw := by
  decide

theorem candidate_source_fields_partition :
    candidateRaw.sourceFields = candidateRaw.dispositionFields :=
  candidate_coverage_valid.2.2.2.2.2.2.2.2.2.2.2

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RadixFourSourceStageCoverageArtifact
