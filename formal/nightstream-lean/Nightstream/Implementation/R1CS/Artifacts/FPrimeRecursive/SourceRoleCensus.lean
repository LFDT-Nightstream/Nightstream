import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.Generated.FPrimeBranchSourceRoleManifestData

/-!
Contract: checked source-role censuses for the fixed F-prime base and recursive
gadget-native branches.

Owns: exactly one executable certificate per generated branch, the resulting
source-only `SourceCensusArtifact` values, exact eligible source-field counts,
and conditional per-field-41 source-capacity floors.

Does not own: encoded-coordinate or CE-coordinate layouts, a coordinate
materializer, selector composition, R1CS row removal, or a lower bound over
alternative encodings or algebraic gate systems.

Emits constraints: no.

Authority boundary: the two `native_decide` certificates establish internal
agreement between the committed packed data and the generic Lean source-census
contract. The values are Rust-conformant source-only evidence only after the
production generator drift gate replays the source trace and byte-compares this
exact generated module. Neither the packed data nor a digest is authority for a
different production trace.

| Branch/result | Mathematical obligation | Guarantee | Assurance boundary |
|---|---|---|---|
| base check | 7,528 runs partition 23,567 source columns with exact role totals | `base_data_check` | committed artifact check |
| recursive check | 327,838 runs partition 8,975,812 source columns with exact role totals | `recursive_data_check` | committed artifact check |
| eligible counts | declared ordinary-private counts and proved run subtotals are exactly 3,226 and 93,896 | `base_eligible_count`, `recursive_eligible_count`, `base_ordinaryRunSubtotal_count`, `recursive_ordinaryRunSubtotal_count` | source-only census |
| width floors | a candidate reserving 41 coordinates per eligible source field needs 132,266 and 3,849,736 coordinates | `base_perField41_width_floor`, `recursive_perField41_width_floor` | conditional capacity theorem |
| combined floor | separate base and recursive candidates need at least 3,982,002 coordinates in total | `combined_perField41_width_floor` | conditional capacity theorem |
| one-million no-go | the recursive per-field-41 candidate alone cannot fit in 1,000,000 coordinates | `recursive_one_million_perField41_budget_is_no_go` | conditional capacity theorem; alternative intra-recursive encodings remain open |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus

open Nightstream.Implementation.R1CS.FPrimeFieldLayout
open Nightstream.Implementation.R1CS.FPrimeBranchSourceRoleManifestData

/-- Sole production-sized executable certificate for the base source census. -/
theorem base_data_check : baseData.check = true := by
  native_decide

/-- Sole production-sized executable certificate for the recursive source
census. -/
theorem recursive_data_check : recursiveData.check = true := by
  native_decide

/-- Checked source-only base-branch census. -/
def baseSourceCensus : SourceCensusArtifact :=
  baseData.toSourceCensusArtifact base_data_check

/-- Checked source-only recursive-branch census. -/
def recursiveSourceCensus : SourceCensusArtifact :=
  recursiveData.toSourceCensusArtifact recursive_data_check

theorem base_eligible_count : baseSourceCensus.eligibleCount = 3226 := by
  rfl

theorem recursive_eligible_count :
    recursiveSourceCensus.eligibleCount = 93896 := by
  rfl

/-- The checked base runs, not merely the generated declaration, contain
exactly 3,226 ordinary-private source columns. -/
theorem base_ordinaryRunSubtotal_count :
    SourceSegment.eligibleCountOf baseSourceCensus.sourceSegments = 3226 := by
  rw [← baseSourceCensus.eligibleCount_eq_ordinaryRunSubtotal]
  exact base_eligible_count

/-- The checked recursive runs, not merely the generated declaration, contain
exactly 93,896 ordinary-private source columns. -/
theorem recursive_ordinaryRunSubtotal_count :
    SourceSegment.eligibleCountOf recursiveSourceCensus.sourceSegments =
      93896 := by
  rw [← recursiveSourceCensus.eligibleCount_eq_ordinaryRunSubtotal]
  exact recursive_eligible_count

/-- Conditional base capacity floor for an architecture reserving one disjoint
41-coordinate word per eligible source field. -/
theorem base_perField41_width_floor {candidateWidth : Nat}
    (capacity :
      baseSourceCensus.PerField41CapacityRequirement candidateWidth) :
    132266 ≤ candidateWidth := by
  simpa [base_eligible_count] using
    baseSourceCensus.perField41_width_floor capacity

/-- Conditional recursive capacity floor for an architecture reserving one
disjoint 41-coordinate word per eligible source field. -/
theorem recursive_perField41_width_floor {candidateWidth : Nat}
    (capacity :
      recursiveSourceCensus.PerField41CapacityRequirement candidateWidth) :
    3849736 ≤ candidateWidth := by
  simpa [recursive_eligible_count] using
    recursiveSourceCensus.perField41_width_floor capacity

/-- Sum of the two conditional source-capacity floors. This does not assert
that production has already materialized either candidate layout. -/
theorem combined_perField41_width_floor
    {baseCandidateWidth recursiveCandidateWidth : Nat}
    (baseCapacity :
      baseSourceCensus.PerField41CapacityRequirement baseCandidateWidth)
    (recursiveCapacity :
      recursiveSourceCensus.PerField41CapacityRequirement
        recursiveCandidateWidth) :
    3982002 ≤ baseCandidateWidth + recursiveCandidateWidth := by
  have baseFloor := base_perField41_width_floor baseCapacity
  have recursiveFloor := recursive_perField41_width_floor recursiveCapacity
  omega

/-- Under the recursive per-field-41 capacity premise alone, a budget of at
most one million coordinates is impossible. Sharing a selector arena with the
base branch cannot change this recursive subtotal; another intra-recursive
encoding architecture remains outside this theorem. -/
theorem recursive_one_million_perField41_budget_is_no_go
    {recursiveCandidateWidth : Nat}
    (recursiveCapacity :
      recursiveSourceCensus.PerField41CapacityRequirement
        recursiveCandidateWidth) :
    ¬ recursiveCandidateWidth ≤ 1000000 := by
  have floor := recursive_perField41_width_floor recursiveCapacity
  omega

end Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus
