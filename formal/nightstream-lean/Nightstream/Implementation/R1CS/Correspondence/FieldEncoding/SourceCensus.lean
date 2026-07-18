import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutWidthFloor

/-!
Contract: source-only schema for separately generated fixed-F-prime base and
recursive role censuses.

Owns: exact partitioning of source columns by `SourceSegment`, exact declared
counts for every `SlotRole`, the ordinary/private versus explicitly excluded
count partition, and prospective 41-coordinate capacity accounting.

Does not own: concrete base or recursive census data, generated files, encoded
or CE coordinate runs, a coordinate materializer, compiler placements,
selector composition, R1CS rows, or a global lower bound over alternative
encodings or algebraic gate systems.

Emits constraints: no.

Authority boundary: a census is non-authoritative structural evidence. A Rust
generator must separately instantiate this schema from the production source
trace. `PerField41CapacityRequirement` is only a conditional capacity test for
the prospective architecture that assigns one disjoint 41-coordinate word to
each eligible source field; it does not prove that production emits such a
layout.

Attribution boundary: every run remains a full `SourceSegment`, including its
`ownerPath`. Generated data can therefore preserve the Rust physical-stage
owner while this module reasons only about source order and role counts.

| Surface | Mathematical obligation | Main result | Assurance tier |
|---|---|---|---|
| source partition | every in-range source column has one unique role; segment uniqueness comes from `sourcePartition` | `sourceColumn_hasUniqueRole` | model-level schema |
| role census | declared count for each role equals its source-run subtotal | `roleCensusExact` | artifact obligation |
| declared total | all declared role counts sum to the exact source universe | `declaredRoleTotal_eq_sourceColumnCount` | model theorem |
| eligibility partition | every role is eligible or explicitly excluded, exclusively | `sourceColumn_hasExactEligibilityClass` | model theorem |
| ordinary subtotal | eligible count is exactly the ordinary-run subtotal | `eligibleCount_eq_ordinaryRunSubtotal` | model theorem |
| capacity test | one 41-coordinate word per eligible field cannot fit below its subtotal | `budget_below_perField41_is_no_go` | conditional accounting theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFieldLayout

open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

/-- The ten fail-closed source roles that are not ordinary private fields. -/
def explicitlyExcludedRoles : List SlotRole :=
  [.constantOne, .privateBoolean, .publicBit, .canonicalU64, .sisOpening,
    .linearlyDerived, .structuralBalancedAlias, .gadgetDerived,
    .productDerived, .gadgetTemporary]

/-- Exhaustive source-role order used only for compact census totals. -/
def sourceCensusRoles : List SlotRole :=
  .ordinaryPrivateField :: explicitlyExcludedRoles

theorem sourceCensusRoles_complete (role : SlotRole) :
    role ∈ sourceCensusRoles := by
  cases role <;> simp [sourceCensusRoles, explicitlyExcludedRoles]

theorem sourceCensusRoles_nodup : sourceCensusRoles.Nodup := by
  decide

/-- Total source-run length carrying one exact role. -/
def roleRunSubtotal (role : SlotRole) : List SourceSegment → Nat
  | [] => 0
  | segment :: tail =>
      (if segment.role = role then segment.source.length else 0) +
        roleRunSubtotal role tail

/-- Total source-run length carrying any explicitly excluded role. -/
def excludedRunSubtotal : List SourceSegment → Nat
  | [] => 0
  | segment :: tail =>
      (if segment.role.ExplicitlyExcluded then segment.source.length else 0) +
        excludedRunSubtotal tail

/-- Smallest source-only artifact contract. Base and recursive generated
modules instantiate separate named values; the schema needs no profile tag. -/
structure SourceCensusArtifact where
  sourceColumnCount : Nat
  sourceSegments : List SourceSegment
  declaredRoleCount : SlotRole → Nat
  sourcePartition : ExactPartition SourceSegment.source
    sourceColumnCount sourceSegments
  roleCensusExact : ∀ role,
    declaredRoleCount role = roleRunSubtotal role sourceSegments

namespace SourceCensusArtifact

/-- Generator-declared ordinary-private-field count. -/
def eligibleCount (artifact : SourceCensusArtifact) : Nat :=
  artifact.declaredRoleCount .ordinaryPrivateField

/-- Generator-declared total over every explicitly excluded role. -/
def excludedCount (artifact : SourceCensusArtifact) : Nat :=
  (explicitlyExcludedRoles.map artifact.declaredRoleCount).sum

/-- Generator-declared total over the complete role universe. -/
def declaredRoleTotal (artifact : SourceCensusArtifact) : Nat :=
  (sourceCensusRoles.map artifact.declaredRoleCount).sum

/-- One role owns a source column when a source segment with that exact role
contains the column. The witness retains its physical-stage `ownerPath`. -/
def RoleOwnsSourceColumn (artifact : SourceCensusArtifact)
    (role : SlotRole) (sourceColumn : Nat) : Prop :=
  ∃ segment : SourceSegment,
    segment ∈ artifact.sourceSegments ∧
      segment.source.Contains sourceColumn ∧
      segment.role = role

private theorem ordinaryRoleRunSubtotal_eq_eligibleCountOf
    (segments : List SourceSegment) :
    roleRunSubtotal .ordinaryPrivateField segments =
      SourceSegment.eligibleCountOf segments := by
  induction segments with
  | nil => rfl
  | cons segment tail inductionHypothesis =>
      simp [roleRunSubtotal, SourceSegment.eligibleCountOf,
        SourceSegment.eligibleFieldCount, inductionHypothesis]

private theorem excludedRoleRunSubtotals_eq_excludedRunSubtotal
    (segments : List SourceSegment) :
    (explicitlyExcludedRoles.map fun role =>
        roleRunSubtotal role segments).sum =
      excludedRunSubtotal segments := by
  induction segments with
  | nil => rfl
  | cons segment tail inductionHypothesis =>
      cases segment with
      | mk ownerPath role source =>
          cases role <;>
            simp [explicitlyExcludedRoles, roleRunSubtotal,
              excludedRunSubtotal, SlotRole.ExplicitlyExcluded] at inductionHypothesis ⊢ <;>
            omega

private theorem allRoleRunSubtotals_eq_totalRunLength
    (segments : List SourceSegment) :
    (sourceCensusRoles.map fun role =>
        roleRunSubtotal role segments).sum =
      totalRunLength SourceSegment.source segments := by
  induction segments with
  | nil => rfl
  | cons segment tail inductionHypothesis =>
      cases segment with
      | mk ownerPath role source =>
          cases role <;>
            simp [sourceCensusRoles, explicitlyExcludedRoles,
              roleRunSubtotal, totalRunLength] at inductionHypothesis ⊢ <;>
            omega

/-- Every in-range source column has one and only one source role. -/
theorem sourceColumn_hasUniqueRole (artifact : SourceCensusArtifact)
    {sourceColumn : Nat}
    (sourceColumnLt : sourceColumn < artifact.sourceColumnCount) :
    ∃ role : SlotRole,
      artifact.RoleOwnsSourceColumn role sourceColumn ∧
        ∀ otherRole : SlotRole,
          artifact.RoleOwnsSourceColumn otherRole sourceColumn →
            otherRole = role := by
  rcases ExactPartition.existsUniqueOwner artifact.sourcePartition
      sourceColumnLt with
    ⟨segment, segmentMember, segmentContains, uniqueSegment⟩
  refine ⟨segment.role, ⟨segment, segmentMember, segmentContains, rfl⟩, ?_⟩
  intro otherRole otherOwns
  rcases otherOwns with
    ⟨other, otherMember, otherContains, otherRoleEq⟩
  have otherEq : other = segment :=
    uniqueSegment other otherMember otherContains
  subst other
  exact otherRoleEq.symm

/-- Every source column's unique role belongs to exactly one side of the
eligible/explicitly-excluded partition. -/
theorem sourceColumn_hasExactEligibilityClass
    (artifact : SourceCensusArtifact)
    {sourceColumn : Nat}
    (sourceColumnLt : sourceColumn < artifact.sourceColumnCount) :
    ∃ role : SlotRole,
      artifact.RoleOwnsSourceColumn role sourceColumn ∧
        ((role.Eligible ∧ ¬ role.ExplicitlyExcluded) ∨
          (role.ExplicitlyExcluded ∧ ¬ role.Eligible)) := by
  rcases artifact.sourceColumn_hasUniqueRole sourceColumnLt with
    ⟨role, roleOwns, uniqueRole⟩
  refine ⟨role, roleOwns, ?_⟩
  rcases SlotRole.eligible_or_explicitlyExcluded role with
    eligible | excluded
  · exact Or.inl ⟨eligible,
      (SlotRole.eligible_iff_not_explicitlyExcluded role).mp eligible⟩
  · exact Or.inr ⟨excluded, fun eligible =>
      (SlotRole.eligible_iff_not_explicitlyExcluded role).mp eligible excluded⟩

/-- The declared eligible count is exactly the ordinary source-run subtotal. -/
theorem eligibleCount_eq_ordinaryRunSubtotal
    (artifact : SourceCensusArtifact) :
    artifact.eligibleCount =
      SourceSegment.eligibleCountOf artifact.sourceSegments := by
  rw [eligibleCount, artifact.roleCensusExact .ordinaryPrivateField]
  exact ordinaryRoleRunSubtotal_eq_eligibleCountOf artifact.sourceSegments

/-- The declared excluded count is exactly the subtotal of all source runs
whose role is explicitly excluded. -/
theorem excludedCount_eq_excludedRunSubtotal
    (artifact : SourceCensusArtifact) :
    artifact.excludedCount =
      excludedRunSubtotal artifact.sourceSegments := by
  calc
    artifact.excludedCount =
        (explicitlyExcludedRoles.map fun role =>
          roleRunSubtotal role artifact.sourceSegments).sum := by
      simp [excludedCount, explicitlyExcludedRoles,
        artifact.roleCensusExact]
    _ = excludedRunSubtotal artifact.sourceSegments :=
      excludedRoleRunSubtotals_eq_excludedRunSubtotal
        artifact.sourceSegments

/-- The sum of all declared role counts equals the exact source universe. The
equality is derived from the partition; it is not a redundant artifact field. -/
theorem declaredRoleTotal_eq_sourceColumnCount
    (artifact : SourceCensusArtifact) :
    artifact.declaredRoleTotal = artifact.sourceColumnCount := by
  calc
    artifact.declaredRoleTotal =
        (sourceCensusRoles.map fun role =>
          roleRunSubtotal role artifact.sourceSegments).sum := by
      simp [declaredRoleTotal, sourceCensusRoles, explicitlyExcludedRoles,
        artifact.roleCensusExact]
    _ = totalRunLength SourceSegment.source artifact.sourceSegments :=
      allRoleRunSubtotals_eq_totalRunLength artifact.sourceSegments
    _ = artifact.sourceColumnCount :=
      ExactPartition.totalRunLength_eq artifact.sourcePartition

theorem declaredRoleTotal_eq_eligibleCount_add_excludedCount
    (artifact : SourceCensusArtifact) :
    artifact.declaredRoleTotal =
      artifact.eligibleCount + artifact.excludedCount := by
  rfl

/-- Source columns partition exactly into eligible ordinary fields and all
explicitly excluded fields. -/
theorem sourceColumnCount_eq_eligibleCount_add_excludedCount
    (artifact : SourceCensusArtifact) :
    artifact.sourceColumnCount =
      artifact.eligibleCount + artifact.excludedCount := by
  rw [← artifact.declaredRoleTotal_eq_eligibleCount_add_excludedCount]
  exact artifact.declaredRoleTotal_eq_sourceColumnCount.symm

/-- Conditional capacity predicate for the prospective architecture that
assigns one disjoint `digitCount`-coordinate word to each eligible field.
This definition does not assert that any production materializer exists. -/
def PerField41CapacityRequirement (artifact : SourceCensusArtifact)
    (coordinateBudget : Nat) : Prop :=
  artifact.eligibleCount * digitCount ≤ coordinateBudget

theorem perField41CapacityRequirement_iff
    (artifact : SourceCensusArtifact) (coordinateBudget : Nat) :
    artifact.PerField41CapacityRequirement coordinateBudget ↔
      artifact.eligibleCount * 41 ≤ coordinateBudget := by
  simp [PerField41CapacityRequirement, digitCount_eq_41]

/-- Conditional width floor for a candidate satisfying the prospective
per-field-41 capacity predicate. This is not a production layout theorem. -/
theorem perField41_width_floor (artifact : SourceCensusArtifact)
    {candidateWidth : Nat}
    (capacity : artifact.PerField41CapacityRequirement candidateWidth) :
    artifact.eligibleCount * 41 ≤ candidateWidth :=
  (artifact.perField41CapacityRequirement_iff candidateWidth).mp capacity

/-- A budget below `eligibleCount * 41` cannot satisfy the prospective
per-field-41 capacity requirement. This is not a lower bound on arbitrary
encodings or on the number of algebraic gates. -/
theorem budget_below_perField41_is_no_go
    (artifact : SourceCensusArtifact) {budget : Nat}
    (tooSmall : budget < artifact.eligibleCount * 41) :
    ¬ artifact.PerField41CapacityRequirement budget := by
  intro capacity
  have floor := artifact.perField41_width_floor capacity
  omega

end SourceCensusArtifact

end Nightstream.Implementation.R1CS.FPrimeFieldLayout
