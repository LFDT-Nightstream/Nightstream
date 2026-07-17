import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange

/-!
Independent NC obligations for the full Phi81 carrier.

Protocol: SuperNeo `Pi_CCS`.
Phase: NC semantic truth before challenge compression or SumCheck.
Constraint family: one strict `b = 2` norm obligation per source and complete
carrier coordinate.

Owns: full-carrier norm truth, canonical materialization equivalence, the
uncompressed flat-column/lane diagonal table, unconditional completeness of
its cubic residual family, and soundness under the minimal no-zero-divisors
assumption.

Does not own: Boolean padding widths, gamma mixing, equality weights, the NC
SumCheck, transcript challenges, production packed-witness decoding, Rust,
R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `Truth` quantifies every coordinate in
`SemanticShape.carrierWidth`, including a suffix that was zero in a fresh
source but may be nonzero in a running CE source. A digest, projection, or
zero mixed claim is not a replacement for this pointwise obligation.

| Protocol | Phase | Family | Mathematical result |
|---|---|---|---|
| `Pi_CCS` | NC semantics | complete carrier | every authoritative coefficient has centered magnitude `< 2` |
| NC source table | flat column / Phi81 lane | one live lane `column mod 54` | table is derived from the authoritative assignment |
| NC residualization | honest completeness | strict norm implies `(z+1)z(z-1)=0` | unconditional |
| NC residualization | soundness | `(z+1)z(z-1)=0` implies strict norm | no zero divisors |
| implementation boundary | typed / list serialization | canonical increasing order | exact equivalence, no production decoder claim |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Strict `b = 2` truth for one complete-carrier assignment. -/
def AssignmentTruth
    {shape : SemanticShape}
    (assignment : PaperLinearAlgebra.Assignment F shape.carrierWidth) : Prop :=
  forall column, centeredMagnitude (assignment column) < 2

/-- Strict `b = 2` truth for every fresh and running source. -/
def Truth
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  forall source, AssignmentTruth (data.assignment source)

/-- Canonical typed materialization preserves one assignment's strict norm
exactly. -/
theorem assignmentTruth_iff_orderedAssignment_normBounded
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount) :
    AssignmentTruth (data.assignment source) <->
      normBounded 2 (data.orderedAssignment source) := by
  constructor
  · intro bounded value member
    rw [Data.orderedAssignment] at member
    rcases List.mem_map.mp member with ⟨column, _, rfl⟩
    exact bounded column
  · intro bounded column
    exact bounded (data.assignment source column) (by
      rw [Data.orderedAssignment]
      exact List.mem_map.mpr
        ⟨column, by simp [canonicalFinIndices], rfl⟩)

/-- The typed batch norm is exactly the concrete list norm on the canonical
materialization of every source. No source or carrier coordinate is omitted. -/
theorem truth_iff_orderedAssignments_normBounded
    {shape : SemanticShape}
    (data : Data shape) :
    Truth data <->
      (∀ assignment ∈ data.orderedAssignments,
        normBounded 2 assignment) := by
  constructor
  · intro truth assignment member
    rcases data.mem_orderedAssignments member with ⟨source, rfl⟩
    exact (assignmentTruth_iff_orderedAssignment_normBounded data source).mp
      (truth source)
  · intro bounded source
    apply (assignmentTruth_iff_orderedAssignment_normBounded data source).mpr
    exact bounded (data.orderedAssignment source)
      (data.orderedAssignment_mem_orderedAssignments source)

/-- Uncompressed NC source table. A flat carrier coordinate is live only in
its canonical Phi81 lane. -/
def diagonal
    {shape : SemanticShape}
    (assignment : PaperLinearAlgebra.Assignment F shape.carrierWidth)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) : F :=
  if lane.val = column.val % ringDegree then assignment column else 0

/-- The unique live lane for one carrier coordinate. -/
def selectedLane
    {shape : SemanticShape}
    (column : Fin shape.carrierWidth) : Fin ringDegree :=
  ⟨column.val % ringDegree, by
    apply Nat.mod_lt
    decide⟩

@[simp] theorem selectedLane_val
    {shape : SemanticShape}
    (column : Fin shape.carrierWidth) :
    (selectedLane column).val = column.val % ringDegree := by
  rfl

/-- Reading the selected lane recovers the authoritative carrier value. -/
theorem diagonal_selectedLane
    {shape : SemanticShape}
    (assignment : PaperLinearAlgebra.Assignment F shape.carrierWidth)
    (column : Fin shape.carrierWidth) :
    diagonal assignment column (selectedLane column) = assignment column := by
  simp [diagonal]

/-- Every uncompressed diagonal-table entry satisfies the semantic cubic
range equation. This proposition is independent of later mixing. -/
def ResidualsZero
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  forall source column lane,
    NormRange.cubicResidual
      (diagonal (data.assignment source) column lane) = 0

/-- Full-carrier strict norm truth makes every uncompressed diagonal cubic
vanish. This honest-completeness direction needs no no-zero-divisor premise:
the three allowed representatives are direct roots of the cubic. -/
theorem residualsZero_of_truth
    {shape : SemanticShape}
    (data : Data shape) :
    Truth data → ResidualsZero data := by
  intro truth source column lane
  by_cases selected : lane.val = column.val % ringDegree
  · have bounded : centeredMagnitude (data.assignment source column) < 2 :=
      truth source column
    rw [diagonal, if_pos selected]
    rcases (NormRange.strictNormTwo_iff_representedRoot
      (data.assignment source column)).mp bounded with negative | zero | one
    · rw [negative]
      decide
    · rw [zero]
      rfl
    · rw [one]
      rfl
  · rw [diagonal, if_neg selected]
    rfl

/-- If every uncompressed diagonal cubic vanishes, then every authoritative
carrier coordinate satisfies the strict norm. Only this soundness direction
uses the no-zero-divisors premise. -/
theorem truth_of_residualsZero
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data → Truth data := by
  intro residuals source column
  apply (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
    noZeroDivisors (data.assignment source column)).mp
  simpa [diagonal_selectedLane] using
    residuals source column (selectedLane column)

/-- The uncompressed diagonal cubic family is sound and complete for the
full-carrier strict norm. The premise is required only by the forward
soundness implication and is not consumed by honest completeness. -/
theorem residualsZero_iff_truth
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data <-> Truth data := by
  exact ⟨truth_of_residualsZero noZeroDivisors data,
    residualsZero_of_truth data⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc
