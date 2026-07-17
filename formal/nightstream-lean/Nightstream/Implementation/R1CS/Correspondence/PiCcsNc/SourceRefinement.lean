import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Semantics.MixedPolynomial

/-!
Refinement from independent full-carrier NC sources to the executable Lean
SplitNc polynomial model.

Protocol: SuperNeo `Pi_CCS`.
Phase: NC source and honest-initial-claim refinement.
Constraint family: direct packed table and pre-SumCheck norm claim.

Owns: construction of the executable NC shape from separately typed domain
widths, exact equality of each canonical materialized source cell, equivalence
of full-carrier semantic norms with the executable input-list predicate, the
coverage premise required by the executable flat column domain, and the
honest-input implication to a zero NC initial claim.

Does not own: FE semantics, the SplitNc verifier/polynomial stack, production
Rust witness decoding, fixed-profile dimensions, gamma-mixing soundness,
SumCheck replay, transcript challenges, terminal authority, R1CS rows, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: `SourceRefinement` imports only SplitNc parameters,
authoritative sources, and NC semantics, then points that narrow semantic
surface into the executable model. It does not depend on the SplitNc barrel or
FE/verifier modules. The zero-initial-claim theorem is only a completeness
direction. No theorem here says a zero mixed claim implies all norms, and no
flat-domain theorem is usable without explicit full-carrier coverage.

| Protocol | Phase | Family | Mathematical guarantee | Permits row removal? |
|---|---|---|---|---|
| SplitNc | parameters | column / lane widths | executable shape is derived from typed domain widths | no |
| SplitNc | source | direct diagonal | every typed carrier cell equals the executable list cell | no |
| SplitNc | semantic bridge | all source norms | exact typed/list predicate equivalence | no |
| SplitNc | domain bridge | flat column coverage | `AssignmentsFitColumnDomain` only under full-carrier coverage | no |
| SplitNc | honest claim | NC initial sum | semantic norm truth implies zero | no converse; no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
open Nightstream.Implementation.R1CS.PiCcsNc

/-- Executable semantic shape for one separately typed flat NC domain. -/
def implementationShape (domain : FlatNcDomain) : PiCcsNc.Shape where
  ellM := domain.columnVariables
  ellD := domain.laneVariables

@[simp] theorem implementationShape_columnDomain
    (domain : FlatNcDomain) :
    (implementationShape domain).columnDomain = domain.columnCount := by
  rfl

@[simp] theorem implementationShape_laneDomain
    (domain : FlatNcDomain) :
    (implementationShape domain).laneDomain = domain.laneCount := by
  rfl

/-- Exact source-cell refinement before Boolean padding or interpolation. -/
theorem directDiagonal_orderedAssignment
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) :
    MixedPolynomial.directDiagonal
        (data.orderedAssignment source) column.val lane.val =
      K.embed
        (Semantics.Nc.diagonal (data.assignment source) column lane) := by
  unfold MixedPolynomial.directDiagonal Semantics.Nc.diagonal
  by_cases selected : lane.val = column.val % ringDegree
  · rw [if_pos ⟨by
        rw [data.orderedAssignment_length source]
        exact column.isLt, selected⟩]
    rw [if_pos selected]
    rw [data.orderedAssignment_getD source column]
  · rw [if_neg (fun live => selected live.2)]
    rw [if_neg selected]
    rfl

/-- The same cell refinement after embedding both typed coordinates into a
padded executable domain under explicit coverage. -/
theorem directDiagonal_at_coveredCoordinates
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (column : Fin shape.carrierWidth)
    (lane : Fin ringDegree) :
    MixedPolynomial.directDiagonal
        (data.orderedAssignment source)
        (domain.carrierColumn covers column).val
        (domain.phi81Lane covers lane).val =
      K.embed
        (Semantics.Nc.diagonal (data.assignment source) column lane) := by
  simpa using directDiagonal_orderedAssignment data source column lane

/-- Independent full-carrier NC truth is exactly the executable model's
input-list norm predicate on canonical materialization. -/
theorem truth_iff_inputsNormBoundedTwo
    {shape : SemanticShape}
    (data : Data shape) :
    Semantics.Nc.Truth data <->
      MixedPolynomial.InputsNormBoundedTwo data.orderedAssignments := by
  exact Semantics.Nc.truth_iff_orderedAssignments_normBounded data

/-- Full-carrier coverage is sufficient for every canonical executable
assignment list to fit the flat column domain. -/
theorem assignmentsFitColumnDomain_of_covers
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape) :
    MixedPolynomial.AssignmentsFitColumnDomain
      (implementationShape domain) data.orderedAssignments := by
  intro assignment member
  rcases data.mem_orderedAssignments member with ⟨source, rfl⟩
  rw [data.orderedAssignment_length source]
  exact covers.1

/-- Honest semantic norm truth makes the executable NC initial sum zero.

This is completeness only. Its converse requires independent gamma-mixing
and SumCheck soundness arguments and is intentionally absent. -/
theorem truth_implies_trueInitial_eq_zero
    (prime : EuclidPrime goldilocksP)
    {shape : SemanticShape}
    (data : Data shape)
    (truth : Semantics.Nc.Truth data)
    (domain : FlatNcDomain)
    (betaM betaA : List K)
    (gamma : K) :
    MixedPolynomial.trueInitial
        (implementationShape domain) betaM betaA gamma
        data.orderedAssignments = K.zero := by
  apply MixedPolynomial.trueInitial_eq_zero_of_normBounded prime
  exact (truth_iff_inputsNormBoundedTwo data).mp truth

end Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement
