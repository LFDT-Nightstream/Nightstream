import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.ArtifactRows

/-!
Semantic refinement for the bounded production strict-`PiDEC` canonical-X
receipt.

Assurance tier: artifact-checked for the fixed `54 x 5`, fourteen-child
profile and only its 4,590 public-X rows.

Owns: transport from satisfaction of the exact live sparse A/B/C rows through
term-order normalization to the independent `UniformXAccepted` predicate, and
the converse honest-satisfaction construction for the same coefficient rows.

Does not own: commitment or evaluation recomposition, any other strict
`PiDEC` row family, assignment-column allocation outside this receipt,
protocol composition, commitment binding, or row-removal authority.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.ArtifactSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler

/-- Satisfaction predicate for exactly the 4,590 coefficient rows carried by
the live canonical-X receipt. Physical indices and owner labels are not used
as semantic authority. -/
def RowsSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies (ArtifactRows.artifactRows.map (fun record => record.row)) assignment

/-- Exact generated-row soundness for the isolated public-X obligation.

The only ambient R1CS assumptions are canonical Goldilocks representatives
and the constant-one wire. No commitment, evaluation, or whole-`PiDEC`
acceptance premise is consumed. -/
theorem rows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied assignment) :
    UniformXAccepted ArtifactRows.layout assignment := by
  apply uniformXRows_sound prime ArtifactRows.layout_shape_valid canonical one
  apply ArtifactRows.satisfies_of_normalizedRows
  rw [← ArtifactRows.coefficients_exact]
  exact satisfies

/-- Honest completeness for the same exact coefficient rows. The trace
premise is the deterministic materialization of each sign-product auxiliary;
it is not a semantic acceptance oracle. -/
theorem rows_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : UniformXAccepted ArtifactRows.layout assignment)
    (traceDefinitions : TraceDefinitions ArtifactRows.layout assignment) :
    RowsSatisfied assignment := by
  unfold RowsSatisfied
  rw [ArtifactRows.coefficients_exact]
  apply ArtifactRows.normalizedRows_satisfy_of
  exact uniformXRows_complete ArtifactRows.layout_shape_valid canonical one
    accepted traceDefinitions

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.ArtifactSemantics
