import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput

/-!
Typed public-input recomposition for the concrete Phi81 `Pi_DEC` algebra.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 14`.
Phase: public parent-input recomposition from the fourteen child public inputs.
Constraint family: semantic public-input recomposition only; this file emits no
rows.

Owns: base-field scaling of one exact ring-aligned public carrier; the
canonical head-first finite base-scalar fold; specialization to the
verifier-fixed `2^i` radix weights; and the exact theorem required by
`Folding.PiDEC.Algebra.publicInput_hom`.

Does not own: assignment splitting, commitments, evaluation recomposition,
public-input authority before projection, transcript binding, Rust/R1CS
refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `recomposePublicInput` consumes only the fourteen public
child inputs and verifier-fixed radix weights. It is executable without child
assignments, default reads, digests, or caller-supplied projection laws. The
homomorphism theorem derives equality from the complete typed assignment.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.public_input_hom.scale` | every public coordinate uses base-field multiplication by the same fixed weight | computed | `publicInputScale`, `projectPublicInput_scale` |
| `nifs.pi_dec.verify.public_input_hom.finite` | assignment and public-input folds use identical head-first base weights | computed / derived | `combinePublicInputs`, `projectPublicInput_combine` |
| `nifs.pi_dec.verify.public_input_hom.radix` | child `i` has verifier-fixed production weight `2^i` | computed | `recomposePublicInput`, `projectPublicInput_recompose` |
| `nifs.pi_dec.verify.public_input_hom.algebra` | theorem has the exact `PiDEC.Algebra.publicInput_hom` field shape | derived | `relation_publicInput_hom` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-! ## Public base-scalar recomposition -/

/-- Base-field scaling of every coordinate in the exact aligned public
carrier. -/
def publicInputScale {shape : Shape}
    (scalar : F) (input : Phi81Relation.PublicInput shape) :
    Phi81Relation.PublicInput shape :=
  fun column => scalar * input column

/-- Canonical head-first base-scalar fold over public inputs. -/
def combinePublicInputs {shape : Shape} :
    {count : Nat} ->
      (Fin count -> F) ->
      (Fin count -> Phi81Relation.PublicInput shape) ->
      Phi81Relation.PublicInput shape
  | 0, _, _ => PiRLCAlgebra.PublicInput.publicZero
  | _ + 1, weights, inputs =>
      PiRLCAlgebra.PublicInput.publicAdd
        (publicInputScale (weights 0) (inputs 0))
        (combinePublicInputs
          (fun index => weights index.succ)
          (fun index => inputs index.succ))

/-- Production Π_DEC public-input recomposition with the verifier-owned
`2^i`, `i in [0, 14)`, base-field weights. -/
def recomposePublicInput {shape : Shape}
    (inputs : Radix.ChildIndex -> Phi81Relation.PublicInput shape) :
    Phi81Relation.PublicInput shape :=
  combinePublicInputs EvaluationHomomorphism.PiDEC.radixWeight inputs

/-! ## Projection linearity -/

/-- Projection commutes with one base-field assignment scale. -/
theorem projectPublicInput_scale {shape : Shape}
    (scalar : F) (assignment : Assignment shape) :
    projectPublicInput (BaseLinear.assignmentScale scalar assignment) =
      publicInputScale scalar (projectPublicInput assignment) := by
  rfl

/-- The complete assignment fold and public-input-only fold agree for every
finite base-scalar family. -/
theorem projectPublicInput_combine {shape : Shape} {count : Nat}
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape) :
    projectPublicInput (BaseLinear.combineAssignments weights assignments) =
      combinePublicInputs weights
        (fun index => projectPublicInput (assignments index)) := by
  induction count with
  | zero => exact PiRLCAlgebra.PublicInput.projectPublicInput_zero
  | succ count inductionHypothesis =>
      rw [BaseLinear.combineAssignments.eq_def, combinePublicInputs,
        PiRLCAlgebra.PublicInput.projectPublicInput_add,
        projectPublicInput_scale]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => assignments index.succ)]

/-- Exact production-radix public-input recomposition. -/
theorem projectPublicInput_recompose {shape : Shape}
    (assignments : Radix.ChildIndex -> Assignment shape) :
    projectPublicInput (Radix.recomposeAssignment assignments) =
      recomposePublicInput
        (fun index => projectPublicInput (assignments index)) := by
  exact projectPublicInput_combine
    EvaluationHomomorphism.PiDEC.radixWeight assignments

/-- Exact public-input field required by the concrete
`Folding.PiDEC.Algebra`. -/
theorem relation_publicInput_hom
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment) :
    forall assignments : Radix.ChildIndex -> Assignment shape,
      (relationSemantics commit).projectPublicInput
          (Radix.recomposeAssignment assignments) =
        recomposePublicInput fun index =>
          (relationSemantics commit).projectPublicInput (assignments index) := by
  intro assignments
  exact projectPublicInput_recompose assignments

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput
