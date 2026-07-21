import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput

/-!
Typed public-input split and recomposition for the concrete Phi81 `Pi_DEC`
verifier.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 14`.
Phase: verifier-owned child public inputs and parent-input recomposition.
Constraint family: semantic public-input operations only; this file emits no
rows.

Owns: the coordinatewise public radix split; its projection-commuting and
recomposition laws; base-field scaling of one exact ring-aligned public
carrier; the canonical head-first finite base-scalar fold; specialization to
the verifier-fixed `2^i` radix weights; and the exact theorem required by
`Folding.PiDEC.Algebra.publicInput_hom`.

Does not own: assignment splitting, commitments, evaluation recomposition,
public-input authority before projection, transcript binding, Rust/R1CS
refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `splitPublicInput` consumes only the public parent and is
the split run by the paper verifier. `recomposePublicInput` consumes only the
fourteen public children and verifier-fixed radix weights. Both are executable
without private assignments, default reads, digests, or caller-supplied laws.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.paper.public_split` | split every public coordinate with the same total radix map | computed | `splitPublicInput` |
| `nifs.pi_dec.paper.public_split.projection` | public splitting commutes with assignment projection | derived | `splitPublicInput_project` |
| `nifs.pi_dec.paper.public_split.recompose` | every public input recomposes from its fourteen public children | derived | `splitPublicInput_recompose` |
| `nifs.pi_dec.verify.public_input_hom.scale` | every public coordinate uses base-field multiplication by the same fixed weight | computed | `publicInputScale`, `projectPublicInput_scale` |
| `nifs.pi_dec.verify.public_input_hom.finite` | assignment and public-input folds use identical head-first base weights | computed / derived | `combinePublicInputs`, `projectPublicInput_combine` |
| `nifs.pi_dec.verify.public_input_hom.radix` | child `i` has verifier-fixed production weight `2^i` | computed | `recomposePublicInput`, `projectPublicInput_recompose` |
| `nifs.pi_dec.verify.public_input_hom.algebra` | theorem has the exact `PiDEC.Algebra.publicInput_hom` field shape | derived | `relation_publicInput_hom` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-! ## Verifier-owned public split -/

/-- Coordinatewise public split computed by the Section-7.5 verifier. -/
def splitPublicInput {shape : Shape}
    (input : Phi81Relation.PublicInput shape)
    (child : Radix.ChildIndex) : Phi81Relation.PublicInput shape :=
  fun column => Radix.splitScalar (input column) child

/-- Splitting the authoritative projected public input is exactly projection
of the corresponding complete-assignment digit. -/
@[simp] theorem splitPublicInput_project {shape : Shape}
    (assignment : Assignment shape) (child : Radix.ChildIndex) :
    splitPublicInput (projectPublicInput assignment) child =
      projectPublicInput (Radix.splitAssignment assignment child) := by
  rfl

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

/-- Public recomposition is coordinatewise scalar recomposition. -/
@[simp] theorem recomposePublicInput_apply {shape : Shape}
    (inputs : Radix.ChildIndex -> Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    recomposePublicInput inputs column =
      Radix.recomposeScalar (fun child => inputs child column) := by
  rfl

/-- The verifier-computed public children recompose exactly to their parent
for every public input, including the total out-of-bound fallback. -/
theorem splitPublicInput_recompose {shape : Shape}
    (input : Phi81Relation.PublicInput shape) :
    recomposePublicInput (splitPublicInput input) = input := by
  funext column
  rw [recomposePublicInput_apply]
  exact Radix.splitScalar_recompose (input column)

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
