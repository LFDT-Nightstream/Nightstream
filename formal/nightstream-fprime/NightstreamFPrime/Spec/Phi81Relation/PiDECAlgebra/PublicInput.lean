import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

/-- SuperNeo v1.2 Section 7.5. -/
def parentBounded {shape : Shape}
    (input : Phi81Relation.PublicInput shape) : Prop :=
  ∀ column, centeredMagnitude (input column) < Radix.combinedBound

def parentBounded_decidable {shape : Shape}
    (input : Phi81Relation.PublicInput shape) : Decidable (parentBounded input) := by
  unfold parentBounded
  infer_instance

theorem parentBounded_project {shape : Shape}
    (assignment : Assignment shape)
    (bounded : assignmentNormBounded productionGlobalParams.bigB assignment) :
    parentBounded (projectPublicInput assignment) := by
  intro column
  simpa [parentBounded, Radix.combinedBound] using!
    bounded (shape.publicColumn column)

/-- SuperNeo v1.2 Section 7.5. -/
def splitPublicInput {shape : Shape}
    (input : Phi81Relation.PublicInput shape)
    (child : Radix.ChildIndex) : Phi81Relation.PublicInput shape :=
  fun column => Radix.splitScalar (input column) child

theorem splitPublicInput_eq_boundedDigit {shape : Shape}
    (input : Phi81Relation.PublicInput shape)
    (bounded : parentBounded input)
    (child : Radix.ChildIndex)
    (column : Fin shape.publicWidth) :
    splitPublicInput input child column =
      Radix.boundedDigit (input column) child := by
  simp [splitPublicInput, Radix.splitScalar, bounded column]

@[simp] theorem splitPublicInput_project {shape : Shape}
    (assignment : Assignment shape) (child : Radix.ChildIndex) :
    splitPublicInput (projectPublicInput assignment) child =
      projectPublicInput (Radix.splitAssignment assignment child) := by
  rfl

def publicInputScale {shape : Shape}
    (scalar : F) (input : Phi81Relation.PublicInput shape) :
    Phi81Relation.PublicInput shape :=
  fun column => scalar * input column

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

def recomposePublicInput {shape : Shape}
    (inputs : Radix.ChildIndex -> Phi81Relation.PublicInput shape) :
    Phi81Relation.PublicInput shape :=
  combinePublicInputs EvaluationHomomorphism.PiDEC.radixWeight inputs

@[simp] theorem recomposePublicInput_apply {shape : Shape}
    (inputs : Radix.ChildIndex -> Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    recomposePublicInput inputs column =
      Radix.recomposeScalar (fun child => inputs child column) := by
  rfl

theorem splitPublicInput_recompose {shape : Shape}
    (input : Phi81Relation.PublicInput shape) :
    recomposePublicInput (splitPublicInput input) = input := by
  funext column
  rw [recomposePublicInput_apply]
  exact Radix.splitScalar_recompose (input column)

theorem projectPublicInput_scale {shape : Shape}
    (scalar : F) (assignment : Assignment shape) :
    projectPublicInput (BaseLinear.assignmentScale scalar assignment) =
      publicInputScale scalar (projectPublicInput assignment) := by
  rfl

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

theorem projectPublicInput_recompose {shape : Shape}
    (assignments : Radix.ChildIndex -> Assignment shape) :
    projectPublicInput (Radix.recomposeAssignment assignments) =
      recomposePublicInput
        (fun index => projectPublicInput (assignments index)) := by
  exact projectPublicInput_combine
    EvaluationHomomorphism.PiDEC.radixWeight assignments

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

end NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput
