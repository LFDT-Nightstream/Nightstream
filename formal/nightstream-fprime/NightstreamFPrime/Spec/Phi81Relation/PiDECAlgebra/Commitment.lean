import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Typed Ajtai commitment recomposition for the concrete Phi81 `Pi_DEC` algebra.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 16`.

Owns: base-field scaling of one typed public Ajtai commitment; the canonical
head-first finite base-scalar fold; specialization to the verifier-fixed
`2^i` radix weights; and the exact theorem required by
`Folding.PiDEC.Algebra.commit_hom`.

Does not own: digit splitting, the Ajtai key, key generation or serialization,
Ajtai binding or MSIS security, child-opening validity, transcript binding,
Rust/R1CS refinement, row removal, or constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Commitment

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def commitmentScale {verifierRows : Nat}
    (scalar : F) (value : PiRLCAlgebra.Commitment.Value verifierRows) :
    PiRLCAlgebra.Commitment.Value verifierRows :=
  fun row => CarrierAction.ringFScale scalar (value row)

def combineCommitments {verifierRows : Nat} :
    {count : Nat} ->
      (Fin count -> F) ->
      (Fin count -> PiRLCAlgebra.Commitment.Value verifierRows) ->
      PiRLCAlgebra.Commitment.Value verifierRows
  | 0, _, _ => PiRLCAlgebra.Commitment.commitmentZero
  | _ + 1, weights, values =>
      PiRLCAlgebra.Commitment.commitmentAdd
        (commitmentScale (weights 0) (values 0))
        (combineCommitments
          (fun index => weights index.succ)
          (fun index => values index.succ))

def recomposeCommitment {verifierRows : Nat}
    (values : Radix.ChildIndex ->
      PiRLCAlgebra.Commitment.Value verifierRows) :
    PiRLCAlgebra.Commitment.Value verifierRows :=
  combineCommitments EvaluationHomomorphism.PiDEC.radixWeight values

private theorem ringFScale_zero (scalar : F) :
    CarrierAction.ringFScale scalar ringFZero = ringFZero := by
  funext lane
  exact Fin.mul_zero scalar

private theorem ringFScale_add (scalar : F) (left right : RingF) :
    CarrierAction.ringFScale scalar (ringFAdd left right) =
      ringFAdd
        (CarrierAction.ringFScale scalar left)
        (CarrierAction.ringFScale scalar right) := by
  funext lane
  exact ConcreteCarrier.baseLaws.left_distrib scalar (left lane) (right lane)

private theorem ringFSum_congr {count : Nat}
    {left right : Fin count -> RingF}
    (equal : forall index, left index = right index) :
    PiRLCAlgebra.Commitment.ringFSum left =
      PiRLCAlgebra.Commitment.ringFSum right := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PiRLCAlgebra.Commitment.ringFSum,
        PiRLCAlgebra.Commitment.ringFSum, equal 0]
      rw [inductionHypothesis (fun index => equal index.succ)]

private theorem ringFSum_scale {count : Nat}
    (scalar : F) (terms : Fin count -> RingF) :
    PiRLCAlgebra.Commitment.ringFSum
        (fun index => CarrierAction.ringFScale scalar (terms index)) =
      CarrierAction.ringFScale scalar
        (PiRLCAlgebra.Commitment.ringFSum terms) := by
  induction count with
  | zero => exact (ringFScale_zero scalar).symm
  | succ count inductionHypothesis =>
      rw [PiRLCAlgebra.Commitment.ringFSum,
        PiRLCAlgebra.Commitment.ringFSum,
        inductionHypothesis
          (fun index => terms index.succ)]
      exact (ringFScale_add scalar (terms 0)
        (PiRLCAlgebra.Commitment.ringFSum
          fun index => terms index.succ)).symm

private theorem assignmentBlock_scale {shape : Shape}
    (scalar : F) (assignment : Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    CarrierAction.assignmentBlock
        (BaseLinear.assignmentScale scalar assignment) block =
      CarrierAction.ringFScale scalar
        (CarrierAction.assignmentBlock assignment block) := by
  rfl

theorem commit_scale {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (scalar : F) (assignment : Assignment shape) :
    PiRLCAlgebra.Commitment.commit key
        (BaseLinear.assignmentScale scalar assignment) =
      commitmentScale scalar
        (PiRLCAlgebra.Commitment.commit key assignment) := by
  funext row
  unfold PiRLCAlgebra.Commitment.commit PiRLCAlgebra.Commitment.ajtaiRow
    PiRLCAlgebra.Commitment.blockSum commitmentScale
  calc
    PiRLCAlgebra.Commitment.ringFSum (fun block =>
        ringFMul (key row block)
          (CarrierAction.assignmentBlock
            (BaseLinear.assignmentScale scalar assignment) block)) =
      PiRLCAlgebra.Commitment.ringFSum (fun block =>
        CarrierAction.ringFScale scalar
          (ringFMul (key row block)
            (CarrierAction.assignmentBlock assignment block))) := by
      apply ringFSum_congr
      intro block
      rw [assignmentBlock_scale, CarrierAction.ringFMul_scale_right]
    _ = CarrierAction.ringFScale scalar
        (PiRLCAlgebra.Commitment.ringFSum fun block =>
          ringFMul (key row block)
            (CarrierAction.assignmentBlock assignment block)) :=
      ringFSum_scale scalar _

theorem commit_combine {shape : Shape} {verifierRows count : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape) :
    PiRLCAlgebra.Commitment.commit key
        (BaseLinear.combineAssignments weights assignments) =
      combineCommitments weights
        (fun index =>
          PiRLCAlgebra.Commitment.commit key (assignments index)) := by
  induction count with
  | zero => exact PiRLCAlgebra.Commitment.commit_zero key
  | succ count inductionHypothesis =>
      rw [BaseLinear.combineAssignments.eq_def, combineCommitments,
        PiRLCAlgebra.Commitment.commit_add, commit_scale]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => assignments index.succ)]

theorem commit_recompose {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (assignments : Radix.ChildIndex -> Assignment shape) :
    PiRLCAlgebra.Commitment.commit key
        (Radix.recomposeAssignment assignments) =
      recomposeCommitment
        (fun index =>
          PiRLCAlgebra.Commitment.commit key (assignments index)) := by
  exact commit_combine key EvaluationHomomorphism.PiDEC.radixWeight assignments

theorem relation_commit_hom {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    forall assignments : Radix.ChildIndex -> Assignment shape,
      (relationSemantics (PiRLCAlgebra.Commitment.commit key)).commit
          (Radix.recomposeAssignment assignments) =
        recomposeCommitment fun index =>
          (relationSemantics
            (PiRLCAlgebra.Commitment.commit key)).commit
            (assignments index) := by
  intro assignments
  exact commit_recompose key assignments

end NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Commitment
