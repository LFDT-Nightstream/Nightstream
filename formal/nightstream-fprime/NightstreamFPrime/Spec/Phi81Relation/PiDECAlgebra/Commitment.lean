import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiDECAlgebra/Commitment.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Typed Ajtai commitment recomposition for the concrete Phi81 `Pi_DEC` algebra.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 16`.
Phase: public parent-commitment recomposition from the sixteen child
commitments.
Constraint family: semantic commitment recomposition only; this file emits no
rows.

Owns: base-field scaling of one typed public Ajtai commitment; the canonical
head-first finite base-scalar fold; specialization to the verifier-fixed
`2^i` radix weights; and the exact theorem required by
`Folding.PiDEC.Algebra.commit_hom`.

Does not own: digit splitting, the Ajtai key, key generation or serialization,
Ajtai binding or MSIS security, child-opening validity, transcript binding,
Rust/R1CS refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `recomposeCommitment` consumes only the sixteen public
child commitments and verifier-fixed radix weights. The typed Ajtai key remains
the sole commitment map input. No assignment, digest, caller-supplied
linearity law, or prover-selected scalar enters public recomposition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.commitment_hom.scale` | base scalar `s` acts coefficientwise on every public Ajtai row | computed | `commitmentScale`, `commit_scale` |
| `nifs.pi_dec.verify.commitment_hom.finite` | assignment and public commitment folds use identical head-first base weights | computed / derived | `combineCommitments`, `commit_combine` |
| `nifs.pi_dec.verify.commitment_hom.radix` | child `i` has verifier-fixed production weight `2^i` | computed | `recomposeCommitment`, `commit_recompose` |
| `nifs.pi_dec.verify.commitment_hom.algebra` | theorem has the exact `PiDEC.Algebra.commit_hom` field shape | derived | `relation_commit_hom` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Commitment

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-! ## Public base-scalar recomposition -/

/-- Base-field scaling of every coefficient in every typed public Ajtai row. -/
def commitmentScale {verifierRows : Nat}
    (scalar : F) (value : PiRLCAlgebra.Commitment.Value verifierRows) :
    PiRLCAlgebra.Commitment.Value verifierRows :=
  fun row => CarrierAction.ringFScale scalar (value row)

/-- Canonical head-first base-scalar fold over public commitments. -/
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

/-- Production Π_DEC commitment recomposition with the verifier-owned
`2^i`, `i in [0, 16)`, base-field weights. -/
def recomposeCommitment {verifierRows : Nat}
    (values : Radix.ChildIndex ->
      PiRLCAlgebra.Commitment.Value verifierRows) :
    PiRLCAlgebra.Commitment.Value verifierRows :=
  combineCommitments EvaluationHomomorphism.PiDEC.radixWeight values

/-! ## Typed Ajtai linearity -/

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

/-- The typed Ajtai commitment commutes with one base-field assignment scale. -/
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

/-- The complete assignment fold and public commitment fold agree for every
finite base-scalar family. -/
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

/-- Exact production-radix commitment recomposition. -/
theorem commit_recompose {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (assignments : Radix.ChildIndex -> Assignment shape) :
    PiRLCAlgebra.Commitment.commit key
        (Radix.recomposeAssignment assignments) =
      recomposeCommitment
        (fun index =>
          PiRLCAlgebra.Commitment.commit key (assignments index)) := by
  exact commit_combine key EvaluationHomomorphism.PiDEC.radixWeight assignments

/-- Exact commitment field required by the concrete
`Folding.PiDEC.Algebra`. -/
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
