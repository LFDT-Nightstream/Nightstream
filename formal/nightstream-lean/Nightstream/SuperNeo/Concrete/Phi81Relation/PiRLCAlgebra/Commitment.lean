import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Typed Ajtai commitment homomorphism for the concrete Phi81 `Pi_RLC` action.

Protocol: SuperNeo Theorem 5, commitment branch of `Pi_RLC`.
Phase: complete assignment action to the verifier-owned Ajtai rows.
Constraint family: semantic commitment combination only; this file emits no
rows.

Owns: an exact finite key over a generic verifier-owned row count and every
complete 54-lane carrier block; the canonical finite Ajtai row equation; the
public commitment-only challenge fold; and the one-action and finite-batch
commitment homomorphisms required by `PiRLC.Algebra.commit_hom`.

Does not own: key generation, key serialization, Ajtai binding or MSIS
security, norm growth, challenge validity, transcript derivation, Rust/R1CS
refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `combineCommitments` consumes only public commitments and
challenges. The key and assignment domains are exact finite types, so neither
the row equation nor its proof uses default reads, truncation, a caller-supplied
linearity law, or a prover-carried digest.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.commitment_hom.key` | every verifier row contains one key block for every complete carrier block | typed input | `Key` |
| `nifs.pi_rlc.verify.commitment_hom.row` | `C_r = sum_j A_(r,j) * z_j` in canonical block order | computed | `ajtaiRow` |
| `nifs.pi_rlc.verify.commitment_hom.action` | `commit (rho * z) = rho * commit z` | derived | `commit_act` |
| `nifs.pi_rlc.verify.commitment_hom.finite` | public commitments use the identical head-first challenge fold | computed/derived | `combineCommitments`, `commit_combine` |
| `nifs.pi_rlc.verify.commitment_hom.algebra` | theorem has the exact algebra-field signature | derived | `relation_commit_hom` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-! ## Exact key and row equation -/

/-- A verifier-owned Ajtai key with exactly one `RingF` block per commitment
row and complete assignment block. -/
abbrev Key (shape : Shape) (verifierRows : Nat) :=
  Fin verifierRows ->
    Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) -> RingF

/-- One typed public Ajtai commitment. -/
abbrev Value (verifierRows : Nat) := Fin verifierRows -> RingF

/-- Canonical head-first sum over an exact finite domain. -/
def ringFSum : {count : Nat} -> (Fin count -> RingF) -> RingF
  | 0, _ => ringFZero
  | _ + 1, terms =>
      ringFAdd (terms 0) (ringFSum fun index => terms index.succ)

/-- Canonical sum over every complete carrier block. -/
def blockSum {shape : Shape}
    (terms : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) -> RingF) :
    RingF :=
  ringFSum terms

/-- One Ajtai row, computed as the exact finite ring inner product between the
verifier-owned key row and all complete assignment blocks. -/
def ajtaiRow {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows) (assignment : Assignment shape)
    (row : Fin verifierRows) : RingF :=
  blockSum fun block =>
    ringFMul (key row block) (CarrierAction.assignmentBlock assignment block)

/-- The complete typed Ajtai commitment. -/
def commit {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows) (assignment : Assignment shape) :
    Value verifierRows :=
  fun row => ajtaiRow key assignment row

/-! ## Public commitment action and finite combination -/

/-- One challenge acts on a public commitment without access to an opening. -/
def commitmentAct {verifierRows : Nat}
    (challenge : RingF) (value : Value verifierRows) : Value verifierRows :=
  fun row => ringFMul challenge (value row)

/-- Pointwise addition of two public commitments. -/
def commitmentAdd {verifierRows : Nat}
    (left right : Value verifierRows) : Value verifierRows :=
  fun row => ringFAdd (left row) (right row)

/-- Canonical zero public commitment. -/
def commitmentZero {verifierRows : Nat} : Value verifierRows :=
  fun _ => ringFZero

/-- Canonical head-first challenge fold computed from public commitments alone. -/
def combineCommitments {verifierRows : Nat} :
    {count : Nat} ->
      (Fin count -> RingF) ->
      (Fin count -> Value verifierRows) -> Value verifierRows
  | 0, _, _ => commitmentZero
  | _ + 1, challenges, values =>
      commitmentAdd
        (commitmentAct (challenges 0) (values 0))
        (combineCommitments
          (fun index => challenges index.succ)
          (fun index => values index.succ))

/-! ## Finite-sum algebra -/

private theorem ringFAdd_assoc (left middle right : RingF) :
    ringFAdd (ringFAdd left middle) right =
      ringFAdd left (ringFAdd middle right) := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem ringFAdd_comm (left right : RingF) :
    ringFAdd left right = ringFAdd right left := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_comm _ _

private theorem ringFZero_add (value : RingF) :
    ringFAdd ringFZero value = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

private theorem ringFSum_congr {count : Nat}
    {left right : Fin count -> RingF}
    (equal : forall index, left index = right index) :
    ringFSum left = ringFSum right := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [ringFSum, ringFSum, equal 0]
      rw [inductionHypothesis (fun index => equal index.succ)]

private theorem ringFSum_zero {count : Nat} :
    ringFSum (fun _ : Fin count => ringFZero) = ringFZero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [ringFSum, inductionHypothesis, ringFZero_add]

private theorem ringFSum_add {count : Nat}
    (left right : Fin count -> RingF) :
    ringFSum (fun index => ringFAdd (left index) (right index)) =
      ringFAdd (ringFSum left) (ringFSum right) := by
  induction count with
  | zero => exact (ringFZero_add ringFZero).symm
  | succ count inductionHypothesis =>
      rw [ringFSum, ringFSum, ringFSum, inductionHypothesis]
      calc
        ringFAdd (ringFAdd (left 0) (right 0))
            (ringFAdd
              (ringFSum fun index => left index.succ)
              (ringFSum fun index => right index.succ)) =
            ringFAdd (left 0)
              (ringFAdd (right 0)
                (ringFAdd
                  (ringFSum fun index => left index.succ)
                  (ringFSum fun index => right index.succ))) :=
          ringFAdd_assoc _ _ _
        _ = ringFAdd (left 0)
              (ringFAdd
                (ringFSum fun index => left index.succ)
                (ringFAdd (right 0)
                  (ringFSum fun index => right index.succ))) := by
          congr 1
          calc
            ringFAdd (right 0)
                (ringFAdd
                  (ringFSum fun index => left index.succ)
                  (ringFSum fun index => right index.succ)) =
                ringFAdd
                  (ringFAdd (right 0)
                    (ringFSum fun index => left index.succ))
                  (ringFSum fun index => right index.succ) :=
              (ringFAdd_assoc _ _ _).symm
            _ = ringFAdd
                  (ringFAdd
                    (ringFSum fun index => left index.succ)
                    (right 0))
                  (ringFSum fun index => right index.succ) := by
              rw [ringFAdd_comm (right 0)
                (ringFSum fun index => left index.succ)]
            _ = ringFAdd
                  (ringFSum fun index => left index.succ)
                  (ringFAdd (right 0)
                    (ringFSum fun index => right index.succ)) :=
              ringFAdd_assoc _ _ _
        _ = ringFAdd
              (ringFAdd (left 0)
                (ringFSum fun index => left index.succ))
              (ringFAdd (right 0)
                (ringFSum fun index => right index.succ)) :=
          (ringFAdd_assoc _ _ _).symm

private theorem ringFSum_act {count : Nat}
    (challenge : RingF) (terms : Fin count -> RingF) :
    ringFSum (fun index => ringFMul challenge (terms index)) =
      ringFMul challenge (ringFSum terms) := by
  induction count with
  | zero => exact (CarrierAction.ringFMul_zero_right challenge).symm
  | succ count inductionHypothesis =>
      rw [ringFSum, ringFSum, inductionHypothesis,
        CarrierAction.ringFMul_add_right]

/-! ## One-action and finite commitment homomorphisms -/

/-- Reading one block commutes definitionally with complete-assignment
addition. -/
private theorem assignmentBlock_add {shape : Shape}
    (left right : Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    CarrierAction.assignmentBlock (BaseLinear.assignmentAdd left right) block =
      ringFAdd
        (CarrierAction.assignmentBlock left block)
        (CarrierAction.assignmentBlock right block) := by
  rfl

/-- Every block of the canonical complete assignment zero is the ring zero. -/
private theorem assignmentBlock_zero {shape : Shape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    CarrierAction.assignmentBlock
        (BaseLinear.assignmentZero : Assignment shape) block = ringFZero := by
  rfl

/-- Ajtai commitment is additive in the complete typed assignment. -/
theorem commit_add {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows) (left right : Assignment shape) :
    commit key (BaseLinear.assignmentAdd left right) =
      commitmentAdd (commit key left) (commit key right) := by
  funext row
  unfold commit ajtaiRow blockSum commitmentAdd
  calc
    ringFSum (fun block =>
        ringFMul (key row block)
          (CarrierAction.assignmentBlock
            (BaseLinear.assignmentAdd left right) block)) =
      ringFSum (fun block =>
        ringFAdd
          (ringFMul (key row block)
            (CarrierAction.assignmentBlock left block))
          (ringFMul (key row block)
            (CarrierAction.assignmentBlock right block))) := by
      apply ringFSum_congr
      intro block
      rw [assignmentBlock_add, CarrierAction.ringFMul_add_right]
    _ = ringFAdd
        (ringFSum fun block =>
          ringFMul (key row block)
            (CarrierAction.assignmentBlock left block))
        (ringFSum fun block =>
          ringFMul (key row block)
            (CarrierAction.assignmentBlock right block)) :=
      ringFSum_add _ _

/-- Ajtai commitment maps the canonical complete assignment zero to the public
commitment zero. -/
theorem commit_zero {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows) :
    commit key (BaseLinear.assignmentZero : Assignment shape) =
      commitmentZero := by
  funext row
  unfold commit ajtaiRow blockSum commitmentZero
  calc
    ringFSum (fun block =>
        ringFMul (key row block)
          (CarrierAction.assignmentBlock BaseLinear.assignmentZero block)) =
      ringFSum (fun _ => ringFZero) := by
      apply ringFSum_congr
      intro block
      rw [assignmentBlock_zero, CarrierAction.ringFMul_zero_right]
    _ = ringFZero := ringFSum_zero

/-- One complete-assignment challenge action commutes with the exact Ajtai
commitment. The product-order step is the symbolic executable Phi81 theorem. -/
theorem commit_act {shape : Shape} {verifierRows : Nat}
    (key : Key shape verifierRows) (challenge : RingF)
    (assignment : Assignment shape) :
    commit key (CarrierAction.act challenge assignment) =
      commitmentAct challenge (commit key assignment) := by
  funext row
  unfold commit ajtaiRow blockSum commitmentAct
  calc
    ringFSum (fun block =>
        ringFMul (key row block)
          (CarrierAction.assignmentBlock
            (CarrierAction.act challenge assignment) block)) =
      ringFSum (fun block =>
        ringFMul challenge
          (ringFMul (key row block)
            (CarrierAction.assignmentBlock assignment block))) := by
      apply ringFSum_congr
      intro block
      rw [CarrierAction.assignmentBlock_act]
      exact RingFLaws.ringFMul_leftActionComm
        (key row block) challenge
        (CarrierAction.assignmentBlock assignment block)
    _ = ringFMul challenge
        (ringFSum fun block =>
          ringFMul (key row block)
            (CarrierAction.assignmentBlock assignment block)) :=
      ringFSum_act challenge _

/-- The exact finite assignment fold and the public commitment-only fold agree
for every verifier-owned key and row count. -/
theorem commit_combine {shape : Shape} {verifierRows count : Nat}
    (key : Key shape verifierRows)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    commit key (PiRLCFinite.combineAssignments challenges assignments) =
      combineCommitments challenges (fun index => commit key (assignments index)) := by
  induction count with
  | zero => exact commit_zero key
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.combineAssignments, combineCommitments,
        commit_add, commit_act]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)]

/-- Exact commitment field required by a future concrete
`Folding.PiRLC.Algebra`. -/
theorem relation_commit_hom {shape : Shape} {verifierRows count : Nat}
    (key : Key shape verifierRows)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    (relationSemantics (commit key)).commit
        (PiRLCFinite.combineAssignments challenges assignments) =
      combineCommitments challenges fun index =>
        (relationSemantics (commit key)).commit (assignments index) := by
  exact commit_combine key challenges assignments

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment
