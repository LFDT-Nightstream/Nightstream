import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.CommitmentRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.EvalARecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.EvalKRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.InputBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.OutputBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.PublicInputSplit
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.Support.VariableSupport

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_2.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.RingRelation.PiDECAlgebra

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  parent : Nat → InputBinding.ParentExpr logicalWidth publicFits
  point : Nat → Fin productionShape.cubeVariables → KExpr
  message : Nat → Radix.ChildIndex → InputBinding.ChildMessageExpr
  digit : Nat → Radix.ChildIndex →
    Fin (PublicInputSplit.coordinateCount logicalWidth publicFits) → Expr

def atOffset
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Interface logicalWidth publicFits where
  parent := fun _ => interface.parent offset
  point := fun _ => interface.point offset
  message := fun _ => interface.message offset
  digit := fun _ => interface.digit offset

def inputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    InputBinding.Interface logicalWidth publicFits where
  parent := interface.parent
  point := interface.point
  message := interface.message

def publicInputInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    PublicInputSplit.Interface logicalWidth publicFits where
  parent := fun offset => (interface.parent offset).publicInput
  digit := interface.digit

def commitmentInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    CommitmentRecomposition.Interface where
  parent := fun offset => (interface.parent offset).commitment
  child := fun offset child => (interface.message offset child).commitment

def evalKInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    EvalKRecomposition.Interface where
  parent := fun offset => (interface.parent offset).evaluation.eval_K
  child := fun offset child =>
    (interface.message offset child).evaluation.eval_K

def evalAInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    EvalARecomposition.Interface where
  parent := fun offset => (interface.parent offset).evaluation.eval_A
  child := fun offset child =>
    (interface.message offset child).evaluation.eval_A

def outputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    OutputBinding.Interface logicalWidth publicFits where
  point := interface.point
  message := interface.message
  publicInput := interface.digit

def inputBindingOffset (offset : Nat) : Nat := offset
def publicInputOffset (offset : Nat) : Nat := offset

def recompositionOffset (offset : Nat) : Nat :=
  offset + PublicInputSplit.exactPrivateCount
def commitmentOffset (offset : Nat) : Nat := recompositionOffset offset
def evalKOffset (offset : Nat) : Nat := recompositionOffset offset
def evalAOffset (offset : Nat) : Nat := recompositionOffset offset
def outputBindingOffset (offset : Nat) : Nat := recompositionOffset offset
def finalOffset (offset : Nat) : Nat := recompositionOffset offset

def inputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  InputBinding.circuit relation (inputBindingInterface interface)

def publicInputCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  PublicInputSplit.circuit (publicInputInterface interface)

def commitmentCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  CommitmentRecomposition.circuit (commitmentInterface interface)

def evalKCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalKRecomposition.circuit (evalKInterface interface)

def evalACircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalARecomposition.circuit (evalAInterface interface)

def outputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  OutputBinding.circuit relation (outputBindingInterface interface)

def childOp (name : String) (child : FormalCircuit) (offset : Nat) : Op :=
  Sequence.childOp name child offset

def opsAt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "pidec.v1_2.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset),
    childOp "pidec.v1_2.public_input_split"
      (publicInputCircuit shared) (publicInputOffset offset),
    childOp "pidec.v1_2.commitment_recomposition"
      (commitmentCircuit shared) (commitmentOffset offset),
    childOp "pidec.v1_2.eval_K_recomposition"
      (evalKCircuit shared) (evalKOffset offset),
    childOp "pidec.v1_2.eval_A_recomposition"
      (evalACircuit shared) (evalAOffset offset),
    childOp "pidec.v1_2.output_binding"
      (outputBindingCircuit relation shared) (outputBindingOffset offset)]

def main
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt relation interface offset)

@[simp] theorem main_ops
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Circuit.ops (main relation interface) offset =
      opsAt relation interface offset := by
  rfl

def logicalPrivateCount : Nat := PublicInputSplit.exactPrivateCount
def logicalRowCount : Nat :=
  PublicInputSplit.exactRowCount + CommitmentRecomposition.coordinateCount +
    RingKRecomposition.coordinateCount EvalKRecomposition.blockCount +
    RingKRecomposition.coordinateCount EvalARecomposition.blockCount

structure InputsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) : Prop where
  point : ∀ coordinate, (interface.point offset coordinate).VarsBelow offset
  parentCommitment : ∀ row lane,
    ((interface.parent offset).commitment row lane).VarsBelow offset
  parentPublicInput : ∀ coordinate,
    ((interface.parent offset).publicInput coordinate).VarsBelow offset
  parentEval_K : ∀ coefficient,
    ((interface.parent offset).evaluation.eval_K coefficient).VarsBelow offset
  parentEval_A : ∀ matrix coefficient,
    ((interface.parent offset).evaluation.eval_A matrix coefficient).VarsBelow offset
  messageCommitment : ∀ child row lane,
    ((interface.message offset child).commitment row lane).VarsBelow offset
  messageEval_K : ∀ child coefficient,
    ((interface.message offset child).evaluation.eval_K coefficient).VarsBelow offset
  messageEval_A : ∀ child matrix coefficient,
    ((interface.message offset child).evaluation.eval_A matrix coefficient).VarsBelow offset
  digit : ∀ child coordinate,
    (interface.digit offset child coordinate).VarsBelow offset

structure Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (_env : Env) : Prop where
  inputs : InputsBelow interface offset

theorem publicInputAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : PublicInputSplit.Assumptions
      (publicInputInterface (atOffset interface offset))
      (publicInputOffset offset) current where
  parentBelow := assumptions.inputs.parentPublicInput
  digitBelow := assumptions.inputs.digit

theorem commitmentAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : CommitmentRecomposition.Assumptions
      (commitmentInterface (atOffset interface offset))
      (commitmentOffset offset) current where
  parentBelow := fun coordinate => by
    apply Expr.VarsBelow.mono _
      (assumptions.inputs.parentCommitment
        (CommitmentRecomposition.coordinates coordinate).1
        (CommitmentRecomposition.coordinates coordinate).2)
    simp [commitmentOffset, recompositionOffset]
  childBelow := fun child coordinate => by
    apply Expr.VarsBelow.mono _
      (assumptions.inputs.messageCommitment child
        (CommitmentRecomposition.coordinates coordinate).1
        (CommitmentRecomposition.coordinates coordinate).2)
    simp [commitmentOffset, recompositionOffset]

theorem evalKAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : EvalKRecomposition.Assumptions
      (evalKInterface (atOffset interface offset)) (evalKOffset offset) current where
  parentBelow := fun coordinate => by
    apply RingKRecomposition.expressionCell_varsBelow
    apply KExpr.varsBelow_mono _
      (assumptions.inputs.parentEval_K
        (EvalKRecomposition.coefficient
          (RingKRecomposition.coordinates coordinate).2.1))
    simp [evalKOffset, recompositionOffset]
  childBelow := fun child coordinate => by
    apply RingKRecomposition.expressionCell_varsBelow
    apply KExpr.varsBelow_mono _
      (assumptions.inputs.messageEval_K child
        (EvalKRecomposition.coefficient
          (RingKRecomposition.coordinates coordinate).2.1))
    simp [evalKOffset, recompositionOffset]

theorem evalAAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : EvalARecomposition.Assumptions
      (evalAInterface (atOffset interface offset)) (evalAOffset offset) current where
  parentBelow := fun coordinate => by
    apply RingKRecomposition.expressionCell_varsBelow
    apply KExpr.varsBelow_mono _
      (assumptions.inputs.parentEval_A
        (RingKRecomposition.coordinates coordinate).1
        (EvalKRecomposition.coefficient
          (RingKRecomposition.coordinates coordinate).2.1))
    simp [evalAOffset, recompositionOffset]
  childBelow := fun child coordinate => by
    apply RingKRecomposition.expressionCell_varsBelow
    apply KExpr.varsBelow_mono _
      (assumptions.inputs.messageEval_A child
        (RingKRecomposition.coordinates coordinate).1
        (EvalKRecomposition.coefficient
          (RingKRecomposition.coordinates coordinate).2.1))
    simp [evalAOffset, recompositionOffset]

structure SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop where
  inputBinding : InputBinding.SpecHolds relation
    (inputBindingInterface (atOffset interface offset))
    (inputBindingOffset offset) env
  publicInput : PublicInputSplit.RelationHolds
    (publicInputInterface (atOffset interface offset))
    (publicInputOffset offset) env
  commitment : CommitmentRecomposition.SpecHolds
    (commitmentInterface (atOffset interface offset))
    (commitmentOffset offset) env
  eval_K : EvalKRecomposition.SpecHolds
    (evalKInterface (atOffset interface offset)) (evalKOffset offset) env
  eval_A : EvalARecomposition.SpecHolds
    (evalAInterface (atOffset interface offset)) (evalAOffset offset) env
  outputBinding : OutputBinding.SpecHolds relation
    (outputBindingInterface (atOffset interface offset))
    (outputBindingOffset offset) env

private theorem childSpec_of_rows
    (name : String) (child : FormalCircuit) (childOffset : Nat)
    (env : Env) (operations : List Op)
    (rows : holds env operations)
    (member : childOp name child childOffset ∈ operations)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have call := rows _ member
  exact call assumptions

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset)) :
    SpecHolds relation interface offset env := by
  let shared := atOffset interface offset
  change holds env (opsAt relation interface offset) at rows
  refine {
    inputBinding := childSpec_of_rows "pidec.v1_2.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset)
      env _ rows (by simp [opsAt, shared]) trivial
    publicInput := childSpec_of_rows "pidec.v1_2.public_input_split"
      (publicInputCircuit shared) (publicInputOffset offset)
      env _ rows (by simp [opsAt, shared])
      (publicInputAssumptions assumptions env)
    commitment := childSpec_of_rows "pidec.v1_2.commitment_recomposition"
      (commitmentCircuit shared) (commitmentOffset offset)
      env _ rows (by simp [opsAt, shared])
      (commitmentAssumptions assumptions env)
    eval_K := childSpec_of_rows "pidec.v1_2.eval_K_recomposition"
      (evalKCircuit shared) (evalKOffset offset)
      env _ rows (by simp [opsAt, shared]) (evalKAssumptions assumptions env)
    eval_A := childSpec_of_rows "pidec.v1_2.eval_A_recomposition"
      (evalACircuit shared) (evalAOffset offset)
      env _ rows (by simp [opsAt, shared]) (evalAAssumptions assumptions env)
    outputBinding := childSpec_of_rows "pidec.v1_2.output_binding"
      (outputBindingCircuit relation shared) (outputBindingOffset offset)
      env _ rows (by simp [opsAt, shared]) trivial }

@[simp] private theorem inputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.input_binding"
      (inputBindingCircuit relation interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact InputBinding.localLength_eq relation
    (inputBindingInterface interface) offset

@[simp] private theorem publicInputOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.public_input_split"
      (publicInputCircuit interface) offset).localLength =
        PublicInputSplit.exactPrivateCount := by
  rw [childOp, Sequence.childOp_localLength]
  change localLength (Circuit.ops
    (PublicInputSplit.circuit (publicInputInterface interface)).main offset) =
      PublicInputSplit.exactPrivateCount
  calc
    _ = (PublicInputSplit.circuit
          (publicInputInterface interface)).privateCount offset :=
      (PublicInputSplit.circuit
        (publicInputInterface interface)).privateCount_eq offset
    _ = PublicInputSplit.logicalPrivateCount logicalWidth publicFits := rfl
    _ = PublicInputSplit.exactPrivateCount :=
      PublicInputSplit.logicalPrivateCount_eq logicalWidth publicFits

@[simp] private theorem commitmentOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.commitment_recomposition"
      (commitmentCircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact CommitmentRecomposition.localLength_eq
    (commitmentInterface interface) offset

@[simp] private theorem evalKOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.eval_K_recomposition"
      (evalKCircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalKRecomposition.localLength_eq (evalKInterface interface) offset

@[simp] private theorem evalAOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.eval_A_recomposition"
      (evalACircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalARecomposition.localLength_eq (evalAInterface interface) offset

@[simp] private theorem outputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.output_binding"
      (outputBindingCircuit relation interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact OutputBinding.localLength_eq relation
    (outputBindingInterface interface) offset

@[simp] private theorem inputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.input_binding"
      (inputBindingCircuit relation interface) offset).rowCount = 0 := by
  rfl

@[simp] private theorem publicInputOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.public_input_split"
      (publicInputCircuit interface) offset).rowCount =
        PublicInputSplit.exactRowCount := by
  change PublicInputSplit.logicalRowCount logicalWidth publicFits =
    PublicInputSplit.exactRowCount
  exact PublicInputSplit.logicalRowCount_eq logicalWidth publicFits

@[simp] private theorem commitmentOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.commitment_recomposition"
      (commitmentCircuit interface) offset).rowCount =
        CommitmentRecomposition.coordinateCount := by
  rfl

@[simp] private theorem evalKOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.eval_K_recomposition"
      (evalKCircuit interface) offset).rowCount =
        RingKRecomposition.coordinateCount EvalKRecomposition.blockCount := by
  rfl

@[simp] private theorem evalAOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.eval_A_recomposition"
      (evalACircuit interface) offset).rowCount =
        RingKRecomposition.coordinateCount EvalARecomposition.blockCount := by
  rfl

@[simp] private theorem outputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_2.output_binding"
      (outputBindingCircuit relation interface) offset).rowCount = 0 := by
  rfl

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (main relation interface) offset) =
      logicalPrivateCount := by
  change localLength (opsAt relation interface offset) = logicalPrivateCount
  simp [opsAt, localLength, logicalPrivateCount]

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (Circuit.ops (main relation interface) offset)).length =
      logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt relation interface offset) = logicalRowCount
  simp [opsAt, rowCount, logicalRowCount, Nat.add_assoc]

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

section Exporter

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The words that the Π_DEC rows read, as columns that `allowed` holds on: the parent's and the
children's commitments and evaluations, the parent's public input and the digit words. -/
structure InputsSatisfy (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop) : Prop where
  parentCommitment : ∀ row lane,
    ((interface.parent offset).commitment row lane).VarsSatisfy allowed
  parentPublicInput : ∀ coordinate,
    ((interface.parent offset).publicInput coordinate).VarsSatisfy allowed
  parentEval_K : ∀ coefficient,
    ((interface.parent offset).evaluation.eval_K coefficient).c0.VarsSatisfy allowed ∧
      ((interface.parent offset).evaluation.eval_K coefficient).c1.VarsSatisfy allowed
  parentEval_A : ∀ matrix coefficient,
    ((interface.parent offset).evaluation.eval_A matrix coefficient).c0.VarsSatisfy allowed ∧
      ((interface.parent offset).evaluation.eval_A matrix coefficient).c1.VarsSatisfy allowed
  messageCommitment : ∀ child row lane,
    ((interface.message offset child).commitment row lane).VarsSatisfy allowed
  messageEval_K : ∀ child coefficient,
    ((interface.message offset child).evaluation.eval_K coefficient).c0.VarsSatisfy allowed ∧
      ((interface.message offset child).evaluation.eval_K coefficient).c1.VarsSatisfy allowed
  messageEval_A : ∀ child matrix coefficient,
    ((interface.message offset child).evaluation.eval_A matrix coefficient).c0.VarsSatisfy
        allowed ∧
      ((interface.message offset child).evaluation.eval_A matrix coefficient).c1.VarsSatisfy
        allowed
  digit : ∀ child coordinate, (interface.digit offset child coordinate).VarsSatisfy allowed

private theorem marks_assertions (rows : List Expr) : marks (rows.map Op.assertZero) = [] := by
  unfold marks
  rw [List.flatMap_eq_nil_iff]
  intro op member
  obtain ⟨_, _, rfl⟩ := List.mem_map.mp member
  rfl

private theorem publicInputSplit_marks_nil (split : PublicInputSplit.Interface logicalWidth publicFits)
    (offset : Nat) :
    marks (Circuit.ops (PublicInputSplit.circuit split).main offset) = [] := by
  show marks (PublicInputSplit.opsPrefix split offset
    (PublicInputSplit.coordinateCount logicalWidth publicFits)) = []
  unfold marks PublicInputSplit.opsPrefix
  rw [List.flatMap_eq_nil_iff]
  intro op member
  obtain ⟨source, _, rfl⟩ := List.mem_map.mp member
  unfold PublicInputSplit.childOp
  split
  · rename_i sourceLt
    rw [Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks]
    show marks (SignedSplitScalar.operations
      (PublicInputSplit.childInterface split offset source sourceLt)
      (PublicInputSplit.sourceOffset offset source)) = []
    rw [SignedSplitScalar.operations, marks_cons, marks_assertions]
    rfl
  · rfl

theorem marks_nil (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    marks (Circuit.ops (main relation interface) offset) = [] := by
  rw [main_ops]
  unfold marks
  rw [List.flatMap_eq_nil_iff]
  intro op member
  simp only [opsAt, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  · rfl
  · rw [childOp, Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks]
    exact publicInputSplit_marks_nil _ _
  · rw [childOp, Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks, commitmentCircuit,
      RadixRecomposition.circuit_ops, RadixRecomposition.operations, marks_assertions]
  · rw [childOp, Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks, evalKCircuit,
      RadixRecomposition.circuit_ops, RadixRecomposition.operations, marks_assertions]
  · rw [childOp, Sequence.childOp, Op.marks, FormalCircuit.asSubcircuit_marks, evalACircuit,
      RadixRecomposition.circuit_ops, RadixRecomposition.operations, marks_assertions]
  · rfl

/-- The rows read the parent, the children's messages, the digit words and the public-input
split's locals. -/
theorem free (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) {live : Nat → Prop}
    (inputs : InputsSatisfy interface offset live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (main relation interface) offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (main relation interface) offset),
      row.VarsSatisfy live := by
  rw [localLength_eq] at locals
  rw [main_ops]
  intro row member
  rcases List.mem_flatMap.mp member with ⟨op, opMember, rowMember⟩
  simp only [opsAt, List.mem_cons, List.not_mem_nil, or_false] at opMember
  rcases opMember with rfl | rfl | rfl | rfl | rfl | rfl <;>
    rw [childOp, Sequence.childOp, Op.flatConstraints, FormalCircuit.asSubcircuit_constraints]
      at rowMember
  · cases rowMember
  · refine PublicInputSplit.flatConstraints_varsSatisfy live
      (publicInputInterface (atOffset interface offset)) offset
      (fun coordinate => inputs.parentPublicInput coordinate)
      (fun child coordinate => inputs.digit child coordinate)
      (fun column lower upper => locals column lower ?_) row rowMember
    rw [PublicInputSplit.logicalPrivateCount_eq] at upper
    exact upper
  · rw [commitmentCircuit, RadixRecomposition.circuit_ops] at rowMember
    exact RadixRecomposition.flatConstraints_varsSatisfy live
      (CommitmentRecomposition.scalarInterface (commitmentInterface (atOffset interface offset)))
      (commitmentOffset offset)
      (fun coordinate => inputs.parentCommitment
        (CommitmentRecomposition.coordinates coordinate).1
        (CommitmentRecomposition.coordinates coordinate).2)
      (fun child coordinate => inputs.messageCommitment child
        (CommitmentRecomposition.coordinates coordinate).1
        (CommitmentRecomposition.coordinates coordinate).2) row rowMember
  · exact RingKRecomposition.flatConstraints_varsSatisfy live
      (EvalKRecomposition.ringInterface (evalKInterface (atOffset interface offset)))
      (evalKOffset offset)
      (fun _ lane => inputs.parentEval_K (EvalKRecomposition.coefficient lane))
      (fun child _ lane => inputs.messageEval_K child (EvalKRecomposition.coefficient lane))
      row rowMember
  · exact RingKRecomposition.flatConstraints_varsSatisfy live
      (EvalARecomposition.ringInterface (evalAInterface (atOffset interface offset)))
      (evalAOffset offset)
      (fun matrix lane => inputs.parentEval_A matrix (EvalKRecomposition.coefficient lane))
      (fun child matrix lane =>
        inputs.messageEval_A child matrix (EvalKRecomposition.coefficient lane))
      row rowMember
  · cases rowMember

end Exporter

end NightstreamFPrime.Lifecycle.PiDEC.v1_2.Formal
