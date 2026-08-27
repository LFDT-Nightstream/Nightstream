import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

/-!
Owns the exact logical PiDEC v1.1 parent assembler.

Child order:
1. zero-row operational input binding;
2. strict parent bound and canonical 16-child public split;
3. commitment recomposition;
4. separate Pad `Eval_K` recomposition;
5. separate 14-matrix `Eval_A` recomposition;
6. zero-row computed-output binding.

The parent owns only shared-value wiring, offsets, operation order, and child
coverage. It adds no witness, copy, or assertion row.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  parent : Nat → InputBinding.ParentExpr logicalWidth publicFits
  point : Nat → Fin productionShape.cubeVariables → KExpr
  message : Nat → Radix.ChildIndex → InputBinding.ChildMessageExpr
  digit : Nat → Radix.ChildIndex →
    Fin (PublicInputSplit.coordinateCount logicalWidth publicFits) → Expr

def atOffset
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Interface logicalWidth publicFits where
  parent := fun _ => interface.parent offset
  point := fun _ => interface.point offset
  message := fun _ => interface.message offset
  digit := fun _ => interface.digit offset

def inputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    InputBinding.Interface logicalWidth publicFits where
  parent := interface.parent
  point := interface.point
  message := interface.message

def publicInputInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    PublicInputSplit.Interface logicalWidth publicFits where
  parent := fun offset => (interface.parent offset).publicInput
  digit := interface.digit

def commitmentInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    CommitmentRecomposition.Interface where
  parent := fun offset => (interface.parent offset).commitment
  child := fun offset child => (interface.message offset child).commitment

def evalKInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    EvalKRecomposition.Interface where
  parent := fun offset => (interface.parent offset).evaluation.eval_K
  child := fun offset child =>
    (interface.message offset child).evaluation.eval_K

def evalAInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    EvalARecomposition.Interface where
  parent := fun offset => (interface.parent offset).evaluation.eval_A
  child := fun offset child =>
    (interface.message offset child).evaluation.eval_A

def outputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    OutputBinding.Interface logicalWidth publicFits where
  point := interface.point
  message := interface.message
  publicInput := interface.digit

def inputBindingOffset (offset : Nat) : Nat := offset
def publicInputOffset (offset : Nat) : Nat := offset

/-- The split child is the only PiDEC child that allocates private cells. -/
def recompositionOffset (offset : Nat) : Nat := offset + 54
def commitmentOffset (offset : Nat) : Nat := recompositionOffset offset
def evalKOffset (offset : Nat) : Nat := recompositionOffset offset
def evalAOffset (offset : Nat) : Nat := recompositionOffset offset
def outputBindingOffset (offset : Nat) : Nat := recompositionOffset offset
def finalOffset (offset : Nat) : Nat := recompositionOffset offset

def inputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  InputBinding.circuit relation (inputBindingInterface interface)

def publicInputCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  PublicInputSplit.circuit (publicInputInterface interface)

def commitmentCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  CommitmentRecomposition.circuit (commitmentInterface interface)

def evalKCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalKRecomposition.circuit (evalKInterface interface)

def evalACircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalARecomposition.circuit (evalAInterface interface)

def outputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  OutputBinding.circuit relation (outputBindingInterface interface)

def childOp (name : String) (child : FormalCircuit) (offset : Nat) : Op :=
  Sequence.childOp name child offset

def opsAt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "pidec.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset),
    childOp "pidec.v1_1.public_input_split"
      (publicInputCircuit shared) (publicInputOffset offset),
    childOp "pidec.v1_1.commitment_recomposition"
      (commitmentCircuit shared) (commitmentOffset offset),
    childOp "pidec.v1_1.eval_K_recomposition"
      (evalKCircuit shared) (evalKOffset offset),
    childOp "pidec.v1_1.eval_A_recomposition"
      (evalACircuit shared) (evalAOffset offset),
    childOp "pidec.v1_1.output_binding"
      (outputBindingCircuit relation shared) (outputBindingOffset offset)]

def main
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : Circuit Unit := fun offset =>
  ((), finalOffset offset, opsAt relation interface offset)

@[simp] theorem main_ops
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Circuit.ops (main relation interface) offset =
      opsAt relation interface offset := by
  rfl

def logicalPrivateCount : Nat := 54
def logicalRowCount : Nat := 3564

structure InputsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (_env : Env) : Prop where
  inputs : InputsBelow interface offset

def publicInputAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Assumptions relation interface offset env)
    (current : Env) : PublicInputSplit.Assumptions
      (publicInputInterface (atOffset interface offset))
      (publicInputOffset offset) current where
  parentBelow := assumptions.inputs.parentPublicInput
  digitBelow := assumptions.inputs.digit

def commitmentAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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

def evalKAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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

def evalAAssumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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
      Phi81CarrierLayout.carrierWidth logicalWidth}
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
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset)) :
    SpecHolds relation interface offset env := by
  let shared := atOffset interface offset
  change holds env (opsAt relation interface offset) at rows
  refine {
    inputBinding := childSpec_of_rows "pidec.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset)
      env _ rows (by simp [opsAt, shared]) trivial
    publicInput := childSpec_of_rows "pidec.v1_1.public_input_split"
      (publicInputCircuit shared) (publicInputOffset offset)
      env _ rows (by simp [opsAt, shared])
      (publicInputAssumptions assumptions env)
    commitment := childSpec_of_rows "pidec.v1_1.commitment_recomposition"
      (commitmentCircuit shared) (commitmentOffset offset)
      env _ rows (by simp [opsAt, shared])
      (commitmentAssumptions assumptions env)
    eval_K := childSpec_of_rows "pidec.v1_1.eval_K_recomposition"
      (evalKCircuit shared) (evalKOffset offset)
      env _ rows (by simp [opsAt, shared]) (evalKAssumptions assumptions env)
    eval_A := childSpec_of_rows "pidec.v1_1.eval_A_recomposition"
      (evalACircuit shared) (evalAOffset offset)
      env _ rows (by simp [opsAt, shared]) (evalAAssumptions assumptions env)
    outputBinding := childSpec_of_rows "pidec.v1_1.output_binding"
      (outputBindingCircuit relation shared) (outputBindingOffset offset)
      env _ rows (by simp [opsAt, shared]) trivial }

@[simp] private theorem inputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.input_binding"
      (inputBindingCircuit relation interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact InputBinding.localLength_eq relation
    (inputBindingInterface interface) offset

@[simp] private theorem publicInputOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.public_input_split"
      (publicInputCircuit interface) offset).localLength = 54 := by
  rw [childOp, Sequence.childOp_localLength]
  change localLength (Circuit.ops
    (PublicInputSplit.circuit (publicInputInterface interface)).main offset) = 54
  calc
    _ = (PublicInputSplit.circuit
          (publicInputInterface interface)).privateCount offset :=
      (PublicInputSplit.circuit
        (publicInputInterface interface)).privateCount_eq offset
    _ = PublicInputSplit.logicalPrivateCount logicalWidth publicFits := rfl
    _ = 54 := PublicInputSplit.logicalPrivateCount_eq logicalWidth publicFits

@[simp] private theorem commitmentOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.commitment_recomposition"
      (commitmentCircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact CommitmentRecomposition.localLength_eq
    (commitmentInterface interface) offset

@[simp] private theorem evalKOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.eval_K_recomposition"
      (evalKCircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalKRecomposition.localLength_eq (evalKInterface interface) offset

@[simp] private theorem evalAOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.eval_A_recomposition"
      (evalACircuit interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalARecomposition.localLength_eq (evalAInterface interface) offset

@[simp] private theorem outputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.output_binding"
      (outputBindingCircuit relation interface) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact OutputBinding.localLength_eq relation
    (outputBindingInterface interface) offset

@[simp] private theorem inputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.input_binding"
      (inputBindingCircuit relation interface) offset).rowCount = 0 := by
  rfl

@[simp] private theorem publicInputOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.public_input_split"
      (publicInputCircuit interface) offset).rowCount = 972 := by
  change PublicInputSplit.logicalRowCount logicalWidth publicFits = 972
  exact PublicInputSplit.logicalRowCount_eq logicalWidth publicFits

@[simp] private theorem commitmentOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.commitment_recomposition"
      (commitmentCircuit interface) offset).rowCount = 972 := by
  change CommitmentRecomposition.coordinateCount = 972
  exact CommitmentRecomposition.coordinateCount_eq

@[simp] private theorem evalKOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.eval_K_recomposition"
      (evalKCircuit interface) offset).rowCount = 108 := by
  change RingKRecomposition.coordinateCount EvalKRecomposition.blockCount = 108
  exact EvalKRecomposition.coordinateCount_eq

@[simp] private theorem evalAOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.eval_A_recomposition"
      (evalACircuit interface) offset).rowCount = 1512 := by
  change RingKRecomposition.coordinateCount EvalARecomposition.blockCount = 1512
  exact EvalARecomposition.coordinateCount_eq

@[simp] private theorem outputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pidec.v1_1.output_binding"
      (outputBindingCircuit relation interface) offset).rowCount = 0 := by
  rfl

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (main relation interface) offset) =
      logicalPrivateCount := by
  change localLength (opsAt relation interface offset) = logicalPrivateCount
  simp [opsAt, localLength, logicalPrivateCount]

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints (Circuit.ops (main relation interface) offset)).length =
      logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  change rowCount (opsAt relation interface offset) = logicalRowCount
  simp [opsAt, rowCount, logicalRowCount]

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal
