import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain

/-!
Owns the exact logical PiRLC v1.1 parent assembler.

Child order:
1. zero-row input binding;
2. 17 transcript-chained strong-set samplers;
3. commitment combination;
4. public-input combination;
5. separate `Eval_K` combination;
6. separate 14-matrix `Eval_A` combination;
7. zero-row computed-output binding.

The parent owns only shared-value wiring, offsets, operation order, and child
coverage. It adds no witness, copy, transcript, or assertion row.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  baseOffset : Nat := 0
  initialState : Nat → SamplerChain.EState
  point : Nat → Fin productionShape.cubeVariables → KExpr
  input : Nat → Fin productionShape.sourceCount →
    InputBinding.InputExpr logicalWidth publicFits

def atOffset
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    Interface logicalWidth publicFits where
  baseOffset := offset
  initialState := fun _ => interface.initialState offset
  point := fun _ => interface.point offset
  input := fun _ => interface.input offset

def inputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    InputBinding.Interface logicalWidth publicFits where
  point := interface.point
  input := interface.input

def samplerInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : SamplerChain.Interface where
  initialState := interface.initialState

def commitmentInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    CommitmentCombination.Interface where
  challenge := fun _ source =>
    SamplerChain.challengeExpr (samplerInterface interface)
      interface.baseOffset source
  input := fun offset source => (interface.input offset source).commitment

def publicInputInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) :
    PublicInputCombination.Interface logicalWidth publicFits where
  challenge := fun _ source =>
    SamplerChain.challengeExpr (samplerInterface interface)
      interface.baseOffset source
  input := fun offset source => (interface.input offset source).publicInput

def evalKInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : EvalKCombination.Interface where
  challenge := fun _ source =>
    SamplerChain.challengeExpr (samplerInterface interface)
      interface.baseOffset source
  input := fun offset source => (interface.input offset source).evaluation.eval_K

def evalAInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : EvalACombination.Interface where
  challenge := fun _ source =>
    SamplerChain.challengeExpr (samplerInterface interface)
      interface.baseOffset source
  input := fun offset source => (interface.input offset source).evaluation.eval_A

def inputBindingOffset (offset : Nat) : Nat := offset
def samplerOffset (offset : Nat) : Nat := offset

def commitmentOffset (offset : Nat) : Nat :=
  samplerOffset offset + SamplerChain.logicalPrivateCount

def publicInputOffset (offset : Nat) : Nat :=
  commitmentOffset offset + 16524

def evalKOffset (offset : Nat) : Nat :=
  publicInputOffset offset + 4590

def evalAOffset (offset : Nat) : Nat :=
  evalKOffset offset + 1836

def outputBindingOffset (offset : Nat) : Nat :=
  evalAOffset offset + 25704

def finalOffset (offset : Nat) : Nat := outputBindingOffset offset

def outputBindingInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (phaseOffset : Nat) :
    OutputBinding.Interface logicalWidth publicFits where
  point := interface.point
  commitment := fun _ => CommitmentCombination.output
    (commitmentInterface interface) (commitmentOffset phaseOffset)
  publicInput := fun _ => PublicInputCombination.output
    (publicInputInterface interface) (publicInputOffset phaseOffset)
  eval_K := fun _ => EvalKCombination.output
    (evalKInterface interface) (evalKOffset phaseOffset)
  eval_A := fun _ => EvalACombination.output
    (evalAInterface interface) (evalAOffset phaseOffset)

def inputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  InputBinding.circuit relation (inputBindingInterface interface)

def samplerCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  SamplerChain.circuit (samplerInterface interface)

def commitmentCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  CommitmentCombination.circuit (commitmentInterface interface)

def publicInputCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  PublicInputCombination.circuit (publicInputInterface interface)

def evalKCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalKCombination.circuit (evalKInterface interface)

def evalACircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) : FormalCircuit :=
  EvalACombination.circuit (evalAInterface interface)

def outputBindingCircuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (phaseOffset : Nat) :
    FormalCircuit :=
  OutputBinding.circuit relation (outputBindingInterface interface phaseOffset)

def childOp (name : String) (child : FormalCircuit) (offset : Nat) : Op :=
  Sequence.childOp name child offset

theorem childOp_privateCount (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).localLength = child.privateCount offset := by
  rfl

def opsAt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "pirlc.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset),
    childOp "pirlc.v1_1.sampler_chain"
      (samplerCircuit shared) (samplerOffset offset),
    childOp "pirlc.v1_1.commitment_combination"
      (commitmentCircuit shared) (commitmentOffset offset),
    childOp "pirlc.v1_1.public_input_combination"
      (publicInputCircuit shared) (publicInputOffset offset),
    childOp "pirlc.v1_1.eval_K_combination"
      (evalKCircuit shared) (evalKOffset offset),
    childOp "pirlc.v1_1.eval_A_combination"
      (evalACircuit shared) (evalAOffset offset),
    childOp "pirlc.v1_1.output_binding"
      (outputBindingCircuit relation shared offset) (outputBindingOffset offset)]

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

def logicalPrivateCount : Nat := 312222
def logicalRowCount : Nat := 313871

structure Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop where
  sampler : SamplerChain.Assumptions (samplerInterface (atOffset interface offset))
    (samplerOffset offset) env
  commitment : CommitmentCombination.Assumptions
    (commitmentInterface (atOffset interface offset)) (commitmentOffset offset) env
  publicInput : PublicInputCombination.Assumptions
    (publicInputInterface (atOffset interface offset)) (publicInputOffset offset) env
  eval_K : EvalKCombination.Assumptions
    (evalKInterface (atOffset interface offset)) (evalKOffset offset) env
  eval_A : EvalACombination.Assumptions
    (evalAInterface (atOffset interface offset)) (evalAOffset offset) env

structure SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop where
  inputBinding : InputBinding.SpecHolds relation
    (inputBindingInterface (atOffset interface offset)) (inputBindingOffset offset) env
  sampler : SamplerChain.RelationHolds
    (samplerInterface (atOffset interface offset)) (samplerOffset offset) env
  commitment : CommitmentCombination.SpecHolds
    (commitmentInterface (atOffset interface offset)) (commitmentOffset offset) env
  publicInput : PublicInputCombination.SpecHolds
    (publicInputInterface (atOffset interface offset)) (publicInputOffset offset) env
  eval_K : EvalKCombination.SpecHolds
    (evalKInterface (atOffset interface offset)) (evalKOffset offset) env
  eval_A : EvalACombination.SpecHolds
    (evalAInterface (atOffset interface offset)) (evalAOffset offset) env
  outputBinding : OutputBinding.SpecHolds relation
    (outputBindingInterface (atOffset interface offset) offset)
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
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (assumptions : Assumptions relation interface offset env)
    (rows : holds env (Circuit.ops (main relation interface) offset)) :
    SpecHolds relation interface offset env := by
  let shared := atOffset interface offset
  change holds env (opsAt relation interface offset) at rows
  refine {
    inputBinding := childSpec_of_rows "pirlc.v1_1.input_binding"
      (inputBindingCircuit relation shared) (inputBindingOffset offset) env _ rows
      (by simp [opsAt, shared]) trivial
    sampler := childSpec_of_rows "pirlc.v1_1.sampler_chain"
      (samplerCircuit shared) (samplerOffset offset) env _ rows
      (by simp [opsAt, shared]) assumptions.sampler
    commitment := childSpec_of_rows "pirlc.v1_1.commitment_combination"
      (commitmentCircuit shared) (commitmentOffset offset) env _ rows
      (by simp [opsAt, shared]) assumptions.commitment
    publicInput := childSpec_of_rows "pirlc.v1_1.public_input_combination"
      (publicInputCircuit shared) (publicInputOffset offset) env _ rows
      (by simp [opsAt, shared]) assumptions.publicInput
    eval_K := childSpec_of_rows "pirlc.v1_1.eval_K_combination"
      (evalKCircuit shared) (evalKOffset offset) env _ rows
      (by simp [opsAt, shared]) assumptions.eval_K
    eval_A := childSpec_of_rows "pirlc.v1_1.eval_A_combination"
      (evalACircuit shared) (evalAOffset offset) env _ rows
      (by simp [opsAt, shared]) assumptions.eval_A
    outputBinding := childSpec_of_rows "pirlc.v1_1.output_binding"
      (outputBindingCircuit relation shared offset) (outputBindingOffset offset)
      env _ rows (by simp [opsAt, shared]) trivial }

@[simp] private theorem inputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.input_binding" (inputBindingCircuit relation interface)
      offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact InputBinding.localLength_eq relation (inputBindingInterface interface) offset

@[simp] private theorem samplerOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.sampler_chain" (samplerCircuit interface) offset).localLength =
      SamplerChain.logicalPrivateCount := by
  rw [childOp, Sequence.childOp_localLength]
  exact SamplerChain.localLength_eq (samplerInterface interface) offset

@[simp] private theorem commitmentOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.commitment_combination" (commitmentCircuit interface)
      offset).localLength = 16524 := by
  rw [childOp, Sequence.childOp_localLength]
  exact CommitmentCombination.localLength_eq (commitmentInterface interface) offset

@[simp] private theorem publicInputOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.public_input_combination" (publicInputCircuit interface)
      offset).localLength = 4590 := by
  rw [childOp, Sequence.childOp_localLength]
  exact PublicInputCombination.localLength_eq (publicInputInterface interface) offset

@[simp] private theorem evalKOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.eval_K_combination" (evalKCircuit interface)
      offset).localLength = 1836 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalKCombination.localLength_eq (evalKInterface interface) offset

@[simp] private theorem evalAOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.eval_A_combination" (evalACircuit interface)
      offset).localLength = 25704 := by
  rw [childOp, Sequence.childOp_localLength]
  exact EvalACombination.localLength_eq (evalAInterface interface) offset

@[simp] private theorem outputBindingOp_localLength
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (phaseOffset offset : Nat) :
    (childOp "pirlc.v1_1.output_binding"
      (outputBindingCircuit relation interface phaseOffset) offset).localLength = 0 := by
  rw [childOp, Sequence.childOp_localLength]
  exact OutputBinding.localLength_eq relation
    (outputBindingInterface interface phaseOffset) offset

@[simp] private theorem inputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.input_binding" (inputBindingCircuit relation interface)
      offset).rowCount = 0 := by rfl

@[simp] private theorem samplerOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.sampler_chain" (samplerCircuit interface) offset).rowCount =
      SamplerChain.logicalRowCount := by rfl

@[simp] private theorem commitmentOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.commitment_combination" (commitmentCircuit interface)
      offset).rowCount = 16524 := by
  change CombinationFamily.logicalRowCount CommitmentCombination.blockCount
    CommitmentCombination.cellCount = 16524
  exact CommitmentCombination.logicalRowCount_eq

@[simp] private theorem publicInputOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.public_input_combination" (publicInputCircuit interface)
      offset).rowCount = 4590 := by
  change CombinationFamily.logicalRowCount PublicInputCombination.blockCount
    PublicInputCombination.cellCount = 4590
  exact PublicInputCombination.logicalRowCount_eq

@[simp] private theorem evalKOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.eval_K_combination" (evalKCircuit interface)
      offset).rowCount = 1836 := by
  change CombinationFamily.logicalRowCount EvalKCombination.blockCount
    RingKCombination.cellCount = 1836
  exact EvalKCombination.logicalRowCount_eq

@[simp] private theorem evalAOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (childOp "pirlc.v1_1.eval_A_combination" (evalACircuit interface)
      offset).rowCount = 25704 := by
  change CombinationFamily.logicalRowCount EvalACombination.blockCount
    RingKCombination.cellCount = 25704
  exact EvalACombination.logicalRowCount_eq

@[simp] private theorem outputBindingOp_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (phaseOffset offset : Nat) :
    (childOp "pirlc.v1_1.output_binding"
      (outputBindingCircuit relation interface phaseOffset) offset).rowCount = 0 := by
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
  simp [opsAt, localLength, logicalPrivateCount,
    SamplerChain.logicalPrivateCount_eq]

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
  simp [opsAt, rowCount, logicalRowCount, SamplerChain.logicalRowCount_eq]

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal
