import NightstreamFPrime.Lifecycle.BoundedDepth.LightFold
import NightstreamFPrime.Lifecycle.BoundedDepth.LightPiCCSLayout
import NightstreamFPrime.Lifecycle.BoundedDepth.LightCommitment

/-! The light Π_RLC layout (CW-2.5d, d3). One sampler draws `ρ` from the transcript state after
the light Π_CCS output binding, and the commitment, public-input and evaluation combinations
follow it from a parent offset. The running inputs are the slot claim's words and the Π_CCS
output's running family; the fresh inputs are the fresh instance's words and the output's fresh
family. `soundness` gives `LightFold.RlcSpecHolds` from the rows, and `wired` gives
`LightFold.RlcWired` from the external words alone. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightRlcLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-- A source's output family as ring blocks: block 0 is the pad family and block `m + 1` is
matrix `m` (`LightEvaluation.familyBlocks`). -/
def familyExpr (output : OutputBinding.ShapeOutputExpr LightShape)
    (source : Fin LightShape.sourceCount) (block : Fin LightEvaluation.blockCount)
    (lane : Fin ringDegree) : KExpr :=
  if first : block.val = 0 then output.padCoordinate source lane
  else output.matrixCoordinate source ⟨block.val - 1, by
    have bound := block.isLt
    simp only [LightEvaluation.blockCount] at bound
    omega⟩ lane

/-- The external words of the light Π_RLC: the slot claim's commitment and public input, and the
fresh instance. -/
structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  runningCommitment : Nat → Fin LightCommitment.blockCount → Fin ringDegree → Expr
  runningPublicInput : Nat → Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  fresh : Nat → StatementAbsorption.FreshExpr logicalWidth publicFits

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
  (interface : Interface logicalWidth publicFits) (offset : Nat)

/-- The sampler starts from the transcript state after the Π_CCS output binding. -/
def samplerInterface : Sampler.Interface where
  initialState := fun _ => OutputBinding.finalState piccs.output piccs.outputOffset

def samplerCircuit : FormalCircuit :=
  Sampler.circuit (samplerInterface piccs) 0

def commitmentOffset : Nat :=
  Formal.nextOffset (samplerCircuit piccs) offset

def commitmentInterface : LightCommitment.Interface where
  challenge := fun _ => Sampler.outputChallenge offset
  running := fun _ => interface.runningCommitment offset
  fresh := fun _ => (interface.fresh offset).commitment Formal.freshIndex

def commitmentCircuit : FormalCircuit :=
  CombinationStep.circuit (LightCommitment.stepInterface (commitmentInterface interface offset))

def publicInputOffset : Nat :=
  Formal.nextOffset (commitmentCircuit interface offset) (commitmentOffset piccs offset)

def publicInputInterface : LightPublicInput.Interface logicalWidth publicFits where
  challenge := fun _ => Sampler.outputChallenge offset
  running := fun _ => interface.runningPublicInput offset
  fresh := fun _ => (interface.fresh offset).publicInput Formal.freshIndex

def publicInputCircuit : FormalCircuit :=
  CombinationStep.circuit
    (LightPublicInput.stepInterface (publicInputInterface interface offset))

def evaluationOffset : Nat :=
  Formal.nextOffset (publicInputCircuit interface offset) (publicInputOffset piccs interface offset)

def evaluationInterface : LightEvaluation.Interface where
  challenge := fun _ => Sampler.outputChallenge offset
  running := fun _ => familyExpr (piccs.output.output piccs.outputOffset) LightFold.runningSource
  fresh := fun _ => familyExpr (piccs.output.output piccs.outputOffset) LightFold.freshSource

def evaluationCircuit : FormalCircuit :=
  CombinationStep.circuit (LightEvaluation.stepInterface (evaluationInterface piccs offset))

def finalOffset : Nat :=
  Formal.nextOffset (evaluationCircuit piccs offset) (evaluationOffset piccs interface offset)

/-- The Π_RLC leaves that this layout places. -/
def leaves : LightFold.RlcLeaves logicalWidth publicFits where
  sampler := samplerInterface piccs
  samplerOffset := offset
  commitment := commitmentInterface interface offset
  commitmentOffset := commitmentOffset piccs offset
  publicInput := publicInputInterface interface offset
  publicInputOffset := publicInputOffset piccs interface offset
  evaluation := evaluationInterface piccs offset
  evaluationOffset := evaluationOffset piccs interface offset

def opsAt : List Op :=
  [Formal.childOp "light.pirlc.sampler" (samplerCircuit piccs) offset,
    Formal.childOp "light.pirlc.commitment" (commitmentCircuit interface offset)
      (commitmentOffset piccs offset),
    Formal.childOp "light.pirlc.public_input" (publicInputCircuit interface offset)
      (publicInputOffset piccs interface offset),
    Formal.childOp "light.pirlc.evaluation" (evaluationCircuit piccs offset)
      (evaluationOffset piccs interface offset)]

def main : Circuit Unit :=
  fun offset => ((), finalOffset piccs interface offset, opsAt piccs interface offset)

/-- Each child's input scoping, which its rows need to give its spec. -/
structure Assumptions (env : Env) : Prop where
  sampler : (samplerCircuit piccs).assumptions offset env
  commitment : (commitmentCircuit interface offset).assumptions (commitmentOffset piccs offset) env
  publicInput : (publicInputCircuit interface offset).assumptions
    (publicInputOffset piccs interface offset) env
  evaluation : (evaluationCircuit piccs offset).assumptions
    (evaluationOffset piccs interface offset) env

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (ops : List Op) (rows : holds env ops)
    (member : Formal.childOp name child childOffset ∈ ops)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  have callHolds := rows (Formal.childOp name child childOffset) member
  change child.assumptions childOffset env → child.spec childOffset env at callHolds
  exact callHolds assumptions

/-- The rows give each Π_RLC leaf's spec. -/
theorem soundness (env : Env) (assumptions : Assumptions piccs interface offset env)
    (rows : holds env (Circuit.ops (main piccs interface) offset)) :
    LightFold.RlcSpecHolds (leaves piccs interface offset) env := by
  change holds env (opsAt piccs interface offset) at rows
  have samplerSpec := childSpec_of_rows "light.pirlc.sampler" (samplerCircuit piccs) offset env
    _ rows (by simp [opsAt]) assumptions.sampler
  have commitmentSpec := childSpec_of_rows "light.pirlc.commitment"
    (commitmentCircuit interface offset) (commitmentOffset piccs offset) env _ rows
    (by simp [opsAt]) assumptions.commitment
  have publicInputSpec := childSpec_of_rows "light.pirlc.public_input"
    (publicInputCircuit interface offset) (publicInputOffset piccs interface offset) env _ rows
    (by simp [opsAt]) assumptions.publicInput
  have evaluationSpec := childSpec_of_rows "light.pirlc.evaluation"
    (evaluationCircuit piccs offset) (evaluationOffset piccs interface offset) env _ rows
    (by simp [opsAt]) assumptions.evaluation
  exact ⟨samplerSpec, commitmentSpec, publicInputSpec, evaluationSpec⟩

/-- The external words carry the slot claim, the fresh instance and the Π_CCS output. -/
structure Represents (slot : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (output : FullOutputCoordinates.FullOutput K LightShape) (env : Env) : Prop where
  runningCommitment : (fun row lane => (interface.runningCommitment offset row lane).eval env) =
    slot.commitment
  runningPublicInput : (fun column => (interface.runningPublicInput offset column).eval env) =
    slot.publicInput
  freshWords : StatementAbsorption.evalFresh (interface.fresh offset) env = fresh
  outputPad : ∀ source coefficient,
    ((piccs.output.output piccs.outputOffset).padCoordinate source coefficient).eval env =
      output.padCoordinate source coefficient
  outputMatrix : ∀ source matrix coefficient,
    ((piccs.output.output piccs.outputOffset).matrixCoordinate source matrix coefficient).eval
      env = output.matrixCoordinate source matrix coefficient

private theorem family_eq (slot : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (output : FullOutputCoordinates.FullOutput K LightShape) (env : Env)
    (represents : Represents piccs interface offset slot fresh output env)
    (source : Fin LightShape.sourceCount) :
    (fun block lane =>
        (familyExpr (piccs.output.output piccs.outputOffset) source block lane).eval env) =
      LightEvaluation.familyBlocks (LightFold.sourceFamily output source) := by
  funext block lane
  unfold familyExpr LightEvaluation.familyBlocks LightFold.sourceFamily
  by_cases first : block.val = 0 <;> simp only [first, ↓reduceDIte]
  · exact represents.outputPad source lane
  · exact represents.outputMatrix source _ lane

/-- The layout's wires: the external words alone give `LightFold.RlcWired`. -/
theorem wired (slot : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (output : FullOutputCoordinates.FullOutput K LightShape) (env : Env)
    (represents : Represents piccs interface offset slot fresh output env) :
    LightFold.RlcWired piccs (leaves piccs interface offset) slot fresh output env where
  samplerStart := by simp only [leaves, samplerInterface, Sampler.evalState]
  commitmentChallenge := rfl
  commitmentRunning := represents.runningCommitment
  commitmentFresh := by
    rw [← represents.freshWords]
    rfl
  publicInputChallenge := rfl
  publicInputRunning := represents.runningPublicInput
  publicInputFresh := by
    rw [← represents.freshWords]
    rfl
  evaluationChallenge := rfl
  evaluationRunning := family_eq piccs interface offset slot fresh output env represents _
  evaluationFresh := family_eq piccs interface offset slot fresh output env represents _

/-- The light fold from rows: the light Π_CCS layout and the light Π_RLC layout after it, with
their external words, make the bounded-depth light fold accept the slot's claim (the fold term of the
light case of `stepHolds_iff`) for the relation of any circuit key. -/
theorem foldHolds (relation target : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (slot : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (piccsInterface : LightPiCCSLayout.Interface logicalWidth 8 publicFits) (piccsOffset : Nat)
    (rlcInterface : Interface logicalWidth publicFits) (rlcOffset : Nat) (env : Env)
    (piccsAssumptions : LightPiCCSLayout.Assumptions relation piccsInterface piccsOffset env)
    (piccsRows : holds env
      (Circuit.ops (LightPiCCSLayout.main relation piccsInterface) piccsOffset))
    (piccsRepresents : LightPiCCSLayout.Represents (LightFold.lightRunning slot)
      (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof piccsInterface piccsOffset
      env)
    (rlcAssumptions : Assumptions (LightPiCCSLayout.leaves relation piccsInterface piccsOffset)
      rlcInterface rlcOffset env)
    (rlcRows : holds env (Circuit.ops
      (main (LightPiCCSLayout.leaves relation piccsInterface piccsOffset) rlcInterface)
      rlcOffset))
    (rlcRepresents : Represents (LightPiCCSLayout.leaves relation piccsInterface piccsOffset)
      rlcInterface rlcOffset slot fresh proof.piCcsOutput env) :
    HyperNova.Construction2.BoundedDepthFolds.folds .light (keys target ajtai)
        (toClaim (refreshKey target ajtai).relationSource slot) fresh (.light proof) =
      some (toClaim (refreshKey target ajtai).relationSource
        (LightFold.nextSlot (LightPiCCSLayout.leaves relation piccsInterface piccsOffset)
          (leaves (LightPiCCSLayout.leaves relation piccsInterface piccsOffset) rlcInterface
            rlcOffset) env)) :=
  LightFold.spec_implies_fold target ajtai slot fresh proof
    (LightPiCCSLayout.leaves relation piccsInterface piccsOffset)
    (leaves (LightPiCCSLayout.leaves relation piccsInterface piccsOffset) rlcInterface rlcOffset)
    env
    (LightPiCCSLayout.soundness relation piccsOffset piccsInterface env piccsAssumptions piccsRows
      target)
    (LightPiCCSLayout.wired relation piccsOffset target ajtai (LightFold.lightRunning slot)
      (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof piccsInterface env
      piccsRepresents)
    (soundness _ rlcInterface rlcOffset env rlcAssumptions rlcRows)
    (wired _ rlcInterface rlcOffset slot fresh proof.piCcsOutput env rlcRepresents)

end NightstreamFPrime.Lifecycle.BoundedDepth.LightRlcLayout
