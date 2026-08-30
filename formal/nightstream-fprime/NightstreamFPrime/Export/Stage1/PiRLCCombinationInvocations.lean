import NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates
import NightstreamFPrime.Layout.Stage1.PiCCSInputs
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.Spartan

/-!
Owns the exact compact-template invocations for all four PiRLC combination
families.

The invocations follow the Lean parent order and the concrete
source/block/lane/cell order. All input, output, and R1CS-fresh columns are
mapped from the canonical cumulative source layout through the one final
Spartan permutation.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Layout.Stage1

def phase : Nat := 7
def sourceCount : Nat := 17

def laneFreshCosts : List Nat :=
  List.ofFn NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount

def laneRowCosts : List Nat := laneFreshCosts.map (fun count => count + 1)

def laneFreshCost (lane : Nat) : Nat := laneFreshCosts.getD lane 0
def laneRowCost (lane : Nat) : Nat := laneRowCosts.getD lane 0
def laneFreshPrefix (lane : Nat) : Nat := (laneFreshCosts.take lane).sum
def laneRowPrefix (lane : Nat) : Nat := (laneRowCosts.take lane).sum

def stepSize (blockCount cellCount : Nat) : Nat :=
  blockCount * (ringDegree * cellCount)

def logicalIndex (cellCount block lane cell : Nat) : Nat :=
  block * (ringDegree * cellCount) + lane * cellCount + cell

def sourceFreshCount (blockCount cellCount : Nat) : Nat :=
  blockCount * cellCount * 8100

def sourceRowCount (blockCount cellCount : Nat) : Nat :=
  blockCount * cellCount * 8154

def coordinateFreshPrefix (cellCount block lane cell : Nat) : Nat :=
  block * cellCount * 8100 + cellCount * laneFreshPrefix lane +
    cell * laneFreshCost lane

def coordinateRowPrefix (cellCount block lane cell : Nat) : Nat :=
  block * cellCount * 8154 + cellCount * laneRowPrefix lane +
    cell * laneRowCost lane

def challengeSourceStart (source : Nat) : Nat :=
  PiRLCStarts.challengeWordStart source

def commitmentValueSourceStart (source block _cell : Nat) : Nat :=
  if source = 0 then
    PiCCSInputs.freshCommitmentStart + block * ringDegree
  else
    PiCCSInputs.runningCommitmentStart (source - 1) + block * ringDegree

def publicInputValueSourceStart (source block _cell : Nat) : Nat :=
  if source = 0 then
    NightstreamFPrime.Layout.PilotProduction.priorPublicInputStart +
      block * ringDegree
  else PiCCSInputs.runningPublicStart (source - 1) + block * ringDegree

def evalKValueSourceStart (source _block cell : Nat) : Nat :=
  PiCCSInputs.outputEvaluationStart + source * 1620 + cell

def evalAValueSourceStart (source block cell : Nat) : Nat :=
  PiCCSInputs.outputEvaluationStart + source * 1620 + 108 +
    block * 108 + cell

theorem commitmentValueSource_affine (source block cell offset : Nat)
    (sourceLt : source < sourceCount) (blockLt : block < 18)
    (offsetLt : offset < ringDegree) :
    Spartan.sourceToSpartan
        (commitmentValueSourceStart source block cell + offset) =
      Spartan.sourceToSpartan
          (commitmentValueSourceStart source block cell) + offset := by
  by_cases first : source = 0
  · subst source
    simp only [commitmentValueSourceStart, if_pos rfl]
    apply Spartan.sourceToSpartan_add_of_proofInput
    · norm_num [PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
        PiCCSInputs.expectedContextWords, Spartan.proofInputSourceStart]
    · norm_num [PiCCSInputs.freshCommitmentStart,
        PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
        PiCCSInputs.expectedContextWords, Spartan.piCcsPhaseOffset,
        ringDegree] at blockLt offsetLt ⊢
      omega
  · simp only [commitmentValueSourceStart, if_neg first]
    apply Spartan.sourceToSpartan_add_of_pilotPriorPrivate
    norm_num [PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      sourceCount, ringDegree] at sourceLt blockLt offsetLt ⊢
    omega

theorem publicInputValueSource_affine (source block cell offset : Nat)
    (sourceLt : source < sourceCount) (blockLt : block < 5)
    (offsetLt : offset < ringDegree) :
    Spartan.sourceToSpartan
        (publicInputValueSourceStart source block cell + offset) =
      Spartan.sourceToSpartan
          (publicInputValueSourceStart source block cell) + offset := by
  by_cases first : source = 0
  · subst source
    simp [publicInputValueSourceStart]
    apply Spartan.sourceToSpartan_add_of_pilotPriorPublic
    · omega
    · norm_num [PilotProduction.outputPreimageStart,
        PilotProduction.priorPublicInputStart,
        PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
        NightstreamFPrime.Lifecycle.PriorStateHash.publicWidth_eq,
        ringDegree] at blockLt offsetLt ⊢
      omega
  · simp only [publicInputValueSourceStart, if_neg first]
    apply Spartan.sourceToSpartan_add_of_pilotPriorPrivate
    norm_num [PiCCSInputs.runningPublicStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      sourceCount, ringDegree] at sourceLt blockLt offsetLt ⊢
    omega

theorem evalKValueSource_affine (source block cell offset : Nat)
    (sourceLt : source < sourceCount) (cellLt : cell < 2)
    (offsetLt : offset < ringDegree) :
    Spartan.sourceToSpartan
        (evalKValueSourceStart source block cell + offset * 2) =
      Spartan.sourceToSpartan (evalKValueSourceStart source block cell) +
        offset * 2 := by
  apply Spartan.sourceToSpartan_add_of_proofInput
  · norm_num [evalKValueSourceStart, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.roundMessageWords, Spartan.proofInputSourceStart]
    omega
  · norm_num [evalKValueSourceStart, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.roundMessageWords, Spartan.piCcsPhaseOffset, sourceCount,
      ringDegree] at sourceLt cellLt offsetLt ⊢
    omega

theorem evalAValueSource_affine (source block cell offset : Nat)
    (sourceLt : source < sourceCount) (blockLt : block < 14)
    (cellLt : cell < 2) (offsetLt : offset < ringDegree) :
    Spartan.sourceToSpartan
        (evalAValueSourceStart source block cell + offset * 2) =
      Spartan.sourceToSpartan (evalAValueSourceStart source block cell) +
        offset * 2 := by
  apply Spartan.sourceToSpartan_add_of_proofInput
  · norm_num [evalAValueSourceStart, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.roundMessageWords, Spartan.proofInputSourceStart]
    omega
  · norm_num [evalAValueSourceStart, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentStart,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.roundMessageWords, Spartan.piCcsPhaseOffset, sourceCount,
      ringDegree] at sourceLt blockLt cellLt offsetLt ⊢
    omega

def inputRanges (logicalStart blockCount cellCount valueStride source block
    lane cell : Nat) (valueSourceStart : Nat → Nat → Nat → Nat) :
    List CompactInputRange :=
  let index := logicalIndex cellCount block lane cell
  let outputSource := logicalStart + source * stepSize blockCount cellCount + index
  let priorSource := if source = 0 then 0 else
    logicalStart + (source - 1) * stepSize blockCount cellCount + index
  [⟨PiRLCCombinationTemplates.challengeInputStart, ringDegree,
      Spartan.sourceToSpartan (challengeSourceStart source), 1⟩,
   ⟨PiRLCCombinationTemplates.valueInputStart, ringDegree,
      Spartan.sourceToSpartan (valueSourceStart source block cell), valueStride⟩,
   ⟨PiRLCCombinationTemplates.priorInput, 1,
      Spartan.sourceToSpartan priorSource, 1⟩,
   ⟨PiRLCCombinationTemplates.outputInput, 1,
      Spartan.sourceToSpartan outputSource, 1⟩]

def invocationFreshSource (freshStart blockCount cellCount source block lane
    cell : Nat) : Nat :=
  freshStart + source * sourceFreshCount blockCount cellCount +
    coordinateFreshPrefix cellCount block lane cell

theorem phaseFreshStart_local :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseFreshStart := by
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem commitmentFreshStart_local :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.commitmentFreshStart := by
  rw [PiRLCStarts.commitmentFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem publicInputFreshStart_local :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.publicInputFreshStart := by
  rw [PiRLCStarts.publicInputFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem evalKFreshStart_local :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.evalKFreshStart := by
  rw [PiRLCStarts.evalKFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem evalAFreshStart_local :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.evalAFreshStart := by
  rw [PiRLCStarts.evalAFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem invocationFreshSource_local (freshStart blockCount cellCount source
    block lane cell : Nat)
    (freshStartLocal : Spartan.piCcsPhaseOffset ≤ freshStart) :
    Spartan.piCcsPhaseOffset ≤
      invocationFreshSource freshStart blockCount cellCount source block lane
        cell := by
  unfold invocationFreshSource
  omega

def sourceInputColumn (logicalStart blockCount cellCount valueStride source
    block lane cell input : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : Nat :=
  let index := logicalIndex cellCount block lane cell
  if input < PiRLCCombinationTemplates.valueInputStart then
    challengeSourceStart source + input
  else if input < PiRLCCombinationTemplates.priorInput then
    valueSourceStart source block cell +
      (input - PiRLCCombinationTemplates.valueInputStart) * valueStride
  else if input = PiRLCCombinationTemplates.priorInput then
    if source = 0 then 0 else
      logicalStart + (source - 1) * stepSize blockCount cellCount + index
  else if input = PiRLCCombinationTemplates.outputInput then
    logicalStart + source * stepSize blockCount cellCount + index
  else 0

def finalInputColumn (logicalStart blockCount cellCount valueStride source
    block lane cell input : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : Nat :=
  Spartan.sourceToSpartan
    (sourceInputColumn logicalStart blockCount cellCount valueStride source
      block lane cell input valueSourceStart)

def firstSource (source : Nat) : Bool := source == 0

def productionSharedInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} : Formal.Interface logicalWidth publicFits :=
  Formal.atOffset
    (PiRLCInputs.interface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) PiRLCInputs.phaseOffset

def productionCommitmentFamilyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    CombinationFamily.Interface CommitmentCombination.blockCount
      CommitmentCombination.cellCount :=
  CommitmentCombination.familyInterface
    (Formal.commitmentInterface
      (productionSharedInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits)))

def productionPublicInputFamilyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    CombinationFamily.Interface PublicInputCombination.blockCount
      PublicInputCombination.cellCount :=
  PublicInputCombination.familyInterface
    (Formal.publicInputInterface
      (productionSharedInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits)))

def productionEvalKFamilyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    CombinationFamily.Interface EvalKCombination.blockCount
      RingKCombination.cellCount :=
  RingKCombination.familyInterface
    (EvalKCombination.ringInterface
      (Formal.evalKInterface
        (productionSharedInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))))

def productionEvalAFamilyInterface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    CombinationFamily.Interface EvalACombination.blockCount
      RingKCombination.cellCount :=
  RingKCombination.familyInterface
    (EvalACombination.ringInterface
      (Formal.evalAInterface
        (productionSharedInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))))

def sourceChallenge (source : Nat) (lane : Fin ringDegree) : Expr :=
  Expr.var (challengeSourceStart source + lane.val) - 2

theorem samplerChallenge_eq_sourceChallenge
    (samplerInterface : SamplerChain.Interface)
    (source : Fin CombinationFamily.sourceCount) (lane : Fin ringDegree) :
    SamplerChain.challengeExpr samplerInterface PiRLCInputs.phaseOffset source lane =
      sourceChallenge source.val lane := by
  unfold SamplerChain.challengeExpr Sampler.outputChallenge Sampler.outputWord
    Sampler.outputSlot NightstreamFPrime.Gadgets.Sampling.First54.output
    NightstreamFPrime.Gadgets.Sampling.First54.valueOffset
    NightstreamFPrime.Gadgets.Sampling.First54.positionOffset
    NightstreamFPrime.Gadgets.Sampling.First54ValueStep.output
    Sampler.selectorOffset Sampler.windowBase SamplerChain.sourceOffset
    sourceChallenge challengeSourceStart
  rw [PiRLCStarts.challengeWordStart_eq]
  congr 3

def sourceValue (valueStride source block cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (lane : Fin ringDegree) : Expr :=
  Expr.var (valueSourceStart source block cell + lane.val * valueStride)

theorem productionCommitmentValue_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin CommitmentCombination.blockCount)
    (lane : Fin ringDegree) :
    (PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).commitment block lane =
      sourceValue 1 source.val block.val 0 commitmentValueSourceStart lane := by
  unfold PiRLCInputs.sourceInput
  split
  · rename_i isFresh
    have freshCount :
        NightstreamFPrime.Lifecycle.productionShape.freshCount = 1 := rfl
    have sourceZero : source.val = 0 := by
      rw [freshCount] at isFresh
      omega
    change Expr.var
        (PiCCSInputs.freshCommitmentStart +
          source.val * PiCCSInputs.freshCommitmentWords +
          block.val * ringDegree + lane.val) =
      Expr.var
        (commitmentValueSourceStart source.val block.val 0 + lane.val * 1)
    simp [commitmentValueSourceStart, sourceZero]
  · rename_i notFresh
    have freshCount :
        NightstreamFPrime.Lifecycle.productionShape.freshCount = 1 := rfl
    have sourceNotZero : source.val ≠ 0 := by
      intro sourceZero
      apply notFresh
      rw [sourceZero, freshCount]
      decide
    change Expr.var
        (PiCCSInputs.runningCommitmentStart
            (source.val - NightstreamFPrime.Lifecycle.productionShape.freshCount) +
          block.val * ringDegree + lane.val) =
      Expr.var
        (commitmentValueSourceStart source.val block.val 0 + lane.val * 1)
    rw [freshCount]
    simp [commitmentValueSourceStart, sourceNotZero]

theorem productionPublicInputValue_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin PublicInputCombination.blockCount)
    (lane : Fin ringDegree) :
    (PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).publicInput
        (PublicInputCombination.publicColumn block lane) =
      sourceValue 1 source.val block.val 0 publicInputValueSourceStart lane := by
  unfold PiRLCInputs.sourceInput
  split
  · rename_i isFresh
    have freshCount :
        NightstreamFPrime.Lifecycle.productionShape.freshCount = 1 := rfl
    have sourceZero : source.val = 0 := by
      rw [freshCount] at isFresh
      omega
    change Expr.var
        (PilotProduction.priorPublicInputStart +
          (block.val * ringDegree + lane.val)) =
      Expr.var
        (publicInputValueSourceStart source.val block.val 0 + lane.val * 1)
    simp [publicInputValueSourceStart, sourceZero, Nat.add_assoc]
  · rename_i notFresh
    have freshCount :
        NightstreamFPrime.Lifecycle.productionShape.freshCount = 1 := rfl
    have sourceNotZero : source.val ≠ 0 := by
      intro sourceZero
      apply notFresh
      rw [sourceZero, freshCount]
      decide
    change Expr.var
        (PiCCSInputs.runningPublicStart
            (source.val - NightstreamFPrime.Lifecycle.productionShape.freshCount) +
          (block.val * ringDegree + lane.val)) =
      Expr.var
        (publicInputValueSourceStart source.val block.val 0 + lane.val * 1)
    rw [freshCount]
    simp [publicInputValueSourceStart, sourceNotZero, Nat.add_assoc]

theorem productionEvalKValue_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin EvalKCombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_K
            (EvalKCombination.coefficient lane)) =
      sourceValue 2 source.val block.val cell.val evalKValueSourceStart lane := by
  unfold PiRLCInputs.sourceInput
  split <;> fin_cases cell <;>
    simp [PiRLCInputs.piCcsInterface, PiCCSInputs.interface,
      PiCCSInputs.outputExpr, PiCCSInputs.outputEval_K, PiCCSInputs.pairAt,
      RingKCombination.expressionCell, EvalKCombination.coefficient,
      sourceValue, evalKValueSourceStart] <;> omega

theorem productionEvalAValue_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin EvalACombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    RingKCombination.expressionCell cell
        ((PiRLCInputs.sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_A block
            (EvalKCombination.coefficient lane)) =
      sourceValue 2 source.val block.val cell.val evalAValueSourceStart lane := by
  unfold PiRLCInputs.sourceInput
  split <;> fin_cases cell <;>
    simp [PiRLCInputs.piCcsInterface, PiCCSInputs.interface,
      PiCCSInputs.outputExpr, PiCCSInputs.outputEval_A, PiCCSInputs.pairAt,
      RingKCombination.expressionCell, EvalKCombination.coefficient,
      sourceValue, evalAValueSourceStart] <;> omega

theorem indexOf_val {blockCount cellCount : Nat}
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    (CombinationStep.indexOf block lane cell).val =
      logicalIndex cellCount block.val lane.val cell.val := by
  simp [CombinationStep.indexOf, finProdFinEquiv, logicalIndex] <;> ring

theorem indexOf_coordinates {blockCount cellCount : Nat}
    (index : Fin (CombinationStep.privateCount blockCount cellCount)) :
    CombinationStep.indexOf (CombinationStep.coordinates index).1
        (CombinationStep.coordinates index).2.1
        (CombinationStep.coordinates index).2.2 = index := by
  unfold CombinationStep.coordinates CombinationStep.indexOf
  unfold CombinationStep.privateCount
  change finProdFinEquiv
      ((finProdFinEquiv.symm index).1,
        finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm index).2)) = index
  simp only [Equiv.apply_symm_apply, Prod.eta]

def sourcePrior (logicalStart blockCount cellCount source block lane cell : Nat) :
    Expr :=
  if source = 0 then 0 else
    Expr.var (logicalStart + (source - 1) * stepSize blockCount cellCount +
      logicalIndex cellCount block lane cell)

def sourceOutput (logicalStart blockCount cellCount source block lane cell : Nat) :
    Expr :=
  Expr.var (logicalStart + source * stepSize blockCount cellCount +
    logicalIndex cellCount block lane cell)

theorem sourceOutput_eq_stepOutput
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart : Nat) (source : Fin CombinationFamily.sourceCount)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    sourceOutput logicalStart blockCount cellCount source.val block.val lane.val
        cell.val =
      CombinationStep.output
        (CombinationFamily.stepOffset logicalStart source.val blockCount cellCount)
        (CombinationStep.indexOf block lane cell) := by
  simp [sourceOutput, CombinationStep.output, CombinationFamily.stepOffset,
    CombinationFamily.stepSize, CombinationStep.privateCount, stepSize,
    indexOf_val]

theorem sourcePrior_eq_priorAt
    {blockCount cellCount : Nat} [NeZero cellCount]
    (logicalStart : Nat) (source : Fin CombinationFamily.sourceCount)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    sourcePrior logicalStart blockCount cellCount source.val block.val lane.val
        cell.val =
      CombinationFamily.priorAt logicalStart source.val block lane cell := by
  by_cases first : source.val = 0
  · simp [sourcePrior, CombinationFamily.priorAt, first]
  · simp [sourcePrior, CombinationFamily.priorAt, first,
      CombinationStep.output, CombinationFamily.stepOffset,
      CombinationFamily.stepSize, CombinationStep.privateCount, stepSize,
      indexOf_val]

theorem stepFlatConstraints_eq_assertions
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationStep.Interface blockCount cellCount) (offset : Nat) :
    flatConstraints (CombinationStep.operations interface offset) =
      List.ofFn fun index =>
        CombinationStep.output offset index -
          CombinationStep.recipe interface offset index := by
  rw [CombinationStep.flatConstraints_operations]
  change recipeConstraints offset
      (List.ofFn (CombinationStep.recipe interface offset)) = _
  exact NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.recipeConstraintsOfFn
    offset (CombinationStep.recipe interface offset)

def sourceConstraint (logicalStart blockCount cellCount valueStride source block
    cell : Nat) (valueSourceStart : Nat → Nat → Nat → Nat)
    (lane : Fin ringDegree) : Expr :=
  sourceOutput logicalStart blockCount cellCount source block lane.val cell -
    (sourcePrior logicalStart blockCount cellCount source block lane.val cell +
      CombinationStep.mulExpr (sourceChallenge source)
        (sourceValue valueStride source block cell valueSourceStart) lane)

theorem sourceConstraint_eq_stepAssertion
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride : Nat)
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (challengeEq : ∀ current,
      interface.challenge logicalStart source current =
        sourceChallenge source.val current)
    (valueEq : ∀ current,
      interface.input logicalStart source block current cell =
        sourceValue valueStride source.val block.val cell.val valueSourceStart
          current) :
    sourceConstraint logicalStart blockCount cellCount valueStride source.val
        block.val cell.val valueSourceStart lane =
      CombinationStep.output
          (CombinationFamily.stepOffset logicalStart source.val blockCount cellCount)
          (CombinationStep.indexOf block lane cell) -
        CombinationStep.recipe
          (CombinationFamily.stepInterface interface logicalStart source.val)
          (CombinationFamily.stepOffset logicalStart source.val blockCount cellCount)
          (CombinationStep.indexOf block lane cell) := by
  have challengeAtEq :
      CombinationFamily.challengeAt interface logicalStart source.val =
        sourceChallenge source.val := by
    funext current
    simpa [CombinationFamily.challengeAt, source.isLt] using challengeEq current
  have valueAtEq :
      CombinationStep.ringExpr
          (fun _ => CombinationFamily.inputAt interface logicalStart source.val)
          (CombinationFamily.stepOffset logicalStart source.val blockCount cellCount)
          block cell =
        sourceValue valueStride source.val block.val cell.val valueSourceStart := by
    funext current
    simpa [CombinationStep.ringExpr, CombinationFamily.inputAt, source.isLt]
      using valueEq current
  unfold sourceConstraint CombinationStep.recipe
  rw [sourceOutput_eq_stepOutput, sourcePrior_eq_priorAt]
  simp only [CombinationStep.blockOf_indexOf, CombinationStep.laneOf_indexOf,
    CombinationStep.cellOf_indexOf]
  simp only [CombinationFamily.stepInterface]
  rw [challengeAtEq, valueAtEq]

theorem commitmentSourceConstraint_eq_stepAssertion
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin CommitmentCombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin CommitmentCombination.cellCount) :
    sourceConstraint PiRLCStarts.commitmentLogicalStart
        CommitmentCombination.blockCount CommitmentCombination.cellCount 1
        source.val block.val cell.val commitmentValueSourceStart lane =
      CombinationStep.output
          (CombinationFamily.stepOffset PiRLCStarts.commitmentLogicalStart
            source.val CommitmentCombination.blockCount
            CommitmentCombination.cellCount)
          (CombinationStep.indexOf block lane cell) -
        CombinationStep.recipe
          (CombinationFamily.stepInterface
            (productionCommitmentFamilyInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))
            PiRLCStarts.commitmentLogicalStart source.val)
          (CombinationFamily.stepOffset PiRLCStarts.commitmentLogicalStart
            source.val CommitmentCombination.blockCount
            CommitmentCombination.cellCount)
          (CombinationStep.indexOf block lane cell) := by
  fin_cases cell
  apply sourceConstraint_eq_stepAssertion
  · intro current
    simpa [productionCommitmentFamilyInterface,
      CommitmentCombination.familyInterface, Formal.commitmentInterface,
      productionSharedInterface, Formal.atOffset] using
        samplerChallenge_eq_sourceChallenge
          (Formal.samplerInterface
            (productionSharedInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))) source current
  · intro current
    simpa [productionCommitmentFamilyInterface,
      CommitmentCombination.familyInterface, Formal.commitmentInterface,
      productionSharedInterface, Formal.atOffset, PiRLCInputs.interface] using
        productionCommitmentValue_eq
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source block current

theorem publicInputSourceConstraint_eq_stepAssertion
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin PublicInputCombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin PublicInputCombination.cellCount) :
    sourceConstraint PiRLCStarts.publicInputLogicalStart
        PublicInputCombination.blockCount PublicInputCombination.cellCount 1
        source.val block.val cell.val publicInputValueSourceStart lane =
      CombinationStep.output
          (CombinationFamily.stepOffset PiRLCStarts.publicInputLogicalStart
            source.val PublicInputCombination.blockCount
            PublicInputCombination.cellCount)
          (CombinationStep.indexOf block lane cell) -
        CombinationStep.recipe
          (CombinationFamily.stepInterface
            (productionPublicInputFamilyInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))
            PiRLCStarts.publicInputLogicalStart source.val)
          (CombinationFamily.stepOffset PiRLCStarts.publicInputLogicalStart
            source.val PublicInputCombination.blockCount
            PublicInputCombination.cellCount)
          (CombinationStep.indexOf block lane cell) := by
  fin_cases cell
  apply sourceConstraint_eq_stepAssertion
  · intro current
    simpa [productionPublicInputFamilyInterface,
      PublicInputCombination.familyInterface, Formal.publicInputInterface,
      productionSharedInterface, Formal.atOffset] using
        samplerChallenge_eq_sourceChallenge
          (Formal.samplerInterface
            (productionSharedInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))) source current
  · intro current
    simpa [productionPublicInputFamilyInterface,
      PublicInputCombination.familyInterface, Formal.publicInputInterface,
      productionSharedInterface, Formal.atOffset, PiRLCInputs.interface] using
        productionPublicInputValue_eq
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source block current

theorem evalKSourceConstraint_eq_stepAssertion
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin EvalKCombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    sourceConstraint PiRLCStarts.evalKLogicalStart EvalKCombination.blockCount
        RingKCombination.cellCount 2 source.val block.val cell.val
        evalKValueSourceStart lane =
      CombinationStep.output
          (CombinationFamily.stepOffset PiRLCStarts.evalKLogicalStart source.val
            EvalKCombination.blockCount RingKCombination.cellCount)
          (CombinationStep.indexOf block lane cell) -
        CombinationStep.recipe
          (CombinationFamily.stepInterface
            (productionEvalKFamilyInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))
            PiRLCStarts.evalKLogicalStart source.val)
          (CombinationFamily.stepOffset PiRLCStarts.evalKLogicalStart source.val
            EvalKCombination.blockCount RingKCombination.cellCount)
          (CombinationStep.indexOf block lane cell) := by
  apply sourceConstraint_eq_stepAssertion
  · intro current
    simpa [productionEvalKFamilyInterface, RingKCombination.familyInterface,
      EvalKCombination.ringInterface, Formal.evalKInterface,
      productionSharedInterface, Formal.atOffset] using
        samplerChallenge_eq_sourceChallenge
          (Formal.samplerInterface
            (productionSharedInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))) source current
  · intro current
    simpa [productionEvalKFamilyInterface, RingKCombination.familyInterface,
      EvalKCombination.ringInterface, Formal.evalKInterface,
      productionSharedInterface, Formal.atOffset, PiRLCInputs.interface] using
        productionEvalKValue_eq
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source block current cell

theorem evalASourceConstraint_eq_stepAssertion
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount)
    (block : Fin EvalACombination.blockCount)
    (lane : Fin ringDegree) (cell : Fin RingKCombination.cellCount) :
    sourceConstraint PiRLCStarts.evalALogicalStart EvalACombination.blockCount
        RingKCombination.cellCount 2 source.val block.val cell.val
        evalAValueSourceStart lane =
      CombinationStep.output
          (CombinationFamily.stepOffset PiRLCStarts.evalALogicalStart source.val
            EvalACombination.blockCount RingKCombination.cellCount)
          (CombinationStep.indexOf block lane cell) -
        CombinationStep.recipe
          (CombinationFamily.stepInterface
            (productionEvalAFamilyInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))
            PiRLCStarts.evalALogicalStart source.val)
          (CombinationFamily.stepOffset PiRLCStarts.evalALogicalStart source.val
            EvalACombination.blockCount RingKCombination.cellCount)
          (CombinationStep.indexOf block lane cell) := by
  apply sourceConstraint_eq_stepAssertion
  · intro current
    simpa [productionEvalAFamilyInterface, RingKCombination.familyInterface,
      EvalACombination.ringInterface, Formal.evalAInterface,
      productionSharedInterface, Formal.atOffset] using
        samplerChallenge_eq_sourceChallenge
          (Formal.samplerInterface
            (productionSharedInterface (logicalWidth := logicalWidth)
              (publicFits := publicFits))) source current
  · intro current
    simpa [productionEvalAFamilyInterface, RingKCombination.familyInterface,
      EvalACombination.ringInterface, Formal.evalAInterface,
      productionSharedInterface, Formal.atOffset, PiRLCInputs.interface] using
        productionEvalAValue_eq
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source block current cell

theorem renamedConstraint_eq_sourceConstraint
    (logicalStart blockCount cellCount valueStride source block cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (lane : Fin ringDegree) :
    CompactRows.renameExpr
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell · valueSourceStart)
        (Expr.var PiRLCCombinationTemplates.outputInput -
          PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      sourceConstraint logicalStart blockCount cellCount valueStride source block
        cell valueSourceStart lane := by
  rw [CompactRows.renameExpr_sub]
  unfold PiRLCCombinationTemplates.outputRecipe
  simp only [CompactRows.renameExpr]
  rw [PiRLCCombinationTemplates.renameExpr_mulExpr]
  have challengeEq :
      (fun current => CompactRows.renameExpr
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell · valueSourceStart)
        (PiRLCCombinationTemplates.challenge current)) =
      sourceChallenge source := by
    funext current
    have currentBound : current.val < 54 := by
      simpa [ringDegree] using current.isLt
    simp [CompactRows.renameExpr, PiRLCCombinationTemplates.challenge,
      sourceChallenge,
      sourceInputColumn, PiRLCCombinationTemplates.challengeInputStart,
      PiRLCCombinationTemplates.valueInputStart, currentBound] <;> congr 3
  have valueEq :
      (fun current => CompactRows.renameExpr
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell · valueSourceStart)
        (PiRLCCombinationTemplates.value current)) =
      sourceValue valueStride source block cell valueSourceStart := by
    funext current
    have currentBound : current.val < 54 := by
      simpa [ringDegree] using current.isLt
    have notChallenge : ¬ 54 + current.val < 54 := by omega
    have beforePrior : 54 + current.val < 108 := by omega
    simp [CompactRows.renameExpr, PiRLCCombinationTemplates.value, sourceValue,
      sourceInputColumn,
      PiRLCCombinationTemplates.valueInputStart,
      PiRLCCombinationTemplates.priorInput, notChallenge, beforePrior]
  rw [challengeEq, valueEq]
  by_cases first : source = 0
  · subst source
    simp [CompactRows.renameExpr, firstSource,
      PiRLCCombinationTemplates.prior,
      sourceConstraint, sourceOutput, sourcePrior, sourceInputColumn,
      PiRLCCombinationTemplates.outputInput,
      PiRLCCombinationTemplates.priorInput,
      PiRLCCombinationTemplates.valueInputStart] <;> congr 3
  · simp [CompactRows.renameExpr, firstSource, first,
      PiRLCCombinationTemplates.prior,
      sourceConstraint, sourceOutput, sourcePrior, sourceInputColumn,
      PiRLCCombinationTemplates.outputInput,
      PiRLCCombinationTemplates.priorInput,
      PiRLCCombinationTemplates.valueInputStart] <;> congr 3

theorem renamedOutputRecipe_eq_sourceRecipe
    (logicalStart blockCount cellCount valueStride source block cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (lane : Fin ringDegree) :
    CompactRows.renameExpr
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell · valueSourceStart)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      sourcePrior logicalStart blockCount cellCount source block lane.val cell +
        CombinationStep.mulExpr (sourceChallenge source)
          (sourceValue valueStride source block cell valueSourceStart) lane := by
  have constraintEq := renamedConstraint_eq_sourceConstraint logicalStart
    blockCount cellCount valueStride source block cell valueSourceStart lane
  have outputEq :
      CompactRows.renameExpr
          (sourceInputColumn logicalStart blockCount cellCount valueStride source
            block lane.val cell · valueSourceStart)
          (Expr.var PiRLCCombinationTemplates.outputInput) =
        sourceOutput logicalStart blockCount cellCount source block lane.val
          cell := by
    simp [CompactRows.renameExpr, sourceInputColumn, sourceOutput,
      PiRLCCombinationTemplates.outputInput,
      PiRLCCombinationTemplates.valueInputStart,
      PiRLCCombinationTemplates.priorInput]
  rw [CompactRows.renameExpr_sub, outputEq] at constraintEq
  unfold sourceConstraint at constraintEq
  change Expr.add _ (Expr.mul _ _) = Expr.add _ (Expr.mul _ _) at constraintEq
  injection constraintEq with _ negativeEq
  injection negativeEq with _ recipeEq

theorem inputColumnOfRanges_eq (logicalStart blockCount cellCount valueStride
    source block lane cell input : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (valueAffine : ∀ offset, offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block cell + offset * valueStride) =
        Spartan.sourceToSpartan (valueSourceStart source block cell) +
          offset * valueStride)
    (inputBound : input < PiRLCCombinationTemplates.inputCount) :
    CompactRows.inputColumnOfRanges
        (inputRanges logicalStart blockCount cellCount valueStride source block
          lane cell valueSourceStart) input =
      finalInputColumn logicalStart blockCount cellCount valueStride source
        block lane cell input valueSourceStart := by
  have phaseLocal : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseLogicalStart := by
    norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.phaseLogicalStart,
      PiRLCInputs.phaseOffset]
  have challengeLocal : Spartan.piCcsPhaseOffset ≤ challengeSourceStart source := by
    unfold challengeSourceStart PiRLCStarts.challengeWordStart
      PiRLCStarts.selectorLogicalStart PiRLCStarts.samplerSourceLogicalStart
      PiRLCStarts.samplerLogicalStart
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
    omega
  have challengeAffine (offset : Nat) :
      Spartan.sourceToSpartan (challengeSourceStart source + offset) =
        Spartan.sourceToSpartan (challengeSourceStart source) + offset :=
    Spartan.sourceToSpartan_add_of_piCcsLocal _ _ challengeLocal
  unfold CompactRows.inputColumnOfRanges compactInputColumn inputRanges
    finalInputColumn
    sourceInputColumn
  dsimp only
  by_cases challenge : input < 54
  · simp [PiRLCCombinationTemplates.challengeInputStart,
      PiRLCCombinationTemplates.valueInputStart,
      PiRLCCombinationTemplates.priorInput,
      PiRLCCombinationTemplates.outputInput, ringDegree, challenge,
      challengeAffine]
  · by_cases value : input < 108
    · have valueStart : 54 ≤ input := by omega
      have valueOffsetBound : input - 54 < ringDegree := by
        norm_num [ringDegree]
        omega
      have valueMapped := valueAffine (input - 54) valueOffsetBound
      simp [PiRLCCombinationTemplates.challengeInputStart,
        PiRLCCombinationTemplates.valueInputStart,
        PiRLCCombinationTemplates.priorInput,
        PiRLCCombinationTemplates.outputInput, ringDegree, challenge, value,
        valueStart, valueMapped]
    · by_cases prior : input = 108
      · subst input
        simp [PiRLCCombinationTemplates.challengeInputStart,
          PiRLCCombinationTemplates.valueInputStart,
          PiRLCCombinationTemplates.priorInput,
          PiRLCCombinationTemplates.outputInput, ringDegree]
      · have output : input = 109 := by
          unfold PiRLCCombinationTemplates.inputCount at inputBound
          omega
        subst input
        simp [PiRLCCombinationTemplates.challengeInputStart,
          PiRLCCombinationTemplates.valueInputStart,
          PiRLCCombinationTemplates.priorInput,
          PiRLCCombinationTemplates.outputInput, ringDegree]

theorem sourceToSpartan_zero : Spartan.sourceToSpartan 0 = 0 := by
  rfl

theorem spartanRemapRows_eq (rows : List R1CS.Row) :
    Spartan.remapRows rows =
      rows.map (CompactRows.renameRow Spartan.sourceToSpartan) := by
  rfl

def invocation (logicalStart rowStart freshStart blockCount cellCount valueStride
    source block lane cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : CompactRowInvocation :=
  let freshSource := invocationFreshSource freshStart blockCount cellCount
    source block lane cell
  { phase := phase
    templateIndex := PiRLCCombinationTemplates.templateIndex source lane
    rowStart := rowStart + source * sourceRowCount blockCount cellCount +
      coordinateRowPrefix cellCount block lane cell
    localStart := Spartan.sourceToSpartan freshSource
    inputRanges := inputRanges logicalStart blockCount cellCount valueStride
      source block lane cell valueSourceStart }

@[simp] theorem invocation_localStart
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block lane cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    (invocation logicalStart rowStart freshStart blockCount cellCount valueStride
      source block lane cell valueSourceStart).localStart =
      Spartan.sourceToSpartan
        (invocationFreshSource freshStart blockCount cellCount source block lane
          cell) := by
  rfl

@[simp] theorem invocation_inputRanges
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block lane cell : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    (invocation logicalStart rowStart freshStart blockCount cellCount valueStride
      source block lane cell valueSourceStart).inputRanges =
      inputRanges logicalStart blockCount cellCount valueStride source block lane
        cell valueSourceStart := by
  rfl

theorem invocationOutputRecipe_eq_remappedSourceRecipe
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block cell : Nat) (lane : Fin ringDegree)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (valueAffine : ∀ offset, offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block cell + offset * valueStride) =
        Spartan.sourceToSpartan (valueSourceStart source block cell) +
          offset * valueStride) :
    CompactRows.renameExpr
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source block lane.val cell valueSourceStart).inputRanges)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      CompactRows.renameExpr Spartan.sourceToSpartan
        (sourcePrior logicalStart blockCount cellCount source block lane.val cell +
          CombinationStep.mulExpr (sourceChallenge source)
            (sourceValue valueStride source block cell valueSourceStart) lane) := by
  have scope :
      (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane).VarsBelow
        PiRLCCombinationTemplates.inputCount :=
    (PiRLCCombinationTemplates.constraint_varsBelow
      (firstSource source) lane).2.2
  have rangeEq :
      CompactRows.renameExpr
          (CompactRows.inputColumnOfRanges
            (inputRanges logicalStart blockCount cellCount valueStride source
              block lane.val cell valueSourceStart))
          (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
        CompactRows.renameExpr
          (finalInputColumn logicalStart blockCount cellCount valueStride source
            block lane.val cell · valueSourceStart)
          (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) := by
    apply CompactRows.renameExpr_congr _ _ _ scope
    intro input inputBound
    exact inputColumnOfRanges_eq logicalStart blockCount cellCount valueStride
      source block lane.val cell input valueSourceStart valueAffine inputBound
  unfold invocation
  dsimp only
  rw [rangeEq]
  change CompactRows.renameExpr
      (Spartan.sourceToSpartan ∘
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell · valueSourceStart))
      (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) = _
  rw [← CompactRows.renameExpr_comp]
  rw [renamedOutputRecipe_eq_sourceRecipe]

theorem commitmentInvocationOutputRecipe_eq_remappedSourceRecipe
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 18) :
    CompactRows.renameExpr
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.commitmentLogicalStart
            PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
            18 1 1 source block lane.val cell
            commitmentValueSourceStart).inputRanges)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      CompactRows.renameExpr Spartan.sourceToSpartan
        (sourcePrior PiRLCStarts.commitmentLogicalStart 18 1 source block
            lane.val cell +
          CombinationStep.mulExpr (sourceChallenge source)
            (sourceValue 1 source block cell commitmentValueSourceStart)
            lane) := by
  apply invocationOutputRecipe_eq_remappedSourceRecipe
  intro offset offsetLt
  simpa using commitmentValueSource_affine source block cell offset sourceLt
    blockLt offsetLt

theorem publicInputInvocationOutputRecipe_eq_remappedSourceRecipe
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 5) :
    CompactRows.renameExpr
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.publicInputLogicalStart
            PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
            5 1 1 source block lane.val cell
            publicInputValueSourceStart).inputRanges)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      CompactRows.renameExpr Spartan.sourceToSpartan
        (sourcePrior PiRLCStarts.publicInputLogicalStart 5 1 source block
            lane.val cell +
          CombinationStep.mulExpr (sourceChallenge source)
            (sourceValue 1 source block cell publicInputValueSourceStart)
            lane) := by
  apply invocationOutputRecipe_eq_remappedSourceRecipe
  intro offset offsetLt
  simpa using publicInputValueSource_affine source block cell offset sourceLt
    blockLt offsetLt

theorem evalKInvocationOutputRecipe_eq_remappedSourceRecipe
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (cellLt : cell < 2) :
    CompactRows.renameExpr
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
            PiRLCStarts.evalKFreshStart 1 2 2 source block lane.val cell
            evalKValueSourceStart).inputRanges)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      CompactRows.renameExpr Spartan.sourceToSpartan
        (sourcePrior PiRLCStarts.evalKLogicalStart 1 2 source block lane.val
            cell +
          CombinationStep.mulExpr (sourceChallenge source)
            (sourceValue 2 source block cell evalKValueSourceStart) lane) := by
  apply invocationOutputRecipe_eq_remappedSourceRecipe
  intro offset offsetLt
  exact evalKValueSource_affine source block cell offset sourceLt cellLt
    offsetLt

theorem evalAInvocationOutputRecipe_eq_remappedSourceRecipe
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 14)
    (cellLt : cell < 2) :
    CompactRows.renameExpr
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
            PiRLCStarts.evalAFreshStart 14 2 2 source block lane.val cell
            evalAValueSourceStart).inputRanges)
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane) =
      CompactRows.renameExpr Spartan.sourceToSpartan
        (sourcePrior PiRLCStarts.evalALogicalStart 14 2 source block lane.val
            cell +
          CombinationStep.mulExpr (sourceChallenge source)
            (sourceValue 2 source block cell evalAValueSourceStart) lane) := by
  apply invocationOutputRecipe_eq_remappedSourceRecipe
  intro offset offsetLt
  exact evalAValueSource_affine source block cell offset sourceLt blockLt
    cellLt offsetLt

/-- One serialized invocation expands to the exact Spartan image of its
Lean-lowered source constraint. The two affine hypotheses are layout facts;
the row expression and all A/B/C entries are fixed by Lean. -/
theorem invocationRows_eq_remappedSource
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block cell : Nat) (lane : Fin ringDegree)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤
      invocationFreshSource freshStart blockCount cellCount source block
        lane.val cell)
    (valueAffine : ∀ offset, offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block cell + offset * valueStride) =
        Spartan.sourceToSpartan (valueSourceStart source block cell) +
          offset * valueStride) :
    CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source block lane.val cell valueSourceStart).inputRanges)
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source block lane.val cell valueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane) =
      Spartan.remapRows
        (R1CS.lowerGenericConstraint
          (sourceConstraint logicalStart blockCount cellCount valueStride source
            block cell valueSourceStart lane)
          (invocationFreshSource freshStart blockCount cellCount source block
            lane.val cell)).rows := by
  let sourceFresh := invocationFreshSource freshStart blockCount cellCount
    source block lane.val cell
  have sourceBound : PiRLCCombinationTemplates.inputCount ≤ sourceFresh := by
    dsimp [sourceFresh]
    norm_num [PiRLCCombinationTemplates.inputCount,
      Spartan.piCcsPhaseOffset] at sourceLocal ⊢
    omega
  have finalLocal : Spartan.piCcsLocalStart ≤
      Spartan.sourceToSpartan sourceFresh :=
    Spartan.piCcsLocalStart_le_sourceToSpartan sourceFresh (by
      simpa [sourceFresh] using sourceLocal)
  have finalBound : PiRLCCombinationTemplates.inputCount ≤
      Spartan.sourceToSpartan sourceFresh := by
    norm_num [PiRLCCombinationTemplates.inputCount,
      Spartan.piCcsLocalStart] at finalLocal ⊢
    omega
  have rangeRows :
      CompactRows.instantiateRows
          (CompactRows.inputColumnOfRanges
            (inputRanges logicalStart blockCount cellCount valueStride source block
              lane.val cell valueSourceStart))
          (Spartan.sourceToSpartan sourceFresh)
          (PiRLCCombinationTemplates.template (firstSource source) lane) =
        CompactRows.instantiateRows
          (finalInputColumn logicalStart blockCount cellCount valueStride source
            block lane.val cell · valueSourceStart)
          (Spartan.sourceToSpartan sourceFresh)
          (PiRLCCombinationTemplates.template (firstSource source) lane) := by
    unfold PiRLCCombinationTemplates.template
    apply CompactRows.instantiate_compactTemplate_congr_inputs
    intro input inputBound
    exact inputColumnOfRanges_eq logicalStart blockCount cellCount valueStride
      source block lane.val cell input valueSourceStart valueAffine inputBound
  have remappedRows := CompactRows.instantiate_compactTemplate_remap
    PiRLCCombinationTemplates.inputCount PiRLCCombinationTemplates.outputInput
    sourceFresh (Spartan.sourceToSpartan sourceFresh)
    (sourceInputColumn logicalStart blockCount cellCount valueStride source block
      lane.val cell · valueSourceStart)
    Spartan.sourceToSpartan
    (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane)
    sourceBound finalBound
    (PiRLCCombinationTemplates.constraint_varsBelow (firstSource source) lane)
    (fun offset => Spartan.sourceToSpartan_add_of_piCcsLocal sourceFresh offset
      (by simpa [sourceFresh] using sourceLocal))
  unfold invocation
  dsimp only
  rw [rangeRows]
  change CompactRows.instantiateRows
      (fun input => Spartan.sourceToSpartan
        (sourceInputColumn logicalStart blockCount cellCount valueStride source
          block lane.val cell input valueSourceStart))
      (Spartan.sourceToSpartan sourceFresh)
      (CompactRows.compactTemplate PiRLCCombinationTemplates.inputCount
        PiRLCCombinationTemplates.outputInput
        (PiRLCCombinationTemplates.outputRecipe (firstSource source) lane)) = _
  rw [remappedRows]
  rw [renamedConstraint_eq_sourceConstraint]
  exact (spartanRemapRows_eq _).symm

theorem commitmentInvocationRows_eq_remappedSource
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 18) :
    CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.commitmentLogicalStart
            PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
            18 1 1 source block lane.val cell
            commitmentValueSourceStart).inputRanges)
        (invocation PiRLCStarts.commitmentLogicalStart
          PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
          18 1 1 source block lane.val cell
          commitmentValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane) =
      Spartan.remapRows
        (R1CS.lowerGenericConstraint
          (sourceConstraint PiRLCStarts.commitmentLogicalStart 18 1 1 source
            block cell commitmentValueSourceStart lane)
          (invocationFreshSource PiRLCStarts.commitmentFreshStart 18 1 source
            block lane.val cell)).rows := by
  apply invocationRows_eq_remappedSource
  · exact invocationFreshSource_local _ _ _ _ _ _ _
      commitmentFreshStart_local
  · intro offset offsetLt
    simpa using commitmentValueSource_affine source block cell offset sourceLt
      blockLt offsetLt

theorem publicInputInvocationRows_eq_remappedSource
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 5) :
    CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.publicInputLogicalStart
            PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
            5 1 1 source block lane.val cell
            publicInputValueSourceStart).inputRanges)
        (invocation PiRLCStarts.publicInputLogicalStart
          PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
          5 1 1 source block lane.val cell
          publicInputValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane) =
      Spartan.remapRows
        (R1CS.lowerGenericConstraint
          (sourceConstraint PiRLCStarts.publicInputLogicalStart 5 1 1 source
            block cell publicInputValueSourceStart lane)
          (invocationFreshSource PiRLCStarts.publicInputFreshStart 5 1 source
            block lane.val cell)).rows := by
  apply invocationRows_eq_remappedSource
  · exact invocationFreshSource_local _ _ _ _ _ _ _
      publicInputFreshStart_local
  · intro offset offsetLt
    simpa using publicInputValueSource_affine source block cell offset sourceLt
      blockLt offsetLt

theorem evalKInvocationRows_eq_remappedSource
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (cellLt : cell < 2) :
    CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
            PiRLCStarts.evalKFreshStart 1 2 2 source block lane.val cell
            evalKValueSourceStart).inputRanges)
        (invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
          PiRLCStarts.evalKFreshStart 1 2 2 source block lane.val cell
          evalKValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane) =
      Spartan.remapRows
        (R1CS.lowerGenericConstraint
          (sourceConstraint PiRLCStarts.evalKLogicalStart 1 2 2 source block
            cell evalKValueSourceStart lane)
          (invocationFreshSource PiRLCStarts.evalKFreshStart 1 2 source block
            lane.val cell)).rows := by
  apply invocationRows_eq_remappedSource
  · exact invocationFreshSource_local _ _ _ _ _ _ _ evalKFreshStart_local
  · intro offset offsetLt
    exact evalKValueSource_affine source block cell offset sourceLt cellLt
      offsetLt

theorem evalAInvocationRows_eq_remappedSource
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 14)
    (cellLt : cell < 2) :
    CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
            PiRLCStarts.evalAFreshStart 14 2 2 source block lane.val cell
            evalAValueSourceStart).inputRanges)
        (invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
          PiRLCStarts.evalAFreshStart 14 2 2 source block lane.val cell
          evalAValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane) =
      Spartan.remapRows
        (R1CS.lowerGenericConstraint
          (sourceConstraint PiRLCStarts.evalALogicalStart 14 2 2 source block
            cell evalAValueSourceStart lane)
          (invocationFreshSource PiRLCStarts.evalAFreshStart 14 2 source block
            lane.val cell)).rows := by
  apply invocationRows_eq_remappedSource
  · exact invocationFreshSource_local _ _ _ _ _ _ _ evalAFreshStart_local
  · intro offset offsetLt
    exact evalAValueSource_affine source block cell offset sourceLt blockLt
      cellLt offsetLt

def familyInvocations (logicalStart rowStart freshStart blockCount cellCount
    valueStride : Nat) (valueSourceStart : Nat → Nat → Nat → Nat) :
    List CompactRowInvocation :=
  (List.range sourceCount).flatMap fun source =>
    List.ofFn fun index : Fin
        (CombinationStep.privateCount blockCount cellCount) =>
      let coordinates := CombinationStep.coordinates index
      invocation logicalStart rowStart freshStart blockCount cellCount
        valueStride source coordinates.1.val coordinates.2.1.val
          coordinates.2.2.val valueSourceStart

def commitmentInvocations : List CompactRowInvocation :=
  familyInvocations PiRLCStarts.commitmentLogicalStart
    PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
    18 1 1 commitmentValueSourceStart

def publicInputInvocations : List CompactRowInvocation :=
  familyInvocations PiRLCStarts.publicInputLogicalStart
    PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
    5 1 1 publicInputValueSourceStart

def evalKInvocations : List CompactRowInvocation :=
  familyInvocations PiRLCStarts.evalKLogicalStart
    PiRLCStarts.evalKRowStart PiRLCStarts.evalKFreshStart
    1 2 2 evalKValueSourceStart

def evalAInvocations : List CompactRowInvocation :=
  familyInvocations PiRLCStarts.evalALogicalStart
    PiRLCStarts.evalARowStart PiRLCStarts.evalAFreshStart
    14 2 2 evalAValueSourceStart

def invocations : List CompactRowInvocation :=
  commitmentInvocations ++ publicInputInvocations ++
    evalKInvocations ++ evalAInvocations

theorem familyInvocation_mem
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat)
    (source : Fin sourceCount)
    (index : Fin (CombinationStep.privateCount blockCount cellCount)) :
    let coordinates := CombinationStep.coordinates index
    invocation logicalStart rowStart freshStart blockCount cellCount valueStride
        source.val coordinates.1.val coordinates.2.1.val coordinates.2.2.val
          valueSourceStart ∈
      familyInvocations logicalStart rowStart freshStart blockCount cellCount
        valueStride valueSourceStart := by
  dsimp only
  unfold familyInvocations
  apply List.mem_flatMap.mpr
  refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
  simp

theorem laneFreshCosts_length : laneFreshCosts.length = 54 := by
  simp [laneFreshCosts, ringDegree]

theorem laneRowCosts_length : laneRowCosts.length = 54 := by
  simp [laneRowCosts, laneFreshCosts, ringDegree]

theorem laneFreshCosts_sum : laneFreshCosts.sum = 8100 := by
  exact PiRLCCombinationTemplates.laneFreshCount_sum

theorem laneRowCosts_sum : laneRowCosts.sum = 8154 := by
  exact PiRLCCombinationTemplates.laneRowCount_sum

theorem familyInvocations_length (logicalStart rowStart freshStart blockCount
    cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    (familyInvocations logicalStart rowStart freshStart blockCount cellCount
      valueStride valueSourceStart).length =
      sourceCount * blockCount * ringDegree * cellCount := by
  simp [familyInvocations, CombinationStep.privateCount]
  ring

theorem commitmentInvocations_length : commitmentInvocations.length = 16524 := by
  rw [commitmentInvocations, familyInvocations_length]
  rfl

theorem publicInputInvocations_length : publicInputInvocations.length = 4590 := by
  rw [publicInputInvocations, familyInvocations_length]
  rfl

theorem evalKInvocations_length : evalKInvocations.length = 1836 := by
  rw [evalKInvocations, familyInvocations_length]
  rfl

theorem evalAInvocations_length : evalAInvocations.length = 25704 := by
  rw [evalAInvocations, familyInvocations_length]
  rfl

theorem invocations_length : invocations.length = 48654 := by
  simp [invocations, commitmentInvocations_length,
    publicInputInvocations_length, evalKInvocations_length,
    evalAInvocations_length]


theorem familyBoundaries_eq :
    PiRLCStarts.commitmentRowStart + sourceCount * sourceRowCount 18 1 =
        PiRLCStarts.publicInputRowStart ∧
    PiRLCStarts.commitmentFreshStart + sourceCount * sourceFreshCount 18 1 =
        PiRLCStarts.publicInputFreshStart ∧
    PiRLCStarts.publicInputRowStart + sourceCount * sourceRowCount 5 1 =
        PiRLCStarts.evalKRowStart ∧
    PiRLCStarts.publicInputFreshStart + sourceCount * sourceFreshCount 5 1 =
        PiRLCStarts.evalKFreshStart ∧
    PiRLCStarts.evalKRowStart + sourceCount * sourceRowCount 1 2 =
        PiRLCStarts.evalARowStart ∧
    PiRLCStarts.evalKFreshStart + sourceCount * sourceFreshCount 1 2 =
        PiRLCStarts.evalAFreshStart ∧
    PiRLCStarts.evalARowStart + sourceCount * sourceRowCount 14 2 =
        PiRLCStarts.outputRowStart ∧
    PiRLCStarts.evalAFreshStart + sourceCount * sourceFreshCount 14 2 =
        PiRLCStarts.outputFreshStart := by
  norm_num [sourceCount, sourceRowCount, sourceFreshCount,
    PiRLCStarts.commitmentRowStart, PiRLCStarts.commitmentFreshStart,
    PiRLCStarts.publicInputRowStart, PiRLCStarts.publicInputFreshStart,
    PiRLCStarts.evalKRowStart, PiRLCStarts.evalKFreshStart,
    PiRLCStarts.evalARowStart, PiRLCStarts.evalAFreshStart,
    PiRLCStarts.outputRowStart, PiRLCStarts.outputFreshStart,
    PiRLCStarts.samplerRowStart, PiRLCStarts.samplerFreshStart,
    PiRLCStarts.phaseRowStart, PiRLCStarts.phaseFreshStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset]

end NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations
