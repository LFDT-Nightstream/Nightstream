import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding
import NightstreamFPrime.Layout.PiRLC.v1_1.Composition
import NightstreamFPrime.Layout.Stage1.PiCCSStarts

/-!
Owns the zero-copy PiCCS-to-PiRLC interface in the cumulative Stage 1 source
layout.

The bridge reuses the exact PiCCS `K + k` commitments and public inputs. It
uses the PiCCS-derived round point, separate output `Eval_K` and `Eval_A`
families, and post-output transcript state. It allocates no boundary column or
row. The later Spartan layout owns the one final column permutation.
-/

namespace NightstreamFPrime.Layout.Stage1.PiRLCInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The exact completed pilot-plus-PiCCS source-column endpoint. -/
def phaseOffset : Nat := 20064823

/-- The completed PiCCS transcript state precedes the physical PiCCS endpoint
that starts PiRLC. The intervening columns are the PiCCS lowering suffix. -/
theorem piCcsLogicalFreshBase_le_phaseOffset :
    PiCCSStarts.logicalFreshBase ≤ phaseOffset := by
  norm_num [PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq,
    phaseOffset]

theorem phaseOffset_matches_piCcs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    phaseOffset = PilotPiCCS.physicalColumnCount relation := by
  rw [PilotPiCCS.physicalColumnCount_eq]
  rfl

def piCcsInterface :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.Interface
      logicalWidth 9 publicFits :=
  PiCCSInputs.interface logicalWidth publicFits

def piCcsSharedInterface :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.Interface
      logicalWidth 9 publicFits :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset
    (piCcsInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) PiCCSInputs.phaseOffset

def piCcsOutputInterface :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingInterface
    (piCcsSharedInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))

/-- Exact circuit-owned post-PiCCS transcript state. -/
def piCcsOutputState :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.EState :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
    (piCcsOutputInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits)) PiCCSStarts.outputBindingWitnessStart

theorem piCcsOutputState_eq_parent
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    piCcsOutputState (logicalWidth := logicalWidth) (publicFits := publicFits) =
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState
        relation
        (piCcsInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) PiCCSInputs.phaseOffset := by
  unfold piCcsOutputState
    piCcsOutputInterface piCcsSharedInterface piCcsInterface
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState
  rw [← PiCCSStarts.outputBindingWitnessStart_matches relation]

/-- The output absorption is nonempty, so its final eight lanes are fresh
variables and therefore affine inputs to the first PiRLC sampler. -/
theorem piCcsOutputState_fresh :
    StateFresh
      (piCcsOutputState (logicalWidth := logicalWidth)
        (publicFits := publicFits)) := by
  unfold piCcsOutputState
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.duplexInterface
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.output
    NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.actions
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.absorbBlock
  apply compile_output_fresh_of_head_absorb
  intro empty
  have lengthZero := congrArg List.length empty
  simp [NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks,
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.blockExpr,
    Spec.Poseidon2.rate] at lengthZero

private def runningIndex (source : Fin productionShape.sourceCount)
    (notFresh : ¬source.val < productionShape.freshCount) :
    Fin productionShape.runningCount :=
  ⟨source.val - productionShape.freshCount, by
    have sourceBound := source.isLt
    simp only [Shape.sourceCount] at sourceBound
    omega⟩

/-- One exact source view. Source zero is fresh; the remaining sixteen
sources are running. The evaluation family always comes from the PiCCS
reduced output at the same unified source index. -/
def sourceInput (source : Fin productionShape.sourceCount) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.InputExpr
      logicalWidth publicFits :=
  let parent := piCcsInterface (logicalWidth := logicalWidth)
    (publicFits := publicFits)
  let output := parent.output PiCCSInputs.phaseOffset
  let evaluation :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.EvaluationExpr :=
    { eval_K := output.padCoordinate source
      eval_A := output.matrixCoordinate source }
  if isFresh : source.val < productionShape.freshCount then
    let fresh : Fin productionShape.freshCount := ⟨source.val, isFresh⟩
    { commitment := (parent.fresh PiCCSInputs.phaseOffset).commitment fresh
      publicInput := (parent.fresh PiCCSInputs.phaseOffset).publicInput fresh
      evaluation := evaluation }
  else
    let running := runningIndex (logicalWidth := logicalWidth) source isFresh
    { commitment := (parent.running PiCCSInputs.phaseOffset).commitment running
      publicInput := (parent.running PiCCSInputs.phaseOffset).publicInput running
      evaluation := evaluation }

/-- The same source view written in the paper's explicit `K + k` order.
This form is used by cross-phase proofs; the production layout continues to
use `sourceInput`. -/
def canonicalSourceInput (source : Fin productionShape.sourceCount) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.InputExpr
      logicalWidth publicFits :=
  let parent := piCcsInterface (logicalWidth := logicalWidth)
    (publicFits := publicFits)
  let output := parent.output PiCCSInputs.phaseOffset
  Fin.addCases
    (fun fresh =>
      { commitment := (parent.fresh PiCCSInputs.phaseOffset).commitment fresh
        publicInput := (parent.fresh PiCCSInputs.phaseOffset).publicInput fresh
        evaluation := {
          eval_K := output.padCoordinate (UnifiedSources.freshSourceIndex fresh)
          eval_A := output.matrixCoordinate
            (UnifiedSources.freshSourceIndex fresh) } })
    (fun running =>
      { commitment := (parent.running PiCCSInputs.phaseOffset).commitment running
        publicInput := (parent.running PiCCSInputs.phaseOffset).publicInput running
        evaluation := {
          eval_K := output.padCoordinate
            (UnifiedSources.runningSourceIndex running)
          eval_A := output.matrixCoordinate
            (UnifiedSources.runningSourceIndex running) } })
    source

/-- The executable source view is exactly the paper's fresh-then-running
partition. -/
theorem sourceInput_eq_canonical
    (source : Fin productionShape.sourceCount) :
    sourceInput (logicalWidth := logicalWidth) (publicFits := publicFits)
        source =
      canonicalSourceInput (logicalWidth := logicalWidth)
        (publicFits := publicFits) source := by
  rcases UnifiedSources.source_eq_fresh_or_running source with
    ⟨fresh, rfl⟩ | ⟨running, rfl⟩
  · have sourceEq : UnifiedSources.freshSourceIndex fresh =
        Fin.castAdd productionShape.runningCount fresh := by
      apply Fin.ext
      rfl
    rw [sourceEq]
    unfold canonicalSourceInput
    rw [Fin.addCases_left]
    have freshPositive : 0 < productionShape.freshCount := by decide
    have freshCount : productionShape.freshCount = 1 := by decide
    have freshEq : fresh = ⟨0, freshPositive⟩ := by
      apply Fin.ext
      have freshBound := fresh.isLt
      omega
    simp [sourceInput, freshPositive, freshEq,
      UnifiedSources.freshSourceIndex]
    constructor <;> congr 1
  · have sourceEq : UnifiedSources.runningSourceIndex running =
        Fin.natAdd productionShape.freshCount running := by
      apply Fin.ext
      rfl
    rw [sourceEq]
    simp [sourceInput, canonicalSourceInput, runningIndex, sourceEq]

/-- The sole production PiRLC parent interface in cumulative source order. -/
def interface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Interface
      logicalWidth publicFits where
  baseOffset := phaseOffset
  initialState := fun _ =>
    piCcsOutputState (logicalWidth := logicalWidth) (publicFits := publicFits)
  point := fun _ =>
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundPoint
      (piCcsInterface (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiCCSInputs.phaseOffset
  input := fun _ => sourceInput (logicalWidth := logicalWidth)
    (publicFits := publicFits)

def samplerInputs :
    NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.InputsAffine
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
          (interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          phaseOffset))
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
        phaseOffset) :=
  ⟨by
    simpa [interface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset] using
        (piCcsOutputState_fresh
          (logicalWidth := logicalWidth) (publicFits := publicFits)).affine⟩

private theorem sourceCommitmentVariable
    (source : Fin productionShape.sourceCount)
    (row : Fin productionProfile.commitmentWidth)
    (lane : Fin ringDegree) :
    ∃ index, (sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).commitment row lane = Expr.var index := by
  unfold sourceInput
  split <;> exact ⟨_, rfl⟩

private theorem sourcePublicInputVariable
    (source : Fin productionShape.sourceCount)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ∃ index, (sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).publicInput column = Expr.var index := by
  unfold sourceInput
  split <;> exact ⟨_, rfl⟩

private theorem sourceEvalKEqOutput
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_K coefficient =
        PiCCSInputs.outputEval_K source coefficient := by
  unfold sourceInput
  split <;> rfl

private theorem sourceEvalAEqOutput
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (sourceInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).evaluation.eval_A matrix coefficient =
        PiCCSInputs.outputEval_A source matrix coefficient := by
  unfold sourceInput
  split <;> rfl

private theorem sourceEvalKComponentVariable
    (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount)
    (cell : Fin
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.cellCount) :
    ∃ index,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.expressionCell
        cell ((sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_K coefficient) =
        Expr.var index := by
  rw [sourceEvalKEqOutput]
  fin_cases cell <;> exact ⟨_, rfl⟩

private theorem sourceEvalAComponentVariable
    (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount)
    (cell : Fin
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.cellCount) :
    ∃ index,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.expressionCell
        cell ((sourceInput (logicalWidth := logicalWidth)
          (publicFits := publicFits) source).evaluation.eval_A matrix
            coefficient) = Expr.var index := by
  rw [sourceEvalAEqOutput]
  fin_cases cell <;> exact ⟨_, rfl⟩

def commitmentProductionInputs :
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.familyInterface
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.commitmentInterface
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
            (interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            phaseOffset)))
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.commitmentOffset
        phaseOffset) := by
  constructor
  · intro source lane
    exact ⟨_, rfl⟩
  · intro source block lane cell
    exact sourceCommitmentVariable source block lane

def publicInputProductionInputs :
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.familyInterface
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.publicInputInterface
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
            (interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            phaseOffset)))
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.publicInputOffset
        phaseOffset) := by
  constructor
  · intro source lane
    exact ⟨_, rfl⟩
  · intro source block lane cell
    exact sourcePublicInputVariable source
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.publicColumn
        block lane)

def evalKProductionInputs :
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.familyInterface
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination.ringInterface
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalKInterface
            (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
              (interface (logicalWidth := logicalWidth)
                (publicFits := publicFits)) phaseOffset))))
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalKOffset phaseOffset) := by
  constructor
  · intro source lane
    exact ⟨_, rfl⟩
  · intro source block lane cell
    exact sourceEvalKComponentVariable source
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination.coefficient lane)
      cell

def evalAProductionInputs :
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination.familyInterface
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.ringInterface
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalAInterface
            (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
              (interface (logicalWidth := logicalWidth)
                (publicFits := publicFits)) phaseOffset))))
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalAOffset phaseOffset) := by
  constructor
  · intro source lane
    exact ⟨_, rfl⟩
  · intro source matrix lane cell
    exact sourceEvalAComponentVariable source matrix
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination.coefficient lane)
      cell

def inputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiRLC.v1_1.InputShapes relation
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits))
      phaseOffset where
  sampler := samplerInputs (logicalWidth := logicalWidth) (publicFits := publicFits)
  commitmentFresh := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCount
      _ _ = 3029400
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCountEqProduction
      _ _ (commitmentProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 150 = 3029400
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.logicalPrivateCount_eq]
  commitmentRows := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCount
      _ _ = 3049596
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCountEqProduction
      _ _ (commitmentProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 151 = 3049596
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination.logicalPrivateCount_eq]
  publicInputFresh := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCount
      _ _ = 688500
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCountEqProduction
      _ _ (publicInputProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 150 = 688500
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.logicalPrivateCount_eq]
  publicInputRows := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCount
      _ _ = 693090
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCountEqProduction
      _ _ (publicInputProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 151 = 693090
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.PublicInputCombination.logicalPrivateCount_eq]
  evalKFresh := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCount
      _ _ = 275400
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCountEqProduction
      _ _ (evalKProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 150 = 275400
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
  evalKRows := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCount
      _ _ = 277236
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCountEqProduction
      _ _ (evalKProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 151 = 277236
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination.logicalPrivateCount_eq]
  evalAFresh := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCount
      _ _ = 3855600
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalFreshColumnCountEqProduction
      _ _ (evalAProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 150 = 3855600
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]
  evalARows := by
    change NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCount
      _ _ = 3881304
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.physicalRowCountEqProduction
      _ _ (evalAProductionInputs (logicalWidth := logicalWidth)
        (publicFits := publicFits))]
    change NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily.logicalPrivateCount
      _ _ * 151 = 3881304
    rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination.logicalPrivateCount_eq]

end NightstreamFPrime.Layout.Stage1.PiRLCInputs
