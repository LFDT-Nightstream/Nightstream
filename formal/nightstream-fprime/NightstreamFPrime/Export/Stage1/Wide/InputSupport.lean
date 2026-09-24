import NightstreamFPrime.Export.Stage1.Wide.FormSupport
import NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness

/-! The concrete Stage 1 PiRLC input forms read only common retained values.
They cannot read a discarded sampler slot or the allocation they complete. -/

namespace NightstreamFPrime.Export.Stage1.Wide.InputSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

private theorem shared_block (program : RetainedLayout.Program) {sourceWidth : Nat}
    (retained : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount)
    (lower : RetainedLayout.sharedStart program ≤ start)
    (upper : start + retained.coordinateCount ≤ RetainedLayout.sharedEnd program) :
    Supported (Common program) (retained.form start fits slot) := by
  apply block
  intro column low high
  exact Or.inr ⟨by omega, by omega⟩

theorem piCcsOutput (program : RetainedLayout.Program)
    (geometry : PiCCSPoseidonPlan.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8) :
    Supported (Common program) (PiCCSPoseidonPlan.outputState geometry invocation lane) := by
  apply external
  intro outputLane
  apply block
  intro column _ high
  apply Or.inl
  have endpoint : PiCCSPoseidonPlan.retainedStart program +
      (PiCCSPoseidonPlan.retainedBlock program).coordinateCount = RetainedLayout.hashEnd program := by
    rw [PiCCSPoseidonPlan.retainedBlock_coordinateCount]
    exact (LaterPoseidonRetainedBlocks.samplerStart_eq program).symm
  exact lt_of_lt_of_eq high endpoint

theorem location (program : RetainedLayout.Program)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (value : PiCCSOrdinaryDirectPlan.Location) :
    Supported (Common program) (value.form geometry) := by
  let within : PiCCSOrdinaryRetainedGeometry.Geometry program (RetainedLayout.sharedEnd program) :=
    ⟨by rw [PiCCSOrdinaryRetainedGeometry.completeLogicalWidth_eq,
      (RetainedLayout.boundaries program).2.2.1]; decide⟩
  have priorLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.priorInputStart program := Nat.le_refl _
  have outputLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.outputInputStart program := by
    change RetainedLayout.sharedStart program ≤ RetainedLayout.sharedStart program + _
    omega
  have freshLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program := by
    rw [(RetainedLayout.boundaries program).2.1]
    unfold PiCCSOrdinaryRetainedGeometry.freshPublicInputStart PiCCSOrdinaryRetainedGeometry.prefixLogicalWidth
    rw [RunningTransitionReducedRetainedBlocks.nextStart_eq]
    decide
  have contextLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.expectedContextStart program := by
    unfold PiCCSOrdinaryRetainedGeometry.expectedContextStart
      PiCCSOrdinaryRetainedGeometry.outputLastStart PiCCSOrdinaryRetainedGeometry.priorLastStart
    omega
  have proofLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.proofLogicalStart program := by
    unfold PiCCSOrdinaryRetainedGeometry.proofLogicalStart
    omega
  have scratchLower : RetainedLayout.sharedStart program ≤
      PiCCSOrdinaryRetainedGeometry.freshStart program := by
    unfold PiCCSOrdinaryRetainedGeometry.freshStart PiCCSOrdinaryRetainedGeometry.outputEndpointStart
    omega
  cases value <;> dsimp only [PiCCSOrdinaryDirectPlan.Location.form]
  case priorInput index =>
    exact shared_block program _ _ _ _ priorLower (PiCCSOrdinaryRetainedGeometry.priorInputFits within)
  case freshPublicInput index =>
    exact shared_block program _ _ _ _ freshLower (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits within)
  case outputInput index =>
    exact shared_block program _ _ _ _ outputLower (PiCCSOrdinaryRetainedGeometry.outputInputFits within)
  case expectedContext index =>
    exact shared_block program _ _ _ _ contextLower (PiCCSOrdinaryRetainedGeometry.expectedContextFits within)
  case proofLogical index =>
    split
    · exact shared_block program _ _ _ _ proofLower (PiCCSOrdinaryRetainedGeometry.proofLogicalFits within)
    · split
      · exact piCcsOutput program _ _ _
      · exact shared_block program _ _ _ _ proofLower (PiCCSOrdinaryRetainedGeometry.proofLogicalFits within)
  case fresh index =>
    exact shared_block program _ _ _ _ scratchLower (PiCCSOrdinaryRetainedGeometry.freshFits within)


end NightstreamFPrime.Export.Stage1.Wide.InputSupport
