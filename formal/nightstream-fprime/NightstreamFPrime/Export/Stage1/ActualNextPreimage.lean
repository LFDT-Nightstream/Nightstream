import NightstreamFPrime.Export.Stage1.ActualPreimageFraming
import NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan

/-!
Owns the row-derived counter and initial-state equations on the actual pilot
preimages. The increment is an equality in Goldilocks; this module makes no
natural-number non-wrap claim and assumes no canonical assignment encoding.
-/

namespace NightstreamFPrime.Export.Stage1.ActualNextPreimage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

/-- The five accepted next-preimage rows increment the actual hashed counter
in the field and preserve all four actual hashed initial-state words. -/
theorem rowsZero_implies_actualNextPreimage
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (NextPreimageDirectPlan.plan geometry).RowsZero assignment) :
    ActualPreimageFraming.outputState geometry assignment
        RunningTransitionInputs.iterationWordIndex =
      ActualPreimageFraming.priorState geometry assignment
          RunningTransitionInputs.iterationWordIndex + 1 ∧
    ∀ index : Lifecycle.Stage1.RunningTransition.StateIndex,
      ActualPreimageFraming.outputState geometry assignment
          (RunningTransitionInputs.initialStateWordStart + index.val) =
        ActualPreimageFraming.priorState geometry assignment
          (RunningTransitionInputs.initialStateWordStart + index.val) := by
  let env := PiCCSAssignmentSoundness.decodedEnv geometry assignment
  have sourcePreserves := PiCCSAssignmentSoundness.decodedEnv_preserves geometry assignment
  have preserves : ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((NextPreimageDirectPlan.inputs geometry).sourceMap index) assignment env
      (NextPreimageDirectPlan.program.row index)
      (NextPreimageDirectPlan.program.bounded index) := by
    intro index
    refine ⟨?_, ?_, ?_⟩
    · intro term member
      exact sourcePreserves ⟨term.1,
        (NextPreimageDirectPlan.program.bounded index).1 term member⟩
    · intro term member
      exact sourcePreserves ⟨term.1,
        (NextPreimageDirectPlan.program.bounded index).2.1 term member⟩
    · intro term member
      exact sourcePreserves ⟨term.1,
        (NextPreimageDirectPlan.program.bounded index).2.2 term member⟩
  have compiledRows :
      (NextPreimageDirectPlan.program.compile
        (NextPreimageDirectPlan.inputs geometry)).toPlan.RowsZero assignment := rows
  have held := (OrdinarySourcePlan.Program.rowsZero_iff
    NextPreimageDirectPlan.program (NextPreimageDirectPlan.inputs geometry)
    assignment env one preserves).mp compiledRows
  have sourceRows := (NextPreimageDirectPlan.program_holds_iff_rowsHold env).mp held
  have mapped := NextPreimagePackage.sourceRows_imply_spec env sourceRows
  have specification := (NextPreimageInputs.spartanSpec_iff_sourceSpec
    NextPreimagePackage.privateStart env).mp mapped
  constructor
  · have bounded : RunningTransitionInputs.iterationWordIndex <
        PilotProduction.stateHashWords := by
      rw [PilotProduction.stateHashWords_eq]
      norm_num [RunningTransitionInputs.iterationWordIndex]
    rw [← ActualPreimageFraming.outputWord_eq geometry assignment
        ⟨RunningTransitionInputs.iterationWordIndex, bounded⟩,
      ← ActualPreimageFraming.priorWord_eq geometry assignment
        ⟨RunningTransitionInputs.iterationWordIndex, bounded⟩]
    exact specification.iteration
  · intro index
    have bounded : RunningTransitionInputs.initialStateWordStart + index.val <
        PilotProduction.stateHashWords := by
      have indexBound := index.isLt
      rw [PilotProduction.stateHashWords_eq]
      norm_num [RunningTransitionInputs.initialStateWordStart,
        Lifecycle.Stage1.RunningTransition.stateWordCount] at indexBound ⊢
      omega
    rw [← ActualPreimageFraming.outputWord_eq geometry assignment
        ⟨RunningTransitionInputs.initialStateWordStart + index.val, bounded⟩,
      ← ActualPreimageFraming.priorWord_eq geometry assignment
        ⟨RunningTransitionInputs.initialStateWordStart + index.val, bounded⟩]
    simpa only [NextPreimageInputs.sourceInterface, Expr.eval,
      NextPreimageInputs.outputInitialStateSource,
      NextPreimageInputs.priorInitialStateSource, Nat.add_assoc] using
      specification.initialState index

/-- The decoded prior counter, incremented as a natural number and encoded
back into the field, is the actual next counter word. All decoded initial
state words are preserved. This does not assert a natural-number non-wrap
property for the decoded output counter. -/
theorem rowsZero_implies_decodedHeaders
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (NextPreimageDirectPlan.plan geometry).RowsZero assignment) :
    natWord (StateDecoder.iteration
        (ActualPreimageFraming.priorState geometry assignment) + 1) =
        ActualPreimageFraming.outputState geometry assignment
          RunningTransitionInputs.iterationWordIndex ∧
      StateDecoder.initialState (ActualPreimageFraming.outputState geometry assignment) =
        StateDecoder.initialState (ActualPreimageFraming.priorState geometry assignment) := by
  have equations := rowsZero_implies_actualNextPreimage geometry assignment one rows
  constructor
  · simpa only [StateDecoder.iteration, StateDecoder.natWord_val_add_one]
      using equations.1.symm
  · unfold StateDecoder.initialState
    apply StateDecoder.slice_congr
    intro index
    exact equations.2 index

end NightstreamFPrime.Export.Stage1.ActualNextPreimage
