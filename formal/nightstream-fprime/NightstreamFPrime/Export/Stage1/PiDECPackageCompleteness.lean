import NightstreamFPrime.Export.Stage1.PackageCompleteness
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns constructive completeness for the canonical PiDEC package-row packet.

The semantic completion runs in Lean source-column order. One proved Spartan
copy then writes exactly the full PiDEC logical and R1CS-fresh suffix into
the final package assignment. The compiled-row equality fixes the final row
order; no exporter or Rust code selects it.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECPackageCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def phaseInterface :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Interface
      Data.logicalWidth Data.publicFits :=
  PiDECArithmetic.phaseInterface Data.logicalWidth Data.publicFits

def piRlcInterface :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Interface
      Data.logicalWidth Data.publicFits :=
  PiRLCInputs.interface
    (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)

def runningTransitionInterface :
    NightstreamFPrime.Lifecycle.Stage1.RunningTransition.Interface
      Data.logicalWidth Data.publicFits :=
  RunningTransitionInputs.interface Data.logicalWidth Data.publicFits

theorem targetStart_eq :
    Spartan.sourceToSpartan PiDECInputs.phaseOffset =
      Data.piDecWitnessStart := by
  rfl

theorem targetEnd_eq :
    Data.piDecWitnessStart + Data.piDecWitnessLength =
      Data.runningTransitionWitnessStart := by
  exact Data.piDecPrivateSegments_contiguous.2.2.2.2.2.1

theorem runningTransitionTargetStart_eq :
    Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset =
      Data.runningTransitionWitnessStart := by
  rfl

theorem runningTransitionTargetLength_eq :
    Data.runningTransitionWitnessLength = 275402 := by
  rfl

theorem runningTransitionTargetEnd_eq :
    Data.runningTransitionWitnessStart +
        Data.runningTransitionWitnessLength = Spartan.privateColumnCount := by
  exact Data.piDecPrivateSegments_contiguous.2.2.2.2.2.2

/-- A valid selected running transition constructs every canonical transition
package row and changes only its final private-column interval. -/
theorem completeRunningTransitionRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env)
    (specification :
      NightstreamFPrime.Lifecycle.Stage1.RunningTransition.SpecHolds
        runningTransitionInterface RunningTransitionInputs.phaseOffset
        (Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed Data.runningTransitionWitnessStart
          Data.runningTransitionWitnessLength ∧
        PackageCompleteness.RunningTransitionRowsHold completed := by
  rcases
      NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physical_complete
        relation (Spartan.pullback env) specification with
    ⟨source, sourceAgrees, sourceRows⟩
  let completed := Spartan.copyMappedInterval env source
    RunningTransitionInputs.phaseOffset Data.runningTransitionWitnessLength
  have sourceAgreesExact : AgreesOutside (Spartan.pullback env) source
      RunningTransitionInputs.phaseOffset
      Data.runningTransitionWitnessLength := by
    simpa [runningTransitionTargetLength_eq] using sourceAgrees
  have targetAgrees : AgreesOutside env completed
      Data.runningTransitionWitnessStart
      Data.runningTransitionWitnessLength := by
    rw [← runningTransitionTargetStart_eq]
    exact Spartan.copyMappedInterval_agreesOutside env source
      RunningTransitionInputs.phaseOffset
      Data.runningTransitionWitnessLength
  have startLocal :
      Spartan.piCcsPhaseOffset ≤ RunningTransitionInputs.phaseOffset := by
    norm_num [Spartan.piCcsPhaseOffset,
      RunningTransitionInputs.phaseOffset]
  have targetPrivate :
      Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset +
          Data.runningTransitionWitnessLength ≤
        Spartan.privateColumnCount := by
    rw [runningTransitionTargetStart_eq, runningTransitionTargetEnd_eq]
  have remappedRows : R1CS.RowsHold completed
      (Spartan.remapRows
        (NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physicalRows
          Data.logicalWidth Data.publicFits)) := by
    exact Spartan.remapRows_hold_copyMappedInterval
      (NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physicalRows
        Data.logicalWidth Data.publicFits)
      env source RunningTransitionInputs.phaseOffset
      Data.runningTransitionWitnessLength startLocal targetPrivate
      sourceAgreesExact sourceRows
  have exactRows := RunningTransitionArithmetic.Plan.rows_to_layout
    (RunningTransitionArithmetic.canonicalPlan
      Data.logicalWidth Data.publicFits)
    (RunningTransitionArithmetic.canonicalLayoutPlan
      Data.logicalWidth Data.publicFits)
    (RunningTransitionArithmetic.canonicalPlan_matches
      Data.logicalWidth Data.publicFits)
  refine ⟨completed, targetAgrees, ⟨?_⟩⟩
  rw [exactRows]
  exact remappedRows

/-- A valid semantic PiDEC phase constructs every canonical PiDEC package row
and changes only its declared final private-column interval. -/
theorem completeRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback env))
    (phase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env)) :
    ∃ completed,
      AgreesOutside env completed Data.piDecWitnessStart
          Data.piDecWitnessLength ∧
        PackageCompleteness.PiDECRowsHold completed := by
  rcases
      NightstreamFPrime.Layout.PiDEC.v1_1.physical_complete_production
        relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env) (PiDECInputs.inputShapes relation)
        assumptions phase with
    ⟨source, sourceAgrees, sourceRows⟩
  let completed := Spartan.copyMappedInterval env source
    PiDECInputs.phaseOffset Data.piDecWitnessLength
  have targetAgrees : AgreesOutside env completed
      Data.piDecWitnessStart Data.piDecWitnessLength := by
    rw [← targetStart_eq]
    exact Spartan.copyMappedInterval_agreesOutside env source
      PiDECInputs.phaseOffset Data.piDecWitnessLength
  have startLocal : Spartan.piCcsPhaseOffset ≤ PiDECInputs.phaseOffset := by
    norm_num [Spartan.piCcsPhaseOffset, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild]
  have targetPrivate :
      Spartan.sourceToSpartan PiDECInputs.phaseOffset +
          Data.piDecWitnessLength ≤
        Spartan.privateColumnCount := by
    rw [targetStart_eq, targetEnd_eq]
    have transitionEnd :=
      Data.piDecPrivateSegments_contiguous.2.2.2.2.2.2
    omega
  have remappedRows : R1CS.RowsHold completed
      (Spartan.remapRows
        (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
          phaseInterface PiDECInputs.phaseOffset)) := by
    exact Spartan.remapRows_hold_copyMappedInterval
      (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
        phaseInterface PiDECInputs.phaseOffset)
      env source PiDECInputs.phaseOffset Data.piDecWitnessLength startLocal
      targetPrivate
      sourceAgrees sourceRows
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  refine ⟨completed, targetAgrees, ⟨?_⟩⟩
  rw [exactRows]
  exact remappedRows

private theorem agreesOutside_widen
    {before after : Env} {start length innerStart innerLength : Nat}
    (inner : AgreesOutside before after innerStart innerLength)
    (starts : start ≤ innerStart)
    (ends : innerStart + innerLength ≤ start + length) :
    AgreesOutside before after start length := by
  intro index outside
  apply inner index
  rcases outside with beforeStart | afterEnd
  · exact Or.inl (lt_of_lt_of_le beforeStart starts)
  · exact Or.inr (Nat.le_trans ends afterEnd)

private theorem agreesOutside_trans
    {before middle after : Env} {start length : Nat}
    (left : AgreesOutside before middle start length)
    (right : AgreesOutside middle after start length) :
    AgreesOutside before after start length := by
  intro index outside
  exact (right index outside).trans (left index outside)

theorem targetAgrees_implies_phaseSuffix
    (before after : Env)
    (agrees : AgreesOutside before after Data.piDecWitnessStart
      Data.piDecWitnessLength) :
    AgreesOutside before after PackageCompleteness.phaseSuffixStart
      PackageCompleteness.phaseSuffixLength := by
  apply agreesOutside_widen agrees
  · norm_num [PackageCompleteness.phaseSuffixStart,
      Data.piDecWitnessStart, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild,
      Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart, PiRLCInputs.phaseOffset]
  · rw [targetEnd_eq, PackageCompleteness.phaseSuffixEnd_eq]
    have transitionEnd :=
      Data.piDecPrivateSegments_contiguous.2.2.2.2.2.2
    omega

theorem runningTransitionTargetAgrees_implies_phaseSuffix
    (before after : Env)
    (agrees : AgreesOutside before after
      Data.runningTransitionWitnessStart
      Data.runningTransitionWitnessLength) :
    AgreesOutside before after PackageCompleteness.phaseSuffixStart
      PackageCompleteness.phaseSuffixLength := by
  apply agreesOutside_widen agrees
  · norm_num [PackageCompleteness.phaseSuffixStart,
      Data.runningTransitionWitnessStart,
      RunningTransitionInputs.phaseOffset,
      PiRLCInputs.phaseOffset,
      Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
      Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
      Spartan.piCcsLocalStart]
  · rw [runningTransitionTargetEnd_eq,
      PackageCompleteness.phaseSuffixEnd_eq]

private theorem pullback_agreesBelow_piDec
    (before after : Env)
    (agrees : AgreesOutside before after Data.piDecWitnessStart
      Data.piDecWitnessLength) :
    ∀ index, index < PiDECInputs.phaseOffset →
      Spartan.pullback after index = Spartan.pullback before index := by
  intro index below
  unfold Spartan.pullback
  apply agrees
  rcases Spartan.sourceToSpartan_before_piCcsLocal index
      PiDECInputs.phaseOffset (by
        norm_num [Spartan.piCcsPhaseOffset, PiDECInputs.phaseOffset,
          PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
          PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          PiDECInputs.publicInputWordsPerChild]) below with
    mappedBefore | mappedPublic
  · apply Or.inl
    rw [← targetStart_eq]
    exact mappedBefore
  · apply Or.inr
    rw [targetEnd_eq]
    have transitionEnd :=
      Data.piDecPrivateSegments_contiguous.2.2.2.2.2.2
    exact Nat.le_trans (by omega) mappedPublic.le

private theorem pullback_agreesBelow_runningTransition
    (before after : Env)
    (agrees : AgreesOutside before after
      Data.runningTransitionWitnessStart
      Data.runningTransitionWitnessLength) :
    ∀ index, index < RunningTransitionInputs.phaseOffset →
      Spartan.pullback after index = Spartan.pullback before index := by
  intro index below
  unfold Spartan.pullback
  apply agrees
  rcases Spartan.sourceToSpartan_before_piCcsLocal index
      RunningTransitionInputs.phaseOffset (by
        norm_num [Spartan.piCcsPhaseOffset,
          RunningTransitionInputs.phaseOffset]) below with
    mappedBefore | mappedPublic
  · apply Or.inl
    rw [← runningTransitionTargetStart_eq]
    exact mappedBefore
  · apply Or.inr
    rw [runningTransitionTargetEnd_eq]
    exact mappedPublic.le

private def recursiveRunningBelowPiDec
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback env)) :
    NightstreamFPrime.Lifecycle.Stage1.RunningTransition.RunningBelow
      (RunningTransitionInputs.recursiveRunningExpr
        Data.logicalWidth Data.publicFits)
      PiDECInputs.phaseOffset := by
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface, phaseInterface,
      PiDECArithmetic.phaseInterface] using assumptions.inputs.point coordinate
  · intro source row coefficient
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface, phaseInterface,
      PiDECArithmetic.phaseInterface] using
      assumptions.inputs.messageCommitment
        (RunningTransitionInputs.childOfRunning source) row coefficient
  · intro source column
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface, phaseInterface,
      PiDECArithmetic.phaseInterface] using
      assumptions.inputs.digit
        (RunningTransitionInputs.childOfRunning source)
        (RunningTransitionInputs.digitCoordinate column)
  · intro source coefficient
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface, phaseInterface,
      PiDECArithmetic.phaseInterface] using
      assumptions.inputs.messageEval_K
        (RunningTransitionInputs.childOfRunning source) coefficient
  · intro source matrix coefficient
    simpa [RunningTransitionInputs.recursiveRunningExpr,
      RunningTransitionInputs.piDecInterface, phaseInterface,
      PiDECArithmetic.phaseInterface] using
      assumptions.inputs.messageEval_A
        (RunningTransitionInputs.childOfRunning source) matrix coefficient

private def outputRunningBelowPiDec :
    NightstreamFPrime.Lifecycle.Stage1.RunningTransition.RunningBelow
      (RunningTransitionInputs.outputRunningExpr
        Data.logicalWidth Data.publicFits)
      PiDECInputs.phaseOffset := by
  apply
    (RunningTransitionInputs.outputRunningBelowOutputDigestStart
      Data.logicalWidth Data.publicFits).mono
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
    PriorStateHash.publicWidth_eq, PiDECInputs.phaseOffset,
    PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
    PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
    PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
    PiDECInputs.publicInputWordsPerChild]

private theorem transitionSpec_of_piDecAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (before after : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback before))
    (agrees : AgreesOutside before after Data.piDecWitnessStart
      Data.piDecWitnessLength)
    (specification :
      NightstreamFPrime.Lifecycle.Stage1.RunningTransition.SpecHolds
        runningTransitionInterface RunningTransitionInputs.phaseOffset
        (Spartan.pullback before)) :
    NightstreamFPrime.Lifecycle.Stage1.RunningTransition.SpecHolds
      runningTransitionInterface RunningTransitionInputs.phaseOffset
      (Spartan.pullback after) := by
  let sourceBefore := Spartan.pullback before
  let sourceAfter := Spartan.pullback after
  have sourceAgrees : ∀ index, index < PiDECInputs.phaseOffset →
      sourceAfter index = sourceBefore index :=
    pullback_agreesBelow_piDec before after agrees
  have iterationBelow :
      (runningTransitionInterface.iteration
        RunningTransitionInputs.phaseOffset).VarsBelow
          PiDECInputs.phaseOffset := by
    norm_num [runningTransitionInterface, RunningTransitionInputs.interface,
      RunningTransitionInputs.iterationExpr,
      RunningTransitionInputs.iterationWordIndex, Expr.VarsBelow,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild,
      PilotProduction.priorPreimageStart]
  have recursiveBelow := recursiveRunningBelowPiDec relation before assumptions
  have iterationEq :
      NightstreamFPrime.Lifecycle.Stage1.RunningTransition.iterationValue
          runningTransitionInterface RunningTransitionInputs.phaseOffset
          sourceAfter =
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.iterationValue
          runningTransitionInterface RunningTransitionInputs.phaseOffset
          sourceBefore :=
    Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      sourceAfter sourceBefore iterationBelow sourceAgrees
  have initialStateEq : ∀ index,
      (runningTransitionInterface.initialState
          RunningTransitionInputs.phaseOffset index).eval sourceAfter =
        (runningTransitionInterface.initialState
          RunningTransitionInputs.phaseOffset index).eval sourceBefore := by
    intro index
    apply Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      sourceAfter sourceBefore
    · simp [runningTransitionInterface, RunningTransitionInputs.interface,
        RunningTransitionInputs.initialStateExpr, Expr.VarsBelow,
        RunningTransitionInputs.initialStateWordStart,
        PilotProduction.priorPreimageStart]
      have bound := index.isLt
      norm_num [
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.stateWordCount,
        PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild] at bound ⊢
      omega
    · exact sourceAgrees
  have currentStateEq : ∀ index,
      (runningTransitionInterface.currentState
          RunningTransitionInputs.phaseOffset index).eval sourceAfter =
        (runningTransitionInterface.currentState
          RunningTransitionInputs.phaseOffset index).eval sourceBefore := by
    intro index
    apply Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      sourceAfter sourceBefore
    · simp [runningTransitionInterface, RunningTransitionInputs.interface,
        RunningTransitionInputs.currentStateExpr, Expr.VarsBelow,
        RunningTransitionInputs.currentStateWordStart,
        PilotProduction.priorPreimageStart]
      have bound := index.isLt
      norm_num [
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.stateWordCount,
        PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
        PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild] at bound ⊢
      omega
    · exact sourceAgrees
  have recursiveEq : ∀ index,
      (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord
          (runningTransitionInterface.recursive
            RunningTransitionInputs.phaseOffset) index).eval sourceAfter =
        (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord
          (runningTransitionInterface.recursive
            RunningTransitionInputs.phaseOffset) index).eval sourceBefore := by
    intro index
    apply Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      sourceAfter sourceBefore
    · simpa [runningTransitionInterface, RunningTransitionInputs.interface]
        using
          NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord_varsBelow
            (RunningTransitionInputs.recursiveRunningExpr
              Data.logicalWidth Data.publicFits)
            PiDECInputs.phaseOffset recursiveBelow index
    · exact sourceAgrees
  have outputEq : ∀ index,
      (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord
          (runningTransitionInterface.output
            RunningTransitionInputs.phaseOffset) index).eval sourceAfter =
        (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord
          (runningTransitionInterface.output
            RunningTransitionInputs.phaseOffset) index).eval sourceBefore := by
    intro index
    apply Expr.eval_eq_of_agree_below _ PiDECInputs.phaseOffset
      sourceAfter sourceBefore
    · simpa [runningTransitionInterface, RunningTransitionInputs.interface]
        using
          NightstreamFPrime.Lifecycle.Stage1.RunningTransition.runningWord_varsBelow
            (RunningTransitionInputs.outputRunningExpr
              Data.logicalWidth Data.publicFits)
            PiDECInputs.phaseOffset outputRunningBelowPiDec index
    · exact sourceAgrees
  refine { initialState := ?_, base := ?_, recursive := ?_ }
  · intro iterationZero index
    have beforeZero :
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.iterationValue
            runningTransitionInterface RunningTransitionInputs.phaseOffset
            sourceBefore = 0 := by
      rw [← iterationEq]
      exact iterationZero
    exact (initialStateEq index).trans
      ((specification.initialState beforeZero index).trans
        (currentStateEq index).symm)
  · intro iterationZero index
    have beforeZero :
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.iterationValue
            runningTransitionInterface RunningTransitionInputs.phaseOffset
            sourceBefore = 0 := by
      rw [← iterationEq]
      exact iterationZero
    exact (outputEq index).trans (specification.base beforeZero index)
  · intro iterationNonzero index
    have beforeNonzero :
        NightstreamFPrime.Lifecycle.Stage1.RunningTransition.iterationValue
            runningTransitionInterface RunningTransitionInputs.phaseOffset
            sourceBefore ≠ 0 := by
      intro beforeZero
      apply iterationNonzero
      rw [iterationEq, beforeZero]
    exact (outputEq index).trans
      ((specification.recursive beforeNonzero index).trans
        (recursiveEq index).symm)

theorem piRlcPhysicalRows_varsBelow
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback env))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback env)) :
    ∀ row ∈ NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset,
      row.VarsBelow PiDECInputs.phaseOffset := by
  have physicalEnd :
      NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount relation
          piRlcInterface PiRLCInputs.phaseOffset ≤
        PiDECInputs.phaseOffset := by
    rw [NightstreamFPrime.Layout.PiRLC.v1_1.physicalColumnCount_eq_production
      relation piRlcInterface PiRLCInputs.phaseOffset
      (PiRLCInputs.inputShapes relation)]
    norm_num [PiRLCInputs.phaseOffset, PiDECInputs.phaseOffset,
      PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
      PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
      PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
      PiDECInputs.publicInputWordsPerChild]
  intro row member
  exact (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows_varsBelow_of_phase
    relation ajtai piRlcInterface PiRLCInputs.phaseOffset
    (Spartan.pullback env) assumptions phase row member).mono row physicalEnd

theorem piRlcPhysicalRows_of_piDecAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (before after : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback before))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback before))
    (agrees : AgreesOutside before after Data.piDecWitnessStart
      Data.piDecWitnessLength)
    (holds : R1CS.RowsHold before (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset))) :
    R1CS.RowsHold after (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset)) := by
  have sourceHolds := (Spartan.remapRows_hold before _).mp holds
  have sourceAfter := R1CS.rowsHold_of_agree_below
    (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
      piRlcInterface PiRLCInputs.phaseOffset)
    PiDECInputs.phaseOffset (Spartan.pullback before)
    (Spartan.pullback after)
    (piRlcPhysicalRows_varsBelow relation ajtai before assumptions phase)
    (pullback_agreesBelow_piDec before after agrees) sourceHolds
  exact (Spartan.remapRows_hold after _).mpr sourceAfter

theorem piRlcPhysicalRows_of_transitionAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (scopeEnv before after : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback scopeEnv))
    (phase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback scopeEnv))
    (agrees : AgreesOutside before after
      Data.runningTransitionWitnessStart
      Data.runningTransitionWitnessLength)
    (holds : R1CS.RowsHold before (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset))) :
    R1CS.RowsHold after (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset)) := by
  have sourceHolds := (Spartan.remapRows_hold before _).mp holds
  have sourceAgrees : ∀ index, index < PiDECInputs.phaseOffset →
      Spartan.pullback after index = Spartan.pullback before index := by
    intro index below
    apply pullback_agreesBelow_runningTransition before after agrees index
    exact lt_of_lt_of_le below (by
      norm_num [PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
        PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
        PiDECInputs.commitmentWordsPerChild,
        PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
        PiDECInputs.publicInputWordsPerChild,
        RunningTransitionInputs.phaseOffset])
  have sourceAfter := R1CS.rowsHold_of_agree_below
    (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
      piRlcInterface PiRLCInputs.phaseOffset)
    PiDECInputs.phaseOffset (Spartan.pullback before)
    (Spartan.pullback after)
    (piRlcPhysicalRows_varsBelow relation ajtai scopeEnv assumptions phase)
    sourceAgrees sourceHolds
  exact (Spartan.remapRows_hold after _).mpr sourceAfter

theorem piDecRows_of_transitionAgreesOutside
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (scopeEnv before after : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback scopeEnv))
    (phase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback scopeEnv))
    (agrees : AgreesOutside before after
      Data.runningTransitionWitnessStart
      Data.runningTransitionWitnessLength)
    (holds : PackageCompleteness.PiDECRowsHold before) :
    PackageCompleteness.PiDECRowsHold after := by
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  have remappedBefore := holds.arithmetic
  rw [exactRows] at remappedBefore
  have sourceBefore := (Spartan.remapRows_hold before _).mp remappedBefore
  have sourceScope :=
    NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows_varsBelow_of_phase
      relation ajtai phaseInterface PiDECInputs.phaseOffset
      (Spartan.pullback scopeEnv) assumptions phase
  have endpoint :
      NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount relation
          phaseInterface PiDECInputs.phaseOffset =
        RunningTransitionInputs.phaseOffset := by
    rw [NightstreamFPrime.Layout.PiDEC.v1_1.physicalColumnCount_eq_production
        relation phaseInterface
        PiDECInputs.phaseOffset (PiDECInputs.inputShapes relation)]
    norm_num [phaseInterface, PiDECArithmetic.phaseInterface,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
      PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
      PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
      RunningTransitionInputs.phaseOffset]
  rw [endpoint] at sourceScope
  have sourceAfter := R1CS.rowsHold_of_agree_below
    (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRows relation
      phaseInterface PiDECInputs.phaseOffset)
    RunningTransitionInputs.phaseOffset (Spartan.pullback before)
    (Spartan.pullback after) sourceScope
    (pullback_agreesBelow_runningTransition before after agrees) sourceBefore
  refine ⟨?_⟩
  rw [exactRows]
  exact (Spartan.remapRows_hold after _).mpr sourceAfter

/-- Completing PiDEC and then the selected running transition after a valid
Pilot/PiCCS/PiRLC prefix produces one assignment satisfying every row of the
current canonical package. -/
theorem completePackageRows
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (env : Env)
    (piRlcAssumptions :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.Assumptions relation
        piRlcInterface PiRLCInputs.phaseOffset (Spartan.pullback env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai piRlcInterface PiRLCInputs.phaseOffset
        (Spartan.pullback env))
    (piDecAssumptions :
      NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        phaseInterface PiDECInputs.phaseOffset (Spartan.pullback env))
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai phaseInterface PiDECInputs.phaseOffset
        (Spartan.pullback env))
    (runningTransition :
      NightstreamFPrime.Lifecycle.Stage1.RunningTransition.SpecHolds
        runningTransitionInterface RunningTransitionInputs.phaseOffset
        (Spartan.pullback env))
    (pilotChains : ∀ chain ∈ [Data.priorChain, Data.outputChain],
      NightstreamFPrime.Export.Package.HashChainHolds
        (Data.circuitPackage ()) chain env)
    (pilotInstructions : ∀ instruction ∈
      Data.liftPilotInstructions
        (NightstreamFPrime.Export.PilotData.witnessInstructions ()),
      instruction.Holds env)
    (pilotAssertions : ∀ row ∈
      Data.liftPilotRows (NightstreamFPrime.Export.PilotData.assertionRows ()),
        row.Holds env)
    (piCcs : PackageCompleteness.PiCCSRowsHold env)
    (piRlcPhysical : R1CS.RowsHold env (Spartan.remapRows
      (NightstreamFPrime.Layout.PiRLC.v1_1.physicalRows relation
        piRlcInterface PiRLCInputs.phaseOffset))) :
    ∃ completed,
      AgreesOutside env completed Data.piDecWitnessStart
          (Data.piDecWitnessLength +
            Data.runningTransitionWitnessLength) ∧
        (Data.circuitPackage ()).RowsHold completed := by
  rcases completeRows relation ajtai env piDecAssumptions piDecPhase with
    ⟨afterPiDec, piDecAgrees, piDec⟩
  have transitionAfterPiDec := transitionSpec_of_piDecAgreesOutside
    relation env afterPiDec piDecAssumptions piDecAgrees runningTransition
  rcases completeRunningTransitionRows relation afterPiDec
      transitionAfterPiDec with
    ⟨completed, transitionAgrees, transitionRows⟩
  have totalAgrees : AgreesOutside env completed Data.piDecWitnessStart
      (Data.piDecWitnessLength +
        Data.runningTransitionWitnessLength) := by
    have transitionAgreesAtEnd : AgreesOutside afterPiDec completed
        (Data.piDecWitnessStart + Data.piDecWitnessLength)
        Data.runningTransitionWitnessLength := by
      rw [targetEnd_eq]
      exact transitionAgrees
    exact piDecAgrees.append transitionAgreesAtEnd
  have piDecSuffixAgrees := targetAgrees_implies_phaseSuffix
    env afterPiDec piDecAgrees
  have transitionSuffixAgrees :=
    runningTransitionTargetAgrees_implies_phaseSuffix
      afterPiDec completed transitionAgrees
  have suffixAgrees := agreesOutside_trans
    piDecSuffixAgrees transitionSuffixAgrees
  have piRlcPhysicalAfterPiDec := piRlcPhysicalRows_of_piDecAgreesOutside
    relation ajtai env afterPiDec piRlcAssumptions piRlcPhase piDecAgrees
    piRlcPhysical
  have piRlcPhysicalAfter := piRlcPhysicalRows_of_transitionAgreesOutside
    relation ajtai env afterPiDec completed piRlcAssumptions piRlcPhase
    transitionAgrees piRlcPhysicalAfterPiDec
  have packets := PiRLCPackageCompleteness.remappedPhysicalRows_imply_packets
    relation completed piRlcPhysicalAfter
  have piRlc := PackageCompleteness.piRlcRowsHold_of_packets completed packets
  have piDecAfter := piDecRows_of_transitionAgreesOutside
    relation ajtai env afterPiDec completed piDecAssumptions piDecPhase
    transitionAgrees piDec
  refine ⟨completed, totalAgrees,
    PackageCompleteness.rowsHold_of_phaseRows
      completed ?_ ?_ ?_ ?_ piRlc piDecAfter transitionRows⟩
  · exact PackageCompleteness.pilotHashChains_of_piRlcAgreesOutside
      env completed suffixAgrees pilotChains
  · exact PackageCompleteness.pilotWitnessInstructions_of_piRlcAgreesOutside
      env completed suffixAgrees pilotInstructions
  · exact PackageCompleteness.pilotAssertionRows_of_piRlcAgreesOutside
      env completed suffixAgrees pilotAssertions
  · exact PackageCompleteness.piCcsRows_of_piRlcAgreesOutside relation
      env completed suffixAgrees piCcs

end NightstreamFPrime.Export.Stage1.PiDECPackageCompleteness
