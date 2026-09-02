import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData
import NightstreamFPrime.Lifecycle.Stage1.NextPreimage

/-!
Owns the zero-copy column map for HyperNova's next-preimage counter and
initial-state equations. The raw interface uses the two pilot preimage blocks.
The Spartan interface applies the one proved source-column permutation.
-/

namespace NightstreamFPrime.Layout.Stage1.NextPreimageInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.Stage1

def priorIterationSource : Nat :=
  PilotProduction.priorPreimageStart + RunningTransitionInputs.iterationWordIndex

def outputIterationSource : Nat :=
  PilotProduction.outputPreimageStart +
    RunningTransitionInputs.iterationWordIndex

def priorInitialStateSource
    (index : RunningTransition.StateIndex) : Nat :=
  PilotProduction.priorPreimageStart +
    RunningTransitionInputs.initialStateWordStart + index.val

def outputInitialStateSource
    (index : RunningTransition.StateIndex) : Nat :=
  PilotProduction.outputPreimageStart +
    RunningTransitionInputs.initialStateWordStart + index.val

def sourceInterface : NextPreimage.Interface where
  priorIteration := fun _ => .var priorIterationSource
  outputIteration := fun _ => .var outputIterationSource
  priorInitialState := fun _ index => .var (priorInitialStateSource index)
  outputInitialState := fun _ index => .var (outputInitialStateSource index)

def spartanInterface : NextPreimage.Interface where
  priorIteration := fun _ => .var (Spartan.sourceToSpartan priorIterationSource)
  outputIteration := fun _ => .var (Spartan.sourceToSpartan outputIterationSource)
  priorInitialState := fun _ index =>
    .var (Spartan.sourceToSpartan (priorInitialStateSource index))
  outputInitialState := fun _ index =>
    .var (Spartan.sourceToSpartan (outputInitialStateSource index))

private theorem priorIterationSource_supported :
    PiCCSOrdinarySourceSupport.Source priorIterationSource := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_prior
  constructor
  · unfold priorIterationSource
    omega
  · unfold priorIterationSource RunningTransitionInputs.iterationWordIndex
    rw [PilotProduction.stateHashWords_eq]
    omega

private theorem outputIterationSource_supported :
    PiCCSOrdinarySourceSupport.Source outputIterationSource := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_output
  constructor
  · unfold outputIterationSource
    omega
  · unfold outputIterationSource RunningTransitionInputs.iterationWordIndex
    rw [PilotProduction.stateHashWords_eq]
    omega

private theorem priorInitialStateSource_supported
    (index : RunningTransition.StateIndex) :
    PiCCSOrdinarySourceSupport.Source (priorInitialStateSource index) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_prior
  constructor
  · unfold priorInitialStateSource
    omega
  · have bound := index.isLt
    unfold priorInitialStateSource
    rw [PilotProduction.stateHashWords_eq]
    norm_num [RunningTransitionInputs.initialStateWordStart,
      RunningTransition.stateWordCount] at bound ⊢
    omega

private theorem outputInitialStateSource_supported
    (index : RunningTransition.StateIndex) :
    PiCCSOrdinarySourceSupport.Source (outputInitialStateSource index) := by
  apply PiCCSOrdinarySourceSupport.external_source
  apply PiCCSOrdinarySourceSupport.external_output
  constructor
  · unfold outputInitialStateSource
    omega
  · have bound := index.isLt
    unfold outputInitialStateSource
    rw [PilotProduction.stateHashWords_eq]
    norm_num [RunningTransitionInputs.initialStateWordStart,
      RunningTransition.stateWordCount] at bound ⊢
    omega

private theorem priorIterationSource_lt :
    priorIterationSource < Spartan.SourceColumnCount := by
  exact PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    priorIterationSource_supported

private theorem outputIterationSource_lt :
    outputIterationSource < Spartan.SourceColumnCount := by
  exact PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    outputIterationSource_supported

private theorem priorInitialStateSource_lt
    (index : RunningTransition.StateIndex) :
    priorInitialStateSource index < Spartan.SourceColumnCount := by
  exact PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (priorInitialStateSource_supported index)

private theorem outputInitialStateSource_lt
    (index : RunningTransition.StateIndex) :
    outputInitialStateSource index < Spartan.SourceColumnCount := by
  exact PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (outputInitialStateSource_supported index)

theorem priorIterationTarget :
    PiCCSOrdinarySourceSupport.Target
      (Spartan.sourceToSpartan priorIterationSource) := by
  exact PiCCSOrdinarySourceSupport.source_target _
    priorIterationSource_supported

theorem outputIterationTarget :
    PiCCSOrdinarySourceSupport.Target
      (Spartan.sourceToSpartan outputIterationSource) := by
  exact PiCCSOrdinarySourceSupport.source_target _
    outputIterationSource_supported

theorem priorInitialStateTarget (index : RunningTransition.StateIndex) :
    PiCCSOrdinarySourceSupport.Target
      (Spartan.sourceToSpartan (priorInitialStateSource index)) := by
  exact PiCCSOrdinarySourceSupport.source_target _
    (priorInitialStateSource_supported index)

theorem outputInitialStateTarget (index : RunningTransition.StateIndex) :
    PiCCSOrdinarySourceSupport.Target
      (Spartan.sourceToSpartan (outputInitialStateSource index)) := by
  exact PiCCSOrdinarySourceSupport.source_target _
    (outputInitialStateSource_supported index)

theorem spartanConstraints_varsSatisfy (offset : Nat) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (NextPreimage.main spartanInterface) offset),
      expression.VarsSatisfy PiCCSOrdinarySourceSupport.Target := by
  intro expression member
  rw [NextPreimage.main_ops, NextPreimage.flatConstraints_opsAt,
    NextPreimage.assertions, List.mem_cons] at member
  rcases member with rfl | initialMember
  · exact Expr.VarsSatisfy.sub _ _ _ (by
      simpa [spartanInterface, Expr.VarsSatisfy] using outputIterationTarget)
      (Expr.VarsSatisfy.add _ _ _ (by
        simpa [spartanInterface, Expr.VarsSatisfy] using priorIterationTarget)
        trivial)
  · rw [NextPreimage.initialStateAssertions, List.mem_map] at initialMember
    rcases initialMember with ⟨index, _indexMember, rfl⟩
    exact Expr.VarsSatisfy.sub _ _ _ (by
      simpa [spartanInterface, Expr.VarsSatisfy] using
        outputInitialStateTarget index) (by
      simpa [spartanInterface, Expr.VarsSatisfy] using
        priorInitialStateTarget index)

theorem sourceAssumptions (env : Env) :
    NextPreimage.Assumptions sourceInterface
      RunningTransitionInputs.phaseOffset env := by
  refine {
    priorIteration := ?_
    outputIteration := ?_
    priorInitialState := fun index => ?_
    outputInitialState := fun index => ?_ }
  · simp only [sourceInterface, Expr.VarsBelow]
    norm_num [priorIterationSource, RunningTransitionInputs.phaseOffset,
      RunningTransitionInputs.iterationWordIndex,
      PilotProduction.priorPreimageStart]
  · simp only [sourceInterface, Expr.VarsBelow]
    norm_num [outputIterationSource, RunningTransitionInputs.phaseOffset,
      RunningTransitionInputs.iterationWordIndex,
      PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq,
      Lifecycle.PriorStateHash.publicWidth, Spec.ringDegree,
      Lifecycle.PaperAlgebra.publicRingColumns]
  · simp only [sourceInterface, Expr.VarsBelow]
    have bound := index.isLt
    norm_num [priorInitialStateSource, RunningTransitionInputs.phaseOffset,
      RunningTransitionInputs.initialStateWordStart,
      RunningTransition.stateWordCount, PilotProduction.priorPreimageStart]
      at bound ⊢
    omega
  · simp only [sourceInterface, Expr.VarsBelow]
    have bound := index.isLt
    norm_num [outputInitialStateSource, RunningTransitionInputs.phaseOffset,
      RunningTransitionInputs.initialStateWordStart,
      RunningTransition.stateWordCount, PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq,
      Lifecycle.PriorStateHash.publicWidth, Spec.ringDegree,
      Lifecycle.PaperAlgebra.publicRingColumns]
      at bound ⊢
    omega

theorem spartanAssumptions (offset : Nat) (env : Env)
    (fits : Spartan.spartanColumnCount ≤ offset) :
    NextPreimage.Assumptions spartanInterface offset env := by
  refine {
    priorIteration := ?_
    outputIteration := ?_
    priorInitialState := fun index => ?_
    outputInitialState := fun index => ?_ }
  all_goals simp only [spartanInterface, Expr.VarsBelow]
  · exact lt_of_lt_of_le
      (Spartan.sourceToSpartan_lt _ priorIterationSource_lt) fits
  · exact lt_of_lt_of_le
      (Spartan.sourceToSpartan_lt _ outputIterationSource_lt) fits
  · exact lt_of_lt_of_le
      (Spartan.sourceToSpartan_lt _ (priorInitialStateSource_lt index)) fits
  · exact lt_of_lt_of_le
      (Spartan.sourceToSpartan_lt _ (outputInitialStateSource_lt index)) fits

theorem spartanSpec_iff_sourceSpec (offset : Nat) (env : Env) :
    NextPreimage.SpecHolds spartanInterface offset env ↔
      NextPreimage.SpecHolds sourceInterface offset (Spartan.pullback env) := by
  constructor
  · intro specification
    refine {
      iteration := ?_
      initialState := fun index => ?_ }
    · simpa [spartanInterface, sourceInterface, Spartan.pullback] using
        specification.iteration
    · simpa [spartanInterface, sourceInterface, Spartan.pullback] using
        specification.initialState index
  · intro specification
    refine {
      iteration := ?_
      initialState := fun index => ?_ }
    · simpa [spartanInterface, sourceInterface, Spartan.pullback] using
        specification.iteration
    · simpa [spartanInterface, sourceInterface, Spartan.pullback] using
        specification.initialState index

end NightstreamFPrime.Layout.Stage1.NextPreimageInputs
