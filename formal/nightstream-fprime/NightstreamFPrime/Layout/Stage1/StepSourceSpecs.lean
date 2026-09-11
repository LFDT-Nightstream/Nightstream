import NightstreamFPrime.Layout.Stage1.StateEncodingReadback
import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Stage1.NextPreimageInputs
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the running-transition and next-preimage specifications read from an
actual selected semantic step. Exact bounded source words identify the prior
and next typed states. The base branch uses the default running vector; only
the positive branch consumes the actual NIFS output equality.
-/

namespace NightstreamFPrime.Layout.Stage1.StepSourceSpecs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem initial_word
    (state : Nat → F)
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (decoded : StateDecoder.preimage logicalWidth publicFits state = value)
    (index : Lifecycle.Stage1.RunningTransition.StateIndex) :
    state (RunningTransitionInputs.initialStateWordStart + index.val) = value.z0.getD index.val 0 := by
  have initial : StateDecoder.initialState state = value.z0 := congrArg (fun preimage => preimage.z0) decoded
  exact (StateDecoder.slice_getD state RunningTransitionInputs.initialStateWordStart
    Lifecycle.Stage1.Application.stateWordCount index.val index.isLt).symm.trans
      (congrArg (fun words => words.getD index.val 0) initial)

private theorem current_word
    (state : Nat → F)
    (value : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (decoded : StateDecoder.preimage logicalWidth publicFits state = value)
    (index : Lifecycle.Stage1.RunningTransition.StateIndex) :
    state (RunningTransitionInputs.currentStateWordStart + index.val) = value.current.getD index.val 0 := by
  have current : StateDecoder.currentState state = value.current := congrArg (fun preimage => preimage.current) decoded
  exact (StateDecoder.slice_getD state RunningTransitionInputs.currentStateWordStart
    Lifecycle.Stage1.Application.stateWordCount index.val index.isLt).symm.trans
      (congrArg (fun words => words.getD index.val 0) current)

/-- Exact bounded source words and the selected step establish both existing
specifications. WellFormed keeps the prior and successor natural counters
below the modulus. No generated phase specification or row is a premise.
The NIFS result is used only in the positive branch. -/
theorem specs_of_step
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (env : Env)
    (step : StepHoldsFor relation ajtai context.toList Lifecycle.Stage1.Poseidon2HashChainV1.program input output)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (setup relation ajtai context.toList) input output))
    (priorWords : ∀ index : Fin PilotProduction.stateHashWords,
      env (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits)
          (priorHashPreimage (setup relation ajtai context.toList) input)).getD index.val 0)
    (nextWords : ∀ index : Fin PilotProduction.stateHashWords,
      env (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits)
          (nextHashPreimage (setup relation ajtai context.toList) input output)).getD index.val 0)
    (recursiveOutput : 0 < input.iteration →
      RunningTransitionInputs.piDecRunningOutput relation env = output.runningNext functionIndex) :
    Lifecycle.Stage1.RunningTransition.SpecHolds (RunningTransitionInputs.interface logicalWidth publicFits)
      RunningTransitionInputs.phaseOffset env ∧
    Lifecycle.Stage1.NextPreimage.SpecHolds NextPreimageInputs.sourceInterface
      RunningTransitionInputs.phaseOffset env := by
  let prior := priorHashPreimage (setup relation ajtai context.toList) input
  let next := nextHashPreimage (setup relation ajtai context.toList) input output
  have priorDecoded := StateEncodingReadback.preimage_eq_of_words prior priorWellFormed
    (fun word => env (PilotProduction.priorPreimageStart + word)) priorWords
  have nextDecoded := StateEncodingReadback.preimage_eq_of_words next nextWellFormed
    (fun word => env (PilotProduction.outputPreimageStart + word)) nextWords
  have priorIteration : (env (PilotProduction.priorPreimageStart +
      RunningTransitionInputs.iterationWordIndex)).val = input.iteration :=
    congrArg (fun preimage => preimage.iteration) priorDecoded
  have nextIteration : (env (PilotProduction.outputPreimageStart +
      RunningTransitionInputs.iterationWordIndex)).val = input.iteration + 1 :=
    congrArg (fun preimage => preimage.iteration) nextDecoded
  have outputRunning : StatementAbsorption.evalRunning
      (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits) env =
      output.runningNext functionIndex :=
    (StateDecoder.evalOutputRunning_eq_running logicalWidth publicFits env).trans
      (congrArg (fun preimage => preimage.running functionIndex) nextDecoded)
  have initial (index : Lifecycle.Stage1.RunningTransition.StateIndex) :
      (RunningTransitionInputs.initialStateExpr index).eval env = input.z0.getD index.val 0 := by
    simpa only [RunningTransitionInputs.initialStateExpr, Expr.eval_var, Nat.add_assoc] using
      initial_word _ prior priorDecoded index
  have current (index : Lifecycle.Stage1.RunningTransition.StateIndex) :
      (RunningTransitionInputs.currentStateExpr index).eval env = input.zi.getD index.val 0 := by
    simpa only [RunningTransitionInputs.currentStateExpr, Expr.eval_var, Nat.add_assoc] using
      current_word _ prior priorDecoded index
  constructor
  · rcases step.2.2.2 with base | recursive
    · have fieldZero : Lifecycle.Stage1.RunningTransition.iterationValue
          (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset env = 0 := by
        apply Fin.ext
        change (env (PilotProduction.priorPreimageStart + RunningTransitionInputs.iterationWordIndex)).val = 0
        exact priorIteration.trans base.1
      have outputDefault := outputRunning.trans (congrFun base.2.2 functionIndex)
      constructor
      · intro _ index
        exact (initial index).trans ((congrArg (fun words => words.getD index.val 0) base.2.1).trans
          (current index).symm)
      · intro _ index
        change (Lifecycle.Stage1.RunningTransition.runningWord
          (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits) index).eval env = _
        rw [Lifecycle.Stage1.RunningTransition.runningWord_eval, outputDefault]
        rfl
      · intro nonzero
        exact False.elim (nonzero fieldZero)
    · rcases recursive with ⟨_, positive, _⟩
      have fieldNonzero : Lifecycle.Stage1.RunningTransition.iterationValue
          (RunningTransitionInputs.interface logicalWidth publicFits) RunningTransitionInputs.phaseOffset env ≠ 0 := by
        intro zero
        have valueZero := congrArg Fin.val zero
        change (env (PilotProduction.priorPreimageStart + RunningTransitionInputs.iterationWordIndex)).val = 0 at valueZero
        rw [priorIteration] at valueZero
        exact (Nat.ne_of_gt positive) valueZero
      have recursiveRunning := (RunningTransitionInputs.eval_recursiveRunningExpr_eq_piDecRunningOutput
        relation env).trans (recursiveOutput positive)
      constructor
      · intro zero
        exact False.elim (fieldNonzero zero)
      · intro zero
        exact False.elim (fieldNonzero zero)
      · intro _ index
        change (Lifecycle.Stage1.RunningTransition.runningWord
          (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits) index).eval env =
          (Lifecycle.Stage1.RunningTransition.runningWord
            (RunningTransitionInputs.recursiveRunningExpr logicalWidth publicFits) index).eval env
        rw [Lifecycle.Stage1.RunningTransition.runningWord_eval,
          Lifecycle.Stage1.RunningTransition.runningWord_eval, outputRunning, recursiveRunning]
  · constructor
    · have words := congrArg natWord nextIteration
      rw [StateDecoder.natWord_val, ← priorIteration, StateDecoder.natWord_val_add_one] at words
      exact words
    · intro index
      have nextInitial := initial_word _ next nextDecoded index
      have priorInitial := initial_word _ prior priorDecoded index
      simpa only [NextPreimageInputs.sourceInterface, NextPreimageInputs.outputInitialStateSource,
        NextPreimageInputs.priorInitialStateSource, Expr.eval_var, Nat.add_assoc] using
        nextInitial.trans priorInitial.symm

end NightstreamFPrime.Layout.Stage1.StepSourceSpecs
