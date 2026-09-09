import NightstreamFPrime.Export.Stage1.DirectAccumulatorCommonSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
import NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO
import NightstreamFPrime.Export.Stage1.PerApplicationVerifierBoundAssignment
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns the deterministic semantic edge from one self-derived per-application
matrix plan to the exact verifier-selected HyperNova augmented step.

The public theorem decodes all typed ABI values from one canonical raw packet.
The relation, application, context key, and Ajtai key are fixed before the
statement is formed. Final package closure must prove that the decoded context
key is the verifier-owned canonical verifier-context digest. This module does not
close package conformance, terminal verification, or security.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev RawValues := PerApplicationCanonicalAssignment.RawValues

abbrev FitsTwoPow28 (application : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 application

abbrev CommitmentSetup (application : Program) :=
  PerApplicationCanonicalPackage.CommitmentSetup application

def relation (application : Program) (fits : FitsTwoPow28 application) :=
  PerApplicationFixedPoint.relation application fits

def geometry (application : Program) :=
  PerApplicationFixedPoint.geometry application

def prefixGeometry (application : Program) :=
  DirectApplicationPrefixPlan.prefixGeometry (geometry application)

def transitionEnv (application : Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) : Env :=
  Spartan.pullback (RunningTransitionDirectPlan.transitionEnv application base)

def commonEnv (application : Program)
    (assignment : PaperLinearAlgebra.Assignment F
      (PerApplicationFixedPoint.logicalWidth application))
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) : Env :=
  Spartan.pullback
    (PiRLCSamplerRetainedCustody.semanticEnv (prefixGeometry application)
      assignment base)

def pilotEnv (application : Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) : Env :=
  PilotSpartan.pullback (PilotOrdinaryDirectPlan.pilotEnv application base)

def applicationEnv (application : Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F) : Env :=
  ApplicationDirectPlan.sourceEnv
    (DirectApplicationPrefixPlan.applicationSource application base)

private theorem slot_eq_functionIndex (slot : Fin slotCount) :
    slot = functionIndex := by
  apply Fin.ext
  have bound := slot.isLt
  change slot.val < 1 at bound
  change slot.val = 0
  omega

/-- Typed meaning of the final direct plan's external values. Every field is
an ABI equality. No field assumes application correctness, hash correctness,
NIFS acceptance, or a terminal relation. -/
private structure Represents
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (vk : KeyDigest)
    (assignment : PaperLinearAlgebra.Assignment F
      (PerApplicationFixedPoint.logicalWidth application))
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (input : Input KeyDigest AppState AppWitness
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Proof (ProductionKey.degreeBound (relation application fits))) slotCount)
    (output : Output Digest AppState
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      slotCount) : Prop where
  priorPreimage : PriorStateHash.RepresentsPreimage
    PilotProduction.priorInterface PilotProduction.witnessOffset
    (pilotEnv application base)
    (priorHashPreimage
      (setup (relation application fits) ajtai vk)
      input)
  priorPublicInput : PriorStateHash.RepresentsPublicInput
    PilotProduction.priorInterface PilotProduction.witnessOffset
    (pilotEnv application base)
    ((machineFor (PerApplicationFixedPoint.publicFits application) application
      ).freshPublic input.fresh)
  nextPreimage : OutputHash.RepresentsPreimage
    PilotProduction.outputInterface
    (Lifecycle.Pilot.outputOffset PilotProduction.interface
      PilotProduction.witnessOffset)
    (pilotEnv application base)
    (nextHashPreimage
      (setup (relation application fits) ajtai vk)
      input output)
  nextDigest : OutputHash.RepresentsDigest PilotProduction.outputInterface
    (Lifecycle.Pilot.outputOffset PilotProduction.interface
      PilotProduction.witnessOffset)
    (pilotEnv application base) output.x
  applicationInput : Lifecycle.Stage1.Application.inputState
    (ApplicationInputs.interface application)
    (ApplicationInputs.localStart application) (applicationEnv application base) =
      input.zi
  applicationWitness : Lifecycle.Stage1.Application.witnessValue
    (ApplicationInputs.interface application)
    (ApplicationInputs.localStart application) (applicationEnv application base) =
      input.witness
  applicationOutput : Lifecycle.Stage1.Application.outputState
    (ApplicationInputs.interface application)
    (ApplicationInputs.localStart application) (applicationEnv application base) =
      output.zNext
  iterationZero : Lifecycle.Stage1.RunningTransition.iterationValue
    (RunningTransitionInputs.interface
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application))
    RunningTransitionInputs.phaseOffset (transitionEnv application base) = 0 ↔
      input.iteration = 0
  initialState : List.ofFn (fun index =>
    (RunningTransitionInputs.initialStateExpr index).eval
      (transitionEnv application base)) = input.z0
  currentState : List.ofFn (fun index =>
    (RunningTransitionInputs.currentStateExpr index).eval
      (transitionEnv application base)) = input.zi
  runningInput : AccumulatorInputs.running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application)
    (commonEnv application assignment base) = input.running functionIndex
  freshInput : AccumulatorInputs.fresh
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application)
    (commonEnv application assignment base) = input.fresh
  proofInput : AccumulatorInputs.proof (relation application fits)
    (commonEnv application assignment base) = input.nifsProof
  accumulatorOutputEnv : AccumulatorInputs.output (relation application fits)
      (commonEnv application assignment base) =
    AccumulatorInputs.output (relation application fits)
      (transitionEnv application base)
  runningOutput : PiCCS.v1_1.StatementAbsorption.evalRunning
    (RunningTransitionInputs.outputRunningExpr
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application))
    (transitionEnv application base) = output.runningNext functionIndex
  priorPc : input.priorPc = 1
  pcNext : output.pcNext = functionIndex

/-- Internal typed-ABI proof. The public theorem below derives every input to
this lemma from one canonical raw packet and accepted rows. -/
private theorem representedSemantics_imply_stepHoldsFor
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (vk : KeyDigest)
    (assignment : PaperLinearAlgebra.Assignment F
      (PerApplicationFixedPoint.logicalWidth application))
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (ApplicationRetainedGeometry.oneColumn (geometry application)) = 1)
    (encodes : DirectApplicationPrefixPlan.Encodes (geometry application)
      assignment base groupValue products)
    (input : Input KeyDigest AppState AppWitness
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Proof (ProductionKey.degreeBound (relation application fits))) slotCount)
    (output : Output Digest AppState
      (Running
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      slotCount)
    (represents : Represents application fits ajtai vk assignment base
      input output)
    (semantics : DirectApplicationPrefixPlan.Semantics
      (relation application fits) (geometry application) assignment base
      groupValue products) :
    StepHoldsFor (relation application fits) ajtai
      vk application input output := by
  have hashSlots := Lifecycle.Pilot.builders_imply_hash_slots
    PilotProduction.interface PilotProduction.witnessOffset
    (pilotEnv application base) (relation application fits) ajtai
    vk application.step input output
    semantics.runningPrefix.prior.pilot represents.priorPreimage
    represents.priorPublicInput represents.nextPreimage represents.nextDigest
  have applicationStep : output.zNext =
      application.step input.zi input.witness := by
    have applicationHolds := semantics.applicationSemantics
    unfold Lifecycle.Stage1.Application.Holds at applicationHolds
    calc
      output.zNext = Lifecycle.Stage1.Application.outputState
          (ApplicationInputs.interface application)
          (ApplicationInputs.localStart application)
          (applicationEnv application base) :=
        represents.applicationOutput.symm
      _ = application.step
          (Lifecycle.Stage1.Application.inputState
            (ApplicationInputs.interface application)
            (ApplicationInputs.localStart application)
            (applicationEnv application base))
          (Lifecycle.Stage1.Application.witnessValue
            (ApplicationInputs.interface application)
            (ApplicationInputs.localStart application)
            (applicationEnv application base)) := applicationHolds
      _ = application.step input.zi input.witness := by
        rw [represents.applicationInput, represents.applicationWitness]
  have accumulator : Lifecycle.Stage1.Accumulator.Holds
      (relation application fits) ajtai vk
      (AccumulatorInputs.running
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (commonEnv application assignment base))
      (AccumulatorInputs.fresh
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (commonEnv application assignment base))
      (AccumulatorInputs.proof (relation application fits)
        (commonEnv application assignment base))
      (AccumulatorInputs.output (relation application fits)
        (commonEnv application assignment base)) := by
    simpa [commonEnv] using
      DirectAccumulatorCommonSemantics.semantics_imply_accumulatorHolds
        (relation application fits) ajtai vk
        (prefixGeometry application) assignment base groupValue products one
        encodes.runningPrefix semantics.runningPrefix
        (PiRLCInputBounds.assumptions (relation application fits)
          (commonEnv application assignment base))
  have runningPhysical : RunningTransitionLayout.PhysicalHolds
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (transitionEnv application base) := by
    simpa [transitionEnv] using semantics.runningPrefix.prior.transition
  have runningSpec := RunningTransitionLayout.physical_implies_specHolds
    (relation application fits) (transitionEnv application base) runningPhysical
  change FixedAugmentedTransition
    (setup (relation application fits) ajtai vk)
    (machineFor (PerApplicationFixedPoint.publicFits application) application)
    functionIndex input output
  refine ⟨represents.pcNext, applicationStep, ?_, ?_⟩
  · simpa [machineFor] using hashSlots.2
  · rcases Nat.eq_zero_or_pos input.iteration with iterationZero |
      iterationPositive
    · have fieldZero : Lifecycle.Stage1.RunningTransition.iterationValue
          (RunningTransitionInputs.interface
            (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application))
          RunningTransitionInputs.phaseOffset (transitionEnv application base) =
          0 := represents.iterationZero.mpr iterationZero
      have initialState : input.z0 = input.zi := by
        calc
          input.z0 = List.ofFn (fun index =>
              (RunningTransitionInputs.initialStateExpr index).eval
                (transitionEnv application base)) :=
            represents.initialState.symm
          _ = List.ofFn (fun index =>
              (RunningTransitionInputs.currentStateExpr index).eval
                (transitionEnv application base)) := by
            apply List.ext_get
            · simp
            · intro index leftBound rightBound
              have bounded : index <
                  Lifecycle.Stage1.RunningTransition.stateWordCount := by
                simpa using leftBound
              simpa using runningSpec.initialState fieldZero ⟨index, bounded⟩
          _ = input.zi := represents.currentState
      have runningBase := RunningTransitionLayout.physical_implies_typed_base
        (relation application fits) (transitionEnv application base)
        runningPhysical fieldZero
      have defaultOutput : output.runningNext = fun _ =>
          (setup (relation application fits) ajtai
            vk).defaultRunning := by
        funext slot
        have slotEq : slot = functionIndex := slot_eq_functionIndex slot
        subst slot
        calc
          output.runningNext functionIndex =
              PiCCS.v1_1.StatementAbsorption.evalRunning
                (RunningTransitionInputs.outputRunningExpr
                  (PerApplicationFixedPoint.logicalWidth application)
                  (PerApplicationFixedPoint.publicFits application))
                (transitionEnv application base) :=
            represents.runningOutput.symm
          _ = defaultRunning
              (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
              (publicFits := PerApplicationFixedPoint.publicFits application) :=
            runningBase
          _ = (setup (relation application fits) ajtai
              vk).defaultRunning := rfl
      exact Or.inl ⟨iterationZero, initialState, defaultOutput⟩
    · have iterationNonzero : input.iteration ≠ 0 :=
        Nat.ne_of_gt iterationPositive
      have fieldNonzero : Lifecycle.Stage1.RunningTransition.iterationValue
          (RunningTransitionInputs.interface
            (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application))
          RunningTransitionInputs.phaseOffset (transitionEnv application base) ≠
          0 := by
        intro fieldZero
        exact iterationNonzero (represents.iterationZero.mp fieldZero)
      have runningRecursive :=
        RunningTransitionLayout.physical_implies_typed_recursive
          (relation application fits) (transitionEnv application base)
          runningPhysical fieldNonzero
      have transitionOutput : output.runningNext functionIndex =
          RunningTransitionInputs.piDecRunningOutput
            (relation application fits) (transitionEnv application base) := by
        calc
          output.runningNext functionIndex =
              PiCCS.v1_1.StatementAbsorption.evalRunning
                (RunningTransitionInputs.outputRunningExpr
                  (PerApplicationFixedPoint.logicalWidth application)
                  (PerApplicationFixedPoint.publicFits application))
                (transitionEnv application base) :=
            represents.runningOutput.symm
          _ = RunningTransitionInputs.piDecRunningOutput
              (relation application fits) (transitionEnv application base) :=
            runningRecursive
      have outputBridge : AccumulatorInputs.output (relation application fits)
          (commonEnv application assignment base) =
          RunningTransitionInputs.piDecRunningOutput
            (relation application fits) (transitionEnv application base) :=
        calc
          AccumulatorInputs.output (relation application fits)
              (commonEnv application assignment base) =
            AccumulatorInputs.output (relation application fits)
              (transitionEnv application base) :=
            represents.accumulatorOutputEnv
          _ = RunningTransitionInputs.piDecRunningOutput
              (relation application fits)
              (transitionEnv application base) := rfl
      have accumulatorOutput : Lifecycle.Stage1.Accumulator.Holds
          (relation application fits) ajtai vk
          (input.running functionIndex) input.fresh input.nifsProof
          (output.runningNext functionIndex) := by
        rw [represents.runningInput, represents.freshInput,
          represents.proofInput, outputBridge, ← transitionOutput] at accumulator
        exact accumulator
      have priorPcValid : InRange slotCount input.priorPc := by
        rw [represents.priorPc]
        norm_num [InRange, slotCount]
      have selectedEq : selectedIndex priorPcValid = functionIndex :=
        slot_eq_functionIndex _
      have selectedNifs : Accepts
          (setup (relation application fits) ajtai vk).nifs
          ((setup (relation application fits) ajtai vk).verifierKeys
              (selectedIndex priorPcValid))
          (input.running (selectedIndex priorPcValid)) input.fresh
          input.nifsProof (output.runningNext (selectedIndex priorPcValid)) := by
        rw [selectedEq]
        simpa only [Accepts, setup, nifsVerifier,
          Lifecycle.Stage1.Accumulator.Holds] using accumulatorOutput
      have unchanged : ∀ slot, slot ≠ selectedIndex priorPcValid →
          output.runningNext slot = input.running slot := by
        intro slot notSelected
        exact False.elim (notSelected
          ((slot_eq_functionIndex slot).trans selectedEq.symm))
      exact Or.inr ⟨priorPcValid, iterationPositive, (by
        simpa [machineFor] using hashSlots.1), selectedNifs, unchanged⟩

/-- One canonical raw assignment and acceptance of the exact self-derived
matrix plan force the verifier-selected HyperNova step. No caller supplies a
typed input, typed output, representation record, constant-column fact, or
retained-block encoding proof. The immediate key is the context key decoded
from the constrained prior state; final package closure must identify it with
the verifier-owned canonical verifier-context digest. -/
theorem rowsZero_implies_stepHoldsFor
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits
      ).RowsZero raw.assignment) :
    StepHoldsFor (relation application fits) ajtai
      (PerApplicationDecodedIO.contextKey raw) application
      (PerApplicationDecodedIO.input application fits raw)
      (PerApplicationDecodedIO.output application raw) := by
  have one := PerApplicationCanonicalAssignment.assignment_one raw
  have encodes := PerApplicationCanonicalEncodes.encodes raw
  have semantics := PerApplicationFixedPoint.rowsZero_implies_semantics
    application fits raw.assignment raw.base raw.groupValue raw.products one
    encodes accepted
  have canonical := PerApplicationDecodedIO.semantics_imply_canonicalStates
    application fits raw semantics
  have represents : Represents application fits ajtai
      (PerApplicationDecodedIO.contextKey raw) raw.assignment raw.base
      (PerApplicationDecodedIO.input application fits raw)
      (PerApplicationDecodedIO.output application raw) := by
    refine {
      priorPreimage := ?_
      priorPublicInput := ?_
      nextPreimage := ?_
      nextDigest := ?_
      applicationInput := ?_
      applicationWitness := ?_
      applicationOutput := ?_
      iterationZero := ?_
      initialState := ?_
      currentState := ?_
      runningInput := ?_
      freshInput := ?_
      proofInput := ?_
      accumulatorOutputEnv := ?_
      runningOutput := ?_
      priorPc := ?_
      pcNext := ?_ }
    · simpa [pilotEnv, PerApplicationDecodedIO.pilotEnv] using
        PerApplicationDecodedIO.priorHashPreimageRepresents application fits
          ajtai raw canonical.1
    · simpa [pilotEnv, PerApplicationDecodedIO.pilotEnv] using
        PerApplicationDecodedIO.priorPublicInputRepresents application fits raw
    · simpa [pilotEnv, PerApplicationDecodedIO.pilotEnv] using
        PerApplicationDecodedIO.nextHashPreimageRepresents application fits
          ajtai raw canonical.2 semantics
    · simpa [pilotEnv, PerApplicationDecodedIO.pilotEnv] using
        PerApplicationDecodedIO.outputDigestRepresents raw
    · simpa [applicationEnv, PerApplicationDecodedIO.applicationEnv] using
        PerApplicationDecodedIO.applicationInputRepresents application fits raw
    · rfl
    · simpa [applicationEnv, PerApplicationDecodedIO.applicationEnv] using
        PerApplicationDecodedIO.applicationOutputRepresents application raw
    · simpa [transitionEnv, PerApplicationDecodedIO.transitionEnv] using
        PerApplicationDecodedIO.iterationZeroRepresents application fits raw
    · simpa [transitionEnv, PerApplicationDecodedIO.transitionEnv] using
        PerApplicationDecodedIO.initialStateRepresents application fits raw
    · simpa [transitionEnv, PerApplicationDecodedIO.transitionEnv] using
        PerApplicationDecodedIO.currentStateRepresents application fits raw
    · simpa [commonEnv, PerApplicationDecodedIO.commonEnv] using
        PerApplicationDecodedIO.runningInputRepresents application fits raw
    · rfl
    · rfl
    · simpa [commonEnv, transitionEnv, PerApplicationDecodedIO.commonEnv,
        PerApplicationDecodedIO.transitionEnv] using
        PerApplicationDecodedIO.accumulatorOutputEnvRepresents application fits
          raw
    · simpa [transitionEnv, PerApplicationDecodedIO.transitionEnv] using
        PerApplicationDecodedIO.runningOutputRepresents application raw
    · rfl
    · rfl
  exact representedSemantics_imply_stepHoldsFor application fits ajtai
    (PerApplicationDecodedIO.contextKey raw) raw.assignment raw.base
    raw.groupValue raw.products one encodes
    (PerApplicationDecodedIO.input application fits raw)
    (PerApplicationDecodedIO.output application raw) represents semantics

/-- Accepted final rows force both the augmented step and the exact recursive
public output exposed by the complete Phi81 assignment. -/
theorem rowsZero_implies_stepHoldsFor_and_publicOutput
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits
      ).RowsZero raw.assignment) :
    StepHoldsFor (relation application fits) ajtai
        (PerApplicationDecodedIO.contextKey raw) application
        (PerApplicationDecodedIO.input application fits raw)
        (PerApplicationDecodedIO.output application raw) ∧
      Spec.Phi81Relation.projectPublicInput raw.completeAssignment =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (PerApplicationDecodedIO.output application raw).x := by
  constructor
  · exact rowsZero_implies_stepHoldsFor application fits ajtai raw accepted
  · simpa [PerApplicationDecodedIO.output,
      PerApplicationDecodedIO.outputDigest] using
      PerApplicationCanonicalAssignment.projectPublicInput_completeAssignment raw

/-- The verifier-owned raw constructor pins the exact canonical context digest
into the state. Acceptance therefore forces the augmented step under that
digest; the prover cannot select the application, package, or static authority.
-/
theorem verifierBoundRowsZero_implies_stepHoldsFor
    (application : Program) (fits : FitsTwoPow28 application)
    (commitmentSetup : CommitmentSetup application)
    (raw : RawValues application)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits
      ).RowsZero
        (PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw
          ).assignment) :
    StepHoldsFor (relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationVerifierBoundAssignment.verifierContextDigest fits
        commitmentSetup)
      application
      (PerApplicationDecodedIO.input application fits
        (PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw))
      (PerApplicationDecodedIO.output application
        (PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw)) := by
  let bound := PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw
  have step := rowsZero_implies_stepHoldsFor application fits
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) bound accepted
  have one := PerApplicationCanonicalAssignment.assignment_one bound
  have encodes := PerApplicationCanonicalEncodes.encodes bound
  have semantics := PerApplicationFixedPoint.rowsZero_implies_semantics
    application fits bound.assignment bound.base bound.groupValue bound.products
    one encodes accepted
  have keyEq :=
    PerApplicationVerifierBoundAssignment.semantics_imply_contextKey
      application fits commitmentSetup raw semantics
  rw [keyEq] at step
  exact step

end NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness
