import NightstreamFPrime.Export.Stage1.DirectAccumulatorCommonSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns the deterministic semantic edge from one self-derived per-application
matrix plan to the exact verifier-selected HyperNova augmented step.

The representation record contains only typed ABI equalities. The relation,
application, package identity, verifier-context descriptor, verification-key
digest, and Ajtai key are fixed before the statement is formed. This module
does not close package conformance, terminal verification, or security.
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

abbrev FitsTwoPow28 (application : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 application

def relation (application : Program) (fits : FitsTwoPow28 application) :=
  PerApplicationFixedPoint.relation application fits

def verifierKeyDigest {application : Program}
    (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)) :
    KeyDigest :=
  (PerApplicationCanonicalPackage.verificationKeyBinding fits ajtai).digest

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
structure Represents
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
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
      (setup (relation application fits) ajtai (verifierKeyDigest fits ajtai))
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
      (setup (relation application fits) ajtai (verifierKeyDigest fits ajtai))
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
  accumulatorOutput : AccumulatorInputs.output (relation application fits)
    (commonEnv application assignment base) = output.runningNext functionIndex
  runningOutput : PiCCS.v1_1.StatementAbsorption.evalRunning
    (RunningTransitionInputs.outputRunningExpr
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application))
    (transitionEnv application base) = output.runningNext functionIndex
  priorPc : input.priorPc = 1
  pcNext : output.pcNext = functionIndex

/-- Acceptance of the exact self-derived matrix plan forces the fixed
per-application HyperNova step under the verifier-key digest derived from the
same package identity and Ajtai key. -/
theorem rowsZero_implies_stepHoldsFor
    (application : Program) (fits : FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
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
    (represents : Represents application fits ajtai assignment base input output)
    (piRlcAssumptions : Lifecycle.PiRLC.v1_1.Formal.Assumptions
      (relation application fits)
      (PiRLCInputs.interface
        (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      PiRLCInputs.phaseOffset (commonEnv application assignment base))
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits
      ).RowsZero assignment) :
    StepHoldsFor (relation application fits) ajtai
      (verifierKeyDigest fits ajtai) application input output := by
  have semantics := PerApplicationFixedPoint.rowsZero_implies_semantics
    application fits assignment base groupValue products one encodes accepted
  have hashSlots := Lifecycle.Pilot.builders_imply_hash_slots
    PilotProduction.interface PilotProduction.witnessOffset
    (pilotEnv application base) (relation application fits) ajtai
    (verifierKeyDigest fits ajtai) application.step input output
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
      (relation application fits) ajtai (verifierKeyDigest fits ajtai)
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
        (relation application fits) ajtai (verifierKeyDigest fits ajtai)
        (prefixGeometry application) assignment base groupValue products one
        encodes.runningPrefix semantics.runningPrefix piRlcAssumptions
  have runningPhysical : RunningTransitionLayout.PhysicalHolds
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (transitionEnv application base) := by
    simpa [transitionEnv] using semantics.runningPrefix.prior.transition
  have runningSpec := RunningTransitionLayout.physical_implies_specHolds
    (relation application fits) (transitionEnv application base) runningPhysical
  change FixedAugmentedTransition
    (setup (relation application fits) ajtai (verifierKeyDigest fits ajtai))
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
            (verifierKeyDigest fits ajtai)).defaultRunning := by
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
              (verifierKeyDigest fits ajtai)).defaultRunning := rfl
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
        represents.accumulatorOutput.trans transitionOutput
      have accumulatorOutput : Lifecycle.Stage1.Accumulator.Holds
          (relation application fits) ajtai (verifierKeyDigest fits ajtai)
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
          (setup (relation application fits) ajtai
            (verifierKeyDigest fits ajtai)).nifs
          ((setup (relation application fits) ajtai
            (verifierKeyDigest fits ajtai)).verifierKeys
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

end NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness
