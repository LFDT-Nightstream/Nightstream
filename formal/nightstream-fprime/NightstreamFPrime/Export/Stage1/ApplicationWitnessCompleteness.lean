import NightstreamFPrime.Export.Stage1.ApplicationCompactWitness
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.Stage1LoweringBridge
import NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan
import NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO
import NightstreamFPrime.Layout.Stage1.ApplicationSemantics
import NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness

/-!
Owns construction of the selected application's private suffix from its real
four-word witness and the state frames constructed by the physical prefix.
The four-word condition is the selected input ABI, not a generated-output
assumption. The source packet retains the preceding prefix through the existing
per-application shift and fills the application interval from its proved builder.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationWitnessCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open Poseidon2HashChainV1Package (application)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem represented_words
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage) (values : Fin PilotProduction.stateHashWords → F)
    (words : ∀ index, values index = (serializePreimage (publicFits := publicFits) preimage).getD index.val 0) :
    List.ofFn values = serializePreimage (publicFits := publicFits) preimage := by
  apply List.ext_get
  · rw [List.length_ofFn, PilotProduction.serializePreimage_length_fixed preimage fixed]
  · intro index leftBound rightBound
    rw [List.get_ofFn, words]
    exact List.getD_eq_getElem (l := serializePreimage (publicFits := publicFits) preimage) (d := 0) rightBound

private theorem complete_suffix
    (target : Env)
    (prior next : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorFixed : PilotProduction.FixedPreimage prior) (nextFixed : PilotProduction.FixedPreimage next)
    (priorWords : ∀ index : Fin PilotProduction.stateHashWords,
      Spartan.pullback target (PilotProduction.priorPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) prior).getD index.val 0)
    (nextWords : ∀ index : Fin PilotProduction.stateHashWords,
      Spartan.pullback target (PilotProduction.outputPreimageStart + index.val) =
        (serializePreimage (publicFits := publicFits) next).getD index.val 0)
    (witness : AppWitness)
    (width : witness.length = Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount)
    (step : next.current = application.step prior.current witness) :
    ∃ suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F,
      (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
        (PerApplicationFixedPoint.geometry application)).RowsZero
        (PerApplicationAssignmentTransportExecution.canonicalRawValues application
          (PerApplicationSourceAssignment.ofCompleted application target suffix)).assignment ∧
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application)
        (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix)) = witness := by
  have priorRepresented : PriorStateHash.RepresentsPreimage PilotProduction.priorInterface
      PilotProduction.witnessOffset (Spartan.pullback target) prior := by
    unfold PriorStateHash.RepresentsPreimage
    rw [PilotProduction.priorInterface_preimage_apply]
    simp only [Gadgets.Poseidon2.Hash.evalList, PilotProduction.priorPreimage, PilotProduction.variableExprs,
      List.map_ofFn, Function.comp_apply, Expr.eval_var]
    exact represented_words prior priorFixed _ priorWords
  have nextRepresented : OutputHash.RepresentsPreimage PilotProduction.outputInterface
      PilotProduction.lifecycleOutputOffset (Spartan.pullback target) next := by
    unfold OutputHash.RepresentsPreimage
    rw [PilotProduction.outputInterface_preimage_apply]
    simp only [Gadgets.Poseidon2.Hash.evalList, PilotProduction.outputPreimage, PilotProduction.variableExprs,
      List.map_ofFn, Function.comp_apply, Expr.eval_var]
    exact represented_words next nextFixed _ nextWords
  have inputValue := ApplicationInputs.inputState_eq_current application target prior priorFixed priorRepresented
  have outputValue := ApplicationInputs.outputState_eq_current application target next nextFixed nextRepresented
  let message : Fin 4 → F := fun lane => witness.getD lane.val 0
  have messageEq : List.ofFn message = witness := by
    apply List.ext_get
    · rw [List.length_ofFn, width]
      rfl
    · intro index leftBound rightBound
      rw [List.get_ofFn]
      exact List.getD_eq_getElem (l := witness) (d := 0) rightBound
  have directStep : (List.ofFn fun lane : Fin 4 => target (ApplicationInputs.outputColumn lane)) =
      application.step (List.ofFn (ApplicationCompactWitness.priorValues target)) (List.ofFn message) := by
    change Lifecycle.Stage1.Application.outputState (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) target = application.step
      (Lifecycle.Stage1.Application.inputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) target) (List.ofFn message)
    rw [inputValue, outputValue, messageEq]
    exact step
  refine ⟨ApplicationCompactWitness.privateSuffix (ApplicationCompactWitness.priorValues target) message,
    ApplicationCompactWitness.complete target message directStep, ?_⟩
  exact (ApplicationCompactWitness.witnessValue target message).trans messageEq

private theorem outputDigest_of_completed
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (prior next : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (digest : Digest)
    (priorFixed : PilotProduction.FixedPreimage prior) (nextFixed : PilotProduction.FixedPreimage next)
    (digestFixed : digest.length = PilotProduction.digestWords)
    (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
    (sources : ∀ index, index < PilotProduction.witnessOffset →
      Spartan.pullback target index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior))
        next digest priorFixed nextFixed digestFixed values context index) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)).outputDigest = digest := by
  let raw := PerApplicationAssignmentTransportExecution.canonicalRawValues application
    (PerApplicationSourceAssignment.ofCompleted application target suffix)
  have copied : ∀ index, index < PilotProduction.witnessOffset →
      PerApplicationDecodedIO.pilotEnv raw index = Spartan.pullback target index := by
    intro index below
    have pilotBound : index < Spartan.pilotSourceColumnCount := by
      rw [PilotProduction.witnessOffset_eq] at below
      norm_num [Spartan.pilotSourceColumnCount] at ⊢
      omega
    have sourceBound : index < Spartan.SourceColumnCount := by
      rw [PilotProduction.witnessOffset_eq] at below
      rw [Spartan.sourceColumnCount_eq]
      omega
    have beforeC : index < PiCCSInputs.phaseOffset := by
      rw [PilotProduction.witnessOffset_eq] at below
      rw [PiCCSInputs.phaseOffset_eq]
      omega
    calc
      _ = PerApplicationDecodedIO.transitionEnv raw index :=
        PerApplicationDecodedIO.pilotEnv_eq_transitionEnv_of_lt raw index pilotBound
      _ = RunningTransitionDirectPlan.packageEnv application raw.base (Spartan.sourceToSpartan index) :=
        RunningTransitionDirectPlan.transitionEnv_of_outside application raw.base index sourceBound (Or.inl beforeC)
      _ = _ := PerApplicationSourceAssignment.source_ofCompleted application target suffix index sourceBound
  have represented := PilotProduction.protocolEnv_represents_of_agreesBelow prior (encHash (stateHash prior))
    next digest priorFixed nextFixed digestFixed (PerApplicationDecodedIO.pilotEnv raw) (by
      intro index below
      exact (copied index below).trans ((sources index below).trans
        (PiCCSProtocolCompleteness.pilot_word prior (encHash (stateHash prior)) next digest
          priorFixed nextFixed digestFixed values context index below)))
  have digestEq := represented.2.2.2
  change digest = raw.outputDigest at digestEq
  exact digestEq.symm

/-- A selected admissible semantic step constructs the physical prefix and
the selected application suffix from its actual witness. The exact four-word
witness condition is the selected input ABI also enforced by the Rust encoder.
All source words and application rows are outputs of the construction. The
canonical carrier public input is the encoding of the same output digest. -/
theorem complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits)) slotCount)
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (step : StepHoldsFor relation ajtai context.toList application input output)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (setup relation ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex)
    (witnessWidth : input.witness.length = Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount) :
    let prior := priorHashPreimage (setup relation ajtai context.toList) input
    let next := nextHashPreimage (setup relation ajtai context.toList) input output
    let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
    ∃ (digestFixed : output.x.length = PilotProduction.digestWords), ∃ target : Env,
      ∃ suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F,
      target Spartan.constantColumn = 1 ∧
      R1CS.RowsHold target (Spartan.remappedRows relation) ∧
      holdsFlat (Spartan.pullback target) (Lifecycle.Stage1.NextPreimage.opsAt
        NextPreimageInputs.sourceInterface RunningTransitionInputs.phaseOffset) ∧
      RunningTransitionInputs.piDecRunningOutput relation (Spartan.pullback target) = result ∧
      (∀ index, index < PilotProduction.witnessOffset ∨ PiCCSOrdinarySourceSupport.External index →
        Spartan.pullback target index = PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior))
          next output.x priorWellFormed.1 nextWellFormed.1 digestFixed values context index) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        Spartan.pullback target (PilotProduction.priorPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) prior).getD index.val 0) ∧
      (∀ index : Fin PilotProduction.stateHashWords,
        Spartan.pullback target (PilotProduction.outputPreimageStart + index.val) =
          (serializePreimage (publicFits := publicFits) next).getD index.val 0) ∧
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application)
        (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix)) = input.witness ∧
      let raw := PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application target suffix)
      raw.outputDigest = output.x ∧
      Phi81Relation.projectPublicInput raw.completeAssignment =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) output.x ∧
      (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
        (PerApplicationFixedPoint.geometry application)).RowsZero raw.assignment := by
  obtain ⟨digestFixed, target, constant, physical, nextRows, nifsOutput, sources, priorWords, nextWords⟩ :=
    StepPhysicalCompleteness.complete relation ajtai context input output result step priorWellFormed nextWellFormed
      freshPublic accepted recursiveResult
  obtain ⟨suffix, applicationRows, actualWitness⟩ := complete_suffix target
    (priorHashPreimage (setup relation ajtai context.toList) input)
    (nextHashPreimage (setup relation ajtai context.toList) input output)
    priorWellFormed.1 nextWellFormed.1 priorWords nextWords input.witness witnessWidth step.2.1
  have outputDigest := outputDigest_of_completed target suffix
    (priorHashPreimage (setup relation ajtai context.toList) input)
    (nextHashPreimage (setup relation ajtai context.toList) input output)
    output.x priorWellFormed.1 nextWellFormed.1 digestFixed
    (PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof) context
    (fun index below => sources index (Or.inl below))
  have publicInput := PerApplicationCanonicalAssignment.projectPublicInput_completeAssignment
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix))
  rw [outputDigest] at publicInput
  exact ⟨digestFixed, target, suffix, constant, physical, nextRows, nifsOutput, sources,
    priorWords, nextWords, actualWitness, outputDigest, publicInput, applicationRows⟩

end NightstreamFPrime.Export.Stage1.ApplicationWitnessCompleteness
