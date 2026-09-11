import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
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

private def loadWitness (env : Env) (witness : AppWitness) : Env :=
  (((env.set ApplicationInputs.witnessStart (witness.getD 0 0)).set
    (ApplicationInputs.witnessStart + 1) (witness.getD 1 0)).set
    (ApplicationInputs.witnessStart + 2) (witness.getD 2 0)).set
    (ApplicationInputs.witnessStart + 3) (witness.getD 3 0)

private theorem loadWitness_read (env : Env) (witness : AppWitness)
    (index : Fin Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount) :
    loadWitness env witness (ApplicationInputs.witnessStart + index.val) = witness.getD index.val 0 := by
  fin_cases index <;> simp [loadWitness, Env.set]

private theorem loadWitness_agrees (env : Env) (witness : AppWitness) :
    AgreesOutside env (loadWitness env witness) ApplicationInputs.witnessStart
      Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount := by
  intro index outside
  have first : index ≠ ApplicationInputs.witnessStart := by
    simp only [Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount] at outside
    omega
  have second : index ≠ ApplicationInputs.witnessStart + 1 := by
    simp only [Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount] at outside
    omega
  have third : index ≠ ApplicationInputs.witnessStart + 2 := by
    simp only [Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount] at outside
    omega
  have fourth : index ≠ ApplicationInputs.witnessStart + 3 := by
    simp only [Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount] at outside
    omega
  simp only [loadWitness, Env.set, if_neg first, if_neg second, if_neg third, if_neg fourth]

private theorem witnessValue_loaded (env : Env) (witness : AppWitness)
    (width : witness.length = Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount) :
    Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) (loadWitness env witness) = witness := by
  change List.ofFn (fun index : Fin Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount =>
    loadWitness env witness (ApplicationInputs.witnessStart + index.val)) = witness
  apply List.ext_get
  · rw [List.length_ofFn, width]
  · intro index leftBound rightBound
    rw [List.get_ofFn, loadWitness_read]
    exact List.getD_eq_getElem (l := witness) (d := 0) rightBound

private theorem input_before_witness (index : Lifecycle.Stage1.Application.StateIndex) :
    ApplicationInputs.inputColumn index < ApplicationInputs.witnessStart := by
  rw [ApplicationInputs.inputColumn_value, ApplicationInputs.witnessStart, Spartan.privateColumnCount_eq]
  have bound := index.isLt
  norm_num [ApplicationInputs.currentWordStart, Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
  omega

private theorem output_before_witness (index : Lifecycle.Stage1.Application.StateIndex) :
    ApplicationInputs.outputColumn index < ApplicationInputs.witnessStart := by
  rw [ApplicationInputs.outputColumn_value, ApplicationInputs.witnessStart, Spartan.privateColumnCount_eq]
  have bound := index.isLt
  norm_num [Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
  omega

private theorem stateValues_loaded (env : Env) (witness : AppWitness) :
    Lifecycle.Stage1.Application.inputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (loadWitness env witness) =
      Lifecycle.Stage1.Application.inputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) env ∧
    Lifecycle.Stage1.Application.outputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (loadWitness env witness) =
      Lifecycle.Stage1.Application.outputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) env := by
  constructor
  · apply congrArg List.ofFn
    funext index
    exact loadWitness_agrees env witness _ (Or.inl (input_before_witness index))
  · apply congrArg List.ofFn
    funext index
    exact loadWitness_agrees env witness _ (Or.inl (output_before_witness index))

private theorem source_end : ApplicationDirectSource.sourceWidth application =
    Spartan.privateColumnCount + PerApplicationPackage.addedPrivateColumnCount application := by
  rw [Poseidon2HashChainV1Package.sourceWidth, Poseidon2HashChainV1Package.addedPrivateColumnCount]
  change Spartan.privateColumnCount + 4 + 7696 = Spartan.privateColumnCount + 7700
  omega

private theorem source_bound : ApplicationDirectSource.sourceWidth application ≤
    PiRLCProductPlan.baseSourceWidth application :=
  DirectApplicationPrefixPlan.applicationSourceWidth_le_baseSourceWidth application

private theorem base_constant : PerApplicationPackage.basePackage.layout.constantColumn =
    Spartan.privateColumnCount := by
  rw [Spartan.privateColumnCount_eq]
  exact Package.circuitPackage_layout_values.2.2.1

private theorem sourceCopied_of_built
    (target built : Env)
    (before : ∀ index, index < Spartan.privateColumnCount → built index = target index) :
    let suffix := fun index : Fin (PerApplicationPackage.addedPrivateColumnCount application) =>
      built (Spartan.privateColumnCount + index.val)
    ∀ index, ApplicationDirectSource.SourceAllowed application index →
      SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix) index = built index := by
  intro suffix
  let base := PerApplicationSourceAssignment.ofCompleted application target suffix
  have insert (index : Fin (PerApplicationPackage.addedPrivateColumnCount application)) :
      SourceCompiler.sourceEnv base (Spartan.privateColumnCount + index.val) =
        built (Spartan.privateColumnCount + index.val) := by
    have bound : Spartan.privateColumnCount + index.val < PiRLCProductPlan.baseSourceWidth application := by
      apply Nat.lt_of_lt_of_le _ source_bound
      rw [source_end]
      exact Nat.add_lt_add_left index.isLt _
    change SourceCompiler.sourceEnv base (⟨Spartan.privateColumnCount + index.val, bound⟩ :
      Fin (PiRLCProductPlan.baseSourceWidth application)).val = _
    rw [SourceCompiler.sourceEnv_at]
    exact PerApplicationSourceAssignment.application_ofCompleted application target suffix index
  have earlier (index : Nat) (below : index < Spartan.privateColumnCount) : SourceCompiler.sourceEnv base index = built index := by
    have bound : index < PiRLCProductPlan.baseSourceWidth application := by
      apply Nat.lt_of_lt_of_le _ source_bound
      rw [source_end]
      exact below.trans_le (Nat.le_add_right _ _)
    change SourceCompiler.sourceEnv base (⟨index, bound⟩ : Fin (PiRLCProductPlan.baseSourceWidth application)).val = _
    rw [SourceCompiler.sourceEnv_at]
    change PerApplicationSourceAssignment.ofCompleted application target suffix ⟨index, bound⟩ = built index
    rw [PerApplicationSourceAssignment.ofCompleted, dif_pos (by rw [base_constant]; exact below)]
    exact (before index below).symm
  have inside (index : Nat) (lower : Spartan.privateColumnCount ≤ index)
      (upper : index < ApplicationDirectSource.sourceWidth application) : SourceCompiler.sourceEnv base index = built index := by
    let slot : Fin (PerApplicationPackage.addedPrivateColumnCount application) :=
      ⟨index - Spartan.privateColumnCount, by rw [source_end] at upper; omega⟩
    have position : Spartan.privateColumnCount + slot.val = index := Nat.add_sub_of_le lower
    simpa only [position] using insert slot
  intro index support
  rcases support with input | witness | output | region
  · obtain ⟨position, rfl⟩ := input
    exact earlier _ (input_before_witness position)
  · obtain ⟨position, rfl⟩ := witness
    apply inside
    · exact Nat.le_add_right _ _
    · rw [source_end]
      have bound := position.isLt
      have fits : application.witnessWordCount ≤ PerApplicationPackage.addedPrivateColumnCount application := by
        rw [Poseidon2HashChainV1Package.addedPrivateColumnCount]
        decide
      simpa only [ApplicationInputs.witnessColumn, ApplicationInputs.witnessStart] using
        (Nat.add_lt_add_left (bound.trans_le fits) Spartan.privateColumnCount)
  · obtain ⟨position, rfl⟩ := output
    exact earlier _ (output_before_witness position)
  · exact inside index ((Nat.le_add_right _ _).trans region.1) region.2

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
      R1CS.RowsHold (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix))
        (ApplicationDirectSource.sourceRows application) ∧
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
  let loaded := loadWitness target witness
  have applicationHolds : Lifecycle.Stage1.Application.Holds application.step
      (ApplicationInputs.interface application) (ApplicationInputs.localStart application) loaded := by
    change Lifecycle.Stage1.Application.outputState _ _ loaded = application.step
      (Lifecycle.Stage1.Application.inputState _ _ loaded) (Lifecycle.Stage1.Application.witnessValue _ _ loaded)
    rw [(stateValues_loaded target witness).1, (stateValues_loaded target witness).2,
      witnessValue_loaded target witness width, inputValue, outputValue]
    exact step
  obtain ⟨built, agrees, logicalRows⟩ := application.completeness
    (ApplicationInputs.interface application) (ApplicationInputs.localStart application) loaded
    (application.assumptions _ _ loaded (ApplicationInputs.externalBelow application)) applicationHolds
  have physicalRows : R1CS.RowsHold built (ApplicationDirectSource.sourceRows application) := by
    rw [ApplicationDirectSource.sourceRows, ApplicationPackage.ofProgram_compiledRows_toR1CS,
      Poseidon2HashChainV1Package.constraints_eq_hashConstraints]
    exact Layout.Poseidon2.hashPhysical_complete _ _ built _ Poseidon2HashChainV1Package.hashInterface_affine logicalRows
  have before : ∀ index, index < Spartan.privateColumnCount → built index = target index := by
    intro index below
    exact (agrees index (Or.inl (below.trans_le (Nat.le_add_right _ _)))).trans
      (loadWitness_agrees target witness index (Or.inl below))
  let suffix := fun index : Fin (PerApplicationPackage.addedPrivateColumnCount application) =>
    built (Spartan.privateColumnCount + index.val)
  have copied := sourceCopied_of_built target built before
  refine ⟨suffix, R1CS.rowsHold_of_agree _ _ built _
    (ApplicationDirectSource.sourceRows_varsSatisfy application) copied physicalRows, ?_⟩
  have witnessSame : Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application)
      (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix)) =
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) built := by
    apply congrArg List.ofFn
    funext index
    exact copied _ (Or.inr (Or.inl ⟨index, rfl⟩))
  rw [witnessSame]
  have builtWitness : Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) built =
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) loaded := by
    apply congrArg List.ofFn
    funext index
    exact agrees _ (Or.inl (by
      change ApplicationInputs.witnessStart + index.val < ApplicationInputs.witnessStart + application.witnessWordCount
      exact Nat.add_lt_add_left index.isLt _))
  exact builtWitness.trans (witnessValue_loaded target witness width)

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

private theorem application_rowsZero_of_source_rows
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : R1CS.RowsHold (SourceCompiler.sourceEnv raw.base) (ApplicationDirectSource.sourceRows application)) :
    (ApplicationDirectPlan.plan Poseidon2HashChainV1Package.fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero raw.assignment := by
  apply (ApplicationDirectPlan.rowsZero_iff_rowsHold Poseidon2HashChainV1Package.fits.package
    (PerApplicationFixedPoint.geometry application) raw.assignment raw.applicationSource
    (PerApplicationCanonicalEncodes.encodes raw).applicationEncoding
    (PerApplicationCanonicalAssignment.assignment_one raw)).2
  apply R1CS.rowsHold_of_agree_below _ _ (SourceCompiler.sourceEnv raw.base)
    (ApplicationDirectPlan.sourceEnv raw.applicationSource)
    (ApplicationDirectSource.sourceRows_varsBelow application) _ rows
  intro index below
  change (if bounded : index < ApplicationRetainedBlocks.sourceWidth application then
      raw.applicationSource ⟨index, bounded⟩ else 0) = _
  have sourceBound : index < ApplicationRetainedBlocks.sourceWidth application := below
  rw [dif_pos sourceBound]
  exact DirectApplicationPrefixPlan.applicationSource_eq_sourceEnv application raw.base ⟨index, sourceBound⟩

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
      R1CS.RowsHold (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix))
        (ApplicationDirectSource.sourceRows application) ∧
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
  have applicationPlanRows := application_rowsZero_of_source_rows
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) applicationRows
  exact ⟨digestFixed, target, suffix, constant, physical, nextRows, nifsOutput, sources,
    priorWords, nextWords, applicationRows, actualWitness, outputDigest, publicInput, applicationPlanRows⟩

end NightstreamFPrime.Export.Stage1.ApplicationWitnessCompleteness
