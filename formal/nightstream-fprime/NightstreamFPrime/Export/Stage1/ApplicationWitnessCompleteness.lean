import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Export.Stage1.Stage1LoweringBridge
import NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan
import NightstreamFPrime.Export.Stage1.PerApplicationDecodedIO
import NightstreamFPrime.Layout.Stage1.ApplicationSemantics
import NightstreamFPrime.Layout.Stage1.StepPhysicalCompleteness

/-!
Owns construction of an application's private suffix from its real witness
and the state frames constructed by the physical prefix, for every application.
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
variable (application : Lifecycle.Stage1.Application.Program)

private def loadWitness (env : Env) (witness : AppWitness) : Env := fun index =>
  if ApplicationInputs.witnessStart ≤ index ∧
      index < ApplicationInputs.witnessStart + application.witnessWordCount then
    witness.getD (index - ApplicationInputs.witnessStart) 0
  else env index

private theorem loadWitness_read (env : Env) (witness : AppWitness)
    (index : Fin application.witnessWordCount) :
    loadWitness application env witness (ApplicationInputs.witnessStart + index.val) =
      witness.getD index.val 0 := by
  have inside : ApplicationInputs.witnessStart ≤ ApplicationInputs.witnessStart + index.val ∧
      ApplicationInputs.witnessStart + index.val <
        ApplicationInputs.witnessStart + application.witnessWordCount :=
    ⟨Nat.le_add_right _ _, Nat.add_lt_add_left index.isLt _⟩
  simp only [loadWitness, if_pos inside, Nat.add_sub_cancel_left]

private theorem loadWitness_agrees (env : Env) (witness : AppWitness) :
    AgreesOutside env (loadWitness application env witness) ApplicationInputs.witnessStart
      application.witnessWordCount := by
  intro index outside
  have notInside : ¬ (ApplicationInputs.witnessStart ≤ index ∧
      index < ApplicationInputs.witnessStart + application.witnessWordCount) := by
    omega
  simp only [loadWitness, if_neg notInside]

private theorem witnessValue_loaded (env : Env) (witness : AppWitness)
    (width : witness.length = application.witnessWordCount) :
    Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
      (ApplicationInputs.localStart application) (loadWitness application env witness) = witness := by
  change List.ofFn (fun index : Fin application.witnessWordCount =>
    loadWitness application env witness (ApplicationInputs.witnessStart + index.val)) = witness
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
        (ApplicationInputs.localStart application) (loadWitness application env witness) =
      Lifecycle.Stage1.Application.inputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) env ∧
    Lifecycle.Stage1.Application.outputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (loadWitness application env witness) =
      Lifecycle.Stage1.Application.outputState (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) env := by
  constructor
  · apply congrArg List.ofFn
    funext index
    exact loadWitness_agrees application env witness _ (Or.inl (input_before_witness index))
  · apply congrArg List.ofFn
    funext index
    exact loadWitness_agrees application env witness _ (Or.inl (output_before_witness index))

private theorem source_end : ApplicationDirectSource.sourceWidth application =
    Spartan.privateColumnCount + PerApplicationPackage.addedPrivateColumnCount application := by
  unfold ApplicationDirectSource.sourceWidth ApplicationPackage.r1csFreshStart
    PerApplicationPackage.addedPrivateColumnCount PerApplicationPackage.applicationPlan
  rw [ApplicationPackage.productionPlan_privateCount]
  change Spartan.privateColumnCount + application.witnessWordCount + _ + _ = _
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
      apply Nat.lt_of_lt_of_le _ (source_bound application)
      rw [source_end application]
      exact Nat.add_lt_add_left index.isLt _
    change SourceCompiler.sourceEnv base (⟨Spartan.privateColumnCount + index.val, bound⟩ :
      Fin (PiRLCProductPlan.baseSourceWidth application)).val = _
    rw [SourceCompiler.sourceEnv_at]
    exact PerApplicationSourceAssignment.application_ofCompleted application target suffix index
  have earlier (index : Nat) (below : index < Spartan.privateColumnCount) : SourceCompiler.sourceEnv base index = built index := by
    have bound : index < PiRLCProductPlan.baseSourceWidth application := by
      apply Nat.lt_of_lt_of_le _ (source_bound application)
      rw [source_end application]
      exact below.trans_le (Nat.le_add_right _ _)
    change SourceCompiler.sourceEnv base (⟨index, bound⟩ : Fin (PiRLCProductPlan.baseSourceWidth application)).val = _
    rw [SourceCompiler.sourceEnv_at]
    change PerApplicationSourceAssignment.ofCompleted application target suffix ⟨index, bound⟩ = built index
    rw [PerApplicationSourceAssignment.ofCompleted, dif_pos (by rw [base_constant]; exact below)]
    exact (before index below).symm
  have inside (index : Nat) (lower : Spartan.privateColumnCount ≤ index)
      (upper : index < ApplicationDirectSource.sourceWidth application) : SourceCompiler.sourceEnv base index = built index := by
    let slot : Fin (PerApplicationPackage.addedPrivateColumnCount application) :=
      ⟨index - Spartan.privateColumnCount, by rw [source_end application] at upper; omega⟩
    have position : Spartan.privateColumnCount + slot.val = index := Nat.add_sub_of_le lower
    simpa only [position] using insert slot
  intro index support
  rcases support with input | witness | output | region
  · obtain ⟨position, rfl⟩ := input
    exact earlier _ (input_before_witness position)
  · obtain ⟨position, rfl⟩ := witness
    apply inside
    · exact Nat.le_add_right _ _
    · rw [source_end application]
      have bound := position.isLt
      have fits : application.witnessWordCount ≤ PerApplicationPackage.addedPrivateColumnCount application := by
        unfold PerApplicationPackage.addedPrivateColumnCount
        exact Nat.le_add_right _ _
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
    (width : witness.length = application.witnessWordCount)
    (step : next.current = application.step prior.current witness)
    (valid : application.valid prior.current witness) :
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
      List.map_ofFn]
    exact represented_words prior priorFixed _ priorWords
  have nextRepresented : OutputHash.RepresentsPreimage PilotProduction.outputInterface
      PilotProduction.lifecycleOutputOffset (Spartan.pullback target) next := by
    unfold OutputHash.RepresentsPreimage
    rw [PilotProduction.outputInterface_preimage_apply]
    simp only [Gadgets.Poseidon2.Hash.evalList, PilotProduction.outputPreimage, PilotProduction.variableExprs,
      List.map_ofFn]
    exact represented_words next nextFixed _ nextWords
  have inputValue := ApplicationInputs.inputState_eq_current application target prior priorFixed priorRepresented
  have outputValue := ApplicationInputs.outputState_eq_current application target next nextFixed nextRepresented
  let interface := ApplicationInputs.interface application
  let localStart := ApplicationInputs.localStart application
  let loaded := loadWitness application target witness
  have applicationHolds : Lifecycle.Stage1.Application.Holds application.step interface localStart loaded := by
    change Lifecycle.Stage1.Application.outputState _ _ loaded = application.step
      (Lifecycle.Stage1.Application.inputState _ _ loaded) (Lifecycle.Stage1.Application.witnessValue _ _ loaded)
    rw [(stateValues_loaded application target witness).1, (stateValues_loaded application target witness).2,
      witnessValue_loaded application target witness width, inputValue, outputValue]
    exact step
  have applicationValid : Lifecycle.Stage1.Application.Valid application.valid interface localStart loaded := by
    change application.valid (Lifecycle.Stage1.Application.inputState _ _ loaded)
      (Lifecycle.Stage1.Application.witnessValue _ _ loaded)
    rw [(stateValues_loaded application target witness).1,
      witnessValue_loaded application target witness width, inputValue]
    exact valid
  have assumptions := application.assumptions interface localStart loaded
    (ApplicationInputs.externalBelow application)
  obtain ⟨built, agrees, logicalRows⟩ :=
    application.completeness interface localStart loaded assumptions
      ⟨applicationHolds, applicationValid⟩
  let constraints := ApplicationPackage.constraints application (ApplicationPackage.productionColumns application)
    localStart
  let freshStart := ApplicationPackage.r1csFreshStart application
    (ApplicationPackage.productionColumns application) localStart
  have scope : ∀ expression ∈ constraints, expression.VarsBelow freshStart :=
    application.scope interface localStart built (ApplicationInputs.externalBelow application)
      (application.assumptions interface localStart built (ApplicationInputs.externalBelow application))
  obtain ⟨lowered, loweredAgrees, loweredRows⟩ :=
    R1CS.lowerConstraints_complete built constraints freshStart scope logicalRows
  have physicalRows : R1CS.RowsHold lowered (ApplicationDirectSource.sourceRows application) := by
    rw [ApplicationDirectSource.sourceRows, ApplicationPackage.ofProgram_compiledRows_toR1CS]
    exact loweredRows
  have localBelow : localStart ≤ freshStart := Nat.le_add_right _ _
  have witnessBelow : ApplicationInputs.witnessStart + application.witnessWordCount = localStart := rfl
  have before : ∀ index, index < Spartan.privateColumnCount → lowered index = target index := by
    intro index below
    have belowLocal : index < localStart := by
      change index < ApplicationInputs.witnessStart + application.witnessWordCount
      exact below.trans_le (Nat.le_add_right _ _)
    exact (loweredAgrees index (Or.inl (belowLocal.trans_le localBelow))).trans
      ((agrees index (Or.inl belowLocal)).trans (loadWitness_agrees application target witness index (Or.inl below)))
  let suffix := fun index : Fin (PerApplicationPackage.addedPrivateColumnCount application) =>
    lowered (Spartan.privateColumnCount + index.val)
  have copied := sourceCopied_of_built application target lowered before
  refine ⟨suffix, R1CS.rowsHold_of_agree _ _ lowered _
    (ApplicationDirectSource.sourceRows_varsSatisfy application) copied physicalRows, ?_⟩
  have witnessSame : Lifecycle.Stage1.Application.witnessValue interface localStart
      (SourceCompiler.sourceEnv (PerApplicationSourceAssignment.ofCompleted application target suffix)) =
      Lifecycle.Stage1.Application.witnessValue interface localStart lowered := by
    apply congrArg List.ofFn
    funext index
    exact copied _ (Or.inr (Or.inl ⟨index, rfl⟩))
  rw [witnessSame]
  have loweredWitness : Lifecycle.Stage1.Application.witnessValue interface localStart lowered =
      Lifecycle.Stage1.Application.witnessValue interface localStart loaded := by
    apply congrArg List.ofFn
    funext index
    have inside : ApplicationInputs.witnessStart + index.val < localStart := by
      rw [← witnessBelow]
      exact Nat.add_lt_add_left index.isLt _
    exact (loweredAgrees _ (Or.inl (inside.trans_le localBelow))).trans (agrees _ (Or.inl inside))
  exact loweredWitness.trans (witnessValue_loaded application target witness width)

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
        RunningTransitionDirectPlan.transitionEnv_of_outside application raw.base index sourceBound
          (Or.inl (Nat.lt_of_lt_of_le beforeC PiCCSStarts.phaseOffset_le_statementWitnessStart))
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
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (raw : PerApplicationCanonicalAssignment.RawValues application)
    (rows : R1CS.RowsHold (SourceCompiler.sourceEnv raw.base) (ApplicationDirectSource.sourceRows application)) :
    (ApplicationDirectPlan.plan fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero raw.assignment := by
  apply (ApplicationDirectPlan.rowsZero_iff_rowsHold fits.package
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

/-- The witness width is the application's input ABI, also enforced by the
Rust encoder. -/
theorem complete
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
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
    (valid : application.valid input.zi input.witness)
    (priorWellFormed : StateEncoding.WellFormed (priorHashPreimage (setup relation ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed (nextHashPreimage (setup relation ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage (setup relation ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex)
    (witnessWidth : input.witness.length = application.witnessWordCount) :
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
      (ApplicationDirectPlan.plan fits.package
        (PerApplicationFixedPoint.geometry application)).RowsZero raw.assignment := by
  obtain ⟨digestFixed, target, constant, physical, nextRows, nifsOutput, sources, priorWords, nextWords⟩ :=
    StepPhysicalCompleteness.complete relation ajtai context input output result application step
      priorWellFormed nextWellFormed freshPublic accepted recursiveResult
  obtain ⟨suffix, applicationRows, actualWitness⟩ := complete_suffix application target
    (priorHashPreimage (setup relation ajtai context.toList) input)
    (nextHashPreimage (setup relation ajtai context.toList) input output)
    priorWellFormed.1 nextWellFormed.1 priorWords nextWords input.witness witnessWidth step.2.1 valid
  have outputDigest := outputDigest_of_completed application target suffix
    (priorHashPreimage (setup relation ajtai context.toList) input)
    (nextHashPreimage (setup relation ajtai context.toList) input output)
    output.x priorWellFormed.1 nextWellFormed.1 digestFixed
    (PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof) context
    (fun index below => sources index (Or.inl below))
  have publicInput := PerApplicationCanonicalAssignment.projectPublicInput_completeAssignment
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix))
  rw [outputDigest] at publicInput
  have applicationPlanRows := application_rowsZero_of_source_rows application fits
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target suffix)) applicationRows
  exact ⟨digestFixed, target, suffix, constant, physical, nextRows, nifsOutput, sources,
    priorWords, nextWords, applicationRows, actualWitness, outputDigest, publicInput, applicationPlanRows⟩

end NightstreamFPrime.Export.Stage1.ApplicationWitnessCompleteness
