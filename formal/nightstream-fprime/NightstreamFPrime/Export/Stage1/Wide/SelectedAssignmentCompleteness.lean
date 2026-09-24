import NightstreamFPrime.Export.Stage1.Wide.CompletedRows
import NightstreamFPrime.Export.Stage1.Wide.CarrierAssignment
import NightstreamFPrime.Layout.Stage1.Wide.StepPhysicalCompleteness

/-! Construct the candidate's complete compact witness from its semantic step
and accepted wide-key NIFS advice. All rows, the carrier norm, public digest
and unchanged application advice refer to this one constructed assignment. -/

namespace NightstreamFPrime.Export.Stage1.Wide.SelectedAssignmentCompleteness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.HyperNova.Construction2.Paper
open Layout.Stage1 ProductionRelation
open Poseidon2HashChainV1Package (application fits)

private theorem current_words {width : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (env : Env) (start : Nat)
    (value : HashPreimage (logicalWidth := width) (publicFits := publicFits))
    (wellFormed : StateEncoding.WellFormed value)
    (words : ∀ index : Fin PilotProduction.stateHashWords, env (start + index.val) =
      (serializePreimage (publicFits := publicFits) value).getD index.val 0) :
    (List.ofFn fun lane : Fin 4 => env (start + 35 + lane.val)) = value.current := by
  have decoded := StateEncodingReadback.preimage_eq_of_words value wellFormed
    (fun index => env (start + index)) words
  have current := congrArg (fun preimage => preimage.current) decoded
  change (List.ofFn fun lane : Fin 4 => env (start + (35 + lane.val))) = value.current at current
  simpa only [Nat.add_assoc] using current

private theorem digest_words {width : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (env : Env) (prior next : HashPreimage (logicalWidth := width) (publicFits := publicFits))
    (digest : Digest) (priorFixed : PilotProduction.FixedPreimage prior)
    (nextFixed : PilotProduction.FixedPreimage next) (digestFixed : digest.length = PilotProduction.digestWords)
    (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
    (sources : ∀ index, index < PilotProduction.witnessOffset → env index =
      PiCCSProtocolCompleteness.environment prior (encHash (stateHash prior)) next digest
        priorFixed nextFixed digestFixed values context index) :
    (List.ofFn fun lane : Fin PilotProduction.digestWords => env (PilotProduction.outputDigestStart + lane.val)) = digest := by
  have represented := PilotProduction.protocolEnv_represents_of_agreesBelow prior (encHash (stateHash prior))
    next digest priorFixed nextFixed digestFixed env (fun index below => (sources index below).trans
      (PiCCSProtocolCompleteness.pilot_word prior (encHash (stateHash prior)) next digest
        priorFixed nextFixed digestFixed values context index below))
  exact represented.2.2.2.symm

/-- The wide-key semantic step constructs a bounded complete CCS witness.
There are no physical-row, sampler-success or witness-transport premises. -/
theorem complete (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (ajtai : AjtaiKey
      (logicalWidth := RetainedLayout.logicalWidth application)
      (publicFits := FixedPoint.publicFits application))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application))
      (Fresh (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application))
      (Lifecycle.Proof 9) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application)) slotCount)
    (result : Running
      (logicalWidth := RetainedLayout.logicalWidth application)
      (publicFits := FixedPoint.publicFits application))
    (step : Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation application compiled fits)
      ajtai context.toList application input output)
    (priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits)
        ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits)
        ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage
        (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits) ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify
      (PiRLC.Wide.Key.key (FixedPoint.relation application compiled fits) ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex)
    (witnessWidth : input.witness.length = Stage1.Poseidon2HashChainV1.messageWordCount) :
    ∃ (env : Env) (message : Fin 4 → F),
      let suffix := ApplicationCompletedAssignment.suffix env message
      let assignment := SourceAssignment.assignment application env suffix
      let carrier := CarrierAssignment.values application env suffix
      (FixedPoint.structuralPlan application compiled fits).RowsZero assignment ∧
      (∀ column, centeredMagnitude (carrier column) < 2) ∧
      Phi81Relation.projectPublicInput carrier = encHash output.x ∧
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application)
        (SourceCompiler.sourceEnv (SourceAssignment.raw application env suffix).base) = input.witness ∧
      (SourceAssignment.raw application env suffix).outputDigest = output.x := by
  let relation := FixedPoint.relation application compiled fits
  obtain ⟨digestFixed, env, physical, nextSpec, _result, sources, priorWords, nextWords⟩ :=
    Layout.Stage1.Wide.StepPhysicalCompleteness.complete relation ajtai context input output result
      step priorWellFormed nextWellFormed freshPublic accepted recursiveResult
  let prior := priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input
  let next := nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup relation ajtai context.toList) input output
  let values := PiCCSProofReadback.ofProof (input.fresh.commitments ⟨0, by decide⟩) input.nifsProof
  let message : Fin 4 → F := fun lane => input.witness.getD lane.val 0
  have messageEq : List.ofFn message = input.witness := by
    apply List.ext_get
    · rw [List.length_ofFn, witnessWidth]
      rfl
    · intro index leftBound rightBound
      rw [List.get_ofFn]
      exact List.getD_eq_getElem (l := input.witness) (d := 0) rightBound
  have inputValues := current_words env PilotProduction.priorPreimageStart prior priorWellFormed priorWords
  have outputValues := current_words env PilotProduction.outputPreimageStart next nextWellFormed nextWords
  have appStep : (List.ofFn fun lane : Fin 4 => env (ApplicationInputs.outputSourceColumn lane)) =
      application.step (List.ofFn fun lane : Fin 4 => env (ApplicationInputs.inputSourceColumn lane)) (List.ofFn message) := by
    change (List.ofFn fun lane : Fin 4 => env (PilotProduction.outputPreimageStart + 35 + lane.val)) =
      application.step (List.ofFn fun lane : Fin 4 => env (PilotProduction.priorPreimageStart + 35 + lane.val)) (List.ofFn message)
    rw [outputValues, inputValues, messageEq]
    exact step.2.1
  let suffix := ApplicationCompletedAssignment.suffix env message
  have rows := CompletedRows.rowsZero compiled relation ajtai input.nifsProof env message
    (PilotProduction.layoutAssumptions env)
    (Layout.PiCCS.v1_1.Assumptions.production relation
      (PiCCSInputs.interface _ _) PiCCSInputs.phaseOffset (PiCCSInputs.externalInputsLinear _ _) env)
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation env)
    (Layout.Stage1.Wide.PiDECInputs.assumptions relation env) physical nextSpec appStep
  rw [FixedPoint.plan_fixedPoint] at rows
  have running := ((Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation env).mp physical).2
  have digest := digest_words env prior next output.x priorWellFormed.1 nextWellFormed.1 digestFixed values context
    (fun index below => sources index (Or.inl below))
  have publicOutput := CarrierAssignment.publicOutput application env suffix
  rw [digest] at publicOutput
  have advice := (ApplicationCompletedAssignment.advice env message).trans messageEq
  have outputDigest := (SourceAssignment.outputDigest application env suffix).trans digest
  exact ⟨env, message, rows, CarrierAssignment.norm application env suffix relation running,
    publicOutput, advice, outputDigest⟩

end NightstreamFPrime.Export.Stage1.Wide.SelectedAssignmentCompleteness
