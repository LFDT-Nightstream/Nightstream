import NightstreamFPrime.Layout.Stage1.PiDECProofInputs
import NightstreamFPrime.Layout.Stage1.PiRLCProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.PiRLCOutputRelocation

/-!
Owns the canonical local C/R/D completeness consumer. Actual NIFS acceptance
supplies the sampler executions and D checks; canonical D source loading
uses the real proof messages and verifier-computed public digits.
-/

namespace NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

private theorem available_of_sampleBatch
    (initial : Transcript.State) (count : Nat) (batch : Transcript.PiRlcSampler.Batch count)
    (success : Transcript.PiRlcSampler.sampleBatch initial count = some batch) :
    Available Transcript.PiRlcSampler.specification count candidateBound initial := by
  induction count with
  | zero => exact ⟨⟨fun coordinate => Fin.elim0 coordinate⟩, trivial⟩
  | succ count ih =>
      rw [Transcript.PiRlcSampler.sampleBatch] at success
      cases previous : Transcript.PiRlcSampler.sampleBatch initial count with
      | none => simp [previous] at success
      | some prior =>
          cases sampled : Transcript.PiRlcSampler.sampleRingChallenge initial count with
          | none => simp [previous, sampled] at success
          | some challenge =>
              obtain ⟨priorExecution, _⟩ := ih prior previous
              obtain ⟨scalar, scalarSample, _⟩ := Option.map_eq_some_iff.mp sampled
              obtain ⟨coefficients, coefficientSample, _⟩ := Option.map_eq_some_iff.mp scalarSample
              obtain ⟨execution, _⟩ := Sampling.FirstAccepted.BoundedExecution.exists_of_bounded_success
                coefficientSample
              refine ⟨{ execution := ?_ }, trivial⟩
              exact Fin.lastCases execution priorExecution.execution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem verifierInputs
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (output : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      running fresh proof = some output) :
    PiCCS.Accepted (ProductionKey.key relation ajtai) running fresh proof ∧
    Available Transcript.PiRlcSampler.specification PiRLC.v1_1.SamplerChain.sourceCount candidateBound
      ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState ∧
    ∃ challenges,
      (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof = some challenges ∧
      PiDEC.PaperVerifier.Accepted (ProductionKey.key relation ajtai).piDecAlgebra
        (ProductionKey.key relation ajtai).piDecPublicInputSplit
        (ProductionKey.key relation ajtai).piDecEvaluationArity
        ((ProductionKey.key relation ajtai).piDecAttemptForParent proof
          ((ProductionKey.key relation ajtai).parentForChallenges running fresh proof challenges)) := by
  let key := ProductionKey.key relation ajtai
  obtain ⟨cCheck, dCheck, _⟩ := (Nifs.PaperNonInteractive.verify_eq_some_iff
    key running fresh proof output).mp accepted
  obtain ⟨attempt, attemptEq, checks⟩ :=
    (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key running fresh proof).mp dCheck
  change (key.parent running fresh proof).map (key.piDecAttemptForParent proof) = some attempt at attemptEq
  obtain ⟨parent, parentEq, attemptValue⟩ := Option.map_eq_some_iff.mp attemptEq
  change (key.piRlcChallenges running fresh proof).map
    (key.parentForChallenges running fresh proof) = some parent at parentEq
  obtain ⟨challenges, sampled, parentValue⟩ := Option.map_eq_some_iff.mp parentEq
  have availability : Available Transcript.PiRlcSampler.specification
      PiRLC.v1_1.SamplerChain.sourceCount candidateBound
      (key.piCcsExecution running fresh proof).outgoingState := by
    change (Transcript.PiRlcSampler.piRlcChallengesWithState
      (key.piCcsExecution running fresh proof).outgoingState Nifs.PaperProfile.arity.total).map
        Transcript.PiRlcSampler.Batch.challenges = some challenges at sampled
    obtain ⟨batch, batchSample, _⟩ := Option.map_eq_some_iff.mp sampled
    exact available_of_sampleBatch _ _ batch batchSample
  refine ⟨cCheck, availability, challenges, sampled, ?_⟩
  rw [parentValue, attemptValue]
  exact checks

private theorem rEnd_before_dInputs :
    PiRLCInputs.phaseOffset + PiRLC.v1_1.Formal.logicalPrivateCount ≤ PiDECInputs.proofInputStart := by
  change PiRLCStarts.phaseFreshStart ≤ PiRLCStarts.outputFreshStart
  unfold PiRLCStarts.outputFreshStart PiRLCStarts.evalAFreshStart PiRLCStarts.evalKFreshStart
    PiRLCStarts.publicInputFreshStart PiRLCStarts.commitmentFreshStart PiRLCStarts.samplerFreshStart
  omega

private theorem point_below_dInputs (coordinate : Fin productionShape.cubeVariables) :
    ((PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)).point
      PiRLCInputs.phaseOffset coordinate).VarsBelow PiDECInputs.proofInputStart := by
  have pointEq :
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)).point
          PiRLCInputs.phaseOffset coordinate =
        RunningTransitionInputs.directRoundPoint PiCCSStarts.roundTranscriptWitnessStart coordinate := by
    exact RunningTransitionInputs.recursivePoint_eq_direct coordinate
  rw [pointEq, PiCCSStarts.roundTranscriptWitnessStart_eq]
  have coordinateBound : coordinate.val < 28 := coordinate.isLt
  simp only [RunningTransitionInputs.directRoundPoint, Quadratic.KExpr.VarsBelow, Expr.VarsBelow]
  norm_num [PiDECInputs.proofInputStart, PiRLCStarts.finalBoundaries_eq.2,
    RunningTransitionInputs.roundStride, RunningTransitionInputs.roundSampleC0Offset,
    RunningTransitionInputs.roundSampleC1Offset]
  omega

private theorem point_ext (left right : PaperAlgebra.Point)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem parent_preserved (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (parentPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiRLC.v1_1.Semantics.evalOutput relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset (PiDECProofInputs.load env proof parentPublic) =
    PiRLC.v1_1.Semantics.evalOutput relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env := by
  let interface := PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)
  let loaded := PiDECProofInputs.load env proof parentPublic
  have preserves := PiDECProofInputs.load_agreesOutside env proof parentPublic
  have pointEq : PiRLC.v1_1.InputBinding.evalPoint (interface.point PiRLCInputs.phaseOffset) env =
      PiRLC.v1_1.InputBinding.evalPoint (interface.point (PiRLCInputs.phaseOffset + 0)) loaded := by
    apply point_ext
    change (List.ofFn fun coordinate => (interface.point PiRLCInputs.phaseOffset coordinate).eval env) =
      List.ofFn fun coordinate => (interface.point PiRLCInputs.phaseOffset coordinate).eval loaded
    apply congrArg List.ofFn
    funext coordinate
    exact Quadratic.KExpr.eval_eq_of_agree_below _ PiDECInputs.proofInputStart env loaded
      (point_below_dInputs coordinate) (fun index below => (preserves index (Or.inl below)).symm)
  have same := PiRLCOutputRelocation.evalOutput_eq_of_shift_agreement relation interface interface
    PiRLCInputs.phaseOffset 0 env loaded pointEq (by
      intro index supported
      rcases supported with impossible | ⟨_, below⟩
      · exact False.elim impossible
      · exact preserves index (Or.inl (Nat.lt_of_lt_of_le below rEnd_before_dInputs)))
  exact same.symm

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem message_ext (left right : PiDEC.PaperVerifier.ChildMessage
    PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem loaded_message (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (parentPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (child : Phi81Relation.PiDECAlgebra.Radix.ChildIndex) :
    PiDEC.v1_1.InputBinding.evalMessage (PiDECInputs.message child)
      (PiDECProofInputs.load env proof parentPublic) =
      { commitment := proof.piDecCommitments child, evaluations := #[proof.piDecEvaluations child] } := by
  apply message_ext
  · funext row lane
    exact PiDECProofInputs.eval_childCommitment env proof parentPublic child row lane
  · apply congrArg (fun value : PaperAlgebra.Evaluation => #[value])
    apply evaluation_ext
    · funext coefficient
      exact PiDECProofInputs.eval_childEvalK env proof parentPublic child coefficient
    · funext matrix coefficient
      exact PiDECProofInputs.eval_childEvalA env proof parentPublic child matrix coefficient

private theorem attempt_ext
    (left right : PiDEC.v1_1.InputBinding.Attempt logicalWidth publicFits)
    (parent : left.parent = right.parent) (messages : left.messages = right.messages) : left = right := by
  cases left
  cases right
  simp_all

private theorem loaded_attempt (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (parentPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiDEC.v1_1.Semantics.inputAttempt relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset (PiDECProofInputs.load env proof parentPublic) =
      (ProductionKey.key relation ajtai).piDecAttemptForParent proof
        (PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset env) := by
  apply attempt_ext
  · exact (AccumulatorSemantics.piDecParent_eq_piRlcOutput relation _).trans
      (parent_preserved relation env proof parentPublic)
  · funext child
    exact loaded_message relation env proof parentPublic child

private theorem instance_ext
    (left right : PiDEC.v1_1.OutputBinding.Output logicalWidth publicFits)
    (system : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem loaded_output (env : Env) (proof : Proof (ProductionKey.degreeBound relation)) :
    let parent := PiRLC.v1_1.Semantics.evalOutput relation
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env
    PiDEC.v1_1.Semantics.output relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset (PiDECProofInputs.load env proof parent.publicInput) =
      PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        ((ProductionKey.key relation ajtai).piDecAttemptForParent proof parent) := by
  dsimp only
  let parent := PiRLC.v1_1.Semantics.evalOutput relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCInputs.phaseOffset env
  funext child
  apply instance_ext
  · rfl
  · funext row lane
    exact PiDECProofInputs.eval_childCommitment env proof parent.publicInput child row lane
  · funext coordinate
    exact PiDECProofInputs.eval_childPublicInput env proof parent.publicInput child coordinate
  · exact congrArg (fun value => value.point) (parent_preserved relation env proof parent.publicInput)
  · exact congrArg (fun message => message.evaluations)
      (loaded_message relation env proof parent.publicInput child)
  · rfl

private theorem children_outputAccepted
    (attempt : PiDEC.v1_1.InputBinding.Attempt logicalWidth publicFits)
    (checks : PiDEC.PaperVerifier.Accepted (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai) (PaperAlgebra.evaluationArity ajtai) attempt) :
    PiDEC.PaperVerifier.OutputAccepted (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai) (PaperAlgebra.evaluationArity ajtai)
      attempt.parent (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai) attempt) := by
  have recovered : PiDEC.PaperVerifier.attemptForOutput attempt.parent
      (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai) attempt) = attempt := by
    cases attempt
    rfl
  exact ⟨by rw [recovered], by rw [recovered]; exact checks⟩

private theorem loaded_phase (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (checks : PiDEC.PaperVerifier.Accepted (ProductionKey.key relation ajtai).piDecAlgebra
      (ProductionKey.key relation ajtai).piDecPublicInputSplit
      (ProductionKey.key relation ajtai).piDecEvaluationArity
      ((ProductionKey.key relation ajtai).piDecAttemptForParent proof
        (PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset env))) :
    PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset (PiDECProofInputs.load env proof
        (PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset env).publicInput) := by
  unfold PiDEC.v1_1.Semantics.PhaseHolds
  rw [loaded_attempt relation ajtai, loaded_output relation ajtai]
  exact children_outputAccepted ajtai _ checks

private theorem computed_output_eq
    (key : ProductionKey.KeyType relation)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (output : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (challenges : Fin key.arity.total → RingF)
    (sampled : key.piRlcChallenges running fresh proof = some challenges)
    (checks : PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit key.piDecEvaluationArity
      (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges)))
    (accepted : Nifs.PaperNonInteractive.verify key running fresh proof = some output) :
    key.outputForAttempt proof
      (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges))
      (key.piDecPublicInputSplit.split (key.parentForChallenges running fresh proof challenges).publicInput) =
      output := by
  have acceptedOutput := (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh proof output).mp accepted |>.2.2
  have computed : key.output running fresh proof = some
      (key.outputForAttempt proof
        (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges))
        (key.piDecPublicInputSplit.split (key.parentForChallenges running fresh proof challenges).publicInput)) := by
    simp only [Nifs.PaperNonInteractive.Key.output, Nifs.PaperNonInteractive.Key.piDecAttempt,
      Nifs.PaperNonInteractive.Key.parent, sampled, Option.map_some, Option.bind_some]
    rw [PiDEC.PaperVerifier.PublicInputSplit.checked_eq_some _ _ checks.parentBounded]
    rfl
  exact Option.some.inj (computed.symm.trans acceptedOutput)

private theorem running_ext
    (left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point) (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs) (evaluations : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem running_of_children (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (parent : PiDEC.v1_1.OutputBinding.Output logicalWidth publicFits)
    (outputs : PiDEC.v1_1.Semantics.output relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env = PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        ((ProductionKey.key relation ajtai).piDecAttemptForParent proof parent)) :
    RunningTransitionInputs.piDecRunningOutput relation env =
      (ProductionKey.key relation ajtai).outputForAttempt proof
        ((ProductionKey.key relation ajtai).piDecAttemptForParent proof parent)
        ((ProductionKey.key relation ajtai).piDecPublicInputSplit.split parent.publicInput) := by
  apply running_ext
  · exact congrArg (fun values => (values ⟨0, by decide⟩).point) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).commitment) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).publicInput) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).evaluations.getD
      0 PaperAlgebra.evaluationZero) outputs

variable
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (priorPublic : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
  (advertised : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (digest : Digest)
  (priorFixed : PilotProduction.FixedPreimage prior)
  (advertisedFixed : PilotProduction.FixedPreimage advertised)
  (digestFixed : digest.length = PilotProduction.digestWords)
  (values : PiCCSProofInputs.ProofValues) (context : VerifierContext.Digest4)
  (template : Proof 9)

/-- An actual accepted NIFS run constructs the canonical local C/R/D
prefixes and their exact running output. C acceptance, sampler availability,
and the D parent bound all follow from that run. State framing and context are
the outer circuit's fixed input prerequisites. No child opening, generated
phase specification, or generated output value is a premise. -/
theorem completePrefix
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (priorPc : prior.pc = 1) (advertisedPc : advertised.pc = 1)
    (priorContext : prior.verifierKeys functionIndex = context.toList)
    (advertisedContext : advertised.verifierKeys functionIndex = context.toList)
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      (prior.running functionIndex)
      (PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values)
      (PiCCSProofInputs.relationProof relation values template) = some result) :
    ∃ c : Sequence.Prefix
        (PiCCSProtocolCompleteness.environment prior priorPublic advertised digest
          priorFixed advertisedFixed digestFixed values context) PiCCSInputs.phaseOffset,
      ∃ r : Sequence.Prefix c.current PiRLCInputs.phaseOffset,
        ∃ d : Sequence.Prefix
            (PiDECProofInputs.load r.current (PiCCSProofInputs.relationProof relation values template)
              (PiRLC.v1_1.Semantics.evalOutput relation
                (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
                PiRLCInputs.phaseOffset r.current).publicInput) PiDECInputs.phaseOffset,
          c.operations = PiCCS.v1_1.Formal.opsAt relation (PiCCSProofInputs.relationInterface relation)
            PiCCSInputs.phaseOffset ∧
          r.operations = PiRLC.v1_1.Formal.opsAt relation
            (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
            PiRLCInputs.phaseOffset ∧
          d.operations = PiDEC.v1_1.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset ∧
          holdsFlat d.current c.operations ∧ holdsFlat d.current r.operations ∧
          PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset d.current ∧
          RunningTransitionInputs.piDecRunningOutput relation d.current = result := by
  let proof := PiCCSProofInputs.relationProof relation values template
  let fresh := PiCCSProofInputs.protocolFresh logicalWidth publicFits priorPublic values
  obtain ⟨cAccepted, available, challenges, sampled, checks⟩ := verifierInputs relation ajtai
    (prior.running functionIndex) fresh proof result accepted
  obtain ⟨c, r, cOperations, rOperations, cRowsAtR, _, rSampled, rParent⟩ :=
    PiRLCProtocolCompleteness.completePrefix relation ajtai prior priorPublic advertised digest
      priorFixed advertisedFixed digestFixed values context template priorPc advertisedPc
      priorContext advertisedContext cAccepted available
  have challengeEq : PiRLC.v1_1.Semantics.evalChallenges
      (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset r.current = challenges := Option.some.inj (rSampled.symm.trans sampled)
  rw [challengeEq] at rParent
  let parent := PiRLC.v1_1.Semantics.evalOutput relation
    (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCInputs.phaseOffset r.current
  let loaded := PiDECProofInputs.load r.current proof parent.publicInput
  have parentChecks : PiDEC.PaperVerifier.Accepted (ProductionKey.key relation ajtai).piDecAlgebra
      (ProductionKey.key relation ajtai).piDecPublicInputSplit
      (ProductionKey.key relation ajtai).piDecEvaluationArity
      ((ProductionKey.key relation ajtai).piDecAttemptForParent proof parent) := by
    change PiDEC.PaperVerifier.Accepted _ _ _
      ((ProductionKey.key relation ajtai).piDecAttemptForParent proof
        (PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset r.current))
    rw [rParent]
    exact checks
  have loadedPhase := loaded_phase relation ajtai r.current proof parentChecks
  obtain ⟨d, dOperations⟩ := PiDEC.v1_1.Formal.completePrefix relation ajtai
    (PiDECInputs.interface logicalWidth publicFits) loaded PiDECInputs.phaseOffset
    (PiDECInputs.assumptions relation loaded) loadedPhase
  have dRows : holds d.current (Circuit.ops
      (PiDEC.v1_1.Formal.main relation (PiDECInputs.interface logicalWidth publicFits)) PiDECInputs.phaseOffset) := by
    change holds d.current (PiDEC.v1_1.Formal.opsAt relation
      (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset)
    rw [← dOperations]
    exact holdsFlat_implies_holds d.current d.operations d.rows
  have dPhase := PiDEC.v1_1.Semantics.spec_implies_phaseHolds relation ajtai
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset d.current
    (PiDEC.v1_1.Formal.soundness relation (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset d.current (PiDECInputs.assumptions relation d.current) dRows)
  have beforeInputs : ∀ index, index < PiDECInputs.proofInputStart → d.current index = r.current index := by
    intro index below
    have beforeD : index < PiDECInputs.phaseOffset :=
      Nat.lt_of_lt_of_le below (Nat.le_add_right _ _)
    exact (d.agrees index (Or.inl beforeD)).trans
      (PiDECProofInputs.load_agreesOutside r.current proof parent.publicInput index (Or.inl below))
  have cEnd : PiCCSInputs.phaseOffset + localLength c.operations ≤ PiDECInputs.proofInputStart := by
    rw [cOperations, ← PiCCS.v1_1.Formal.main_ops, PiCCS.v1_1.Formal.localLength_eq]
    change NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset ≤ _
    rw [← PiCCSStarts.logicalFreshBase_eq_layout relation]
    exact Nat.le_trans PiRLCInputs.piCcsLogicalFreshBase_le_phaseOffset
      (Nat.le_trans (Nat.le_add_right _ _) rEnd_before_dInputs)
  have rEnd : PiRLCInputs.phaseOffset + localLength r.operations ≤ PiDECInputs.proofInputStart := by
    rw [rOperations, ← PiRLC.v1_1.Formal.main_ops, PiRLC.v1_1.Formal.localLength_eq]
    exact rEnd_before_dInputs
  have cRows : holdsFlat d.current c.operations := by
    intro expression member
    have same := expression.eval_eq_of_agree_below
      (PiCCSInputs.phaseOffset + localLength c.operations) d.current r.current
      (c.scope expression member) (fun index below => beforeInputs index (Nat.lt_of_lt_of_le below cEnd))
    exact same.trans (cRowsAtR expression member)
  have rRows : holdsFlat d.current r.operations := by
    intro expression member
    have same := expression.eval_eq_of_agree_below
      (PiRLCInputs.phaseOffset + localLength r.operations) d.current r.current
      (r.scope expression member) (fun index below => beforeInputs index (Nat.lt_of_lt_of_le below rEnd))
    exact same.trans (r.rows expression member)
  have outputPreserved := PiDEC.v1_1.Semantics.output_eq_of_agree relation
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset loaded d.current
    (PiDECInputs.assumptions relation loaded) (fun index below => (d.agrees index (Or.inl below)).symm)
  have family := outputPreserved.symm.trans (loaded_output relation ajtai r.current proof)
  have runningOutput := running_of_children relation ajtai d.current proof parent family
  have actualOutput := computed_output_eq relation (ProductionKey.key relation ajtai)
    (prior.running functionIndex) fresh proof result challenges sampled checks accepted
  have parentIdentity : parent = (ProductionKey.key relation ajtai).parentForChallenges
      (prior.running functionIndex) fresh proof challenges := rParent
  rw [parentIdentity] at runningOutput
  exact ⟨c, r, d, cOperations, rOperations, dOperations, cRows, rRows, dPhase,
    runningOutput.trans actualOutput⟩

end NightstreamFPrime.Layout.Stage1.PiDECProtocolCompleteness
