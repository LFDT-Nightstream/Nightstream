import NightstreamFPrime.Spec.Folding.Nifs.PaperCausalReplay
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakCompleteness

/-!
Honest completeness for the existing one-message NIFS verifier. The causal
PiCCS prover generates its messages on the verifier's own transcript. After
actual PiRLC sampler success, the honest weak suffix supplies the PiDEC
messages and witnesses. No probability or executable work bound is asserted.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)
open PiRLC.CoordinateForkLaw

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}
  (key : Key K Commitment PublicInput Scalar State shape columns blockCount width)
  (running : Running K Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)

private def replayCoins (messages : Fin shape.cubeVariables → FixedPolynomial K width) :
    FiatShamir.DerivedCoins K State shape :=
  FiatShamir.derive key.oracle.transcript
    ({ priorState := key.publicInputState running fresh
       input := (key.statement running fresh).verifierInput key.lift } :
      PiCCS.TranscriptReplay.Statement K State shape)
    { rounds := fun round => (messages round).toMessage }

private def replayProbe (messages : Fin shape.cubeVariables → FixedPolynomial K width)
    (fullOutput : FullOutputCoordinates.FullOutput K shape) : Probe K shape :=
  let coins := replayCoins key running fresh messages
  { coins := { alpha := coins.alpha, gamma := coins.gamma, roundPoint := coins.roundPoint }
    response := {
      rounds := FixedPhase.RawCertificate.encode { rounds := List.ofFn messages }
      fullOutput := fullOutput } }

private theorem run_probe_eq (prover : CausalExecution.Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables)
    (probe : Probe K shape) (witness : OutputWitness shape columns)
    (rounds : List (FixedPolynomial K width))
    (returned : CausalExecution.run prover alpha gamma point = some (probe, witness))
    (receipt : issued (prover.rounds alpha gamma) [] point.coordinates = some rounds) :
    probe = {
      coins := { alpha, gamma, roundPoint := point }
      response := {
        rounds := FixedPhase.RawCertificate.encode { rounds }
        fullOutput := probe.response.fullOutput } } := by
  cases outputEq : prover.output alpha gamma point.coordinates with
  | none =>
      simp only [CausalExecution.run, receipt, outputEq, Option.map_none] at returned
      cases returned
  | some output =>
      have equal :
          (({ coins := { alpha, gamma, roundPoint := point }
              response := {
                rounds := FixedPhase.RawCertificate.encode { rounds }
                fullOutput := output.1 } } : Probe K shape), output.2) =
            (probe, witness) := by
        apply Option.some.inj
        simpa only [CausalExecution.run, receipt, outputEq, Option.map_some] using returned
      have probeEq := congrArg Prod.fst equal
      have fullOutputEq : output.1 = probe.response.fullOutput :=
        congrArg (fun value : Probe K shape => value.response.fullOutput) probeEq
      rw [fullOutputEq] at probeEq
      exact probeEq.symm

private theorem exists_honest_prefix (witness : OutputWitness shape columns)
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness) :
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K width)
      (fullOutput : FullOutputCoordinates.FullOutput K shape),
      (replayProbe key running fresh messages fullOutput).FixedWidthAccepted
        key.extensionOps key.lift (key.statement running fresh) width ∧
      ∀ source, CE.Holds key.piRlcSemantics key.params
        ((key.statement running fresh).publicOutput
          (replayProbe key running fresh messages fullOutput) source)
        (witness.assignments source) := by
  obtain ⟨prover, honest⟩ :=
    PaperStrongCompleteness.exists_honest_piCcs_prover key running fresh witness valid
  let context : PiCCS.TranscriptReplay.Statement K State shape := {
    priorState := key.publicInputState running fresh
    input := (key.statement running fresh).verifierInput key.lift }
  let pre := FiatShamir.derivePreSumcheck key.oracle.transcript context
  have total : ∀ point : CubePoint K shape.cubeVariables,
      ∃ (probe : Probe K shape) (value : OutputWitness shape columns),
        CausalExecution.run prover pre.alpha pre.gamma point = some (probe, value) := by
    intro point
    obtain ⟨probe, returned, _, _⟩ := honest pre.alpha pre.gamma point
    exact ⟨probe, witness, returned⟩
  obtain ⟨messages, _, _, receipt, _⟩ := PaperCausalReplay.prover_generated_messages_replay
    key.oracle.transcript prover pre.alpha pre.gamma pre.state total
  let coins := replayCoins key running fresh messages
  obtain ⟨probe, returned, accepted, openings⟩ := honest pre.alpha pre.gamma coins.roundPoint
  have actualReceipt : issued (prover.rounds pre.alpha pre.gamma) [] coins.roundPoint.coordinates =
      some (List.ofFn messages) := receipt
  have same : replayProbe key running fresh messages probe.response.fullOutput = probe :=
    (run_probe_eq prover pre.alpha pre.gamma coins.roundPoint probe witness
      (List.ofFn messages) returned actualReceipt).symm
  refine ⟨messages, probe.response.fullOutput, ?_, ?_⟩
  · rw [same]
    exact accepted
  · rw [same]
    exact openings

private theorem piCcsProbe_eq_replay (proof : Proof K Commitment shape width) :
    key.piCcsProbe running fresh proof =
      replayProbe key running fresh proof.piCcsRounds proof.piCcsOutput := by
  have raw : (key.piCcsCertificate running fresh proof).toFinite =
      FixedPhase.RawCertificate.encode { rounds := List.ofFn proof.piCcsRounds } := by
    simp only [Key.piCcsCertificate, PiCCS.TranscriptReplay.Certificate.toFinite,
      PiCCS.TranscriptReplay.Certificate.toTranscript, FiatShamir.Certificate.toFinite,
      FixedPhase.RawCertificate.encode, List.map_ofFn, Function.comp_def]
  simp only [Key.piCcsProbe, raw, key.piCcsExecution_coins_eq_derive,
    replayProbe, replayCoins]

private theorem singleton_of_size_one {Value : Type*} (values : Array Value)
    (size : values.size = 1) :
    #[values[0]'(by omega)] = values := by
  obtain ⟨value, equal⟩ := Array.size_eq_one_iff.mp size
  subst values
  rfl

private theorem childMessage_ext {Evaluation : Type*}
    (left right : PiDEC.PaperVerifier.ChildMessage Evaluation Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  cases commitment
  cases evaluations
  rfl

private theorem package_suffix
    (messages : Fin shape.cubeVariables → FixedPolynomial K width)
    (fullOutput : FullOutputCoordinates.FullOutput K shape)
    (witness : OutputWitness shape columns)
    (accepted : (replayProbe key running fresh messages fullOutput).FixedWidthAccepted
      key.extensionOps key.lift (key.statement running fresh) width)
    (openings : ∀ source, CE.Holds key.piRlcSemantics key.params
      ((key.statement running fresh).publicOutput
        (replayProbe key running fresh messages fullOutput) source)
      (witness.assignments source))
    (rho : Fin key.arity.total → Scalar)
    (sampled : key.piRlcResponse
      (key.absorbPiCcsOutput (replayCoins key running fresh messages).finalState fullOutput) = some rho) :
    ∃ (proof : Proof K Commitment shape width)
      (result : Running K Commitment PublicInput shape)
      (childWitnesses : Fin shape.runningCount → PaperLinearAlgebra.Assignment F columns),
      proof.piCcsRounds = messages ∧ proof.piCcsOutput = fullOutput ∧
      key.piRlcChallenges running fresh proof = some rho ∧
      verify key running fresh proof = some result ∧
      ∀ child, CE.Holds key.piRlcSemantics key.params
        (PiDEC.OutputWitnessConsumer.runningStatement key result child) (childWitnesses child) := by
  let probe := replayProbe key running fresh messages fullOutput
  let batch := PaperStrongInterface.piRlcBatchForProbe key running fresh probe
  let vector : Fin key.arity.total → Challenge key.piRlcAlgebra := fun index =>
    ⟨rho index, key.piRlcResponseValid _ rho sampled index⟩
  let assignments : Fin key.arity.total → PaperLinearAlgebra.Assignment F columns :=
    fun index => witness.assignments (Fin.cast key.total_eq_sourceCount index)
  let reply := PaperWeakCompleteness.honestReply key.piRlcAlgebra batch key.piDecAlgebra vector assignments
  let attempt := PaperWeakSuffix.attempt key.piRlcAlgebra batch vector reply
  have inputFresh : ∀ index, (batch.inputs index).stage = .fresh := fun _ => rfl
  have inputHolds : ∀ index, CE.Holds key.piRlcSemantics key.params
      (batch.inputs index) (assignments index) :=
    fun index => openings (Fin.cast key.total_eq_sourceCount index)
  have complete := PaperWeakCompleteness.honest_complete key.piRlcAlgebra batch key.piDecAlgebra
    key.piDecPublicInputSplit key.piDecEvaluationArity vector assignments inputFresh inputHolds
  have messageSize : ∀ child, (reply.messages child).evaluations.size = 1 := by
    intro child
    exact key.piRlcEvaluationsSize _ _ _
  let proof : Proof K Commitment shape width := {
    piCcsRounds := messages
    piCcsOutput := fullOutput
    piDecCommitments := fun child =>
      (reply.messages (Fin.cast key.outputCount_eq.symm child)).commitment
    piDecEvaluations := fun child =>
      (reply.messages (Fin.cast key.outputCount_eq.symm child)).evaluations[0]'(by
        rw [messageSize]
        decide) }
  have probeEq : key.piCcsProbe running fresh proof = probe :=
    piCcsProbe_eq_replay key running fresh proof
  have sampleEq : key.piRlcChallenges running fresh proof = some rho := by
    unfold Key.piRlcChallenges
    rw [key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput,
      key.piCcsExecution_coins_eq_derive]
    exact sampled
  have scalarEq : scalarVector key.piRlcAlgebra vector = rho := rfl
  have parentEq : key.parentForChallenges running fresh proof rho = attempt.parent := by
    change PiRLC.combinedOutput key.piRlcAlgebra key.relationSource
      (key.piCcsProbe running fresh proof).coins.roundPoint
      (fun index => (key.statement running fresh).publicOutput
        (key.piCcsProbe running fresh proof) (Fin.cast key.total_eq_sourceCount index)) rho = _
    rw [probeEq, ← scalarEq]
    rfl
  have messagesEq : (key.piDecAttemptForParent proof attempt.parent).messages = attempt.messages := by
    funext child
    have castBack : Fin.cast key.outputCount_eq.symm (Fin.cast key.outputCount_eq child) = child :=
      Fin.ext rfl
    apply childMessage_ext
    · change (reply.messages
        (Fin.cast key.outputCount_eq.symm (Fin.cast key.outputCount_eq child))).commitment =
        (reply.messages child).commitment
      rw [castBack]
    · change #[proof.piDecEvaluations (Fin.cast key.outputCount_eq child)] =
        (reply.messages child).evaluations
      dsimp only [proof]
      simp only [castBack]
      exact singleton_of_size_one (reply.messages child).evaluations (messageSize child)
  have attemptValue : key.piDecAttemptForParent proof attempt.parent = attempt := by
    exact congrArg (fun value => PiDEC.PaperVerifier.Attempt.mk attempt.parent value) messagesEq
  have attemptEq : key.piDecAttempt running fresh proof = some attempt := by
    unfold Key.piDecAttempt Key.parent
    rw [sampleEq]
    simp only [Option.map_some]
    rw [parentEq, attemptValue]
  let result := key.outputForAttempt proof attempt
    (key.piDecPublicInputSplit.split attempt.parent.publicInput)
  have verified : verify key running fresh proof = some result := by
    apply (verify_eq_some_iff key running fresh proof result).mpr
    refine ⟨?_, ?_, ?_⟩
    · apply (piCcsCheck_eq_true_iff_fixedWidthAccepted key running fresh proof).mpr
      rw [probeEq]
      exact accepted
    · exact (piDecCheck_eq_true_iff key running fresh proof).mpr ⟨attempt, attemptEq, complete.1⟩
    · exact key.output_eq_some_of_parentBounded running fresh proof attempt attemptEq complete.1.parentBounded
  let childWitnesses : Fin shape.runningCount → PaperLinearAlgebra.Assignment F columns :=
    fun child => reply.assignments (Fin.cast key.outputCount_eq.symm child)
  refine ⟨proof, result, childWitnesses, rfl, rfl, sampleEq, verified, ?_⟩
  intro child
  let index : Fin key.params.k := Fin.cast key.outputCount_eq.symm child
  have castBack : Fin.cast key.outputCount_eq index = child := Fin.ext rfl
  have same := PiDEC.OutputWitnessConsumer.runningStatement_eq_child key running fresh proof result
    attempt attemptEq verified index
  rw [castBack] at same
  rw [same]
  exact complete.2 index

/-- Source witnesses construct the causal C prefix before the PiRLC response
is known. For any actual successful response from its complete-output state,
the honest split constructs one normal NIFS proof, the verifier's returned
running product, and valid witnesses for every exact returned child. Sampler
success is explicit. Neither accepted messages nor valid children are inputs.
The existential causal prover carries no executable work or sampling claim. -/
theorem exists_honest_proof_of_sampler_success
    (witness : OutputWitness shape columns)
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness) :
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K width)
      (fullOutput : FullOutputCoordinates.FullOutput K shape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K State shape)
        { rounds := fun round => (messages round).toMessage }
      let probe : Probe K shape := {
        coins := { alpha := coins.alpha, gamma := coins.gamma, roundPoint := coins.roundPoint }
        response := {
          rounds := FixedPhase.RawCertificate.encode { rounds := List.ofFn messages }
          fullOutput := fullOutput } }
      probe.FixedWidthAccepted key.extensionOps key.lift (key.statement running fresh) width ∧
      (∀ source, CE.Holds key.piRlcSemantics key.params
        ((key.statement running fresh).publicOutput probe source) (witness.assignments source)) ∧
      ∀ rho : Fin key.arity.total → Scalar,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
        ∃ (proof : Proof K Commitment shape width)
          (result : Running K Commitment PublicInput shape)
          (childWitnesses : Fin shape.runningCount → PaperLinearAlgebra.Assignment F columns),
          proof.piCcsRounds = messages ∧ proof.piCcsOutput = fullOutput ∧
          key.piRlcChallenges running fresh proof = some rho ∧
          verify key running fresh proof = some result ∧
          ∀ child, CE.Holds key.piRlcSemantics key.params
            (PiDEC.OutputWitnessConsumer.runningStatement key result child) (childWitnesses child) := by
  obtain ⟨messages, fullOutput, accepted, openings⟩ :=
    exists_honest_prefix key running fresh witness valid
  refine ⟨messages, fullOutput, accepted, openings, ?_⟩
  intro rho sampled
  exact package_suffix key running fresh messages fullOutput witness accepted openings rho sampled

end NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness
