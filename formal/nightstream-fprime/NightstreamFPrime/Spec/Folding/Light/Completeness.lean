import NightstreamFPrime.Spec.Folding.Light.Verifier
import NightstreamFPrime.Spec.Folding.Nifs.PaperCausalReplay

/-! Cyclo (ePrint 2026/359) Theorem 3, item 1, for the light fold under Fiat-Shamir. An honest
prover with a fresh witness of norm below `b` and an accumulator witness at bound `β` gets a
proof that `verify` accepts, and the output claim holds at `β + T · (b − 1)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.FieldCausalTrace (issued)
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Fresh Running)

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}

private theorem run_rounds_eq (prover : CausalExecution.Prover shape columns width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables)
    (probe : Probe K shape) (witness : OutputWitness shape columns)
    (rounds : List (FixedPolynomial K width))
    (returned : CausalExecution.run prover alpha gamma point = some (probe, witness))
    (receipt : issued (prover.rounds alpha gamma) [] point.coordinates = some rounds) :
    probe.response.rounds = FixedPhase.RawCertificate.encode { rounds } := by
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
      simp only at probeEq
      rw [← probeEq]

/-- Cyclo Theorem 3, item 1, for one fresh source and one running source. -/
theorem complete
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (witness : OutputWitness shape columns)
    (bound : Nat)
    (freshOpening : Opening.Holds
      (paperRelationSemantics (shape := shape) (blockCount := blockCount)
        key.baseOps key.extensionOps key.lift key.openingMaps)
      key.params.b ((key.statement running fresh).commitments key.freshSource)
      ((key.statement running fresh).publicInputs key.freshSource)
      (witness.assignments key.freshSource))
    (runningOpening : Opening.Holds
      (paperRelationSemantics (shape := shape) (blockCount := blockCount)
        key.baseOps key.extensionOps key.lift key.openingMaps)
      bound ((key.statement running fresh).commitments key.runningSource)
      ((key.statement running fresh).publicInputs key.runningSource)
      (witness.assignments key.runningSource))
    (truth : PiCCS.SemanticTruth
      (((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs key.baseOps)
      key.baseOps key.extensionOps key.lift) :
    ∃ rounds : Fin shape.cubeVariables → FixedPolynomial K width,
      let proof : Proof shape width := {
        piCcsRounds := rounds
        piCcsOutput := FullOutputCoordinates.FullOutput.honestAt key.baseOps key.extensionOps
          key.lift ((key.statement running fresh).sourceConnectedInputs witness)
          (key.coins running fresh rounds).roundPoint }
      ∀ challenge : Scalar, key.challenge running fresh proof = some challenge →
        ∃ result,
          verify key running fresh proof = some result ∧
          Rlc.Holds key.piRlcSemantics (bound + key.params.expansionT * (key.params.b - 1))
            result
            (Rlc.witness key.piRlcAlgebra key.one challenge
              (witness.assignments key.freshSource) (witness.assignments key.runningSource)) := by
  obtain ⟨prover, honest⟩ := PiCCS.exists_honest_prover key.baseOps key.baseLaws key.baseZero
    key.noZeroDivisors key.extensionOps key.extensionLaws key.extensionZeroLaws key.lift
    key.liftLaws (key.statement running fresh) key.constantLaw witness width
    (key.statement_sumcheckDegreeBound_eq running fresh) truth
  let pre := FiatShamir.derivePreSumcheck key.oracle.transcript
    (key.transcriptStatement running fresh)
  have total : ∀ point : CubePoint K shape.cubeVariables,
      ∃ (probe : Probe K shape) (value : OutputWitness shape columns),
        CausalExecution.run prover pre.alpha pre.gamma point = some (probe, value) := by
    intro point
    obtain ⟨probe, returned, _⟩ := honest pre.alpha pre.gamma point
    exact ⟨probe, witness, returned⟩
  obtain ⟨messages, _, _, receipt, _⟩ := Nifs.PaperCausalReplay.prover_generated_messages_replay
    key.oracle.transcript prover pre.alpha pre.gamma pre.state total
  refine ⟨messages, ?_⟩
  intro proof challenge sampled
  let coins := key.coins running fresh messages
  obtain ⟨probe, returned, accepted, coinsEq, outputEq⟩ :=
    honest pre.alpha pre.gamma coins.roundPoint
  have actualReceipt : issued (prover.rounds pre.alpha pre.gamma) []
      coins.roundPoint.coordinates = some (List.ofFn messages) := receipt
  have roundsEq := run_rounds_eq prover pre.alpha pre.gamma coins.roundPoint probe witness
    (List.ofFn messages) returned actualReceipt
  have probeEq : probe = key.probe running fresh proof := by
    obtain ⟨probeCoins, probeRounds, probeOutput⟩ := probe
    simp only at coinsEq roundsEq outputEq
    rw [coinsEq, roundsEq, outputEq]
    rfl
  have checkTrue : PiCCS.check key.extensionOps key.lift (key.statement running fresh) width
      (key.probe running fresh proof) = true := by
    rw [← probeEq]
    exact accepted
  refine ⟨key.outputFor running fresh proof challenge, ?_, ?_⟩
  · simp only [verify, checkTrue, ↓reduceIte, sampled, Option.map_some]
  · let outputs := (key.statement running fresh).publicOutput (key.probe running fresh proof)
    have holds : ∀ (source : Fin shape.sourceCount) (sourceBound : Nat),
        Opening.Holds
          (paperRelationSemantics (shape := shape) (blockCount := blockCount)
            key.baseOps key.extensionOps key.lift key.openingMaps)
          sourceBound ((key.statement running fresh).commitments source)
          ((key.statement running fresh).publicInputs source)
          (witness.assignments source) →
        Rlc.Holds key.piRlcSemantics sourceBound (outputs source) (witness.assignments source) := by
      intro source sourceBound opening
      refine ⟨(key.openingAgreement _ _ _ _).mp opening,
        (key.evaluationAgreement (witness.assignments source) coins.roundPoint).1, ?_⟩
      change key.piRlcSemantics.evaluations key.relationSource (witness.assignments source)
          coins.roundPoint = #[{
            pad := proof.piCcsOutput.padCoordinate source
            matrix := proof.piCcsOutput.matrixCoordinate source }]
      unfold Key.relationSource
      rw [(key.evaluationAgreement (witness.assignments source) coins.roundPoint).2]
      rfl
    exact Rlc.complete key.piRlcAlgebra key.one key.normLaw challenge
      (key.piRlcResponseValid _ _ sampled) key.relationSource
      (key.probe running fresh proof).coins.roundPoint
      (outputs key.freshSource) (outputs key.runningSource)
      (witness.assignments key.freshSource) (witness.assignments key.runningSource) bound
      rfl rfl rfl rfl
      (holds key.freshSource key.params.b freshOpening)
      (holds key.runningSource bound runningOpening)

end NightstreamFPrime.Spec.Folding.Light
