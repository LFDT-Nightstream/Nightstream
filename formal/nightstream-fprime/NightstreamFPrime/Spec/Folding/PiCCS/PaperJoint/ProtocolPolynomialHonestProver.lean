import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree
import NightstreamFPrime.Spec.SumCheck.FixedPhase.Canonical

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ProtocolPolynomialHonestProver.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Interactive honest prover for the paper-joint `Pi_CCS` polynomial.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: verifier-owned public alpha/gamma followed by causal SumCheck rounds.
Constraint family: paper semantics only; this file emits no rows.

Owns: construction of raw canonical round messages before their corresponding
verifier challenges, exact honest output evaluation at the resulting typed
point, and perfect logical acceptance of the raw finite SumCheck verifier.

Does not own: Fiat--Shamir, a random oracle, Poseidon2, commitment security,
the strong extractor game, generated artifacts, Rust, R1CS, or costs.

Emits constraints: no.

The verifier fixes `alpha` and `gamma` before this construction. Each call to
`challengeStep` receives only the current verifier state and the current raw
canonical message. No future challenge or terminal is an input to the honest
round constructor.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

universe uField uState

/-- Replay raw messages in message-before-challenge order. The verifier owns
the state transition and returned challenge. -/
def runRaw
    {Field : Type uField}
    {State : Type uState}
    (challengeStep : State -> SumCheck.Finite.Message Field -> Field × State) :
    State -> List (SumCheck.Finite.Message Field) -> List Field × State
  | state, [] => ([], state)
  | state, message :: messages =>
      let sample := challengeStep state message
      let tail := runRaw challengeStep sample.2 messages
      (sample.1 :: tail.1, tail.2)

/-- Fixed-width sequential replay is exactly raw replay after canonicalizing
each message. This changes no order and gives the challenge transition no
access to the fixed-width proof carrier. -/
theorem sequentialRun_eq_runRaw
    {Field : Type uField}
    {State : Type uState}
    {degree : Nat}
    [DecidableEq Field]
    (ops : SumCheck.Finite.Ops Field)
    (challengeStep : State -> SumCheck.Finite.Message Field -> Field × State)
    (state : State)
    (rounds : List (SumCheck.Finite.FixedPolynomial Field degree)) :
    SumCheck.Finite.FixedPhase.Sequential.run
        (fun current polynomial =>
          challengeStep current
            (SumCheck.Finite.FixedPolynomial.canonicalMessage ops polynomial))
        state rounds =
      runRaw challengeStep state
        (rounds.map
          (SumCheck.Finite.FixedPolynomial.canonicalMessage ops)) := by
  induction rounds generalizing state with
  | nil => rfl
  | cons polynomial polynomials inductionHypothesis =>
      simp only [SumCheck.Finite.FixedPhase.Sequential.run, List.map_cons,
        runRaw]
      rw [inductionHypothesis]

/-- Exact interactive perfect completeness for the actual paper polynomial.

`sourceTruth` is the unsampled CCS/norm/carried relation. It derives the
initial equality for every verifier-owned `alpha` and `gamma`; it is not a
sampled-equality or acceptance premise. The output and terminal are computed
only after causal replay derives the typed round point. -/
theorem exists_honest_accepted
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challengeStep :
      State -> SumCheck.Finite.Message Field -> Field × State)
    (initialState : State)
    (sourceTruth :
      (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold) :
    ∃ certificate : SumCheck.Finite.Certificate Field,
    ∃ roundPoint : CubePoint Field shape.cubeVariables,
    ∃ output : ProtocolPolynomial.OutputMessage Field shape,
    ∃ finalState : State,
      certificate.rounds.length = shape.cubeVariables ∧
      runRaw challengeStep initialState certificate.rounds =
        (roundPoint.coordinates, finalState) ∧
      output = ProtocolPolynomial.messageAt ops data roundPoint ∧
      ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput
          alpha gamma roundPoint output =
        ProtocolPolynomial.polynomial ops data alpha gamma
          roundPoint.coordinates ∧
      SumCheck.Finite.Accepted ops.toOps
        data.toVerifierInput.sumcheckDegreeBound
        (data.toVerifierInput.initial ops gamma)
        roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput
          alpha gamma roundPoint output)
        certificate := by
  let q := ProtocolPolynomial.polynomial ops data alpha gamma
  let degree := data.toVerifierInput.sumcheckDegreeBound
  let fixedLaws := ProtocolPolynomialDegree.Support.polynomialLaws laws
  rcases SumCheck.Finite.FixedPhase.Sequential.exists_honest_run
      ops.toOps q degree shape.cubeVariables
      (fun state polynomial =>
        challengeStep state
          (SumCheck.Finite.FixedPolynomial.canonicalMessage
            ops.toOps polynomial))
      (by
        simpa [q, degree] using
          ProtocolPolynomialDegree.sequentialRoundRepresentable
            ops laws data alpha gamma)
      initialState with
    ⟨fixedCertificate, challenges, finalState, fixedRoundsLength,
      challengesLength, fixedReplay, fixedHonest⟩
  let roundPoint : CubePoint Field shape.cubeVariables := {
    coordinates := challenges
    dimension := challengesLength
  }
  let certificate : SumCheck.Finite.Certificate Field :=
    SumCheck.Finite.FixedPhase.Canonical.toFinite
      ops.toOps fixedCertificate
  let output : ProtocolPolynomial.OutputMessage Field shape :=
    ProtocolPolynomial.messageAt ops data roundPoint
  have coefficientTruth :
      SignedCoefficientObject.CoefficientTruth ops
        (data.toJointData ops) :=
    (SignedCoefficientObject.coefficientTruth_iff_tableObligations
      ops zeroLaws (data.toJointData ops)).2 sourceTruth
  have sampledZero :
      (SignedCoefficientPolynomial.polynomial ops
        (data.toJointData ops) alpha).evaluate ops.toOps gamma = ops.zero :=
    SignedCoefficientObject.evaluate_eq_zero_of_coefficientTruth
      ops laws (data.toJointData ops) alpha gamma coefficientTruth
  have jointInitialTrue :
      SumCheckInitial.verifierInitial ops (data.toJointData ops) gamma =
        SumCheckInitial.semanticInitial ops (data.toJointData ops)
          alpha gamma := by
    have claimTrue :=
      (SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero
        ops laws (data.toJointData ops) alpha gamma degree 0
        challenges (q challenges) certificate
        (ProtocolPolynomial.canonicalExpected ops data alpha gamma
          challenges)).2 sampledZero
    simpa [SumCheck.Claim.True, SumCheckInitial.symbolicInstance] using
      claimTrue
  have initialIsTrue :
      data.toVerifierInput.initial ops gamma =
        SumCheck.Finite.FixedPhase.semanticInitial ops.toOps q
          challenges.length := by
    rw [challengesLength]
    unfold SumCheck.Finite.FixedPhase.semanticInitial
    dsimp only [q]
    rw [ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ
      ops laws data alpha gamma]
    rw [ProtocolPolynomial.verifierInput_initial_eq_joint_initial]
    exact jointInitialTrue
  have fixedAccepted :
      SumCheck.Finite.FixedPhase.Accepted ops.toOps q
        (data.toVerifierInput.initial ops gamma) challenges
        fixedCertificate :=
    SumCheck.Finite.FixedPhase.complete ops.toOps q
      (data.toVerifierInput.initial ops gamma) challenges fixedCertificate
      initialIsTrue fixedHonest
  have rawAcceptedAtQ :
      SumCheck.Finite.Accepted ops.toOps degree
        (data.toVerifierInput.initial ops gamma) challenges
        (q challenges) certificate :=
    SumCheck.Finite.FixedPhase.Canonical.accepted_toFinite
      ops.toOps fixedLaws q (data.toVerifierInput.initial ops gamma)
      challenges fixedCertificate fixedAccepted
  have rawRoundsLength :
      certificate.rounds.length = shape.cubeVariables := by
    simp [certificate, SumCheck.Finite.FixedPhase.Canonical.toFinite,
      fixedRoundsLength]
  have rawReplay :
      runRaw challengeStep initialState certificate.rounds =
        (roundPoint.coordinates, finalState) := by
    dsimp only [certificate, SumCheck.Finite.FixedPhase.Canonical.toFinite]
    rw [← sequentialRun_eq_runRaw ops.toOps challengeStep]
    simpa [roundPoint] using fixedReplay
  have terminalExact :
      ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput
          alpha gamma roundPoint output =
        ProtocolPolynomial.polynomial ops data alpha gamma
          roundPoint.coordinates := by
    dsimp only [output]
    unfold ProtocolPolynomial.polynomial
    rw [dif_pos roundPoint.dimension]
    rfl
  have rawAccepted :
      SumCheck.Finite.Accepted ops.toOps
        data.toVerifierInput.sumcheckDegreeBound
        (data.toVerifierInput.initial ops gamma)
        roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage ops data.toVerifierInput
          alpha gamma roundPoint output)
        certificate := by
    rw [terminalExact]
    simpa [q, degree, roundPoint] using rawAcceptedAtQ
  exact ⟨certificate, roundPoint, output, finalState, rawRoundsLength,
    rawReplay, rfl, terminalExact, rawAccepted⟩

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver
