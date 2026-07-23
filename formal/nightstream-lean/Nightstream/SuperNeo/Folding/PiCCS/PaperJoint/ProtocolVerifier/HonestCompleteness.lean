import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialHonestProver
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier

/-!
Honest completeness of the transcript-bound paper-joint `Pi_CCS` verifier.

Owns: the exact equality between the causal honest prover's
message-before-challenge replay and `ProtocolVerifier`'s Fiat--Shamir replay,
and construction of one accepted non-interactive certificate from independent
paper source truth.

Does not own: `Pi_RLC`, `Pi_DEC`, extraction, random-oracle security, a
concrete hash, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

The proof vectorizes an exact-length message list only after the honest causal
run has been constructed.  No challenge, terminal value, or acceptance bit is
copied into the certificate.

| Protocol phase | Mathematical obligation | Lean owner |
|---|---|---|
| message indexing | preserve the exact honest message list and order | `roundsOfList`, `ofFn_roundsOfList` |
| causal transcript | equate honest message-before-challenge replay with verifier replay | `runRaw_indexedChallengeStep` |
| certificate | construct one accepted certificate from independent table truth | `exists_certificate_checked` |
| complete output | additionally expose equality between certificate output and the honest terminal message | `exists_certificate_checked_with_output` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.HonestCompleteness

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uState

/-- Re-index an exact-length honest message list by the verifier's static
round type. -/
def roundsOfList
    {Field : Type uField}
    {roundCount : Nat}
    (rounds : List (SumCheck.Finite.Message Field))
    (lengthEq : rounds.length = roundCount) :
    Fin roundCount -> SumCheck.Finite.Message Field :=
  fun index => rounds.get (Fin.cast lengthEq.symm index)

/-- Vectorization preserves the complete list and its canonical order. -/
theorem ofFn_roundsOfList
    {Field : Type uField}
    {roundCount : Nat}
    (rounds : List (SumCheck.Finite.Message Field))
    (lengthEq : rounds.length = roundCount) :
    List.ofFn (roundsOfList rounds lengthEq) = rounds := by
  subst roundCount
  change List.ofFn (fun index : Fin rounds.length => rounds.get index) = rounds
  exact List.ofFn_getElem

/-- Total state step used by the honest prover.  The fallback branch is
unreachable in the exact-length run; it merely keeps the function total when
there is no remaining typed round index. -/
def indexedChallengeStep
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (fallback : Field) :
    (List (Fin shape.cubeVariables) × State) ->
      SumCheck.Finite.Message Field ->
        Field × (List (Fin shape.cubeVariables) × State)
  | ([], state), _ => (fallback, ([], state))
  | (round :: remaining, state), message =>
      let absorbed := oracle.absorbRound state round message
      let sample := oracle.squeeze absorbed (.sumcheck round)
      (sample.1, (remaining, sample.2))

/-- Causal raw replay over the canonical message order is exactly the replay
used by the verifier's indexed Fiat--Shamir machine. -/
theorem runRaw_indexedChallengeStep
    {Field : Type uField}
    {State : Type uState}
    {shape : Shape}
    (oracle : FiatShamir.Oracle
      (ProtocolVerifier.Statement Field State shape) Field State shape)
    (fallback : Field)
    (rounds : Fin shape.cubeVariables -> SumCheck.Finite.Message Field)
    (state : State)
    (indices : List (Fin shape.cubeVariables)) :
    ProtocolPolynomialHonestProver.runRaw
        (indexedChallengeStep oracle fallback) (indices, state)
        (indices.map rounds) =
      let replay := FiatShamir.deriveRoundsFrom oracle rounds state indices
      (replay.1, ([], replay.2)) := by
  induction indices generalizing state with
  | nil => rfl
  | cons round remaining inductionHypothesis =>
      simp only [List.map_cons, ProtocolPolynomialHonestProver.runRaw,
        indexedChallengeStep, FiatShamir.deriveRoundsFrom]
      rw [inductionHypothesis]

private theorem cubePoint_eq_of_coordinates_eq
    {Field : Type uField}
    {variables : Nat}
    (left right : CubePoint Field variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  cases coordinates
  rfl

private theorem finiteCertificate_eq_of_rounds_eq
    {Field : Type uField}
    (left right : SumCheck.Finite.Certificate Field)
    (rounds : left.rounds = right.rounds) :
    left = right := by
  cases left
  cases right
  cases rounds
  rfl

/-- Independent paper source truth constructs an accepted certificate whose
output is exactly the honest paper message at the verifier-derived point.
Every verifier challenge is replayed from the statement and preceding
messages. -/
theorem exists_certificate_checked_with_output
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (sourceTruth :
      (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold) :
    exists certificate : ProtocolVerifier.Certificate Field shape,
      ProtocolVerifier.check oracle priorState ops data.toVerifierInput
        certificate = true /\
      certificate.output = ProtocolPolynomial.messageAt ops data
        (ProtocolVerifier.derive oracle priorState data.toVerifierInput
          certificate).coins.roundPoint := by
  let statement : ProtocolVerifier.Statement Field State shape := {
    priorState := priorState
    input := data.toVerifierInput
  }
  let pre := FiatShamir.derivePreSumcheck oracle.transcript statement
  let indices := canonicalFinIndices shape.cubeVariables
  let step := indexedChallengeStep oracle.transcript pre.gamma
  rcases ProtocolPolynomialHonestProver.exists_honest_accepted
      ops laws zeroLaws data pre.alpha pre.gamma step
      (indices, pre.state) sourceTruth with
    ⟨rawCertificate, roundPoint, output, finalState, roundsLength,
      rawReplay, outputExact, _terminalExact, rawAccepted⟩
  let rounds := roundsOfList rawCertificate.rounds roundsLength
  let certificate : ProtocolVerifier.Certificate Field shape := {
    rounds := rounds
    output := output
  }
  have roundsList : List.ofFn rounds = rawCertificate.rounds := by
    exact ofFn_roundsOfList rawCertificate.rounds roundsLength
  have indexedMessages : indices.map rounds = rawCertificate.rounds := by
    calc
      indices.map rounds = List.ofFn rounds := by
        simp [indices, canonicalFinIndices]
      _ = rawCertificate.rounds := roundsList
  let replay := FiatShamir.deriveRoundsFrom oracle.transcript rounds
    pre.state indices
  have verifierReplay :
      ProtocolPolynomialHonestProver.runRaw step (indices, pre.state)
          (indices.map rounds) =
        (replay.1, ([], replay.2)) := by
    exact runRaw_indexedChallengeStep oracle.transcript pre.gamma rounds
      pre.state indices
  have rawReplay' :
      ProtocolPolynomialHonestProver.runRaw step (indices, pre.state)
          (indices.map rounds) =
        (roundPoint.coordinates, finalState) := by
    rw [indexedMessages]
    exact rawReplay
  have replayCoordinates : replay.1 = roundPoint.coordinates := by
    exact congrArg Prod.fst (verifierReplay.symm.trans rawReplay')
  have derivedPoint :
      (ProtocolVerifier.derive oracle priorState data.toVerifierInput
        certificate).coins.roundPoint = roundPoint := by
    apply cubePoint_eq_of_coordinates_eq
    exact replayCoordinates
  have derivedAlpha :
      (ProtocolVerifier.derive oracle priorState data.toVerifierInput
        certificate).coins.alpha = pre.alpha := by
    rfl
  have derivedGamma :
      (ProtocolVerifier.derive oracle priorState data.toVerifierInput
        certificate).coins.gamma = pre.gamma := by
    rfl
  have finiteCertificate : certificate.toFinite = rawCertificate := by
    apply finiteCertificate_eq_of_rounds_eq
    exact roundsList
  have certificateOutput : certificate.output = output := by
    rfl
  refine ⟨certificate, ?_, ?_⟩
  · apply (ProtocolVerifier.check_eq_true_iff_accepted
      oracle priorState ops data.toVerifierInput certificate).2
    dsimp only
    rw [derivedAlpha, derivedGamma, derivedPoint, finiteCertificate,
      certificateOutput]
    exact rawAccepted
  · rw [certificateOutput, outputExact, derivedPoint]

/-- Compatibility projection retaining the original checker-only result. -/
theorem exists_certificate_checked
    {Field : Type uField}
    {State : Type uState}
    [DecidableEq Field]
    {shape : Shape}
    (oracle : ProtocolVerifier.Oracle Field State shape)
    (priorState : State)
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (sourceTruth :
      (TableResidualData.toTableObligations ops
        (SignedCoefficientObject.toTableResidualData ops
          (data.toJointData ops))).AllHold) :
    exists certificate : ProtocolVerifier.Certificate Field shape,
      ProtocolVerifier.check oracle priorState ops data.toVerifierInput
        certificate = true := by
  rcases exists_certificate_checked_with_output oracle priorState ops laws
      zeroLaws data sourceTruth with ⟨certificate, checked, _⟩
  exact ⟨certificate, checked⟩

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.HonestCompleteness
