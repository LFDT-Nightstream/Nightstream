import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance

/-!
Sequential honest-prover construction for the production combined block/lane
NC polynomial carrying one delayed packed-`yZcol` projection.

Assurance tier: model-level registered-deviation refinement.

Owns: prefix-causal construction of the exact five-slot block-then-lane
messages, replay from one NC entry state, and honest completeness when the
pending parent scalar equals the authoritative running-assignment projection.

Does not own: FE replay, selection of the pending iteration, commitment
binding, terminal discharge, concrete hashing, Rust, R1CS, generated rows,
costs, or row removal.

Emits constraints: none.

Authority boundary: the polynomial reads the current authoritative
`Sources.Data` and an independently supplied old block. The parent scalar is
accepted only through the explicit equality to
`authoritativeRunningProjection`; it is never copied from an output sidecar.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.delayed.degree` | represent every block/lane round in exactly five coefficient slots | derived | `roundRepresentable` |
| `pi_ccs.nc.delayed.messages` | construct messages before their corresponding replayed challenge | computed | `exists_honest_certificate` |
| `pi_ccs.nc.delayed.initial` | bind the pending scalar to the authoritative old-block projection | checked | `complete_of_truth_and_parentProjection` |
| `pi_ccs.nc.delayed.complete` | construct accepted combined-NC verification from paper NC truth | derived | `complete_of_truth_and_parentProjection` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.DelayedHonestProver

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

universe uState

/-- Every semantic combined-NC round is representable by the unchanged
quartic/five-slot physical message, using only the already-derived prefix. -/
theorem roundRepresentable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    RoundRepresentable ops.toOps
      (sumcheckPolynomial covers data coins weights producerBeta batchWeight
        oldBlock)
      ncSumcheckDegreeBound (Transcript.Nc.BlockLane.roundCount domain) := by
  intro fixed remaining length
  exact expectedRound_quartic covers data coins weights producerBeta
    batchWeight oldBlock fixed remaining length

/-- Construct exactly one message per block/lane coordinate. The certificate
contains messages only; every challenge is derived after absorbing its
corresponding message. -/
theorem exists_honest_certificate
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      FixedPhase.Honest ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        (Transcript.Nc.BlockLane.derive machine initialState certificate
          ).challengePoint.coordinates
        certificate.toSumCheck := by
  let polynomial := sumcheckPolynomial covers data coins weights producerBeta
    batchWeight oldBlock
  rcases exists_honest_run ops.toOps polynomial ncSumcheckDegreeBound
      (Transcript.Nc.BlockLane.roundCount domain)
      (Transcript.Nc.runRound machine)
      (by
        simpa [polynomial] using roundRepresentable covers data coins weights
          producerBeta batchWeight oldBlock)
      (machine.enterNc initialState) with
    ⟨fixedCertificate, challenges, finalState, roundsLength, _, replay,
      honest⟩
  let certificate : Transcript.Nc.BlockLane.Certificate domain := {
    rounds := functionOfExactList fixedCertificate.rounds roundsLength
  }
  have roundsExact :
      certificate.rawRounds = fixedCertificate.rounds := by
    dsimp only [certificate, Transcript.Nc.BlockLane.Certificate.rawRounds]
    exact ofFn_functionOfExactList fixedCertificate.rounds roundsLength
  have transcriptReplay :
      Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
          certificate.rawRounds =
        (challenges, finalState) := by
    rw [roundsExact, ← HonestProver.sequentialRun_eq_runRoundsFrom]
    exact replay
  have derivedChallenges :
      (Transcript.Nc.BlockLane.derive machine initialState certificate
        ).challengePoint.coordinates = challenges := by
    rw [Transcript.Nc.BlockLane.derive_point_coordinates, transcriptReplay]
  refine ⟨certificate, ?_⟩
  unfold FixedPhase.Honest at honest ⊢
  rw [derivedChallenges]
  change FixedPhase.Representations ops.toOps certificate.rawRounds
    (FixedPhase.expectedRounds ops.toOps polynomial challenges)
  rw [roundsExact]
  exact honest

/-- Honest current NC truth and an authoritative pending-parent projection make
the verifier's weighted initial claim equal the combined polynomial's full
Boolean cube, including degenerate `batchWeight = 0`. -/
theorem initial_eq_semanticInitial_of_truth_and_parentProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (truth : Semantics.Nc.Truth data)
    (parentBound : parentProjection =
      authoritativeRunningProjection covers data weights producerBeta
        oldBlock) :
    K.mul batchWeight parentProjection =
      FixedPhase.semanticInitial ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        point.coordinates.length := by
  calc
    K.mul batchWeight parentProjection =
        K.add K.zero (K.mul batchWeight parentProjection) :=
      (laws.zero_add _).symm
    _ = K.add (InitialSum.mixedResidualAtBeta covers data coins)
        (K.mul batchWeight
          (authoritativeRunningProjection covers data weights producerBeta
            oldBlock)) := by
      rw [InitialSum.mixedResidualAtBeta_eq_zero_of_truth covers data coins
        truth, parentBound]
    _ = FixedPhase.semanticInitial ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        point.coordinates.length :=
      (Acceptance.semanticInitial_eq_ordinary_add_weightedProjection covers data
        coins weights producerBeta batchWeight oldBlock point).symm

/-- Honest completeness for the delayed combined-NC phase. The NC transcript
continues from the supplied state, and its certificate is accepted against its
own replay-derived block/lane point. -/
theorem complete_of_truth_and_parentProjection
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight parentProjection : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (truth : Semantics.Nc.Truth data)
    (parentBound : parentProjection =
      authoritativeRunningProjection covers data weights producerBeta
        oldBlock) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      FixedPhase.Accepted ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        (K.mul batchWeight parentProjection)
        (Transcript.Nc.BlockLane.derive machine initialState certificate
          ).challengePoint.coordinates
        certificate.toSumCheck := by
  rcases exists_honest_certificate covers data machine initialState coins
      weights producerBeta batchWeight oldBlock with
    ⟨certificate, honest⟩
  let point := (Transcript.Nc.BlockLane.derive machine initialState certificate
    ).challengePoint
  let polynomial := sumcheckPolynomial covers data coins weights producerBeta
    batchWeight oldBlock
  have initialIsTrue :
      K.mul batchWeight parentProjection =
        FixedPhase.semanticInitial ops.toOps polynomial
          point.coordinates.length := by
    exact initial_eq_semanticInitial_of_truth_and_parentProjection covers data
      coins weights producerBeta batchWeight parentProjection oldBlock point
      truth parentBound
  refine ⟨certificate, ?_⟩
  exact FixedPhase.complete ops.toOps polynomial
    (K.mul batchWeight parentProjection) point.coordinates
    certificate.toSumCheck initialIsTrue honest

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.DelayedHonestProver
