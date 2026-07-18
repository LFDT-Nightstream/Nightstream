import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck
import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Sequential honest-prover construction for canonical block×lane NC.

Assurance tier: model-level.

Owns: prefix-local construction of exactly one five-slot message per block or
lane coordinate, replay of each message before its challenge, and honest
completeness at the certificate's own transcript-derived point.

Does not own: output-message construction, packed `yZcol` authority,
Poseidon2 encoding, Fiat--Shamir probability, FE, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: no future challenge is an input to message construction.
The block prefix and lane suffix are one replay entered once; the certificate
contains messages only.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.prover.round` | derive one quartic/five-slot round from the current prefix | derived | `roundRepresentable` |
| `nifs.pi_ccs.nc.block_lane.prover.replay` | sequential message-before-challenge execution is the canonical NC replay | direct dataflow | `sequentialRun_eq_runRoundsFrom` |
| `nifs.pi_ccs.nc.block_lane.prover.certificate` | exact list transport preserves all nine symbolic rounds | direct dataflow | `exists_honest_certificate` |
| `nifs.pi_ccs.nc.block_lane.prover.completeness` | NC truth yields transcript-bound acceptance | derived | `complete_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

/-- Generic sequential replay and canonical NC replay have the same explicit
state transition at every message. -/
theorem sequentialRun_eq_runRoundsFrom
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (state : State)
    (rounds : List Transcript.Nc.RoundMessage) :
    run (Transcript.Nc.runRound machine) state rounds =
      Transcript.Nc.runRoundsFrom machine state rounds := by
  induction rounds generalizing state with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      simp only [run, Transcript.Nc.runRoundsFrom]
      rw [inductionHypothesis]

/-- The independent quartic slice theorem supplies the exact prefix-local
representability premise used by the sequential constructor. -/
theorem roundRepresentable
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    RoundRepresentable ops.toOps
      (InitialSum.sumcheckPolynomial covers data coins)
      Polynomial.Nc.Degree.ncSumcheckDegreeBound
      (Transcript.Nc.BlockLane.roundCount domain) := by
  intro fixed remaining length
  exact Degree.SumCheck.expectedRound_quartic
    covers data coins fixed remaining length

/-- Construct the exact-count block-then-lane message list while deriving
every challenge only after its message has been absorbed. -/
theorem exists_honest_certificate
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      FixedPhase.Honest ops.toOps
        (InitialSum.sumcheckPolynomial covers data coins)
        (Transcript.Nc.BlockLane.derive machine initialState certificate).challengePoint.coordinates
        certificate.toSumCheck := by
  let polynomial := InitialSum.sumcheckPolynomial covers data coins
  rcases exists_honest_run ops.toOps polynomial
      Polynomial.Nc.Degree.ncSumcheckDegreeBound
      (Transcript.Nc.BlockLane.roundCount domain)
      (Transcript.Nc.runRound machine)
      (by simpa [polynomial] using roundRepresentable covers data coins)
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
    rw [roundsExact, ← sequentialRun_eq_runRoundsFrom]
    exact replay
  have derivedChallenges :
      (Transcript.Nc.BlockLane.derive machine initialState
        certificate).challengePoint.coordinates = challenges := by
    rw [Transcript.Nc.BlockLane.derive_point_coordinates, transcriptReplay]
  refine ⟨certificate, ?_⟩
  unfold FixedPhase.Honest at honest ⊢
  rw [derivedChallenges]
  change FixedPhase.Representations ops.toOps certificate.rawRounds
    (FixedPhase.expectedRounds ops.toOps polynomial challenges)
  rw [roundsExact]
  exact honest

/-- Honest NC truth yields a message-only certificate accepted at the
terminal recomputed from the same transcript-derived block×lane point. -/
theorem complete_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      Transcript.Nc.BlockLane.Accepted machine initialState
        InitialSum.claimedInitial
        (InitialSum.sumcheckPolynomial covers data coins
          (Transcript.Nc.BlockLane.derive machine initialState
            certificate).challengePoint.coordinates)
        certificate := by
  rcases exists_honest_certificate covers data machine initialState coins with
    ⟨certificate, honest⟩
  let challenges :=
    (Transcript.Nc.BlockLane.derive machine initialState
      certificate).challengePoint.coordinates
  let polynomial := InitialSum.sumcheckPolynomial covers data coins
  have initialIsTrue :
      InitialSum.claimedInitial =
        FixedPhase.semanticInitial ops.toOps polynomial challenges.length := by
    rw [InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
      covers data coins truth]
    unfold InitialSum.sumcheckHypercubeSum FixedPhase.semanticInitial
    have challengeLength :
        challenges.length =
          domain.blockVariables + domain.laneVariables :=
      (Transcript.Nc.BlockLane.derive machine initialState
        certificate).challengePoint.coordinates_length
    rw [challengeLength]
  refine ⟨certificate, ?_⟩
  unfold Transcript.Nc.BlockLane.Accepted
  change FixedPhase.Chain ops.toOps InitialSum.claimedInitial
    certificate.toSumCheck.rounds challenges (polynomial challenges)
  exact FixedPhase.complete ops.toOps polynomial InitialSum.claimedInitial
    challenges certificate.toSumCheck initialIsTrue honest

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver
