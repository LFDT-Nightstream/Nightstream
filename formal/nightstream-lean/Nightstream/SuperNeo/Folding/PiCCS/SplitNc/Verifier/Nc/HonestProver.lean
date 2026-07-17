import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc
import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Sequential honest-prover construction for the concrete Split-NC NC phase.

Owns: the bridge from the independently derived NC quartic slices to the
actual message-before-challenge transcript replay, exact conversion of the
constructed round list into the finite-index NC certificate, and honest
semantic completeness at the certificate's own derived challenge point.

Does not own: output-message construction, `yZcol` source authority,
Poseidon2 encoding, Fiat--Shamir probability, FE, Rust, R1CS, rows, costs, or
row removal.

Emits constraints: no.

Authority boundary: each round polynomial is derived from the current
challenge prefix and remaining Boolean dimension before `runRound` absorbs
that polynomial and derives the next challenge. No theorem receives a future
challenge vector, semantic attempt witness, or caller-supplied terminal.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.prover.round` | derive one five-slot round from the current prefix only | derived | `roundRepresentable` |
| `nifs.pi_ccs.nc.prover.replay` | generic sequential replay is the concrete NC replay | direct dataflow | `sequentialRun_eq_runRoundsFrom` |
| `nifs.pi_ccs.nc.prover.certificate` | exact-list reindexing preserves every round and its order | direct dataflow | `exists_honest_certificate` |
| `nifs.pi_ccs.nc.prover.honesty` | the constructed certificate is honest at its own derived challenges | derived | `exists_honest_certificate` |
| `nifs.pi_ccs.nc.prover.completeness` | NC truth yields transcript-bound semantic acceptance | derived | `complete_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Sequential

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

/-- The generic message-before-challenge replay is definitionally the same
ordered state machine as the concrete NC transcript replay. -/
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

/-- The independently proved quartic slice theorem supplies the exact
prefix-local premise required by the sequential constructor. -/
theorem roundRepresentable
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Polynomial.Nc.Mixing.Coins domain) :
    RoundRepresentable ops.toOps
      (Polynomial.Nc.InitialSum.sumcheckPolynomial
        convention covers data coins)
      Polynomial.Nc.Degree.ncSumcheckDegreeBound
      (Transcript.Nc.roundCount domain) := by
  intro fixed remaining length
  exact Polynomial.Nc.Degree.expectedRound_quartic
    convention covers data coins fixed remaining length

/-- Construct an exact-count NC transcript certificate whose rounds are
honest at the challenges produced by replaying those same rounds.

This closes the fixed-point gap left by post-challenge completeness: the
certificate is constructed sequentially, and no future challenge is an input
to the round constructor. -/
theorem exists_honest_certificate
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain) :
    ∃ certificate : Transcript.Nc.Certificate domain,
      FixedPhase.Honest ops.toOps
        (Polynomial.Nc.InitialSum.sumcheckPolynomial
          convention covers data coins)
        (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates
        certificate.toSumCheck := by
  let polynomial :=
    Polynomial.Nc.InitialSum.sumcheckPolynomial
      convention covers data coins
  rcases exists_honest_run ops.toOps polynomial
      Polynomial.Nc.Degree.ncSumcheckDegreeBound
      (Transcript.Nc.roundCount domain)
      (Transcript.Nc.runRound machine)
      (by
        simpa [polynomial] using
          roundRepresentable convention covers data coins)
      (machine.enterNc initialState) with
    ⟨fixedCertificate, challenges, finalState, roundsLength, _,
      replay, honest⟩
  let certificate : Transcript.Nc.Certificate domain := {
    rounds := functionOfExactList
      fixedCertificate.rounds roundsLength
  }
  have roundsExact :
      List.ofFn certificate.rounds = fixedCertificate.rounds := by
    dsimp only [certificate]
    exact ofFn_functionOfExactList
      fixedCertificate.rounds roundsLength
  have transcriptReplay :
      Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
          (List.ofFn certificate.rounds) =
        (challenges, finalState) := by
    rw [roundsExact]
    rw [← sequentialRun_eq_runRoundsFrom]
    exact replay
  have derivedChallenges :
      (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates =
        challenges := by
    change
      (Transcript.Nc.runRoundsFrom machine (machine.enterNc initialState)
        (List.ofFn certificate.rounds)).1 = challenges
    rw [transcriptReplay]
  refine ⟨certificate, ?_⟩
  unfold FixedPhase.Honest at honest ⊢
  rw [derivedChallenges]
  change FixedPhase.Representations ops.toOps
    (List.ofFn certificate.rounds)
    (FixedPhase.expectedRounds ops.toOps polynomial challenges)
  rw [roundsExact]
  exact honest

/-- Honest full-carrier NC truth yields a transcript-bound semantic
certificate accepted at the terminal recomputed from the certificate's own
derived challenge vector. -/
theorem complete_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    ∃ certificate : Transcript.Nc.Certificate domain,
      Transcript.Nc.Accepted machine initialState
        Polynomial.Nc.InitialSum.claimedInitial
        (Polynomial.Nc.InitialSum.sumcheckPolynomial
          convention covers data coins
          (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates)
        certificate := by
  rcases exists_honest_certificate convention covers data machine
      initialState coins with ⟨certificate, honest⟩
  let challenges :=
    (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates
  let polynomial :=
    Polynomial.Nc.InitialSum.sumcheckPolynomial
      convention covers data coins
  have initialIsTrue :
      Polynomial.Nc.InitialSum.claimedInitial =
        FixedPhase.semanticInitial ops.toOps polynomial challenges.length := by
    rw [Polynomial.Nc.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
      convention covers data coins truth]
    unfold Polynomial.Nc.InitialSum.sumcheckHypercubeSum
      FixedPhase.semanticInitial
    have challengeLength :
        challenges.length = Transcript.Nc.roundCount domain := by
      exact Transcript.Nc.derive_challenges_length
        machine initialState certificate
    rw [challengeLength]
  refine ⟨certificate, ?_⟩
  unfold Transcript.Nc.Accepted
  change FixedPhase.Chain ops.toOps
    Polynomial.Nc.InitialSum.claimedInitial
    certificate.toSumCheck.rounds challenges
    (polynomial challenges)
  exact FixedPhase.complete ops.toOps polynomial
    Polynomial.Nc.InitialSum.claimedInitial challenges
    certificate.toSumCheck initialIsTrue honest

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver
