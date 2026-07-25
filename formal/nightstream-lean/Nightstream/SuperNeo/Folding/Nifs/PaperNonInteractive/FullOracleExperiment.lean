import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PiCcsPrefixOracleWorld
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixWorldSoundness

/-!
Full finite oracle experiment for the paper non-interactive NIFS.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6.

Owns: composition of one correlated `Pi_CCS` prefix seed with the exact
finite Appendix-D.5 post-prefix coordinate experiment; one dependent outcome
whose inner verifier key is owned by both oracle realizations; pullbacks of
the accepted, transition, residual, and six Fiat--Shamir events; the
pointwise soundness cover; exact zero ideal strong-set sampling loss; and the
global `(ell + 1) / |C|` programming bound.

Does not own: an ideal-random-oracle support, the accepted target-witness
extraction bound, the four interactive-event bounds, the four transcript-
collision bounds, Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or
costs.

Emits constraints: no.

The outer support preserves adversary randomness and prefix-oracle
multiplicity.  Every component then preserves the coordinate-fork seed
multiplicity.  No outcome or oracle world is deduplicated.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState uSeed

/-- One global outcome: the outer seed selects the correlated prefix oracle,
public input, and malicious prover; the inner outcome selects the exact
post-prefix PiRLC oracle vector and coordinate forks. -/
structure FullOracleOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key) where
  prefixSeed : prefixExperiment.Seed
  inner :
    RewindablePiRlcWorldOutcome
      (prefixExperiment.realizedKey prefixSeed)

/-- The exact finite mixture: first the caller-supplied correlated prefix
support, then the paper-selected coordinate-fork seed support. -/
def fullOracleForkMixture
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar) :
    Mixture prefixExperiment.Seed (FullOracleOutcome prefixExperiment) where
  prefixes := prefixExperiment.support
  component := fun prefixSeed =>
    (postPrefixForkExperiment
      (prefixExperiment.running prefixSeed)
      (prefixExperiment.fresh prefixSeed)
      (prefixExperiment.prover prefixSeed)
      alphabet alphabetValid).map fun inner => {
        prefixSeed := prefixSeed
        inner := inner
      }

/-- Executable base acceptance in the two-level realized oracle world. -/
def FullOracleAcceptedOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldAcceptedOutcome outcome.inner

/-- Independent paper transition in the two-level realized oracle world. -/
def FullOracleTransitionOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldTransitionOutcome outcome.inner

/-- One accepted target-witness extraction event plus four interactive
residual events in the two-level world. -/
def FullOracleResidualFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldResidualFailure outcome.inner

/-- Six exact Fiat--Shamir predicates evaluated under both oracle
realizations owned by each outcome. -/
def fullOracleEventPredicates
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key) :
    EventPredicates (FullOracleOutcome prefixExperiment) where
  publicInputBindingCollision := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).publicInputBindingCollision outcome.inner
  transcriptReplayCollision := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).transcriptReplayCollision outcome.inner
  transcriptStateCollision := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).transcriptStateCollision outcome.inner
  outputAbsorptionCollision := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).outputAbsorptionCollision outcome.inner
  challengeSamplingFailure := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).challengeSamplingFailure outcome.inner
  multiForkProgrammingFailure := fun outcome =>
    (piRlcWorldEventPredicates
      (prefixExperiment.realizedKey outcome.prefixSeed)
      ).multiForkProgrammingFailure outcome.inner

/-- Complete eleven-event failure union in the two-level oracle world. -/
def FullOracleNonInteractiveFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  FullOracleResidualFailure outcome \/
    AnyFailure (fullOracleEventPredicates prefixExperiment) outcome

/-- Pointwise soundness survives both dependent oracle realizations. -/
theorem fullOracleAccepted_implies_transition_or_failure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : FullOracleOutcome prefixExperiment)
    (accepted : FullOracleAcceptedOutcome outcome) :
    FullOracleTransitionOutcome outcome \/
      FullOracleNonInteractiveFailure outcome := by
  exact
    piRlcWorldAccepted_implies_transition_or_failure
      (key := prefixExperiment.realizedKey outcome.prefixSeed)
      laws strongSet outcome.inner accepted

/-- Ideal strong-set sampling failure is impossible in every complete oracle
outcome. -/
theorem not_fullOracleChallengeSamplingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) :
    ¬ (fullOracleEventPredicates
      prefixExperiment).challengeSamplingFailure outcome := by
  exact not_piRlcWorldChallengeSamplingFailure outcome.inner

/-- The full finite mixture assigns exactly zero probability to ideal
strong-set sampling failure. -/
theorem fullOracleChallengeSamplingFailure_probability_eq_zero
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar) :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).challengeSamplingFailure =
      0 := by
  have eventEq :
      (fullOracleEventPredicates
          prefixExperiment).challengeSamplingFailure =
        (fun _ : FullOracleOutcome prefixExperiment => False) := by
    funext outcome
    apply propext
    constructor
    · exact fun failure =>
        not_fullOracleChallengeSamplingFailure outcome failure
    · exact False.elim
  rw [eventEq, Mixture.probability_false]

/-- Appendix D.5's programming bound averages over the complete correlated
prefix support without multiplying the loss. -/
theorem fullOracleProgrammingFailure_probability_le_paper
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar) :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).multiForkProgrammingFailure ≤
      ratio (key.arity.total + 1) alphabet.cardinality := by
  apply Mixture.probability_le_of_components
  intro prefixSeed _
  change
    ((postPrefixForkExperiment
      (prefixExperiment.running prefixSeed)
      (prefixExperiment.fresh prefixSeed)
      (prefixExperiment.prover prefixSeed)
      alphabet alphabetValid).map fun inner => {
        prefixSeed := prefixSeed
        inner := inner
      }).probability
        (fullOracleEventPredicates
          prefixExperiment).multiForkProgrammingFailure ≤
      ratio (key.arity.total + 1) alphabet.cardinality
  rw [Experiment.map_probability]
  exact
    piRlcWorldProgrammingFailure_probability_le_paper
      (prefixExperiment.running prefixSeed)
      (prefixExperiment.fresh prefixSeed)
      (prefixExperiment.prover prefixSeed)
      alphabet alphabetValid

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
