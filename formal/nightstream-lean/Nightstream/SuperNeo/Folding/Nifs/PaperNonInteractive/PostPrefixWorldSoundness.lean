import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixForkExperiment
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableOracleSoundness

/-!
Soundness predicates over dependent post-prefix oracle worlds.

Source: SuperNeo Sections 7.3--7.5, Appendices D.3--D.6, and Appendix D.5's
fixed-prefix coordinate experiment.

Owns: accepted, transition, residual, and six Fiat--Shamir predicates whose
verifier key is the realization carried by each outcome; exact agreement
between generated-world acceptance and the finite runner; the accepted
structural soundness cover over dependent worlds; impossibility of sampling
failure; and the concrete paper bound for accepted multi-fork programming
failure; generic eleven-event probability composition; and the finite
fixed-prefix subtractive soundness theorem with only one target-witness
extraction, four interactive, and four collision bounds left as premises.

Does not own: the distribution of the preceding `Pi_CCS` oracle prefix, the
target-witness extraction bound, four transcript-collision bounds,
interactive reduction event bounds, Poseidon2, Ajtai, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

The last Fiat--Shamir predicate is deliberately
`accepted ∧ ¬ programmingReceipt`.  Rejecting base worlds are not extraction
failures and are not charged to the D.5 loss.

The `Pi_DEC` extraction predicate is likewise acceptance-gated: rejecting
worlds do not consume its target-witness extraction budget.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uExtension uCommitment uPublicInput uScalar uState uWeight

/-- The exact aligned outcome selected by one dependent oracle world. -/
def PiRlcWorldAlignedOutcome
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
    (outcome : RewindablePiRlcWorldOutcome key) :
    AlignedForkOutcome outcome.realizedKey :=
  outcome.toRewindableForkOutcome.toAlignedForkOutcome

/-- Executable base acceptance in the key realized by this world. -/
def PiRlcWorldAcceptedOutcome
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  AcceptedOutcome (PiRlcWorldAlignedOutcome outcome)

/-- Independent paper transition in the key realized by this world. -/
def PiRlcWorldTransitionOutcome
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  TransitionOutcome (PiRlcWorldAlignedOutcome outcome)

/-- The five residual interactive predicates evaluated in the outcome's
realized key. -/
def PiRlcWorldResidualFailure
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  ResidualFailure (PiRlcWorldAlignedOutcome outcome)

/-- Exact accepted-base multi-fork programming failure in one dependent
world. -/
def PiRlcWorldMultiForkProgrammingFailure
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiRlcWorldAcceptedOutcome outcome /\
    MultiForkProgrammingFailure (PiRlcWorldAlignedOutcome outcome)

/-- The six exact Fiat--Shamir predicates pulled back through each
world-owned realized key. -/
def piRlcWorldEventPredicates
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    EventPredicates (RewindablePiRlcWorldOutcome key) where
  publicInputBindingCollision := fun outcome =>
    (nifsEventPredicates outcome.realizedKey).publicInputBindingCollision
      (PiRlcWorldAlignedOutcome outcome)
  transcriptReplayCollision := fun outcome =>
    (nifsEventPredicates outcome.realizedKey).transcriptReplayCollision
      (PiRlcWorldAlignedOutcome outcome)
  transcriptStateCollision := fun outcome =>
    (nifsEventPredicates outcome.realizedKey).transcriptStateCollision
      (PiRlcWorldAlignedOutcome outcome)
  outputAbsorptionCollision := fun outcome =>
    (nifsEventPredicates outcome.realizedKey).outputAbsorptionCollision
      (PiRlcWorldAlignedOutcome outcome)
  challengeSamplingFailure := fun outcome =>
    (nifsEventPredicates outcome.realizedKey).challengeSamplingFailure
      (PiRlcWorldAlignedOutcome outcome)
  multiForkProgrammingFailure := PiRlcWorldMultiForkProgrammingFailure

/-- Exact accepted dependent-world `Pi_DEC` child-extraction predicate. -/
def PiRlcWorldPiDecChildExtractionEvent
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiDecChildExtractionEvent (PiRlcWorldAlignedOutcome outcome)

/-- Exact dependent-world `Pi_RLC` accepted-fork sampling predicate. -/
def PiRlcWorldForkSamplingEvent
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiRlcForkSamplingEvent (PiRlcWorldAlignedOutcome outcome)

/-- Exact dependent-world `Pi_CCS` SumCheck predicate. -/
def PiRlcWorldPiCcsSumCheckEvent
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiCcsSumCheckEvent (PiRlcWorldAlignedOutcome outcome)

/-- Exact dependent-world `Pi_CCS` mixing-root predicate. -/
def PiRlcWorldPiCcsMixingRootEvent
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiCcsMixingRootEvent (PiRlcWorldAlignedOutcome outcome)

/-- Exact dependent-world parent-opening binding predicate. -/
def PiRlcWorldParentBindingCollisionEvent
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  ParentBindingCollisionEvent (PiRlcWorldAlignedOutcome outcome)

/-- Independent bounds for one accepted target-witness extraction predicate
and four exact interactive residual predicates over any dependent-world
experiment. -/
structure PiRlcWorldInteractiveResidualContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {probabilityScale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment probabilityScale
        (RewindablePiRlcWorldOutcome key))
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight) : Prop where
  piDecChildExtraction :
    probabilityScale.le
      (experiment.probability PiRlcWorldPiDecChildExtractionEvent)
      extractionBudget.piDecTargetWitnessFailure
  piRlcForkSampling :
    probabilityScale.le
      (experiment.probability PiRlcWorldForkSamplingEvent)
      budget.piRlcForkSampling
  piCcsSumCheck :
    probabilityScale.le
      (experiment.probability PiRlcWorldPiCcsSumCheckEvent)
      budget.piCcsSumCheck
  piCcsMixingRoot :
    probabilityScale.le
      (experiment.probability PiRlcWorldPiCcsMixingRootEvent)
      budget.piCcsSchwartzZippel
  parentBindingCollision :
    probabilityScale.le
      (experiment.probability PiRlcWorldParentBindingCollisionEvent)
      budget.adjustedRelaxedBinding

/-- Exact six-event random-oracle contract over dependent worlds. -/
abbrev PiRlcWorldExplicitRandomOracleContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {probabilityScale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment probabilityScale
        (RewindablePiRlcWorldOutcome key))
    (budget : FiatShamirErrorBudget Weight) :=
  ExplicitRandomOracleContract experiment
    (piRlcWorldEventPredicates key) budget

/-- Base acceptance of a generated world is exactly the finite runner's base
Boolean. -/
theorem postPrefixOutcome_worldAccepted_iff
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (seed : ForkSeed alphabet key.arity.total) :
    PiRlcWorldAcceptedOutcome
        (postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
          seed) ↔
      prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
          (decodeWord seed.val) =
        true := by
  let outcome :=
    postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid seed
  have runnerChecks :=
    acceptsInPiRlcWorld_decodeWord_eq_checks running fresh prover alphabet
      alphabetValid seed
  have proofEquation :=
    outcome.realizedProver_baseProof_eq_proofAt_world
  have runnerChecks' :
      prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
          (decodeWord seed.val) =
        (piCcsCheck outcome.realizedKey running fresh
            (outcome.realizedProver.proofAt outcome.world.challenges) &&
          piDecCheck outcome.realizedKey running fresh
            (outcome.realizedProver.proofAt outcome.world.challenges)) := by
    change
      prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
          (decodeWord seed.val) =
        (piCcsCheck outcome.realizedKey running fresh
            (outcome.realizedProver.proofAt outcome.world.challenges) &&
          piDecCheck outcome.realizedKey running fresh
            (outcome.realizedProver.proofAt outcome.world.challenges))
      at runnerChecks
    exact runnerChecks
  have proofEquation' :
      outcome.realizedProver.baseProof running fresh =
        outcome.realizedProver.proofAt outcome.world.challenges := by
    change
      outcome.realizedProver.baseProof running fresh =
        outcome.realizedProver.proofAt outcome.world.challenges
      at proofEquation
    exact proofEquation
  constructor
  · intro accepted
    change
      verify outcome.realizedKey running fresh
          (outcome.realizedProver.baseProof running fresh) =
        some
          (outcome.realizedKey.output running fresh
            (outcome.realizedProver.baseProof running fresh)) at accepted
    have checks :=
      (verify_eq_some_iff outcome.realizedKey running fresh
        (outcome.realizedProver.baseProof running fresh)
        (outcome.realizedKey.output running fresh
          (outcome.realizedProver.baseProof running fresh))).mp accepted
    rw [runnerChecks']
    rw [← proofEquation']
    exact Bool.and_eq_true_iff.mpr ⟨checks.1, checks.2.1⟩
  · intro accepted
    rw [runnerChecks'] at accepted
    have checks : piCcsCheck outcome.realizedKey running fresh
          (outcome.realizedProver.baseProof running fresh) = true /\
        piDecCheck outcome.realizedKey running fresh
          (outcome.realizedProver.baseProof running fresh) = true := by
      rw [proofEquation']
      exact Bool.and_eq_true_iff.mp accepted
    change
      verify outcome.realizedKey running fresh
          (outcome.realizedProver.baseProof running fresh) =
        some
          (outcome.realizedKey.output running fresh
            (outcome.realizedProver.baseProof running fresh))
    exact
      (verify_eq_some_iff outcome.realizedKey running fresh
        (outcome.realizedProver.baseProof running fresh)
        (outcome.realizedKey.output running fresh
          (outcome.realizedProver.baseProof running fresh))).mpr
        ⟨checks.1, checks.2, rfl⟩

/-- The complete dependent-world failure union: one target-witness extraction
and four interactive residuals, followed by the six exact Fiat--Shamir
predicates. -/
def PiRlcWorldNonInteractiveFailure
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
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  PiRlcWorldResidualFailure outcome \/
    AnyFailure (piRlcWorldEventPredicates key) outcome

/-- Pull an exact six-event failure from a realized key into the dependent
world.  The programming branch retains the already-established base
acceptance instead of charging rejecting worlds. -/
theorem realizedAnyFailure_implies_worldAnyFailure
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
    (outcome : RewindablePiRlcWorldOutcome key)
    (accepted : PiRlcWorldAcceptedOutcome outcome)
    (failure :
      AnyFailure (nifsEventPredicates outcome.realizedKey)
        (PiRlcWorldAlignedOutcome outcome)) :
    AnyFailure (piRlcWorldEventPredicates key) outcome := by
  rcases failure with
    publicInput | replay | state | output | sampling | programming
  · exact Or.inl publicInput
  · exact Or.inr (Or.inl replay)
  · exact Or.inr (Or.inr (Or.inl state))
  · exact Or.inr (Or.inr (Or.inr (Or.inl output)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl sampling))))
  · exact Or.inr
      (Or.inr (Or.inr (Or.inr (Or.inr ⟨accepted, programming⟩))))

/-- Structural soundness over actual dependent oracle worlds. -/
theorem piRlcWorldAccepted_implies_transition_or_failure
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
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : RewindablePiRlcWorldOutcome key)
    (accepted : PiRlcWorldAcceptedOutcome outcome) :
    PiRlcWorldTransitionOutcome outcome \/
      PiRlcWorldNonInteractiveFailure outcome := by
  have core :=
    acceptedOutcome_implies_transition_or_failure
      (key := outcome.realizedKey) laws strongSet
      (PiRlcWorldAlignedOutcome outcome) accepted
  rcases core with transition | failure
  · exact Or.inl transition
  · right
    rcases failure with residual | randomOracle
    · exact Or.inl residual
    · exact Or.inr
        (realizedAnyFailure_implies_worldAnyFailure outcome accepted
          randomOracle)

/-- A realized key's strong-set contract excludes challenge-sampling failure
in every dependent world. -/
theorem not_piRlcWorldChallengeSamplingFailure
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
    (outcome : RewindablePiRlcWorldOutcome key) :
    ¬ (piRlcWorldEventPredicates key).challengeSamplingFailure outcome := by
  exact not_piRlcSamplingSetFailure outcome.realizedKey outcome.running
    outcome.fresh
    (outcome.toRewindableForkOutcome.prover.baseProof outcome.running
      outcome.fresh)

/-- In the concrete conditional experiment, the sampling-failure event has
exactly zero probability. -/
theorem postPrefixChallengeSamplingFailure_probability_eq_zero
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar) :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability
        (piRlcWorldEventPredicates key).challengeSamplingFailure =
      0 := by
  have eventEq :
      (piRlcWorldEventPredicates key).challengeSamplingFailure =
        (fun _ : RewindablePiRlcWorldOutcome key => False) := by
    funext outcome
    apply propext
    constructor
    · exact fun failure =>
        not_piRlcWorldChallengeSamplingFailure outcome failure
    · exact False.elim
  rw [eventEq, Experiment.probability_false]

/-- A generated dependent-world programming failure is exactly within the
accepted-base event counted by the finite runner. -/
theorem generatedWorldProgrammingFailure_implies_postPrefixFailure
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (seed : ForkSeed alphabet key.arity.total)
    (failure :
      PiRlcWorldMultiForkProgrammingFailure
        (postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
          seed)) :
    PostPrefixMultiForkProgrammingFailure alphabet alphabetValid
      (postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
        seed) := by
  refine ⟨?_, failure.2⟩
  let accepts :=
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
  have baseAccepted : accepts (decodeWord seed.val) = true :=
    (postPrefixOutcome_worldAccepted_iff running fresh prover alphabet
      alphabetValid seed).mp failure.1
  change accepts (run accepts seed.val).sample.base = true
  simpa only [RunResult.sample, run_base] using baseAccepted

/-- Appendix D.5's selected programming loss now bounds the exact
dependent-world event used by structural soundness. -/
theorem piRlcWorldProgrammingFailure_probability_le_paper
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar) :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability PiRlcWorldMultiForkProgrammingFailure ≤
      ratio (key.arity.total + 1) alphabet.cardinality := by
  have eventMonotone :
      (forkExperiment alphabet key.arity.total).probability
          (fun seed =>
            PiRlcWorldMultiForkProgrammingFailure
              (postPrefixOutcomeOfSeed running fresh prover alphabet
                alphabetValid seed)) ≤
        (forkExperiment alphabet key.arity.total).probability
          (fun seed =>
            PostPrefixMultiForkProgrammingFailure alphabet alphabetValid
              (postPrefixOutcomeOfSeed running fresh prover alphabet
                alphabetValid seed)) := by
    apply Experiment.probability_mono
    intro seed failure
    exact generatedWorldProgrammingFailure_implies_postPrefixFailure running
      fresh prover alphabet alphabetValid seed failure
  exact Rat.le_trans eventMonotone
    (postPrefixProgrammingFailure_probability_le_paper running fresh prover
      alphabet alphabetValid)

/-- The accepted target-witness extraction bound and four exact dependent-
world interactive residual bounds imply the NIFS extraction plus
composition-ordered interactive total. This theorem performs only union
bookkeeping; it proves none of the five contract fields. -/
theorem piRlcWorldResidualFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {probabilityScale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (RewindablePiRlcWorldOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight)
    (contract :
      PiRlcWorldInteractiveResidualContract experiment extractionBudget
        budget) :
    probabilityScale.le
      (experiment.probability PiRlcWorldResidualFailure)
      (nifsInteractiveTotal probabilityScale extractionBudget budget) := by
  have sumCheckMixing :
      probabilityScale.le
        (experiment.probability fun outcome =>
          PiRlcWorldPiCcsSumCheckEvent outcome \/
            PiRlcWorldPiCcsMixingRootEvent outcome)
        (probabilityScale.add budget.piCcsSumCheck
          budget.piCcsSchwartzZippel) :=
    probabilityScale.le_trans
      (unionLaw.unionBound PiRlcWorldPiCcsSumCheckEvent
        PiRlcWorldPiCcsMixingRootEvent)
      (scaleLaws.add_mono contract.piCcsSumCheck
        contract.piCcsMixingRoot)
  have piCcsBinding :
      probabilityScale.le
        (experiment.probability fun outcome =>
          (PiRlcWorldPiCcsSumCheckEvent outcome \/
              PiRlcWorldPiCcsMixingRootEvent outcome) \/
            PiRlcWorldParentBindingCollisionEvent outcome)
        (probabilityScale.add
          (probabilityScale.add budget.piCcsSumCheck
            budget.piCcsSchwartzZippel)
          budget.adjustedRelaxedBinding) :=
    probabilityScale.le_trans
      (unionLaw.unionBound
        (fun outcome =>
          PiRlcWorldPiCcsSumCheckEvent outcome \/
            PiRlcWorldPiCcsMixingRootEvent outcome)
        PiRlcWorldParentBindingCollisionEvent)
      (scaleLaws.add_mono sumCheckMixing
        contract.parentBindingCollision)
  have forkTail :
      probabilityScale.le
        (experiment.probability fun outcome =>
          PiRlcWorldForkSamplingEvent outcome \/
            (PiRlcWorldPiCcsSumCheckEvent outcome \/
                PiRlcWorldPiCcsMixingRootEvent outcome) \/
              PiRlcWorldParentBindingCollisionEvent outcome)
        (probabilityScale.add budget.piRlcForkSampling
          (probabilityScale.add
            (probabilityScale.add budget.piCcsSumCheck
              budget.piCcsSchwartzZippel)
            budget.adjustedRelaxedBinding)) :=
    probabilityScale.le_trans
      (unionLaw.unionBound PiRlcWorldForkSamplingEvent
        (fun outcome =>
          (PiRlcWorldPiCcsSumCheckEvent outcome \/
              PiRlcWorldPiCcsMixingRootEvent outcome) \/
            PiRlcWorldParentBindingCollisionEvent outcome))
      (scaleLaws.add_mono contract.piRlcForkSampling piCcsBinding)
  exact
    probabilityScale.le_trans
      (unionLaw.unionBound PiRlcWorldPiDecChildExtractionEvent
        (fun outcome =>
          PiRlcWorldForkSamplingEvent outcome \/
            (PiRlcWorldPiCcsSumCheckEvent outcome \/
                PiRlcWorldPiCcsMixingRootEvent outcome) \/
              PiRlcWorldParentBindingCollisionEvent outcome))
      (by
        simpa [PiRlcWorldResidualFailure, ResidualFailure,
          PiRlcWorldPiDecChildExtractionEvent,
          PiRlcWorldForkSamplingEvent,
          PiRlcWorldPiCcsSumCheckEvent,
          PiRlcWorldPiCcsMixingRootEvent,
          PiRlcWorldParentBindingCollisionEvent,
          nifsInteractiveTotal,
          InteractiveErrorBudget.strongWeakTotal] using
          scaleLaws.add_mono contract.piDecChildExtraction forkTail)

/-- The interactive and explicit random-oracle contracts bound the complete
dependent-world failure union. -/
theorem piRlcWorldNonInteractiveFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {probabilityScale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (RewindablePiRlcWorldOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      PiRlcWorldInteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      PiRlcWorldExplicitRandomOracleContract experiment fiatShamirBudget) :
    probabilityScale.le
      (experiment.probability PiRlcWorldNonInteractiveFailure)
      (nonInteractiveTotal probabilityScale extractionBudget
        interactiveBudget fiatShamirBudget) := by
  have interactiveBound :=
    piRlcWorldResidualFailure_probability_le_total scaleLaws experiment
      unionLaw extractionBudget interactiveBudget interactiveContract
  have fiatShamirBound :=
    anyFailure_probability_le_total scaleLaws experiment unionLaw
      (piRlcWorldEventPredicates key) fiatShamirBudget randomOracleContract
  exact
    probabilityScale.le_trans
      (unionLaw.unionBound PiRlcWorldResidualFailure
        (AnyFailure (piRlcWorldEventPredicates key)))
      (by
        simpa [PiRlcWorldNonInteractiveFailure, nonInteractiveTotal] using
          scaleLaws.add_mono interactiveBound fiatShamirBound)

/-- Conditional soundness over any actual dependent-world experiment.  The
probability of accepted execution, minus precisely the interactive and six
Fiat--Shamir budgets, is at most the probability of the independently stated
paper transition. -/
theorem piRlcWorldAccepted_probability_sub_total_le_transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {probabilityScale : ProbabilityScale Weight}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (RewindablePiRlcWorldOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      PiRlcWorldInteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      PiRlcWorldExplicitRandomOracleContract experiment fiatShamirBudget) :
    probabilityScale.le
      (probabilityScale.subtract
        (experiment.probability PiRlcWorldAcceptedOutcome)
        (nonInteractiveTotal probabilityScale extractionBudget
          interactiveBudget fiatShamirBudget))
      (experiment.probability PiRlcWorldTransitionOutcome) := by
  apply loss_le_of_cover scaleLaws experiment unionLaw
    PiRlcWorldAcceptedOutcome PiRlcWorldTransitionOutcome
    PiRlcWorldNonInteractiveFailure
    (nonInteractiveTotal probabilityScale extractionBudget
      interactiveBudget fiatShamirBudget)
  · exact piRlcWorldAccepted_implies_transition_or_failure laws strongSet
  · exact
      piRlcWorldNonInteractiveFailure_probability_le_total scaleLaws
        experiment unionLaw extractionBudget interactiveBudget
        fiatShamirBudget interactiveContract randomOracleContract

/-- Caller-owned bounds for the four exact transcript-collision predicates.
The finite post-prefix experiment proves the sampling and programming fields
itself, so neither appears here. -/
structure PostPrefixCollisionBudget where
  publicInputBindingCollision : Rat
  transcriptReplayCollision : Rat
  transcriptStateCollision : Rat
  outputAbsorptionCollision : Rat

/-- The exact Fiat--Shamir budget selected for the finite conditional
post-prefix experiment: four caller-owned collision terms, zero ideal
strong-set sampling loss, and Appendix D.5's `(ell + 1) / |C|` programming
loss. -/
def postPrefixFiatShamirBudget
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (alphabet : Support Scalar)
    (collisionBudget : PostPrefixCollisionBudget) :
    FiatShamirErrorBudget Rat where
  publicInputBindingCollision :=
    collisionBudget.publicInputBindingCollision
  transcriptReplayCollision := collisionBudget.transcriptReplayCollision
  transcriptStateCollision := collisionBudget.transcriptStateCollision
  outputAbsorptionCollision := collisionBudget.outputAbsorptionCollision
  challengeSamplingFailure := 0
  multiForkProgrammingFailure :=
    ratio (key.arity.total + 1) alphabet.cardinality

/-- The only random-oracle premises left by the finite conditional
post-prefix experiment: one bound for each exact transcript-collision
predicate. -/
structure PostPrefixCollisionContract
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (budget : PostPrefixCollisionBudget) : Prop where
  publicInputBindingCollision :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability
        (piRlcWorldEventPredicates key).publicInputBindingCollision ≤
      budget.publicInputBindingCollision
  transcriptReplayCollision :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability
        (piRlcWorldEventPredicates key).transcriptReplayCollision ≤
      budget.transcriptReplayCollision
  transcriptStateCollision :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability
        (piRlcWorldEventPredicates key).transcriptStateCollision ≤
      budget.transcriptStateCollision
  outputAbsorptionCollision :
    (postPrefixForkExperiment running fresh prover alphabet
      alphabetValid).probability
        (piRlcWorldEventPredicates key).outputAbsorptionCollision ≤
      budget.outputAbsorptionCollision

/-- The four exact collision premises, together with the kernel-checked zero
sampling result and D.5 programming theorem, instantiate the complete
six-event random-oracle contract for the actual finite pushforward
experiment. -/
theorem postPrefixExplicitRandomOracleContract
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
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (collisionBudget : PostPrefixCollisionBudget)
    (collisionContract :
      PostPrefixCollisionContract running fresh prover alphabet alphabetValid
        collisionBudget) :
    PiRlcWorldExplicitRandomOracleContract
      (postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).toProbabilityExperiment
      (postPrefixFiatShamirBudget key alphabet collisionBudget) := by
  refine {
    publicInputBindingCollision := collisionContract.publicInputBindingCollision
    transcriptReplayCollision := collisionContract.transcriptReplayCollision
    transcriptStateCollision := collisionContract.transcriptStateCollision
    outputAbsorptionCollision := collisionContract.outputAbsorptionCollision
    challengeSamplingFailure := ?_
    multiForkProgrammingFailure := ?_
  }
  · change
      (postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).probability
          (piRlcWorldEventPredicates key).challengeSamplingFailure ≤ 0
    rw [postPrefixChallengeSamplingFailure_probability_eq_zero]
    exact Rat.le_refl
  · change
      (postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).probability PiRlcWorldMultiForkProgrammingFailure ≤
        ratio (key.arity.total + 1) alphabet.cardinality
    exact
      piRlcWorldProgrammingFailure_probability_le_paper running fresh prover
        alphabet alphabetValid

/-- Appendix D.5 instantiated over its actual finite conditional experiment.
Only the accepted target-witness extraction bound, four exact interactive-
reduction bounds, and four exact transcript-collision bounds remain premises.
Ideal strong-set sampling has zero loss and the multi-fork programming loss
is fixed to `(ell + 1) / |C|`.

This theorem is conditional on the fixed `Pi_CCS` prefix and continuation
carried by `prover`; it does not claim a distribution for the preceding
random-oracle transcript. -/
theorem postPrefixAccepted_probability_sub_total_le_transition
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
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (prover : RewindableProver key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (extractionBudget : NifsExtractionErrorBudget Rat)
    (interactiveBudget : InteractiveErrorBudget Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      PiRlcWorldInteractiveResidualContract
        (postPrefixForkExperiment running fresh prover alphabet
          alphabetValid).toProbabilityExperiment
        extractionBudget interactiveBudget)
    (collisionContract :
      PostPrefixCollisionContract running fresh prover alphabet alphabetValid
        collisionBudget) :
    scale.le
      (scale.subtract
        ((postPrefixForkExperiment running fresh prover alphabet
          alphabetValid).probability PiRlcWorldAcceptedOutcome)
        (nonInteractiveTotal scale extractionBudget interactiveBudget
          (postPrefixFiatShamirBudget key alphabet collisionBudget)))
      ((postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).probability PiRlcWorldTransitionOutcome) := by
  exact
    piRlcWorldAccepted_probability_sub_total_le_transition laws strongSet
      scaleLaws
      (postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).toProbabilityExperiment
      (postPrefixForkExperiment running fresh prover alphabet
        alphabetValid).toProbabilityUnionBound
      extractionBudget interactiveBudget
      (postPrefixFiatShamirBudget key alphabet collisionBudget)
      interactiveContract
      (postPrefixExplicitRandomOracleContract running fresh prover alphabet
        alphabetValid collisionBudget collisionContract)

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
