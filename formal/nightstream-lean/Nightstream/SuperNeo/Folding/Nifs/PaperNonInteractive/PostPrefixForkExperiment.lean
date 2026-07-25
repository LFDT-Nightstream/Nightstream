import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixOracleWorld
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment.Headline

/-!
Finite post-prefix coordinate-fork experiment for the paper NIFS.

Source: SuperNeo Appendix C, Theorem 10, and Appendix D.5.  The fixed
`(pp, s, u₁, st)` prefix is represented by one key, public input, and
rewindable prover; the experiment samples the complete `Pi_RLC` vector
uniformly from the configured finite alphabet.

Owns: the executable Boolean acceptance predicate in a one-point-programmed
world; the uniform finite experiment over world-owned outcomes; exact
alignment of each world's base challenge; and conversion of runner success
into the concrete coordinate-programming receipt.

Does not own: the distribution of the preceding `Pi_CCS` random-oracle
prefix, a global Fiat--Shamir experiment, collision bounds, success of a
continuation, extraction event bounds, Poseidon2, Ajtai, Rust, R1CS,
artifacts, minimality, or costs.

Emits constraints: no.

This is the conditional D.5 experiment after a fixed prefix.  It does not
pretend that one realized key contains multiple random-oracle worlds.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState

namespace PiRlcVectorWorld

/-- Turn a vector whose coordinates belong to the finite strong-set alphabet
into a valid post-prefix oracle world. -/
def ofAlphabetVector
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
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext)
    (member : forall index, challenges index ∈ alphabet.values) :
    PiRlcVectorWorld key where
  challenges := challenges
  valid := fun index => alphabetValid _ (member index)

/-- The valid world represented by one finite coordinate-fork seed. -/
def ofForkSeed
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
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (seed : ForkSeed alphabet key.arity.total) :
    PiRlcVectorWorld key :=
  ofAlphabetVector key alphabet alphabetValid (decodeWord seed.val)
    (decodedWord_mem alphabet seed.property)

end PiRlcVectorWorld

namespace RewindableProver

/-- The actual Boolean NIFS verifier under a complete challenge-vector
programming.  Vectors outside the declared finite alphabet are rejected;
inside the alphabet, the exact post-prefix query is reprogrammed and both
operational paper checks are evaluated. -/
def acceptsInPiRlcWorld
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
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) : Bool :=
  if member : forall index, challenges index ∈ alphabet.values then
    let world :=
      PiRlcVectorWorld.ofAlphabetVector key alphabet alphabetValid
        challenges member
    let realizedKey :=
      key.reprogramPiRlcAt (prover.prefixState running fresh) world
    let realizedProver :=
      prover.inPiRlcWorld running fresh world
    let proof := realizedProver.proofAt challenges
    piCcsCheck realizedKey running fresh proof &&
      piDecCheck realizedKey running fresh proof
  else
    false

/-- On an alphabet vector, the runner predicate is definitionally the two
operational verifier checks in the corresponding oracle world. -/
theorem acceptsInPiRlcWorld_of_mem
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
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext)
    (member : forall index, challenges index ∈ alphabet.values) :
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
        challenges =
      let world :=
        PiRlcVectorWorld.ofAlphabetVector key alphabet alphabetValid
          challenges member
      let realizedKey :=
        key.reprogramPiRlcAt (prover.prefixState running fresh) world
      let realizedProver :=
        prover.inPiRlcWorld running fresh world
      let proof := realizedProver.proofAt challenges
      piCcsCheck realizedKey running fresh proof &&
        piDecCheck realizedKey running fresh proof := by
  simp [acceptsInPiRlcWorld, member]

end RewindableProver

/-- The world-owned outcome generated by one exact finite forking seed. -/
def postPrefixOutcomeOfSeed
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
    RewindablePiRlcWorldOutcome key where
  running := running
  fresh := fresh
  prover := prover
  world := PiRlcVectorWorld.ofForkSeed key alphabet alphabetValid seed
  sample :=
    (run
      (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)
      seed.val).sample

/-- Base acceptance for one seed is exactly the two operational checks in
that seed's realized world. -/
theorem acceptsInPiRlcWorld_decodeWord_eq_checks
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
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
        (decodeWord seed.val) =
      let outcome :=
        postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid seed
      piCcsCheck outcome.realizedKey running fresh
          (outcome.realizedProver.proofAt outcome.world.challenges) &&
        piDecCheck outcome.realizedKey running fresh
          (outcome.realizedProver.proofAt outcome.world.challenges) := by
  rw [RewindableProver.acceptsInPiRlcWorld_of_mem
    prover running fresh alphabet alphabetValid (decodeWord seed.val)
    (decodedWord_mem alphabet seed.property)]
  rfl

/-- The actual finite-uniform conditional D.5 experiment.  The outcome map
preserves seed multiplicity. -/
def postPrefixForkExperiment
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
    Experiment (RewindablePiRlcWorldOutcome key) :=
  (forkExperiment alphabet key.arity.total).map
    (postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid)

/-- The concrete extractor makes at most `ell + 1` verifier calls in
expectation.  Mapping seeds to world-owned outcomes preserves the seed and
therefore the exact trace cost. -/
theorem postPrefixForkExperiment_expectedQueriesAtMost
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
      alphabetValid).ExpectedQueriesAtMost
        (fun seed =>
          (run
            (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)
            seed.val).trace)
        (key.arity.total + 1) := by
  exact expected_queries_at_most alphabet
    (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)

/-- A semantic accepted coordinate fork supplies every structural field of
the exact programming receipt, while base alignment follows from the
world-owned key. -/
theorem postPrefixOutcome_programmed_of_acceptedCoordinateFork
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
    (acceptedFork :
      AcceptedCoordinateFork key.piRlcAlgebra.challengeValid
        (fun challenges (_answer : Unit) =>
          prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
              challenges =
            true)
        (fun _ => ())
        (run
        (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)
        seed.val).sample) :
    CoordinateProgrammingReceipt
      ((postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
        seed).toRewindableForkOutcome.toAlignedForkOutcome) := by
  let accepts :=
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
  constructor
  · let outcome :=
      postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid seed
    have derivedWorld :
        outcome.realizedKey.piRlcChallenges running fresh
            (outcome.realizedProver.baseProof running fresh) =
          outcome.world.challenges := by
      rw [RewindableProver.piRlcChallenges_baseProof]
      exact outcome.toRewindableForkOutcome_baseChallenges
    change
      (run accepts seed.val).sample.base =
        outcome.realizedKey.piRlcChallenges running fresh
          (outcome.realizedProver.baseProof running fresh)
    rw [derivedWorld]
    change (run accepts seed.val).base = decodeWord seed.val
    exact run_base accepts seed.val
  · exact acceptedFork.baseValid
  · exact acceptedFork.forkValid
  · exact acceptedFork.agreeExcept
  · exact acceptedFork.changed

/-- Executable runner success implies the exact world-owned programming
receipt. -/
theorem postPrefixOutcome_programmed_of_success
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
    (success :
      (run
        (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)
        seed.val).successBool
          (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid) =
        true) :
    CoordinateProgrammingReceipt
      ((postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
        seed).toRewindableForkOutcome.toAlignedForkOutcome) := by
  let accepts :=
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
  have acceptedFork :
      AcceptedCoordinateFork key.piRlcAlgebra.challengeValid
        (fun challenges (_answer : Unit) => accepts challenges = true)
        (fun _ => ())
        (run accepts seed.val).sample := by
    exact successBool_implies_acceptedCoordinateFork alphabet
      key.piRlcAlgebra.challengeValid
      (fun challenges (_answer : Unit) => accepts challenges = true)
      (fun _ => ()) accepts (fun _ => Iff.rfl) alphabetValid
      seed.property success
  exact postPrefixOutcome_programmed_of_acceptedCoordinateFork running fresh
    prover alphabet alphabetValid seed acceptedFork

/-- The exact finite-experiment programming failure: the base execution is
accepted by the real Boolean paper verifier, but the runner does not produce
a complete world-aligned receipt.  Rejecting base executions are not
programming failures. -/
def PostPrefixMultiForkProgrammingFailure
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
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (outcome : RewindablePiRlcWorldOutcome key) : Prop :=
  outcome.prover.acceptsInPiRlcWorld outcome.running outcome.fresh alphabet
      alphabetValid outcome.sample.base =
    true /\
  MultiForkProgrammingFailure
    outcome.toRewindableForkOutcome.toAlignedForkOutcome

/-- A programming failure in one generated world inhabits the concrete bad
word counted by the finite coordinate-fork theorem. -/
theorem postPrefixProgrammingFailure_implies_wordBad
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
      PostPrefixMultiForkProgrammingFailure alphabet alphabetValid
        (postPrefixOutcomeOfSeed running fresh prover alphabet alphabetValid
          seed)) :
    wordBad
        (prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid)
        seed.val =
      true := by
  let accepts :=
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
  rcases failure with ⟨baseAcceptedOutcome, notProgrammed⟩
  have baseAccepted : accepts (decodeWord seed.val) = true := by
    change accepts
        (run accepts seed.val).sample.base =
      true at baseAcceptedOutcome
    simpa only [RunResult.sample, run_base] using baseAcceptedOutcome
  rcases base_success_or_wordBad accepts seed.val baseAccepted with
    success | bad
  · exact False.elim
      (notProgrammed
        (postPrefixOutcome_programmed_of_success running fresh prover alphabet
          alphabetValid seed success))
  · exact bad

/-- The exact accepted-base programming-failure probability is bounded by
the sharp finite-coordinate loss `ell / |C|`. -/
theorem postPrefixProgrammingFailure_probability_le_sharp
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
          (PostPrefixMultiForkProgrammingFailure alphabet alphabetValid) ≤
      ratio key.arity.total alphabet.cardinality := by
  let accepts :=
    prover.acceptsInPiRlcWorld running fresh alphabet alphabetValid
  have eventMonotone :
      (forkExperiment alphabet key.arity.total).probability
          (fun seed =>
            PostPrefixMultiForkProgrammingFailure alphabet alphabetValid
              (postPrefixOutcomeOfSeed running fresh prover alphabet
                alphabetValid seed)) ≤
        (forkExperiment alphabet key.arity.total).probability
          (fun seed => wordBad accepts seed.val = true) := by
    apply Experiment.probability_mono
    intro seed failure
    exact postPrefixProgrammingFailure_implies_wordBad running fresh prover
      alphabet alphabetValid seed failure
  rw [Experiment.probability_bool_event] at eventMonotone
  exact Rat.le_trans eventMonotone
    (bad_probability_le_sharp alphabet accepts)

/-- The same concrete event satisfies Appendix D.5's selected
`(ell + 1) / |C|` loss. -/
theorem postPrefixProgrammingFailure_probability_le_paper
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
          (PostPrefixMultiForkProgrammingFailure alphabet alphabetValid) ≤
      ratio (key.arity.total + 1) alphabet.cardinality := by
  apply Rat.le_trans
    (postPrefixProgrammingFailure_probability_le_sharp running fresh prover
      alphabet alphabetValid)
  unfold ratio
  apply div_le_div_of_le
  · exact Rat.natCast_le_natCast.mpr (Nat.le_succ key.arity.total)
  · exact Rat.natCast_pos.mpr alphabet.cardinality_pos

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
