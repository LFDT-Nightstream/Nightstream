import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCoupling
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixForkExperiment

/-!
Causal interactive-composition bridge through the post-prefix oracle world.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6, at the permitted
explicit random-oracle boundary.

Owns: the exact post-prefix outcome generated from one causal prefix seed and
one finite coordinate-fork seed; equality of its programmed NIFS `Pi_DEC`
execution with the operational interactive composition; equality of that
execution's public attempt with the actual one-message NIFS attempt; and the
exact D.6 target-success equivalence.

Does not own: existence of an ideal-random-oracle coupling, collision
probabilities, first-success conditioning, target child witnesses, a claim
that public acceptance implies target membership, Poseidon2, Ajtai, Rust,
R1CS, artifacts, minimality, or costs.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState

/-- The literal post-prefix outcome generated from one causal product seed
and one finite coordinate-fork seed. -/
noncomputable def causalPostPrefixOutcomeOfSeed
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
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (prefixSeed : contract.toPrefixExperiment.Seed)
    (scalarAlphabet : Support Scalar)
    (scalarAlphabetValid : forall scalar,
      scalar ∈ scalarAlphabet.values ->
        key.piRlcAlgebra.challengeValid scalar)
    (forkSeed : ForkSeed scalarAlphabet key.arity.total) :
    RewindablePiRlcWorldOutcome
      (contract.toPrefixExperiment.realizedKey prefixSeed) :=
  postPrefixOutcomeOfSeed running fresh
    (contract.toPrefixExperiment.prover prefixSeed)
    scalarAlphabet scalarAlphabetValid forkSeed

namespace CausalPrefixCouplingContract

/-- The operational interactive `Pi_DEC` execution and the execution owned by
the programmed NIFS world are the same complete typed value. -/
theorem interactivePiDecExecution_eq_postPrefix
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
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (prefixSeed : contract.toPrefixExperiment.Seed)
    (scalarAlphabet : Support Scalar)
    (scalarAlphabetValid : forall scalar,
      scalar ∈ scalarAlphabet.values ->
        key.piRlcAlgebra.challengeValid scalar)
    (forkSeed : ForkSeed scalarAlphabet key.arity.total) :
    let outcome := causalPostPrefixOutcomeOfSeed contract prefixSeed
      scalarAlphabet scalarAlphabetValid forkSeed
    PiRlcComposition.PiDec.piDecExecution
        (key.compatibleContext running fresh)
        (key.compatiblePiDecContext running fresh)
        contract.adversary
        (causalPrefixRun key running fresh contract.adversary prefixSeed)
        outcome.world.challenges =
      outcome.toRewindableForkOutcome.piDecExecutionAt
        outcome.world.challenges := by
  simp only
  rw [contract.interactivePiDecExecution_eq_continuation]
  rfl

end CausalPrefixCouplingContract

namespace RewindablePiRlcWorldOutcome

/-- At the programmed vector, the continuation execution's public attempt is
exactly the attempt evaluated by the actual NIFS verifier in that world. -/
theorem piDecExecutionAt_world_attempt_eq_nifs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindablePiRlcWorldOutcome key) :
    (outcome.toRewindableForkOutcome.piDecExecutionAt
      outcome.world.challenges).attempt =
        outcome.realizedKey.piDecAttempt outcome.running outcome.fresh
          (outcome.realizedProver.baseProof
            outcome.running outcome.fresh) := by
  rw [← outcome.toRewindableForkOutcome_baseChallenges]
  exact RewindableProver.continuationPiDecExecution_baseChallenges_attempt
    outcome.toRewindableForkOutcome.prover
    outcome.running outcome.fresh

/-- D.6 success in a programmed world is exactly public NIFS `Pi_DEC`
acceptance plus valid child witnesses for that same attempt.  The target
witness conjunct remains explicit and non-derivable from acceptance alone. -/
theorem continuationSuccessAt_world_iff_nifs_target
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindablePiRlcWorldOutcome key) :
    outcome.toRewindableForkOutcome.ContinuationSuccessAt
        outcome.world.challenges <->
      PiDEC.PaperVerifier.Accepted outcome.realizedKey.piDecAlgebra
          outcome.realizedKey.piDecEvaluationArity
          (outcome.realizedKey.piDecAttempt
            outcome.running outcome.fresh
            (outcome.realizedProver.baseProof
              outcome.running outcome.fresh)) /\
        forall child,
          CE.Holds outcome.realizedKey.semantics
            outcome.realizedKey.params
            (PiDEC.PaperVerifier.children
              outcome.realizedKey.piDecPublicInputSplit
              (outcome.realizedKey.piDecAttempt
                outcome.running outcome.fresh
                (outcome.realizedProver.baseProof
                  outcome.running outcome.fresh))
              child)
            ((outcome.realizedProver.reply
              outcome.world.challenges).childAssignments child) := by
  rw [← outcome.toRewindableForkOutcome_baseChallenges]
  exact
    outcome.toRewindableForkOutcome.continuationSuccessAt_baseChallenges_iff

end RewindablePiRlcWorldOutcome

namespace CausalPrefixCouplingContract

/-- The interactive composition's D.6 success event is exactly the
programmed NIFS verifier's public `Pi_DEC` acceptance plus target child
membership for the same messages and assignments. -/
theorem interactivePiDecSuccess_iff_postPrefixNifsTarget
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
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (prefixSeed : contract.toPrefixExperiment.Seed)
    (scalarAlphabet : Support Scalar)
    (scalarAlphabetValid : forall scalar,
      scalar ∈ scalarAlphabet.values ->
        key.piRlcAlgebra.challengeValid scalar)
    (forkSeed : ForkSeed scalarAlphabet key.arity.total) :
    let outcome := causalPostPrefixOutcomeOfSeed contract prefixSeed
      scalarAlphabet scalarAlphabetValid forkSeed
    PiDEC.PaperReduction.Success
        (key.compatiblePiDecContext running fresh).paper
        (PiRlcComposition.PiDec.piDecExecution
          (key.compatibleContext running fresh)
          (key.compatiblePiDecContext running fresh)
          contract.adversary
          (causalPrefixRun key running fresh contract.adversary prefixSeed)
          outcome.world.challenges) <->
      PiDEC.PaperVerifier.Accepted outcome.realizedKey.piDecAlgebra
          outcome.realizedKey.piDecEvaluationArity
          (outcome.realizedKey.piDecAttempt
            outcome.running outcome.fresh
            (outcome.realizedProver.baseProof
              outcome.running outcome.fresh)) /\
        forall child,
          CE.Holds outcome.realizedKey.semantics
            outcome.realizedKey.params
            (PiDEC.PaperVerifier.children
              outcome.realizedKey.piDecPublicInputSplit
              (outcome.realizedKey.piDecAttempt
                outcome.running outcome.fresh
                (outcome.realizedProver.baseProof
                  outcome.running outcome.fresh))
              child)
            ((outcome.realizedProver.reply
              outcome.world.challenges).childAssignments child) := by
  simp only
  rw [contract.interactivePiDecExecution_eq_postPrefix]
  exact
    (causalPostPrefixOutcomeOfSeed contract prefixSeed scalarAlphabet
      scalarAlphabetValid forkSeed).continuationSuccessAt_world_iff_nifs_target

end CausalPrefixCouplingContract

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
