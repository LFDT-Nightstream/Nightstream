import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.InteractiveCompositionBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PiCcsPrefixOracleWorld
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins

/-!
Explicit causal coupling contract for the Fiat--Shamir `Pi_CCS` prefix.

Source: SuperNeo Section 7.3 and Appendix D.4, at the permitted explicit
random-oracle boundary.

Owns: a Cartesian product of malicious-prover randomness and independent
finite verifier-coin randomness; one sequential interactive
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` adversary; realized prefix-oracle worlds and NIFS
continuations over that product; exact replay alignment with the causal
interactive prefix; exact continuation alignment with the same interactive
adversary; and conversion to the global prefix experiment.

Does not own: construction of an ideal-random-oracle world, collision
probabilities, existence of a coupling for a concrete oracle, D.4
first-success conditioning, target-witness success, Poseidon2, Ajtai, Rust,
R1CS, artifacts, minimality, or costs.

Emits constraints: no.

This is a fail-closed contract: a caller must exhibit every realized oracle
world and prove both alignment equations.  The contract cannot be obtained
from the typed oracle interface alone.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState
  uProverSeed uProverTape

/-- Forget the final `Pi_DEC` interface while preserving exactly the same
prover support, causal strategy, tape map, and recomposed assignment reply. -/
noncomputable def causalPiRlcAdversary
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (adversary : PiRlcComposition.PiDec.Adversary
      (key.compatibleContext running fresh) ProverSeed ProverTape) :
    PiRlcComposition.Adversary
      (key.compatibleContext running fresh) ProverSeed ProverTape :=
  PiRlcComposition.PiDec.toPiRlc
    (key.compatibleContext running fresh)
    (key.compatiblePiDecContext running fresh)
    adversary

/-- Causal interactive prefix selected by one independent prover/verifier seed
pair. -/
noncomputable def causalPrefixRun
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (adversary : PiRlcComposition.PiDec.Adversary
      (key.compatibleContext running fresh) ProverSeed ProverTape)
    (seed : PiRlcComposition.PrefixSeed Extension shape ProverSeed) :
    PrefixExecution Extension shape :=
  PiRlcComposition.prefixExecution
    (key.compatibleContext running fresh)
    (causalPiRlcAdversary key running fresh adversary)
    seed

/-- Explicit random-oracle coupling boundary for one fixed public NIFS
statement.  Public inputs cannot depend on verifier coins because they are
parameters outside the product seed. -/
structure CausalPrefixCouplingContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (alphabet : Support Extension) where
  ProverSeed : Type uProverSeed
  ProverTape : Type uProverTape
  adversary : PiRlcComposition.PiDec.Adversary
    (key.compatibleContext running fresh) ProverSeed ProverTape
  world :
    PiRlcComposition.PrefixSeed Extension shape ProverSeed ->
      PiCcsPrefixOracleWorld key
  prover : (seed : PiRlcComposition.PrefixSeed Extension shape ProverSeed) ->
    RewindableProver (key.inPiCcsPrefixWorld (world seed))
  prefixAligned : forall seed,
    CausalPrefixAlignment
      (key.inPiCcsPrefixWorld (world seed))
      running fresh
      ((prover seed).baseProof running fresh)
      (causalPrefixRun key running fresh adversary seed)
  replyAligned : forall seed challenges,
    (prover seed).toInteractivePiDecReply running fresh challenges =
      adversary.reply
        (causalPrefixRun key running fresh adversary seed)
        challenges

namespace CausalPrefixCouplingContract

/-- The exact independent product support required before the causal prefix
runs. -/
noncomputable def support
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet) :
    Support (PiRlcComposition.PrefixSeed
      Extension shape contract.ProverSeed) :=
  PiRlcComposition.prefixSupport
    (key.compatibleContext running fresh)
    alphabet
    (causalPiRlcAdversary key running fresh contract.adversary)

/-- Forget only the causal proof fields and expose the exact carrier consumed
by the global prefix/post-prefix experiment. -/
noncomputable def toPrefixExperiment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet) :
    PiCcsPrefixExperiment key where
  Seed := PiRlcComposition.PrefixSeed Extension shape contract.ProverSeed
  support := contract.support
  world := contract.world
  running := fun _ => running
  fresh := fun _ => fresh
  prover := contract.prover

/-- The outer carrier is definitionally the prover-support/verifier-coin
Cartesian product. -/
theorem support_eq_product
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet) :
    contract.support =
      contract.adversary.proverSupport.product
        (VerifierCoins.support alphabet shape.cubeVariables) := by
  rfl

/-- Membership exposes independent provenance of both seed coordinates. -/
theorem mem_support_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet)
    (seed : PiRlcComposition.PrefixSeed
      Extension shape contract.ProverSeed) :
    seed ∈ contract.support.values <->
      seed.1 ∈ contract.adversary.proverSupport.values /\
      seed.2 ∈
        (VerifierCoins.support alphabet shape.cubeVariables).values := by
  rw [support_eq_product, Support.mem_product_iff]

/-- Exact factorized seed-space size; no post-hoc outcome deduplication is
permitted. -/
theorem support_cardinality
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet) :
    contract.support.cardinality =
      contract.adversary.proverSupport.cardinality *
        alphabet.cardinality ^ (2 * shape.cubeVariables + 1) := by
  rw [support_eq_product, Support.product_cardinality,
    VerifierCoins.support_cardinality_pow]

/-- Every outcome in the exported prefix experiment retains the exact causal
replay receipt; conversion does not erase provenance. -/
theorem toPrefixExperiment_prefixAligned
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet)
    (seed : contract.toPrefixExperiment.Seed) :
    CausalPrefixAlignment
      (contract.toPrefixExperiment.realizedKey seed)
      (contract.toPrefixExperiment.running seed)
      (contract.toPrefixExperiment.fresh seed)
      ((contract.toPrefixExperiment.prover seed).baseProof
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed))
      (causalPrefixRun key running fresh contract.adversary seed) := by
  exact contract.prefixAligned seed

/-- Every realized causal prefix inherits the exact fixed-witness PiCCS
reduction at the NIFS gate.  This is pointwise over the independent product
support; first-success conditioning and probability bounds remain separate. -/
theorem toPrefixExperiment_piCcsCheck_extracts_sourceValid_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet)
    (seed : contract.toPrefixExperiment.Seed)
    (witness : OutputWitness shape columns)
    (ambient : AmbientTargetOpenings
      (contract.toPrefixExperiment.realizedKey seed)
      (contract.toPrefixExperiment.running seed)
      (contract.toPrefixExperiment.fresh seed)
      ((contract.toPrefixExperiment.prover seed).baseProof
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed))
      witness)
    (accepted : piCcsCheck
      (contract.toPrefixExperiment.realizedKey seed)
      (contract.toPrefixExperiment.running seed)
      (contract.toPrefixExperiment.fresh seed)
      ((contract.toPrefixExperiment.prover seed).baseProof
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed)) = true) :
    SourceValid
        (contract.toPrefixExperiment.realizedKey seed)
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed) witness \/
      PiCcsMixingRoot
        (contract.toPrefixExperiment.realizedKey seed)
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed)
        ((contract.toPrefixExperiment.prover seed).baseProof
          (contract.toPrefixExperiment.running seed)
          (contract.toPrefixExperiment.fresh seed))
        witness \/
      PiCcsSumCheckCollision
        (contract.toPrefixExperiment.realizedKey seed)
        (contract.toPrefixExperiment.running seed)
        (contract.toPrefixExperiment.fresh seed)
        ((contract.toPrefixExperiment.prover seed).baseProof
          (contract.toPrefixExperiment.running seed)
          (contract.toPrefixExperiment.fresh seed))
        witness := by
  exact piCcsCheck_extracts_sourceValid_or_badEvent
    (contract.toPrefixExperiment_prefixAligned seed) witness ambient accepted

/-- Therefore the exported NIFS prefix reaches exactly the interactive
coefficient-complete batch. -/
theorem toPrefixExperiment_batch_eq
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet)
    (seed : contract.toPrefixExperiment.Seed) :
    ((contract.toPrefixExperiment.realizedKey seed).compatibleContext
      running fresh).batchOfPrefix
        (causalPrefixRun key running fresh contract.adversary seed) =
      (contract.toPrefixExperiment.realizedKey seed).nifsPiRlcBatch
        running fresh
        ((contract.toPrefixExperiment.prover seed).baseProof running fresh) := by
  exact batchOfPrefix_eq_nifsPiRlcBatch
    (contract.toPrefixExperiment_prefixAligned seed)

/-- For every product seed and every post-prefix vector, the full interactive
`Pi_DEC` execution is literally the continuation execution in the realized
NIFS prefix world.  Target-relation success remains a separate premise. -/
theorem interactivePiDecExecution_eq_continuation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {alphabet : Support Extension}
    (contract : CausalPrefixCouplingContract key running fresh alphabet)
    (seed : contract.toPrefixExperiment.Seed)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    PiRlcComposition.PiDec.piDecExecution
        (key.compatibleContext running fresh)
        (key.compatiblePiDecContext running fresh)
        contract.adversary
        (causalPrefixRun key running fresh contract.adversary seed)
        challenges =
      (contract.prover seed).continuationPiDecExecution
        running fresh challenges := by
  exact RewindableProver.interactivePiDecExecution_eq_continuation
    running fresh (contract.prover seed) contract.adversary
    (causalPrefixRun key running fresh contract.adversary seed)
    challenges (contract.prefixAligned seed)
    (contract.replyAligned seed challenges)

end CausalPrefixCouplingContract

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
