import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixOracleWorld
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Oracle worlds for the `Pi_CCS` prefix of the paper non-interactive NIFS.

Source: SuperNeo Section 7.3 and Appendix D.4, followed by Appendix D.5's
fixed-prefix experiment.

Owns: the three abstract oracle surfaces used before `Pi_RLC`; a key
realization that changes only those surfaces; and one finite correlated
experiment whose seed owns the realized prefix oracle, public NIFS input,
malicious prefix, continuation, and adversary randomness.

Does not own: an ideal-random-oracle distribution, collision bounds, the
post-prefix `Pi_RLC` vector distribution, event bounds, Poseidon2, Ajtai,
Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The experiment's deterministic `prover` projection is one malicious prover
algorithm over its explicit finite seed support.  The support may preserve
multiple adversary-randomness seeds that realize the same oracle world; no
deduplication occurs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState uSeed

/-- One realization of every abstract random-oracle surface reached before
the `Pi_RLC` vector query.  Semantic relation data and commitment algebra do
not belong to this world. -/
structure PiCcsPrefixOracleWorld
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  initialTranscriptState : State
  absorbPublicInput : State ->
    Running Extension Commitment PublicInput shape ->
    Fresh Commitment PublicInput shape ->
    State
  absorbPiCcsOutput : State ->
    FullOutputCoordinates.FullOutput Extension shape -> State
  oracle : ProtocolVerifier.Oracle Extension State shape

namespace Key

/-- Realize one prefix oracle world while preserving every semantic,
commitment, arity, and `Pi_RLC` field of the template key. -/
def inPiCcsPrefixWorld
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (world : PiCcsPrefixOracleWorld key) :
    Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound :=
  { key with
    oracle := world.oracle
    initialTranscriptState := world.initialTranscriptState
    absorbPublicInput := world.absorbPublicInput
    absorbPiCcsOutput := world.absorbPiCcsOutput }

@[simp] theorem inPiCcsPrefixWorld_oracle
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (world : PiCcsPrefixOracleWorld key) :
    (key.inPiCcsPrefixWorld world).oracle = world.oracle := by
  rfl

@[simp] theorem inPiCcsPrefixWorld_initialTranscriptState
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (world : PiCcsPrefixOracleWorld key) :
    (key.inPiCcsPrefixWorld world).initialTranscriptState =
      world.initialTranscriptState := by
  rfl

@[simp] theorem inPiCcsPrefixWorld_absorbPublicInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (world : PiCcsPrefixOracleWorld key) :
    (key.inPiCcsPrefixWorld world).absorbPublicInput =
      world.absorbPublicInput := by
  rfl

@[simp] theorem inPiCcsPrefixWorld_absorbPiCcsOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (world : PiCcsPrefixOracleWorld key) :
    (key.inPiCcsPrefixWorld world).absorbPiCcsOutput =
      world.absorbPiCcsOutput := by
  rfl

end Key

/-- A finite correlated prefix experiment.  One seed owns the oracle
realization, public NIFS input, and malicious rewindable prover that produced
the fixed `Pi_CCS` prefix and its `Pi_DEC` continuation. -/
structure PiCcsPrefixExperiment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  Seed : Type uSeed
  support : Support Seed
  world : Seed -> PiCcsPrefixOracleWorld key
  running : Seed -> Running Extension Commitment PublicInput shape
  fresh : Seed -> Fresh Commitment PublicInput shape
  prover : (seed : Seed) ->
    RewindableProver (key.inPiCcsPrefixWorld (world seed))

namespace PiCcsPrefixExperiment

/-- Exact key realization owned by one outer seed. -/
def realizedKey
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (experiment : PiCcsPrefixExperiment key)
    (seed : experiment.Seed) :
    Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound :=
  key.inPiCcsPrefixWorld (experiment.world seed)

end PiCcsPrefixExperiment

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
