import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation

/-!
Post-prefix oracle worlds for the paper non-interactive NIFS.

Source: SuperNeo Section 7.4 and Appendix D.5.  The Appendix-D.5 adversary
fixes `(pp, s, u₁, st)` and receives the complete `Pi_RLC` challenge vector
as its sole varying input.

Owns: one valid complete `Pi_RLC` challenge vector; exact reprogramming of the
single post-`Pi_CCS` response point while retaining every response at every
other state; transport of one fixed rewindable prover into that world; and a
dependent outcome whose projected fork uses the world-owned key.

Does not own: a distribution on worlds, the preceding random-oracle prefix,
an acceptance runner, a forking probability theorem, event bounds,
Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The template key's `Pi_CCS` oracle and public-input absorption remain fixed.
Only `piRlcResponse` at the exact state after the fixed prefix is changed.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

universe uExtension uCommitment uPublicInput uScalar uState

/-- One valid, complete `Pi_RLC` challenge vector at a fixed transcript
prefix. -/
structure PiRlcVectorWorld
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  challenges : PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext
  valid : forall index,
    key.piRlcAlgebra.challengeValid (challenges index)

namespace Key

/-- Reprogram the complete `Pi_RLC` vector at one exact transcript state.
All other response points retain the template key's response. -/
def reprogramPiRlcAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (targetState : State)
    (world : PiRlcVectorWorld key) :
    Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound :=
  { key with
    piRlcResponse := fun state index =>
      if state = targetState then
        world.challenges index
      else
        key.piRlcResponse state index
    piRlcResponseValid := by
      intro state index
      by_cases atTarget : state = targetState
      · simp only [atTarget, ↓reduceIte]
        exact world.valid index
      · simp only [atTarget, ↓reduceIte]
        exact key.piRlcResponseValid state index }

/-- Reprogramming returns the world vector at the selected response point. -/
@[simp] theorem reprogramPiRlcAt_response_target
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (targetState : State)
    (world : PiRlcVectorWorld key)
    (index : Fin key.arity.total) :
    (key.reprogramPiRlcAt targetState world).piRlcResponse
        targetState index =
      world.challenges index := by
  simp [reprogramPiRlcAt]

/-- Reprogramming retains the template response away from the selected
point. -/
theorem reprogramPiRlcAt_response_other
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (targetState state : State)
    (world : PiRlcVectorWorld key)
    (different : state ≠ targetState)
    (index : Fin key.arity.total) :
    (key.reprogramPiRlcAt targetState world).piRlcResponse state index =
      key.piRlcResponse state index := by
  simp [reprogramPiRlcAt, different]

end Key

namespace PrefixMessage

/-- Exact `Pi_CCS` certificate owned by the challenge-independent prefix. -/
def piCcsCertificate
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
    (message : PrefixMessage Extension shape degreeBound) :
    ProtocolVerifier.Certificate Extension shape where
  rounds := fun round => (message.piCcsRounds round).toMessage
  output := (key.statement running fresh).projectOutput message.piCcsOutput

/-- Exact post-output transcript state computed without constructing any
challenge-dependent continuation reply. -/
def outgoingState
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
    (message : PrefixMessage Extension shape degreeBound) : State :=
  (ProtocolVerifier.derive key.oracle
    (key.publicInputState running fresh)
    ((key.statement running fresh).verifierInput key.lift)
    (message.piCcsCertificate key running fresh)).outgoingState

end PrefixMessage

namespace RewindableProver

/-- Transcript state after the fixed `Pi_CCS` prefix and its complete output
have been replayed. -/
def prefixState
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) : State :=
  prover.piCcsPrefix.outgoingState key running fresh

/-- Replaying any continuation reply reaches the state owned solely by the
fixed prefix. -/
@[simp] theorem piCcsExecution_outgoingState_eq_prefixState
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    (key.piCcsExecution running fresh
        (prover.proofAt challenges)).outgoingState =
      prover.prefixState running fresh := by
  rfl

/-- Transport one fixed continuation into a key whose only changed field is
the post-prefix `Pi_RLC` response function. -/
def reprogramPiRlcAt
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
    (prover : RewindableProver key)
    (targetState : State)
    (world : PiRlcVectorWorld key) :
    RewindableProver (key.reprogramPiRlcAt targetState world) where
  piCcsPrefix := prover.piCcsPrefix
  reply := fun challenges => {
    piDecCommitments := (prover.reply challenges).piDecCommitments
    piDecEvaluations := (prover.reply challenges).piDecEvaluations
    childAssignments := (prover.reply challenges).childAssignments
  }

/-- Transport the continuation into the world programmed at its exact fixed
prefix state. -/
def inPiRlcWorld
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
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (world : PiRlcVectorWorld key) :
    RewindableProver
      (key.reprogramPiRlcAt (prover.prefixState running fresh) world) :=
  prover.reprogramPiRlcAt (prover.prefixState running fresh) world

/-- The transported prover receives exactly the vector installed at the
fixed post-prefix response point. -/
@[simp] theorem inPiRlcWorld_baseChallenges
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
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (world : PiRlcVectorWorld key) :
    (prover.inPiRlcWorld running fresh world).baseChallenges running fresh =
      world.challenges := by
  funext index
  unfold baseChallenges Key.piRlcChallenges
  rw [piCcsExecution_outgoingState_eq_prefixState]
  have samePrefixState :
      (prover.inPiRlcWorld running fresh world).prefixState running fresh =
        prover.prefixState running fresh := by
    rfl
  rw [samePrefixState]
  exact Key.reprogramPiRlcAt_response_target key
    (prover.prefixState running fresh) world index

end RewindableProver

/-- One outcome over a world-specific post-prefix oracle realization.  The
template key, running/fresh input, and malicious continuation are fixed data;
the complete challenge vector and fork sample belong to this outcome. -/
structure RewindablePiRlcWorldOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  running : Running Extension Commitment PublicInput shape
  fresh : Fresh Commitment PublicInput shape
  prover : RewindableProver key
  world : PiRlcVectorWorld key
  sample : ForkSample Scalar key.arity.total

namespace RewindablePiRlcWorldOutcome

/-- The exact verifier key realized by this outcome. -/
def realizedKey
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
    Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound :=
  key.reprogramPiRlcAt
    (outcome.prover.prefixState outcome.running outcome.fresh)
    outcome.world

/-- The fixed malicious continuation transported into this outcome's exact
oracle world. -/
def realizedProver
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
    RewindableProver outcome.realizedKey :=
  outcome.prover.inPiRlcWorld outcome.running outcome.fresh outcome.world

/-- Project the dependent world into the existing owned-continuation
soundness carrier. -/
def toRewindableForkOutcome
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
    RewindableForkOutcome outcome.realizedKey where
  running := outcome.running
  fresh := outcome.fresh
  prover := outcome.realizedProver
  sample := outcome.sample

/-- The projected base proof is driven by the world-owned challenge vector. -/
@[simp] theorem toRewindableForkOutcome_baseChallenges
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
    outcome.toRewindableForkOutcome.prover.baseChallenges
        outcome.running outcome.fresh =
      outcome.world.challenges := by
  exact outcome.prover.inPiRlcWorld_baseChallenges
    outcome.running outcome.fresh outcome.world

/-- The realized base proof is the fixed continuation's reply to the exact
world-owned vector. -/
@[simp] theorem realizedProver_baseProof_eq_proofAt_world
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
    outcome.realizedProver.baseProof outcome.running outcome.fresh =
      outcome.realizedProver.proofAt outcome.world.challenges := by
  unfold RewindableProver.baseProof
  exact congrArg outcome.realizedProver.proofAt
    outcome.toRewindableForkOutcome_baseChallenges

end RewindablePiRlcWorldOutcome

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
