import Nightstream.Protocol.Nebula.PaperFingerprint

/-!
Contract: Nebula Layer-1 finalization and Layer-2 segment continuity.

Assurance tier: model-level.

Owns the explicit paper obligations for a finalized memory segment: all four
products start at one, the operations commitment matches the program-advice
commitment, challenges are derived from committed multisets, terminal products
balance, all three finalized proofs verify, and the Layer-2 state carries the
final-memory commitment into the next segment.

Proof verification, commitment binding, challenge derivation, and proof
folding are verifier-owned functions. This module does not instantiate them,
prove a collision probability, connect them to F-prime rows, or claim Nebula
Theorem 9.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.PaperFinalization

open Nightstream.SuperNeo.Concrete
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.Memory

universe uState uCommitment uProof uAccumulator

/-- Commitments parsed from the finalized program, operations, and scan
proofs. The four component commitments are the challenge preimage. -/
structure Commitments (Commitment : Type uCommitment) where
  programAdvice : Commitment
  operations : Commitment
  scan : Commitment
  reads : Commitment
  writes : Commitment
  initialMemory : Commitment
  finalMemory : Commitment

/-- The three Layer-1 proofs folded by one `F_final` invocation. -/
structure Proofs (Proof : Type uProof) where
  program : Proof
  operations : Proof
  scan : Proof

/-- Public data parsed from one finalized memory segment. -/
structure Segment
    (State : Type uState)
    (Commitment : Type uCommitment)
    (Proof : Type uProof) where
  iterations : Nat
  stateIn : State
  stateOut : State
  timestampIn : Nat
  timestampOut : Nat
  commitments : Commitments Commitment
  challenges : Challenges
  initialProducts : Fin 4 → K
  finalProducts : Fin 4 → K
  proofs : Proofs Proof

/-- Layer-2 state carried between finalized segments. -/
structure Running
    (State : Type uState)
    (Commitment : Type uCommitment)
    (Accumulator : Type uAccumulator) where
  step : Nat
  initialState : State
  currentState : State
  timestamp : Nat
  finalMemory : Commitment
  accumulator : Accumulator

/-- Verifier-owned functions used by the abstract finalizer. -/
structure Semantics
    (State : Type uState)
    (Commitment : Type uCommitment)
    (Proof : Type uProof)
    (Accumulator : Type uAccumulator) where
  deriveChallenges : Commitments Commitment → Challenges
  verifyProgram : Segment State Commitment Proof → Bool
  verifyOperations : Segment State Commitment Proof → Bool
  verifyScan : Segment State Commitment Proof → Bool
  foldProofs : Accumulator → Proofs Proof → Option Accumulator

/-- The four `F_ops` and `F_scan` products start at one. -/
def ProductsStartAtOne (values : Fin 4 → K) : Prop :=
  ∀ index, values index = K.one

/-- Exact Layer-1 finalization checks from Nebula Section 4.3. -/
def Layer1Holds
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    (semantics : Semantics State Commitment Proof Accumulator)
    (segment : Segment State Commitment Proof) : Prop :=
  ProductsStartAtOne segment.initialProducts ∧
  segment.commitments.operations = segment.commitments.programAdvice ∧
  segment.challenges = semantics.deriveChallenges segment.commitments ∧
  Balanced segment.finalProducts ∧
  semantics.verifyProgram segment = true ∧
  semantics.verifyOperations segment = true ∧
  semantics.verifyScan segment = true

/-- Exact Layer-2 `F_final` transition. The initial-memory commitment of this
segment must equal the final-memory commitment carried by the prior state. -/
def Advances
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    (semantics : Semantics State Commitment Proof Accumulator)
    (prior : Running State Commitment Accumulator)
    (segment : Segment State Commitment Proof)
    (next : Running State Commitment Accumulator) : Prop :=
  Layer1Holds semantics segment ∧
  segment.stateIn = prior.currentState ∧
  segment.timestampIn = prior.timestamp ∧
  segment.commitments.initialMemory = prior.finalMemory ∧
  semantics.foldProofs prior.accumulator segment.proofs =
    some next.accumulator ∧
  next.step = prior.step + segment.iterations ∧
  next.initialState = prior.initialState ∧
  next.currentState = segment.stateOut ∧
  next.timestamp = segment.timestampOut ∧
  next.finalMemory = segment.commitments.finalMemory

theorem advances_implies_layer1
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    {semantics : Semantics State Commitment Proof Accumulator}
    {prior : Running State Commitment Accumulator}
    {segment : Segment State Commitment Proof}
    {next : Running State Commitment Accumulator}
    (advance : Advances semantics prior segment next) :
    Layer1Holds semantics segment :=
  advance.1

theorem advances_memory_continuity
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    {semantics : Semantics State Commitment Proof Accumulator}
    {prior : Running State Commitment Accumulator}
    {segment : Segment State Commitment Proof}
    {next : Running State Commitment Accumulator}
    (advance : Advances semantics prior segment next) :
    segment.commitments.initialMemory = prior.finalMemory ∧
      next.finalMemory = segment.commitments.finalMemory :=
  ⟨advance.2.2.2.1, advance.2.2.2.2.2.2.2.2.2⟩

theorem advances_next_state_exact
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    {semantics : Semantics State Commitment Proof Accumulator}
    {prior : Running State Commitment Accumulator}
    {segment : Segment State Commitment Proof}
    {next : Running State Commitment Accumulator}
    (advance : Advances semantics prior segment next) :
    next.step = prior.step + segment.iterations ∧
      next.initialState = prior.initialState ∧
      next.currentState = segment.stateOut ∧
      next.timestamp = segment.timestampOut :=
  ⟨advance.2.2.2.2.2.1,
    advance.2.2.2.2.2.2.1,
    advance.2.2.2.2.2.2.2.1,
    advance.2.2.2.2.2.2.2.2.1⟩

/-- A chain of finalized segments. Every constructor enforces the Layer-2
continuity equation at its boundary. -/
inductive Chain
    {State : Type uState}
    {Commitment : Type uCommitment}
    {Proof : Type uProof}
    {Accumulator : Type uAccumulator}
    (semantics : Semantics State Commitment Proof Accumulator) :
    Running State Commitment Accumulator →
      List (Segment State Commitment Proof) →
      Running State Commitment Accumulator → Prop
  | nil (state : Running State Commitment Accumulator) :
      Chain semantics state [] state
  | cons
      {prior middle final : Running State Commitment Accumulator}
      {segment : Segment State Commitment Proof}
      {rest : List (Segment State Commitment Proof)}
      (advance : Advances semantics prior segment middle)
      (tail : Chain semantics middle rest final) :
      Chain semantics prior (segment :: rest) final

end Nightstream.Protocol.Nebula.PaperFinalization
