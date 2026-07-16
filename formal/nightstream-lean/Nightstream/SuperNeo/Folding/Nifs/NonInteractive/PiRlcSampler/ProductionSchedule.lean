import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Independent block schedule for production-shaped `Pi_RLC` coefficient sampling.

Protocol: SuperNeo `Pi_RLC` inside the candidate noninteractive NIFS.
Phase: transcript derivation of the complete scalar-challenge batch.
Constraint family: per-scalar domain separation, four digest blocks,
candidate flattening, and successor-state threading.

Owns: an abstract deterministic block machine whose one `digestBlock` call
jointly returns both its successor state and 16 candidates; the exact scalar
and per-block counter schedule; the flattened candidate stream; the fixed
four-block successor state; and the proof that every successful 54-of-64
reference execution advances by exactly those four complete blocks.

Does not own: a hash permutation, field or byte encoding, Poseidon2 tags,
Goldilocks lane canonicality, counter wrapping, probability, rotation/ring
assembly, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `Machine.enterScalar` and `Machine.digestBlock` are
verifier-owned deterministic functions. Stream candidates and `nextState`
cannot be supplied independently: both are projections of the same sequence
of block calls. A later concrete theorem must instantiate this machine with
the exact Poseidon2 transcript and prove Rust/R1CS correspondence.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | scalar domain | `Machine.enterScalar` | derive scalar coordinate `i` from the prior verifier state |
| `Pi_RLC` | digest block | `Machine.digestBlock` | jointly derive one successor state and exactly 16 candidates |
| `Pi_RLC` | counter schedule | `stateBeforeBlock` | block `r` uses counter `i+r` |
| `Pi_RLC` | candidate order | `candidateStream` | flatten blocks and their within-block indices in order |
| `Pi_RLC` | fixed circuit | `source` | expose the first 64 candidates and the state after four complete blocks |
| `Pi_RLC` | native equivalence condition | `successful_execution_uses_four_blocks` | a successful least cursor uses exactly four whole blocks |
| `Pi_RLC` | batch chaining | `stateAt_succ_eq_referenceBlockState` | each next scalar begins from the same state reached by the successful reference run |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

open Nightstream.SuperNeo.Sampling
open ProductionAlphabet

universe uState uScalar

/-- Abstract deterministic transcript machine for the sampler schedule. One
block call jointly owns its candidates and successor state. -/
structure Machine (State : Type uState) where
  enterScalar : State -> Nat -> State
  digestBlock : State -> Nat ->
    State × (Fin chunksPerDigest -> Chunk)

/-- State before block `round`, after all earlier complete block calls. -/
def stateBeforeBlock
    {State : Type uState}
    (machine : Machine State)
    (entered : State)
    (seed : Nat) : Nat -> State
  | 0 => entered
  | round + 1 =>
      (machine.digestBlock
        (stateBeforeBlock machine entered seed round)
        (seed + round)).1

/-- Candidate vector jointly returned by block `round`. -/
def chunksAt
    {State : Type uState}
    (machine : Machine State)
    (entered : State)
    (seed round : Nat) : Fin chunksPerDigest -> Chunk :=
  (machine.digestBlock
    (stateBeforeBlock machine entered seed round)
    (seed + round)).2

@[simp] theorem stateBeforeBlock_zero
    {State : Type uState}
    (machine : Machine State)
    (entered : State)
    (seed : Nat) :
    stateBeforeBlock machine entered seed 0 = entered := by
  rfl

@[simp] theorem stateBeforeBlock_succ
    {State : Type uState}
    (machine : Machine State)
    (entered : State)
    (seed round : Nat) :
    stateBeforeBlock machine entered seed (round + 1) =
      (machine.digestBlock
        (stateBeforeBlock machine entered seed round)
        (seed + round)).1 := by
  rfl

/-- Infinite candidate stream obtained by flattening complete blocks. Only its
first 64 candidates are used by the bounded recursive verifier. -/
def candidateStream
    {State : Type uState}
    (machine : Machine State)
    (entered : State)
    (seed : Nat) : FirstAccepted.CandidateStream Chunk :=
  fun position =>
    chunksAt machine entered seed (position / chunksPerDigest)
      ⟨position % chunksPerDigest, Nat.mod_lt _ (by decide)⟩

/-- One jointly owned source for scalar coordinate `coordinate`. -/
def source
    {State : Type uState}
    (machine : Machine State)
    (state : State)
    (coordinate : Nat) : Source State Chunk :=
  let entered := machine.enterScalar state coordinate
  {
    stream := candidateStream machine entered coordinate
    nextState := stateBeforeBlock machine entered coordinate digestRounds
  }

/-- Generic PiRLC sampler specialization induced by the block machine. Scalar
assembly remains an explicit independent parameter. -/
def specification
    {State : Type uState}
    {Scalar : Type uScalar}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar) :
    Specification State Chunk Coefficient Scalar where
  coefficientCount := coefficientCount
  source := source machine
  verifier := verifier
  assemble := assemble

@[simp] theorem specification_coefficientCount
    {State : Type uState}
    {Scalar : Type uScalar}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar) :
    (specification machine assemble).coefficientCount = coefficientCount := by
  rfl

@[simp] theorem specification_source
    {State : Type uState}
    {Scalar : Type uScalar}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar)
    (state : State)
    (coordinate : Nat) :
    (specification machine assemble).source state coordinate =
      source machine state coordinate := by
  rfl

/-- Flattening preserves the usual quotient/remainder position exactly. -/
theorem candidate_position_decomposition (position : Nat) :
    position / chunksPerDigest * chunksPerDigest +
        position % chunksPerDigest = position := by
  simpa [chunksPerDigest, Nat.mul_comm] using
    Nat.div_add_mod position chunksPerDigest

/-- Every candidate in the fixed prefix belongs to one of the first four
blocks. This is an index theorem, not a hash or encoding theorem. -/
theorem candidate_before_bound_uses_first_four_blocks
    {position : Nat}
    (beforeBound : position < candidateBound) :
    position / chunksPerDigest < digestRounds := by
  have numeric : position < 64 := by
    simpa [candidateBound] using beforeBound
  change position / 16 < 4
  omega

/-- Number of complete digest blocks a block-oriented execution has generated
when its least accepted cursor has consumed `consumed` candidates. -/
def blocksUsed (consumed : Nat) : Nat :=
  (consumed + (chunksPerDigest - 1)) / chunksPerDigest

/-- Any cursor in the fourth candidate window has generated exactly four whole
blocks, even when the final accepted coefficient occurs before lane 16. -/
theorem fourth_window_uses_four_blocks
    {consumed : Nat}
    (window :
      3 * chunksPerDigest < consumed /\
        consumed ≤ digestRounds * chunksPerDigest) :
    blocksUsed consumed = digestRounds := by
  have lower : 48 < consumed := by
    simpa [chunksPerDigest] using window.1
  have upper : consumed ≤ 64 := by
    simpa [chunksPerDigest, digestRounds] using window.2
  change (consumed + 15) / 16 = 4
  omega

/-- A successful production bounded execution and the unbounded reference run
therefore consume exactly four whole digest blocks. -/
theorem successful_execution_uses_four_blocks
    {stream : FirstAccepted.CandidateStream Chunk}
    (execution : FirstAccepted.BoundedExecution verifier coefficientCount
      stream candidateBound) :
    blocksUsed execution.consumed = digestRounds :=
  fourth_window_uses_four_blocks
    (successful_cursor_in_fourth_digest_window execution)

/-- The fixed source's successor state is exactly the block-machine state
reached by the successful reference execution. Candidate output and successor
state cannot drift independently at this abstract boundary. -/
theorem source_nextState_eq_referenceBlockState
    {State : Type uState}
    {Scalar : Type uScalar}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar)
    (initial : State)
    (coordinate : Nat)
    (execution : CoefficientExecution (specification machine assemble)
      candidateBound initial coordinate) :
    (sourceAt (specification machine assemble) initial coordinate).nextState =
      stateBeforeBlock machine
        (machine.enterScalar
          (stateAt (specification machine assemble) initial coordinate)
          coordinate)
        coordinate (blocksUsed execution.consumed) := by
  have blocks := successful_execution_uses_four_blocks execution
  change stateBeforeBlock machine
      (machine.enterScalar
        (stateAt (specification machine assemble) initial coordinate)
        coordinate)
      coordinate digestRounds =
    stateBeforeBlock machine
      (machine.enterScalar
        (stateAt (specification machine assemble) initial coordinate)
        coordinate)
      coordinate (blocksUsed execution.consumed)
  rw [blocks]

/-- Batch state threading agrees coordinate-by-coordinate with the successful
reference execution's whole-block state. -/
theorem stateAt_succ_eq_referenceBlockState
    {State : Type uState}
    {Scalar : Type uScalar}
    {challengeCount : Nat}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar)
    (initial : State)
    (batch : BatchExecution (specification machine assemble) challengeCount
      candidateBound initial)
    (coordinate : Fin challengeCount) :
    stateAt (specification machine assemble) initial (coordinate.val + 1) =
      stateBeforeBlock machine
        (machine.enterScalar
          (stateAt (specification machine assemble) initial coordinate.val)
          coordinate.val)
        coordinate.val (blocksUsed (batch.execution coordinate).consumed) := by
  rw [stateAt_succ]
  exact source_nextState_eq_referenceBlockState machine assemble initial
    coordinate.val (batch.execution coordinate)

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
