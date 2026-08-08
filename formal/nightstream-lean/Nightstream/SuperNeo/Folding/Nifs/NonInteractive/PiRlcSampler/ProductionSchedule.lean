import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Independent block schedule for production-shaped `Pi_RLC` coefficient sampling.

Protocol: SuperNeo `Pi_RLC` inside the candidate noninteractive NIFS.
Phase: transcript derivation of the complete scalar-challenge batch.
Constraint family: per-scalar domain separation, eight digest blocks,
candidate flattening, and successor-state threading.

Owns: an abstract deterministic block machine whose one `digestBlock` call
jointly returns both its successor state and eight candidates; the exact scalar
and per-block counter schedule; the flattened candidate stream; the fixed
eight-block successor state; and its direct use for every next scalar.

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
| `Pi_RLC` | digest block | `Machine.digestBlock` | jointly derive one successor state and exactly eight candidates |
| `Pi_RLC` | counter schedule | `stateBeforeBlock` | block `r` uses counter `i+r` |
| `Pi_RLC` | candidate order | `candidateStream` | flatten blocks and their within-block indices in order |
| `Pi_RLC` | fixed circuit | `source` | expose the first 64 candidates and the state after eight complete blocks |
| `Pi_RLC` | batch chaining | `stateAt_succ_eq_fixedBlockState` | each next scalar begins from the verifier state after all eight blocks |
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

/-- Every candidate in the fixed prefix belongs to one of the first eight
blocks. This is an index theorem, not a hash or encoding theorem. -/
theorem candidate_before_bound_uses_first_eight_blocks
    {position : Nat}
    (beforeBound : position < candidateBound) :
    position / chunksPerDigest < digestRounds := by
  have numeric : position < 64 := by
    simpa [candidateBound] using beforeBound
  change position / 8 < 8
  omega

/-- The fixed source's successor state is exactly the state after all eight
block-machine calls. Candidate output and successor state cannot drift at this
abstract boundary. -/
theorem source_nextState_eq_fixedBlockState
    {State : Type uState}
    {Scalar : Type uScalar}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar)
    (initial : State)
    (coordinate : Nat) :
    (sourceAt (specification machine assemble) initial coordinate).nextState =
      stateBeforeBlock machine
        (machine.enterScalar
          (stateAt (specification machine assemble) initial coordinate)
          coordinate)
        coordinate digestRounds := by
  rfl

/-- Batch state threading agrees coordinate-by-coordinate with the successful
reference execution's whole-block state. -/
theorem stateAt_succ_eq_fixedBlockState
    {State : Type uState}
    {Scalar : Type uScalar}
    {challengeCount : Nat}
    (machine : Machine State)
    (assemble : (Fin coefficientCount -> Coefficient) -> Scalar)
    (initial : State)
    (coordinate : Fin challengeCount) :
    stateAt (specification machine assemble) initial (coordinate.val + 1) =
      stateBeforeBlock machine
        (machine.enterScalar
          (stateAt (specification machine assemble) initial coordinate.val)
          coordinate.val)
        coordinate.val digestRounds := by
  rw [stateAt_succ]
  exact source_nextState_eq_fixedBlockState machine assemble initial
    coordinate.val

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
