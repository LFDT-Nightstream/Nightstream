import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Pure production-shaped transcript machine for `Pi_RLC` coefficient blocks.

Protocol: noninteractive SuperNeo `Pi_RLC` inside the recursive NIFS.
Phase: one scalar domain separator followed by four rejection-sampler digests.
Constraint family: transcript overwrite/permute schedule and little-endian
16-bit candidate extraction.

Owns: an eight-lane canonical Goldilocks state and rate cursor; raw-field
pair absorption including the length word; the exact `[0, coordinate]` and
`[1, counter]` schedules; `digest32`'s one-word squeeze gate; the extracted
Poseidon2 permutation function; and lane-major little-endian chunk order.

Does not own: the state reached after the complete `Pi_CCS` transcript,
equality with native `neo-transcript`, equality with `TranscriptGadget`, a
generated R1CS call/column schedule, bit-decomposition rows, rejection or
selection rows, distribution or bias, ring-scalar assembly, or row counts.

Emits constraints: no. This is executable implementation semantics consumed
by later Rust and R1CS refinement theorems.

Authority boundary: candidate chunks and successor transcript state are
computed by one deterministic machine. Neither can be supplied separately by
the prover. The permutation is the function independently extracted from the
exact 600-row Poseidon2 SSA artifact, not a carried digest value.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | state | `State` | eight canonical Goldilocks lanes and cursor `0..4` |
| `Pi_RLC` | raw absorb | `appendRawPair` | absorb length `2`, then both fields, with overwrite semantics |
| `Pi_RLC` | scalar domain | `enterScalar` | append `[0, coordinate mod 2^64]` |
| `Pi_RLC` | digest domain | `digestBlock` | append `[1, counter mod 2^64]`, absorb one, permute |
| `Pi_RLC` | candidate bytes | `digestChunks` | four little-endian 16-bit chunks from each of four canonical lanes |
| `Pi_RLC` | joint source | `machine` | expose chunks and successor state from that same digest execution |
| `Pi_RLC` | bounded success | `successfulExecution_successorState` | successful 54-of-64 sampling reaches the same four-block state |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

set_option maxHeartbeats 1000000

/-- Production Poseidon2 state width. -/
def width : Nat := 8

/-- Production transcript overwrite rate. -/
def rate : Nat := 4

/-- Modulus of Rust's `u64` wrapping counters. -/
def u64Modulus : Nat := 2 ^ 64

/-- Canonical Goldilocks field value represented without trusting a witness. -/
abbrev Field := Fin goldilocksP

/-- Canonical reduction into the production base field. -/
def fieldValue (value : Nat) : Field :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

/-- Rust `as u64`/wrapping-word semantics followed by field conversion. -/
def wordField (value : Nat) : Field :=
  fieldValue (value % u64Modulus)

/-- Pure native-transcript state. The types enforce lane canonicality and the
complete legal cursor range. -/
structure State where
  lanes : Fin width -> Field
  absorbed : Fin (rate + 1)

/-- Total natural-valued lane view expected by the extracted permutation. -/
def laneNat (state : State) (lane : Nat) : Nat :=
  if laneLt : lane < width then state.lanes ⟨lane, laneLt⟩ |>.val else 0

/-- Extracted Poseidon2 permutation, normalized into canonical field lanes. -/
def permute (state : State) : State where
  lanes := fun lane => fieldValue
    (Poseidon2PermutationSound.permute (laneNat state) lane.val)
  absorbed := ⟨0, by decide⟩

/-- Overwrite one lane, matching `Poseidon2Transcript::absorb_elem`. -/
def overwriteLane (lanes : Fin width -> Field) (index : Nat)
    (value : Field) : Fin width -> Field :=
  fun lane => if lane.val = index then value else lanes lane

/-- Absorb one canonical field element. A full cursor first permutes, then
overwrites lane zero. -/
def absorbElem (state : State) (value : Field) : State :=
  if room : state.absorbed.val < rate then
    { lanes := overwriteLane state.lanes state.absorbed.val value
      absorbed := ⟨state.absorbed.val + 1, by
        change state.absorbed.val + 1 < 5
        have roomNumeric : state.absorbed.val < 4 := by
          simpa [rate] using room
        omega⟩ }
  else
    let ready := permute state
    { lanes := overwriteLane ready.lanes 0 value
      absorbed := ⟨1, by decide⟩ }

/-- Exact `append_fields_raw(&[first, second])` shape: the length word `2`
is part of transcript semantics. -/
def appendRawPair (state : State) (first second : Nat) : State :=
  absorbElem
    (absorbElem (absorbElem state (wordField 2)) (wordField first))
    (wordField second)

/-- Exact `digest32` state transition and its four canonical output lanes. -/
def digest (state : State) : State × (Fin 4 -> Field) :=
  let next := permute (absorbElem state (wordField 1))
  (next, fun lane => next.lanes ⟨lane.val, by
    have laneLt := lane.isLt
    change lane.val < 8
    omega⟩)

/-- The `part`th little-endian 16-bit chunk of one canonical lane. -/
def laneChunk (lane : Field) (part : Fin 4) : Chunk :=
  ⟨(lane.val / (2 ^ (16 * part.val))) % chunkModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Four chunks per lane, four lanes per digest, in Rust's lane-major order. -/
def digestChunks (lanes : Fin 4 -> Field) :
    Fin chunksPerDigest -> Chunk :=
  fun position =>
    let laneIndex : Fin 4 := ⟨position.val / 4, by
      have positionLt : position.val < 16 := by
        simpa [chunksPerDigest] using position.isLt
      omega⟩
    let part : Fin 4 := ⟨position.val % 4, Nat.mod_lt _ (by decide)⟩
    laneChunk (lanes laneIndex) part

/-- Per-scalar domain separation. -/
def enterScalar (state : State) (coordinate : Nat) : State :=
  appendRawPair state 0 coordinate

/-- One counter block jointly returns its successor state and all 16 chunks. -/
def digestBlock (state : State) (counter : Nat) :
    State × (Fin chunksPerDigest -> Chunk) :=
  let result := digest (appendRawPair state 1 counter)
  (result.1, digestChunks result.2)

/-- Concrete instantiation of the abstract jointly owned production schedule. -/
def machine : Machine State where
  enterScalar := enterScalar
  digestBlock := digestBlock

/-- Complete production-shaped coefficient sampler over this transcript
machine. Scalar output remains the exact 54-coordinate coefficient vector. -/
def specification :
    Specification State Chunk Coefficient Scalar :=
  ProductionSchedule.specification machine assembleCoefficients

@[simp] theorem machine_enterScalar (state : State) (coordinate : Nat) :
    machine.enterScalar state coordinate = enterScalar state coordinate := by
  rfl

@[simp] theorem machine_digestBlock (state : State) (counter : Nat) :
    machine.digestBlock state counter = digestBlock state counter := by
  rfl

/-- The implementation model fixes the exact lane/part quotient-remainder
ordering; no caller supplies a byte permutation. -/
theorem digestChunks_lane_part
    (lanes : Fin 4 -> Field) (lane part : Fin 4) :
    digestChunks lanes
      ⟨lane.val * 4 + part.val, by
        have laneLt := lane.isLt
        have partLt := part.isLt
        change lane.val * 4 + part.val < chunksPerDigest
        simp only [chunksPerDigest]
        omega⟩ =
      laneChunk (lanes lane) part := by
  unfold digestChunks
  dsimp only
  have laneIndexEq :
      (⟨(lane.val * 4 + part.val) / 4, by
        have laneLt := lane.isLt
        have partLt := part.isLt
        omega⟩ : Fin 4) = lane := by
    apply Fin.ext
    change (lane.val * 4 + part.val) / 4 = lane.val
    have decomposition := Nat.div_add_mod (lane.val * 4 + part.val) 4
    have remainderLt := Nat.mod_lt (lane.val * 4 + part.val) (by decide : 0 < 4)
    have partLt := part.isLt
    omega
  have partEq :
      (⟨(lane.val * 4 + part.val) % 4, Nat.mod_lt _ (by decide)⟩ :
        Fin 4) = part := by
    apply Fin.ext
    change (lane.val * 4 + part.val) % 4 = part.val
    have decomposition := Nat.div_add_mod (lane.val * 4 + part.val) 4
    have remainderLt := Nat.mod_lt (lane.val * 4 + part.val) (by decide : 0 < 4)
    have partLt := part.isLt
    omega
  rw [laneIndexEq, partEq]

@[simp] theorem digestBlock_absorbed_zero (state : State) (counter : Nat) :
    (digestBlock state counter).1.absorbed.val = 0 := by
  rfl

/-- Conditional state/candidate agreement inherited from the independent
first-accepted semantics: if the fixed sampler succeeds, its successor is
exactly the state after the same four concrete digest blocks. -/
theorem successfulExecution_successorState
    (initial : State)
    (coordinate : Nat)
    (execution : CoefficientExecution specification candidateBound
      initial coordinate) :
    (sourceAt specification initial coordinate).nextState =
      stateBeforeBlock machine
        (enterScalar (stateAt specification initial coordinate) coordinate)
        coordinate (blocksUsed execution.consumed) := by
  exact source_nextState_eq_referenceBlockState machine
    assembleCoefficients initial coordinate execution

end Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
