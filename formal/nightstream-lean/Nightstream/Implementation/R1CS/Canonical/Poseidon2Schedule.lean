import Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Program

/-!
Contract: the concrete Poseidon2 round schedule, and the permutation program
built from it.

Owns: the symbolic state after every round; the map from an S-box index to the
linear combination that S-box consumes; and the resulting fully concrete row
program.

Does not own: matrix values (`Poseidon2Matrices`), row/column allocation
(`Poseidon2Program`), absorption or padding (Phase 3).

## What this closes

`Poseidon2Program` deliberately left `Schedule` abstract so allocation,
ownership and row counts could be proved independently of matrix arithmetic.
That was the right factoring, but it meant the 352/344 figures were conditional
on a conforming schedule that did not exist.  This module supplies it, so
`canonicalProgram` is a closed term and `canonicalProgram_cost` is a cost of
something real rather than of a hypothesis.

Round constants stay a parameter.  Every theorem here is universally quantified
over `Constants`, which is strictly stronger than pinning the 86 sampled values
and avoids importing a ChaCha8 stream into the kernel.  Constant *values* are
`POSEIDON2-ROUND-CONSTANT-CONFORMANCE` and are only needed for bit-for-bit
claims against the Rust digest.

## Round structure

Transcribed from `absorb_words_then_permute_values` in
`crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs`:

    state = external(input)                          -- pre-layer, no S-box
    4x:  sbox all 8 lanes (+ initial[r][lane]); state = external(sbox)
    22x: sbox lane 0 only  (+ internal[r]);      state = internal(sbox_in)
    4x:  sbox all 8 lanes (+ terminal[r][lane]); state = external(sbox)

A full round S-boxes every lane, so the next state depends only on the eight
fresh output columns — that is why `initialState` and `terminalState` are not
recursive.  A partial round S-boxes lane 0 only and lanes 1..7 flow onward
unchanged, so `partialState` *is* recursive.  That asymmetry is exactly what
drives the support recurrence.

S-box index families, verified by `sboxIndex_partition`:

    [0, 32)   initial full   index = round * 8 + lane
    [32, 54)  partial        index = 32 + round
    [54, 86)  terminal full  index = 54 + round * 8 + lane
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program

/-! ## Round constants

Shaped to mirror `Poseidon2Constants` in the Rust source: two full-round
tables and one internal table for the partial rounds. -/

structure Constants where
  initial : Nat → Fin width → Nat
  internal : Nat → Nat
  terminal : Nat → Fin width → Nat

/-- `POSEIDON2_HALF_FULL_ROUNDS`.  The eight external rounds split evenly
around the partial block. -/
def halfFullRounds : Nat := 4

theorem externalRounds_split : externalRounds = 2 * halfFullRounds := by decide

/-! ## S-box output columns

`sboxColumn` is indexed by `Fin sboxCount`; the round recursion is over `Nat`,
so it uses the raw arithmetic form and `sboxOutput_eq_sboxColumn` records that
the two agree. -/

/-- The output column of S-box `index` — slot 3 of its four-column frame. -/
def sboxOutput (layout : Layout) (index : Nat) : Nat :=
  layout.auxBase + columnsPerSbox * index + 3

theorem sboxOutput_eq_sboxColumn (layout : Layout) (index : Fin sboxCount) :
    sboxOutput layout index.val = sboxColumn layout index ⟨3, by decide⟩ := rfl

/-! ## Index families -/

def initialSboxIndex (round lane : Nat) : Nat := round * width + lane

def partialSboxIndex (round : Nat) : Nat := halfFullRounds * width + round

def terminalSboxIndex (round lane : Nat) : Nat :=
  halfFullRounds * width + partialRounds + round * width + lane

/-- **The three families partition `[0, 86)`.** -/
theorem sboxIndex_partition :
    ∀ index : Fin sboxCount,
      index.val < 32 ∨ (32 ≤ index.val ∧ index.val < 54) ∨
        (54 ≤ index.val ∧ index.val < 86) := by decide

theorem initialSboxIndex_range :
    ∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      initialSboxIndex round.val lane.val < 32 := by decide

theorem partialSboxIndex_range :
    ∀ round : Fin partialRounds,
      32 ≤ partialSboxIndex round.val ∧ partialSboxIndex round.val < 54 := by
  decide

theorem terminalSboxIndex_range :
    ∀ round : Fin halfFullRounds, ∀ lane : Fin width,
      54 ≤ terminalSboxIndex round.val lane.val ∧
        terminalSboxIndex round.val lane.val < 86 := by decide

/-! ### The decomposition used by `scheduleOf` inverts the index maps

Without these, the `/ width` and `% width` case analysis below would be an
unchecked guess at which round and lane an index denotes. -/

theorem initialSboxIndex_roundtrip :
    ∀ index : Fin 32, initialSboxIndex (index.val / width) (index.val % width)
      = index.val := by decide

theorem partialSboxIndex_roundtrip :
    ∀ index : Fin 22, partialSboxIndex index.val = 32 + index.val := by decide

theorem terminalSboxIndex_roundtrip :
    ∀ offset : Fin 32,
      terminalSboxIndex (offset.val / width) (offset.val % width)
        = 54 + offset.val := by decide

/-! ## Symbolic round evolution

Each state is one linear combination per lane, carried without materializing
any intermediate value. -/

/-- State entering initial full round `round`; `0` is the post-pre-layer
state. -/
def initialState (layout : Layout) : Nat → State
  | 0 => applyMatrix externalMatrix (fun lane => [(layout.inputPort lane, 1)])
  | round + 1 =>
      applyMatrix externalMatrix
        (fun lane => [(sboxOutput layout (initialSboxIndex round lane.val), 1)])


/-! ## Generalized entry state

`initialState` derives its starting state from declared input ports, one column
per lane.  A sponge needs to enter a permutation carrying `state + chunk` — two
terms per lane — which that shape cannot express
(`POSEIDON2-INITIAL-STATE-GENERALIZATION`).

`initialStateFrom` takes the entry state as a parameter.  It is added rather
than substituted so no existing theorem changes: `initialState_eq_from` is the
bridge, and migrating the downstream results is a separate step. -/

def initialStateFrom (layout : Layout) (entry : State) : Nat → State
  | 0 => applyMatrix externalMatrix entry
  | round + 1 =>
      applyMatrix externalMatrix
        (fun lane => [(sboxOutput layout (initialSboxIndex round lane.val), 1)])

/-- **The port-derived form is the parameterised one at the port entry.**  So
every theorem proved about `initialState` is a theorem about `initialStateFrom`
at that entry, and the sponge's different entry is the only thing needing new
work. -/
theorem initialState_eq_from (layout : Layout) (round : Nat) :
    initialState layout round
      = initialStateFrom layout
          (fun lane => [(layout.inputPort lane, 1)]) round := by
  cases round <;> rfl

/-- Past the first round the entry is irrelevant: a full round has replaced
every lane with a fresh output.  This is what will let the sponge inherit the
support recurrence unchanged from round one onward. -/
theorem initialStateFrom_succ_entry_irrelevant
    (layout : Layout) (first second : State) (round : Nat) :
    initialStateFrom layout first (round + 1)
      = initialStateFrom layout second (round + 1) := rfl


/-- **Everything from the partial block onward is entry-independent.**  The
partial block enters at `initialStateFrom layout entry halfFullRounds`, and
`halfFullRounds` is `3 + 1`, so the entry has already been erased by a full
round.  A sponge therefore reuses `partialState`, `terminalState`, `finalState`
and the partial and terminal schedule families **verbatim** — only the initial
family sees the carried entry at all.

This is what makes the sponge cheap to build on Phase 2: 54 of the 86 S-boxes,
the whole support recurrence past round 0, and every terminal result apply
unchanged. -/
theorem initialStateFrom_halfFull_eq
    (layout : Layout) (entry : State) :
    initialStateFrom layout entry halfFullRounds
      = initialState layout halfFullRounds := by
  rw [initialState_eq_from]
  exact initialStateFrom_succ_entry_irrelevant layout entry _ 3

/-- State entering partial round `round`.  Recursive: lanes 1..7 are not
S-boxed and flow into the internal layer unchanged. -/
def partialState (layout : Layout) : Nat → State
  | 0 => initialState layout halfFullRounds
  | round + 1 =>
      applyMatrix internalMatrix
        (fun lane =>
          if lane.val = 0 then [(sboxOutput layout (partialSboxIndex round), 1)]
          else partialState layout round lane)

/-- State entering terminal full round `round`. -/
def terminalState (layout : Layout) : Nat → State
  | 0 => partialState layout partialRounds
  | round + 1 =>
      applyMatrix externalMatrix
        (fun lane => [(sboxOutput layout (terminalSboxIndex round lane.val), 1)])

/-- The state after the complete permutation, before terminal output binding. -/
def finalState (layout : Layout) : State := terminalState layout halfFullRounds

/-! ## The schedule

Every S-box consumes its lane's current combination plus that round's constant
on the constant wire — never a materialized column. -/

def scheduleOf (layout : Layout) (constants : Constants) :
    Poseidon2Program.Schedule :=
  fun index =>
    if index.val < halfFullRounds * width then
      addConstant
        (constants.initial (index.val / width)
          ⟨index.val % width, Nat.mod_lt _ (by decide)⟩)
        (initialState layout (index.val / width)
          ⟨index.val % width, Nat.mod_lt _ (by decide)⟩)
    else if index.val < halfFullRounds * width + partialRounds then
      addConstant (constants.internal (index.val - halfFullRounds * width))
        (partialState layout (index.val - halfFullRounds * width)
          ⟨0, by decide⟩)
    else
      addConstant
        (constants.terminal
          ((index.val - (halfFullRounds * width + partialRounds)) / width)
          ⟨(index.val - (halfFullRounds * width + partialRounds)) % width,
            Nat.mod_lt _ (by decide)⟩)
        (terminalState layout
          ((index.val - (halfFullRounds * width + partialRounds)) / width)
          ⟨(index.val - (halfFullRounds * width + partialRounds)) % width,
            Nat.mod_lt _ (by decide)⟩)


/-! ## The schedule and program on a carried entry

Only the initial full-round family consults the entry —
`initialStateFrom_halfFull_eq` shows the partial and terminal families are
entry-independent — so `scheduleOfFrom` differs from `scheduleOf` in exactly one
branch and delegates the rest.

This is what a sponge needs to emit rows for a call it enters carrying
`state + chunk` rather than declared input ports. -/

def scheduleOfFrom (layout : Layout) (entry : State) (constants : Constants) :
    Poseidon2Program.Schedule :=
  fun index =>
    if index.val < halfFullRounds * width then
      addConstant
        (constants.initial (index.val / width)
          ⟨index.val % width, Nat.mod_lt _ (by decide)⟩)
        (initialStateFrom layout entry (index.val / width)
          ⟨index.val % width, Nat.mod_lt _ (by decide)⟩)
    else scheduleOf layout constants index

/-- **At the port entry it is the existing schedule.**  So every theorem proved
about `scheduleOf` is a theorem about `scheduleOfFrom` at that entry, and only
the sponge's different entry needs new work. -/
theorem scheduleOfFrom_port_entry (layout : Layout) (constants : Constants) :
    scheduleOfFrom layout (fun lane => [(layout.inputPort lane, 1)]) constants
      = scheduleOf layout constants := by
  funext index
  unfold scheduleOfFrom
  split
  · rw [← initialState_eq_from]
    unfold scheduleOf
    rw [if_pos (by assumption)]
  · rfl

/-- The permutation program for a call entered on a carried state. -/
def canonicalProgramFrom
    (layout : Layout) (entry : State) (constants : Constants) : List Row :=
  permutationProgram layout (scheduleOfFrom layout entry constants)
    (finalState layout)

/-- **Row count is unchanged by the entry.**  `sboxRows` emits four rows
whatever combination it is handed, so a carried entry costs no extra row — the
absorption really is free. -/
theorem canonicalProgramFrom_length
    (layout : Layout) (entry : State) (constants : Constants) :
    (canonicalProgramFrom layout entry constants).length = 352 :=
  permutationProgram_length_eq layout (scheduleOfFrom layout entry constants)
    (finalState layout)

/-! ## The concrete program

`Schedule` is no longer a hypothesis: `canonicalProgram` is a closed term, so
the row and column counts below describe an object that exists. -/

def canonicalProgram (layout : Layout) (constants : Constants) : List Row :=
  permutationProgram layout (scheduleOf layout constants) (finalState layout)

/-- **The concrete permutation is 352 rows.**  Same figure as
`permutationProgram_length_eq`, now discharged of its schedule hypothesis. -/
theorem canonicalProgram_length (layout : Layout) (constants : Constants) :
    (canonicalProgram layout constants).length = 352 :=
  permutationProgram_length_eq layout (scheduleOf layout constants)
    (finalState layout)

/-- **The concrete permutation cost**, every component a receipt fold. -/
theorem canonicalProgram_cost (layout : Layout) (constants : Constants) :
    permutationCost layout (scheduleOf layout constants) (finalState layout)
      = ⟨352, 0, 0, 344⟩ :=
  permutationProgram_cost_eq_receiptFold layout (scheduleOf layout constants)
    (finalState layout)

/-- **Every S-box of the concrete program computes `x⁷` on its scheduled
combination.**  Inherited from the assembled-program chain lemma; what it does
*not* yet say is that those combinations are Poseidon2's successive states,
which is `POSEIDON2-ROUND-INDUCTION`. -/
theorem canonicalProgram_sbox_chains
    (layout : Layout) (constants : Constants)
    (z : Nat → Nat) (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies (canonicalProgram layout constants) z)
    (index : Fin sboxCount) :
    let frame := frameAt layout index (scheduleOf layout constants index)
    z frame.square = lcEval z frame.input * lcEval z frame.input % goldilocksP ∧
      z frame.fourth = z frame.square * z frame.square % goldilocksP ∧
      z frame.sixth = z frame.square * z frame.fourth % goldilocksP ∧
      z frame.output = lcEval z frame.input * z frame.sixth % goldilocksP :=
  permutationProgram_sbox_chains layout (scheduleOf layout constants)
    (finalState layout) z residues satisfied index

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
