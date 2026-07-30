import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile

/-!
Contract: emit the Poseidon2-duplex portion of the production-shaped
`Pi_RLC` coefficient sampler from Lean.

`PiRlcCanonicalMachine` fixes the value-level schedule.  This module gives the
same schedule a symbolic row program:

* absorb the raw-pair length word and scalar tag;
* for each of four blocks, absorb the raw-pair length word, block tag, and
  exact `coordinate + round` counter;
* force the pre-squeeze gate permutation; and
* carry the four freshly permuted digest lanes into the next decoding layer.

The refinement theorems are driven only by satisfaction of
`SymbolicDuplex.rows`.  No digest, transcript state, or verifier conclusion is
accepted as a premise.

This module does not yet decompose the four digest lanes into sixteen checked
16-bit chunks or implement first-accepted selection.  Consequently it owns
the transcript schedule and digest-lane source, but not a complete physical
sampler or `nifsVerify` recipe.

Assurance tier: canonical model/R1CS refinement.  No generated row or Rust
result is imported.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule

/-- One verifier word on the shared constant wire.  The `u64` conversion and
the subsequent Goldilocks reduction are both explicit. -/
def fieldWord (value : Nat) : LinCombNormal.LinComb :=
  [(0, (PiRlcCanonicalMachine.word value) % goldilocksP)]

theorem fieldWord_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (value : Nat) :
    lcEval assignment (fieldWord value) =
      (PiRlcCanonicalMachine.word value) % goldilocksP := by
  unfold fieldWord
  rw [lcEval_eq_rawSum, rawSum_cons, constantWire]
  simp only [rawSum, List.foldl_nil, Nat.add_zero, Nat.mul_one,
    Nat.mod_mod]

/-- Exact symbolic serialization of `append_fields_raw(&[first, second])`. -/
def rawPairFields (first second : Nat) : List LinCombNormal.LinComb :=
  [fieldWord 2, fieldWord first, fieldWord second]

theorem rawPairFields_values
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (first second : Nat) :
    (rawPairFields first second).map (lcEval assignment) =
      [ PiRlcCanonicalMachine.word 2 % goldilocksP,
        PiRlcCanonicalMachine.word first % goldilocksP,
        PiRlcCanonicalMachine.word second % goldilocksP ] := by
  simp only [rawPairFields, List.map_cons, List.map_nil,
    fieldWord_eval assignment constantWire]

/-- Absorption already reduces its input, so pre-reducing a serialized word
does not change the duplex transition. -/
theorem absorbElem_mod
    (constants : Constants) (value : Nat) (state : Poseidon2Duplex.State) :
    Poseidon2Duplex.absorbElem constants (value % goldilocksP) state =
      Poseidon2Duplex.absorbElem constants value state := by
  unfold Poseidon2Duplex.absorbElem
  simp only [Nat.mod_mod]

/-- Symbolic raw-pair absorption. -/
def appendRawPair
    (base first second : Nat) (builder : SymbolicDuplex.Builder) :
    SymbolicDuplex.Builder :=
  SymbolicDuplex.absorbMany base (rawPairFields first second) builder

theorem appendRawPair_extends
    (base first second : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (appendRawPair base first second builder) :=
  SymbolicDuplexSemantics.absorbMany_extends base
    (rawPairFields first second) builder

/-- A raw pair starting at cursor zero emits no permutation and leaves three
absorbed words pending. -/
theorem appendRawPair_shape_of_zero
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (cursorZero : builder.absorbed = 0) :
    (appendRawPair base first second builder).entries.length =
        builder.entries.length ∧
      (appendRawPair base first second builder).absorbed = 3 := by
  simp [appendRawPair, rawPairFields, SymbolicDuplex.absorbMany,
    SymbolicDuplex.absorb, SymbolicDuplex.guarded,
    cursorZero, Poseidon2Sponge.rate]

/-- A raw pair starting at cursor one emits no permutation and fills the
four-word rate.  The next write, rather than this one, performs the guarded
flush. -/
theorem appendRawPair_shape_of_one
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (cursorOne : builder.absorbed = 1) :
    (appendRawPair base first second builder).entries.length =
        builder.entries.length ∧
      (appendRawPair base first second builder).absorbed = 4 := by
  simp [appendRawPair, rawPairFields, SymbolicDuplex.absorbMany,
    SymbolicDuplex.absorb, SymbolicDuplex.guarded,
    cursorOne, Poseidon2Sponge.rate]

/-- A raw pair starting at cursor three flushes exactly once and leaves two
absorbed words pending. -/
theorem appendRawPair_shape_of_three
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (cursorThree : builder.absorbed = 3) :
    (appendRawPair base first second builder).entries.length =
        builder.entries.length + 1 ∧
      (appendRawPair base first second builder).absorbed = 2 := by
  simp [appendRawPair, rawPairFields, SymbolicDuplex.absorbMany,
    SymbolicDuplex.absorb, SymbolicDuplex.guarded,
    SymbolicDuplex.permute, cursorThree, Poseidon2Sponge.rate]

/-- A raw pair starting at a full cursor flushes exactly once and leaves
three words pending. -/
theorem appendRawPair_shape_of_four
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (cursorFour : builder.absorbed = 4) :
    (appendRawPair base first second builder).entries.length =
        builder.entries.length + 1 ∧
      (appendRawPair base first second builder).absorbed = 3 := by
  simp [appendRawPair, rawPairFields, SymbolicDuplex.absorbMany,
    SymbolicDuplex.absorb, SymbolicDuplex.guarded,
    SymbolicDuplex.permute, cursorFour, Poseidon2Sponge.rate]

/-- Symbolic raw-pair absorption denotes the value-level raw-pair operation. -/
theorem decoded_appendRawPair
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (first second : Nat) (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (appendRawPair base first second builder)) :
    decodedBuilder assignment (appendRawPair base first second builder) =
      PiRlcCanonicalMachine.appendRawPair constants
        (decodedBuilder assignment builder) first second := by
  unfold appendRawPair at valid ⊢
  rw [SymbolicDuplexSemantics.decodedBuilder_absorbMany
    base constants assignment (rawPairFields first second) builder valid,
    rawPairFields_values assignment constantWire]
  unfold PiRlcCanonicalMachine.appendRawPair
  simp only [Poseidon2Duplex.absorbList]
  rw [absorbElem_mod, absorbElem_mod, absorbElem_mod]

/-- Per-scalar domain entry. -/
def enterScalar
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) : SymbolicDuplex.Builder :=
  appendRawPair base 0 coordinate builder

theorem enterScalar_extends
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) :
    Extends builder (enterScalar base builder coordinate) :=
  appendRawPair_extends base 0 coordinate builder

theorem decoded_enterScalar
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder) (coordinate : Nat)
    (valid :
      Valid base constants assignment
        (enterScalar base builder coordinate)) :
    decodedBuilder assignment (enterScalar base builder coordinate) =
      PiRlcCanonicalMachine.enterScalar constants
        (decodedBuilder assignment builder) coordinate :=
  decoded_appendRawPair base constants assignment constantWire
    0 coordinate builder valid

/-- One block's successor builder.  The four digest lanes are projections of
this builder, so the state and candidate source cannot drift independently. -/
def digestBlock
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) : SymbolicDuplex.Builder :=
  SymbolicDuplex.gate base (appendRawPair base 1 counter builder)

/-- The four freshly permuted digest-lane expressions. -/
def digestLanes
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) : Fin 4 → LinCombNormal.LinComb :=
  let next := digestBlock base builder counter
  fun lane =>
    next.lanes ⟨lane.val, by
      have laneLt := lane.isLt
      change lane.val < width
      simp only [width]
      omega⟩

theorem digestBlock_extends
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) :
    Extends builder (digestBlock base builder counter) :=
  (appendRawPair_extends base 1 counter builder).trans
    (SymbolicDuplexSemantics.gate_extends base
      (appendRawPair base 1 counter builder))

/-- A digest block beginning at cursor zero emits its forced gate
permutation and no automatic flush. -/
theorem digestBlock_entries_length_of_zero
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (cursorZero : builder.absorbed = 0) :
    (digestBlock base builder counter).entries.length =
      builder.entries.length + 1 := by
  have pairShape :=
    appendRawPair_shape_of_zero base 1 counter builder cursorZero
  simp [digestBlock, SymbolicDuplex.gate, SymbolicDuplex.absorb,
    SymbolicDuplex.guarded, SymbolicDuplex.permute,
    pairShape.1, pairShape.2, Poseidon2Sponge.rate]

/-- The first digest block after a three-word scalar prefix emits one
automatic flush plus its forced gate permutation. -/
theorem digestBlock_entries_length_of_three
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (cursorThree : builder.absorbed = 3) :
    (digestBlock base builder counter).entries.length =
      builder.entries.length + 2 := by
  have pairShape :=
    appendRawPair_shape_of_three base 1 counter builder cursorThree
  simp [digestBlock, SymbolicDuplex.gate, SymbolicDuplex.absorb,
    SymbolicDuplex.guarded, SymbolicDuplex.permute,
    pairShape.1, pairShape.2, Poseidon2Sponge.rate]

/-- A digest block beginning at a full cursor emits the guarded flush for its
first raw-pair word and its forced gate permutation. -/
theorem digestBlock_entries_length_of_four
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (cursorFour : builder.absorbed = 4) :
    (digestBlock base builder counter).entries.length =
      builder.entries.length + 2 := by
  have pairShape :=
    appendRawPair_shape_of_four base 1 counter builder cursorFour
  simp [digestBlock, SymbolicDuplex.gate, SymbolicDuplex.absorb,
    SymbolicDuplex.guarded, SymbolicDuplex.permute,
    pairShape.1, pairShape.2, Poseidon2Sponge.rate]

/-- One symbolic digest block has exactly the value-level successor state. -/
theorem decoded_digestBlock
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder) (counter : Nat)
    (valid :
      Valid base constants assignment
        (digestBlock base builder counter)) :
    decodedBuilder assignment (digestBlock base builder counter) =
      (PiRlcCanonicalMachine.digestBlock constants
        (decodedBuilder assignment builder) counter).1 := by
  unfold digestBlock PiRlcCanonicalMachine.digestBlock
    PiRlcCanonicalMachine.digest
  rw [SymbolicDuplexSemantics.decodedBuilder_gate
    base constants assignment
      (appendRawPair base 1 counter builder) constantWire valid]
  congr 1
  apply decoded_appendRawPair base constants assignment constantWire
  exact valid.of_extends
    (SymbolicDuplexSemantics.gate_extends base
      (appendRawPair base 1 counter builder))

/-- Every exposed symbolic digest lane is the matching lane of the same
value-level block execution. -/
theorem digestLanes_eval
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder) (counter : Nat)
    (valid :
      Valid base constants assignment
        (digestBlock base builder counter))
    (lane : Fin 4) :
    lcEval assignment (digestLanes base builder counter lane) =
      (PiRlcCanonicalMachine.digest constants
        (PiRlcCanonicalMachine.appendRawPair constants
          (decodedBuilder assignment builder) 1 counter)).2 lane := by
  have blockEq :=
    decoded_digestBlock base constants assignment constantWire
      builder counter valid
  have laneEq := congrArg
    (fun state => state.lanes
      ⟨lane.val, by
        have laneLt := lane.isLt
        change lane.val < width
        simp only [width]
        omega⟩)
    blockEq
  simpa only [digestLanes, decodedBuilder, evalState,
    PiRlcCanonicalMachine.digestBlock,
    PiRlcCanonicalMachine.digest] using laneEq

/-- Symbolic state before block `round`, with the exact `seed + round`
counter schedule. -/
def stateBeforeBlock
    (base : Nat) (entered : SymbolicDuplex.Builder)
    (seed : Nat) : Nat → SymbolicDuplex.Builder
  | 0 => entered
  | round + 1 =>
      digestBlock base (stateBeforeBlock base entered seed round)
        (seed + round)

theorem stateBeforeBlock_extends
    (base : Nat) (entered : SymbolicDuplex.Builder) (seed : Nat) :
    ∀ round, Extends entered (stateBeforeBlock base entered seed round)
  | 0 => Extends.refl entered
  | round + 1 =>
      (stateBeforeBlock_extends base entered seed round).trans
        (digestBlock_extends base
          (stateBeforeBlock base entered seed round) (seed + round))

/-- The complete symbolic block recurrence refines the abstract production
schedule, round by round. -/
theorem decoded_stateBeforeBlock
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (entered : SymbolicDuplex.Builder) (seed : Nat) :
    ∀ round,
      Valid base constants assignment
          (stateBeforeBlock base entered seed round) →
      decodedBuilder assignment
          (stateBeforeBlock base entered seed round) =
        ProductionSchedule.stateBeforeBlock
          (PiRlcCanonicalMachine.machine constants)
          (decodedBuilder assignment entered) seed round
  | 0, _ => rfl
  | round + 1, valid => by
      have priorValid :
          Valid base constants assignment
            (stateBeforeBlock base entered seed round) :=
        valid.of_extends
          (digestBlock_extends base
            (stateBeforeBlock base entered seed round) (seed + round))
      have priorEq :=
        decoded_stateBeforeBlock base constants assignment constantWire
          entered seed round priorValid
      have blockEq :=
        decoded_digestBlock base constants assignment constantWire
          (stateBeforeBlock base entered seed round) (seed + round) valid
      calc
        decodedBuilder assignment
            (stateBeforeBlock base entered seed (round + 1)) =
          ((PiRlcCanonicalMachine.machine constants).digestBlock
            (decodedBuilder assignment
              (stateBeforeBlock base entered seed round))
            (seed + round)).1 := by
              simpa only [stateBeforeBlock,
                PiRlcCanonicalMachine.machine_digestBlock] using blockEq
        _ =
          ((PiRlcCanonicalMachine.machine constants).digestBlock
            (ProductionSchedule.stateBeforeBlock
              (PiRlcCanonicalMachine.machine constants)
              (decodedBuilder assignment entered) seed round)
            (seed + round)).1 := by
              exact congrArg
                (fun state =>
                  ((PiRlcCanonicalMachine.machine constants).digestBlock
                    state (seed + round)).1)
                priorEq
        _ =
          ProductionSchedule.stateBeforeBlock
            (PiRlcCanonicalMachine.machine constants)
            (decodedBuilder assignment entered) seed (round + 1) := rfl

/-- One complete scalar source: enter its coordinate and execute all four
digest blocks, irrespective of the later rejection-selection outcome. -/
def scalarBuilder
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) : SymbolicDuplex.Builder :=
  stateBeforeBlock base (enterScalar base builder coordinate)
    coordinate digestRounds

theorem scalarBuilder_extends
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) :
    Extends builder (scalarBuilder base builder coordinate) :=
  (enterScalar_extends base builder coordinate).trans
    (stateBeforeBlock_extends base
      (enterScalar base builder coordinate) coordinate digestRounds)

@[simp] theorem scalarBuilder_absorbed
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) :
    (scalarBuilder base builder coordinate).absorbed = 0 := by
  rfl

/-- From a freshly gated cursor, one scalar uses five permutations: the first
block flushes the three-word scalar prefix while absorbing its own raw pair,
and each of the four blocks ends in one forced gate permutation. -/
theorem scalarBuilder_entries_length_of_zero
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (cursorZero : builder.absorbed = 0) :
    (scalarBuilder base builder coordinate).entries.length =
      builder.entries.length + 5 := by
  let entered := enterScalar base builder coordinate
  let block0 := digestBlock base entered coordinate
  let block1 := digestBlock base block0 (coordinate + 1)
  let block2 := digestBlock base block1 (coordinate + 2)
  let block3 := digestBlock base block2 (coordinate + 3)
  have enteredShape :=
    appendRawPair_shape_of_zero base 0 coordinate builder cursorZero
  have block0Length :=
    digestBlock_entries_length_of_three base entered coordinate
      (by simpa [entered, enterScalar] using enteredShape.2)
  have block1Length :=
    digestBlock_entries_length_of_zero base block0 (coordinate + 1)
      (by rfl)
  have block2Length :=
    digestBlock_entries_length_of_zero base block1 (coordinate + 2)
      (by rfl)
  have block3Length :=
    digestBlock_entries_length_of_zero base block2 (coordinate + 3)
      (by rfl)
  have scalarEq : scalarBuilder base builder coordinate = block3 := by
    rfl
  rw [scalarEq]
  rw [block3Length, block2Length, block1Length, block0Length]
  change entered.entries.length + 2 + 1 + 1 + 1 =
    builder.entries.length + 5
  rw [show entered.entries.length = builder.entries.length by
    simpa [entered, enterScalar] using enteredShape.1]

/-- From the selected post-PiCCS cursor one, one scalar still uses five
permutations: the scalar prefix fills the rate, the first digest block flushes
it once, and every digest block ends in one forced permutation. -/
theorem scalarBuilder_entries_length_of_one
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (cursorOne : builder.absorbed = 1) :
    (scalarBuilder base builder coordinate).entries.length =
      builder.entries.length + 5 := by
  let entered := enterScalar base builder coordinate
  let block0 := digestBlock base entered coordinate
  let block1 := digestBlock base block0 (coordinate + 1)
  let block2 := digestBlock base block1 (coordinate + 2)
  let block3 := digestBlock base block2 (coordinate + 3)
  have enteredShape :=
    appendRawPair_shape_of_one base 0 coordinate builder cursorOne
  have block0Length :=
    digestBlock_entries_length_of_four base entered coordinate
      (by simpa [entered, enterScalar] using enteredShape.2)
  have block1Length :=
    digestBlock_entries_length_of_zero base block0 (coordinate + 1)
      (by rfl)
  have block2Length :=
    digestBlock_entries_length_of_zero base block1 (coordinate + 2)
      (by rfl)
  have block3Length :=
    digestBlock_entries_length_of_zero base block2 (coordinate + 3)
      (by rfl)
  have scalarEq : scalarBuilder base builder coordinate = block3 := by
    rfl
  rw [scalarEq, block3Length, block2Length, block1Length, block0Length]
  change entered.entries.length + 2 + 1 + 1 + 1 =
    builder.entries.length + 5
  rw [show entered.entries.length = builder.entries.length by
    simpa [entered, enterScalar] using enteredShape.1]

/-- One symbolic scalar source reaches exactly the successor state owned by
the value-level sampler source. -/
theorem decoded_scalarBuilder
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (builder : SymbolicDuplex.Builder) (coordinate : Nat)
    (valid :
      Valid base constants assignment
        (scalarBuilder base builder coordinate)) :
    decodedBuilder assignment (scalarBuilder base builder coordinate) =
      (ProductionSchedule.source
        (PiRlcCanonicalMachine.machine constants)
        (decodedBuilder assignment builder) coordinate).nextState := by
  unfold scalarBuilder ProductionSchedule.source
  simp only
  rw [decoded_stateBeforeBlock base constants assignment constantWire
    (enterScalar base builder coordinate) coordinate digestRounds valid]
  congr 1
  apply decoded_enterScalar base constants assignment constantWire
  exact valid.of_extends
    (stateBeforeBlock_extends base
      (enterScalar base builder coordinate) coordinate digestRounds)

/-- Value-level recurrence induced by the exact production source.  It is the
specialized `PiRlcSampler.stateAt` recurrence, kept local so subsequent
symbolic proofs do not repeatedly unfold the complete sampler structure. -/
def valueStateAt
    (constants : Constants) (initial : Poseidon2Duplex.State) :
    Nat → Poseidon2Duplex.State
  | 0 => initial
  | coordinate + 1 =>
      (ProductionSchedule.source
        (PiRlcCanonicalMachine.machine constants)
        (valueStateAt constants initial coordinate) coordinate).nextState

/-- Thread the exact source successor across `count` scalar coordinates. -/
def stateAt
    (base : Nat) (initial : SymbolicDuplex.Builder) :
    Nat → SymbolicDuplex.Builder
  | 0 => initial
  | coordinate + 1 =>
      scalarBuilder base (stateAt base initial coordinate) coordinate

theorem stateAt_extends
    (base : Nat) (initial : SymbolicDuplex.Builder) :
    ∀ coordinate, Extends initial (stateAt base initial coordinate)
  | 0 => Extends.refl initial
  | coordinate + 1 =>
      (stateAt_extends base initial coordinate).trans
        (scalarBuilder_extends base
          (stateAt base initial coordinate) coordinate)

/-- Exact permutation count for a fixed-size scalar batch beginning at a
freshly gated cursor. -/
theorem stateAt_entries_length_of_zero
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (cursorZero : initial.absorbed = 0) :
    ∀ coordinate,
      (stateAt base initial coordinate).entries.length =
        initial.entries.length + coordinate * 5
  | 0 => rfl
  | coordinate + 1 => by
      rw [stateAt,
        scalarBuilder_entries_length_of_zero base
          (stateAt base initial coordinate) coordinate
          (by
            cases coordinate <;>
              simp only [stateAt, cursorZero, scalarBuilder_absorbed]),
        stateAt_entries_length_of_zero base initial cursorZero coordinate]
      omega

/-- Exact permutation count for a fixed-size scalar batch beginning at the
selected post-PiCCS cursor one.  The first scalar consumes five calls and
leaves cursor zero; every later scalar uses the existing cursor-zero law. -/
theorem stateAt_entries_length_of_one
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (cursorOne : initial.absorbed = 1) :
    ∀ coordinate,
      (stateAt base initial coordinate).entries.length =
        initial.entries.length + coordinate * 5
  | 0 => rfl
  | coordinate + 1 => by
      rw [stateAt]
      by_cases first : coordinate = 0
      · subst coordinate
        change
          (scalarBuilder base initial 0).entries.length =
            initial.entries.length + (0 + 1) * 5
        rw [scalarBuilder_entries_length_of_one base initial 0 cursorOne]
      · have stateCursorZero :
          (stateAt base initial coordinate).absorbed = 0 := by
            cases coordinate with
            | zero => exact False.elim (first rfl)
            | succ previous =>
                simp only [stateAt, scalarBuilder_absorbed]
        rw [scalarBuilder_entries_length_of_zero base
            (stateAt base initial coordinate) coordinate stateCursorZero,
          stateAt_entries_length_of_one base initial cursorOne coordinate]
        omega

/-- The fixed-active fifteen-scalar suffix contributes exactly 75 canonical
Poseidon2 permutations once the preceding replay hands off a cursor-zero
state. -/
theorem fixedActive_entries_length_of_zero
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (cursorZero : initial.absorbed = 0) :
    (stateAt base initial 15).entries.length =
      initial.entries.length + 75 := by
  simpa using stateAt_entries_length_of_zero base initial cursorZero 15

/-- The selected cursor-one handoff contributes the same exact 75
permutations as the standalone cursor-zero schedule. -/
theorem fixedActive_entries_length_of_one
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (cursorOne : initial.absorbed = 1) :
    (stateAt base initial 15).entries.length =
      initial.entries.length + 75 := by
  simpa using stateAt_entries_length_of_one base initial cursorOne 15

/-- The fifteen-scalar duplex suffix therefore contributes exactly 26,400
Poseidon2 rows beyond the rows already present in the incoming builder. -/
theorem fixedActive_rows_length_of_zero
    (base : Nat) (constants : Constants)
    (initial : SymbolicDuplex.Builder)
    (cursorZero : initial.absorbed = 0) :
    (SymbolicDuplex.rows base constants
        (stateAt base initial 15)).length =
      initial.entries.length * 352 + 26400 := by
  rw [SymbolicDuplex.rows_length,
    fixedActive_entries_length_of_zero base initial cursorZero]
  omega

/-- The symbolic batch schedule is the exact `stateAt` recurrence of the
canonical production specification. -/
theorem decoded_stateAt
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (initial : SymbolicDuplex.Builder) (coordinate : Nat)
    (valid :
      Valid base constants assignment (stateAt base initial coordinate)) :
    decodedBuilder assignment (stateAt base initial coordinate) =
      valueStateAt constants
        (decodedBuilder assignment initial) coordinate := by
  induction coordinate with
  | zero => rfl
  | succ coordinate inductionHypothesis =>
      have priorValid :
          Valid base constants assignment
            (stateAt base initial coordinate) :=
        valid.of_extends
          (scalarBuilder_extends base
            (stateAt base initial coordinate) coordinate)
      have priorEq :=
        inductionHypothesis priorValid
      have scalarEq :=
        decoded_scalarBuilder base constants assignment constantWire
          (stateAt base initial coordinate) coordinate valid
      calc
        decodedBuilder assignment
            (stateAt base initial (coordinate + 1)) =
          (ProductionSchedule.source
            (PiRlcCanonicalMachine.machine constants)
            (decodedBuilder assignment
              (stateAt base initial coordinate)) coordinate).nextState := by
                simpa only [stateAt] using scalarEq
        _ =
          (ProductionSchedule.source
            (PiRlcCanonicalMachine.machine constants)
            (valueStateAt constants
              (decodedBuilder assignment initial) coordinate)
            coordinate).nextState := by
              exact congrArg
                (fun state =>
                  (ProductionSchedule.source
                    (PiRlcCanonicalMachine.machine constants)
                    state coordinate).nextState)
                priorEq
        _ =
          valueStateAt constants
            (decodedBuilder assignment initial) (coordinate + 1) := rfl

/-- The selected fixed-active profile derives exactly fifteen scalar sources.
This theorem states the typed count; it does not claim that chunk selection
rows have already been emitted. -/
theorem fixedActive_challengeCount :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total = 15 :=
  Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity_total

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine
