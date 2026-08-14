import Mathlib.Data.Nat.Digits.Lemmas
import Nightstream.Protocol.Nebula.Snapshot

/-!
Contract: exact integer relation for the two V2 snapshot timestamp checks.

Assurance tier: implementation model.

Owns a fixed 23-bit little-endian word, the non-wrapping slack relation used
to prove `left <= right`, and one exact relation instance for every initial
and final snapshot cell.

The soundness theorem derives both segment-relative timestamp inequalities
from bit bounds, decoding equations, and `left + slack = right`. The desired
inequalities are not fields of the witness.

Does not own concrete R1CS column numbers, generated row inclusion, Rust
serialization, or the final V2 row census. Those are separate refinement
obligations over this relation.

Emits constraints: no. It defines the relation that generated constraints
must refine.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.SnapshotRows

open Nightstream.Protocol.Nebula

/-- Exactly one little-endian 23-bit unsigned integer. -/
structure Word where
  digits : List Nat
  lengthExact : digits.length = timestampBits
  binary : ∀ digit ∈ digits, digit < 2
deriving Repr

namespace Word

def decode (word : Word) : Nat :=
  Nat.ofDigits 2 word.digits

theorem decode_lt_limit (word : Word) : word.decode < timestampLimit := by
  have bounded :=
    Nat.ofDigits_lt_base_pow_length (b := 2) (by decide) word.binary
  simpa [decode, timestampLimit, word.lengthExact] using bounded

/-- Canonical fixed-width word used by the honest witness generator. -/
def ofNat (value : Nat) (bounded : value < timestampLimit) : Word where
  digits := Nat.digitsAppend 2 timestampBits value
  lengthExact := by
    exact Nat.length_digitsAppend (by decide) timestampBits
      (by simpa [timestampLimit] using bounded)
  binary := by
    intro digit member
    exact Nat.lt_of_mem_digitsAppend (by decide) timestampBits digit member

theorem decode_ofNat (value : Nat) (bounded : value < timestampLimit) :
    (ofNat value bounded).decode = value := by
  change
    Nat.ofDigits 2
      (Nat.digits 2 value ++
        List.replicate (timestampBits - (Nat.digits 2 value).length) 0) =
      value
  rw [Nat.ofDigits_append_replicate_zero, Nat.ofDigits_digits]

end Word

/-- Exact arithmetic witness for one 23-bit `left <= right` comparison.
Generated rows must range-check the three words, decode them, and enforce the
single slack equation. -/
structure LeqWitness (left right : Nat) where
  leftWord : Word
  rightWord : Word
  slackWord : Word
  leftDecoded : leftWord.decode = left
  rightDecoded : rightWord.decode = right
  slackEquation : left + slackWord.decode = right

namespace LeqWitness

theorem sound {left right : Nat} (witness : LeqWitness left right) :
    left ≤ right := by
  have equation := witness.slackEquation
  omega

theorem left_bounded {left right : Nat} (witness : LeqWitness left right) :
    left < timestampLimit := by
  rw [← witness.leftDecoded]
  exact witness.leftWord.decode_lt_limit

theorem right_bounded {left right : Nat} (witness : LeqWitness left right) :
    right < timestampLimit := by
  rw [← witness.rightDecoded]
  exact witness.rightWord.decode_lt_limit

/-- Honest construction. The slack is an ordinary natural subtraction, so
the equation cannot wrap through the Goldilocks field. -/
def ofBounded
    {left right : Nat}
    (leftBounded : left < timestampLimit)
    (rightBounded : right < timestampLimit)
    (ordered : left ≤ right) : LeqWitness left right where
  leftWord := Word.ofNat left leftBounded
  rightWord := Word.ofNat right rightBounded
  slackWord := Word.ofNat (right - left) (by omega)
  leftDecoded := Word.decode_ofNat left leftBounded
  rightDecoded := Word.decode_ofNat right rightBounded
  slackEquation := by
    rw [Word.decode_ofNat]
    omega

end LeqWitness

/-- One exact checked scan row at a structural memory index. Values are
range-checked here; addresses are structural and cannot be supplied by the
prover. -/
structure Row
    (initial final : CellState)
    (segmentStart segmentEnd : Nat) where
  initialValueBounded : initial.value < valueLimit
  finalValueBounded : final.value < valueLimit
  initialTimestamp : LeqWitness initial.lastTimestamp segmentStart
  finalTimestamp : LeqWitness final.lastTimestamp segmentEnd

/-- The complete scan relation has one row for every structural index. -/
def Accepts
    (initial final : Snapshot)
    (segmentStart segmentEnd : Nat) : Prop :=
  ∀ index, Nonempty (Row (initial index) (final index) segmentStart segmentEnd)

theorem accepts_sound
    {initial final : Snapshot}
    {segmentStart segmentEnd : Nat}
    (accepts : Accepts initial final segmentStart segmentEnd) :
    initial.ValidAt segmentStart ∧ final.ValidAt segmentEnd := by
  constructor
  · intro index
    rcases accepts index with ⟨row⟩
    exact
      ⟨row.initialValueBounded,
        row.initialTimestamp.sound⟩
  · intro index
    rcases accepts index with ⟨row⟩
    exact
      ⟨row.finalValueBounded,
        row.finalTimestamp.sound⟩

theorem accepts_complete
    {initial final : Snapshot}
    {segmentStart segmentEnd : Nat}
    (startBounded : segmentStart < timestampLimit)
    (endBounded : segmentEnd < timestampLimit)
    (initialValid : initial.ValidAt segmentStart)
    (finalValid : final.ValidAt segmentEnd) :
    Accepts initial final segmentStart segmentEnd := by
  intro index
  exact ⟨
    { initialValueBounded := (initialValid index).1
      finalValueBounded := (finalValid index).1
      initialTimestamp :=
        LeqWitness.ofBounded
          (lt_of_le_of_lt (initialValid index).2 startBounded)
          startBounded
          (initialValid index).2
      finalTimestamp :=
        LeqWitness.ofBounded
          (lt_of_le_of_lt (finalValid index).2 endBounded)
          endBounded
          (finalValid index).2 }⟩

end Nightstream.Implementation.Nebula.SnapshotRows
