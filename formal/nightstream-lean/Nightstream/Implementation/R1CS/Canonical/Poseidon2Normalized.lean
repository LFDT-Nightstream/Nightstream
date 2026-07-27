import Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest

/-!
Contract: the program the encoding actually emits.

Owns: the normalizing row transformation, the emitted program built with it,
and the transfer of satisfaction, cost, soundness and honest completeness onto
that program.

Does not own: the round structure, the schedule, or the reference.

## The defect this repairs

`Poseidon2Coefficients.rowTermCount` applies `normalize` *inside the metric*,
while `canonicalProgram` emits raw `flatMap` combinations.  So
`canonicalProgram_termCount` costed a representation that was never
constructed — a count not derived from an emitted row program.  Row and column
counts were never affected, since those do not depend on operand shape; only
the coefficient figure was attached to a term that did not exist.

The repair is to emit the normalized program and measure it with a metric that
does no normalizing of its own.  `rawTermCount` counts entries as they are;
`normalizedCanonicalProgram` is what carries them.

Everything already proved about `canonicalProgram` transfers, because
`normalizeRow` changes no row's meaning: `LinCombNormal.lcEval_fieldNormalize`
gives operandwise equality of value, so `RowHolds` is preserved in both
directions and `Satisfies` follows.  Nothing is reproved from scratch and no
statement is weakened.

The emitted rows carry the **field-canonical** form: every coefficient is
reduced modulo the prime and every zero is dropped.  So the emitted entry count
is the number of nonzero field coefficients, and it is at most — not equal to —
the merge-only decomposition, the gap being exactly the coefficients that
cancel.  Closing that gap is `POSEIDON2-NO-CANCELLATION`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## The emitted form -/

def normalizeRow (row : Row) : Row where
  a := fieldNormalize row.a
  b := fieldNormalize row.b
  c := fieldNormalize row.c

def normalizeProgram (rows : List Row) : List Row := rows.map normalizeRow

def normalizedCanonicalProgram (layout : Layout) (constants : Constants) :
    List Row :=
  normalizeProgram (canonicalProgram layout constants)

/-- The emitted permutation program when the eight initial lanes are carried
as arbitrary sparse linear combinations.  This is the form used between
successive sponge calls. -/
def normalizedCanonicalProgramFrom
    (layout : Layout) (entry : State) (constants : Constants) : List Row :=
  normalizeProgram (canonicalProgramFrom layout entry constants)

/-! ## Meaning is preserved -/

/-- **Normalizing a row changes nothing about when it holds.** -/
theorem rowHolds_normalizeRow (z : Nat → Nat) (row : Row) :
    RowHolds z (normalizeRow row) ↔ RowHolds z row := by
  simp only [RowHolds, normalizeRow, lcEval_fieldNormalize]

theorem satisfies_normalizeProgram (rows : List Row) (z : Nat → Nat) :
    Satisfies (normalizeProgram rows) z ↔ Satisfies rows z := by
  constructor
  · intro holds row member
    exact (rowHolds_normalizeRow z row).1
      (holds _ (List.mem_map.2 ⟨row, member, rfl⟩))
  · intro holds row member
    rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
    exact (rowHolds_normalizeRow z source).2 (holds source sourceMember)

/-- **Normalizing a row introduces no column**, so conservation carries over
operandwise.

Only one direction holds, and that is deliberate: `fieldNormalize` DROPS a
column whose coefficient vanishes modulo the prime, so the emitted row can
reference strictly fewer columns than the raw one.  Conservation needs exactly
this direction — no row touches a column outside its allocation — so nothing is
lost.  Support *equality* would be false. -/
theorem mentions_normalizeRow (row : Row) (column : Nat) :
    (Mentions (normalizeRow row).a column → Mentions row.a column)
      ∧ (Mentions (normalizeRow row).b column → Mentions row.b column)
      ∧ (Mentions (normalizeRow row).c column → Mentions row.c column) :=
  ⟨fun m => (mentions_normalize _ _).1 (mentions_fieldNormalize_subset _ _ m),
   fun m => (mentions_normalize _ _).1 (mentions_fieldNormalize_subset _ _ m),
   fun m => (mentions_normalize _ _).1 (mentions_fieldNormalize_subset _ _ m)⟩

/-! ## Counting the emitted program

`rawTermCount` counts entries as they stand.  Applied to the normalized
program it agrees with `rowTermCount` on the raw one — but now the number
describes a term that exists. -/

def rawTermCount (row : Row) : Nat :=
  row.a.length + row.b.length + row.c.length

def rawProgramTermCount (rows : List Row) : Nat := (rows.map rawTermCount).sum

/-- The emitted row's entry count is at most the merge-only count, with the
difference being exactly the coefficients that vanish modulo the prime. -/
theorem rawTermCount_normalizeRow_le (row : Row) :
    rawTermCount (normalizeRow row) ≤ rowTermCount row := by
  simp only [rawTermCount, normalizeRow, rowTermCount]
  have a := fieldNormalize_length_le row.a
  have b := fieldNormalize_length_le row.b
  have c := fieldNormalize_length_le row.c
  omega

theorem rawProgramTermCount_normalizeProgram_le (rows : List Row) :
    rawProgramTermCount (normalizeProgram rows) ≤ programTermCount rows := by
  induction rows with
  | nil => simp [rawProgramTermCount, normalizeProgram, programTermCount]
  | cons head tail hypothesis =>
      simp only [rawProgramTermCount, normalizeProgram, programTermCount,
        List.map_cons, List.sum_cons] at hypothesis ⊢
      have step := rawTermCount_normalizeRow_le head
      omega

/-! ## Every emitted coefficient is a canonical nonzero residue

This is what a row-level comparison against Rust needs, and it is what makes
`rawProgramTermCount` of the emitted program the NONZERO coefficient count
rather than an upper bound on it. -/

theorem normalizeRow_entries
    (row : Row) (operand : List (Nat × Nat))
    (isOperand : operand = (normalizeRow row).a ∨ operand = (normalizeRow row).b
      ∨ operand = (normalizeRow row).c) :
    ∀ term ∈ operand, term.2 < goldilocksP ∧ term.2 ≠ 0 := by
  intro term member
  rcases isOperand with rfl | rfl | rfl <;>
    exact ⟨fieldNormalize_canonical _ term member,
      fieldNormalize_nonzero _ term member⟩

/-! ## Everything transfers -/

theorem normalizedCanonicalProgram_length
    (layout : Layout) (constants : Constants) :
    (normalizedCanonicalProgram layout constants).length = 352 := by
  unfold normalizedCanonicalProgram normalizeProgram
  rw [List.length_map]
  exact canonicalProgram_length layout constants

theorem normalizedCanonicalProgramFrom_length
    (layout : Layout) (entry : State) (constants : Constants) :
    (normalizedCanonicalProgramFrom layout entry constants).length = 352 := by
  unfold normalizedCanonicalProgramFrom normalizeProgram
  rw [List.length_map]
  exact canonicalProgramFrom_length layout entry constants

/-- **The coefficient total of the program that is emitted.**  Same figure as
`canonicalProgram_termCount`, now attached to a constructed term and measured
by a metric that does no normalizing of its own. -/
theorem normalizedCanonicalProgram_termCount_le
    (layout : Layout) (constants : Constants) :
    rawProgramTermCount (normalizedCanonicalProgram layout constants)
      ≤ 3 * (scheduledSizes layout constants).sum + 9 * sboxCount
        + ((finalSizes layout).sum + 2 * width) := by
  unfold normalizedCanonicalProgram
  rw [← canonicalProgram_termCount layout constants]
  exact rawProgramTermCount_normalizeProgram_le _

/-- **Soundness transfers.** -/
theorem normalizedCanonicalProgram_computes_reference
    (layout : Layout) (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (normalizedCanonicalProgram layout constants) z)
    (lane : Fin width) :
    z (layout.outputPort lane)
      = referencePermutation constants (inputValues layout z) lane :=
  canonicalProgram_computes_reference layout constants z residues constantWire
    ((satisfies_normalizeProgram _ z).1 satisfied) lane

/-- **Carried-entry soundness transfers to the emitted normalized form.** -/
theorem normalizedCanonicalProgramFrom_computes_reference
    (layout : Layout) (constants : Constants) (z : Nat → Nat)
    (entry : State) (entryValues : Values)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (entryAgrees : ∀ lane : Fin width, lcEval z (entry lane) = entryValues lane)
    (satisfied :
      Satisfies (normalizedCanonicalProgramFrom layout entry constants) z)
    (lane : Fin width) :
    z (layout.outputPort lane)
      = referencePermutation constants entryValues lane :=
  canonicalProgramFrom_computes_reference layout constants z entry entryValues
    residues constantWire entryAgrees
    ((satisfies_normalizeProgram _ z).1 satisfied) lane

/-- **Honest completeness transfers.** -/
theorem honest_satisfies_normalized
    (constants : Constants) (input : Values)
    (inputResidues : ∀ lane, input lane < goldilocksP) :
    Satisfies (normalizedCanonicalProgram canonicalLayout constants)
      (honestAssignment constants input) :=
  (satisfies_normalizeProgram _ _).2 (honest_satisfies constants input inputResidues)

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
