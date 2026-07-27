import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants

/-!
Contract: a compact kernel certificate that no coefficient in the 22-round
Poseidon2 partial-state recurrence vanishes over Goldilocks.

Owns: a fixed 8-by-30 coefficient table, the exact matrix recurrence, a bounded
Boolean certificate, and extraction of any active nonzero coefficient.

Does not own: sparse-column placement or the theorem relating this coefficient
table to `partialState`; that semantic bridge is separate.  Keeping the
certificate coefficient-only avoids reducing the much larger proof-carrying
row program.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices

def supportWidth : Nat := width + partialRounds
abbrev Table := List (List Nat)

def tableValue (table : Table) (lane : Fin width) (label : Nat) : Nat :=
  (table.getD lane.val []).getD label 0

def tableOf (value : Fin width → Fin supportWidth → Nat) : Table :=
  List.ofFn (fun lane => List.ofFn (value lane))

theorem tableValue_tableOf
    (value : Fin width → Fin supportWidth → Nat)
    (lane : Fin width) (label : Fin supportWidth) :
    tableValue (tableOf value) lane label.val = value lane label := by
  simp [tableValue, tableOf]

def initialTable : Table :=
  tableOf fun lane label =>
    if below : label.val < width then
      externalMatrix lane ⟨label.val, below⟩
    else 0

/-- One partial round: lane zero is replaced by the fresh S-box output labelled
`width + round`; lanes one through seven carry the previous coefficient table
through the selected internal matrix. -/
def nextTable (round : Nat) (previous : Table) : Table :=
  tableOf fun target label =>
    applyMatrixValues internalMatrix
      (fun source =>
        if source.val = 0 then
          if label.val = width + round then 1 else 0
        else tableValue previous source label.val)
      target

def activeNonzero (round : Nat) (table : Table) : Bool :=
  (List.finRange width).all fun lane =>
    (List.finRange (width + round)).all fun label =>
      decide (tableValue table lane label.val ≠ 0)

/-- Advance a materialized table without rebuilding earlier layers. -/
def advance : Nat → Nat → Table → Table
  | 0, _, table => table
  | steps + 1, round, table =>
      advance steps (round + 1) (nextTable round table)

def tableAt (round : Nat) : Table := advance round 0 initialTable

theorem tableAt_zero : tableAt 0 = initialTable := rfl

theorem advance_succ (steps round : Nat) (table : Table) :
    advance (steps + 1) round table =
      nextTable (round + steps) (advance steps round table) := by
  induction steps generalizing round table with
  | zero => simp [advance]
  | succ previous hypothesis =>
      change
        advance (previous + 1) (round + 1) (nextTable round table) =
          nextTable (round + (previous + 1))
            (advance previous (round + 1) (nextTable round table))
      rw [hypothesis]
      congr 1
      omega

theorem tableAt_succ (round : Nat) :
    tableAt (round + 1) = nextTable round (tableAt round) := by
  unfold tableAt
  rw [advance_succ]
  simp

theorem tableValue_initial
    (lane : Fin width) (label : Fin supportWidth) :
    tableValue initialTable lane label.val =
      if below : label.val < width then
        externalMatrix lane ⟨label.val, below⟩
      else 0 := by
  exact tableValue_tableOf _ lane label

theorem tableValue_next
    (round : Nat) (previous : Table)
    (lane : Fin width) (label : Fin supportWidth) :
    tableValue (nextTable round previous) lane label.val =
      applyMatrixValues internalMatrix
        (fun source =>
          if source.val = 0 then
            if label.val = width + round then 1 else 0
          else tableValue previous source label.val)
        lane := by
  exact tableValue_tableOf _ lane label

theorem applyMatrixValues_zero
    (matrix : Fin width → Fin width → Nat) (lane : Fin width) :
    applyMatrixValues matrix (fun _ => 0) lane = 0 := by
  unfold applyMatrixValues
  have zeroSum :
      ∀ items : List (Fin width),
        (items.map (fun _ => 0)).sum = 0 := by
    intro items
    induction items with
    | nil => rfl
    | cons head tail hypothesis =>
        simp only [List.map_cons, List.sum_cons]
        simpa using hypothesis
  simp only [Nat.mul_zero]
  rw [zeroSum]
  exact Nat.zero_mod _

/-- Labels outside the active `8 + round` prefix remain exactly zero.  This is
the companion invariant to `activeNonzero`: the next partial-round output is
not present before the round that introduces it. -/
theorem tableAt_inactive_zero
    (round : Nat) (lane : Fin width) (label : Nat)
    (inactive : width + round ≤ label) (labelBound : label < supportWidth) :
    tableValue (tableAt round) lane label = 0 := by
  induction round generalizing lane with
  | zero =>
      let supportLabel : Fin supportWidth := ⟨label, labelBound⟩
      rw [tableAt_zero, show label = supportLabel.val from rfl,
        tableValue_initial lane supportLabel]
      change (if below : label < width then
        externalMatrix lane ⟨label, below⟩ else 0) = 0
      rw [dif_neg (by omega)]
  | succ previous hypothesis =>
      let supportLabel : Fin supportWidth := ⟨label, labelBound⟩
      rw [tableAt_succ, show label = supportLabel.val from rfl,
        tableValue_next previous (tableAt previous) lane supportLabel]
      have sources :
          (fun source : Fin width =>
            if source.val = 0 then
              if supportLabel.val = width + previous then 1 else 0
            else tableValue (tableAt previous) source supportLabel.val) =
          (fun _ => 0) := by
        funext source
        change
          (if source.val = 0 then
            if label = width + previous then 1 else 0
          else tableValue (tableAt previous) source label) = 0
        by_cases sourceZero : source.val = 0
        · simp [sourceZero]
          omega
        · simp [sourceZero]
          exact hypothesis source (by omega)
      rw [sources, applyMatrixValues_zero]

def verify : Nat → Nat → Table → Bool
  | round, 0, table => activeNonzero round table
  | round, remaining + 1, table =>
      activeNonzero round table &&
        verify (round + 1) remaining (nextTable round table)

/-- Exact dimensions of the compact certificate carrier. -/
theorem table_shapes :
    initialTable.length = width ∧
      (∀ row ∈ initialTable, row.length = supportWidth) := by
  decide

set_option maxRecDepth 100000 in
/-- The entire 23-state recurrence (the initial state and 22 successors) has
no zero in any active coefficient position.  This is the only closed finite
computation; its carrier is 8 by 30, not the 2,464-row program. -/
theorem selected_partial_coefficients_check :
    verify 0 partialRounds initialTable = true := by
  decide

theorem verify_current
    (round remaining : Nat) (table : Table)
    (checked : verify round remaining table = true) :
    activeNonzero round table = true := by
  cases remaining with
  | zero => exact checked
  | succ remaining =>
      exact (Bool.and_eq_true_iff.mp (by simpa [verify] using checked)).1

theorem verify_successor
    (round remaining : Nat) (table : Table)
    (checked : verify round (remaining + 1) table = true) :
    verify (round + 1) remaining (nextTable round table) = true :=
  (Bool.and_eq_true_iff.mp (by simpa [verify] using checked)).2

theorem verify_at
    (round remaining offset : Nat) (table : Table)
    (offsetBound : offset ≤ remaining)
    (checked : verify round remaining table = true) :
    activeNonzero (round + offset) (advance offset round table) = true := by
  induction offset generalizing round remaining table with
  | zero =>
      simpa [advance] using verify_current round remaining table checked
  | succ previous hypothesis =>
      cases remaining with
      | zero => omega
      | succ remaining =>
          have previousBound : previous ≤ remaining := by omega
          have nextChecked :=
            verify_successor round remaining table (by simpa using checked)
          have result := hypothesis (round + 1) remaining
            (nextTable round table) previousBound nextChecked
          simpa [advance, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using result

private theorem activeNonzero_entry
    (round : Nat) (table : Table)
    (checked : activeNonzero round table = true)
    (lane : Fin width) (label : Nat) (labelBound : label < width + round) :
    tableValue table lane label ≠ 0 := by
  have laneChecked :=
    (List.all_eq_true.mp checked) lane (List.mem_finRange lane)
  let labelIndex : Fin (width + round) := ⟨label, labelBound⟩
  have valueChecked :=
    (List.all_eq_true.mp laneChecked) labelIndex (List.mem_finRange labelIndex)
  exact of_decide_eq_true valueChecked

/-- Extract any active coefficient from the bounded certificate. -/
theorem tableAt_nonzero
    (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (label : Nat) (labelBound : label < width + round) :
    tableValue (tableAt round) lane label ≠ 0 := by
  have checked := verify_at 0 partialRounds round initialTable
    roundBound selected_partial_coefficients_check
  rw [Nat.zero_add] at checked
  change activeNonzero round (tableAt round) = true at checked
  exact activeNonzero_entry round (tableAt round) checked lane label labelBound

end Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate
