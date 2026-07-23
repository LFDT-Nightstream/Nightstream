import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.SelectorRows

/-!
Exact selector rows for the bounded fixed-point profile.

Owns: equality between the four Rust-projected proof-free rows and the exact
three selector-domain plus one selector-total coefficient schedule, including
unique physical ownership.

Does not own: decoded semantics, selector values, retained-row coverage,
branch semantics, CCS/CE membership, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.selector.domain` | three exact domain rows | checked |
| `f_prime.fixed_point.selector.total` | one exact sum row | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def relationRows : Nat := 14944219
def relationColumns : Nat := 11437038
def selectorCount : Nat := 3
def selectorStart : Nat := 270
def totalEmittedRow : Nat := 4729579
def negativeOneWord : Nat := 18446744069414584320

def rawRows : List RawRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.SelectorRows.rawRows

def emptyPort : RawPort := { explicit := [], geometric := [] }

def unitPort (column : Nat) : RawPort :=
  { explicit := [{ column, coefficient := 1 }], geometric := [] }

def selectorColumn (arm : Fin selectorCount) : Nat :=
  selectorStart + arm.val

def expectedSelectorPort (arm : Fin selectorCount) (port : Fin 13) : RawPort :=
  if port.val = 0 then unitPort (selectorColumn arm)
  else if port.val = 1 then unitPort 0
  else emptyPort

def expectedSelectorRow (arm : Fin selectorCount) : RawRow :=
  { schemaVersion := 1
    rows := relationRows
    columns := relationColumns
    emittedRow := arm.val
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := List.ofFn (expectedSelectorPort arm) }

def totalPort : Fin 13 → RawPort := fun port =>
  if port.val = 1 then unitPort 0
  else if port.val = 4 then
    { explicit :=
        [ { column := 0, coefficient := negativeOneWord }
        , { column := 270, coefficient := 1 }
        , { column := 271, coefficient := 1 }
        , { column := 272, coefficient := 1 }
        ]
      geometric := [] }
  else emptyPort

def expectedTotalRow : RawRow :=
  { schemaVersion := 1
    rows := relationRows
    columns := relationColumns
    emittedRow := totalEmittedRow
    runIndex := 5
    family := .oneHot
    arm := none
    ports := List.ofFn totalPort }

def expectedRow (index : Fin 4) : RawRow :=
  if selector : index.val < selectorCount then
    expectedSelectorRow ⟨index.val, selector⟩
  else
    expectedTotalRow

def expectedRows : List RawRow :=
  List.ofFn expectedRow

/-- The certificate compares exactly four proof-free `RawRow` records. -/
theorem generated_rows_exact : rawRows = expectedRows := by
  native_decide

theorem generated_row_count : rawRows.length = 4 := by
  rw [generated_rows_exact]
  simp [expectedRows]

theorem expectedRow_emittedRow (index : Fin 4) :
    (expectedRow index).emittedRow =
      if index.val < selectorCount then index.val else totalEmittedRow := by
  by_cases selector : index.val < selectorCount
  · simp [expectedRow, selector, expectedSelectorRow]
  · simp [expectedRow, selector, expectedTotalRow]

theorem expectedRow_injective : Function.Injective expectedRow := by
  intro left right equal
  have emitted := congrArg RawRow.emittedRow equal
  rw [expectedRow_emittedRow, expectedRow_emittedRow] at emitted
  have leftBound := left.isLt
  have rightBound := right.isLt
  by_cases leftSelector : left.val < selectorCount
  · by_cases rightSelector : right.val < selectorCount
    · rw [if_pos leftSelector, if_pos rightSelector] at emitted
      exact Fin.ext emitted
    · rw [if_pos leftSelector, if_neg rightSelector] at emitted
      simp only [selectorCount] at leftSelector rightSelector
      simp only [totalEmittedRow] at emitted
      omega
  · by_cases rightSelector : right.val < selectorCount
    · rw [if_neg leftSelector, if_pos rightSelector] at emitted
      simp only [selectorCount] at leftSelector rightSelector
      simp only [totalEmittedRow] at emitted
      omega
    · apply Fin.ext
      simp only [selectorCount] at leftSelector rightSelector
      omega

/-- Every generated row belongs to exactly one of the four equation owners. -/
theorem generated_row_has_unique_owner {row : RawRow} (member : row ∈ rawRows) :
    ∃ index : Fin 4,
      row = expectedRow index ∧
        ∀ other : Fin 4, row = expectedRow other → other = index := by
  rw [generated_rows_exact, expectedRows, List.mem_ofFn] at member
  rcases member with ⟨index, expectedEq⟩
  refine ⟨index, expectedEq.symm, ?_⟩
  intro other otherEq
  exact (expectedRow_injective (expectedEq.trans otherEq)).symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors
