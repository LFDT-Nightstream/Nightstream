import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.PrivatePaddingRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows

/-!
Exact private-alignment padding rows for the bounded fixed-point profile.

Owns: equality between the 38 Rust-projected proof-free rows and the complete
expected thirteen-port coefficient schedule, including physical-row order.

Does not own: decoded row semantics, constant-one authority, private values,
CCS/CE membership, commitment alignment, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.private_padding.rows` | 38 exact emitted rows | checked |
| `f_prime.fixed_point.private_padding.coefficients` | ports 1/4 contain the two unit terms | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def relationRows : Nat :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows.relationRows
def relationColumns : Nat :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows.relationColumns
def firstEmittedRow : Nat := 4729593
def emitterRunIndex : Nat := 7
def constantColumn : Nat := 0
def firstPaddingColumn : Nat := 273
def paddingWidth : Nat := 38

def rawRows : List RawRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.PrivatePaddingRows.rawRows

def emptyPort : RawPort := { explicit := [], geometric := [] }

def unitPort (column : Nat) : RawPort :=
  { explicit := [{ column, coefficient := 1 }], geometric := [] }

def expectedPort (paddingColumn : Nat) (port : Fin 13) : RawPort :=
  if port.val = 1 then unitPort constantColumn
  else if port.val = 4 then unitPort paddingColumn
  else emptyPort

def expectedPorts (paddingColumn : Nat) : List RawPort :=
  List.ofFn (expectedPort paddingColumn)

def expectedRow (offset : Nat) : RawRow :=
  { schemaVersion := 1
    rows := relationRows
    columns := relationColumns
    emittedRow := firstEmittedRow + offset
    runIndex := emitterRunIndex
    family := .privatePadding
    arm := none
    ports := expectedPorts (firstPaddingColumn + offset) }

def expectedRows : List RawRow :=
  (List.range paddingWidth).map expectedRow

/-- The certificate compares exactly 38 proof-free `RawRow` records. -/
theorem generated_rows_exact : rawRows = expectedRows := by
  native_decide

theorem generated_row_count : rawRows.length = paddingWidth := by
  rw [generated_rows_exact]
  simp [expectedRows]

theorem expectedRow_dimensions (offset : Nat) :
    (expectedRow offset).rows = relationRows ∧
      (expectedRow offset).columns = relationColumns := by
  exact ⟨rfl, rfl⟩

theorem expectedRow_emittedRow (offset : Nat) :
    (expectedRow offset).emittedRow = firstEmittedRow + offset := by
  rfl

theorem expectedRow_paddingColumn (offset : Fin paddingWidth) :
    firstPaddingColumn + offset.val < relationColumns := by
  have intervalBound : firstPaddingColumn + paddingWidth ≤ relationColumns := by
    decide
  have offsetBound := offset.isLt
  omega

theorem expectedRow_emittedRow_bound (offset : Fin paddingWidth) :
    firstEmittedRow + offset.val < relationRows := by
  have intervalBound : firstEmittedRow + paddingWidth ≤ relationRows := by
    decide
  have offsetBound := offset.isLt
  omega

/-- Every generated record has one unique offset in the exact 38-row
physical interval. -/
theorem generated_row_has_unique_offset {row : RawRow}
    (member : row ∈ rawRows) :
    ∃ offset : Fin paddingWidth,
      row = expectedRow offset.val ∧
        ∀ other : Fin paddingWidth,
          row = expectedRow other.val → other = offset := by
  rw [generated_rows_exact, expectedRows, List.mem_map] at member
  rcases member with ⟨value, valueMem, rfl⟩
  have valueLt : value < paddingWidth := by
    simpa using valueMem
  refine ⟨⟨value, valueLt⟩, rfl, ?_⟩
  intro other equal
  apply Fin.ext
  have emitted := congrArg RawRow.emittedRow equal
  simp only [expectedRow_emittedRow] at emitted
  exact (Nat.add_left_cancel emitted).symm

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
