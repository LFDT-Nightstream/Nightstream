import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.PublicPaddingRows

/-!
Exact public-padding row certificate for the bounded fixed-point carrier.

Owns: the 13 generated sparse rows, their exact interval, coefficient shape,
and unique physical-row ownership.

Does not own: row semantics, constant-one authority, source-field decoding,
CCS/CE membership, commitment alignment, or row removal.

Emits constraints: no.

| Stage path | Exact equation | Authority class | Rust owner | Lean owner | Multiplicity |
|---|---|---|---|---|---:|
| `f_prime.fixed_point.public_padding` | `-(z[0] * z[257+i]) = 0` | checked coefficients | selective final emitter / fixed-point artifact generator | `generated_rows_exact`, `generated_row_has_unique_offset` | 13 |

All 13 physical rows are retained in this artifact; this leaf grants no
permission to eliminate them.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def relationRows : Nat := 14944219
def relationColumns : Nat := 11437038
def firstEmittedRow : Nat := 4729580
def emitterRunIndex : Nat := 6
def constantColumn : Nat := 0
def firstPaddingColumn : Nat := 257
def paddingWidth : Nat := 13

def rawRows : List RawRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.PublicPaddingRows.rawRows

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
    family := .publicPadding
    arm := none
    ports := expectedPorts (firstPaddingColumn + offset) }

def expectedRows : List RawRow :=
  (List.range paddingWidth).map expectedRow

/-- The certificate compares exactly 13 proof-free `RawRow` records. -/
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
  have offsetBound := offset.isLt
  simp only [paddingWidth, firstPaddingColumn, relationColumns] at offsetBound ⊢
  omega

theorem expectedRow_emittedRow_bound (offset : Fin paddingWidth) :
    firstEmittedRow + offset.val < relationRows := by
  have offsetBound := offset.isLt
  simp only [paddingWidth, firstEmittedRow, relationRows] at offsetBound ⊢
  omega

/-- Every generated record has one unique offset in the exact 13-row
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

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding
