import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedPublicInput
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout

/-!
Exact logical-column map required by an aligned F' R1CS compiler.

Owns: the old-to-aligned scalar index function; its partial inverse; the
thirteen-coordinate public padding hole; injectivity and bounds; and the
resulting 54-lane block/coefficient coordinates.

Does not own: Rust sparse-matrix storage, assignment serialization, Ajtai
matrix values, commitment equality, generated rows, or permission to change
the production compiler.

Emits constraints: no.

Authority boundary: old verifier-owned columns are mapped injectively. The
thirteen new positions have no old preimage and are reserved for fixed zero
coefficients; a prover-supplied value cannot be interpreted as an old column.

| Protocol | Phase | Constraint family | Mathematical obligation | Result |
|---|---|---|---|---|
| F' / CCS | compiler | logical public columns | columns `0..256` retain their indices | `alignedIndex_public` |
| F' / CCS | compiler | private columns | columns from `257` shift right by 13 | `alignedIndex_private` |
| F' / CCS | compiler | column connectivity | every old column has one distinct in-range image | `alignedIndex_injective`, `alignedIndex_lt` |
| F' / CCS | compiler | fixed padding | exactly columns `257..269` have no old preimage | `unalignIndex?_eq_none_iff` |
| F' / CCS | compiler | inverse connectivity | every decoded aligned column returns to that same column | `alignedIndex_of_unalignIndex?_eq_some` |
| SuperNeo | coefficient packing | block/lane placement | quotient/remainder flatten to the aligned scalar index | `packedFlatIndex` |
| SuperNeo | setup shape | Ajtai ring columns | the exact fixed carrier uses five blocks before and after repair | `fixedCarrier_blockCounts` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput

/-- Old scalar column to aligned scalar column. Public columns stay fixed;
the old private suffix moves past the thirteen verifier-fixed zeros. -/
def alignedIndex (column : Nat) : Nat :=
  if column < logicalPublicWidth then column else column + paddingWidth

/-- Partial inverse of `alignedIndex`. `none` denotes a verifier-fixed padding
coordinate rather than a prover-owned old column. -/
def unalignIndex? (column : Nat) : Option Nat :=
  if column < logicalPublicWidth then
    some column
  else if column < alignedPublicWidth then
    none
  else
    some (column - paddingWidth)

theorem alignedIndex_public (column : Nat)
    (isPublic : column < logicalPublicWidth) :
    alignedIndex column = column := by
  simp [alignedIndex, isPublic]

theorem alignedIndex_private (column : Nat)
    (isPrivate : logicalPublicWidth ≤ column) :
    alignedIndex column = column + paddingWidth := by
  simp [alignedIndex, Nat.not_lt.mpr isPrivate]

theorem alignedIndex_injective : Function.Injective alignedIndex := by
  intro left right equal
  unfold alignedIndex at equal
  split at equal <;> split at equal <;>
    simp_all [logicalPublicWidth, paddingWidth] <;> omega

theorem alignedIndex_lt {columns : Nat} (column : Fin columns) :
    alignedIndex column.val < columns + paddingWidth := by
  unfold alignedIndex
  split <;> simp_all [paddingWidth] <;> omega

/-- Finite-column form consumed by matrix and assignment refinements. -/
def alignColumn {columns : Nat} (column : Fin columns) :
    Fin (columns + paddingWidth) :=
  ⟨alignedIndex column.val, alignedIndex_lt column⟩

@[simp] theorem alignColumn_val {columns : Nat} (column : Fin columns) :
    (alignColumn column).val = alignedIndex column.val := rfl

theorem alignColumn_injective {columns : Nat} :
    Function.Injective (@alignColumn columns) := by
  intro left right equal
  apply Fin.ext
  apply alignedIndex_injective
  exact congrArg Fin.val equal

/-- Every old column round-trips through the partial inverse. -/
theorem unalignIndex?_alignedIndex (column : Nat) :
    unalignIndex? (alignedIndex column) = some column := by
  by_cases isPublic : column < logicalPublicWidth
  · simp [alignedIndex, unalignIndex?, isPublic]
  · have isPrivate : logicalPublicWidth ≤ column := Nat.not_lt.mp isPublic
    have beyondPadding : alignedPublicWidth ≤ column + paddingWidth := by
      simp [logicalPublicWidth, alignedPublicWidth, paddingWidth] at isPrivate ⊢
      omega
    have oldNotPublic : ¬ column < 257 := by
      simpa [logicalPublicWidth] using isPublic
    have newNotPublic : ¬ column + 13 < 257 := by omega
    have newNotPadding : ¬ column + 13 < 270 := by
      simpa [alignedPublicWidth, paddingWidth] using
        Nat.not_lt.mpr beyondPadding
    simp [alignedIndex, unalignIndex?, logicalPublicWidth,
      alignedPublicWidth, paddingWidth, oldNotPublic, newNotPublic,
      newNotPadding]

/-- The only aligned scalar positions without an old-column owner are the
thirteen public padding coordinates. -/
theorem unalignIndex?_eq_none_iff (column : Nat) :
    unalignIndex? column = none ↔
      logicalPublicWidth ≤ column ∧ column < alignedPublicWidth := by
  unfold unalignIndex?
  split
  next isPublic =>
    simp [logicalPublicWidth, alignedPublicWidth] at isPublic ⊢
    omega
  next isNotPublic =>
    split
    next isPadding =>
      simp [logicalPublicWidth, alignedPublicWidth] at isNotPublic isPadding ⊢
      constructor <;> omega
    next isNotPadding =>
      simp [logicalPublicWidth, alignedPublicWidth] at isNotPublic isNotPadding ⊢
      omega

/-- Every aligned coordinate that decodes to an old column is exactly that
old column's aligned image. -/
theorem alignedIndex_of_unalignIndex?_eq_some {column oldColumn : Nat}
    (decoded : unalignIndex? column = some oldColumn) :
    alignedIndex oldColumn = column := by
  unfold unalignIndex? at decoded
  split at decoded
  next isPublic =>
    have oldEqual : oldColumn = column := (Option.some.inj decoded).symm
    subst oldColumn
    exact alignedIndex_public column isPublic
  next isNotPublic =>
    split at decoded
    next isPadding => contradiction
    next isNotPadding =>
      have oldEqual : oldColumn = column - paddingWidth :=
        (Option.some.inj decoded).symm
      subst oldColumn
      have oldPrivate : logicalPublicWidth ≤ column - paddingWidth := by
        simp [logicalPublicWidth, alignedPublicWidth, paddingWidth]
          at isNotPadding ⊢
        omega
      rw [alignedIndex_private _ oldPrivate]
      simp [paddingWidth, alignedPublicWidth] at isNotPadding ⊢
      omega

/-- Decoding an in-range aligned column cannot produce an old column beyond
the old compiler width. -/
theorem unalignedIndex_lt {columns column oldColumn : Nat}
    (hasPublic : logicalPublicWidth ≤ columns)
    (columnLt : column < columns + paddingWidth)
    (decoded : unalignIndex? column = some oldColumn) :
    oldColumn < columns := by
  have mapped := alignedIndex_of_unalignIndex?_eq_some decoded
  unfold alignedIndex at mapped
  split at mapped <;> simp [paddingWidth] at mapped columnLt ⊢ <;> omega

/-- Block coordinate after alignment. -/
def packedBlock (column : Nat) : Nat :=
  alignedIndex column / ringDegree

/-- Coefficient coordinate after alignment. -/
def packedLane (column : Nat) : Nat :=
  alignedIndex column % ringDegree

/-- The block/lane pair is an exact decomposition of the aligned scalar
column; there is no hidden packing permutation. -/
theorem packedFlatIndex (column : Nat) :
    packedBlock column * ringDegree + packedLane column =
      alignedIndex column := by
  simpa [packedBlock, packedLane, Nat.mul_comm] using
    Nat.div_add_mod (alignedIndex column) ringDegree

/-- Exact transition at the logical-public/private boundary. This pins the
off-by-thirteen and ring-boundary behavior independently of Rust. -/
theorem boundary_coordinates :
    alignedIndex 256 = 256 ∧
      alignedIndex 257 = 270 ∧
      packedBlock 256 = 4 ∧
      packedLane 256 = 40 ∧
      packedBlock 257 = 5 ∧
      packedLane 257 = 0 := by
  decide

/-- The exact current 257-scalar carrier and repaired 270-scalar carrier both
occupy five 54-coefficient ring columns. This is a shape fact only: their
coefficient ownership differs, so it does not imply commitment equality. -/
theorem fixedCarrier_blockCounts :
    Phi81ColumnLayout.blockCount logicalPublicWidth = 5 ∧
      Phi81ColumnLayout.blockCount alignedPublicWidth = 5 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap
