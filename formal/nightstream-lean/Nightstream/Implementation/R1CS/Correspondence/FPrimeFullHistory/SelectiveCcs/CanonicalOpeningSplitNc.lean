import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.BorrowChunk
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane

/-!
Contract: connect optimized canonical-opening columns to the complete Split-NC
assignment of the selective F′ relation.

Assurance tier: model-level. A production compiler artifact must instantiate
`ProductionLayout`; this file does not infer physical columns from Rust counts.

Owns: a typed, injective owner for all 41 digit and 20 retained-borrow columns
of every opening; their embedding into the complete Phi81 carrier; exact
relabeling to the 21-row canonicality model; and the composed theorem that
Split-NC at `b = 2` plus those rows implies a canonical Goldilocks opening.

Does not own: a generated Rust-to-Lean layout instance, matrix-row refinement,
Fiat-Shamir soundness, or unconditional conversion of verifier acceptance into
`Nc.Truth`.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Split-NC source shape owned by one selective F′ relation profile. -/
def ncShape
    {rows columns : Nat}
    (profile : RelationProfile.Profile rows columns)
    (freshCount runningCount : Nat) : SemanticShape where
  rowVariables := profile.rowVariables
  logicalWidth := columns
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := RelationProfile.matrixCount

@[simp] theorem ncShape_logicalWidth
    {rows columns freshCount runningCount : Nat}
    (profile : RelationProfile.Profile rows columns) :
    (ncShape profile freshCount runningCount).logicalWidth = columns := by
  rfl

/-- Exhaustive semantic role of one distinct optimized canonical word.

`openingCount` counts memoized words, not commitment uses. Multiple Ajtai maps
may reference the same opening index without allocating another 61 columns. -/
inductive Coordinate (openingCount : Nat) where
  | digit (opening : Fin openingCount) (index : Fin digitCount)
  | borrow (opening : Fin openingCount) (index : Fin chunkBorrowCount)
deriving DecidableEq, Repr

/-- Lean-owned production-layout contract. Injectivity forbids accidental
aliasing between distinct memoized words, digits, and borrow endpoints. -/
structure ProductionLayout (columns openingCount : Nat) where
  column : Coordinate openingCount → Fin columns
  injective : Function.Injective column

namespace ProductionLayout

theorem column_has_unique_owner
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (coordinate : Coordinate openingCount) :
    ∃ owner : Coordinate openingCount,
      layout.column owner = layout.column coordinate ∧
        ∀ other : Coordinate openingCount,
          layout.column other = layout.column coordinate →
            other = owner := by
  refine ⟨coordinate, rfl, ?_⟩
  intro other equal
  exact layout.injective equal

end ProductionLayout

/-- Embed one owned logical F′ column into the complete Phi81 carrier. -/
def carrierColumn
    {columns openingCount : Nat}
    (layout : ProductionLayout columns openingCount)
    (coordinate : Coordinate openingCount) :
    Fin (Phi81CarrierLayout.carrierWidth columns) :=
  Phi81CarrierLayout.embedLogical (layout.column coordinate)

/-- Authoritative value of one optimized opening coordinate in one Split-NC
source assignment. -/
def coordinateValue
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (coordinate : Coordinate openingCount) : F :=
  data.assignment source (carrierColumn layout coordinate)

/-- Canonical block×lane decoding reads each owned opening coordinate from the
same authoritative Split-NC assignment, with no omitted carrier position. -/
theorem blockLane_reads_coordinate
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (coordinate : Coordinate openingCount) :
    Semantics.Nc.BlockLane.value (data.assignment source)
        (Phi81ColumnLayout.decode (carrierColumn layout coordinate)).1
        (Phi81ColumnLayout.decode (carrierColumn layout coordinate)).2 =
      coordinateValue layout data source coordinate := by
  exact Semantics.Nc.BlockLane.value_decode
    (data.assignment source) (carrierColumn layout coordinate)

/-- Interpret the local columns used by the 21-row proof. Columns outside its
41-digit and 20-borrow interface are deliberately absent. -/
def localCoordinate?
    {openingCount : Nat}
    (opening : Fin openingCount) (column : Nat) :
    Option (Coordinate openingCount) :=
  if digitRange : 58 ≤ column ∧ column < 58 + digitCount then
    some (.digit opening ⟨column - 58, by omega⟩)
  else if borrowRange :
      chunkBorrowColumnBase ≤ column ∧
        column < chunkBorrowColumnBase + chunkBorrowCount then
    some (.borrow opening
      ⟨column - chunkBorrowColumnBase, by omega⟩)
  else
    none

/-- Relabel one production opening to the compact assignment consumed by the
21-row canonicality theorem. -/
def localAssignment
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) : Nat → Nat :=
  fun column =>
    match localCoordinate? opening column with
    | some coordinate => (coordinateValue layout data source coordinate).val
    | none => 0

@[simp] theorem localCoordinate?_digit
    {openingCount : Nat}
    (opening : Fin openingCount) (index : Fin digitCount) :
    localCoordinate? opening
        (ShiftedTernary.digitCols.getD index.val 0) =
      some (.digit opening index) := by
  rw [digitColumn_formula index.isLt]
  unfold localCoordinate?
  have digitRange :
      58 ≤ 58 + index.val ∧
        58 + index.val < 58 + digitCount := by
    constructor
    · omega
    · exact Nat.add_lt_add_left index.isLt 58
  rw [dif_pos digitRange]
  congr 2
  apply Fin.ext
  simp

@[simp] theorem localCoordinate?_borrow
    {openingCount : Nat}
    (opening : Fin openingCount) (index : Fin chunkBorrowCount) :
    localCoordinate? opening (chunkBorrowColumnBase + index.val) =
      some (.borrow opening index) := by
  unfold localCoordinate?
  have outsideDigit :
      ¬ (58 ≤ chunkBorrowColumnBase + index.val ∧
        chunkBorrowColumnBase + index.val < 58 + digitCount) := by
    simp [chunkBorrowColumnBase, digitCount]
  rw [dif_neg outsideDigit]
  have borrowRange :
      chunkBorrowColumnBase ≤ chunkBorrowColumnBase + index.val ∧
        chunkBorrowColumnBase + index.val <
          chunkBorrowColumnBase + chunkBorrowCount := by
    constructor
    · omega
    · exact Nat.add_lt_add_left index.isLt chunkBorrowColumnBase
  rw [dif_pos borrowRange]
  congr 2
  apply Fin.ext
  simp

@[simp] theorem localAssignment_digit
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) (index : Fin digitCount) :
    localAssignment layout data source opening
        (ShiftedTernary.digitCols.getD index.val 0) =
      (coordinateValue layout data source (.digit opening index)).val := by
  unfold localAssignment
  rw [localCoordinate?_digit]

@[simp] theorem localAssignment_borrow
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) (index : Fin chunkBorrowCount) :
    localAssignment layout data source opening
        (chunkBorrowColumnBase + index.val) =
      (coordinateValue layout data source (.borrow opening index)).val := by
  unfold localAssignment
  rw [localCoordinate?_borrow]

/-- Split-NC truth supplies the required strict `b = 2` bound to every
coordinate owned by the production opening layout. -/
theorem splitNc_covers_coordinate
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (coordinate : Coordinate openingCount) :
    NormBoundTwo (coordinateValue layout data source coordinate).val := by
  apply normBoundTwo_iff_centeredResidue.mpr
  apply (concrete_norm_two_iff_centeredResidue _).mp
  exact truth source (carrierColumn layout coordinate)

theorem splitNc_covers_digit
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) (index : Fin digitCount) :
    NormBoundTwo
      (localAssignment layout data source opening
        (ShiftedTernary.digitCols.getD index.val 0)) := by
  rw [localAssignment_digit]
  exact splitNc_covers_coordinate layout data truth source
    (.digit opening index)

theorem splitNc_covers_borrow
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) (index : Fin chunkBorrowCount) :
    NormBoundTwo
      (localAssignment layout data source opening
        (chunkBorrowColumnBase + index.val)) := by
  rw [localAssignment_borrow]
  exact splitNc_covers_coordinate layout data truth source
    (.borrow opening index)

/-- Exact requested coverage statement: all 41 digits and all 20 retained
borrow endpoints of one opening are in Split-NC's strict `b = 2` domain. -/
def OpeningCoordinatesBoundTwo
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) : Prop :=
  (∀ index : Fin digitCount,
    NormBoundTwo
      (localAssignment layout data source opening
        (ShiftedTernary.digitCols.getD index.val 0))) ∧
  (∀ index : Fin chunkBorrowCount,
    NormBoundTwo
      (localAssignment layout data source opening
        (chunkBorrowColumnBase + index.val)))

theorem splitNc_covers_opening
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) :
    OpeningCoordinatesBoundTwo layout data source opening := by
  exact ⟨splitNc_covers_digit layout data truth source opening,
    splitNc_covers_borrow layout data truth source opening⟩

theorem splitNc_supplies_digitNormBoundTwo
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) :
    DigitNormBoundTwo (localAssignment layout data source opening) := by
  intro index indexLt
  exact splitNc_covers_digit layout data truth source opening
    ⟨index, indexLt⟩

/-- The production-layout interpretation of the exact 21 canonicality rows. -/
def CanonicalRowsHold
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount) : Prop :=
  ChunkScheduleHolds (localAssignment layout data source opening)

/-- End-to-end model theorem: the complete Split-NC relation supplies the
alphabet premise consumed by the exact 21-row chain, hence the represented
41-trit integer is strictly below the Goldilocks modulus. -/
theorem splitNc_and_canonicalRows_encoded_lt_modulus
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (rowsHold : CanonicalRowsHold layout data source opening) :
    lowValue
        (assignmentTritMod
          (localAssignment layout data source opening))
        digitCount <
      goldilocksP := by
  exact chunkSchedule_encoded_lt_modulus
    (splitNc_supplies_digitNormBoundTwo
      layout data truth source opening)
    rowsHold

/-- The same conclusion for every optimized opening owned by the layout. -/
theorem splitNc_and_allCanonicalRows_encoded_lt_modulus
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (truth : Semantics.Nc.Truth data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (rowsHold : ∀ opening,
      CanonicalRowsHold layout data source opening) :
    ∀ opening,
      lowValue
          (assignmentTritMod
            (localAssignment layout data source opening))
          digitCount <
        goldilocksP := by
  intro opening
  exact splitNc_and_canonicalRows_encoded_lt_modulus
    layout data truth source opening (rowsHold opening)

/-- Exact block×lane residual coverage can replace a caller-supplied truth
premise; no transcript or probabilistic acceptance claim is introduced. -/
theorem blockLaneResiduals_and_canonicalRows_encoded_lt_modulus
    (noZeroDivisors :
      NormRange.BaseFieldNoZeroDivisors)
    {rows columns freshCount runningCount openingCount : Nat}
    {profile : RelationProfile.Profile rows columns}
    (layout : ProductionLayout columns openingCount)
    (data : Data (ncShape profile freshCount runningCount))
    (residuals : Semantics.Nc.BlockLane.ResidualsZero data)
    (source : Fin (ncShape profile freshCount runningCount).sourceCount)
    (opening : Fin openingCount)
    (rowsHold : CanonicalRowsHold layout data source opening) :
    lowValue
        (assignmentTritMod
          (localAssignment layout data source opening))
        digitCount <
      goldilocksP := by
  apply splitNc_and_canonicalRows_encoded_lt_modulus
    layout data
      (Semantics.Nc.BlockLane.truth_of_residualsZero
        noZeroDivisors data residuals)
    source opening rowsHold

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc
