import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

/-!
Focused model-level regressions for the typed five-ring F' carrier.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| F' / CCS | public shape | five complete rings | public width is exactly 270 |
| F' / CCS | logical lowering | private shift | old column 257 moves to column 270 without changing value |
| F' / CCS | fresh assignment | fixed padding | every coordinate 257 through 269 is zero |
| F' / CCS | public projection | exact boundary | projection is the legacy prefix followed by fixed zeros |
| F' / CCS | matrix source | aligned column ownership | old coefficients relocate injectively and padding is zero |
| F' / CCS | matrix source | little-endian row order | numeric rows `0..3` decode as `00,10,01,11` |
| F' / CCS | matrix source | finite-row padding | one actual row is preserved and the second Boolean row is zero |
| F' / CCS | matrix evaluation | tensor weight | numeric formula equals the independent Boolean equality weight |
| assurance | necessity | tail value one | relaxed boundary accepts it while fixed padding rejects it |
-/

namespace tests.FPrimeCarrier270

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.RowPadding

#check dimensions_exact
#check Dimensions.shape_publicRingColumns
#check Dimensions.shape_publicWidth
#check alignedIndex_injective
#check assignment_private_shift
#check assignment_fixedPublicPadding
#check projectPublicInput_exact
#check legacyIndex?_alignedIndex
#check legacyIndex?_eq_none_iff
#check alignedMatrix_at_alignedIndex
#check alignedMatrix_padding_zero
#check carrierMatrix_at_alignedCarrierIndex
#check carrierMatrix_completion_zero
#check rowIndex_lt_twoPow
#check rowIndex_rowVertex
#check rowVertex_rowIndex
#check productionTensorWeight_eq_equalityWeight
#check padRows_at_numericRow
#check padRows_atPadding
#check padRows_oneRow_actual
#check padRows_oneRow_padding
#check tailOne_normBounded
#check omittingFixedPadding_enlargesFreshBoundary

/-! Numeric row one sets the head coordinate, not the tail coordinate. These
four regressions distinguish the production little-endian row convention from
the canonical `BooleanVertex.all` list positions. -/

example : rowVertex 2 (0 : Fin (2 ^ 2)) =
    .cons false (.cons false .nil) := by decide

example : rowVertex 2 (1 : Fin (2 ^ 2)) =
    .cons true (.cons false .nil) := by decide

example : rowVertex 2 (2 : Fin (2 ^ 2)) =
    .cons false (.cons true .nil) := by decide

example : rowVertex 2 (3 : Fin (2 ^ 2)) =
    .cons true (.cons true .nil) := by decide

example :
    [rowIndex (.cons false (.cons false .nil)),
      rowIndex (.cons true (.cons false .nil)),
      rowIndex (.cons false (.cons true .nil)),
      rowIndex (.cons true (.cons true .nil))] = [0, 1, 2, 3] := by
  decide

/-! The model-level one-row specialization is exact: numeric row zero is
preserved, while the remaining row in the one-variable Boolean cube is zero. -/

def oneRowMatrix : NumericMatrix Nat 1 2 :=
  fun _ column => column.val + 7

example (column : Fin 2) :
    padRows (rowVariables := 1) oneRowMatrix (rowVertex 1 ⟨0, by decide⟩)
        column =
      oneRowMatrix ⟨0, by decide⟩ column := by
  exact padRows_oneRow_actual oneRowMatrix column

example (column : Fin 2) :
    padRows (rowVariables := 1) oneRowMatrix (rowVertex 1 ⟨1, by decide⟩)
        column = 0 := by
  exact padRows_oneRow_padding oneRowMatrix column

/-- One private coordinate makes the fixture distinguish insertion from simple
end padding. -/
def dimensions : Dimensions where
  rowVariables := 0
  legacyLogicalWidth := 258
  matrixCount := 0
  legacyPublicFits := by decide

def legacyAssignment : LegacyAssignment dimensions :=
  fun column => if column.val = 257 then 7 else 0

def legacyMatrix :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.BooleanMatrix
      F dimensions.rowVariables dimensions.legacyLogicalWidth :=
  fun _ column => if column.val = 257 then 9 else 0

def lastLegacyPublic : Fin dimensions.legacyLogicalWidth := ⟨256, by decide⟩

def firstLegacyPrivate : Fin dimensions.legacyLogicalWidth := ⟨257, by decide⟩

example : dimensions.shape.publicRingColumns = 5 := by
  exact Dimensions.shape_publicRingColumns dimensions

example : dimensions.shape.publicWidth = 270 := by
  exact Dimensions.shape_publicWidth dimensions

example : (alignedIndex dimensions lastLegacyPublic).val = 256 := by
  exact alignedIndex_public dimensions lastLegacyPublic (by decide)

example : (alignedIndex dimensions firstLegacyPrivate).val = 270 := by
  exact alignedIndex_private dimensions firstLegacyPrivate (by decide)

example : legacyIndex? dimensions (alignedIndex dimensions firstLegacyPrivate) =
    some firstLegacyPrivate := by
  exact legacyIndex?_alignedIndex dimensions firstLegacyPrivate

example : alignedMatrix dimensions legacyMatrix .nil
    (alignedIndex dimensions firstLegacyPrivate) = 9 := by
  rw [alignedMatrix_at_alignedIndex]
  simp [legacyMatrix, firstLegacyPrivate]

example : alignedMatrix dimensions legacyMatrix .nil
    (paddingLogicalColumn dimensions firstPaddingOffset) = 0 := by
  exact alignedMatrix_padding_zero dimensions legacyMatrix .nil
    firstPaddingOffset

example : carrierMatrix dimensions legacyMatrix .nil
    (alignedCarrierIndex dimensions firstLegacyPrivate) = 9 := by
  rw [carrierMatrix_at_alignedCarrierIndex]
  simp [legacyMatrix, firstLegacyPrivate]

example :
    assignment dimensions legacyAssignment
        (alignedCarrierIndex dimensions firstLegacyPrivate) = 7 := by
  rw [assignment_at_alignedIndex]
  simp [legacyAssignment, firstLegacyPrivate]

example : FixedPublicPadding dimensions
    (assignment dimensions legacyAssignment) := by
  exact assignment_fixedPublicPadding dimensions legacyAssignment

example :
    projectPublicInput (assignment dimensions legacyAssignment) =
      expectedPublicInput dimensions legacyAssignment := by
  exact projectPublicInput_exact dimensions legacyAssignment

example : assignmentNormBounded 2 (tailOneAssignment dimensions) := by
  exact tailOne_normBounded dimensions

example : ¬ FixedPublicPadding dimensions (tailOneAssignment dimensions) := by
  exact tailOne_not_fixedPublicPadding dimensions

example :
    exists candidate,
      RelaxedFreshBoundary dimensions (zeroLegacyAssignment dimensions)
          candidate /\
        ¬ FixedFreshBoundary dimensions (zeroLegacyAssignment dimensions)
          candidate := by
  exact omittingFixedPadding_enlargesFreshBoundary dimensions

end tests.FPrimeCarrier270
