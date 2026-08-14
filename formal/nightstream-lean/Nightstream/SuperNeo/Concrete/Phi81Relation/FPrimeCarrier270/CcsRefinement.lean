import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericRowMap

/-!
Typed CCS preservation across the five-ring F' carrier repair.

Protocol: SuperNeo CCS/CE relation specialized to the F' source relation.
Phase: legacy CCS source to the aligned 270-public-coordinate Phi81 carrier.
Constraint family: matrix image and CCS residual semantics only; this file
emits no rows.

Owns: exact preservation of the canonical finite matrix-vector product when
the matrix and assignment are relocated together; preservation through
zero-completion to the full Phi81 carrier; construction of the typed relation
structure with the identical explicit constraint polynomial; and exact
`residualAt` / `ConstraintSatisfied` equivalence.

Does not own: the concrete eight F' matrices or polynomial, Rust sparse-row
storage, numeric-row artifact conformance, Ajtai commitments, CE coefficient
images, PiCCS, PiRLC, PiDEC, NIFS, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: a caller supplies one legacy matrix family, one legacy
assignment, and one explicit sparse constraint polynomial. The thirteen
inserted columns and the carrier-completion suffix are derived zeros in both
the matrix and assignment. The lifted polynomial is definitionally the same
object; no caller supplies a residual equivalence or satisfaction witness.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.ccs.carrier.columns.aligned` | jointly relocating `M` and `z` preserves `(M z)(x)` | derived | `alignedMatrixVectorAt_eq` |
| `fprime.ccs.carrier.columns.completed` | jointly zero-completing `M` and `z` preserves `(M z)(x)` | derived | `carrierMatrixVectorAt_eq` |
| `fprime.ccs.carrier.rows.numeric` | numeric little-endian rows select the same typed Boolean row | derived | `carrierMatrixVectorAt_numericRow_eq`, `carrierMatrixVectorAt_rowIndex_eq` |
| `fprime.ccs.carrier.structure.polynomial` | the lifted structure carries the identical sparse polynomial | direct dataflow | `liftStructure_constraintPolynomial` |
| `fprime.ccs.carrier.structure.images` | every lifted matrix image equals its legacy image | derived | `matrixImagesAt_eq` |
| `fprime.ccs.carrier.structure.residual` | the identical polynomial sees identical image vectors | derived | `residualAt_eq` |
| `fprime.ccs.carrier.structure.membership` | legacy and completed CCS zero sets are equivalent | derived | `constraintSatisfied_iff` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap

/-! ## Canonical finite-sum normalization -/

private theorem sumRange_zero (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps 0 term = 0 := by
  rfl

private theorem sumRange_succ (count : Nat) (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps (count + 1) term =
      sumRange ConcreteCarrier.baseOps count term + term count := by
  rfl

private theorem sumRange_append
    (leftCount rightCount : Nat) (term : Nat -> F) :
    sumRange ConcreteCarrier.baseOps (leftCount + rightCount) term =
      sumRange ConcreteCarrier.baseOps leftCount term +
        sumRange ConcreteCarrier.baseOps rightCount
          (fun index => term (leftCount + index)) := by
  induction rightCount with
  | zero =>
      rw [Nat.add_zero, sumRange_zero]
      exact (ConcreteCarrier.baseLaws.add_zero _).symm
  | succ rightCount inductionHypothesis =>
      rw [Nat.add_succ, sumRange_succ, sumRange_succ,
        inductionHypothesis]
      exact Lean.Grind.Fin.add_assoc _ _ _

private theorem sumRange_count_congr
    (leftCount rightCount : Nat) (term : Nat -> F)
    (countsEqual : leftCount = rightCount) :
    sumRange ConcreteCarrier.baseOps leftCount term =
      sumRange ConcreteCarrier.baseOps rightCount term := by
  exact congrArg (fun count => sumRange ConcreteCarrier.baseOps count term)
    countsEqual

private def listSum {Index : Type}
    (indices : List Index) (term : Index -> F) : F :=
  match indices with
  | [] => 0
  | index :: rest => term index + listSum rest term

private theorem foldl_eq_add_listSum
    {Index : Type} (indices : List Index) (term : Index -> F)
    (initial : F) :
    indices.foldl (fun accumulated index => accumulated + term index) initial =
      initial + listSum indices term := by
  induction indices generalizing initial with
  | nil => exact (ConcreteCarrier.baseLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem listSum_map
    {Left Right : Type} (indices : List Left) (map : Left -> Right)
    (term : Right -> F) :
    listSum (indices.map map) term =
      listSum indices (fun index => term (map index)) := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, listSum, inductionHypothesis]

private theorem listSum_append
    {Index : Type} (left right : List Index) (term : Index -> F) :
    listSum (left ++ right) term =
      listSum left term + listSum right term := by
  induction left with
  | nil => exact (ConcreteCarrier.baseLaws.zero_add _).symm
  | cons index left inductionHypothesis =>
      simp only [List.cons_append, listSum, inductionHypothesis]
      exact (ConcreteCarrier.baseLaws.add_assoc _ _ _).symm

private theorem listSum_range_eq_sumRange
    (count : Nat) (term : Nat -> F) :
    listSum (List.range count) term =
      sumRange ConcreteCarrier.baseOps count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, listSum_append, listSum, inductionHypothesis,
        sumRange_succ]
      rw [listSum]
      rw [Fin.add_zero]

/-- Natural-number view of one typed matrix-vector contribution. Indices
outside the declared width are definitionally zero. -/
private def matrixVectorTerm
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex variables) (index : Nat) : F :=
  if indexLt : index < columns then
    matrix vertex ⟨index, indexLt⟩ * assignment ⟨index, indexLt⟩
  else
    0

private theorem matrixVectorAt_eq_sumRange
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ConcreteCarrier.baseOps matrix assignment vertex =
      sumRange ConcreteCarrier.baseOps columns
        (matrixVectorTerm matrix assignment vertex) := by
  unfold matrixVectorAt
  calc
    (canonicalFinIndices columns).foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * assignment column) 0 =
      0 + listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
          foldl_eq_add_listSum _ _ 0
    _ = listSum (canonicalFinIndices columns)
        (fun column => matrix vertex column * assignment column) :=
      ConcreteCarrier.baseLaws.zero_add _
    _ = listSum ((canonicalFinIndices columns).map Fin.val)
        (matrixVectorTerm matrix assignment vertex) := by
      rw [listSum_map]
      apply congrArg (listSum (canonicalFinIndices columns))
      funext column
      simp [matrixVectorTerm, column.isLt]
    _ = listSum (List.range columns)
        (matrixVectorTerm matrix assignment vertex) := by
      rw [canonicalFinIndices_values]
    _ = _ := listSum_range_eq_sumRange _ _

/-! ## Column-alignment preservation -/

private def legacyPrivateWidth (dimensions : Dimensions) : Nat :=
  dimensions.legacyLogicalWidth - legacyPublicWidth

private theorem legacyWidth_eq (dimensions : Dimensions) :
    dimensions.legacyLogicalWidth =
      legacyPublicWidth + legacyPrivateWidth dimensions := by
  unfold legacyPrivateWidth
  exact (Nat.add_sub_of_le dimensions.legacyPublicFits).symm

private theorem alignedWidth_eq (dimensions : Dimensions) :
    dimensions.alignedLogicalWidth =
      legacyPublicWidth + fixedPaddingWidth + legacyPrivateWidth dimensions := by
  rw [Dimensions.alignedLogicalWidth, legacyWidth_eq]
  omega

private theorem alignedTerm_public
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables)
    (index : Nat) (indexLt : index < legacyPublicWidth) :
    matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex index =
      matrixVectorTerm legacyMatrix legacyAssignment vertex index := by
  have legacyBound : index < dimensions.legacyLogicalWidth :=
    Nat.lt_of_lt_of_le indexLt dimensions.legacyPublicFits
  have alignedBound : index < dimensions.alignedLogicalWidth := by
    have legacyFits := dimensions.legacyPublicFits
    simp only [Dimensions.alignedLogicalWidth, legacyPublicWidth,
      fixedPaddingWidth] at legacyFits indexLt ⊢
    omega
  let legacyColumn : Fin dimensions.legacyLogicalWidth := ⟨index, legacyBound⟩
  let alignedColumn : Fin dimensions.alignedLogicalWidth := ⟨index, alignedBound⟩
  have mapped : alignedColumn = alignedIndex dimensions legacyColumn := by
    apply Fin.ext
    symm
    exact alignedIndex_public dimensions legacyColumn indexLt
  simp only [matrixVectorTerm, dif_pos legacyBound, dif_pos alignedBound]
  change alignedMatrix dimensions legacyMatrix vertex alignedColumn *
      alignedLogicalAssignment dimensions legacyAssignment alignedColumn =
    legacyMatrix vertex legacyColumn * legacyAssignment legacyColumn
  rw [mapped, alignedMatrix_at_alignedIndex,
    alignedLogicalAssignment_at_alignedIndex]

private theorem alignedTerm_padding
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables)
    (offset : Nat) (offsetLt : offset < fixedPaddingWidth) :
    matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex
        (legacyPublicWidth + offset) = 0 := by
  have alignedBound :
      legacyPublicWidth + offset < dimensions.alignedLogicalWidth := by
    have legacyFits := dimensions.legacyPublicFits
    simp only [Dimensions.alignedLogicalWidth, legacyPublicWidth,
      fixedPaddingWidth] at legacyFits offsetLt ⊢
    omega
  let paddingOffset : Fin fixedPaddingWidth := ⟨offset, offsetLt⟩
  let alignedColumn : Fin dimensions.alignedLogicalWidth :=
    ⟨legacyPublicWidth + offset, alignedBound⟩
  have mapped : alignedColumn = paddingLogicalColumn dimensions paddingOffset := by
    apply Fin.ext
    rfl
  simp only [matrixVectorTerm, dif_pos alignedBound]
  change alignedMatrix dimensions legacyMatrix vertex alignedColumn *
      alignedLogicalAssignment dimensions legacyAssignment alignedColumn = 0
  rw [mapped, alignedMatrix_padding_zero,
    alignedLogicalAssignment_padding_zero, Fin.zero_mul]

private theorem alignedTerm_private
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables)
    (offset : Nat) (offsetLt : offset < legacyPrivateWidth dimensions) :
    matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex
        (legacyPublicWidth + fixedPaddingWidth + offset) =
      matrixVectorTerm legacyMatrix legacyAssignment vertex
        (legacyPublicWidth + offset) := by
  have legacyBound :
      legacyPublicWidth + offset < dimensions.legacyLogicalWidth := by
    rw [legacyWidth_eq dimensions]
    omega
  have alignedBound :
      legacyPublicWidth + fixedPaddingWidth + offset <
        dimensions.alignedLogicalWidth := by
    rw [alignedWidth_eq dimensions]
    omega
  let legacyColumn : Fin dimensions.legacyLogicalWidth :=
    ⟨legacyPublicWidth + offset, legacyBound⟩
  let alignedColumn : Fin dimensions.alignedLogicalWidth :=
    ⟨legacyPublicWidth + fixedPaddingWidth + offset, alignedBound⟩
  have privateColumn : legacyPublicWidth <= legacyColumn.val := by
    simp only [legacyColumn]
    omega
  have mapped : alignedColumn = alignedIndex dimensions legacyColumn := by
    apply Fin.ext
    symm
    rw [alignedIndex_private dimensions legacyColumn privateColumn]
    simp only [legacyColumn, alignedColumn]
    omega
  simp only [matrixVectorTerm, dif_pos legacyBound, dif_pos alignedBound]
  change alignedMatrix dimensions legacyMatrix vertex alignedColumn *
      alignedLogicalAssignment dimensions legacyAssignment alignedColumn =
    legacyMatrix vertex legacyColumn * legacyAssignment legacyColumn
  rw [mapped, alignedMatrix_at_alignedIndex,
    alignedLogicalAssignment_at_alignedIndex]

/-- Inserting the thirteen fixed public zeros in both the matrix and
assignment preserves the exact canonical finite dot product at every typed
Boolean row. -/
theorem alignedMatrixVectorAt_eq
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables) :
    matrixVectorAt ConcreteCarrier.baseOps
        (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex =
      matrixVectorAt ConcreteCarrier.baseOps legacyMatrix legacyAssignment
        vertex := by
  rw [matrixVectorAt_eq_sumRange, matrixVectorAt_eq_sumRange]
  have publicSum :
      sumRange ConcreteCarrier.baseOps legacyPublicWidth
          (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
            (alignedLogicalAssignment dimensions legacyAssignment) vertex) =
        sumRange ConcreteCarrier.baseOps legacyPublicWidth
          (matrixVectorTerm legacyMatrix legacyAssignment vertex) := by
    apply sumRange_congr
    intro index indexLt
    exact alignedTerm_public dimensions legacyMatrix legacyAssignment vertex
      index indexLt
  have paddingSum :
      sumRange ConcreteCarrier.baseOps fixedPaddingWidth
          (fun offset =>
            matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
              (alignedLogicalAssignment dimensions legacyAssignment) vertex
              (legacyPublicWidth + offset)) = 0 := by
    apply sumRange_eq_zero ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
    intro offset offsetLt
    exact alignedTerm_padding dimensions legacyMatrix legacyAssignment vertex
      offset offsetLt
  have privateSum :
      sumRange ConcreteCarrier.baseOps (legacyPrivateWidth dimensions)
          (fun offset =>
            matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
              (alignedLogicalAssignment dimensions legacyAssignment) vertex
              (legacyPublicWidth + fixedPaddingWidth + offset)) =
        sumRange ConcreteCarrier.baseOps (legacyPrivateWidth dimensions)
          (fun offset => matrixVectorTerm legacyMatrix legacyAssignment vertex
            (legacyPublicWidth + offset)) := by
    apply sumRange_congr
    intro offset offsetLt
    exact alignedTerm_private dimensions legacyMatrix legacyAssignment vertex
      offset offsetLt
  calc
    sumRange ConcreteCarrier.baseOps dimensions.alignedLogicalWidth
        (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
          (alignedLogicalAssignment dimensions legacyAssignment) vertex) =
      sumRange ConcreteCarrier.baseOps
          (legacyPublicWidth + fixedPaddingWidth +
            legacyPrivateWidth dimensions)
          (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
            (alignedLogicalAssignment dimensions legacyAssignment) vertex) := by
        exact sumRange_count_congr _ _ _ (alignedWidth_eq dimensions)
    _ =
      (sumRange ConcreteCarrier.baseOps legacyPublicWidth
          (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
            (alignedLogicalAssignment dimensions legacyAssignment) vertex) +
        sumRange ConcreteCarrier.baseOps fixedPaddingWidth
          (fun offset =>
            matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
              (alignedLogicalAssignment dimensions legacyAssignment) vertex
              (legacyPublicWidth + offset))) +
      sumRange ConcreteCarrier.baseOps (legacyPrivateWidth dimensions)
        (fun offset =>
          matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
            (alignedLogicalAssignment dimensions legacyAssignment) vertex
            (legacyPublicWidth + fixedPaddingWidth + offset)) := by
        rw [sumRange_append
          (legacyPublicWidth + fixedPaddingWidth)
          (legacyPrivateWidth dimensions),
          sumRange_append legacyPublicWidth fixedPaddingWidth]
    _ =
      sumRange ConcreteCarrier.baseOps legacyPublicWidth
          (matrixVectorTerm legacyMatrix legacyAssignment vertex) +
        sumRange ConcreteCarrier.baseOps (legacyPrivateWidth dimensions)
          (fun offset => matrixVectorTerm legacyMatrix legacyAssignment vertex
            (legacyPublicWidth + offset)) := by
        rw [publicSum, paddingSum, privateSum,
          Fin.add_zero]
    _ = sumRange ConcreteCarrier.baseOps
        (legacyPublicWidth + legacyPrivateWidth dimensions)
        (matrixVectorTerm legacyMatrix legacyAssignment vertex) := by
      rw [sumRange_append]
    _ = sumRange ConcreteCarrier.baseOps dimensions.legacyLogicalWidth
        (matrixVectorTerm legacyMatrix legacyAssignment vertex) := by
      exact sumRange_count_congr _ _ _ (legacyWidth_eq dimensions).symm

/-! ## Total-carrier completion preservation -/

private theorem carrierTerm_prefix
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables)
    (index : Nat) (indexLt : index < dimensions.alignedLogicalWidth) :
    matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
        (assignment dimensions legacyAssignment) vertex index =
      matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex index := by
  have carrierBound : index < dimensions.shape.carrierWidth :=
    Nat.lt_of_lt_of_le indexLt
      (Phi81CarrierLayout.logicalWidth_le_carrierWidth
        dimensions.alignedLogicalWidth)
  let logicalColumn : Fin dimensions.alignedLogicalWidth := ⟨index, indexLt⟩
  let carrierColumn : Fin dimensions.shape.carrierWidth := ⟨index, carrierBound⟩
  have mapped : carrierColumn = Phi81CarrierLayout.embedLogical logicalColumn := by
    apply Fin.ext
    rfl
  simp only [matrixVectorTerm, dif_pos indexLt, dif_pos carrierBound]
  change carrierMatrix dimensions legacyMatrix vertex carrierColumn *
      assignment dimensions legacyAssignment carrierColumn =
    alignedMatrix dimensions legacyMatrix vertex logicalColumn *
      alignedLogicalAssignment dimensions legacyAssignment logicalColumn
  rw [mapped]
  unfold carrierMatrix assignment
  congr 1
  · exact Phi81CarrierLayout.extendMatrix_embedLogical 0
      (alignedMatrix dimensions legacyMatrix) vertex logicalColumn
  · exact Phi81CarrierLayout.extendAssignment_embedLogical 0
      (alignedLogicalAssignment dimensions legacyAssignment) logicalColumn

private theorem carrierTerm_tail
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables)
    (offset : Nat)
    (offsetLt : offset <
      dimensions.shape.carrierWidth - dimensions.alignedLogicalWidth) :
    matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
        (assignment dimensions legacyAssignment) vertex
        (dimensions.alignedLogicalWidth + offset) = 0 := by
  have logicalFits : dimensions.alignedLogicalWidth <=
      dimensions.shape.carrierWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth
      dimensions.alignedLogicalWidth
  have carrierBound :
      dimensions.alignedLogicalWidth + offset <
        dimensions.shape.carrierWidth := by
    omega
  let column : Fin dimensions.shape.carrierWidth :=
    ⟨dimensions.alignedLogicalWidth + offset, carrierBound⟩
  have isCompletion : dimensions.alignedLogicalWidth <= column.val := by
    simp only [column]
    omega
  simp only [matrixVectorTerm, dif_pos carrierBound]
  rw [carrierMatrix_completion_zero dimensions legacyMatrix vertex column
      isCompletion,
    assignment_completion_zero dimensions legacyAssignment column isCompletion,
    Fin.zero_mul]

/-- Completing the already aligned matrix and fresh assignment with the same
canonical zero suffix preserves their exact finite dot product. -/
theorem carrierMatrixVectorAt_eq
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables) :
    matrixVectorAt ConcreteCarrier.baseOps
        (carrierMatrix dimensions legacyMatrix)
        (assignment dimensions legacyAssignment) vertex =
      matrixVectorAt ConcreteCarrier.baseOps legacyMatrix legacyAssignment
        vertex := by
  rw [matrixVectorAt_eq_sumRange]
  have logicalFits : dimensions.alignedLogicalWidth <=
      dimensions.shape.carrierWidth :=
    Phi81CarrierLayout.logicalWidth_le_carrierWidth
      dimensions.alignedLogicalWidth
  have carrierWidthEq :
      dimensions.shape.carrierWidth = dimensions.alignedLogicalWidth +
        (dimensions.shape.carrierWidth - dimensions.alignedLogicalWidth) := by
    omega
  have prefixSum :
      sumRange ConcreteCarrier.baseOps dimensions.alignedLogicalWidth
          (matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
            (assignment dimensions legacyAssignment) vertex) =
        sumRange ConcreteCarrier.baseOps dimensions.alignedLogicalWidth
          (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
            (alignedLogicalAssignment dimensions legacyAssignment) vertex) := by
    apply sumRange_congr
    intro index indexLt
    exact carrierTerm_prefix dimensions legacyMatrix legacyAssignment vertex
      index indexLt
  have tailSum :
      sumRange ConcreteCarrier.baseOps
          (dimensions.shape.carrierWidth - dimensions.alignedLogicalWidth)
          (fun offset =>
            matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
              (assignment dimensions legacyAssignment) vertex
              (dimensions.alignedLogicalWidth + offset)) = 0 := by
    apply sumRange_eq_zero ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
    intro offset offsetLt
    exact carrierTerm_tail dimensions legacyMatrix legacyAssignment vertex
      offset offsetLt
  calc
    sumRange ConcreteCarrier.baseOps dimensions.shape.carrierWidth
        (matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
          (assignment dimensions legacyAssignment) vertex) =
      sumRange ConcreteCarrier.baseOps
          (dimensions.alignedLogicalWidth +
            (dimensions.shape.carrierWidth -
              dimensions.alignedLogicalWidth))
          (matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
            (assignment dimensions legacyAssignment) vertex) := by
        exact sumRange_count_congr _ _ _ carrierWidthEq
    _ =
      sumRange ConcreteCarrier.baseOps dimensions.alignedLogicalWidth
          (matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
            (assignment dimensions legacyAssignment) vertex) +
        sumRange ConcreteCarrier.baseOps
          (dimensions.shape.carrierWidth - dimensions.alignedLogicalWidth)
          (fun offset =>
            matrixVectorTerm (carrierMatrix dimensions legacyMatrix)
              (assignment dimensions legacyAssignment) vertex
              (dimensions.alignedLogicalWidth + offset)) := by
        rw [sumRange_append]
    _ = sumRange ConcreteCarrier.baseOps dimensions.alignedLogicalWidth
        (matrixVectorTerm (alignedMatrix dimensions legacyMatrix)
          (alignedLogicalAssignment dimensions legacyAssignment) vertex) := by
      rw [prefixSum, tailSum, Fin.add_zero]
    _ = matrixVectorAt ConcreteCarrier.baseOps
        (alignedMatrix dimensions legacyMatrix)
        (alignedLogicalAssignment dimensions legacyAssignment) vertex := by
      rw [matrixVectorAt_eq_sumRange]
    _ = matrixVectorAt ConcreteCarrier.baseOps legacyMatrix legacyAssignment
        vertex := alignedMatrixVectorAt_eq dimensions legacyMatrix
          legacyAssignment vertex

/-- Numeric little-endian row decoding is orthogonal to the column repair:
the completed image at a numeric row equals the legacy image at the exact
typed Boolean row decoded by `rowVertex`. -/
theorem carrierMatrixVectorAt_numericRow_eq
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (row : Fin (2 ^ dimensions.rowVariables)) :
    matrixVectorAt ConcreteCarrier.baseOps
        (carrierMatrix dimensions legacyMatrix)
        (assignment dimensions legacyAssignment)
        (rowVertex dimensions.rowVariables row) =
      matrixVectorAt ConcreteCarrier.baseOps legacyMatrix legacyAssignment
        (rowVertex dimensions.rowVariables row) := by
  exact carrierMatrixVectorAt_eq dimensions legacyMatrix legacyAssignment _

/-- Encoding a typed Boolean row numerically and decoding it again cannot
change the matrix image before or after carrier repair. -/
theorem carrierMatrixVectorAt_rowIndex_eq
    (dimensions : Dimensions)
    (legacyMatrix : BooleanMatrix F dimensions.rowVariables
      dimensions.legacyLogicalWidth)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables) :
    matrixVectorAt ConcreteCarrier.baseOps
        (carrierMatrix dimensions legacyMatrix)
        (assignment dimensions legacyAssignment)
        (rowVertex dimensions.rowVariables
          ⟨rowIndex vertex, rowIndex_lt_twoPow vertex⟩) =
      matrixVectorAt ConcreteCarrier.baseOps legacyMatrix legacyAssignment
        vertex := by
  rw [rowVertex_rowIndex]
  exact carrierMatrixVectorAt_eq dimensions legacyMatrix legacyAssignment vertex

/-! ## Structure and CCS residual preservation -/

/-- Legacy CCS structure over the pre-alignment logical width. -/
abbrev LegacyStructure (dimensions : Dimensions) :=
  CCSResidualTable.Structure F dimensions.shape.sourceShape
    dimensions.legacyLogicalWidth

/-- Lift one legacy CCS source into the typed five-ring Phi81 relation. The
explicit sparse constraint polynomial is reused unchanged. -/
def liftStructure (dimensions : Dimensions)
    (legacy : LegacyStructure dimensions) :
    Structure dimensions.shape where
  matrices := fun matrix => alignedMatrix dimensions (legacy.matrices matrix)
  constraintPolynomial := legacy.constraintPolynomial

/-- The carrier repair does not rewrite, reinterpret, or replace the legacy
constraint polynomial. -/
@[simp] theorem liftStructure_constraintPolynomial
    (dimensions : Dimensions) (legacy : LegacyStructure dimensions) :
    (liftStructure dimensions legacy).constraintPolynomial =
      legacy.constraintPolynomial := by
  rfl

/-- The CCS matrix family exposed by the typed relation is exactly the
jointly relocated and completed legacy matrix family. -/
theorem liftStructure_matrixSource_matrix
    (dimensions : Dimensions) (legacy : LegacyStructure dimensions)
    (matrix : Fin dimensions.matrixCount)
    (vertex : BooleanVertex dimensions.rowVariables)
    (column : Fin dimensions.shape.carrierWidth) :
    (liftStructure dimensions legacy).matrixSource.system.matrices matrix
        vertex column =
      carrierMatrix dimensions (legacy.matrices matrix) vertex column := by
  rfl

/-- Every finite matrix-image coordinate of the lifted typed structure is
exactly its legacy image. -/
theorem matrixImagesAt_eq
    (dimensions : Dimensions) (legacy : LegacyStructure dimensions)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables) :
    CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
        (liftStructure dimensions legacy).matrixSource.system
        (assignment dimensions legacyAssignment) vertex =
      CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps legacy
        legacyAssignment vertex := by
  funext matrix
  exact carrierMatrixVectorAt_eq dimensions (legacy.matrices matrix)
    legacyAssignment vertex

/-- With the same explicit polynomial and identical derived matrix images,
the completed typed relation has exactly the legacy CCS residual at every
Boolean row. -/
theorem residualAt_eq
    (dimensions : Dimensions) (legacy : LegacyStructure dimensions)
    (legacyAssignment : LegacyAssignment dimensions)
    (vertex : BooleanVertex dimensions.rowVariables) :
    CCSResidualTable.residualAt ConcreteCarrier.baseOps
        (liftStructure dimensions legacy).matrixSource.system
        (assignment dimensions legacyAssignment) vertex =
      CCSResidualTable.residualAt ConcreteCarrier.baseOps legacy
        legacyAssignment vertex := by
  unfold CCSResidualTable.residualAt
  rw [matrixImagesAt_eq]
  rfl

/-- The five-ring carrier repair is semantics-preserving for the independent
paper CCS zero-set predicate. This is a model-level theorem only: it does not
claim that production Rust supplies `legacy` or implements these maps. -/
theorem constraintSatisfied_iff
    (dimensions : Dimensions) (legacy : LegacyStructure dimensions)
    (legacyAssignment : LegacyAssignment dimensions) :
    CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
        (liftStructure dimensions legacy).matrixSource.system
        (assignment dimensions legacyAssignment) <->
      CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps legacy
        legacyAssignment := by
  constructor <;> intro satisfied vertex
  · rw [← residualAt_eq]
    exact satisfied vertex
  · rw [residualAt_eq]
    exact satisfied vertex

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
