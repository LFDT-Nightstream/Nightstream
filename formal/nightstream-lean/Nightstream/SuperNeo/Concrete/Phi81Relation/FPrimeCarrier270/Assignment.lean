import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics

/-!
Typed assignment boundary for the five-ring F' public carrier.

Protocol: SuperNeo CCS/CE relation specialized to the F' public interface.
Phase: legacy logical assignment to complete Phi81 carrier.
Constraint family: semantic column ownership only; this file emits no rows.

Owns: the exact `257 + 13 = 270 = 54 * 5` dimensions; an explicit
legacy-width premise; insertion of thirteen definitionally fixed zeros before
the private suffix; canonical completion of the total carrier; and exact
public-projection, injectivity, and private-shift theorems.

Does not own: CCS satisfaction, CE evaluation equivalence, commitments, Ajtai
setup, PiCCS, PiRLC, PiDEC, NIFS, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: legacy columns are mapped injectively. Public coordinates
257 through 269 have no legacy owner and are definitions equal to zero. Total
carrier coordinates beyond the aligned logical width are also definitions
equal to zero for this fresh assignment constructor.

| Protocol | Phase | Family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| F' / CCS | public shape | five complete rings | `270 = 54 * 5` | `dimensions_exact` |
| F' / CCS | logical lowering | old public / private | preserve `<257`; shift `>=257` by 13 | `alignedIndex_public`, `alignedIndex_private` |
| F' / CCS | logical lowering | column connectivity | every legacy column has one distinct image | `alignedIndex_injective` |
| F' / CCS | fresh assignment | fixed public padding | columns 257 through 269 are zero | `assignment_fixedPublicPadding` |
| F' / CCS | public projection | complete public input | first 257 legacy values followed by 13 zeros | `projectPublicInput_exact` |
| coefficient carrier | fresh completion | total tail | coordinates beyond the aligned logical width are zero | `assignment_completion_zero` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Existing F' scalar interface: one constant plus 256 digest bits. -/
def legacyPublicWidth : Nat := 257

/-- Missing coefficients required to complete the fifth public Phi81 ring. -/
def fixedPaddingWidth : Nat := 13

/-- The paper-visible F' public carrier consists of five complete rings. -/
def publicRingColumns : Nat := 5

/-- Exact field width of the five-ring public carrier. -/
def alignedPublicWidth : Nat := ringDegree * publicRingColumns

/-- Dimensions supplied by the legacy logical relation. The premise states
explicitly that its assignment contains the existing 257-field public prefix. -/
structure Dimensions where
  rowVariables : Nat
  legacyLogicalWidth : Nat
  matrixCount : Nat
  legacyPublicFits : legacyPublicWidth <= legacyLogicalWidth

namespace Dimensions

/-- Logical width after inserting the fixed public-padding block. -/
def alignedLogicalWidth (dimensions : Dimensions) : Nat :=
  dimensions.legacyLogicalWidth + fixedPaddingWidth

theorem alignedPublicFitsLogical (dimensions : Dimensions) :
    alignedPublicWidth <= dimensions.alignedLogicalWidth := by
  have legacyFits := dimensions.legacyPublicFits
  simp only [alignedPublicWidth, publicRingColumns, ringDegree,
    alignedLogicalWidth, fixedPaddingWidth, legacyPublicWidth] at legacyFits |- 
  omega

theorem alignedPublicFitsCarrier (dimensions : Dimensions) :
    alignedPublicWidth <=
      Phi81CarrierLayout.carrierWidth dimensions.alignedLogicalWidth := by
  exact Nat.le_trans dimensions.alignedPublicFitsLogical
    (Phi81CarrierLayout.logicalWidth_le_carrierWidth
      dimensions.alignedLogicalWidth)

/-- Exact typed Phi81 relation shape for this legacy relation. -/
def shape (dimensions : Dimensions) : Shape where
  rowVariables := dimensions.rowVariables
  logicalWidth := dimensions.alignedLogicalWidth
  matrixCount := dimensions.matrixCount
  publicRingColumns := publicRingColumns
  publicFits := dimensions.alignedPublicFitsCarrier

@[simp] theorem shape_logicalWidth (dimensions : Dimensions) :
    dimensions.shape.logicalWidth = dimensions.alignedLogicalWidth := by
  rfl

@[simp] theorem shape_publicRingColumns (dimensions : Dimensions) :
    dimensions.shape.publicRingColumns = 5 := by
  rfl

@[simp] theorem shape_publicWidth (dimensions : Dimensions) :
    dimensions.shape.publicWidth = 270 := by
  simp [shape, Shape.publicWidth, publicRingColumns, ringDegree]

end Dimensions

theorem dimensions_exact :
    legacyPublicWidth = 257 /\ fixedPaddingWidth = 13 /\
      alignedPublicWidth = 270 /\ alignedPublicWidth = ringDegree * 5 := by
  decide

/-- Legacy field assignment before the public carrier is ring-aligned. -/
abbrev LegacyAssignment (dimensions : Dimensions) :=
  PaperLinearAlgebra.Assignment F dimensions.legacyLogicalWidth

/-- Intermediate logical assignment after inserting thirteen public zeros. -/
abbrev AlignedLogicalAssignment (dimensions : Dimensions) :=
  PaperLinearAlgebra.Assignment F dimensions.alignedLogicalWidth

/-- Old scalar column to aligned scalar column. Public columns stay fixed;
private columns move past the thirteen verifier-fixed public coordinates. -/
def alignedIndex (dimensions : Dimensions)
    (column : Fin dimensions.legacyLogicalWidth) :
    Fin dimensions.alignedLogicalWidth :=
  if isPublic : column.val < legacyPublicWidth then
    ⟨column.val, by
      have columnBound := column.isLt
      simp only [Dimensions.alignedLogicalWidth, fixedPaddingWidth]
      omega⟩
  else
    ⟨column.val + fixedPaddingWidth, by
      have columnBound := column.isLt
      simp only [Dimensions.alignedLogicalWidth]
      omega⟩

theorem alignedIndex_public (dimensions : Dimensions)
    (column : Fin dimensions.legacyLogicalWidth)
    (isPublic : column.val < legacyPublicWidth) :
    (alignedIndex dimensions column).val = column.val := by
  simp [alignedIndex, isPublic]

theorem alignedIndex_private (dimensions : Dimensions)
    (column : Fin dimensions.legacyLogicalWidth)
    (isPrivate : legacyPublicWidth <= column.val) :
    (alignedIndex dimensions column).val =
      column.val + fixedPaddingWidth := by
  simp [alignedIndex, Nat.not_lt.mpr isPrivate]

theorem alignedIndex_injective (dimensions : Dimensions) :
    Function.Injective (alignedIndex dimensions) := by
  intro left right equal
  have values := congrArg Fin.val equal
  by_cases leftPublic : left.val < legacyPublicWidth
  · rw [alignedIndex_public dimensions left leftPublic] at values
    by_cases rightPublic : right.val < legacyPublicWidth
    · rw [alignedIndex_public dimensions right rightPublic] at values
      exact Fin.ext values
    · rw [alignedIndex_private dimensions right
          (Nat.not_lt.mp rightPublic)] at values
      have rightPrivate := Nat.not_lt.mp rightPublic
      simp only [legacyPublicWidth] at leftPublic rightPrivate
      simp only [fixedPaddingWidth] at values
      omega
  · rw [alignedIndex_private dimensions left
        (Nat.not_lt.mp leftPublic)] at values
    by_cases rightPublic : right.val < legacyPublicWidth
    · rw [alignedIndex_public dimensions right rightPublic] at values
      have leftPrivate := Nat.not_lt.mp leftPublic
      simp only [legacyPublicWidth] at leftPrivate rightPublic
      simp only [fixedPaddingWidth] at values
      omega
    · rw [alignedIndex_private dimensions right
          (Nat.not_lt.mp rightPublic)] at values
      apply Fin.ext
      simp only [fixedPaddingWidth] at values
      omega

/-- Insert the fixed public-padding block at the logical assignment layer. -/
def alignedLogicalAssignment (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    AlignedLogicalAssignment dimensions :=
  fun column =>
    if isPublic : column.val < legacyPublicWidth then
      legacy ⟨column.val,
        Nat.lt_of_lt_of_le isPublic dimensions.legacyPublicFits⟩
    else if isPadding : column.val < alignedPublicWidth then
      0
    else
      legacy ⟨column.val - fixedPaddingWidth, by
        have columnBound :
            column.val < dimensions.legacyLogicalWidth + fixedPaddingWidth := by
          simpa only [Dimensions.alignedLogicalWidth] using column.isLt
        have beyondPublic : alignedPublicWidth <= column.val :=
          Nat.not_lt.mp isPadding
        simp only [alignedPublicWidth, publicRingColumns, ringDegree,
          fixedPaddingWidth] at beyondPublic columnBound |- 
        omega⟩

theorem alignedLogicalAssignment_at_alignedIndex
    (dimensions : Dimensions) (legacy : LegacyAssignment dimensions)
    (column : Fin dimensions.legacyLogicalWidth) :
    alignedLogicalAssignment dimensions legacy (alignedIndex dimensions column) =
      legacy column := by
  by_cases isPublic : column.val < legacyPublicWidth
  · simp [alignedLogicalAssignment, alignedIndex, isPublic]
  · have isPrivate : legacyPublicWidth <= column.val := Nat.not_lt.mp isPublic
    have mapped : alignedIndex dimensions column =
        ⟨column.val + fixedPaddingWidth, by
          have columnBound := column.isLt
          simp only [Dimensions.alignedLogicalWidth]
          omega⟩ := by
      apply Fin.ext
      exact alignedIndex_private dimensions column isPrivate
    have notPublicAfterShift :
        ¬ column.val + fixedPaddingWidth < legacyPublicWidth := by
      simp only [legacyPublicWidth, fixedPaddingWidth] at isPrivate |- 
      omega
    have notPaddingAfterShift :
        ¬ column.val + fixedPaddingWidth < alignedPublicWidth := by
      simp only [legacyPublicWidth, fixedPaddingWidth, alignedPublicWidth,
        publicRingColumns, ringDegree] at isPrivate |- 
      omega
    rw [mapped]
    unfold alignedLogicalAssignment
    rw [dif_neg notPublicAfterShift, dif_neg notPaddingAfterShift]
    apply congrArg legacy
    apply Fin.ext
    simp [fixedPaddingWidth]

/-- Logical coordinate of one of the thirteen fixed public zeros. -/
def paddingLogicalColumn (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth) :
    Fin dimensions.alignedLogicalWidth :=
  ⟨legacyPublicWidth + offset.val, by
    have legacyFits := dimensions.legacyPublicFits
    have offsetBound := offset.isLt
    simp only [Dimensions.alignedLogicalWidth, legacyPublicWidth,
      fixedPaddingWidth] at legacyFits offsetBound |- 
    omega⟩

@[simp] theorem paddingLogicalColumn_val (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth) :
    (paddingLogicalColumn dimensions offset).val =
      legacyPublicWidth + offset.val := by
  rfl

theorem alignedLogicalAssignment_padding_zero
    (dimensions : Dimensions) (legacy : LegacyAssignment dimensions)
    (offset : Fin fixedPaddingWidth) :
    alignedLogicalAssignment dimensions legacy
      (paddingLogicalColumn dimensions offset) = 0 := by
  have offsetBound := offset.isLt
  simp only [fixedPaddingWidth] at offsetBound
  have notPublic : ¬ 257 + offset.val < 257 := by omega
  have isPadding : 257 + offset.val < 270 := by omega
  simp [alignedLogicalAssignment, paddingLogicalColumn, legacyPublicWidth,
    alignedPublicWidth, publicRingColumns, ringDegree, notPublic, isPadding]

/-- Complete fresh assignment consumed by the typed Phi81 relation. -/
def assignment (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : Assignment dimensions.shape :=
  Phi81CarrierLayout.extendAssignment 0
    (alignedLogicalAssignment dimensions legacy)

/-- Complete-carrier location of an old logical column. -/
def alignedCarrierIndex (dimensions : Dimensions)
    (column : Fin dimensions.legacyLogicalWidth) :
    Fin dimensions.shape.carrierWidth :=
  Phi81CarrierLayout.embedLogical (alignedIndex dimensions column)

theorem assignment_at_alignedIndex (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (column : Fin dimensions.legacyLogicalWidth) :
    assignment dimensions legacy (alignedCarrierIndex dimensions column) =
      legacy column := by
  rw [assignment, alignedCarrierIndex,
    Phi81CarrierLayout.extendAssignment_embedLogical]
  exact alignedLogicalAssignment_at_alignedIndex dimensions legacy column

theorem assignment_private_shift (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (column : Fin dimensions.legacyLogicalWidth)
    (isPrivate : legacyPublicWidth <= column.val) :
    (alignedCarrierIndex dimensions column).val =
        column.val + fixedPaddingWidth /\
      assignment dimensions legacy (alignedCarrierIndex dimensions column) =
        legacy column := by
  exact ⟨alignedIndex_private dimensions column isPrivate,
    assignment_at_alignedIndex dimensions legacy column⟩

/-- Complete-carrier location of one fixed public-padding coordinate. -/
def paddingCarrierColumn (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth) :
    Fin dimensions.shape.carrierWidth :=
  Phi81CarrierLayout.embedLogical (paddingLogicalColumn dimensions offset)

/-- Retained fresh-input obligation: every inserted public coordinate is zero. -/
def FixedPublicPadding (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape) : Prop :=
  forall offset,
    candidate (paddingCarrierColumn dimensions offset) = 0

theorem assignment_fixedPublicPadding (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    FixedPublicPadding dimensions (assignment dimensions legacy) := by
  intro offset
  rw [assignment, paddingCarrierColumn,
    Phi81CarrierLayout.extendAssignment_embedLogical]
  exact alignedLogicalAssignment_padding_zero dimensions legacy offset

theorem assignment_completion_zero (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (column : Fin dimensions.shape.carrierWidth)
    (isCompletion : dimensions.alignedLogicalWidth <= column.val) :
    assignment dimensions legacy column = 0 := by
  exact Phi81CarrierLayout.extendAssignment_tail_zero 0
    (alignedLogicalAssignment dimensions legacy) column isCompletion

/-- Aligned logical coordinate corresponding to a typed public coordinate. -/
def publicLogicalColumn (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) :
    Fin dimensions.alignedLogicalWidth :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt
    dimensions.alignedPublicFitsLogical⟩

theorem publicColumn_eq_embedLogical (dimensions : Dimensions)
    (column : Fin dimensions.shape.publicWidth) :
    dimensions.shape.publicColumn column =
      Phi81CarrierLayout.embedLogical
        (publicLogicalColumn dimensions column) := by
  apply Fin.ext
  rfl

/-- Exact typed public input: the legacy 257-field prefix followed by thirteen
definitionally fixed zeros. -/
def expectedPublicInput (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) : PublicInput dimensions.shape :=
  fun column =>
    if isLegacy : column.val < legacyPublicWidth then
      legacy ⟨column.val,
        Nat.lt_of_lt_of_le isLegacy dimensions.legacyPublicFits⟩
    else
      0

theorem projectPublicInput_exact (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    projectPublicInput (assignment dimensions legacy) =
      expectedPublicInput dimensions legacy := by
  funext column
  rw [projectPublicInput, publicColumn_eq_embedLogical]
  change Phi81CarrierLayout.extendAssignment 0
      (alignedLogicalAssignment dimensions legacy)
      (Phi81CarrierLayout.embedLogical
        (publicLogicalColumn dimensions column)) =
    expectedPublicInput dimensions legacy column
  rw [Phi81CarrierLayout.extendAssignment_embedLogical]
  have columnBound : column.val < alignedPublicWidth := by
    have widthEquality :
        dimensions.shape.publicWidth = alignedPublicWidth := by
      simp [alignedPublicWidth, publicRingColumns, ringDegree]
    rw [← widthEquality]
    exact column.isLt
  by_cases isLegacy : column.val < legacyPublicWidth
  · simp [alignedLogicalAssignment, publicLogicalColumn,
      expectedPublicInput, isLegacy]
  · simp [alignedLogicalAssignment, publicLogicalColumn,
      expectedPublicInput, isLegacy, columnBound]

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
