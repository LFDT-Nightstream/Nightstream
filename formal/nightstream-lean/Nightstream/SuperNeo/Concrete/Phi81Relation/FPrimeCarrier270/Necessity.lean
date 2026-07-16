import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Necessity of the thirteen fixed zeros in the five-ring F' public carrier.

Protocol: SuperNeo CCS/CE fresh-input boundary for F'.
Phase: retained public-padding obligation.
Constraint family: semantic necessity witness only; this file emits no rows.

Owns: a boundary language that preserves the legacy 257-field public view and
the strict `b = 2` assignment norm; an exact candidate with value one at the
first inserted coordinate; and a kernel-checked proof that this candidate is
accepted by the relaxed boundary but rejected by fixed public padding.

Does not own: CCS satisfaction, CE evaluations, commitment binding, Ajtai,
PiCCS, NIFS, Rust/R1CS acceptance, or a security reduction. In particular,
"fresh boundary" below is not the complete CCS language.

Emits constraints: no.

Authority boundary: strict norm does not imply that an inserted public
coordinate is zero. Value one is norm-valid at `b = 2`, so the zero condition
must remain an independent verifier-owned obligation.

| Protocol | Phase | Family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| F' / CCS | fresh boundary | legacy public view | first 257 values remain unchanged | `SameLegacyPublic` |
| F' / CCS | fresh boundary | strict norm | every complete-carrier value has magnitude below 2 | `RelaxedFreshBoundary` |
| F' / CCS | fresh boundary | fixed public padding | all thirteen inserted values equal zero | `FixedFreshBoundary` |
| assurance | necessity | first padding value | value one preserves view and norm but violates padding | `tailOne_relaxedFreshBoundary`, `tailOne_not_fixedPublicPadding` |
| assurance | necessity | language inclusion | removing padding strictly admits an extra boundary member | `omittingFixedPadding_enlargesFreshBoundary` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Embed one coordinate of the legacy 257-field public view into the legacy
logical assignment. -/
def legacyPublicLegacyColumn (dimensions : Dimensions)
    (column : Fin legacyPublicWidth) :
    Fin dimensions.legacyLogicalWidth :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt dimensions.legacyPublicFits⟩

/-- The same legacy public coordinate in the complete aligned carrier. -/
def legacyPublicCarrierColumn (dimensions : Dimensions)
    (column : Fin legacyPublicWidth) :
    Fin dimensions.shape.carrierWidth :=
  alignedCarrierIndex dimensions (legacyPublicLegacyColumn dimensions column)

@[simp] theorem legacyPublicCarrierColumn_val (dimensions : Dimensions)
    (column : Fin legacyPublicWidth) :
    (legacyPublicCarrierColumn dimensions column).val = column.val := by
  change (alignedIndex dimensions
    (legacyPublicLegacyColumn dimensions column)).val = column.val
  exact alignedIndex_public dimensions
    (legacyPublicLegacyColumn dimensions column) column.isLt

/-- First coordinate in the verifier-fixed thirteen-value padding block. -/
def firstPaddingOffset : Fin fixedPaddingWidth := ⟨0, by decide⟩

@[simp] theorem firstPaddingCarrierColumn_val (dimensions : Dimensions) :
    (paddingCarrierColumn dimensions firstPaddingOffset).val =
      legacyPublicWidth := by
  rfl

/-- Candidate assignments expose the same legacy public view. This predicate
deliberately does not constrain the thirteen new public coordinates. -/
def SameLegacyPublic (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (candidate : Assignment dimensions.shape) : Prop :=
  forall column,
    candidate (legacyPublicCarrierColumn dimensions column) =
      legacy (legacyPublicLegacyColumn dimensions column)

/-- Boundary obtained if the fixed-zero obligation is omitted. -/
def RelaxedFreshBoundary (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (candidate : Assignment dimensions.shape) : Prop :=
  SameLegacyPublic dimensions legacy candidate /\
    assignmentNormBounded 2 candidate

/-- Intended boundary: the relaxed obligations plus all thirteen fixed zeros. -/
def FixedFreshBoundary (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (candidate : Assignment dimensions.shape) : Prop :=
  RelaxedFreshBoundary dimensions legacy candidate /\
    FixedPublicPadding dimensions candidate

theorem fixedFreshBoundary_implies_relaxed (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (candidate : Assignment dimensions.shape) :
    FixedFreshBoundary dimensions legacy candidate ->
      RelaxedFreshBoundary dimensions legacy candidate := by
  exact fun accepted => accepted.1

/-- All-zero legacy assignment used only to isolate padding necessity. -/
def zeroLegacyAssignment (dimensions : Dimensions) :
    LegacyAssignment dimensions :=
  fun _ => 0

/-- Exact necessity witness: only the first fixed-padding coordinate is one. -/
def tailOneAssignment (dimensions : Dimensions) :
    Assignment dimensions.shape :=
  fun column =>
    if column = paddingCarrierColumn dimensions firstPaddingOffset then 1 else 0

theorem tailOne_sameLegacyPublic (dimensions : Dimensions) :
    SameLegacyPublic dimensions (zeroLegacyAssignment dimensions)
      (tailOneAssignment dimensions) := by
  intro column
  have distinct :
      legacyPublicCarrierColumn dimensions column ≠
        paddingCarrierColumn dimensions firstPaddingOffset := by
    intro equal
    have values := congrArg Fin.val equal
    simp only [legacyPublicCarrierColumn_val,
      firstPaddingCarrierColumn_val] at values
    exact Nat.ne_of_lt column.isLt values
  simp [tailOneAssignment, distinct, zeroLegacyAssignment]

/-- Value one and value zero both satisfy the strict `b = 2` norm. -/
theorem tailOne_normBounded (dimensions : Dimensions) :
    assignmentNormBounded 2 (tailOneAssignment dimensions) := by
  intro column
  by_cases isTail :
      column = paddingCarrierColumn dimensions firstPaddingOffset
  · rw [show tailOneAssignment dimensions column = (1 : F) by
      simp [tailOneAssignment, isTail]]
    decide
  · rw [show tailOneAssignment dimensions column = (0 : F) by
      simp [tailOneAssignment, isTail]]
    decide

theorem tailOne_relaxedFreshBoundary (dimensions : Dimensions) :
    RelaxedFreshBoundary dimensions (zeroLegacyAssignment dimensions)
      (tailOneAssignment dimensions) := by
  exact ⟨tailOne_sameLegacyPublic dimensions,
    tailOne_normBounded dimensions⟩

theorem tailOne_not_fixedPublicPadding (dimensions : Dimensions) :
    ¬ FixedPublicPadding dimensions (tailOneAssignment dimensions) := by
  intro fixed
  have first := fixed firstPaddingOffset
  have oneEqualsZero : (1 : F) = 0 := by
    simpa [tailOneAssignment] using first
  exact (by decide : (1 : F) ≠ 0) oneEqualsZero

/-- Model-level necessity result. Omitting the fixed-zero obligation strictly
admits a norm-valid carrier with the same 257-field legacy public view. -/
theorem omittingFixedPadding_enlargesFreshBoundary
    (dimensions : Dimensions) :
    exists candidate,
      RelaxedFreshBoundary dimensions (zeroLegacyAssignment dimensions)
          candidate /\
        ¬ FixedFreshBoundary dimensions (zeroLegacyAssignment dimensions)
          candidate := by
  refine ⟨tailOneAssignment dimensions,
    tailOne_relaxedFreshBoundary dimensions, ?_⟩
  intro fixed
  exact tailOne_not_fixedPublicPadding dimensions fixed.2

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
