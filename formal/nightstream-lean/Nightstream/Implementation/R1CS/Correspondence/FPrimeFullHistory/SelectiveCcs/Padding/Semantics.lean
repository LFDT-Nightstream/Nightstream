import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.SelectiveLayout

/-!
Contract: typed mathematical semantics for a selective-CCS public-padding
row at the repaired F' carrier boundary.

Owns: the normalized zero-set equation for the intended specialization with
`GENERAL_SELECTOR = z[0]`, `C = z[padding]`, and all other ports zero; the
explicit constant-one premise needed to recover a zero pin; and honest
completeness for the canonical `FPrimeCarrier270.assignment` constructor.

Does not own: the complete 13-port sparse polynomial, Rust port numbering,
sparse matrix emission, row ordering, multiplicity, or production
conformance. A raw compiler artifact must establish those facts before this
predicate can discharge an emitted row.

Emits constraints: no. This file specifies one mathematical leaf.

| Stage path | Equation | Authority class | Rust owner | Lean owner | Multiplicity | Removal status |
|---|---|---|---|---|---:|---|
| `f_prime.selective_ccs.padding.zero_pin` | `z[0] * z[257+i] = 0` (normalized from the selective `-C` residual) | checked for raw input | selective CCS zero-row emitter | `zeroPinHolds_iff` | `i < 13` | exact polynomial specialization proved in `Padding.Refinement`; row artifact pending |
| `f_prime.carrier270.fresh.padding` | inserted padding is definitionally zero | computed | fresh carrier encoder | `canonicalAssignment_complete` | `13` | candidate for derivation |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Typed public coordinate containing the relation's constant-one wire. -/
def constantPublicCoordinate (dimensions : Dimensions) :
    Fin dimensions.shape.publicWidth :=
  ⟨0, by rw [Dimensions.shape_publicWidth]; decide⟩

/-- Complete-carrier location of the constant-one wire. -/
def constantColumn (dimensions : Dimensions) :
    Fin dimensions.shape.carrierWidth :=
  dimensions.shape.publicColumn (constantPublicCoordinate dimensions)

@[simp] theorem constantColumn_val (dimensions : Dimensions) :
    (constantColumn dimensions).val = 0 := by
  rfl

theorem constantColumn_ne_padding (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth) :
    constantColumn dimensions ≠ paddingCarrierColumn dimensions offset := by
  intro equal
  have values := congrArg Fin.val equal
  simp [constantColumn, constantPublicCoordinate, paddingCarrierColumn,
    paddingLogicalColumn, Phi81CarrierLayout.embedLogical,
    legacyPublicWidth] at values
  omega

theorem paddingCarrierColumn_injective (dimensions : Dimensions) :
    Function.Injective (paddingCarrierColumn dimensions) := by
  intro left right equal
  have values := congrArg Fin.val equal
  apply Fin.ext
  simp [paddingCarrierColumn, paddingLogicalColumn,
    Phi81CarrierLayout.embedLogical, legacyPublicWidth] at values
  omega

/-- Normalized zero set of the intended selective-polynomial padding
specialization. The exact sparse residual has the opposite sign;
`Padding.Refinement` proves that specialization without changing this file's
semantic ownership. -/
def zeroPinProduct (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (offset : Fin fixedPaddingWidth) : F :=
  candidate (constantColumn dimensions) *
    candidate (paddingCarrierColumn dimensions offset)

def ZeroPinHolds (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (offset : Fin fixedPaddingWidth) : Prop :=
  zeroPinProduct dimensions candidate offset = 0

def AllZeroPinsHold (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape) : Prop :=
  ∀ offset, ZeroPinHolds dimensions candidate offset

/-- A zero-row is a zero pin only under explicit constant-one authority. -/
theorem zeroPinHolds_iff (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (constantOne : candidate (constantColumn dimensions) = 1)
    (offset : Fin fixedPaddingWidth) :
    ZeroPinHolds dimensions candidate offset ↔
      candidate (paddingCarrierColumn dimensions offset) = 0 := by
  unfold ZeroPinHolds zeroPinProduct
  rw [constantOne, Fin.one_mul]

theorem allZeroPinsHold_iff_fixedPublicPadding (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (constantOne : candidate (constantColumn dimensions) = 1) :
    AllZeroPinsHold dimensions candidate ↔
      FixedPublicPadding dimensions candidate := by
  constructor
  · intro rows offset
    exact (zeroPinHolds_iff dimensions candidate constantOne offset).mp
      (rows offset)
  · intro padding offset
    exact (zeroPinHolds_iff dimensions candidate constantOne offset).mpr
      (padding offset)

/-- The canonical typed fresh carrier satisfies every intended padding row by
construction, independently of the value in its constant coordinate. -/
theorem canonicalAssignment_complete (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) :
    AllZeroPinsHold dimensions (assignment dimensions legacy) := by
  intro offset
  unfold ZeroPinHolds zeroPinProduct
  rw [assignment_fixedPublicPadding dimensions legacy offset, Fin.mul_zero]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics
