import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.SelectiveLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.ArtifactRefinement

/-!
Artifact-checked selective-compiler bridge for the 270-coordinate public
carrier slice.

Owns: exact equality between the Rust-exported public layout and the
independent `SelectiveLayout`; fail-closed decoding and coefficient
classification of all thirteen public-padding rows; and their local
soundness/completeness bridge to typed `FPrimeCarrier270` zero-pin semantics.

Does not own: the complete fixed-point F′ relation, private-column decoding,
matrix/assignment refinement beyond the public prefix, Pi_CCS/CE membership,
Ajtai commitments, NIFS soundness, or permission to remove rows.

Emits constraints: no. It interprets exact emitted rows.

| Stage path | Mathematical obligation | Artifact owner | Lean result |
|---|---|---|---|
| `f_prime.compiler.public.layout` | `257 + 13 = 270`; selectors begin at 270; private branch begins at 311 | generated prepared-layout values | `generated_layout_refines_model` |
| `f_prime.selective_ccs.padding.rows` | exactly thirteen decoded rows at columns `257+i` | generated final matrices | `generated_padding_rows_shape` |
| `f_prime.selective_ccs.padding.refinement` | row residual vanishes iff the typed carrier zero pin holds | decoded coefficients plus explicit prefix agreement | `generated_padding_row_iff_zeroPin` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.ArtifactRefinement
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout

/-- Exact generated layout values, stated against the independent formulas
rather than a second handwritten list of expected numbers. -/
theorem generated_layout_refines_model :
    logicalPublicInputLen = legacyPublicWidth ∧
      publicInputLen = alignedPublicWidth ∧
      publicPaddingColumns =
        List.ofFn (fun offset : Fin fixedPaddingWidth =>
          publicPaddingStart + offset.val) ∧
      selectorColumns = List.ofFn selectorColumn ∧
      privateAlignmentPaddingColumns =
        List.ofFn (fun offset : Fin privateAlignmentPaddingWidth =>
          privateAlignmentStart + offset.val) ∧
      sharedPrivateStart = branchPrivateStart ∧
      sharedPrivateEnd = branchPrivateStart ∧
      branchStart = branchPrivateStart ∧
      branchEnd = branchPrivateStart ∧
      ringAlignmentPaddingStart = branchPrivateStart ∧
      ringAlignmentPaddingEnd = 324 := by
  native_decide

/-- Decode every generated row fail-closed. An empty result would make the
subsequent exact-count and shape theorems false. -/
def decodedPaddingRows : List DecodedRow :=
  (rawPaddingRows.mapM decodeRow).getD []

theorem decodedPaddingRows_length :
    decodedPaddingRows.length = fixedPaddingWidth := by
  native_decide

/-- Generated row corresponding to one typed padding offset. -/
def decodedPaddingRow (offset : Fin fixedPaddingWidth) : DecodedRow :=
  decodedPaddingRows.get ⟨offset.val, by
    rw [decodedPaddingRows_length]
    exact offset.isLt⟩

theorem decodedPaddingRow_columns :
    ∀ offset : Fin fixedPaddingWidth,
      (decodedPaddingRow offset).columns = 324 := by
  native_decide

/-- Production encoded column carrying the verifier-owned constant. -/
def encodedConstantColumn (offset : Fin fixedPaddingWidth) :
    Fin (decodedPaddingRow offset).columns :=
  ⟨0, by
    have columns := decodedPaddingRow_columns offset
    omega⟩

/-- Production encoded column carrying one fixed public zero. -/
def encodedPaddingColumn (offset : Fin fixedPaddingWidth) :
    Fin (decodedPaddingRow offset).columns :=
  ⟨legacyPublicWidth + offset.val, by
    have columns := decodedPaddingRow_columns offset
    have bound := offset.isLt
    simp only [legacyPublicWidth, fixedPaddingWidth] at bound ⊢
    omega⟩

/-- Metadata and coefficient classification for every one of the thirteen
physical rows. The family tag is checked separately from the sparse shape and
therefore is not semantic authority. -/
theorem generated_padding_rows_shape :
    ∀ offset : Fin fixedPaddingWidth,
      (decodedPaddingRow offset).rows = 836 ∧
      (decodedPaddingRow offset).columns = 324 ∧
      (decodedPaddingRow offset).emittedRow.val = 4 + offset.val ∧
      (decodedPaddingRow offset).runIndex = 6 ∧
      (decodedPaddingRow offset).family = .publicPadding ∧
      (decodedPaddingRow offset).arm = none ∧
      (encodedConstantColumn offset).val = 0 ∧
      (encodedPaddingColumn offset).val = 257 + offset.val ∧
      IsPaddingAt (decodedPaddingRow offset)
        (encodedConstantColumn offset) (encodedPaddingColumn offset) := by
  native_decide

def validatedPaddingRow (offset : Fin fixedPaddingWidth) :
    ValidatedPaddingRow (decodedPaddingRow offset) where
  constantColumn := encodedConstantColumn offset
  paddingColumn := encodedPaddingColumn offset
  shape := (generated_padding_rows_shape offset).2.2.2.2.2.2.2.2

theorem encodedPaddingColumn_matches_typedCarrier
    (dimensions : Dimensions) (offset : Fin fixedPaddingWidth) :
    (encodedPaddingColumn offset).val =
      (paddingCarrierColumn dimensions offset).val := by
  rfl

/-- Explicit local connectivity premise between the production encoded
prefix and the independent typed relation carrier. -/
def PublicPrefixAgrees
    (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth)
    (encoded : Fin (decodedPaddingRow offset).columns → F)
    (candidate : Assignment dimensions.shape) : Prop :=
  encoded (encodedConstantColumn offset) =
      candidate (constantColumn dimensions) ∧
    encoded (encodedPaddingColumn offset) =
      candidate (paddingCarrierColumn dimensions offset)

/-- Exact local soundness and completeness of one generated row for the
independent typed zero-pin predicate. -/
theorem generated_padding_row_iff_zeroPin
    (dimensions : Dimensions)
    (offset : Fin fixedPaddingWidth)
    (encoded : Fin (decodedPaddingRow offset).columns → F)
    (candidate : Assignment dimensions.shape)
    (agrees : PublicPrefixAgrees dimensions offset encoded candidate) :
    residual (decodedPaddingRow offset) encoded = 0 ↔
      ZeroPinHolds dimensions candidate offset := by
  rw [residual_eq_neg_product
    (decodedPaddingRow offset) (validatedPaddingRow offset) encoded]
  unfold ZeroPinHolds zeroPinProduct
  change
    -(encoded (encodedConstantColumn offset) *
        encoded (encodedPaddingColumn offset)) = 0 ↔
      candidate (constantColumn dimensions) *
        candidate (paddingCarrierColumn dimensions offset) = 0
  rw [agrees.1, agrees.2]
  constructor
  · intro negated
    have := congrArg (fun value : F => -value) negated
    simpa only [Lean.Grind.AddCommGroup.neg_neg,
      Lean.Grind.AddCommGroup.neg_zero] using this
  · intro zeroPin
    rw [zeroPin]
    rfl

/-- Honest canonical carrier completion satisfies each exact generated row,
provided the encoded prefix is connected to that canonical carrier. -/
theorem generated_padding_row_canonical_complete
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (offset : Fin fixedPaddingWidth)
    (encoded : Fin (decodedPaddingRow offset).columns → F)
    (agrees : PublicPrefixAgrees dimensions offset encoded
      (assignment dimensions legacy)) :
    residual (decodedPaddingRow offset) encoded = 0 := by
  exact (generated_padding_row_iff_zeroPin dimensions offset encoded
    (assignment dimensions legacy) agrees).2
      (canonicalAssignment_complete dimensions legacy offset)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ProductionCarrier
