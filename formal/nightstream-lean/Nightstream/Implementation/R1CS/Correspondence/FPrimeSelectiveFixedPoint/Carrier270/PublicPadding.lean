import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Semantic refinement of the bounded fixed-point public-padding rows.

Owns: fail-closed decoding of the exact 13 generated rows; coefficient-based
classification independent of family labels; the exact residual
`-(z[0] * z[257 + i])`; soundness under constant-one; honest zero extension;
and equality with the independent `FixedPublicPadding` obligation under an
explicit assignment-coordinate agreement.

Does not own: authority for the constant-one coordinate, decoding of private
columns, complete matrix/assignment refinement, CCS/CE membership,
commitment alignment, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.public_padding.decode` | every exact raw row decodes | checked |
| `f_prime.fixed_point.public_padding.residual` | decoded equation is `-(z0*zpad)` | derived |
| `f_prime.fixed_point.public_padding.typed` | 13 physical zeros equal `FixedPublicPadding` | direct dataflow plus derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement

namespace Artifact

abbrev expectedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow
abbrev paddingWidth :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth
abbrev relationColumns :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationColumns
abbrev relationRows :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationRows
abbrev firstEmittedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstEmittedRow
abbrev emitterRunIndex :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.emitterRunIndex
abbrev firstPaddingColumn :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstPaddingColumn

end Artifact

def constantColumn : Fin Artifact.relationColumns :=
  ⟨0, by decide⟩

def paddingColumn (offset : Fin Artifact.paddingWidth) :
    Fin Artifact.relationColumns :=
  ⟨Artifact.firstPaddingColumn + offset.val,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow_paddingColumn
      offset⟩

def expectedDecodedPort (offset : Fin Artifact.paddingWidth)
    (port : Fin 13) : DecodedPort Artifact.relationColumns :=
  if port.val = 1 then
    unitDecodedPort Artifact.relationColumns constantColumn.val constantColumn.isLt
  else if port.val = 4 then
    unitDecodedPort Artifact.relationColumns (paddingColumn offset).val
      (paddingColumn offset).isLt
  else
    emptyDecodedPort Artifact.relationColumns

def expectedDecodedRow (offset : Fin Artifact.paddingWidth) : DecodedRow :=
  { rows := Artifact.relationRows
    columns := Artifact.relationColumns
    rowsPositive := by decide
    columnsPositive := by decide
    emittedRow :=
      ⟨Artifact.firstEmittedRow + offset.val,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow_emittedRow_bound
          offset⟩
    runIndex := Artifact.emitterRunIndex
    family := .publicPadding
    arm := none
    ports := expectedDecodedPort offset }

/-- Pointwise kernel decoding lemma. No generated list or proof-carrying
structure is evaluated here. -/
theorem expectedRow_decode_exact (offset : Fin Artifact.paddingWidth) :
    decodeRow (Artifact.expectedRow offset.val) =
      some (expectedDecodedRow offset) := by
  have rowBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow_emittedRow_bound
      offset
  have concreteRowBound : 5589380 + offset.val < 17669277 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstEmittedRow,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationRows]
      using rowBound
  have paddingBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow_paddingColumn
      offset
  have concretePaddingBound : 257 + offset.val < 14338890 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstPaddingColumn,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationColumns]
      using paddingBound
  have decodedPaddingPort :=
    decodePort_unit 14338890 (257 + offset.val) concretePaddingBound
  simp [Artifact.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedPorts,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.expectedPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.unitPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.emptyPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationRows,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.relationColumns,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstEmittedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.emitterRunIndex,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.constantColumn,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.firstPaddingColumn,
    decodeRow, supportedSchemaVersion, decodedPaddingPort,
    decodePort_empty, decodePort_unit, expectedDecodedRow]
  constructor
  · exact concreteRowBound
  · funext port
    change
      (List.ofFn (expectedDecodedPort offset)).get
          ⟨port.val, by simp⟩ =
        expectedDecodedPort offset port
    rw [List.get_eq_getElem, List.getElem_ofFn]

theorem expectedRow_decodes (offset : Fin Artifact.paddingWidth) :
    ∃ row : DecodedRow,
      decodeRow (Artifact.expectedRow offset.val) = some row :=
  ⟨expectedDecodedRow offset, expectedRow_decode_exact offset⟩

theorem expectedDecodedRow_shape (offset : Fin Artifact.paddingWidth) :
    IsPaddingAt (expectedDecodedRow offset) constantColumn
      (paddingColumn offset) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [expectedDecodedRow, expectedDecodedPort, Role.index,
      DecodedRow.port]
  · change
      expandedFieldTerms (expectedDecodedPort offset (4 : Fin 13)) =
        [(paddingColumn offset, 1)]
    unfold expectedDecodedPort
    rw [if_neg (by decide), if_pos (by decide)]
    exact expandedFieldTerms_unit Artifact.relationColumns
      (paddingColumn offset).val (paddingColumn offset).isLt
  · intro port generalNe cNe
    have notOne : port.val ≠ 1 := by
      intro value
      apply generalNe
      apply Fin.ext
      simpa [Role.index] using value
    have notFour : port.val ≠ 4 := by
      intro value
      apply cNe
      apply Fin.ext
      simpa [Role.index] using value
    simp [expectedDecodedRow, expectedDecodedPort, DecodedRow.port,
      notOne, notFour]

def validatedExpectedRow (offset : Fin Artifact.paddingWidth) :
    ValidatedPaddingRow (expectedDecodedRow offset) :=
  { constantColumn
    paddingColumn := paddingColumn offset
    shape := expectedDecodedRow_shape offset }

/-- Exact semantic equation for one generated public-padding owner. -/
theorem expectedRow_residual_eq (offset : Fin Artifact.paddingWidth)
    (assignment : Fin Artifact.relationColumns → F) :
    residual (expectedDecodedRow offset) assignment =
      -(assignment constantColumn * assignment (paddingColumn offset)) := by
  exact residual_eq_neg_product (expectedDecodedRow offset)
    (validatedExpectedRow offset) assignment

/-- Soundness of one physical public-padding equation under the separately
owned constant-one invariant. -/
theorem expectedRow_satisfied_iff_padding_zero
    (offset : Fin Artifact.paddingWidth)
    (assignment : Fin Artifact.relationColumns → F)
    (constantOne : assignment constantColumn = 1) :
    RowSatisfied (expectedDecodedRow offset) assignment ↔
      assignment (paddingColumn offset) = 0 := by
  exact residual_eq_zero_iff (expectedDecodedRow offset)
    (validatedExpectedRow offset) assignment constantOne

def GeneratedRowsSatisfied
    (assignment : Fin Artifact.relationColumns → F) : Prop :=
  ∀ offset : Fin Artifact.paddingWidth,
    RowSatisfied (expectedDecodedRow offset) assignment

theorem generatedRowsSatisfied_iff_padding_zero
    (assignment : Fin Artifact.relationColumns → F)
    (constantOne : assignment constantColumn = 1) :
    GeneratedRowsSatisfied assignment ↔
      ∀ offset : Fin Artifact.paddingWidth,
        assignment (paddingColumn offset) = 0 := by
  constructor
  · intro satisfied offset
    exact (expectedRow_satisfied_iff_padding_zero offset assignment
      constantOne).1 (satisfied offset)
  · intro paddingZero offset
    exact (expectedRow_satisfied_iff_padding_zero offset assignment
      constantOne).2 (paddingZero offset)

/-- Zero only the generated public-padding interval `[257, 270)`. -/
def withPublicPaddingZero
    (assignment : Fin Artifact.relationColumns → F) :
    Fin Artifact.relationColumns → F :=
  fun column =>
    if Artifact.firstPaddingColumn ≤ column.val ∧
        column.val < Artifact.firstPaddingColumn + Artifact.paddingWidth then
      0
    else
      assignment column

theorem withPublicPaddingZero_at_padding
    (assignment : Fin Artifact.relationColumns → F)
    (offset : Fin Artifact.paddingWidth) :
    withPublicPaddingZero assignment (paddingColumn offset) = 0 := by
  have offsetBound := offset.isLt
  simp [withPublicPaddingZero, paddingColumn, Artifact.firstPaddingColumn,
    Artifact.paddingWidth]

/-- Honest completeness for the exact 13-row physical interval. -/
theorem withPublicPaddingZero_satisfies
    (assignment : Fin Artifact.relationColumns → F) :
    GeneratedRowsSatisfied (withPublicPaddingZero assignment) := by
  intro offset
  rw [RowSatisfied, expectedRow_residual_eq,
    withPublicPaddingZero_at_padding]
  rw [Fin.mul_zero]
  rfl

/-- Convert one artifact offset to the independently typed 13-coordinate
padding domain. -/
def typedOffset (offset : Fin Artifact.paddingWidth) :
    Fin fixedPaddingWidth :=
  ⟨offset.val, by
    simpa [Artifact.paddingWidth,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth,
      fixedPaddingWidth] using offset.isLt⟩

/-- Explicit coordinate agreement between the final selective assignment and
the independent 270-coordinate carrier. No digest can discharge this. -/
def PublicPaddingAgrees
    (dimensions : Dimensions)
    (encoded : Fin Artifact.relationColumns → F)
    (candidate : Assignment dimensions.shape) : Prop :=
  ∀ offset : Fin Artifact.paddingWidth,
    encoded (paddingColumn offset) =
      candidate (paddingCarrierColumn dimensions (typedOffset offset))

/-- The generated rows are exactly the typed fixed-public-padding obligation,
provided the actual assignment decoder identifies their coordinates. -/
theorem generatedRowsSatisfied_iff_fixedPublicPadding
    (dimensions : Dimensions)
    (encoded : Fin Artifact.relationColumns → F)
    (candidate : Assignment dimensions.shape)
    (constantOne : encoded constantColumn = 1)
    (agrees : PublicPaddingAgrees dimensions encoded candidate) :
    GeneratedRowsSatisfied encoded ↔
      FixedPublicPadding dimensions candidate := by
  rw [generatedRowsSatisfied_iff_padding_zero encoded constantOne]
  constructor
  · intro zeros typed
    let artifact : Fin Artifact.paddingWidth :=
      ⟨typed.val, by
        simpa [Artifact.paddingWidth,
          Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth,
          fixedPaddingWidth] using typed.isLt⟩
    have offsetEq : typedOffset artifact = typed := by
      apply Fin.ext
      rfl
    rw [← offsetEq]
    exact (agrees artifact).symm.trans (zeros artifact)
  · intro fixed offset
    exact (agrees offset).trans (fixed (typedOffset offset))

/-- Honest typed carrier assignments satisfy the generated rows whenever the
concrete assignment decoder supplies the explicit coordinate agreement. -/
theorem generatedRowsSatisfied_of_typedAssignment
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (encoded : Fin Artifact.relationColumns → F)
    (constantOne : encoded constantColumn = 1)
    (agrees : PublicPaddingAgrees dimensions encoded
      (assignment dimensions legacy)) :
    GeneratedRowsSatisfied encoded := by
  exact (generatedRowsSatisfied_iff_fixedPublicPadding dimensions encoded
    (assignment dimensions legacy) constantOne agrees).2
      (assignment_fixedPublicPadding dimensions legacy)

/-- Every actual generated raw record has exactly one semantic owner and
decodes to the corresponding equation. -/
theorem generated_raw_row_refines {raw :
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.RawRow}
    (member : raw ∈
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.rawRows) :
    ∃ offset : Fin Artifact.paddingWidth,
      raw = Artifact.expectedRow offset.val ∧
        decodeRow raw = some (expectedDecodedRow offset) ∧
        ∀ assignment : Fin Artifact.relationColumns → F,
          assignment constantColumn = 1 →
          (RowSatisfied (expectedDecodedRow offset) assignment ↔
            assignment (paddingColumn offset) = 0) := by
  rcases
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.generated_row_has_unique_offset
        member with
    ⟨offset, rawExact, _unique⟩
  refine ⟨offset, rawExact, ?_, ?_⟩
  · rw [rawExact]
    exact expectedRow_decode_exact offset
  · intro assignment constantOne
    exact expectedRow_satisfied_iff_padding_zero offset assignment constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement
