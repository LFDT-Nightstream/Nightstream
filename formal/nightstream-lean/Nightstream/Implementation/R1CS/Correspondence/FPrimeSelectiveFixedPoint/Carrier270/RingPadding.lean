import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.RingPadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding

/-!
Semantic refinement of the bounded fixed-point final ring-padding rows.

Owns: fail-closed decoding of the exact 52 generated rows, their coefficient
shape, the residual `-(z[0] * z[11_725_454 + i])`, soundness under the
separately owned constant-one invariant, and honest zero extension.

Does not own: the earlier 38-coordinate private-alignment interval,
constant-one authority, decoding of non-padding coordinates, CCS/CE
membership, commitment alignment, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.ring_padding.decode` | every exact raw row decodes | checked |
| `f_prime.fixed_point.ring_padding.residual` | decoded equation is `-(z0*zpad)` | derived |
| `f_prime.fixed_point.ring_padding.zero` | active row iff named padding value is zero | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPaddingRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement

namespace Artifact

abbrev expectedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow
abbrev paddingWidth :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.paddingWidth
abbrev relationColumns :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationColumns
abbrev relationRows :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationRows
abbrev firstEmittedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.firstEmittedRow
abbrev firstPaddingColumn :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.firstPaddingColumn
abbrev emitterRunIndex :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.emitterRunIndex

end Artifact

def constantColumn : Fin Artifact.relationColumns :=
  ⟨0, by decide⟩

def paddingColumn (offset : Fin Artifact.paddingWidth) :
    Fin Artifact.relationColumns :=
  ⟨Artifact.firstPaddingColumn + offset.val,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow_paddingColumn
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
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow_emittedRow_bound
          offset⟩
    runIndex := Artifact.emitterRunIndex
    family := .ringPadding
    arm := none
    ports := expectedDecodedPort offset }

/-- Pointwise kernel decoding lemma. Its input is one symbolic row, not the
52-row generated list. -/
theorem expectedRow_decode_exact (offset : Fin Artifact.paddingWidth) :
    decodeRow (Artifact.expectedRow offset.val) =
      some (expectedDecodedRow offset) := by
  have rowBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow_emittedRow_bound
      offset
  have concreteRowBound :
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.firstEmittedRow +
          offset.val < 14946911 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationRows]
      using rowBound
  have paddingBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow_paddingColumn
      offset
  have concretePaddingBound : 11725454 + offset.val < 11725506 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.firstPaddingColumn,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.unpaddedColumns,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationColumns]
      using paddingBound
  have decodedPaddingPort :=
    decodePort_unit 11725506 (11725454 + offset.val)
      concretePaddingBound
  simp [Artifact.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedPorts,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.expectedPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.unitPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.emptyPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationRows,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.relationColumns,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.unpaddedColumns,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.firstPaddingColumn,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.constantColumn,
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

theorem expectedRow_residual_eq (offset : Fin Artifact.paddingWidth)
    (assignment : Fin Artifact.relationColumns → F) :
    residual (expectedDecodedRow offset) assignment =
      -(assignment constantColumn * assignment (paddingColumn offset)) := by
  exact residual_eq_neg_product (expectedDecodedRow offset)
    (validatedExpectedRow offset) assignment

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

/-- Zero exactly the final ring-alignment interval
`[11_725_454, 11_725_506)`. -/
def withRingPaddingZero
    (assignment : Fin Artifact.relationColumns → F) :
    Fin Artifact.relationColumns → F :=
  fun column =>
    if Artifact.firstPaddingColumn ≤ column.val then 0 else assignment column

theorem withRingPaddingZero_at_padding
    (assignment : Fin Artifact.relationColumns → F)
    (offset : Fin Artifact.paddingWidth) :
    withRingPaddingZero assignment (paddingColumn offset) = 0 := by
  simp [withRingPaddingZero, paddingColumn, Artifact.firstPaddingColumn]

theorem withRingPaddingZero_satisfies
    (assignment : Fin Artifact.relationColumns → F) :
    GeneratedRowsSatisfied (withRingPaddingZero assignment) := by
  intro offset
  rw [RowSatisfied, expectedRow_residual_eq,
    withRingPaddingZero_at_padding]
  rw [Fin.mul_zero]
  rfl

/-- Every generated raw record has one semantic owner and decodes to its
coefficient-derived zero equation. -/
theorem generated_raw_row_refines {raw :
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.RawRow}
    (member : raw ∈
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.rawRows) :
    ∃ offset : Fin Artifact.paddingWidth,
      raw = Artifact.expectedRow offset.val ∧
        decodeRow raw = some (expectedDecodedRow offset) ∧
        ∀ assignment : Fin Artifact.relationColumns → F,
          assignment constantColumn = 1 →
          (RowSatisfied (expectedDecodedRow offset) assignment ↔
            assignment (paddingColumn offset) = 0) := by
  rcases
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPadding.generated_row_has_unique_offset
        member with
    ⟨offset, rawExact, _unique⟩
  refine ⟨offset, rawExact, ?_, ?_⟩
  · rw [rawExact]
    exact expectedRow_decode_exact offset
  · intro assignment constantOne
    exact expectedRow_satisfied_iff_padding_zero offset assignment constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPaddingRefinement
