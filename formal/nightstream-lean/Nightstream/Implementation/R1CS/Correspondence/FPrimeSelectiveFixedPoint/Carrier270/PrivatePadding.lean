import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Refinement

/-!
Semantic refinement of the bounded fixed-point private-alignment padding
rows.

Owns: fail-closed decoding of the exact 38 generated rows; coefficient-based
classification independent of family labels; the exact residual
`-(z[0] * z[273 + i])`; soundness under constant-one; and an honest zero
extension.

Does not own: authority for the constant-one coordinate, ownership of later
private columns, complete matrix/assignment decoding, CCS/CE membership,
commitment alignment, or row removal.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.private_padding.decode` | every exact raw row decodes | checked |
| `f_prime.fixed_point.private_padding.residual` | decoded equation is `-(z0*zpad)` | derived |
| `f_prime.fixed_point.private_padding.zero` | active row iff named padding value is zero | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

namespace Artifact

abbrev expectedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow
abbrev paddingWidth :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.paddingWidth
abbrev relationColumns :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationColumns
abbrev relationRows :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationRows
abbrev firstEmittedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstEmittedRow
abbrev emitterRunIndex :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.emitterRunIndex
abbrev firstPaddingColumn :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstPaddingColumn

end Artifact

/-- Exact coefficient shape of one compact private-padding row. Diagnostic
family metadata is deliberately excluded. -/
def IsPaddingAt (row : DecodedRow)
    (constantColumn paddingColumn : Fin row.columns) : Prop :=
  expandedFieldTerms (row.port Role.generalSelector.index) =
      [(constantColumn, 1)] ∧
    expandedFieldTerms (row.port Role.c.index) =
      [(paddingColumn, 1)] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.c.index →
      expandedFieldTerms (row.port port) = []

instance (row : DecodedRow)
    (constantColumn paddingColumn : Fin row.columns) :
    Decidable (IsPaddingAt row constantColumn paddingColumn) := by
  unfold IsPaddingAt
  infer_instance

structure ValidatedPaddingRow (row : DecodedRow) where
  constantColumn : Fin row.columns
  paddingColumn : Fin row.columns
  shape : IsPaddingAt row constantColumn paddingColumn

private theorem action_unit (row : DecodedRow) (port : Fin 13)
    (column : Fin row.columns)
    (shape : expandedFieldTerms (row.port port) = [(column, 1)])
    (assignment : Fin row.columns → F) :
    action (row.port port) assignment = assignment column := by
  simp only [action, shape, List.foldl_cons, List.foldl_nil,
    Fin.zero_add]
  exact Fin.one_mul _

private theorem action_empty (row : DecodedRow) (port : Fin 13)
    (shape : expandedFieldTerms (row.port port) = [])
    (assignment : Fin row.columns → F) :
    action (row.port port) assignment = 0 := by
  simp [action, shape]

theorem rowPoint_eq_paddingPortPoint
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      paddingPortPoint
        (assignment validated.constantColumn)
        (assignment validated.paddingColumn) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint]
    rw [action_unit row Role.generalSelector.index
      validated.constantColumn validated.shape.1 assignment]
    simp [paddingPortPoint, Role.index]
  · by_cases cPort : port = Role.c.index
    · subst port
      simp only [rowPoint]
      rw [action_unit row Role.c.index validated.paddingColumn
        validated.shape.2.1 assignment]
      simp [paddingPortPoint, Role.index]
    · simp only [rowPoint]
      rw [action_empty row port
        (validated.shape.2.2 port generalPort cPort) assignment]
      have generalPortValue : port ≠ (1 : Fin 13) := by
        intro equal
        apply generalPort
        simpa [Role.index] using equal
      have cPortValue : port ≠ (4 : Fin 13) := by
        intro equal
        apply cPort
        simpa [Role.index] using equal
      simp [paddingPortPoint, generalPortValue, cPortValue]

theorem residual_eq_neg_product
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      -(assignment validated.constantColumn *
        assignment validated.paddingColumn) := by
  rw [residual,
    rowPoint_eq_paddingPortPoint row validated assignment,
    evaluate_paddingPortPoint]

theorem residual_eq_zero_iff
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F)
    (constantOne : assignment validated.constantColumn = 1) :
    residual row assignment = 0 ↔
      assignment validated.paddingColumn = 0 := by
  rw [residual_eq_neg_product row validated assignment, constantOne,
    Fin.one_mul]
  constructor
  · intro negated
    have bothSides := congrArg (fun value : F => -value) negated
    simpa only [Lean.Grind.AddCommGroup.neg_neg,
      Lean.Grind.AddCommGroup.neg_zero] using bothSides
  · intro paddingZero
    rw [paddingZero]
    rfl

def constantColumn : Fin Artifact.relationColumns :=
  ⟨0, by decide⟩

def paddingColumn (offset : Fin Artifact.paddingWidth) :
    Fin Artifact.relationColumns :=
  ⟨Artifact.firstPaddingColumn + offset.val,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow_paddingColumn
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
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow_emittedRow_bound
          offset⟩
    runIndex := Artifact.emitterRunIndex
    family := .privatePadding
    arm := none
    ports := expectedDecodedPort offset }

/-- Pointwise kernel decoding lemma. Its input is one symbolic row, not the
38-row generated list, and no proof-carrying structure is passed to
`native_decide`. -/
theorem expectedRow_decode_exact (offset : Fin Artifact.paddingWidth) :
    decodeRow (Artifact.expectedRow offset.val) =
      some (expectedDecodedRow offset) := by
  have rowBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow_emittedRow_bound
      offset
  have concreteRowBound : 4729593 + offset.val < 14946911 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstEmittedRow,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationRows]
      using rowBound
  have paddingBound :=
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow_paddingColumn
      offset
  have concretePaddingBound : 273 + offset.val < 11725506 := by
    simpa [
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstPaddingColumn,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationColumns]
      using paddingBound
  have decodedPaddingPort :=
    decodePort_unit 11725506 (273 + offset.val) concretePaddingBound
  simp [Artifact.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedPorts,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.expectedPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.unitPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.emptyPort,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationRows,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.relationColumns,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstEmittedRow,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.emitterRunIndex,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.constantColumn,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.firstPaddingColumn,
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

@[simp] theorem expandedFieldTerms_empty (columns : Nat) :
    expandedFieldTerms (emptyDecodedPort columns) = [] := by
  rfl

@[simp] theorem expandedFieldTerms_unit (columns column : Nat)
    (columnInRange : column < columns) :
    expandedFieldTerms (unitDecodedPort columns column columnInRange) =
      [(⟨column, columnInRange⟩, 1)] := by
  simp [expandedFieldTerms, unitDecodedPort, unitDecodedTerm,
    termAsFieldTerm]

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

/-- Exact semantic equation for one generated private-padding owner. -/
theorem expectedRow_residual_eq (offset : Fin Artifact.paddingWidth)
    (assignment : Fin Artifact.relationColumns → F) :
    residual (expectedDecodedRow offset) assignment =
      -(assignment constantColumn * assignment (paddingColumn offset)) := by
  exact residual_eq_neg_product (expectedDecodedRow offset)
    (validatedExpectedRow offset) assignment

/-- Soundness of the physical private-padding equation under the separately
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

/-- Zero only the prepared private-alignment interval `[273, 311)`. -/
def withPrivatePaddingZero
    (assignment : Fin Artifact.relationColumns → F) :
    Fin Artifact.relationColumns → F :=
  fun column =>
    if Artifact.firstPaddingColumn ≤ column.val ∧
        column.val < Artifact.firstPaddingColumn + Artifact.paddingWidth then
      0
    else
      assignment column

theorem withPrivatePaddingZero_at_padding
    (assignment : Fin Artifact.relationColumns → F)
    (offset : Fin Artifact.paddingWidth) :
    withPrivatePaddingZero assignment (paddingColumn offset) = 0 := by
  have offsetBound := offset.isLt
  simp [withPrivatePaddingZero, paddingColumn, Artifact.firstPaddingColumn,
    Artifact.paddingWidth]

/-- Honest completeness: materializing the compiler-owned zero interval
constructs an assignment satisfying all 38 generated equations. -/
theorem withPrivatePaddingZero_satisfies
    (assignment : Fin Artifact.relationColumns → F) :
    GeneratedRowsSatisfied (withPrivatePaddingZero assignment) := by
  intro offset
  rw [RowSatisfied, expectedRow_residual_eq,
    withPrivatePaddingZero_at_padding]
  rw [Fin.mul_zero]
  rfl

/-- Every actual generated raw record has exactly one semantic owner and
decodes to the corresponding equation. -/
theorem generated_raw_row_refines {raw :
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.RawRow}
    (member : raw ∈
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.rawRows) :
    ∃ offset : Fin Artifact.paddingWidth,
      raw = Artifact.expectedRow offset.val ∧
        decodeRow raw = some (expectedDecodedRow offset) ∧
        ∀ assignment : Fin Artifact.relationColumns → F,
          assignment constantColumn = 1 →
          (RowSatisfied (expectedDecodedRow offset) assignment ↔
            assignment (paddingColumn offset) = 0) := by
  rcases
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePadding.generated_row_has_unique_offset
        member with
    ⟨offset, rawExact, _unique⟩
  refine ⟨offset, rawExact, ?_, ?_⟩
  · rw [rawExact]
    exact expectedRow_decode_exact offset
  · intro assignment constantOne
    exact expectedRow_satisfied_iff_padding_zero offset assignment constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement
