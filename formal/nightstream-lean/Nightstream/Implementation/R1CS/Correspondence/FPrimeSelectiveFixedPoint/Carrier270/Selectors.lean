import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Selectors
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.SelectorComposition

/-!
Semantic refinement of the bounded fixed-point selector rows.

Owns: fail-closed decoding of the exact three selector-domain rows and one
selector-total row; coefficient-based reduction to the Boolean and sum-to-one
equations; soundness under constant-one and Goldilocks no-zero-products; and
an honest unit-selector extension.

Does not own: authority for the constant-one coordinate, selector-gated branch
rows, retained-row coverage, branch-to-paper refinement, CCS/CE membership,
commitment alignment, or row removal. In particular, the Boolean rows remain
candidate eliminations at the model level; this file only states what the
physical rows currently enforce.

Emits constraints: no.

| Stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.selector.domain` | three exact Boolean rows | checked/derived |
| `f_prime.fixed_point.selector.total` | exact selector sum equals one | checked/derived |
| `f_prime.fixed_point.selector.honest` | unit selector satisfies all four rows | computed |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

abbrev booleanResidual :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.booleanResidual
abbrev productResidual :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.productResidual

namespace Artifact

abbrev expectedSelectorRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorRow
abbrev expectedTotalRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedTotalRow
abbrev expectedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedRow
abbrev rawRows :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.rawRows
abbrev relationColumns :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationColumns
abbrev relationRows :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationRows
abbrev selectorCount :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorCount
abbrev selectorStart :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorStart
abbrev totalEmittedRow :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalEmittedRow
abbrev negativeOneWord :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.negativeOneWord
abbrev expectedSelectorPort :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort
abbrev totalPort :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalPort

end Artifact

def constantColumn : Fin 11437038 :=
  ⟨0, by decide⟩

def selectorColumn (arm : Fin 3) : Fin 11437038 :=
  ⟨270 + arm.val, by omega⟩

def expectedSelectorDecodedPort (arm : Fin 3)
    (port : Fin 13) : DecodedPort 11437038 :=
  if port.val = 0 then
    unitDecodedPort 11437038 (selectorColumn arm).val
      (selectorColumn arm).isLt
  else if port.val = 1 then
    unitDecodedPort 11437038 constantColumn.val
      constantColumn.isLt
  else
    emptyDecodedPort 11437038

def expectedSelectorDecodedRow (arm : Fin 3) :
    DecodedRow :=
  { rows := 14944219
    columns := 11437038
    rowsPositive := by decide
    columnsPositive := by decide
    emittedRow := ⟨arm.val, by omega⟩
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := fun port =>
      (List.ofFn (expectedSelectorDecodedPort arm)).get
        ⟨port.val, by simp⟩ }

@[simp] theorem expectedSelectorDecodedRow_port (arm : Fin 3)
    (port : Fin 13) :
    (expectedSelectorDecodedRow arm).port port =
      expectedSelectorDecodedPort arm port := by
  unfold DecodedRow.port expectedSelectorDecodedRow
  change
    (List.ofFn (expectedSelectorDecodedPort arm)).get
        ⟨port.val, by simp⟩ =
      expectedSelectorDecodedPort arm port
  rw [List.get_eq_getElem, List.getElem_ofFn]

private theorem mapM_decodePorts_of_pointwise {columns count : Nat}
    (raw : Fin count → RawPort)
    (decoded : Fin count → DecodedPort columns)
    (pointwise : ∀ index, decodePort columns (raw index) =
      some (decoded index)) :
    (List.ofFn raw).mapM (decodePort columns) =
      some (List.ofFn decoded) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.ofFn_succ]
      simp only [List.mapM_cons, pointwise]
      rw [inductionHypothesis (raw := fun index => raw index.succ)
        (decoded := fun index => decoded index.succ)
        (pointwise := fun index => pointwise index.succ)]
      rfl

private theorem expectedSelectorPort_decode_exact (arm : Fin 3)
    (port : Fin 13) :
    decodePort 11437038
        (Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort
          arm port) =
      some (expectedSelectorDecodedPort arm port) := by
  by_cases bitPort : port.val = 0
  · simpa [Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.unitPort,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorColumn,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorStart,
      expectedSelectorDecodedPort, selectorColumn, bitPort] using
        (decodePort_unit 11437038 (270 + arm.val) (by omega))
  · by_cases generalPort : port.val = 1
    · simpa [Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.unitPort,
        expectedSelectorDecodedPort, constantColumn, bitPort, generalPort] using
          (decodePort_unit 11437038 0 (by decide))
    · simp [Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.emptyPort,
        expectedSelectorDecodedPort, bitPort, generalPort,
        decodePort_empty]

theorem expectedSelectorRow_decode_exact
    (arm : Fin 3) :
    decodeRow (Artifact.expectedSelectorRow arm) =
      some (expectedSelectorDecodedRow arm) := by
  have decodedPorts := mapM_decodePorts_of_pointwise
    (fun port : Fin 13 =>
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorPort
        arm port)
    (expectedSelectorDecodedPort arm)
    (expectedSelectorPort_decode_exact arm)
  unfold Artifact.expectedSelectorRow
  unfold Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedSelectorRow
  unfold decodeRow
  simp only [
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationRows,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationColumns,
    supportedSchemaVersion]
  have emittedBound : arm.val < 14944219 := by
    have armBound := arm.isLt
    omega
  rw [dif_pos True.intro, dif_pos (by decide), dif_pos (by decide),
    dif_pos emittedBound]
  rw [decodedPorts]
  rfl

def negativeOne : F :=
  ⟨Artifact.negativeOneWord, by decide⟩

theorem negativeOne_eq_neg_one : negativeOne = -1 := by
  decide

theorem negativeOne_nonzero : negativeOne ≠ 0 := by
  decide

def decodedFirstSelectorTerm : DecodedTerm 11437038 :=
  canonicalDecodedTerm 11437038 270 1 (by decide) (by decide) (by decide)

def decodedSecondSelectorTerm : DecodedTerm 11437038 :=
  canonicalDecodedTerm 11437038 271 1 (by decide) (by decide) (by decide)

def decodedThirdSelectorTerm : DecodedTerm 11437038 :=
  canonicalDecodedTerm 11437038 272 1 (by decide) (by decide) (by decide)

def decodedNegativeOneTerm : DecodedTerm 11437038 :=
  canonicalDecodedTerm 11437038 constantColumn.val
    Artifact.negativeOneWord constantColumn.isLt (by decide)
      negativeOne_nonzero

def expectedTotalCDecodedPort : DecodedPort 11437038 :=
  { explicit :=
      [ decodedNegativeOneTerm
      , decodedFirstSelectorTerm
      , decodedSecondSelectorTerm
      , decodedThirdSelectorTerm
      ]
    geometric := [] }

def expectedTotalDecodedPort (port : Fin 13) :
    DecodedPort 11437038 :=
  if port.val = 1 then
    unitDecodedPort 11437038 constantColumn.val
      constantColumn.isLt
  else if port.val = 4 then
    expectedTotalCDecodedPort
  else
    emptyDecodedPort 11437038

def expectedTotalDecodedRow : DecodedRow :=
  { rows := 14944219
    columns := 11437038
    rowsPositive := by decide
    columnsPositive := by decide
    emittedRow := ⟨Artifact.totalEmittedRow, by decide⟩
    runIndex := 5
    family := .oneHot
    arm := none
    ports := fun port =>
      (List.ofFn expectedTotalDecodedPort).get ⟨port.val, by simp⟩ }

@[simp] theorem expectedTotalDecodedRow_port (port : Fin 13) :
    expectedTotalDecodedRow.port port = expectedTotalDecodedPort port := by
  unfold DecodedRow.port expectedTotalDecodedRow
  change
    (List.ofFn expectedTotalDecodedPort).get ⟨port.val, by simp⟩ =
      expectedTotalDecodedPort port
  rw [List.get_eq_getElem, List.getElem_ofFn]

private theorem decodeTotalCPort :
    decodePort 11437038
        { explicit :=
            [ { column := 0, coefficient := Artifact.negativeOneWord }
            , { column := 270, coefficient := 1 }
            , { column := 271, coefficient := 1 }
            , { column := 272, coefficient := 1 }
            ]
          geometric := [] } =
      some expectedTotalCDecodedPort := by
  have decodedNegative :=
    decodeTerm_canonical 11437038 0
      Artifact.negativeOneWord (by decide) (by decide)
        negativeOne_nonzero
  have decodedFirst :=
    decodeTerm_canonical 11437038 270 1
      (by decide) (by decide) (by decide)
  have decodedSecond :=
    decodeTerm_canonical 11437038 271 1
      (by decide) (by decide) (by decide)
  have decodedThird :=
    decodeTerm_canonical 11437038 272 1
      (by decide) (by decide) (by decide)
  unfold decodePort
  simp [decodedNegative, decodedFirst, decodedSecond, decodedThird,
    expectedTotalCDecodedPort, decodedNegativeOneTerm,
    decodedFirstSelectorTerm, decodedSecondSelectorTerm,
    decodedThirdSelectorTerm, constantColumn]

private theorem expectedTotalPort_decode_exact (port : Fin 13) :
    decodePort 11437038 (Artifact.totalPort port) =
      some (expectedTotalDecodedPort port) := by
  by_cases generalPort : port.val = 1
  · simpa [Artifact.totalPort,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalPort,
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.unitPort,
      constantColumn, generalPort, expectedTotalDecodedPort] using
        (decodePort_unit 11437038 0 (by decide))
  · by_cases cPort : port.val = 4
    · simp [Artifact.totalPort,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalPort,
        cPort, expectedTotalDecodedPort]
      exact decodeTotalCPort
    · simp [Artifact.totalPort,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalPort,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.emptyPort,
        generalPort, cPort, expectedTotalDecodedPort, decodePort_empty]

theorem expectedTotalRow_decode_exact :
    decodeRow Artifact.expectedTotalRow = some expectedTotalDecodedRow := by
  have decodedPorts := mapM_decodePorts_of_pointwise
    (Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalPort)
    expectedTotalDecodedPort expectedTotalPort_decode_exact
  unfold Artifact.expectedTotalRow
  unfold Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedTotalRow
  unfold decodeRow
  simp only [
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationRows,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.relationColumns,
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.totalEmittedRow,
    supportedSchemaVersion]
  rw [dif_pos True.intro, dif_pos (by decide), dif_pos (by decide),
    dif_pos (by decide)]
  rw [decodedPorts]
  rfl

def expectedDecodedRow (index : Fin 4) : DecodedRow :=
  if selector : index.val < 3 then
    expectedSelectorDecodedRow ⟨index.val, selector⟩
  else
    expectedTotalDecodedRow

theorem expectedRow_decode_exact (index : Fin 4) :
    decodeRow (Artifact.expectedRow index) = some (expectedDecodedRow index) := by
  by_cases selector : index.val < 3
  · have artifactSelector :
        index.val <
          Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorCount := by
      simpa [Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorCount]
        using selector
    rw [show Artifact.expectedRow index =
        Artifact.expectedSelectorRow ⟨index.val, artifactSelector⟩ by
      simp [Artifact.expectedRow,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedRow,
        artifactSelector]]
    simpa [expectedDecodedRow, selector] using
      (expectedSelectorRow_decode_exact ⟨index.val, selector⟩)
  · have artifactSelector :
        ¬index.val <
          Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorCount := by
      simpa [Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.selectorCount]
        using selector
    rw [show Artifact.expectedRow index = Artifact.expectedTotalRow by
      simp [Artifact.expectedRow,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.expectedRow,
        artifactSelector]]
    simpa [expectedDecodedRow, selector] using expectedTotalRow_decode_exact

/-- Every physical generated row has one exact owner and fail-closed decoded
equation. -/
theorem generated_raw_row_decodes {raw :
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.RawRow}
    (member : raw ∈ Artifact.rawRows) :
    ∃ index : Fin 4,
      raw = Artifact.expectedRow index ∧
        decodeRow raw = some (expectedDecodedRow index) ∧
        ∀ other : Fin 4, raw = Artifact.expectedRow other → other = index := by
  rcases
      Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Selectors.generated_row_has_unique_owner
        member with
    ⟨index, rawExact, unique⟩
  refine ⟨index, rawExact, ?_, unique⟩
  rw [rawExact]
  exact expectedRow_decode_exact index

@[simp] theorem expandedFieldTerms_empty (columns : Nat) :
    expandedFieldTerms (emptyDecodedPort columns) = [] := by
  rfl

@[simp] theorem expandedFieldTerms_unit (columns column : Nat)
    (columnInRange : column < columns) :
    expandedFieldTerms (unitDecodedPort columns column columnInRange) =
      [(⟨column, columnInRange⟩, 1)] := by
  simp [expandedFieldTerms, unitDecodedPort, unitDecodedTerm,
    termAsFieldTerm]

@[simp] theorem termAsFieldTerm_decodedFirst :
    termAsFieldTerm decodedFirstSelectorTerm =
      (selectorColumn 0, 1) := by
  decide

@[simp] theorem termAsFieldTerm_decodedSecond :
    termAsFieldTerm decodedSecondSelectorTerm =
      (selectorColumn 1, 1) := by
  decide

@[simp] theorem termAsFieldTerm_decodedThird :
    termAsFieldTerm decodedThirdSelectorTerm =
      (selectorColumn 2, 1) := by
  decide

@[simp] theorem expandedFieldTerms_decodedNegativeOne :
    termAsFieldTerm decodedNegativeOneTerm = (constantColumn, -1) := by
  apply Prod.ext
  · rfl
  · exact negativeOne_eq_neg_one

def IsBooleanAt (row : DecodedRow)
    (bitColumn selectorColumn : Fin row.columns) : Prop :=
  expandedFieldTerms (row.port Role.bit.index) = [(bitColumn, 1)] ∧
    expandedFieldTerms (row.port Role.generalSelector.index) =
      [(selectorColumn, 1)] ∧
    ∀ port : Fin 13,
      port ≠ Role.bit.index →
      port ≠ Role.generalSelector.index →
      expandedFieldTerms (row.port port) = []

structure ValidatedBooleanRow (row : DecodedRow) where
  bitColumn : Fin row.columns
  selectorColumn : Fin row.columns
  shape : IsBooleanAt row bitColumn selectorColumn

def IsThreeSelectorTotalAt (row : DecodedRow)
    (constant first second third : Fin row.columns) : Prop :=
  expandedFieldTerms (row.port Role.generalSelector.index) =
      [(constant, 1)] ∧
    expandedFieldTerms (row.port Role.c.index) =
      [(constant, -1), (first, 1), (second, 1), (third, 1)] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.c.index →
      expandedFieldTerms (row.port port) = []

structure ValidatedThreeSelectorTotalRow (row : DecodedRow) where
  constantColumn : Fin row.columns
  firstColumn : Fin row.columns
  secondColumn : Fin row.columns
  thirdColumn : Fin row.columns
  shape : IsThreeSelectorTotalAt row constantColumn firstColumn secondColumn
    thirdColumn

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

theorem booleanRowPoint_eq_booleanPoint
    (row : DecodedRow) (validated : ValidatedBooleanRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      booleanPoint (assignment validated.selectorColumn)
        (assignment validated.bitColumn) := by
  funext port
  by_cases bitPort : port = Role.bit.index
  · subst port
    simp only [rowPoint]
    rw [action_unit row Role.bit.index validated.bitColumn
      validated.shape.1 assignment]
    simp [booleanPoint, sparsePoint, Role.index]
  · by_cases generalPort : port = Role.generalSelector.index
    · subst port
      simp only [rowPoint]
      rw [action_unit row Role.generalSelector.index
        validated.selectorColumn validated.shape.2.1 assignment]
      simp [booleanPoint, sparsePoint, Role.index]
    · simp only [rowPoint]
      rw [action_empty row port
        (validated.shape.2.2 port bitPort generalPort) assignment]
      have notZero : port ≠ (0 : Fin 13) := by
        simpa only [Role.index] using bitPort
      have notOne : port ≠ (1 : Fin 13) := by
        simpa only [Role.index] using generalPort
      simp [booleanPoint, sparsePoint, notZero, notOne]

theorem booleanResidual_eq
    (row : DecodedRow) (validated : ValidatedBooleanRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      booleanResidual
        (booleanPoint (assignment validated.selectorColumn)
          (assignment validated.bitColumn)) := by
  rw [residual, booleanRowPoint_eq_booleanPoint row validated assignment,
    evaluate_booleanPoint]

theorem totalRowPoint_eq_selectorTotalPoint
    (row : DecodedRow) (validated : ValidatedThreeSelectorTotalRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint
        (assignment validated.constantColumn)
        (assignment validated.firstColumn)
        (assignment validated.secondColumn)
        (assignment validated.thirdColumn) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint]
    rw [action_unit row Role.generalSelector.index validated.constantColumn
      validated.shape.1 assignment]
    simp [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
      productPoint, sparsePoint, Role.index]
  · by_cases cPort : port = Role.c.index
    · subst port
      simp only [rowPoint, action]
      rw [validated.shape.2.1]
      simp [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
        productPoint, sparsePoint, Role.index, Lean.Grind.Fin.neg_mul,
        Fin.one_mul]
    · simp only [rowPoint]
      rw [action_empty row port
        (validated.shape.2.2 port generalPort cPort) assignment]
      have notOne : port ≠ (1 : Fin 13) := by
        simpa only [Role.index] using generalPort
      have notFour : port ≠ (4 : Fin 13) := by
        simpa only [Role.index] using cPort
      simp [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
        productPoint, sparsePoint, Role.index, notOne, notFour]

theorem totalResidual_eq_selectorGap
    (row : DecodedRow) (validated : ValidatedThreeSelectorTotalRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      -(assignment validated.constantColumn *
        (-assignment validated.constantColumn +
          assignment validated.firstColumn +
          assignment validated.secondColumn +
          assignment validated.thirdColumn)) := by
  rw [residual,
    totalRowPoint_eq_selectorTotalPoint row validated assignment,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorTotalPoint,
    evaluate_productPoint]
  simp [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.productResidual,
    productPoint, sparsePoint, Role.index,
    Fin.mul_zero, Fin.zero_add]

theorem expectedSelectorDecodedRow_shape
    (arm : Fin 3) :
    IsBooleanAt (expectedSelectorDecodedRow arm) (selectorColumn arm)
      constantColumn := by
  refine ⟨?_, ?_, ?_⟩
  · change
      expandedFieldTerms
          (expectedSelectorDecodedPort arm (0 : Fin 13)) =
        [(selectorColumn arm, 1)]
    rw [show expectedSelectorDecodedPort arm (0 : Fin 13) =
        unitDecodedPort 11437038 (selectorColumn arm).val
          (selectorColumn arm).isLt by
      simp [expectedSelectorDecodedPort]]
    exact expandedFieldTerms_unit 11437038 (selectorColumn arm).val
      (selectorColumn arm).isLt
  · change
      expandedFieldTerms
          (expectedSelectorDecodedPort arm (1 : Fin 13)) =
        [(constantColumn, 1)]
    rw [show expectedSelectorDecodedPort arm (1 : Fin 13) =
        unitDecodedPort 11437038 constantColumn.val
          constantColumn.isLt by
      simp [expectedSelectorDecodedPort]]
    exact expandedFieldTerms_unit 11437038 constantColumn.val
      constantColumn.isLt
  · intro port bitNe generalNe
    have notZero : port.val ≠ 0 := by
      intro value
      apply bitNe
      apply Fin.ext
      simpa [Role.index] using value
    have notOne : port.val ≠ 1 := by
      intro value
      apply generalNe
      apply Fin.ext
      simpa [Role.index] using value
    rw [expectedSelectorDecodedRow_port]
    rw [show expectedSelectorDecodedPort arm port =
        emptyDecodedPort 11437038 by
      simp [expectedSelectorDecodedPort, notZero, notOne]]
    exact expandedFieldTerms_empty 11437038

def validatedExpectedSelectorRow (arm : Fin 3) :
    ValidatedBooleanRow (expectedSelectorDecodedRow arm) :=
  { bitColumn := selectorColumn arm
    selectorColumn := constantColumn
    shape := expectedSelectorDecodedRow_shape arm }

theorem expectedTotalDecodedRow_shape :
    IsThreeSelectorTotalAt expectedTotalDecodedRow constantColumn
      (selectorColumn 0) (selectorColumn 1) (selectorColumn 2) := by
  refine ⟨?_, ?_, ?_⟩
  · change
      expandedFieldTerms (expectedTotalDecodedPort (1 : Fin 13)) =
        [(constantColumn, 1)]
    rw [show expectedTotalDecodedPort (1 : Fin 13) =
        unitDecodedPort 11437038 constantColumn.val
          constantColumn.isLt by
      simp [expectedTotalDecodedPort]]
    exact expandedFieldTerms_unit 11437038 constantColumn.val
      constantColumn.isLt
  · change
      expandedFieldTerms (expectedTotalDecodedPort (4 : Fin 13)) =
        [(constantColumn, -1), (selectorColumn 0, 1),
          (selectorColumn 1, 1), (selectorColumn 2, 1)]
    rw [show expectedTotalDecodedPort (4 : Fin 13) =
        expectedTotalCDecodedPort by rfl]
    simp [expectedTotalCDecodedPort, expandedFieldTerms]
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
    rw [expectedTotalDecodedRow_port]
    rw [show expectedTotalDecodedPort port = emptyDecodedPort 11437038 by
      simp [expectedTotalDecodedPort, notOne, notFour]]
    exact expandedFieldTerms_empty 11437038

def validatedExpectedTotalRow :
    ValidatedThreeSelectorTotalRow expectedTotalDecodedRow :=
  { constantColumn
    firstColumn := selectorColumn 0
    secondColumn := selectorColumn 1
    thirdColumn := selectorColumn 2
    shape := expectedTotalDecodedRow_shape }

theorem expectedSelectorResidual_eq
    (arm : Fin 3)
    (assignment : Fin 11437038 → F) :
    residual (expectedSelectorDecodedRow arm) assignment =
      booleanResidual
        (booleanPoint (assignment constantColumn)
          (assignment (selectorColumn arm))) := by
  exact booleanResidual_eq (expectedSelectorDecodedRow arm)
    (validatedExpectedSelectorRow arm) assignment

theorem expectedTotalResidual_eq
    (assignment : Fin 11437038 → F) :
    residual expectedTotalDecodedRow assignment =
      -(assignment constantColumn *
        (-assignment constantColumn + assignment (selectorColumn 0) +
          assignment (selectorColumn 1) +
          assignment (selectorColumn 2))) := by
  exact totalResidual_eq_selectorGap expectedTotalDecodedRow
    validatedExpectedTotalRow assignment

private theorem fmul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := congrArg Neg.neg (Fin.mul_comm _ _)

private theorem fadd_neg_cancel (value : F) : value + -value = 0 := by
  rw [Lean.Grind.Fin.add_comm]
  exact Lean.Grind.Fin.neg_add_cancel value

private theorem booleanResidual_one_eq_factor (bit : F) :
    booleanResidual (booleanPoint 1 bit) = bit * (bit + -1) := by
  calc
    booleanResidual (booleanPoint 1 bit) =
        bit * bit + -bit := by
          simp [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components.booleanResidual,
            booleanPoint, sparsePoint, Role.index,
            Fin.mul_one]
    _ = bit * bit + bit * -1 := by
          rw [fmul_neg, Fin.mul_one]
    _ = bit * (bit + -1) :=
      (Lean.Grind.Fin.left_distrib bit bit (-1)).symm

private theorem add_neg_one_eq_zero_iff (value : F) :
    value + -1 = 0 ↔ value = 1 := by
  constructor
  · intro zero
    calc
      value = value + 0 := by rw [Fin.add_zero]
      _ = value + (-1 + 1) := by
        rw [Lean.Grind.Fin.neg_add_cancel]
      _ = (value + -1) + 1 := by
        rw [Lean.Grind.Fin.add_assoc]
      _ = 0 + 1 := by rw [zero]
      _ = 1 := Fin.zero_add _
  · rintro rfl
    exact fadd_neg_cancel 1

theorem booleanResidual_one_eq_zero_iff
    (prime : EuclidPrime goldilocksP) (bit : F) :
    booleanResidual (booleanPoint 1 bit) = 0 ↔ bit = 0 ∨ bit = 1 := by
  rw [booleanResidual_one_eq_factor]
  constructor
  · intro zero
    rcases goldilocks_noZeroProducts prime bit (bit + -1) zero with
      bitZero | gapZero
    · exact Or.inl bitZero
    · exact Or.inr ((add_neg_one_eq_zero_iff bit).1 gapZero)
  · rintro (rfl | rfl)
    · exact Fin.zero_mul _
    · rw [fadd_neg_cancel, Fin.mul_zero]

theorem expectedSelectorRow_satisfied_iff_boolean
    (prime : EuclidPrime goldilocksP)
    (arm : Fin 3)
    (assignment : Fin 11437038 → F)
    (constantOne : assignment constantColumn = 1) :
    RowSatisfied (expectedSelectorDecodedRow arm) assignment ↔
      assignment (selectorColumn arm) = 0 ∨
        assignment (selectorColumn arm) = 1 := by
  rw [RowSatisfied, expectedSelectorResidual_eq, constantOne]
  exact booleanResidual_one_eq_zero_iff prime _

theorem expectedTotalRow_satisfied_iff_total
    (assignment : Fin 11437038 → F)
    (constantOne : assignment constantColumn = 1) :
    RowSatisfied expectedTotalDecodedRow assignment ↔
      SelectorTotal
        (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.threeWeights
          (assignment (selectorColumn 0))
          (assignment (selectorColumn 1))
          (assignment (selectorColumn 2))) := by
  rw [RowSatisfied, expectedTotalResidual_eq]
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorGap_eq_zero_iff_total
      _ _ _ _ constantOne

def GeneratedRowsSatisfied
    (assignment : Fin 11437038 → F) : Prop :=
  (∀ arm : Fin 3,
      RowSatisfied (expectedSelectorDecodedRow arm) assignment) ∧
    RowSatisfied expectedTotalDecodedRow assignment

theorem generatedRowsSatisfied_iff
    (prime : EuclidPrime goldilocksP)
    (assignment : Fin 11437038 → F)
    (constantOne : assignment constantColumn = 1) :
    GeneratedRowsSatisfied assignment ↔
      (∀ arm : Fin 3,
        assignment (selectorColumn arm) = 0 ∨
          assignment (selectorColumn arm) = 1) ∧
      SelectorTotal
        (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.threeWeights
          (assignment (selectorColumn 0))
          (assignment (selectorColumn 1))
          (assignment (selectorColumn 2))) := by
  constructor
  · intro satisfied
    exact ⟨fun arm =>
      (expectedSelectorRow_satisfied_iff_boolean prime arm assignment
        constantOne).1 (satisfied.1 arm),
      (expectedTotalRow_satisfied_iff_total assignment constantOne).1
        satisfied.2⟩
  · rintro ⟨boolean, total⟩
    exact ⟨fun arm =>
      (expectedSelectorRow_satisfied_iff_boolean prime arm assignment
        constantOne).2 (boolean arm),
      (expectedTotalRow_satisfied_iff_total assignment constantOne).2 total⟩

/-- Replace exactly the prepared selector interval `[270, 273)` by the
canonical unit vector for `selected`. -/
def withUnitSelectors (selected : Fin 3)
    (assignment : Fin 11437038 → F) :
    Fin 11437038 → F :=
  fun column =>
    if inSelectorRange :
        270 ≤ column.val ∧ column.val < 273 then
      unitWeights selected
        ⟨column.val - 270, by omega⟩
    else
      assignment column

theorem withUnitSelectors_at_selector
    (selected arm : Fin 3)
    (assignment : Fin 11437038 → F) :
    withUnitSelectors selected assignment (selectorColumn arm) =
      unitWeights selected arm := by
  unfold withUnitSelectors
  have inRange : 270 ≤ (selectorColumn arm).val ∧
      (selectorColumn arm).val < 273 := by
    have armBound := arm.isLt
    simp only [selectorColumn]
    omega
  rw [dif_pos inRange]
  apply congrArg (unitWeights selected)
  apply Fin.ext
  simp [selectorColumn]

theorem withUnitSelectors_at_constant
    (selected : Fin 3)
    (assignment : Fin 11437038 → F) :
    withUnitSelectors selected assignment constantColumn =
      assignment constantColumn := by
  unfold withUnitSelectors
  rw [dif_neg (by simp [constantColumn])]

/-- Honest completeness for the exact four physical selector equations. -/
theorem withUnitSelectors_satisfies
    (prime : EuclidPrime goldilocksP)
    (selected : Fin 3)
    (assignment : Fin 11437038 → F)
    (constantOne : assignment constantColumn = 1) :
    GeneratedRowsSatisfied (withUnitSelectors selected assignment) := by
  apply (generatedRowsSatisfied_iff prime _ ?_).2
  · constructor
    · intro arm
      rw [withUnitSelectors_at_selector]
      unfold unitWeights
      split <;> simp_all
    · simpa only [withUnitSelectors_at_selector] using
        (unitWeights_total selected)
  · rw [withUnitSelectors_at_constant]
    exact constantOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement
