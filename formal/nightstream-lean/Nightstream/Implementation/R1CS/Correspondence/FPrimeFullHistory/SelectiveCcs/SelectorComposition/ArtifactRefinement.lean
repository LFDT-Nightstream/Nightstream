import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.SelectorComposition

/-!
Contract: exact generated-row refinement for the three-arm 270-carrier
selector prefix.

Owns: fail-closed decoding of all three selector-domain rows, the selector
total row, and one representative retained arm row; coefficient-derived row
classification; and their exact connection to the model-level selector total
and gated-source equations.

Does not own: the full production F' relation, every arm row, constant-one
connectivity beyond an explicit premise, branch-to-paper semantics, constraint
necessity, a trusted row count, or permission to remove rows.

Emits constraints: no. The imported data comes from a deterministic compiler
fixture whose final matrices are regenerated and drift-checked by Rust.

| Stage path | Artifact multiplicity | Exact mathematical result | Assurance tier |
|---|---:|---|---|
| `f_prime.selective_ccs.branch.selector_domain` | 3 | each row is `G * s * (s-1)` | artifact-checked fixture |
| `f_prime.selective_ccs.branch.total` | 1 | with constant one, acceptance iff `sum s = 1` | artifact-checked fixture |
| `f_prime.selective_ccs.branch.gate[0].representative` | 1 | residual is `s0 * (A*B-C)` | artifact-checked fixture |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

def decodedSelectorRows : List DecodedRow :=
  (rawSelectorRows.mapM decodeRow).getD []

theorem decodedSelectorRows_length : decodedSelectorRows.length = 3 := by
  decide

def decodedSelectorRow (arm : Fin 3) : DecodedRow :=
  decodedSelectorRows.get ⟨arm.val, by
    rw [decodedSelectorRows_length]
    exact arm.isLt⟩

theorem decodedSelectorRow_columns :
    ∀ arm : Fin 3, (decodedSelectorRow arm).columns = 324 := by
  decide

def selectorConstantColumn (arm : Fin 3) :
    Fin (decodedSelectorRow arm).columns :=
  ⟨0, by
    have columns := decodedSelectorRow_columns arm
    omega⟩

def selectorValueColumn (arm : Fin 3) :
    Fin (decodedSelectorRow arm).columns :=
  ⟨270 + arm.val, by
    have columns := decodedSelectorRow_columns arm
    have armLt := arm.isLt
    omega⟩

theorem generated_selector_rows_shape :
    ∀ arm : Fin 3,
      (decodedSelectorRow arm).rows = 836 ∧
      (decodedSelectorRow arm).columns = 324 ∧
      (decodedSelectorRow arm).emittedRow.val = arm.val ∧
      (decodedSelectorRow arm).runIndex = 0 ∧
      (decodedSelectorRow arm).family = .selectorDomain ∧
      (decodedSelectorRow arm).arm = none ∧
      (selectorConstantColumn arm).val = 0 ∧
      (selectorValueColumn arm).val = 270 + arm.val ∧
      IsBooleanAt (decodedSelectorRow arm)
        (selectorValueColumn arm) (selectorConstantColumn arm) := by
  decide

def validatedSelectorRow (arm : Fin 3) :
    ValidatedBooleanRow (decodedSelectorRow arm) where
  bitColumn := selectorValueColumn arm
  selectorColumn := selectorConstantColumn arm
  shape := (generated_selector_rows_shape arm).2.2.2.2.2.2.2.2

theorem generated_selector_row_residual
    (arm : Fin 3)
    (assignment : Fin (decodedSelectorRow arm).columns → F) :
    residual (decodedSelectorRow arm) assignment =
      booleanResidual
        (booleanPoint
          (assignment (selectorConstantColumn arm))
          (assignment (selectorValueColumn arm))) := by
  exact residual_eq_booleanResidual
    (decodedSelectorRow arm) (validatedSelectorRow arm) assignment

def decodedTotalRow : DecodedRow :=
  (decodeRow rawOneHotRow).get (by decide)

theorem decodedTotalRow_columns : decodedTotalRow.columns = 324 := by
  decide

def totalConstantColumn : Fin decodedTotalRow.columns :=
  ⟨0, by
    have columns := decodedTotalRow_columns
    omega⟩

def totalSelectorColumn (arm : Fin 3) : Fin decodedTotalRow.columns :=
  ⟨270 + arm.val, by
    have columns := decodedTotalRow_columns
    have armLt := arm.isLt
    omega⟩

theorem generated_total_row_shape :
    decodedTotalRow.rows = 836 ∧
    decodedTotalRow.columns = 324 ∧
    decodedTotalRow.emittedRow.val = 3 ∧
    decodedTotalRow.runIndex = 5 ∧
    decodedTotalRow.family = .oneHot ∧
    decodedTotalRow.arm = none ∧
    (totalConstantColumn).val = 0 ∧
    (totalSelectorColumn 0).val = 270 ∧
    (totalSelectorColumn 1).val = 271 ∧
    (totalSelectorColumn 2).val = 272 ∧
    IsThreeSelectorTotalAt decodedTotalRow totalConstantColumn
      (totalSelectorColumn 0) (totalSelectorColumn 1)
      (totalSelectorColumn 2) := by
  decide

def validatedTotalRow : ValidatedThreeSelectorTotalRow decodedTotalRow where
  constantColumn := totalConstantColumn
  firstColumn := totalSelectorColumn 0
  secondColumn := totalSelectorColumn 1
  thirdColumn := totalSelectorColumn 2
  shape := generated_total_row_shape.2.2.2.2.2.2.2.2.2.2

def totalWeights (assignment : Fin decodedTotalRow.columns → F) : Fin 3 → F :=
  threeWeights
    (assignment (totalSelectorColumn 0))
    (assignment (totalSelectorColumn 1))
    (assignment (totalSelectorColumn 2))

theorem generated_total_row_residual
    (assignment : Fin decodedTotalRow.columns → F) :
    residual decodedTotalRow assignment =
      -(assignment totalConstantColumn *
        (-assignment totalConstantColumn +
          assignment (totalSelectorColumn 0) +
          assignment (totalSelectorColumn 1) +
          assignment (totalSelectorColumn 2))) := by
  exact residual_eq_selectorGap decodedTotalRow validatedTotalRow assignment

theorem generated_total_row_iff_selectorTotal
    (assignment : Fin decodedTotalRow.columns → F)
    (constantOne : assignment totalConstantColumn = 1) :
    residual decodedTotalRow assignment = 0 ↔
      SelectorTotal (totalWeights assignment) := by
  rw [generated_total_row_residual]
  exact selectorGap_eq_zero_iff_total
    (assignment totalConstantColumn)
    (assignment (totalSelectorColumn 0))
    (assignment (totalSelectorColumn 1))
    (assignment (totalSelectorColumn 2))
    constantOne

def decodedGatedRow : DecodedRow :=
  (decodeRow rawGatedRow).get (by decide)

theorem decodedGatedRow_columns : decodedGatedRow.columns = 324 := by
  decide

def gatedConstantColumn : Fin decodedGatedRow.columns :=
  ⟨0, by
    have columns := decodedGatedRow_columns
    omega⟩

def gatedBitColumn : Fin decodedGatedRow.columns :=
  ⟨1, by
    have columns := decodedGatedRow_columns
    omega⟩

def gatedSelectorColumn : Fin decodedGatedRow.columns :=
  ⟨270, by
    have columns := decodedGatedRow_columns
    omega⟩

theorem generated_gated_row_shape :
    decodedGatedRow.rows = 836 ∧
    decodedGatedRow.columns = 324 ∧
    decodedGatedRow.emittedRow.val = 55 ∧
    decodedGatedRow.runIndex = 8 ∧
    decodedGatedRow.family = .retained ∧
    decodedGatedRow.arm = some 0 ∧
    IsProductGateAt decodedGatedRow gatedSelectorColumn ∧
    (decodedGatedRow.port
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.a.index).terms =
        [{ column := gatedBitColumn, coefficient := 1 }] ∧
    (decodedGatedRow.port
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.b.index).terms =
        [{ column := gatedConstantColumn, coefficient := -1 },
          { column := gatedBitColumn, coefficient := 1 }] ∧
    (decodedGatedRow.port
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.Role.c.index).terms = [] := by
  decide

def validatedGatedRow : ValidatedProductGateRow decodedGatedRow where
  selectorColumn := gatedSelectorColumn
  shape := generated_gated_row_shape.2.2.2.2.2.2.1

theorem generated_gated_row_residual
    (assignment : Fin decodedGatedRow.columns → F) :
    residual decodedGatedRow assignment =
      assignment gatedSelectorColumn *
        sourceResidual decodedGatedRow assignment := by
  exact residual_eq_gatedSource decodedGatedRow validatedGatedRow assignment

theorem generated_gated_source_residual
    (assignment : Fin decodedGatedRow.columns → F) :
    sourceResidual decodedGatedRow assignment =
      assignment gatedBitColumn *
        (-assignment gatedConstantColumn + assignment gatedBitColumn) := by
  unfold sourceResidual action
  rw [generated_gated_row_shape.2.2.2.2.2.2.2.1,
    generated_gated_row_shape.2.2.2.2.2.2.2.2.1,
    generated_gated_row_shape.2.2.2.2.2.2.2.2.2]
  simp [Fin.one_mul, Fin.zero_add, Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Lean.Grind.AddCommGroup.neg_zero]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement
