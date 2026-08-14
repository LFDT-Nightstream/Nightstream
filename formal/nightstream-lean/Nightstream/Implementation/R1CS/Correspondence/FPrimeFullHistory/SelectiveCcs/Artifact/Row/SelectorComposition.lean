import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics

/-!
Contract: coefficient-driven row classifiers for selector composition.

Owns: the exact three-selector total-row shape, the generic product-gate row
shape, their matrix actions, and reduction of the selective polynomial to the
sum-to-one and selector-times-source-residual equations.

Does not own: any generated row, Rust family labels, constant-one connectivity,
coverage of every arm row, branch-to-paper refinement, or row-removal authority.

Emits constraints: no.

| Stage path | Coefficient obligation | Mathematical result |
|---|---|---|
| `f_prime.selective_ccs.branch.total` | `G=z[0]`, `C=-z[0]+s0+s1+s2` | residual is `z[0] * (z[0] - sum selectors)` |
| `f_prime.selective_ccs.branch.gate[*]` | `G=selector`, only `A/B/C` otherwise active | residual is `selector * (A*B-C)` |
| `f_prime.selective_ccs.branch.selector_domain` | delegated to `Row.Boolean` | residual is the Boolean polynomial |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) :=
  Lean.Grind.Fin.add_assoc _ _ _

private theorem fadd_comm (left right : F) : left + right = right + left :=
  Lean.Grind.Fin.add_comm _ _

private theorem fadd_neg_cancel (value : F) : value + -value = 0 := by
  rw [fadd_comm]
  exact Lean.Grind.Fin.neg_add_cancel value

private theorem fmul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem fmul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right :=
  Lean.Grind.Fin.left_distrib _ _ _

private theorem fmul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = (-right) * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := congrArg Neg.neg (Fin.mul_comm _ _)

/-- Exact fixed-profile shape of the compiler's three-selector sum row. -/
def IsThreeSelectorTotalAt (row : DecodedRow)
    (constantColumn firstColumn secondColumn thirdColumn : Fin row.columns) : Prop :=
  (row.port Role.generalSelector.index).terms =
      [{ column := constantColumn, coefficient := 1 }] ∧
    (row.port Role.c.index).terms =
      [{ column := constantColumn, coefficient := -1 },
        { column := firstColumn, coefficient := 1 },
        { column := secondColumn, coefficient := 1 },
        { column := thirdColumn, coefficient := 1 }] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.c.index →
      (row.port port).terms = []

instance (row : DecodedRow)
    (constantColumn firstColumn secondColumn thirdColumn : Fin row.columns) :
    Decidable (IsThreeSelectorTotalAt row constantColumn firstColumn
      secondColumn thirdColumn) := by
  unfold IsThreeSelectorTotalAt
  infer_instance

structure ValidatedThreeSelectorTotalRow (row : DecodedRow) where
  constantColumn : Fin row.columns
  firstColumn : Fin row.columns
  secondColumn : Fin row.columns
  thirdColumn : Fin row.columns
  shape : IsThreeSelectorTotalAt row constantColumn firstColumn secondColumn
    thirdColumn

def validateThreeSelectorTotalAt (row : DecodedRow)
    (constantColumn firstColumn secondColumn thirdColumn : Fin row.columns) :
    Option (ValidatedThreeSelectorTotalRow row) :=
  if shape : IsThreeSelectorTotalAt row constantColumn firstColumn
      secondColumn thirdColumn then
    some ⟨constantColumn, firstColumn, secondColumn, thirdColumn, shape⟩
  else
    none

def selectorTotalPoint (constant first second third : F) : Fin 13 → F :=
  productPoint constant 0 0 (-constant + first + second + third)

theorem rowPoint_eq_selectorTotalPoint
    (row : DecodedRow) (validated : ValidatedThreeSelectorTotalRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      selectorTotalPoint
        (assignment validated.constantColumn)
        (assignment validated.firstColumn)
        (assignment validated.secondColumn)
        (assignment validated.thirdColumn) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [selectorTotalPoint, productPoint, sparsePoint, Role.index,
      Fin.one_mul, Fin.zero_add]
  · by_cases cPort : port = Role.c.index
    · subst port
      simp only [rowPoint, action]
      rw [validated.shape.2.1]
      simp [selectorTotalPoint, productPoint, sparsePoint, Role.index,
        Fin.one_mul, Fin.zero_add, Lean.Grind.Fin.neg_mul]
    · simp only [rowPoint, action]
      rw [validated.shape.2.2 port generalPort cPort]
      have generalPortValue : port ≠ (1 : Fin 13) := by
        simpa only [Role.index] using generalPort
      have cPortValue : port ≠ (4 : Fin 13) := by
        simpa only [Role.index] using cPort
      simp [selectorTotalPoint, productPoint, sparsePoint, Role.index,
        generalPortValue, cPortValue]

theorem residual_eq_selectorGap
    (row : DecodedRow) (validated : ValidatedThreeSelectorTotalRow row)
    (assignment : Fin row.columns → F) :
    Boolean.residual row assignment =
      -(assignment validated.constantColumn *
        (-assignment validated.constantColumn +
          assignment validated.firstColumn +
          assignment validated.secondColumn +
          assignment validated.thirdColumn)) := by
  rw [Boolean.residual,
    rowPoint_eq_selectorTotalPoint row validated assignment,
    selectorTotalPoint,
    evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint, Role.index,
    Fin.mul_zero, Fin.zero_add]

/-- Fixed three-arm adapter into the generic finite selector semantics. -/
def threeWeights (first second third : F) : Fin 3 → F :=
  Fin.cases first (Fin.cases second (Fin.cases third Fin.elim0))

theorem selectorSum_three (first second third : F) :
    selectorSum (threeWeights first second third) =
      first + (second + (third + 0)) := by
  rfl

private theorem selectorLinear_eq_sumSubOne (first second third : F) :
    -1 + first + second + third =
      selectorSum (threeWeights first second third) + -1 := by
  rw [selectorSum_three, Fin.add_zero]
  calc
    -1 + first + second + third =
        (-1 + (first + second)) + third := by
          rw [fadd_assoc (-1) first second]
    _ = -1 + ((first + second) + third) := by
          rw [fadd_assoc (-1) (first + second) third]
    _ = (first + second + third) + -1 :=
          fadd_comm _ _
    _ = first + (second + third) + -1 := by
          rw [fadd_assoc first second third]

theorem selectorGap_eq_zero_iff_total
    (constant first second third : F)
    (constantOne : constant = 1) :
    -(constant * (-constant + first + second + third)) = 0 ↔
      SelectorTotal (threeWeights first second third) := by
  subst constant
  rw [Fin.one_mul, selectorLinear_eq_sumSubOne]
  unfold SelectorTotal
  constructor
  · intro negated
    have sumSubOneZero :
        selectorSum (threeWeights first second third) + -1 = 0 := by
      have := congrArg (fun value : F => -value) negated
      simpa only [Lean.Grind.AddCommGroup.neg_neg,
        Lean.Grind.AddCommGroup.neg_zero] using this
    calc
      selectorSum (threeWeights first second third) =
          selectorSum (threeWeights first second third) + 0 := by
            rw [Fin.add_zero]
      _ = selectorSum (threeWeights first second third) + (-1 + 1) := by
            rw [Lean.Grind.Fin.neg_add_cancel]
      _ = (selectorSum (threeWeights first second third) + -1) + 1 := by
            rw [fadd_assoc]
      _ = 0 + 1 := by rw [sumSubOneZero]
      _ = 1 := Fin.zero_add _
  · intro total
    rw [total, fadd_neg_cancel, Lean.Grind.AddCommGroup.neg_zero]

/-- Coefficient-only classifier for an ordinary selector-gated R1CS product
row. The `A`, `B`, and `C` ports remain arbitrary decoded linear forms. -/
def IsProductGateAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) : Prop :=
  (row.port Role.generalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.a.index →
      port ≠ Role.b.index →
      port ≠ Role.c.index →
      (row.port port).terms = []

instance (row : DecodedRow) (selectorColumn : Fin row.columns) :
    Decidable (IsProductGateAt row selectorColumn) := by
  unfold IsProductGateAt
  infer_instance

structure ValidatedProductGateRow (row : DecodedRow) where
  selectorColumn : Fin row.columns
  shape : IsProductGateAt row selectorColumn

def validateProductGateAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) :
    Option (ValidatedProductGateRow row) :=
  if shape : IsProductGateAt row selectorColumn then
    some ⟨selectorColumn, shape⟩
  else
    none

def sourceResidual (row : DecodedRow)
    (assignment : Fin row.columns → F) : F :=
  action (row.port Role.a.index) assignment *
      action (row.port Role.b.index) assignment +
    -action (row.port Role.c.index) assignment

theorem rowPoint_eq_productPoint
    (row : DecodedRow) (validated : ValidatedProductGateRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      productPoint
        (assignment validated.selectorColumn)
        (action (row.port Role.a.index) assignment)
        (action (row.port Role.b.index) assignment)
        (action (row.port Role.c.index) assignment) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [productPoint, sparsePoint, Role.index, Fin.one_mul, Fin.zero_add]
  · by_cases aPort : port = Role.a.index
    · subst port
      simp [rowPoint, productPoint, sparsePoint, Role.index]
    · by_cases bPort : port = Role.b.index
      · subst port
        simp [rowPoint, productPoint, sparsePoint, Role.index]
      · by_cases cPort : port = Role.c.index
        · subst port
          simp [rowPoint, productPoint, sparsePoint, Role.index]
        · simp only [rowPoint, action]
          rw [validated.shape.2 port generalPort aPort bPort cPort]
          have generalPortValue : port ≠ (1 : Fin 13) := by
            simpa only [Role.index] using generalPort
          have aPortValue : port ≠ (2 : Fin 13) := by
            simpa only [Role.index] using aPort
          have bPortValue : port ≠ (3 : Fin 13) := by
            simpa only [Role.index] using bPort
          have cPortValue : port ≠ (4 : Fin 13) := by
            simpa only [Role.index] using cPort
          simp [productPoint, sparsePoint, Role.index, generalPortValue,
            aPortValue, bPortValue, cPortValue]

theorem residual_eq_gatedSource
    (row : DecodedRow) (validated : ValidatedProductGateRow row)
    (assignment : Fin row.columns → F) :
    Boolean.residual row assignment =
      assignment validated.selectorColumn * sourceResidual row assignment := by
  rw [Boolean.residual,
    rowPoint_eq_productPoint row validated assignment,
    evaluate_productPoint]
  simp [productResidual, productPoint, sparsePoint, Role.index,
    sourceResidual]
  rw [fmul_assoc, ← fmul_neg, ← fmul_add]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition
