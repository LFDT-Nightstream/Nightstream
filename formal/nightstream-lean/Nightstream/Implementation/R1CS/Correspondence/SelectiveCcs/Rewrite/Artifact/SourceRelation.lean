import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.GroupedProduct

/-!
Contract: source-relation semantics for decoded grouped-product artifacts.

Assurance tier: model-level.

Owns: evaluation of decoded source R1CS rows and executable rewrite steps on
one explicit source assignment and one explicit derived-value assignment.

Does not own: a generated source-row artifact, Rust conformance, source-row
coverage, final low-norm encoding, or proof that a rewrite is equivalent to
the source rows it replaces.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

/-- Semantic value of one decoded source linear combination. -/
abbrev linearValue {columns : Nat}
    (value : DecodedSourceLinearCombination columns)
    (assignment : Fin columns → F) (constantWire : F) : F :=
  directSourceLinearValue value assignment constantWire

/-- One decoded source R1CS equation holds on the supplied assignment. -/
def RowHolds {rows columns : Nat}
    (row : DecodedSourceR1csRow rows columns)
    (assignment : Fin columns → F) (constantWire : F) : Prop :=
  linearValue row.a assignment constantWire *
      linearValue row.b assignment constantWire =
    linearValue row.c assignment constantWire

/-- Additive residual of one decoded source R1CS equation. -/
def rowResidual {rows columns : Nat}
    (row : DecodedSourceR1csRow rows columns)
    (assignment : Fin columns → F) (constantWire : F) : F :=
  linearValue row.a assignment constantWire *
      linearValue row.b assignment constantWire -
    linearValue row.c assignment constantWire

theorem rowHolds_iff_residual_zero {rows columns : Nat}
    (row : DecodedSourceR1csRow rows columns)
    (assignment : Fin columns → F) (constantWire : F) :
    RowHolds row assignment constantWire ↔
      rowResidual row assignment constantWire = 0 := by
  unfold RowHolds rowResidual
  rw [Lean.Grind.AddCommGroup.sub_eq_zero_iff]

/-- Every row in a decoded source fragment holds on one assignment. -/
def RowsHold {rows columns : Nat}
    (sourceRows : List (DecodedSourceR1csRow rows columns))
    (assignment : Fin columns → F) (constantWire : F) : Prop :=
  ∀ row ∈ sourceRows, RowHolds row assignment constantWire

theorem rowHolds_of_rowsHold {rows columns : Nat}
    {sourceRows : List (DecodedSourceR1csRow rows columns)}
    {assignment : Fin columns → F} {constantWire : F}
    (holds : RowsHold sourceRows assignment constantWire)
    (index : Fin sourceRows.length) :
    RowHolds (sourceRows.get index) assignment constantWire :=
  holds _ (List.get_mem sourceRows index)

/-- Value of one scaled product in an executable rewrite step. -/
def factorValue {columns : Nat}
    (factor : DecodedFactor columns)
    (assignment : Fin columns → F) (constantWire : F) : F :=
  factor.coefficient * linearValue factor.left assignment constantWire *
    linearValue factor.right assignment constantWire

/-- Ordered contribution of all products in one executable rewrite step. -/
def factorSum {columns : Nat}
    (assignment : Fin columns → F) (constantWire : F) :
    List (DecodedFactor columns) → F
  | [] => 0
  | factor :: tail =>
      factorValue factor assignment constantWire +
        factorSum assignment constantWire tail

/-- Value selected by a source or derived rewrite output. -/
def outputValue {columns : Nat}
    (assignment : Fin columns → F) (constantWire : F)
    (derived : Nat → F) : DecodedOutput columns → F
  | .source value => linearValue value assignment constantWire
  | .derivedProductSum index => derived index

/-- Value selected by an optional derived predecessor. -/
def previousValue (derived : Nat → F) : Option Nat → F
  | none => 0
  | some index => derived index

/-- Exact source-level equation implemented by one decoded rewrite step. -/
def StepHolds {rows columns : Nat}
    (step : DecodedStep rows columns)
    (assignment : Fin columns → F) (constantWire : F)
    (derived : Nat → F) : Prop :=
  outputValue assignment constantWire derived step.output =
    previousValue derived step.previous +
      (linearValue step.base assignment constantWire +
        factorSum assignment constantWire step.factors)

/-- The decoded step is the existing grouped-product model after evaluation. -/
def evaluatedStep {rows columns : Nat}
    (step : DecodedStep rows columns)
    (assignment : Fin columns → F) (constantWire : F)
    (derived : Nat → F) : GroupedProduct.Step where
  base := linearValue step.base assignment constantWire
  products := step.factors.map fun factor =>
    { left := factor.coefficient *
        linearValue factor.left assignment constantWire
      right := linearValue factor.right assignment constantWire }
  output := outputValue assignment constantWire derived step.output

private theorem groupedProductSum_map {columns : Nat}
    (factors : List (DecodedFactor columns))
    (assignment : Fin columns → F) (constantWire : F) :
    GroupedProduct.productSum
        (factors.map fun factor =>
          { left := factor.coefficient *
              linearValue factor.left assignment constantWire
            right := linearValue factor.right assignment constantWire }) =
      factorSum assignment constantWire factors := by
  induction factors with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, GroupedProduct.productSum,
        GroupedProduct.Product.value, factorSum, factorValue,
        inductionHypothesis]

/-- Source-level step semantics are exactly the generic grouped-product step
semantics. -/
theorem stepHolds_iff_grouped {rows columns : Nat}
    (step : DecodedStep rows columns)
    (assignment : Fin columns → F) (constantWire : F)
    (derived : Nat → F) :
    StepHolds step assignment constantWire derived ↔
      (evaluatedStep step assignment constantWire derived).holds
        (previousValue derived step.previous) := by
  unfold StepHolds evaluatedStep GroupedProduct.Step.holds
    GroupedProduct.StepSpec.contribution
  rw [groupedProductSum_map]

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
