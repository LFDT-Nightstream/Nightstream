import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalAssignment

/-!
Contract: same-assignment bridge from the six exact final grouped-product
rows to their six source-level recurrence equations.

Assurance tier: artifact-checked fixture.

Owns: factor padding, equality between a five-port final row and one decoded
rewrite step, and active-row transport for the deterministic fixture.

Does not own: source-only temporary reconstruction, low-norm validity,
production-family coverage, or permission to remove production coordinates.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowSourceBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_comm⟩

/-- Value of one factor position, padded with zero after the factor list. -/
def factorValueAt {columns : Nat}
    (step : DecodedStep sourceRowCount columns)
    (assignment : Fin columns → F) (index : Nat) : F :=
  match step.factors[index]? with
  | some factor => factorValue factor assignment 1
  | none => 0

/-- Exact five-factor value accepted by one evaluation row. -/
def paddedFactorSum {columns : Nat}
    (step : DecodedStep sourceRowCount columns)
    (assignment : Fin columns → F) : F :=
  factorValueAt step assignment 0 + factorValueAt step assignment 1 +
    factorValueAt step assignment 2 + factorValueAt step assignment 3 +
    factorValueAt step assignment 4

/-- Padding to five positions preserves every factor sum whose decoded list
has the enforced length bound. -/
theorem factorSum_eq_padded {columns : Nat}
    (step : DecodedStep sourceRowCount columns)
    (assignment : Fin columns → F) :
    factorSum assignment 1 step.factors = paddedFactorSum step assignment := by
  have bound := step.factorsBound
  cases factorsEq : step.factors with
  | nil => simp [paddedFactorSum, factorValueAt, factorSum, factorsEq]
  | cons factor0 tail0 =>
      cases tail0 with
      | nil => simp [paddedFactorSum, factorValueAt, factorSum, factorsEq]
      | cons factor1 tail1 =>
          cases tail1 with
          | nil => simp [paddedFactorSum, factorValueAt, factorSum, factorsEq]
          | cons factor2 tail2 =>
              cases tail2 with
              | nil =>
                  simp [paddedFactorSum, factorValueAt, factorSum, factorsEq]
                  ac_rfl
              | cons factor3 tail3 =>
                  cases tail3 with
                  | nil =>
                      simp [paddedFactorSum, factorValueAt, factorSum, factorsEq]
                      ac_rfl
                  | cons factor4 tail4 =>
                      cases tail4 with
                      | nil =>
                          simp [paddedFactorSum, factorValueAt, factorSum,
                            factorsEq]
                          ac_rfl
                      | cons factor5 tail5 =>
                          simp [factorsEq] at bound

private theorem restoreDifference (output base previous value : F)
    (difference : output - base - previous = value) :
    output = previous + (base + value) := by
  simp only [Fin.sub_eq_add_neg] at difference
  have baseCancel : -base + base = 0 :=
    Lean.Grind.Fin.neg_add_cancel base
  have previousCancel : -previous + previous = 0 :=
    Lean.Grind.Fin.neg_add_cancel previous
  calc
    output = previous + (base + ((output + -base) + -previous)) := by
      calc
        output = output + (-base + base) + (-previous + previous) := by
          simp only [baseCancel, previousCancel, Fin.add_zero]
        _ = previous + (base + ((output + -base) + -previous)) := by
          ac_rfl
    _ = previous + (base + value) := by rw [difference]

private theorem removeDifference (output base previous value : F)
    (step : output = previous + (base + value)) :
    output - base - previous = value := by
  rw [step]
  simp only [Fin.sub_eq_add_neg]
  have baseCancel : -base + base = 0 :=
    Lean.Grind.Fin.neg_add_cancel base
  have previousCancel : -previous + previous = 0 :=
    Lean.Grind.Fin.neg_add_cancel previous
  calc
    (previous + (base + value) + -base) + -previous =
        value + (-base + base) + (-previous + previous) := by
      ac_rfl
    _ = value := by simp only [baseCancel, previousCancel, Fin.add_zero]

private theorem difference_iff_step
    (output base previous value : F) :
    output - base - previous = value ↔
      output = previous + (base + value) := by
  constructor
  · exact restoreDifference output base previous value
  · exact removeDifference output base previous value

/-- Active algebraic equation of one fixed final fixture row. -/
def FixedEquation (index : Fin 6)
    (assignment : Fin finalColumnCount → F) : Prop :=
  action ((decodedFixedRow index).ports Role.c.index) assignment =
    fiveProductSum
      (action ((decodedFixedRow index).ports Role.bit.index) assignment)
      (action ((decodedFixedRow index).ports Role.a.index) assignment)
      (action ((decodedFixedRow index).ports Role.b.index) assignment)
      (action ((decodedFixedRow index).ports Role.sboxInput.index) assignment)
      (action ((decodedFixedRow index).ports Role.centeredUnit.index) assignment)
      (action ((decodedFixedRow index).ports Role.canonicalDigit.index) assignment)
      (action ((decodedFixedRow index).ports Role.canonicalBorrow.index) assignment)
      (action ((decodedFixedRow index).ports Role.canonicalNextBorrow.index) assignment)
      (action ((decodedFixedRow index).ports Role.canonicalBoundDigit.index) assignment)
      (action ((decodedFixedRow index).ports Role.evalTailRight.index) assignment)

def fixtureSourceAssignment (assignment : Fin finalColumnCount → F) :
    Fin sourceColumnCount → F :=
  sourceAssignment finalColumnCount_positive decodedSourceSlots
    decodedSourceDefinitions sourceFuel assignment

def fixtureDerivedAssignment (assignment : Fin finalColumnCount → F) :
    Nat → F :=
  derivedAssignment decodedDerivedSlots assignment

private theorem factorPortProduct
    (index : Fin 6) (factor : Fin 5)
    (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1) :
    action
          ((decodedFixedRow index).ports (factorRoles factor).1.index)
          assignment *
        action
          ((decodedFixedRow index).ports (factorRoles factor).2.index)
          assignment =
      factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
        factor.val := by
  have actions := generated_factor_actions_eq_source_images index factor assignment
  rw [actions.1, actions.2]
  unfold factorValueAt fixtureSourceAssignment
  cases factorAt : (decodedStep index).factors[factor.val]? with
  | none =>
      simp only [factorFormAt, factorAt, Form.evaluate_zero, Fin.zero_mul]
  | some decodedFactor =>
      have values := evaluate_factorForms finalColumnCount_positive
        decodedSourceSlots decodedSourceDefinitions sourceFuel factor.val
        (decodedStep index) decodedFactor factorAt assignment constantOne
      rw [values.1, values.2]
      rfl

/-- The exact fixed-row equation is the exact decoded source recurrence on
the source and derived values decoded from the same final assignment. -/
theorem fixedEquation_iff_stepHolds
    (index : Fin 6) (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1) :
    FixedEquation index assignment ↔
      StepHolds (decodedStep index) (fixtureSourceAssignment assignment) 1
        (fixtureDerivedAssignment assignment) := by
  have product0 :
      action ((decodedFixedRow index).ports Role.bit.index) assignment *
          action ((decodedFixedRow index).ports Role.a.index) assignment =
        factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
          0 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct index 0 assignment constantOne
  have product1 :
      action ((decodedFixedRow index).ports Role.b.index) assignment *
          action ((decodedFixedRow index).ports Role.sboxInput.index) assignment =
        factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
          1 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct index 1 assignment constantOne
  have product2 :
      action ((decodedFixedRow index).ports Role.centeredUnit.index) assignment *
          action ((decodedFixedRow index).ports Role.canonicalDigit.index) assignment =
        factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
          2 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct index 2 assignment constantOne
  have product3 :
      action ((decodedFixedRow index).ports Role.canonicalBorrow.index) assignment *
          action ((decodedFixedRow index).ports Role.canonicalNextBorrow.index) assignment =
        factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
          3 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct index 3 assignment constantOne
  have product4 :
      action ((decodedFixedRow index).ports Role.canonicalBoundDigit.index) assignment *
          action ((decodedFixedRow index).ports Role.evalTailRight.index) assignment =
        factorValueAt (decodedStep index) (fixtureSourceAssignment assignment)
          4 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct index 4 assignment constantOne
  unfold FixedEquation fiveProductSum
  rw [generated_c_action_eq_source_image index assignment]
  rw [product0, product1, product2, product3, product4]
  rw [evaluate_expectedCForm finalColumnCount_positive decodedSourceSlots
    decodedSourceDefinitions decodedDerivedSlots sourceFuel
    (decodedStep index) assignment constantOne]
  change _ =
      paddedFactorSum (decodedStep index) (fixtureSourceAssignment assignment) ↔ _
  rw [← factorSum_eq_padded (decodedStep index)
    (fixtureSourceAssignment assignment)]
  unfold StepHolds fixtureSourceAssignment fixtureDerivedAssignment
  exact difference_iff_step _ _ _ _

/-- The six exact final rows, stated without a dependent cast. -/
def ActiveRowsHold (assignment : Fin finalColumnCount → F) : Prop :=
  residual (decodedRow 0) assignment = 0 ∧
    residual (decodedRow 1) assignment = 0 ∧
    residual (decodedRow 2) assignment = 0 ∧
    residual (decodedRow 3) assignment = 0 ∧
    residual (decodedRow 4) assignment = 0 ∧
    residual (decodedRow 5) assignment = 0

private theorem fixedEquation_of_row00
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 0) = 1)
    (holds : residual (decodedRow 0) assignment = 0) :
    FixedEquation 0 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 0 assignment selectorOne).mp holds
  exact equation

private theorem fixedEquation_of_row01
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 1) = 1)
    (holds : residual (decodedRow 1) assignment = 0) :
    FixedEquation 1 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 1 assignment selectorOne).mp holds
  exact equation

private theorem fixedEquation_of_row02
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 2) = 1)
    (holds : residual (decodedRow 2) assignment = 0) :
    FixedEquation 2 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 2 assignment selectorOne).mp holds
  exact equation

private theorem fixedEquation_of_row03
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 3) = 1)
    (holds : residual (decodedRow 3) assignment = 0) :
    FixedEquation 3 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 3 assignment selectorOne).mp holds
  exact equation

private theorem fixedEquation_of_row04
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 4) = 1)
    (holds : residual (decodedRow 4) assignment = 0) :
    FixedEquation 4 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 4 assignment selectorOne).mp holds
  exact equation

private theorem fixedEquation_of_row05
    (assignment : Fin finalColumnCount → F)
    (selectorOne : assignment (selectorColumn 5) = 1)
    (holds : residual (decodedRow 5) assignment = 0) :
    FixedEquation 5 assignment := by
  have equation :=
    (generated_row_zero_iff_fiveProduct 5 assignment selectorOne).mp holds
  exact equation

/-- All six generated branch selectors are active. -/
def SelectorsOne (assignment : Fin finalColumnCount → F) : Prop :=
  assignment (selectorColumn 0) = 1 ∧
    assignment (selectorColumn 1) = 1 ∧
    assignment (selectorColumn 2) = 1 ∧
    assignment (selectorColumn 3) = 1 ∧
    assignment (selectorColumn 4) = 1 ∧
    assignment (selectorColumn 5) = 1

/-- Satisfaction of the six generated rows gives their six exact fixed
equations on the same assignment. -/
theorem fixedEquations_of_activeRows
    (assignment : Fin finalColumnCount → F)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ∀ index : Fin 6, FixedEquation index assignment := by
  rcases selectors with
    ⟨selector00, selector01, selector02, selector03, selector04, selector05⟩
  rcases holds with ⟨row00, row01, row02, row03, row04, row05⟩
  intro index
  exact Fin.cases
    (fixedEquation_of_row00 assignment selector00 row00)
    (Fin.cases
      (fixedEquation_of_row01 assignment selector01 row01)
      (Fin.cases
        (fixedEquation_of_row02 assignment selector02 row02)
        (Fin.cases
          (fixedEquation_of_row03 assignment selector03 row03)
          (Fin.cases
            (fixedEquation_of_row04 assignment selector04 row04)
            (Fin.cases
              (fixedEquation_of_row05 assignment selector05 row05)
              (fun impossible => Fin.elim0 impossible)))))) index

/-- The six active final rows imply all six decoded source recurrences on the
source values extracted from that same final assignment. -/
theorem stepHolds_of_activeRows
    (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ∀ index : Fin 6,
      StepHolds (decodedStep index) (fixtureSourceAssignment assignment) 1
        (fixtureDerivedAssignment assignment) := by
  intro index
  exact (fixedEquation_iff_stepHolds index assignment constantOne).mp
    (fixedEquations_of_activeRows assignment selectors holds index)

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowSourceBridge
