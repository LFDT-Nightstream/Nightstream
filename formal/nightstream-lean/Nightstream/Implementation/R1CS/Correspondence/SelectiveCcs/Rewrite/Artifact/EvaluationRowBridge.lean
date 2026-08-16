import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Evaluation
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalAssignment

/-!
Contract: generic same-assignment bridge from one decoded low-norm
evaluation row to one decoded grouped-product source recurrence.

Assurance tier: model-level artifact interpreter.

Owns: five-factor padding, matched final-port evaluation, and the equivalence
between an active final row and its source recurrence on values decoded from
the same final assignment.

Does not own: a concrete generated artifact, selector dispatch, source-row
coverage, compiler rewrite coverage, norm enforcement, or lifecycle
soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.EvaluationRowBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Evaluation
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_comm⟩

/-- Value of one factor position, padded with zero after the factor list. -/
def factorValueAt {rows columns : Nat}
    (step : DecodedStep rows columns)
    (assignment : Fin columns → F) (index : Nat) : F :=
  match step.factors[index]? with
  | some factor => factorValue factor assignment 1
  | none => 0

/-- Exact five-factor value accepted by one evaluation row. -/
def paddedFactorSum {rows columns : Nat}
    (step : DecodedStep rows columns)
    (assignment : Fin columns → F) : F :=
  factorValueAt step assignment 0 + factorValueAt step assignment 1 +
    factorValueAt step assignment 2 + factorValueAt step assignment 3 +
    factorValueAt step assignment 4

/-- Padding to five positions preserves every factor sum whose decoded list
has the decoder-enforced length bound. -/
theorem factorSum_eq_padded {rows columns : Nat}
    (step : DecodedStep rows columns)
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
                      simp [paddedFactorSum, factorValueAt, factorSum,
                        factorsEq]
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

/-- The active five-product equation of arbitrary decoded final ports. -/
def FiveProductEquation {columns : Nat}
    (ports : Fin 13 → DecodedPort columns)
    (assignment : Fin columns → F) : Prop :=
  action (ports Role.c.index) assignment =
    fiveProductSum
      (action (ports Role.bit.index) assignment)
      (action (ports Role.a.index) assignment)
      (action (ports Role.b.index) assignment)
      (action (ports Role.sboxInput.index) assignment)
      (action (ports Role.centeredUnit.index) assignment)
      (action (ports Role.canonicalDigit.index) assignment)
      (action (ports Role.canonicalBorrow.index) assignment)
      (action (ports Role.canonicalNextBorrow.index) assignment)
      (action (ports Role.canonicalBoundDigit.index) assignment)
      (action (ports Role.evalTailRight.index) assignment)

private theorem factorPortProduct
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat)
    (imageMatch :
      PortImagesMatch columnsPositive ports step slots definitions derived fuel)
    (factor : Fin 5) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    action (ports (factorRoles factor).1.index) assignment *
        action (ports (factorRoles factor).2.index) assignment =
      factorValueAt step
        (sourceAssignment columnsPositive slots definitions fuel assignment)
        factor.val := by
  have actions := matched_port_factor_actions columnsPositive ports step slots
    definitions derived fuel imageMatch factor assignment
  rw [actions.1, actions.2]
  unfold factorValueAt
  cases factorAt : step.factors[factor.val]? with
  | none =>
      simp only [factorFormAt, factorAt, Form.evaluate_zero, Fin.zero_mul]
  | some decodedFactor =>
      have values := evaluate_factorForms columnsPositive slots definitions fuel
        factor.val step decodedFactor factorAt assignment constantOne
      rw [values.1, values.2]
      rfl

/-- A matched five-product equation is exactly the decoded source recurrence
on source and derived values read from the same final assignment. -/
theorem fiveProductEquation_iff_stepHolds
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat)
    (imageMatch :
      PortImagesMatch columnsPositive ports step slots definitions derived fuel)
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    FiveProductEquation ports assignment ↔
      StepHolds step
        (sourceAssignment columnsPositive slots definitions fuel assignment) 1
        (derivedAssignment derived assignment) := by
  have product0 :
      action (ports Role.bit.index) assignment *
          action (ports Role.a.index) assignment =
        factorValueAt step
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          0 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct columnsPositive ports step slots definitions derived fuel
        imageMatch 0 assignment constantOne
  have product1 :
      action (ports Role.b.index) assignment *
          action (ports Role.sboxInput.index) assignment =
        factorValueAt step
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          1 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct columnsPositive ports step slots definitions derived fuel
        imageMatch 1 assignment constantOne
  have product2 :
      action (ports Role.centeredUnit.index) assignment *
          action (ports Role.canonicalDigit.index) assignment =
        factorValueAt step
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          2 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct columnsPositive ports step slots definitions derived fuel
        imageMatch 2 assignment constantOne
  have product3 :
      action (ports Role.canonicalBorrow.index) assignment *
          action (ports Role.canonicalNextBorrow.index) assignment =
        factorValueAt step
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          3 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct columnsPositive ports step slots definitions derived fuel
        imageMatch 3 assignment constantOne
  have product4 :
      action (ports Role.canonicalBoundDigit.index) assignment *
          action (ports Role.evalTailRight.index) assignment =
        factorValueAt step
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          4 := by
    simpa [factorRoles, factorRolePairs] using
      factorPortProduct columnsPositive ports step slots definitions derived fuel
        imageMatch 4 assignment constantOne
  unfold FiveProductEquation fiveProductSum
  rw [matched_port_c_action columnsPositive ports step slots definitions derived
    fuel imageMatch assignment]
  rw [product0, product1, product2, product3, product4]
  rw [evaluate_expectedCForm columnsPositive slots definitions derived fuel step
    assignment constantOne]
  change _ = paddedFactorSum step
      (sourceAssignment columnsPositive slots definitions fuel assignment) ↔ _
  rw [← factorSum_eq_padded step
    (sourceAssignment columnsPositive slots definitions fuel assignment)]
  unfold StepHolds
  exact difference_iff_step _ _ _ _

/-- One active decoded evaluation row is exactly its decoded source
recurrence on values read from the same final assignment. -/
theorem residual_zero_iff_stepHolds
    {rows sourceColumns : Nat}
    (row : DecodedRow) (validated : ValidatedEvaluationRow row)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot row.columns))
    (fuel : Nat)
    (imageMatch : StepImagesMatch row step slots definitions derived fuel)
    (assignment : Fin row.columns → F)
    (constantOne : assignment ⟨0, row.columnsPositive⟩ = 1)
    (selectorOne : assignment validated.selectorColumn = 1) :
    residual row assignment = 0 ↔
      StepHolds step
        (sourceAssignment row.columnsPositive slots definitions fuel assignment)
        1 (derivedAssignment derived assignment) := by
  rw [residual_zero_iff_fiveProduct row validated assignment selectorOne]
  change FiveProductEquation row.ports assignment ↔ _
  exact fiveProductEquation_iff_stepHolds row.columnsPositive row.ports step
    slots definitions derived fuel imageMatch assignment constantOne

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.EvaluationRowBridge
