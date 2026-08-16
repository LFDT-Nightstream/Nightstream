import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.EvaluationRowBridge

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
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.EvaluationRowBridge
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

/-- Active algebraic equation of one fixed final fixture row. -/
def FixedEquation (index : Fin 6)
    (assignment : Fin finalColumnCount → F) : Prop :=
  FiveProductEquation (decodedFixedRow index).ports assignment

def fixtureSourceAssignment (assignment : Fin finalColumnCount → F) :
    Fin sourceColumnCount → F :=
  sourceAssignment finalColumnCount_positive decodedSourceSlots
    decodedSourceDefinitions sourceFuel assignment

def fixtureDerivedAssignment (assignment : Fin finalColumnCount → F) :
    Nat → F :=
  derivedAssignment decodedDerivedSlots assignment

/-- The exact fixed-row equation is the exact decoded source recurrence on
the source and derived values decoded from the same final assignment. -/
theorem fixedEquation_iff_stepHolds
    (index : Fin 6) (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1) :
    FixedEquation index assignment ↔
      StepHolds (decodedStep index) (fixtureSourceAssignment assignment) 1
        (fixtureDerivedAssignment assignment) := by
  unfold FixedEquation fixtureSourceAssignment fixtureDerivedAssignment
  exact fiveProductEquation_iff_stepHolds finalColumnCount_positive
    (decodedFixedRow index).ports (decodedStep index) decodedSourceSlots
    decodedSourceDefinitions decodedDerivedSlots sourceFuel
    (generated_step_images_match index) assignment constantOne

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
