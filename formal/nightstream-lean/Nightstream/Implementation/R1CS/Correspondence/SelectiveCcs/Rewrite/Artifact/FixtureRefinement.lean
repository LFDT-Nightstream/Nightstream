import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Evaluation
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.Decoder
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceRelation

/-!
Contract: exact artifact refinement for one complete grouped-product rewrite.

Assurance tier: artifact-checked fixture.

Owns: fail-closed decoding of all six executable Rust steps, their exact
source R1CS rows, source and derived low-norm slots, and all six final matrix
rows; their exact emitted-row join; coefficient-derived evaluation-row
classification; the source-to-final linear images; and the five-product
equation of each active row.

Does not own: production-family coverage, complete recursive or terminal
conformance, norm enforcement, constraint necessity, or permission to remove
a production row or coordinate.

Emits constraints: no. Rust regenerates this artifact from the exact
provenance projector after that projector reproduces every final row.

| Fixture obligation | Exact result | Assurance tier |
|---|---|---|
| executable source data | six bounded, canonical steps with at most five factors | artifact-checked fixture |
| final row data | six decoded 13-port rows | artifact-checked fixture |
| provenance join | equal emitted rows, product-sum tag, arm zero | artifact-checked fixture |
| assignment image | every source and derived port equals its decoded low-norm image | proved from coefficients |
| row algebra | active row zero iff its five-product equation | proved from coefficients |
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Evaluation
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Wire

def decodedSteps : List (DecodedStep sourceRowCount sourceColumnCount) :=
  (rawSteps.mapM (decodeStep sourceRowCount sourceColumnCount)).getD []

def decodedRows : List DecodedRow :=
  (rawRows.mapM decodeRow).getD []

structure FixedDecodedRow where
  ports : Fin 13 → DecodedPort finalColumnCount

def decodeFixedRow (raw : RawRow) : Option FixedDecodedRow := do
  if raw.columns = finalColumnCount then
    let ports ← raw.ports.mapM (decodePort finalColumnCount)
    if portCount : ports.length = 13 then
      pure
        { ports := fun port => ports.get ⟨port.val, by
            rw [portCount]
            exact port.isLt⟩ }
    else
      none
  else
    none

def emptyPort : DecodedPort finalColumnCount where
  terms := []
  columnsStrict := by simp
  coefficientsNonzero := by simp

def emptyFixedRow : FixedDecodedRow where
  ports := fun _ => emptyPort

def decodedFixedRow00 : FixedDecodedRow :=
  (decodeFixedRow rawRow00).getD emptyFixedRow

def decodedFixedRow01 : FixedDecodedRow :=
  (decodeFixedRow rawRow01).getD emptyFixedRow

def decodedFixedRow02 : FixedDecodedRow :=
  (decodeFixedRow rawRow02).getD emptyFixedRow

def decodedFixedRow03 : FixedDecodedRow :=
  (decodeFixedRow rawRow03).getD emptyFixedRow

def decodedFixedRow04 : FixedDecodedRow :=
  (decodeFixedRow rawRow04).getD emptyFixedRow

def decodedFixedRow05 : FixedDecodedRow :=
  (decodeFixedRow rawRow05).getD emptyFixedRow

def decodedFixedRows : List FixedDecodedRow :=
  [decodedFixedRow00, decodedFixedRow01, decodedFixedRow02,
    decodedFixedRow03, decodedFixedRow04, decodedFixedRow05]

def decodedSourceSlots :
    List (DecodedSourceSlot sourceColumnCount finalColumnCount) :=
  (rawSourceSlots.mapM
    (decodeSourceSlot sourceColumnCount finalColumnCount)).getD []

def decodedSourceDefinitions :
    List (DecodedSourceDefinition sourceColumnCount) :=
  (rawSourceDefinitions.mapM
    (decodeSourceDefinition sourceColumnCount)).getD []

def decodedSourceRows :
    List (DecodedSourceR1csRow sourceRowCount sourceColumnCount) :=
  (rawSourceRows.mapM
    (decodeSourceR1csRow sourceRowCount sourceColumnCount)).getD []

def decodedDerivedSlots : List (DecodedDerivedSlot finalColumnCount) :=
  (rawDerivedSlots.mapM (decodeDerivedSlot finalColumnCount)).getD []

theorem decodedSteps_length : decodedSteps.length = 6 := by
  decide

theorem decodedRows_length : decodedRows.length = 6 := by
  decide

theorem decodedFixedRows_length : decodedFixedRows.length = 6 := by
  decide

theorem decodedSourceSlots_length : decodedSourceSlots.length = 27 := by
  decide

theorem decodedSourceDefinitions_length :
    decodedSourceDefinitions.length = 1 := by
  decide

theorem decodedSourceRows_length : decodedSourceRows.length = 33 := by
  native_decide

/-- The artifact contains each claimed source row exactly once, with no gap or
extra row outside the rewrite interval `[4, 37)`. -/
theorem decodedSourceRows_exact_interval :
    decodedSourceRows.map (fun row => row.row.val) = List.range' 4 33 := by
  native_decide

theorem decodedDerivedSlots_length : decodedDerivedSlots.length = 3 := by
  decide

def decodedStep (index : Fin 6) :
    DecodedStep sourceRowCount sourceColumnCount :=
  decodedSteps.get ⟨index.val, by
    rw [decodedSteps_length]
    exact index.isLt⟩

def decodedRow (index : Fin 6) : DecodedRow :=
  decodedRows.get ⟨index.val, by
    rw [decodedRows_length]
    exact index.isLt⟩

def decodedFixedRow (index : Fin 6) : FixedDecodedRow :=
  decodedFixedRows.get ⟨index.val, by
    rw [decodedFixedRows_length]
    exact index.isLt⟩

theorem decodedRow_columns :
    ∀ index : Fin 6, (decodedRow index).columns = 1458 := by
  decide

def sourceFuel : Nat := 2

theorem finalColumnCount_positive : 0 < finalColumnCount := by
  decide

def selectorColumn (index : Fin 6) : Fin (decodedRow index).columns :=
  ⟨54, by
    have columns := decodedRow_columns index
    omega⟩

/-- The join uses actual decoded coefficients for the evaluation selector.
The generated family and rewrite fields are checked only as separate
provenance metadata. -/
theorem generated_steps_and_rows_join :
    ∀ index : Fin 6,
      (decodedStep index).emittedRow = (decodedRow index).emittedRow.val ∧
      (decodedStep index).rewriteId = 0 ∧
      (decodedStep index).kind = RawKind.productSum ∧
      (decodedStep index).factors.length ≤ 5 ∧
      (decodedRow index).family =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire.RawFamily.productSum ∧
      (decodedRow index).arm = some arm ∧
      IsEvaluationAt (decodedRow index) (selectorColumn index) := by
  decide

private theorem generated_step_images_match_00 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow00.ports
      (decodedStep 0) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

private theorem generated_step_images_match_01 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow01.ports
      (decodedStep 1) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

private theorem generated_step_images_match_02 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow02.ports
      (decodedStep 2) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

private theorem generated_step_images_match_03 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow03.ports
      (decodedStep 3) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

private theorem generated_step_images_match_04 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow04.ports
      (decodedStep 4) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

private theorem generated_step_images_match_05 :
    PortImagesMatch finalColumnCount_positive decodedFixedRow05.ports
      (decodedStep 5) decodedSourceSlots decodedSourceDefinitions
      decodedDerivedSlots sourceFuel := by
  native_decide

/-- The exact final port coefficients are the independent expansion of every
source linear combination and every compiler-derived accumulator slot. This
is the bounded same-assignment bridge that the earlier fixture lacked. -/
theorem generated_step_images_match :
    ∀ index : Fin 6,
      PortImagesMatch finalColumnCount_positive
        (decodedFixedRow index).ports (decodedStep index)
        decodedSourceSlots decodedSourceDefinitions decodedDerivedSlots
        sourceFuel := by
  exact Fin.cases generated_step_images_match_00
    (Fin.cases generated_step_images_match_01
      (Fin.cases generated_step_images_match_02
          (Fin.cases generated_step_images_match_03
            (Fin.cases generated_step_images_match_04
            (Fin.cases generated_step_images_match_05
              (fun index => Fin.elim0 index))))))

def validatedRow (index : Fin 6) :
    ValidatedEvaluationRow (decodedRow index) where
  selectorColumn := selectorColumn index
  shape := (generated_steps_and_rows_join index).2.2.2.2.2.2

/-- Each exact generated row has precisely the independent five-product
equation when its branch selector is active. -/
theorem generated_row_zero_iff_fiveProduct
    (index : Fin 6)
    (assignment : Fin (decodedRow index).columns → F)
    (selectorOne : assignment (selectorColumn index) = 1) :
    residual (decodedRow index) assignment = 0 ↔
      action ((decodedRow index).port Role.c.index) assignment =
        Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.fiveProductSum
          (action ((decodedRow index).port Role.bit.index) assignment)
          (action ((decodedRow index).port Role.a.index) assignment)
          (action ((decodedRow index).port Role.b.index) assignment)
          (action ((decodedRow index).port Role.sboxInput.index) assignment)
          (action ((decodedRow index).port Role.centeredUnit.index) assignment)
          (action ((decodedRow index).port Role.canonicalDigit.index) assignment)
          (action ((decodedRow index).port Role.canonicalBorrow.index) assignment)
          (action ((decodedRow index).port Role.canonicalNextBorrow.index) assignment)
          (action ((decodedRow index).port Role.canonicalBoundDigit.index) assignment)
          (action ((decodedRow index).port Role.evalTailRight.index) assignment) :=
  residual_zero_iff_fiveProduct
    (decodedRow index) (validatedRow index) assignment selectorOne

theorem generated_c_action_eq_source_image
    (index : Fin 6)
    (assignment : Fin finalColumnCount → F) :
    action ((decodedFixedRow index).ports Role.c.index) assignment =
      Form.evaluate
        (expectedCForm finalColumnCount_positive decodedSourceSlots
          decodedSourceDefinitions decodedDerivedSlots sourceFuel
          (decodedStep index))
        assignment :=
  matched_port_c_action finalColumnCount_positive
    (decodedFixedRow index).ports (decodedStep index) decodedSourceSlots
    decodedSourceDefinitions decodedDerivedSlots sourceFuel
    (generated_step_images_match index) assignment

theorem generated_factor_actions_eq_source_images
    (index : Fin 6) (factor : Fin 5)
    (assignment : Fin finalColumnCount → F) :
    action
          ((decodedFixedRow index).ports
            (factorRoles factor).1.index)
          assignment =
        Form.evaluate
          (factorFormAt finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel factor.val true
            (decodedStep index)) assignment ∧
      action
          ((decodedFixedRow index).ports
            (factorRoles factor).2.index)
          assignment =
        Form.evaluate
          (factorFormAt finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel factor.val false
            (decodedStep index)) assignment :=
  matched_port_factor_actions finalColumnCount_positive
    (decodedFixedRow index).ports (decodedStep index) decodedSourceSlots
    decodedSourceDefinitions decodedDerivedSlots sourceFuel
    (generated_step_images_match index) factor assignment

/-- The generated factor that uses source column 15 crosses the exported
affine definition `column 15 = column 3 + 3`. -/
def generatedAffineFactor : DecodedFactor sourceColumnCount :=
  (decodedStep 2).factors.get ⟨0, by decide⟩

theorem generatedAffineFactor_at :
    (decodedStep 2).factors[0]? = some generatedAffineFactor := by
  rw [List.getElem?_eq_getElem (by decide)]
  rfl

theorem generated_affine_definition_exact :
    (decodedSourceDefinitions.get ⟨0, by
      rw [decodedSourceDefinitions_length]
      decide⟩).target.val = 15 ∧
    (decodedSourceDefinitions.get ⟨0, by
      rw [decodedSourceDefinitions_length]
      decide⟩).value.constant = (3 : F) ∧
    (decodedSourceDefinitions.get ⟨0, by
      rw [decodedSourceDefinitions_length]
      decide⟩).value.terms.length = 1 := by
  decide

/-- The exact final factor ports evaluate the source recurrence after the
nonempty affine definition is recursively expanded. -/
theorem generated_affine_factor_actions_eq_source_values
    (assignment : Fin finalColumnCount → F) :
    action
          ((decodedFixedRow 2).ports
            (factorRoles 0).1.index)
          assignment =
        generatedAffineFactor.coefficient *
          sourceLinearValue finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel generatedAffineFactor.left
            assignment ∧
      action
          ((decodedFixedRow 2).ports
            (factorRoles 0).2.index)
          assignment =
        sourceLinearValue finalColumnCount_positive decodedSourceSlots
          decodedSourceDefinitions sourceFuel generatedAffineFactor.right
          assignment :=
  matched_port_factor_actions_eq_source_values finalColumnCount_positive
    (decodedFixedRow 2).ports (decodedStep 2) decodedSourceSlots
    decodedSourceDefinitions decodedDerivedSlots sourceFuel
    (generated_step_images_match 2) 0 generatedAffineFactor
    generatedAffineFactor_at assignment

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
