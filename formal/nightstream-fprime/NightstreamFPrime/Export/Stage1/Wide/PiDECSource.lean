import NightstreamFPrime.Export.Stage1.PiDECDirectPlan
import NightstreamFPrime.Layout.Stage1.Wide.PiDECStarts
import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionValues

/-! Read the unchanged PiDEC fields from the new physical source layout.
Parent products and the PiDEC suffix move by different amounts. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECSource

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open PiRLC.v1_1

abbrev Location := PiDECDirectPlan.Location

def parentCommitmentStart : Nat := CombinationFamily.stepOffset
  Layout.Stage1.Wide.PiRLCStarts.commitmentLogicalStart CombinationFamily.finalSource.val
  CommitmentCombination.blockCount CommitmentCombination.cellCount

def parentPublicInputStart : Nat := CombinationFamily.stepOffset
  Layout.Stage1.Wide.PiRLCStarts.publicInputLogicalStart CombinationFamily.finalSource.val
  PublicInputCombination.blockCount PublicInputCombination.cellCount

def parentEvalKStart : Nat := CombinationFamily.stepOffset
  Layout.Stage1.Wide.PiRLCStarts.evalKLogicalStart CombinationFamily.finalSource.val
  EvalKCombination.blockCount RingKCombination.cellCount

def parentEvalAStart : Nat := CombinationFamily.stepOffset
  Layout.Stage1.Wide.PiRLCStarts.evalALogicalStart CombinationFamily.finalSource.val
  EvalACombination.blockCount RingKCombination.cellCount

/-- The existing seven PiDEC location kinds keep their slot meaning. -/
def column : Location → Nat
  | .parentCommitment index => parentCommitmentStart + index.val
  | .parentPublicInput index => parentPublicInputStart + index.val
  | .parentEvalK index => parentEvalKStart + index.val
  | .parentEvalA index => parentEvalAStart + index.val
  | .proof index => Layout.Stage1.Wide.PiDECInputs.proofInputStart + index.val
  | .logical index => Layout.Stage1.Wide.PiDECStarts.phaseLogicalStart + index.val
  | .fresh index => Layout.Stage1.Wide.PiDECStarts.phaseFreshStart + index.val

theorem parent_starts : [parentCommitmentStart, parentPublicInputStart, parentEvalKStart, parentEvalAStart] =
    [19587528, 19593036, 19595034, 19619334] := by rfl

/-- A single global offset is invalid: parent products move within the logical
region, while the PiDEC suffix follows all PiRLC R1CS scratch fields. -/
theorem source_offsets (location : Location) :
    location.sourceColumn = column location +
      match location with
      | .parentCommitment _ | .parentPublicInput _ | .parentEvalK _ | .parentEvalA _ => 208165
      | .proof _ | .logical _ | .fresh _ => 925480 := by
  cases location with
  | parentCommitment index => change 19795693 + index.val = 19587528 + index.val + 208165; omega
  | parentPublicInput index => change 19801201 + index.val = 19593036 + index.val + 208165; omega
  | parentEvalK index => change 19803199 + index.val = 19595034 + index.val + 208165; omega
  | parentEvalA index => change 19827499 + index.val = 19619334 + index.val + 208165; omega
  | proof index => change 28421542 + index.val = 27496062 + index.val + 925480; omega
  | logical index => change 28470790 + index.val = 27545310 + index.val + 925480; omega
  | fresh index => change 28471060 + index.val = 27545580 + index.val + 925480; omega

theorem column_before_running (location : Location) :
    column location < Layout.Stage1.Wide.RunningTransitionInputs.phaseOffset := by
  cases location with
  | parentCommitment index => have bound : index.val < 1188 := index.isLt; change 19587528 + index.val < 27563400; omega
  | parentPublicInput index => have bound : index.val < 270 := index.isLt; change 19593036 + index.val < 27563400; omega
  | parentEvalK index => have bound : index.val < 108 := index.isLt; change 19595034 + index.val < 27563400; omega
  | parentEvalA index => have bound : index.val < 1512 := index.isLt; change 19619334 + index.val < 27563400; omega
  | proof index => have bound : index.val < 49248 := index.isLt; change 27496062 + index.val < 27563400; omega
  | logical index => have bound : index.val < 270 := index.isLt; change 27545310 + index.val < 27563400; omega
  | fresh index => have bound : index.val < 17820 := index.isLt; change 27545580 + index.val < 27563400; omega

/-- Read one checked location directly from the wide physical witness. -/
def value (env : Env) (location : Location) : F := env (column location)

open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Spec.Phi81Relation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem commitment_value (env : Env) (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    ((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).commitment row lane).eval env =
      value env (.parentCommitment (CombinationStep.indexOf row lane CommitmentCombination.cell)) := by
  rfl

theorem publicInput_value (env : Env) (index : Fin (FullShape logicalWidth publicFits).publicWidth) :
    ((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).publicInput index).eval env =
      value env (.parentPublicInput (CombinationStep.indexOf
        (blockCount := PublicInputCombination.blockCount) (cellCount := PublicInputCombination.cellCount)
        (PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape logicalWidth publicFits) index)
        (PiRLCAlgebra.PublicInput.publicLaneIndex index) PublicInputCombination.cell)) := by
  rfl

theorem evalK_values (env : Env) (coefficient : Fin productionShape.coefficientCount) :
    (((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).evaluation.eval_K coefficient).c0.eval env =
      value env (.parentEvalK (CombinationStep.indexOf EvalKCombination.block
        (Fin.cast EvalKCombination.coefficientCount_eq coefficient) RingKCombination.c0Cell))) ∧
    (((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).evaluation.eval_K coefficient).c1.eval env =
      value env (.parentEvalK (CombinationStep.indexOf EvalKCombination.block
        (Fin.cast EvalKCombination.coefficientCount_eq coefficient) RingKCombination.c1Cell))) := by
  exact ⟨rfl, rfl⟩

theorem evalA_values (env : Env) (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).evaluation.eval_A matrix coefficient).c0.eval env =
      value env (.parentEvalA (CombinationStep.indexOf matrix
        (Fin.cast EvalKCombination.coefficientCount_eq coefficient) RingKCombination.c0Cell))) ∧
    (((Layout.Stage1.Wide.PiDECInputs.parent logicalWidth publicFits).evaluation.eval_A matrix coefficient).c1.eval env =
      value env (.parentEvalA (CombinationStep.indexOf matrix
        (Fin.cast EvalKCombination.coefficientCount_eq coefficient) RingKCombination.c1Cell))) := by
  exact ⟨rfl, rfl⟩

end NightstreamFPrime.Export.Stage1.Wide.PiDECSource
