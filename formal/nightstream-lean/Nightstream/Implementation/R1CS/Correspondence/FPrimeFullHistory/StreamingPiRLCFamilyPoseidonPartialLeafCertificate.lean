import Batteries.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonPartialLeaf
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafReconstruction

/-!
Contract: exact operand-permutation certificate for the partial-start
production PiRLC Poseidon2 leaf.

Assurance tier: artifact-checked and Rust-conformant leaf certificate.

Owns: exact decoding in two 43-entry leaves, the eight-row external-A operand
rotation, invariance of source and final actions under that rotation, and
transport of exact partial-row satisfaction to the direct leaf relation.

Does not own: replay-batch coverage, absolute assignment placement, selector
authority, lifecycle semantics, or permission to remove constraints.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonPartialLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction

def partialRawStepHead : List RawStep := rawSteps.take 43
def partialRawStepTail : List RawStep := rawSteps.drop 43
def partialRawRowHead : List RawRow := rawRows.take 43
def partialRawRowTail : List RawRow := rawRows.drop 43

def partialDecodedStepHead : List Wire.Step :=
  (partialRawStepHead.mapM Wire.decodeStep).getD []

def partialDecodedStepTail : List Wire.Step :=
  (partialRawStepTail.mapM Wire.decodeStep).getD []

def partialDecodedRowHead : List Wire.Row :=
  (partialRawRowHead.mapM Wire.decodeRow).getD []

def partialDecodedRowTail : List Wire.Row :=
  (partialRawRowTail.mapM Wire.decodeRow).getD []

def partialDecodedSteps : List Wire.Step :=
  partialDecodedStepHead ++ partialDecodedStepTail

def partialDecodedRows : List Wire.Row :=
  partialDecodedRowHead ++ partialDecodedRowTail

/-- Each exact reduction is bounded by one generated 43-entry leaf. -/
theorem partial_raw_step_head_decodes :
    partialRawStepHead.mapM Wire.decodeStep = some partialDecodedStepHead := by
  rfl

theorem partial_raw_step_tail_decodes :
    partialRawStepTail.mapM Wire.decodeStep = some partialDecodedStepTail := by
  rfl

theorem partial_raw_row_head_decodes :
    partialRawRowHead.mapM Wire.decodeRow = some partialDecodedRowHead := by
  rfl

theorem partial_raw_row_tail_decodes :
    partialRawRowTail.mapM Wire.decodeRow = some partialDecodedRowTail := by
  rfl

/-- Move the first two two-element blocks past each other. -/
def rotateFour {α : Type} : List α → List α
  | a :: b :: c :: d :: tail => c :: d :: a :: b :: tail
  | values => values

theorem rotateFour_perm {α : Type} :
    ∀ values : List α, (rotateFour values).Perm values
  | [] => List.Perm.refl []
  | [a] => List.Perm.refl [a]
  | [a, b] => List.Perm.refl [a, b]
  | [a, b, c] => List.Perm.refl [a, b, c]
  | a :: b :: c :: d :: tail => by
      simpa only [rotateFour, List.append_assoc] using
        ((List.perm_append_comm :
          List.Perm ([c, d] ++ [a, b]) ([a, b] ++ [c, d])).append_right tail)

def normalizeSourceLinearCombination
    (value : SourceLinearCombination) : SourceLinearCombination :=
  { value with terms := rotateFour value.terms }

def normalizeStep (step : Wire.Step) : Wire.Step :=
  { step with input := normalizeSourceLinearCombination step.input }

def normalizePort (port : Port) : Port :=
  { port with geometric := rotateFour port.geometric }

def normalizeRow (row : Wire.Row) : Wire.Row where
  rowOffset := row.rowOffset
  ports := row.ports.map normalizePort
  portsLength := by simpa using row.portsLength

private theorem sum_eq_of_perm {left right : List F}
    (permutation : left.Perm right) : sum left = sum right := by
  induction permutation with
  | nil => rfl
  | cons _ _ hypothesis =>
      simp only [sum]
      rw [hypothesis]
  | swap left right tail =>
      simp only [sum]
      calc
        right + (left + sum tail) = (right + left) + sum tail :=
          (Lean.Grind.Fin.add_assoc _ _ _).symm
        _ = (left + right) + sum tail :=
          congrArg (fun value => value + sum tail)
            (Lean.Grind.Fin.add_comm right left)
        _ = left + (right + sum tail) :=
          Lean.Grind.Fin.add_assoc _ _ _
  | trans _ _ leftHypothesis rightHypothesis =>
      exact leftHypothesis.trans rightHypothesis

theorem sourceAction_normalizeSourceLinearCombination
    (value : SourceLinearCombination) (source : SourceAssignment) :
    sourceAction (normalizeSourceLinearCombination value) source =
      sourceAction value source := by
  unfold sourceAction normalizeSourceLinearCombination
  rw [sum_eq_of_perm
    ((rotateFour_perm value.terms).map fun term =>
      term.coefficient * sourceValue source term.column)]

theorem portAction_normalizePort
    (port : Port) (final : FinalAssignment) :
    portAction (normalizePort port) final = portAction port final := by
  unfold portAction normalizePort
  rw [sum_eq_of_perm
    ((rotateFour_perm port.geometric).map fun run =>
      geometricAction run final)]

theorem normalizeRow_port (row : Wire.Row) (index : Fin 13) :
    (normalizeRow row).port index = normalizePort (row.port index) := by
  simp [Wire.Row.port, normalizeRow]

theorem point_normalizeRow
    (row : Wire.Row) (final : FinalAssignment) :
    point (normalizeRow row) final = point row final := by
  funext index
  simp only [point, normalizeRow_port, portAction_normalizePort]

theorem residual_normalizeRow
    (row : Wire.Row) (final : FinalAssignment) :
    residual (normalizeRow row) final = residual row final := by
  unfold residual
  rw [point_normalizeRow]

theorem stepSboxHolds_normalizeStep
    (source : SourceAssignment) (step : Wire.Step) :
    StepSboxHolds source (normalizeStep step) ↔
      StepSboxHolds source step := by
  unfold StepSboxHolds normalizeStep
  rw [sourceAction_normalizeSourceLinearCombination]

def canonicalStepHead : List Wire.Step :=
  (partialDecodedStepHead.take 8).map normalizeStep ++
    partialDecodedStepHead.drop 8

def canonicalRowHead : List Wire.Row :=
  (partialDecodedRowHead.take 8).map normalizeRow ++
    partialDecodedRowHead.drop 8

/-- These exact checks are each bounded by one 43-entry leaf. The semantic
normalizer is proved permutation-preserving above. -/
theorem canonical_step_head_eq_direct :
    canonicalStepHead = decodedStepHead := by
  rfl

theorem partial_step_tail_eq_direct :
    partialDecodedStepTail = decodedStepTail := by
  rfl

theorem canonical_row_head_eq_direct :
    canonicalRowHead = decodedRowHead := by
  rfl

theorem partial_row_tail_eq_direct :
    partialDecodedRowTail = decodedRowTail := by
  rfl

theorem canonical_steps_eq_direct :
    canonicalStepHead ++ partialDecodedStepTail = decodedSteps := by
  rw [canonical_step_head_eq_direct, partial_step_tail_eq_direct]
  rfl

theorem canonical_rows_eq_direct :
    canonicalRowHead ++ partialDecodedRowTail = decodedRows := by
  rw [canonical_row_head_eq_direct, partial_row_tail_eq_direct]
  rfl

private theorem canonical_row_head_holds
    (final : FinalAssignment)
    (holds : ∀ row ∈ partialDecodedRows, residual row final = 0) :
    ∀ row ∈ canonicalRowHead, residual row final = 0 := by
  intro row member
  rw [canonicalRowHead, List.mem_append] at member
  rcases member with prefixMember | suffixMember
  · rcases List.mem_map.mp prefixMember with
      ⟨original, originalMember, rfl⟩
    rw [residual_normalizeRow]
    apply holds original
    apply List.mem_append_left partialDecodedRowTail
    rw [← List.take_append_drop 8 partialDecodedRowHead]
    exact List.mem_append_left _ originalMember
  · apply holds row
    apply List.mem_append_left partialDecodedRowTail
    rw [← List.take_append_drop 8 partialDecodedRowHead]
    exact List.mem_append_right _ suffixMember

private theorem canonical_rows_hold
    (final : FinalAssignment)
    (holds : ∀ row ∈ partialDecodedRows, residual row final = 0) :
    ∀ row ∈ canonicalRowHead ++ partialDecodedRowTail,
      residual row final = 0 := by
  intro row member
  rw [List.mem_append] at member
  rcases member with headMember | tailMember
  · exact canonical_row_head_holds final holds row headMember
  · exact holds row (List.mem_append_right partialDecodedRowHead tailMember)

/-- Exact partial-start rows imply the same 86-step direct Poseidon2 S-box
relation on one final assignment. -/
theorem partial_rows_imply_direct_reconstructed_step_sboxes
    (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (selectorOne : final.explicit .selector = 1)
    (holds : ∀ row ∈ partialDecodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps,
      StepSboxHolds (reconstructedSource final) step := by
  apply decoded_rows_imply_reconstructed_step_sboxes final one selectorOne
  intro row member
  apply canonical_rows_hold final holds row
  rw [canonical_rows_eq_direct]
  exact member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate
