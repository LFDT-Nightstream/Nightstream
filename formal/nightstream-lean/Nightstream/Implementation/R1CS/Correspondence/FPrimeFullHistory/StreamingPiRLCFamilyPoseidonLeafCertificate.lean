import Batteries.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafModel

/-!
Contract: bounded structural certificate for one generated production PiRLC
Poseidon2 leaf.

Assurance tier: artifact-checked leaf certificate.

Owns: exact decoding of all 86 steps and rows in two 43-entry leaves, exact
decoded lengths, and index-preserving pairing of every row with its typed
S-box input and output step.

Does not own: source-slot reconstruction, row satisfaction, Poseidon2 replay,
absolute Rust column placement, lifecycle semantics, or permission to remove
constraints.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

def rawStepHead : List RawStep := rawSteps.take 43
def rawStepTail : List RawStep := rawSteps.drop 43
def rawRowHead : List RawRow := rawRows.take 43
def rawRowTail : List RawRow := rawRows.drop 43

def decodedStepHead : List Wire.Step :=
  (rawStepHead.mapM Wire.decodeStep).getD []

def decodedStepTail : List Wire.Step :=
  (rawStepTail.mapM Wire.decodeStep).getD []

def decodedRowHead : List Wire.Row :=
  (rawRowHead.mapM Wire.decodeRow).getD []

def decodedRowTail : List Wire.Row :=
  (rawRowTail.mapM Wire.decodeRow).getD []

def decodedSteps : List Wire.Step := decodedStepHead ++ decodedStepTail
def decodedRows : List Wire.Row := decodedRowHead ++ decodedRowTail

theorem raw_steps_partition : rawSteps = rawStepHead ++ rawStepTail := by
  simp [rawStepHead, rawStepTail]

theorem raw_rows_partition : rawRows = rawRowHead ++ rawRowTail := by
  simp [rawRowHead, rawRowTail]

/-- Each reduction is bounded by one generated 43-entry step leaf. -/
theorem raw_step_head_decodes :
    rawStepHead.mapM Wire.decodeStep = some decodedStepHead := by
  rfl

theorem raw_step_tail_decodes :
    rawStepTail.mapM Wire.decodeStep = some decodedStepTail := by
  rfl

/-- Each reduction is bounded by one generated 43-entry row shard. -/
theorem raw_row_head_decodes :
    rawRowHead.mapM Wire.decodeRow = some decodedRowHead := by
  rfl

theorem raw_row_tail_decodes :
    rawRowTail.mapM Wire.decodeRow = some decodedRowTail := by
  rfl

theorem raw_steps_decode :
    rawSteps.mapM Wire.decodeStep = some decodedSteps := by
  rw [raw_steps_partition, List.mapM_append, raw_step_head_decodes,
    raw_step_tail_decodes]
  rfl

theorem raw_rows_decode :
    rawRows.mapM Wire.decodeRow = some decodedRows := by
  rw [raw_rows_partition, List.mapM_append, raw_row_head_decodes,
    raw_row_tail_decodes]
  rfl

theorem decoded_steps_length : decodedSteps.length = 86 := by
  rfl

theorem decoded_rows_length : decodedRows.length = 86 := by
  rfl

private theorem head_offsets_exact :
    decodedStepHead.map (fun step => step.rowOffset) =
      decodedRowHead.map (fun row => row.rowOffset) := by
  rfl

private theorem tail_offsets_exact :
    decodedStepTail.map (fun step => step.rowOffset) =
      decodedRowTail.map (fun row => row.rowOffset) := by
  rfl

theorem decoded_offsets_exact :
    decodedSteps.map (fun step => step.rowOffset) =
      decodedRows.map (fun row => row.rowOffset) := by
  simp only [decodedSteps, decodedRows, List.map_append]
  rw [head_offsets_exact, tail_offsets_exact]

theorem paired_of_offsets :
    ∀ (steps : List Wire.Step) (rows : List Wire.Row),
      steps.map (fun step => step.rowOffset) =
          rows.map (fun row => row.rowOffset) →
        List.Forall₂ (fun step row => step.rowOffset = row.rowOffset)
          steps rows
  | [], [], _ => .nil
  | [], _ :: _, offsets => by simp at offsets
  | _ :: _, [], offsets => by simp at offsets
  | step :: steps, row :: rows, offsets => by
      simp only [List.map_cons, List.cons.injEq] at offsets
      exact .cons offsets.1 (paired_of_offsets steps rows offsets.2)

/-- Every generated row is paired, at the same list index and row offset,
with the decoded step that owns its typed S-box input and output expressions.
-/
theorem decoded_rows_pair_with_steps :
    List.Forall₂ (fun step row => step.rowOffset = row.rowOffset)
      decodedSteps decodedRows :=
  paired_of_offsets decodedSteps decodedRows decoded_offsets_exact

theorem decoded_row_head_shapes_checked :
    decodedRowHead.all sboxShapeCheck = true := by
  rfl

theorem decoded_row_tail_shapes_checked :
    decodedRowTail.all sboxShapeCheck = true := by
  rfl

theorem shapes_of_all_checked :
    ∀ (rows : List Wire.Row), rows.all sboxShapeCheck = true →
      ∀ row ∈ rows, IsSboxShape row
  | [], _ => by simp
  | head :: tail, checked => by
      simp only [List.all_cons, Bool.and_eq_true] at checked
      intro row member
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact sboxShapeCheck_sound _ checked.1
      · exact shapes_of_all_checked tail checked.2 row tailMember

theorem decoded_rows_have_sbox_shape :
    ∀ row ∈ decodedRows, IsSboxShape row := by
  intro row member
  rw [decodedRows, List.mem_append] at member
  rcases member with headMember | tailMember
  · exact shapes_of_all_checked decodedRowHead
      decoded_row_head_shapes_checked row headMember
  · exact shapes_of_all_checked decodedRowTail
      decoded_row_tail_shapes_checked row tailMember

structure StepRealized
    (source : SourceAssignment) (final : FinalAssignment)
    (step : Wire.Step) (row : Wire.Row) : Prop where
  input :
    portAction (row.port Role.sboxInput.index) final =
      sourceAction step.input source
  output :
    portAction (row.port Role.c.index) final =
      sourceAction step.output source

structure StepPortsRealized
    (source : SourceAssignment) (final : FinalAssignment)
    (step : Wire.Step) (row : Wire.Row) : Prop where
  input : PortRealized step.input
    (row.port Role.sboxInput.index) source final
  output : PortRealized step.output
    (row.port Role.c.index) source final

private theorem steps_realized_of_ports
    (source : SourceAssignment) (final : FinalAssignment) :
    ∀ {steps : List Wire.Step} {rows : List Wire.Row},
      List.Forall₂ (StepPortsRealized source final) steps rows →
      List.Forall₂ (StepRealized source final) steps rows := by
  intro steps rows links
  induction links with
  | nil => exact .nil
  | cons head tail inductionHypothesis =>
      exact .cons
        { input := portRealized_action head.input
          output := portRealized_action head.output }
        inductionHypothesis

def StepSboxHolds (source : SourceAssignment) (step : Wire.Step) : Prop :=
  sourceAction step.input source * sourceAction step.input source *
      sourceAction step.input source * sourceAction step.input source *
      sourceAction step.input source * sourceAction step.input source *
      sourceAction step.input source =
    sourceAction step.output source

theorem realized_rows_imply_step_sboxes
    (source : SourceAssignment) (final : FinalAssignment)
    (selectorOne : final.explicit .selector = 1) :
    ∀ {steps : List Wire.Step} {rows : List Wire.Row},
      List.Forall₂ (StepRealized source final) steps rows →
      (∀ row ∈ rows, IsSboxShape row) →
      (∀ row ∈ rows, residual row final = 0) →
      ∀ step ∈ steps, StepSboxHolds source step := by
  intro steps rows realized
  induction realized with
  | nil =>
      intro _ _ step member
      simp at member
  | @cons step row steps rows realization tail inductionHypothesis =>
      intro shapes holds candidate member
      rcases List.mem_cons.mp member with rfl | tailMember
      · have equation :=
          (residual_zero_iff_sbox_of_shape row final
            (shapes row List.mem_cons_self) selectorOne).mp
            (holds row List.mem_cons_self)
        unfold StepSboxHolds
        rw [← realization.input, ← realization.output]
        exact equation
      · apply inductionHypothesis
        · intro candidate member
          exact shapes candidate (List.mem_cons_of_mem row member)
        · intro candidate member
          exact holds candidate (List.mem_cons_of_mem row member)
        · exact tailMember

/-- On one same assignment, zero residuals for all generated leaf rows imply
all 86 typed source S-box equations. `StepRealized` is the explicit remaining
source-slot reconstruction boundary. -/
theorem decoded_rows_imply_step_sboxes
    (source : SourceAssignment) (final : FinalAssignment)
    (selectorOne : final.explicit .selector = 1)
    (realized : List.Forall₂ (StepRealized source final)
      decodedSteps decodedRows)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps, StepSboxHolds source step :=
  realized_rows_imply_step_sboxes source final selectorOne realized
    decoded_rows_have_sbox_shape holds

/-- Exact per-run source-to-final links plus all active generated rows imply
all 86 typed source S-box equations on the same assignment. -/
theorem decoded_rows_and_port_links_imply_step_sboxes
    (source : SourceAssignment) (final : FinalAssignment)
    (selectorOne : final.explicit .selector = 1)
    (links : List.Forall₂ (StepPortsRealized source final)
      decodedSteps decodedRows)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps, StepSboxHolds source step :=
  decoded_rows_imply_step_sboxes source final selectorOne
    (steps_realized_of_ports source final links) holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate
