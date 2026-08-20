import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCompactTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafReconstruction

/-!
Contract: structural same-assignment bridge for any recursive-terminal XOut
Poseidon2 call.

Assurance tier: artifact-checked row interpretation once the complete terminal
matrix discharges `FinalRowSliceExact`.

Owns: fail-closed interpretation of the eight Rust-emitted source images,
the 86 local S-box-output slots, transport from exact final row actions to the
canonical S-box equations, and the independent Poseidon2 reference result.

Does not own: exact call-to-round alignment, the complete terminal matrix,
terminal assignment satisfaction, four final output-copy rows, lifecycle
composition, or collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Program

private abbrev decodedRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedRows

private abbrev decodedSteps :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps

private abbrev LeafStepSboxHolds :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds

private def decodeField :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField

def fieldValue (value : Nat) : F :=
  (decodeField value).getD 0

@[simp] theorem fieldValue_of_lt (value : Nat)
    (canonical : value < goldilocksModulus) :
    fieldValue value = ⟨value, canonical⟩ := by
  simp [fieldValue, decodeField,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
    canonical]

abbrev AbsoluteAssignment (placement : PoseidonCallPlacement) :=
  Fin placement.finalColumns → F

def absoluteValue {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (column : Nat) : F :=
  if bounded : column < placement.finalColumns then
    assignment ⟨column, bounded⟩
  else
    0

def explicitAction {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (term : AbsoluteTerm) : F :=
  fieldValue term.coefficient * absoluteValue assignment term.column

def geometricRunAction {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement)
    (run : AbsoluteGeometricRun) : F :=
  sum (List.range run.length |>.map fun index =>
    geometricCoefficient (fieldValue run.initial) (fieldValue run.ratio) index *
      absoluteValue assignment (run.columnStart + index))

def absolutePortAction {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (port : AbsolutePort) : F :=
  sum (port.explicit.map (explicitAction assignment)) +
    sum (port.geometric.map (geometricRunAction assignment))

def emptySourceImage : SourceImage :=
  { sourceColumn := 0, port := { explicit := [], geometric := [] } }

def inputImage (placement : PoseidonCallPlacement) (lane : Fin width) :
    SourceImage :=
  placement.inputImages.getD lane.val emptySourceImage

def inputValue (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) : F :=
  absolutePortAction assignment (inputImage placement lane).port

def projectFinalAssignment (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) : FinalAssignment where
  explicit
    | .one => absoluteValue assignment 0
    | .selector => absoluteValue assignment placement.selectorColumn
  digit slot digit :=
    match slot with
    | .externalA lane =>
        if digit.val = 0 then
          inputValue placement assignment
            ⟨lane.val, by
              have laneBound := lane.isLt
              simpa only [width] using (show lane.val < 8 by omega)⟩
        else
          0
    | .externalB lane =>
        if digit.val = 0 then
          inputValue placement assignment
            ⟨4 + lane.val, by
              have laneBound := lane.isLt
              simpa only [width] using
                (show 4 + lane.val < 8 by omega)⟩
        else
          0
    | .previousLocal _ => 0
    | .local index =>
        absoluteValue assignment
          (placement.localSlotStart +
            index.val * placement.slotWidth + digit.val)

@[simp] theorem projected_one
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) :
    (projectFinalAssignment placement assignment).explicit .one =
      absoluteValue assignment 0 := by
  rfl

@[simp] theorem projected_selector
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) :
    (projectFinalAssignment placement assignment).explicit .selector =
      absoluteValue assignment placement.selectorColumn := by
  rfl

/-- One absolute 41-digit run is the corresponding current local slot. This
adapter is structural in one run and does not inspect a generated row list. -/
theorem localGeometricRunAction
    {placement : PoseidonCallPlacement} (valid : placement.Valid)
    (assignment : AbsoluteAssignment placement) (slot : Fin 86)
    (initial : Nat) :
    geometricRunAction assignment
        { columnStart := placement.localSlotStart + slot.val * 41
          length := 41
          initial := initial
          ratio := 3 } =
      fieldValue initial *
        slotValue (projectFinalAssignment placement assignment) (.local slot) := by
  rcases valid with ⟨_, _, _, _, _, slotWidth, _, _, _, _⟩
  have ratioExact : fieldValue 3 = (3 : F) := by
    rw [fieldValue_of_lt 3 (by decide)]
    apply Fin.ext
    exact (Nat.mod_eq_of_lt (by decide)).symm
  have actionExact :
      geometricRunAction assignment
          { columnStart := placement.localSlotStart + slot.val * 41
            length := 41
            initial := initial
            ratio := 3 } =
        geometricAction
          { slot := .local slot, initial := fieldValue initial, ratio := 3 }
          (projectFinalAssignment placement assignment) := by
    unfold geometricRunAction geometricAction
    rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range]
    simp only [List.map_map, Function.comp_apply]
    congr 2
    funext digit
    rw [ratioExact]
    simp [projectFinalAssignment, slotWidth]
  rw [actionExact]
  exact geometricAction_eq_scaled_slotValue _ _ rfl

private def terminalOutputTerms
    (lane : Fin width) : List (Fin 8 × Fin 7) :=
  match lane.val with
  | 0 => [(7, 1), (6, 1), (5, 3), (4, 2), (3, 2), (2, 2), (1, 6), (0, 4)]
  | 1 => [(7, 1), (6, 3), (5, 2), (4, 1), (3, 2), (2, 6), (1, 4), (0, 2)]
  | 2 => [(7, 3), (6, 2), (5, 1), (4, 1), (3, 6), (2, 4), (1, 2), (0, 2)]
  | 3 => [(7, 2), (6, 1), (5, 1), (4, 3), (3, 4), (2, 2), (1, 2), (0, 6)]
  | 4 => [(7, 2), (6, 2), (5, 6), (4, 4), (3, 1), (2, 1), (1, 3), (0, 2)]
  | 5 => [(7, 2), (6, 6), (5, 4), (4, 2), (3, 1), (2, 3), (1, 2), (0, 1)]
  | 6 => [(7, 6), (6, 4), (5, 2), (4, 2), (3, 3), (2, 2), (1, 1), (0, 1)]
  | _ => [(7, 4), (6, 2), (5, 2), (4, 6), (3, 2), (2, 1), (1, 1), (0, 3)]

private def terminalOutputSlot (slot : Fin 8) : Fin 86 :=
  ⟨78 + slot.val, by
    have bounded := slot.isLt
    omega⟩

private def terminalOutputOffset (slot : Fin 8) : Fin 600 :=
  ⟨555 + 4 * slot.val, by
    have bounded := slot.isLt
    omega⟩

private def terminalOutputRun
    (placement : PoseidonCallPlacement) (term : Fin 8 × Fin 7) :
    AbsoluteGeometricRun where
  columnStart :=
    placement.localSlotStart + (terminalOutputSlot term.1).val * 41
  length := 41
  initial := term.2.val
  ratio := 3

private def terminalOutputCoefficient (coefficient : Fin 7) : F :=
  ⟨coefficient.val, lt_trans coefficient.isLt (by decide)⟩

private theorem fieldValue_terminalOutputCoefficient (coefficient : Fin 7) :
    fieldValue coefficient.val = terminalOutputCoefficient coefficient := by
  rw [fieldValue_of_lt coefficient.val
    (lt_trans coefficient.isLt (by decide))]
  apply Fin.ext
  rfl

private def terminalOutputSourceTerm
    (term : Fin 8 × Fin 7) : Wire.SourceTerm where
  column := .local (terminalOutputOffset term.1)
  coefficient := terminalOutputCoefficient term.2

private def terminalOutputSource
    (lane : Fin width) : Wire.SourceLinearCombination where
  constant := 0
  terms := (terminalOutputTerms lane).map terminalOutputSourceTerm

private theorem terminal_output_sourceValue
    (final : FinalAssignment) (slot : Fin 8) :
    sourceValue (reconstructedSource final)
        (.local (terminalOutputOffset slot)) =
      slotValue final (.local (terminalOutputSlot slot)) := by
  change
    (match sourceSlot decodedSteps (.local (terminalOutputOffset slot)) with
      | some owner => slotValue final owner
      | none => 0) =
        slotValue final (.local (terminalOutputSlot slot))
  have owned :
      sourceSlot decodedSteps (.local (terminalOutputOffset slot)) =
        some (.local (terminalOutputSlot slot)) := by
    fin_cases slot <;> rfl
  rw [owned]

/-- Absolute compact-output image for one selected call. The generated source
images use this exact reverse operand order. -/
def callOutputPort
    (placement : PoseidonCallPlacement) (lane : Fin width) : AbsolutePort where
  explicit := []
  geometric := (terminalOutputTerms lane).map (terminalOutputRun placement)

def appendAbsolutePort (left right : AbsolutePort) : AbsolutePort where
  explicit := left.explicit ++ right.explicit
  geometric := left.geometric ++ right.geometric

private theorem sum_append (left right : List F) :
    sum (left ++ right) = sum left + sum right := by
  induction left with
  | nil => simp [sum]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, sum]
      rw [inductionHypothesis, Lean.Grind.Fin.add_assoc]

theorem absolutePortAction_append
    {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement) (left right : AbsolutePort) :
    absolutePortAction assignment (appendAbsolutePort left right) =
      absolutePortAction assignment left +
        absolutePortAction assignment right := by
  unfold absolutePortAction appendAbsolutePort
  simp only [List.map_append, sum_append]
  ac_rfl

private theorem terminal_output_source_exact
    (final : FinalAssignment) (lane : Fin width) :
    lcEval (sourcePhysical (reconstructedSource final)) (traceFinalForm lane) =
      (sourceAction (terminalOutputSource lane)
        (reconstructedSource final)).val := by
  have traceTermsExact :
      (traceFinalForm lane).reverse =
        (terminalOutputTerms lane).map fun term =>
          (564 + 4 * term.1.val, term.2.val) := by
    fin_cases lane <;> rfl
  have sourceTermsExact :
      sourceTerms (terminalOutputSource lane) =
        (terminalOutputTerms lane).map (fun term =>
          (564 + 4 * term.1.val, term.2.val)) ++ [(0, 0)] := by
    unfold sourceTerms terminalOutputSource
    simp only [List.map_map]
    congr 1
    apply List.map_congr_left
    intro term _
    apply Prod.ext
    · simp only [Function.comp_apply, terminalOutputSourceTerm,
        sourceColumnIndex, terminalOutputOffset]
      omega
    · rfl
  have appendZero (terms : List (Nat × Nat)) :
      lcEval (sourcePhysical (reconstructedSource final))
          (terms ++ [(0, 0)]) =
        lcEval (sourcePhysical (reconstructedSource final)) terms := by
    simp [lcEval, List.foldl_append]
  calc
    lcEval (sourcePhysical (reconstructedSource final)) (traceFinalForm lane) =
        lcEval (sourcePhysical (reconstructedSource final))
          (traceFinalForm lane).reverse :=
      lcEval_eq_of_perm _ (List.reverse_perm (traceFinalForm lane)).symm
    _ = lcEval (sourcePhysical (reconstructedSource final))
          ((terminalOutputTerms lane).map fun term =>
            (564 + 4 * term.1.val, term.2.val)) := by
      rw [traceTermsExact]
    _ = lcEval (sourcePhysical (reconstructedSource final))
          (sourceTerms (terminalOutputSource lane)) := by
      rw [sourceTermsExact, appendZero]
    _ = (sourceAction (terminalOutputSource lane)
          (reconstructedSource final)).val :=
      lcEval_sourceTerms _ _

private theorem terminal_output_run_action
    {placement : PoseidonCallPlacement} (valid : placement.Valid)
    (assignment : AbsoluteAssignment placement) (term : Fin 8 × Fin 7) :
    geometricRunAction assignment (terminalOutputRun placement term) =
      (terminalOutputSourceTerm term).coefficient *
        sourceValue
          (reconstructedSource
            (projectFinalAssignment placement assignment))
          (terminalOutputSourceTerm term).column := by
  unfold terminalOutputRun terminalOutputSourceTerm
  rw [localGeometricRunAction valid assignment
    (terminalOutputSlot term.1) term.2.val]
  rw [terminal_output_sourceValue,
    fieldValue_terminalOutputCoefficient]

private theorem terminal_output_runs_action
    {placement : PoseidonCallPlacement} (valid : placement.Valid)
    (assignment : AbsoluteAssignment placement) :
    ∀ terms : List (Fin 8 × Fin 7),
      sum ((terms.map (terminalOutputRun placement)).map
          (geometricRunAction assignment)) =
        sum ((terms.map terminalOutputSourceTerm).map fun term =>
          term.coefficient *
            sourceValue
              (reconstructedSource
                (projectFinalAssignment placement assignment)) term.column)
  | [] => rfl
  | term :: tail => by
      simp only [List.map_cons, sum]
      rw [terminal_output_run_action valid assignment term,
        terminal_output_runs_action valid assignment tail]

private theorem callOutputPort_action_field
    {placement : PoseidonCallPlacement} (valid : placement.Valid)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) :
    absolutePortAction assignment (callOutputPort placement lane) =
      sourceAction (terminalOutputSource lane)
        (reconstructedSource
          (projectFinalAssignment placement assignment)) := by
  unfold absolutePortAction callOutputPort sourceAction terminalOutputSource
  simp only [List.map_nil, sum, Fin.zero_add]
  exact terminal_output_runs_action valid assignment
    (terminalOutputTerms lane)

/-- The absolute output port and the independent compact-trace output use the
same assignment coordinates and the same field value. -/
theorem callOutputPort_action
    {placement : PoseidonCallPlacement} (valid : placement.Valid)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) :
    (absolutePortAction assignment (callOutputPort placement lane)).val =
      lcEval
        (sourcePhysical
          (reconstructedSource
            (projectFinalAssignment placement assignment)))
        (traceFinalForm lane) := by
  calc
    (absolutePortAction assignment (callOutputPort placement lane)).val =
        (sourceAction (terminalOutputSource lane)
          (reconstructedSource
            (projectFinalAssignment placement assignment))).val :=
      congrArg Fin.val (callOutputPort_action_field valid assignment lane)
    _ = lcEval
          (sourcePhysical
            (reconstructedSource
              (projectFinalAssignment placement assignment)))
          (traceFinalForm lane) :=
      (terminal_output_source_exact
        (projectFinalAssignment placement assignment) lane).symm

private theorem slotValue_projected_externalA
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin 4) :
    slotValue (projectFinalAssignment placement assignment) (.externalA lane) =
      inputValue placement assignment
        ⟨lane.val, by
          have laneBound := lane.isLt
          simpa only [width] using (show lane.val < 8 by omega)⟩ := by
  unfold slotValue geometricAction
  simp [projectFinalAssignment, sum, geometricCoefficient]

private theorem slotValue_projected_externalB
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin 4) :
    slotValue (projectFinalAssignment placement assignment) (.externalB lane) =
      inputValue placement assignment
        ⟨4 + lane.val, by
          have laneBound := lane.isLt
          simpa only [width] using
            (show 4 + lane.val < 8 by omega)⟩ := by
  unfold slotValue geometricAction
  simp [projectFinalAssignment, sum, geometricCoefficient]

theorem sourceInput_projected
    (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (lane : Fin width) :
    sourceInput
        (reconstructedSource (projectFinalAssignment placement assignment)) lane =
      inputValue placement assignment lane := by
  fin_cases lane <;>
    simp [sourceInput, reconstructedSource,
      slotValue_projected_externalA, slotValue_projected_externalB]

def absolutePoint (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (row : Wire.Row) : Fin 13 → F :=
  point row (projectFinalAssignment placement assignment)

def absoluteResidual (placement : PoseidonCallPlacement)
    (assignment : AbsoluteAssignment placement) (row : Wire.Row) : F :=
  residual row (projectFinalAssignment placement assignment)

theorem final_row_range_exact
    (placement : PoseidonCallPlacement) (valid : placement.Valid) :
    placement.finalRows.start + decodedRows.length =
      placement.finalRows.stop := by
  unfold PoseidonCallPlacement.Valid at valid
  rcases valid with ⟨_, rangeValid, _, lengthExact, _, _, _, _, _, _⟩
  unfold Range.Valid at rangeValid
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decoded_rows_length]
  omega

def finalRowIndex
    (placement : PoseidonCallPlacement) (valid : placement.Valid)
    {rows : Nat} (rowsFit : placement.finalRows.stop ≤ rows)
    (offset : Fin decodedRows.length) : Fin rows :=
  ⟨placement.finalRows.start + offset.val, by
    have offsetLt := offset.isLt
    have rangeExact := final_row_range_exact placement valid
    omega⟩

/-- Exact action of the complete final matrix on one compact call slice.
The complete terminal artifact must prove this obligation structurally. -/
structure FinalRowSliceExact
    (placement : PoseidonCallPlacement) (valid : placement.Valid)
    {rows : Nat}
    (relation : InterpretedRelation rows placement.finalColumns)
    (assignment : AbsoluteAssignment placement) : Prop where
  rowsFit : placement.finalRows.stop ≤ rows
  pointExact : ∀ offset : Fin decodedRows.length,
    rowPoint relation assignment (finalRowIndex placement valid rowsFit offset) =
      absolutePoint placement assignment (decodedRows.get offset)

def AllRowsSatisfied
    {placement : PoseidonCallPlacement} {rows : Nat}
    (relation : InterpretedRelation rows placement.finalColumns)
    (assignment : AbsoluteAssignment placement) : Prop :=
  ∀ row, residualAt relation assignment row = 0

theorem final_rows_imply_decoded_rows
    {placement : PoseidonCallPlacement} {valid : placement.Valid}
    {rows : Nat}
    {relation : InterpretedRelation rows placement.finalColumns}
    {assignment : AbsoluteAssignment placement}
    (exact : FinalRowSliceExact placement valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment) :
    ∀ row ∈ decodedRows,
      residual row (projectFinalAssignment placement assignment) = 0 := by
  intro row member
  rcases List.mem_iff_get.mp member with ⟨offset, rowExact⟩
  subst row
  have finalZero :=
    satisfied (finalRowIndex placement valid exact.rowsFit offset)
  rw [residualAt_eq_evaluate, exact.pointExact offset] at finalZero
  exact finalZero

theorem final_rows_imply_step_sboxes
    {placement : PoseidonCallPlacement} {valid : placement.Valid}
    {rows : Nat}
    {relation : InterpretedRelation rows placement.finalColumns}
    {assignment : AbsoluteAssignment placement}
    (exact : FinalRowSliceExact placement valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment placement.selectorColumn = 1) :
    ∀ step ∈ decodedSteps,
      LeafStepSboxHolds
        (reconstructedSource
          (projectFinalAssignment placement assignment)) step := by
  exact decoded_rows_imply_reconstructed_step_sboxes
    (projectFinalAssignment placement assignment) one selectorOne
    (final_rows_imply_decoded_rows exact satisfied)

/-- One exact final call slice forces the independent production Poseidon2
permutation on the eight Rust-emitted source-image actions. -/
theorem final_rows_compute_reference
    {placement : PoseidonCallPlacement} {valid : placement.Valid}
    {rows : Nat}
    {relation : InterpretedRelation rows placement.finalColumns}
    {assignment : AbsoluteAssignment placement}
    (exact : FinalRowSliceExact placement valid relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment placement.selectorColumn = 1)
    (lane : Fin width) :
    lcEval
        (sourcePhysical
          (reconstructedSource
            (projectFinalAssignment placement assignment)))
        (traceFinalForm lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane => (inputValue placement assignment inputLane).val) lane := by
  rw [step_sboxes_compute_reference _
    (final_rows_imply_step_sboxes exact satisfied one selectorOne) lane]
  congr 2
  funext inputLane
  exact congrArg Fin.val
    (sourceInput_projected placement assignment inputLane)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
