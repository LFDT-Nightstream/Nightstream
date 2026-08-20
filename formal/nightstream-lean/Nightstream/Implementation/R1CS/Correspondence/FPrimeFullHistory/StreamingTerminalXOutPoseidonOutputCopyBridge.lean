import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallPermutation

/-!
Contract: same-assignment semantic bridge for the four recursive-terminal XOut
Poseidon2 output-copy rows.

Owns: structural interpretation of the exact Rust-emitted output-copy ports,
reduction of each selected product row to equality with one final Poseidon2
output lane, and composition with the pure nine-call terminal hash theorem.

Does not own: the complete terminal matrix, relation satisfaction, public-word
serialization, lifecycle composition, or collision resistance.

Assurance tier: artifact-checked once the explicit final-row premises hold.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallPermutation
open Nightstream.Implementation.R1CS.Poseidon2Sponge

def emptyAbsolutePort : AbsolutePort where
  explicit := []
  geometric := []

private def emptyOutputCopyPlacement : OutputCopyPlacement where
  lane := 0
  rewriteId := 0
  sourceRows := { start := 0, stop := 0 }
  finalRow := 0
  finalRows := 0
  finalColumns := 0
  selectorColumn := 0
  outputSourceColumn := 0
  linearFormConstant := 0
  linearFormTerms := []
  finalPorts := []

/-- The exact Rust-emitted output-copy placement for one public hash lane. -/
def outputCopyAt (lane : Fin 4) : OutputCopyPlacement :=
  outputCopies.getD lane.val emptyOutputCopyPlacement

/-- The exact Rust-emitted binary output image for one public hash lane. -/
def outputImageAt (lane : Fin 4) : SourceImage :=
  outputImages.getD lane.val emptySourceImage

def outputLane (lane : Fin 4) : Fin 8 :=
  ⟨lane.val, by
    have bounded := lane.isLt
    omega⟩

def outputImageValue
    (assignment : AbsoluteAssignment callPlacement8) (lane : Fin 4) : Nat :=
  (absolutePortAction assignment (outputImageAt lane).port).val

def negateAbsoluteTerm (term : AbsoluteTerm) : AbsoluteTerm where
  column := term.column
  coefficient := (-fieldValue term.coefficient).val

def negateAbsoluteGeometricRun
    (run : AbsoluteGeometricRun) : AbsoluteGeometricRun where
  columnStart := run.columnStart
  length := run.length
  initial := (-fieldValue run.initial).val
  ratio := run.ratio

/-- Coefficient negation preserves the physical columns and run geometry. -/
def negateAbsolutePort (port : AbsolutePort) : AbsolutePort where
  explicit := port.explicit.map negateAbsoluteTerm
  geometric := port.geometric.map negateAbsoluteGeometricRun

@[simp] theorem fieldValue_neg_val (value : Nat) :
    fieldValue (-fieldValue value).val = -fieldValue value := by
  rw [fieldValue_of_lt _ (Fin.isLt _)]

private theorem geometricCoefficient_neg
    (initial ratio : F) (index : Nat) :
    geometricCoefficient (-initial) ratio index =
      -geometricCoefficient initial ratio index := by
  induction index with
  | zero => rfl
  | succ index inductionHypothesis =>
      simp only [geometricCoefficient, inductionHypothesis, neg_mul]

private theorem sum_map_neg : ∀ values : List F,
    sum (values.map fun value => -value) = -sum values
  | [] => by simp [sum]
  | head :: tail => by
      simp only [List.map_cons, sum, sum_map_neg tail, neg_add]

private theorem explicitAction_negate
    (assignment : AbsoluteAssignment callPlacement8) (term : AbsoluteTerm) :
    explicitAction assignment (negateAbsoluteTerm term) =
      -explicitAction assignment term := by
  simp [explicitAction, negateAbsoluteTerm]

private theorem geometricRunAction_negate
    (assignment : AbsoluteAssignment callPlacement8)
    (run : AbsoluteGeometricRun) :
    geometricRunAction assignment (negateAbsoluteGeometricRun run) =
      -geometricRunAction assignment run := by
  let values := List.range run.length |>.map fun index =>
    geometricCoefficient (fieldValue run.initial) (fieldValue run.ratio) index *
      absoluteValue assignment (run.columnStart + index)
  unfold geometricRunAction
  simp only [negateAbsoluteGeometricRun, fieldValue_neg_val,
    geometricCoefficient_neg, neg_mul]
  simpa only [values, List.map_map, Function.comp_apply] using
    sum_map_neg values

/-- Port negation is semantic field negation for every assignment. -/
theorem absolutePortAction_negate
    (assignment : AbsoluteAssignment callPlacement8) (port : AbsolutePort) :
    absolutePortAction assignment (negateAbsolutePort port) =
      -absolutePortAction assignment port := by
  unfold absolutePortAction negateAbsolutePort
  have explicitExact :
      (port.explicit.map negateAbsoluteTerm).map
          (explicitAction assignment) =
        (port.explicit.map (explicitAction assignment)).map fun value =>
          -value := by
    simp only [List.map_map, Function.comp_apply]
    apply List.map_congr_left
    intro term _
    exact explicitAction_negate assignment term
  have geometricExact :
      (port.geometric.map negateAbsoluteGeometricRun).map
          (geometricRunAction assignment) =
        (port.geometric.map (geometricRunAction assignment)).map fun value =>
          -value := by
    simp only [List.map_map, Function.comp_apply]
    apply List.map_congr_left
    intro run _
    exact geometricRunAction_negate assignment run
  rw [explicitExact, geometricExact, sum_map_neg, sum_map_neg, neg_add]

def reverseAbsolutePort (port : AbsolutePort) : AbsolutePort where
  explicit := port.explicit.reverse
  geometric := port.geometric.reverse

/-- Reversing operand order does not change a port action. -/
theorem absolutePortAction_reverse
    (assignment : AbsoluteAssignment callPlacement8) (port : AbsolutePort) :
    absolutePortAction assignment (reverseAbsolutePort port) =
      absolutePortAction assignment port := by
  unfold absolutePortAction reverseAbsolutePort
  rw [sum_map_eq_of_perm (List.reverse_perm port.explicit)
      (explicitAction assignment),
    sum_map_eq_of_perm (List.reverse_perm port.geometric)
      (geometricRunAction assignment)]

def selectorPort : AbsolutePort where
  explicit := [{ column := callPlacement8.selectorColumn, coefficient := 1 }]
  geometric := []

@[simp] private theorem fieldValue_one : fieldValue 1 = 1 := by
  rw [fieldValue_of_lt 1 (by decide)]
  apply Fin.ext
  rfl

private theorem absolutePortAction_empty
    (assignment : AbsoluteAssignment callPlacement8) :
    absolutePortAction assignment emptyAbsolutePort = 0 := by
  simp [absolutePortAction, emptyAbsolutePort, sum]

private theorem absolutePortAction_selector
    (assignment : AbsoluteAssignment callPlacement8) :
    absolutePortAction assignment selectorPort =
      absoluteValue assignment callPlacement8.selectorColumn := by
  simp [absolutePortAction, selectorPort, explicitAction, sum]

private theorem absolutePortAction_outputCopyC
    (assignment : AbsoluteAssignment callPlacement8) (lane : Fin 4) :
    absolutePortAction assignment
        (appendAbsolutePort (outputImageAt lane).port
          (reverseAbsolutePort
            (negateAbsolutePort
              (callOutputPort callPlacement8 (outputLane lane))))) =
      absolutePortAction assignment (outputImageAt lane).port -
        absolutePortAction assignment
          (callOutputPort callPlacement8 (outputLane lane)) := by
  rw [absolutePortAction_append, absolutePortAction_reverse,
    absolutePortAction_negate]
  exact (sub_eq_add_neg _ _).symm

def expectedOutputCopyPorts (lane : Fin 4) : List AbsolutePort :=
  [emptyAbsolutePort,
   selectorPort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   appendAbsolutePort (outputImageAt lane).port
     (reverseAbsolutePort
       (negateAbsolutePort
         (callOutputPort callPlacement8 (outputLane lane)))),
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort,
   emptyAbsolutePort]

/-- Four small leaf certificates for the exact Rust-emitted matrix ports. -/
theorem outputCopyAt_finalPorts (lane : Fin 4) :
    (outputCopyAt lane).finalPorts = expectedOutputCopyPorts lane := by
  fin_cases lane <;> rfl

def outputCopyPoint
    (assignment : AbsoluteAssignment callPlacement8)
    (placement : OutputCopyPlacement) : Fin 13 → F :=
  fun port => absolutePortAction assignment
    (placement.finalPorts.getD port.val emptyAbsolutePort)

theorem outputCopyPoint_exact
    (assignment : AbsoluteAssignment callPlacement8) (lane : Fin 4) :
    outputCopyPoint assignment (outputCopyAt lane) =
      productPoint
        (absoluteValue assignment callPlacement8.selectorColumn) 0 0
        (absolutePortAction assignment (outputImageAt lane).port -
          absolutePortAction assignment
            (callOutputPort callPlacement8 (outputLane lane))) := by
  unfold outputCopyPoint
  rw [outputCopyAt_finalPorts]
  funext port
  fin_cases port <;>
    simp [expectedOutputCopyPorts, productPoint, sparsePoint,
      absolutePortAction_empty, absolutePortAction_selector,
      absolutePortAction_outputCopyC]

def outputCopyRowIndex
    (placement : OutputCopyPlacement) {rows : Nat}
    (rowFit : placement.finalRow < rows) : Fin rows :=
  ⟨placement.finalRow, rowFit⟩

/-- Exact action of one output-copy row in the complete final matrix. The
complete terminal artifact must prove this obligation structurally. -/
structure OutputCopyRowExact
    (placement : OutputCopyPlacement) {rows : Nat}
    (relation : InterpretedRelation rows callPlacement8.finalColumns)
    (assignment : AbsoluteAssignment callPlacement8) : Prop where
  rowFit : placement.finalRow < rows
  pointExact :
    rowPoint relation assignment (outputCopyRowIndex placement rowFit) =
      outputCopyPoint assignment placement

/-- One selected output-copy row binds its public image to the corresponding
final Poseidon2 call output on the same assignment. -/
theorem output_copy_row_implies_call_output_field
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (lane : Fin 4)
    (exact : OutputCopyRowExact (outputCopyAt lane) relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1) :
    absolutePortAction assignment (outputImageAt lane).port =
      absolutePortAction assignment
        (callOutputPort callPlacement8 (outputLane lane)) := by
  let row := outputCopyRowIndex (outputCopyAt lane) exact.rowFit
  let gap :=
    absolutePortAction assignment (outputImageAt lane).port -
      absolutePortAction assignment
        (callOutputPort callPlacement8 (outputLane lane))
  have pointExact :
      rowPoint relation assignment row =
        productPoint
          (absoluteValue assignment callPlacement8.selectorColumn) 0 0 gap := by
    exact exact.pointExact.trans (outputCopyPoint_exact assignment lane)
  have rowZero := satisfied row
  rw [residualAt_productPoint relation assignment row
    (absoluteValue assignment callPlacement8.selectorColumn) 0 0 gap
    pointExact] at rowZero
  have negGapZero : -gap = 0 := by
    simpa [productResidual, productPoint, sparsePoint, selectorOne] using rowZero
  exact sub_eq_zero.mp (neg_eq_zero.mp negGapZero)

theorem output_copy_row_implies_call_output
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (lane : Fin 4)
    (exact : OutputCopyRowExact (outputCopyAt lane) relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1) :
    outputImageValue assignment lane =
      callOutputValue callPlacement8 assignment (outputLane lane) := by
  calc
    outputImageValue assignment lane =
        (absolutePortAction assignment
          (callOutputPort callPlacement8 (outputLane lane))).val :=
      congrArg Fin.val
        (output_copy_row_implies_call_output_field lane exact satisfied
          selectorOne)
    _ = callOutputValue callPlacement8 assignment (outputLane lane) :=
      callOutputPort_action callPlacement8_valid assignment (outputLane lane)

/-- The nine exact call slices and four exact output-copy rows compute the
public four-field Poseidon2 hash of the ordered 32-field XOut frame. -/
theorem final_rows_compute_public_terminal_x_out_hash
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (exact0 : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      assignment)
    (exact1 : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      assignment)
    (exact2 : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      assignment)
    (exact3 : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      assignment)
    (exact4 : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      assignment)
    (exact5 : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      assignment)
    (exact6 : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      assignment)
    (exact7 : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      assignment)
    (exact8 : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      assignment)
    (copyExact : ∀ lane : Fin 4,
      OutputCopyRowExact (outputCopyAt lane) relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1)
    (lane : Fin 4) :
    outputImageValue assignment lane =
      runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0)
        lane.val := by
  calc
    outputImageValue assignment lane =
        callOutputValue callPlacement8 assignment (outputLane lane) :=
      output_copy_row_implies_call_output lane (copyExact lane) satisfied
        selectorOne
    _ = runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0)
          (outputLane lane).val :=
      final_rows_compute_terminal_x_out_hash exact0 exact1 exact2 exact3 exact4
        exact5 exact6 exact7 exact8 satisfied one selectorOne (outputLane lane)
    _ = runValueRounds rounds (terminalXOutValues assignment) (fun _ => 0)
          lane.val := rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge
