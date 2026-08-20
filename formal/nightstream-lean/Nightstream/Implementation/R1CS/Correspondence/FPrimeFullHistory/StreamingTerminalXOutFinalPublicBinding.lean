import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge

/-!
Contract: final selective-row binding from the terminal XOut Poseidon2 output
to the verifier-owned four-word public input.

Owns the 256 public equality-row semantics, the public-prefix projection, and
the field reconstruction of each 64-value public word. Exact final-row
identity stays an explicit generated-artifact obligation.

Does not own the complete terminal matrix, lifecycle composition, recursive
size closure, or collision resistance.

Assurance tier: artifact-checked once the explicit final-row premises hold for
the Nightstream b2/k16 terminal profile.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutFinalPublicBinding

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge
open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash

private abbrev callPlacement8 :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.callPlacement8

def publicBitColumn (lane : Fin 4) (bit : Fin 64) : Nat :=
  1 + lane.val * 64 + bit.val

def outputBitColumn (lane : Fin 4) (bit : Fin 64) : Nat :=
  22054892 + lane.val * 64 + bit.val

private theorem publicBitColumn_bound (lane : Fin 4) (bit : Fin 64) :
    publicBitColumn lane bit < callPlacement8.finalColumns := by
  norm_num [publicBitColumn, callPlacement8,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.callPlacement8]
  omega

private theorem outputBitColumn_bound (lane : Fin 4) (bit : Fin 64) :
    outputBitColumn lane bit < callPlacement8.finalColumns := by
  norm_num [outputBitColumn, callPlacement8,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.callPlacement8]
  omega

def publicBitField
    (assignment : AbsoluteAssignment callPlacement8)
    (lane : Fin 4) (bit : Fin 64) : F :=
  absoluteValue assignment (publicBitColumn lane bit)

def outputBitField
    (assignment : AbsoluteAssignment callPlacement8)
    (lane : Fin 4) (bit : Fin 64) : F :=
  absoluteValue assignment (outputBitColumn lane bit)

/-- Final row index of one public-to-canonical bit equality. The first public
word block starts at row 5007881, with 69 canonical rows before its 64 links;
each complete word block has 133 rows. -/
def publicLinkFinalRow (lane : Fin 4) (bit : Fin 64) : Nat :=
  5007950 + lane.val * 133 + bit.val

def publicLinkRowIndex
    (lane : Fin 4) (bit : Fin 64) {rows : Nat}
    (rowFit : publicLinkFinalRow lane bit < rows) : Fin rows :=
  ⟨publicLinkFinalRow lane bit, rowFit⟩

def publicLinkPoint
    (assignment : AbsoluteAssignment callPlacement8)
    (lane : Fin 4) (bit : Fin 64) : Fin 13 → F :=
  productPoint
    (absoluteValue assignment callPlacement8.selectorColumn) 0 0
    (outputBitField assignment lane bit - publicBitField assignment lane bit)

/-- Exact action of one rewritten public equality row. The complete terminal
artifact must discharge this obligation from the emitted final matrix. -/
structure PublicLinkRowExact
    (lane : Fin 4) (bit : Fin 64) {rows : Nat}
    (relation : InterpretedRelation rows callPlacement8.finalColumns)
    (assignment : AbsoluteAssignment callPlacement8) : Prop where
  rowFit : publicLinkFinalRow lane bit < rows
  pointExact :
    rowPoint relation assignment (publicLinkRowIndex lane bit rowFit) =
      publicLinkPoint assignment lane bit

/-- One satisfied rewritten public link identifies a canonical output bit
with its verifier-owned public-prefix value. -/
theorem public_link_row_implies_bit
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    {assignment : AbsoluteAssignment callPlacement8}
    (lane : Fin 4) (bit : Fin 64)
    (rowExact : PublicLinkRowExact lane bit relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (selectorOne :
      absoluteValue assignment callPlacement8.selectorColumn = 1) :
    outputBitField assignment lane bit =
      publicBitField assignment lane bit := by
  let row := publicLinkRowIndex lane bit rowExact.rowFit
  let gap := outputBitField assignment lane bit -
    publicBitField assignment lane bit
  have pointExact :
      rowPoint relation assignment row =
        productPoint
          (absoluteValue assignment callPlacement8.selectorColumn) 0 0 gap :=
    rowExact.pointExact
  have rowZero := satisfied row
  rw [residualAt_productPoint relation assignment row
    (absoluteValue assignment callPlacement8.selectorColumn) 0 0 gap
    pointExact] at rowZero
  have negGapZero : -gap = 0 := by
    simpa [productResidual, productPoint, sparsePoint, selectorOne] using rowZero
  exact sub_eq_zero.mp (neg_eq_zero.mp negGapZero)

private theorem publicBitColumn_source_exact
    (lane : Fin 4) (index : Nat) (bounded : index < 64) :
    (publicWordAt lane).publicBitColumns.getD index 0 =
      1 + lane.val * 64 + index := by
  fin_cases lane <;> interval_cases index <;> rfl

private theorem outputImageAt_port_exact (lane : Fin 4) :
    (outputImageAt lane).port =
      { explicit := []
        geometric :=
          [{ columnStart := 22054892 + lane.val * 64
             length := 64
             initial := 1
             ratio := 2 }] } := by
  fin_cases lane <;> rfl

private theorem absolutePortAction_single_geometric
    {placement : PoseidonCallPlacement}
    (assignment : AbsoluteAssignment placement)
    (run : AbsoluteGeometricRun) :
    absolutePortAction assignment
        { explicit := [], geometric := [run] } =
      geometricRunAction assignment run := by
  simp [absolutePortAction, sum]

private def residue (value : Nat) : F :=
  ⟨value % goldilocksP, Nat.mod_lt value (by decide)⟩

private theorem residue_add (left right : Nat) :
    residue (left + right) = residue left + residue right := by
  apply Fin.ext
  change
    (left + right) % goldilocksModulus =
      (left % goldilocksModulus + right % goldilocksModulus) %
        goldilocksModulus
  exact Nat.add_mod left right goldilocksModulus

private theorem residue_mul (left right : Nat) :
    residue (left * right) = residue left * residue right := by
  apply Fin.ext
  change
    (left * right) % goldilocksModulus =
      (left % goldilocksModulus * (right % goldilocksModulus)) %
        goldilocksModulus
  exact Nat.mul_mod left right goldilocksModulus

private theorem residue_eq_fieldValue
    (value : Nat) (canonical : value < goldilocksP) :
    residue value = fieldValue value := by
  rw [fieldValue_of_lt value canonical]
  apply Fin.ext
  simp [residue, Nat.mod_eq_of_lt canonical]

private theorem geometricCoefficient_binary (index : Nat) :
    geometricCoefficient (fieldValue 1) (fieldValue 2) index =
      residue (2 ^ index) := by
  induction index with
  | zero =>
      rw [geometricCoefficient]
      exact (residue_eq_fieldValue 1 (by decide)).symm
  | succ index inductionHypothesis =>
      rw [geometricCoefficient, inductionHypothesis,
        ← residue_eq_fieldValue 2 (by decide), ← residue_mul]
      simp [Nat.pow_succ]

private theorem ofDigits_range_map
    (base count : Nat) (digit : Nat → Nat) :
    Nat.ofDigits base ((List.range count).map digit) =
      ((List.range count).map fun index => digit index * base ^ index).sum := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.range_succ, Nat.ofDigits_append, inductionHypothesis,
        Nat.mul_comm]

private theorem residue_weighted_sum
    (digit : Nat → Nat) : ∀ indices : List Nat,
    sum (indices.map fun index =>
        residue (2 ^ index) * residue (digit index)) =
      residue ((indices.map fun index => digit index * 2 ^ index).sum)
  | [] => rfl
  | index :: tail => by
      simp only [List.map_cons, sum, List.sum_cons,
        residue_weighted_sum digit tail]
      rw [← residue_mul, Nat.mul_comm, ← residue_add]

private theorem publicWordValue_direct
    (assignment : Nat → Nat) (lane : Fin 4) :
    publicWordValue assignment lane =
      Nat.ofDigits 2 ((List.range 64).map fun index =>
        assignment (1 + lane.val * 64 + index)) := by
  unfold publicWordValue
  congr 1
  apply List.map_congr_left
  intro index member
  rw [publicBitColumn_source_exact lane index (List.mem_range.mp member)]

private theorem output_run_terms_eq_public_terms
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    (source : Nat → Nat)
    (canonical : ∀ column, source column < goldilocksP)
    (lane : Fin 4)
    (links : ∀ bit : Fin 64,
      PublicLinkRowExact lane bit relation
        (projectedFinalAssignment source canonical))
    (satisfied : AllRowsSatisfied relation
      (projectedFinalAssignment source canonical))
    (selectorOne :
      absoluteValue (projectedFinalAssignment source canonical)
        callPlacement8.selectorColumn = 1) :
    (List.range 64).map (fun index =>
        geometricCoefficient (fieldValue 1) (fieldValue 2) index *
          absoluteValue (projectedFinalAssignment source canonical)
            (22054892 + lane.val * 64 + index)) =
      (List.range 64).map (fun index =>
        residue (2 ^ index) *
          residue
            (projectedFinalValues source (1 + lane.val * 64 + index))) := by
  apply List.map_congr_left
  intro index member
  have bounded : index < 64 := List.mem_range.mp member
  let bit : Fin 64 := ⟨index, bounded⟩
  rw [geometricCoefficient_binary]
  have linked := public_link_row_implies_bit lane bit (links bit) satisfied
    selectorOne
  have projected := absoluteValue_projected_eq_fieldValue source canonical
    (publicBitColumn lane bit) (publicBitColumn_bound lane bit)
  change
    residue (2 ^ index) *
        outputBitField (projectedFinalAssignment source canonical) lane bit =
      residue (2 ^ index) *
        residue (projectedFinalValues source (1 + lane.val * 64 + index))
  rw [linked]
  have publicExact :
      publicBitField (projectedFinalAssignment source canonical) lane bit =
        fieldValue
          (projectedFinalValues source (1 + lane.val * 64 + index)) := by
    simpa [publicBitField, publicBitColumn, projectedFinalValues, bit,
      Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using projected
  rw [publicExact, ← residue_eq_fieldValue _
    (by
      simpa [projectedFinalValues] using
        canonical ((1 + lane.val * 64 + index) + 1))]

private theorem output_image_field_eq_public_word
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    (source : Nat → Nat)
    (canonical : ∀ column, source column < goldilocksP)
    (lane : Fin 4)
    (links : ∀ bit : Fin 64,
      PublicLinkRowExact lane bit relation
        (projectedFinalAssignment source canonical))
    (satisfied : AllRowsSatisfied relation
      (projectedFinalAssignment source canonical))
    (selectorOne :
      absoluteValue (projectedFinalAssignment source canonical)
        callPlacement8.selectorColumn = 1) :
    absolutePortAction (projectedFinalAssignment source canonical)
        (outputImageAt lane).port =
      residue (publicWordValue (projectedFinalValues source) lane) := by
  rw [outputImageAt_port_exact lane]
  rw [absolutePortAction_single_geometric]
  unfold geometricRunAction
  rw [output_run_terms_eq_public_terms source canonical lane links satisfied
    selectorOne]
  rw [residue_weighted_sum]
  rw [← ofDigits_range_map]
  rw [← publicWordValue_direct (projectedFinalValues source)]

/-- Exact final call, output-copy, and public-link rows derive the public
terminal XOut hash from verifier-owned public input. No digest equality is a
premise. -/
theorem final_rows_imply_outer_hash
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    (source : Nat → Nat)
    (canonical : ∀ column, source column < goldilocksP)
    (exact0 : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      (projectedFinalAssignment source canonical))
    (exact1 : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      (projectedFinalAssignment source canonical))
    (exact2 : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      (projectedFinalAssignment source canonical))
    (exact3 : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      (projectedFinalAssignment source canonical))
    (exact4 : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      (projectedFinalAssignment source canonical))
    (exact5 : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      (projectedFinalAssignment source canonical))
    (exact6 : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      (projectedFinalAssignment source canonical))
    (exact7 : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      (projectedFinalAssignment source canonical))
    (exact8 : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      (projectedFinalAssignment source canonical))
    (copyExact : ∀ lane : Fin 4,
      OutputCopyRowExact (outputCopyAt lane) relation
        (projectedFinalAssignment source canonical))
    (linkExact : ∀ lane : Fin 4, ∀ bit : Fin 64,
      PublicLinkRowExact lane bit relation
        (projectedFinalAssignment source canonical))
    (satisfied : AllRowsSatisfied relation
      (projectedFinalAssignment source canonical))
    (one : absoluteValue (projectedFinalAssignment source canonical) 0 = 1)
    (selectorOne :
      absoluteValue (projectedFinalAssignment source canonical)
        callPlacement8.selectorColumn = 1)
    (publicXOut :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Digest)
    (binding : PublicAssignmentBinding (projectedFinalValues source)
      publicXOut) :
    outerHash
        (terminalXOutValues (projectedFinalAssignment source canonical)) =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
        publicXOut := by
  have schedules :
      valueSchedules rounds =
        valueSchedules
          Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds := by
    exact valueSchedules_exact.trans
      Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds_schedule.symm
  have sameHash := runValueRounds_eq_of_schedules schedules
    (terminalXOutValues (projectedFinalAssignment source canonical))
    (fun _ => 0)
  funext lane
  calc
    outerHash (terminalXOutValues (projectedFinalAssignment source canonical))
        lane =
        runValueRounds
          Nightstream.Implementation.Nebula.StateOutputPoseidonRows.representativeRounds
          (terminalXOutValues (projectedFinalAssignment source canonical))
          (fun _ => 0) lane.val := rfl
    _ = runValueRounds rounds
          (terminalXOutValues (projectedFinalAssignment source canonical))
          (fun _ => 0) lane.val := (congrFun sameHash lane.val).symm
    _ = outputImageValue (projectedFinalAssignment source canonical) lane :=
      (final_rows_compute_public_terminal_x_out_hash exact0 exact1 exact2
        exact3 exact4 exact5 exact6 exact7 exact8 copyExact satisfied one
        selectorOne lane).symm
    _ = (absolutePortAction (projectedFinalAssignment source canonical)
          (outputImageAt lane).port).val := rfl
    _ = (residue
          (publicWordValue (projectedFinalValues source) lane)).val :=
      congrArg Fin.val
        (output_image_field_eq_public_word source canonical lane
          (linkExact lane) satisfied selectorOne)
    _ = publicWordValue (projectedFinalValues source) lane := by
      simp only [residue]
      rw [Nat.mod_eq_of_lt]
      exact binding.value lane ▸
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues_canonical
          publicXOut lane
    _ = Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.digestValues
          publicXOut lane := binding.value lane

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutFinalPublicBinding
