import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalSourceBindingRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallPermutation

/-!
Contract: exact same-assignment bridge from the 32 terminal source-binding
decoder outputs to the final selective-CCS ports consumed by the terminal
XOut Poseidon2 hash.

Owns the source-binding column shift, the exact pairing of the 32 Rust
decoder bindings with the 32 Rust source images, and their field evaluation.

Does not own source-row transport into the final selective relation, public
word binding, lifecycle composition, or collision resistance.

Assurance tier: artifact-checked for the Nightstream b2/k16 terminal profile
once the exact source-binding rows hold.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBindingRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallPermutation

private abbrev sourceArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding.rawArtifact

private abbrev sourceDecoderGroups :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding.decoderGroups

private abbrev xOutImages :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.xOutImages

private abbrev callPlacement8 :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.callPlacement8

private structure FieldBinding where
  sourceColumn : Nat
  decodedColumn : Nat
  terms : List (Nat × Nat)

private def emptyFieldBinding : FieldBinding where
  sourceColumn := 0
  decodedColumn := 0
  terms := []

private def blockBindings (block : DecoderBlock) : List FieldBinding :=
  (List.range block.count).map fun index =>
    { sourceColumn := block.sourceFields.start + index
      decodedColumn := block.decodedColumns.start + index
      terms := block.termsAt index }

private def groupBindings : DecoderGroup → List FieldBinding
  | .block block => blockBindings block
  | .composite decoder =>
      [{ sourceColumn := decoder.sourceField
         decodedColumn := decoder.decodedColumn
         terms := decoder.terms }]

/-- The generated source-binding artifact places all 13 XOut decoder groups
first. Their flattened binding list has the fixed 32-field lifecycle shape. -/
private def xOutGroups : List DecoderGroup :=
  sourceDecoderGroups.take 13

private def xOutBindings : List FieldBinding :=
  xOutGroups.flatMap groupBindings

private theorem xOutBindings_length : xOutBindings.length = 32 := by
  rfl

private def xOutBindingAt (lane : Fin 32) : FieldBinding :=
  xOutBindings.getD lane.val emptyFieldBinding

private def xOutImageAt (lane : Fin 32) : SourceImage :=
  xOutImages.getD lane.val emptySourceImage

private theorem getD_mem_of_lt {alpha : Type}
    (fallback : alpha) {entries : List alpha} {index : Nat}
    (bounded : index < entries.length) :
    entries.getD index fallback ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD fallback] at member

private theorem xOutBindingAt_member (lane : Fin 32) :
    xOutBindingAt lane ∈ xOutBindings := by
  change xOutBindings.getD lane.val emptyFieldBinding ∈ xOutBindings
  apply getD_mem_of_lt emptyFieldBinding
  rw [xOutBindings_length]
  exact lane.isLt

private theorem groupBindings_hold
    (assignment : Nat → Nat) (group : DecoderGroup)
    (holds : group.Holds assignment) :
    ∀ binding ∈ groupBindings group,
      assignment binding.decodedColumn = lcEval assignment binding.terms := by
  cases group with
  | block block =>
      change block.Holds assignment at holds
      intro binding member
      rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
      exact holds index (List.mem_range.mp indexMember)
  | composite decoder =>
      change decoder.Holds assignment at holds
      intro binding member
      simp only [groupBindings, List.mem_singleton] at member
      subst binding
      exact holds

private theorem xOutBindingAt_holds
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : sourceArtifact.Satisfied assignment)
    (lane : Fin 32) :
    assignment (xOutBindingAt lane).decodedColumn =
      lcEval assignment (xOutBindingAt lane).terms := by
  rcases List.mem_flatMap.mp (xOutBindingAt_member lane) with
    ⟨group, groupMember, bindingMember⟩
  have fullMember : group ∈ sourceDecoderGroups :=
    List.mem_of_mem_take groupMember
  have groupHolds := rows_imply_decoder_groups assignment canonical one
    satisfied group fullMember
  exact groupBindings_hold assignment group groupHolds
    (xOutBindingAt lane) bindingMember

private def shiftExplicitTerm (term : AbsoluteTerm) : Nat × Nat :=
  (term.column + 1, term.coefficient)

private def shiftedRunTerms (run : AbsoluteGeometricRun) : List (Nat × Nat) :=
  decoderTerms (run.columnStart + 1) run.length run.ratio run.initial

/-- Decoder segments use ascending final-column order. Rust source images
store composite geometric runs in reverse operand order, so this normal form
reverses only that outer run list. -/
private def shiftedPortTerms (port : AbsolutePort) : List (Nat × Nat) :=
  port.explicit.map shiftExplicitTerm ++
    port.geometric.reverse.flatMap shiftedRunTerms

private structure BindingImageExact
    (binding : FieldBinding) (lane : Fin 32) : Prop where
  sourceColumn : binding.sourceColumn = (xOutImageAt lane).sourceColumn
  decodedColumn : binding.decodedColumn = 28041899 + lane.val
  terms : binding.terms = shiftedPortTerms (xOutImageAt lane).port

/-- One compact exact-data certificate pairs each Rust source decoder with
the corresponding Rust final source image. It does not inspect any row set. -/
private theorem xOutBindingAt_exact (lane : Fin 32) :
    BindingImageExact (xOutBindingAt lane) lane := by
  fin_cases lane <;> exact ⟨rfl, rfl, rfl⟩

private def PortReady (port : AbsolutePort) (bound : Nat) : Prop :=
  (∀ term ∈ port.explicit,
      term.column < bound ∧ term.coefficient < goldilocksP) ∧
    ∀ run ∈ port.geometric,
      run.columnStart + run.length ≤ bound ∧
        run.initial < goldilocksP ∧ run.ratio < goldilocksP

/-- Small fixed geometry certificate for the 32 final XOut ports. -/
private theorem xOutImageAt_ready (lane : Fin 32) :
    PortReady (xOutImageAt lane).port callPlacement8.finalColumns := by
  fin_cases lane <;>
    norm_num [PortReady, xOutImageAt,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.xOutImages,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash.callPlacement8,
      goldilocksP, goldilocksModulus]

private def shiftedField
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) : F :=
  ⟨assignment (column + 1), by
    simpa [goldilocksP, goldilocksModulus] using canonical (column + 1)⟩

/-- Numeric view of the final selective assignment inside the source-binding
row system. Source column zero is the row-system constant, so final column
`c` is source column `c + 1`. -/
def projectedFinalValues (assignment : Nat → Nat) (column : Nat) : Nat :=
  assignment (column + 1)

/-- The source-binding schema has one leading row-system constant column.
Thus final selective column `c` is source-binding column `c + 1`. -/
def projectedFinalAssignment
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    AbsoluteAssignment callPlacement8 :=
  fun column => shiftedField assignment canonical column.val

theorem absoluteValue_projected
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) (bounded : column < callPlacement8.finalColumns) :
    absoluteValue (projectedFinalAssignment assignment canonical) column =
      shiftedField assignment canonical column := by
  simp [absoluteValue, bounded, projectedFinalAssignment]

/-- The one-column source-to-final shift preserves every canonical numeric
value as its Goldilocks field element. -/
theorem absoluteValue_projected_eq_fieldValue
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) (bounded : column < callPlacement8.finalColumns) :
    absoluteValue (projectedFinalAssignment assignment canonical) column =
      fieldValue (assignment (column + 1)) := by
  rw [absoluteValue_projected assignment canonical column bounded]
  rw [fieldValue_of_lt _ (canonical (column + 1))]
  rfl

private def residue (value : Nat) : F :=
  ⟨value % goldilocksP, by
    simpa [goldilocksP, goldilocksModulus] using
      Nat.mod_lt value (by decide : 0 < goldilocksP)⟩

private theorem residue_eq_fieldValue
    (value : Nat) (bounded : value < goldilocksP) :
    residue value = fieldValue value := by
  rw [fieldValue_of_lt value bounded]
  apply Fin.ext
  exact Nat.mod_eq_of_lt bounded

private theorem residue_assignment_eq_shiftedField
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    residue (assignment (column + 1)) =
      shiftedField assignment canonical column := by
  apply Fin.ext
  exact Nat.mod_eq_of_lt (canonical (column + 1))

private theorem residue_mod (value : Nat) :
    residue (value % goldilocksP) = residue value := by
  apply Fin.ext
  exact Nat.mod_mod value goldilocksP

private def fieldEval
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) : F :=
  ⟨lcEval assignment terms, by
    unfold lcEval
    simpa [goldilocksP, goldilocksModulus] using
      Nat.mod_lt
        (terms.foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) 0)
        (by decide : 0 < goldilocksP)⟩

private theorem fieldEval_cons
    (assignment : Nat → Nat) (term : Nat × Nat)
    (rest : List (Nat × Nat)) :
    fieldEval assignment (term :: rest) =
      residue term.2 * residue (assignment term.1) +
        fieldEval assignment rest := by
  apply Fin.ext
  simp only [fieldEval, residue, Fin.val_add, Fin.val_mul, Fin.val_mk]
  simp [lcEval_eq_raw_mod, rawLcEval, Nat.add_mod, Nat.mul_mod,
    goldilocksP, goldilocksModulus]

private theorem rawLcEval_append
    (assignment : Nat → Nat) (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem fieldEval_append
    (assignment : Nat → Nat) (left right : List (Nat × Nat)) :
    fieldEval assignment (left ++ right) =
      fieldEval assignment left + fieldEval assignment right := by
  apply Fin.ext
  simp only [fieldEval, Fin.val_add, Fin.val_mk]
  rw [lcEval_eq_raw_mod, rawLcEval_append, Nat.add_mod]
  rw [← lcEval_eq_raw_mod, ← lcEval_eq_raw_mod]
  simp [goldilocksP, goldilocksModulus]

private theorem explicitTerm_action
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (term : AbsoluteTerm)
    (columnBound : term.column < callPlacement8.finalColumns)
    (coefficientBound : term.coefficient < goldilocksP) :
    residue term.coefficient * residue (assignment (term.column + 1)) =
      explicitAction (projectedFinalAssignment assignment canonical) term := by
  unfold explicitAction
  rw [absoluteValue_projected assignment canonical term.column columnBound]
  rw [residue_eq_fieldValue term.coefficient coefficientBound,
    residue_assignment_eq_shiftedField assignment canonical term.column]

private theorem fieldEval_explicitTerms
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ terms : List AbsoluteTerm,
      (∀ term ∈ terms,
        term.column < callPlacement8.finalColumns ∧
          term.coefficient < goldilocksP) →
      fieldEval assignment (terms.map shiftExplicitTerm) =
        sum (terms.map
          (explicitAction (projectedFinalAssignment assignment canonical))) := by
  intro terms ready
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [fieldEval_cons]
      simp only [shiftExplicitTerm]
      rw [explicitTerm_action assignment canonical head
        (ready head (by simp)).1 (ready head (by simp)).2]
      rw [inductionHypothesis (fun term member =>
        ready term (by simp [member]))]

private theorem geometricCoefficient_exact
    (initial ratio : Nat)
    (initialBound : initial < goldilocksP)
    (ratioBound : ratio < goldilocksP) :
    ∀ index,
      geometricCoefficient (fieldValue initial) (fieldValue ratio) index =
        residue (initial * ratio ^ index) := by
  intro index
  induction index with
  | zero =>
      rw [geometricCoefficient, fieldValue_of_lt initial initialBound]
      apply Fin.ext
      simpa [residue, goldilocksP, goldilocksModulus] using
        (Nat.mod_eq_of_lt initialBound).symm
  | succ index inductionHypothesis =>
      rw [geometricCoefficient, inductionHypothesis,
        fieldValue_of_lt ratio ratioBound]
      apply Fin.ext
      simp only [residue, Fin.val_mul, Fin.val_mk]
      simp only [goldilocksP, goldilocksModulus]
      rw [Nat.pow_succ, Nat.mod_mul_mod]
      rw [Nat.mul_assoc]

private theorem shiftedRunTerm_action
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (run : AbsoluteGeometricRun)
    (columnBound : run.columnStart + run.length ≤
      callPlacement8.finalColumns)
    (initialBound : run.initial < goldilocksP)
    (ratioBound : run.ratio < goldilocksP)
    (index : Nat) (indexBound : index < run.length) :
    residue ((run.initial * run.ratio ^ index) % goldilocksP) *
        residue (assignment (run.columnStart + 1 + index)) =
      geometricCoefficient (fieldValue run.initial) (fieldValue run.ratio)
          index *
        absoluteValue (projectedFinalAssignment assignment canonical)
          (run.columnStart + index) := by
  have absoluteBound :
      run.columnStart + index < callPlacement8.finalColumns := by
    omega
  rw [geometricCoefficient_exact run.initial run.ratio initialBound
      ratioBound index,
    absoluteValue_projected assignment canonical
      (run.columnStart + index) absoluteBound]
  rw [show run.columnStart + 1 + index =
      run.columnStart + index + 1 by omega]
  rw [residue_mod,
    residue_assignment_eq_shiftedField assignment canonical
      (run.columnStart + index)]

private theorem fieldEval_runIndices
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (run : AbsoluteGeometricRun)
    (columnBound : run.columnStart + run.length ≤
      callPlacement8.finalColumns)
    (initialBound : run.initial < goldilocksP)
    (ratioBound : run.ratio < goldilocksP) :
    ∀ indices : List Nat,
      (∀ index ∈ indices, index < run.length) →
      fieldEval assignment
          (indices.map fun index =>
            (run.columnStart + 1 + index,
              (run.initial * run.ratio ^ index) % goldilocksP)) =
        sum (indices.map fun index =>
          geometricCoefficient (fieldValue run.initial)
              (fieldValue run.ratio) index *
            absoluteValue (projectedFinalAssignment assignment canonical)
              (run.columnStart + index)) := by
  intro indices indicesBound
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [fieldEval_cons]
      rw [shiftedRunTerm_action assignment canonical run columnBound
        initialBound ratioBound head (indicesBound head (by simp))]
      rw [inductionHypothesis (fun index member =>
        indicesBound index (by simp [member]))]

private theorem fieldEval_shiftedRunTerms
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (run : AbsoluteGeometricRun)
    (ready : run.columnStart + run.length ≤ callPlacement8.finalColumns ∧
      run.initial < goldilocksP ∧ run.ratio < goldilocksP) :
    fieldEval assignment (shiftedRunTerms run) =
      geometricRunAction (projectedFinalAssignment assignment canonical) run := by
  unfold shiftedRunTerms decoderTerms geometricRunAction
  exact fieldEval_runIndices assignment canonical run ready.1 ready.2.1
    ready.2.2 (List.range run.length)
    (fun index member => List.mem_range.mp member)

private theorem fieldEval_shiftedRuns
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ runs : List AbsoluteGeometricRun,
      (∀ run ∈ runs,
        run.columnStart + run.length ≤ callPlacement8.finalColumns ∧
          run.initial < goldilocksP ∧ run.ratio < goldilocksP) →
      fieldEval assignment (runs.flatMap shiftedRunTerms) =
        sum (runs.map
          (geometricRunAction
            (projectedFinalAssignment assignment canonical))) := by
  intro runs ready
  induction runs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, fieldEval_append]
      simp only [List.map_cons, sum]
      rw [fieldEval_shiftedRunTerms assignment canonical head
        (ready head (by simp))]
      rw [inductionHypothesis (fun run member =>
        ready run (by simp [member]))]

private theorem fieldEval_shiftedPortTerms
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (port : AbsolutePort)
    (ready : PortReady port callPlacement8.finalColumns) :
    fieldEval assignment (shiftedPortTerms port) =
      absolutePortAction (projectedFinalAssignment assignment canonical) port := by
  unfold shiftedPortTerms absolutePortAction
  rw [fieldEval_append]
  rw [fieldEval_explicitTerms assignment canonical port.explicit ready.1]
  rw [fieldEval_shiftedRuns assignment canonical port.geometric.reverse]
  · rw [sum_map_eq_of_perm (List.reverse_perm port.geometric)
      (geometricRunAction
        (projectedFinalAssignment assignment canonical))]
  · intro run member
    exact ready.2 run (by simpa using member)

private theorem lcEval_shiftedPortTerms
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (port : AbsolutePort)
    (ready : PortReady port callPlacement8.finalColumns) :
    lcEval assignment (shiftedPortTerms port) =
      (absolutePortAction
        (projectedFinalAssignment assignment canonical) port).val := by
  exact congrArg Fin.val
    (fieldEval_shiftedPortTerms assignment canonical port ready)

/-- Exact source-binding rows recover one decoded XOut field as the same
final source-image value consumed by the terminal Poseidon2 call sequence. -/
theorem rows_imply_decoded_x_out_value
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : sourceArtifact.Satisfied assignment)
    (lane : Fin 32) :
    assignment (28041899 + lane.val) =
      (absolutePortAction (projectedFinalAssignment assignment canonical)
        (xOutImageAt lane).port).val := by
  have exact := xOutBindingAt_exact lane
  calc
    assignment (28041899 + lane.val) =
        assignment (xOutBindingAt lane).decodedColumn := by
      rw [exact.decodedColumn]
    _ = lcEval assignment (xOutBindingAt lane).terms :=
      xOutBindingAt_holds assignment canonical one satisfied lane
    _ = lcEval assignment (shiftedPortTerms (xOutImageAt lane).port) := by
      rw [exact.terms]
    _ = (absolutePortAction (projectedFinalAssignment assignment canonical)
          (xOutImageAt lane).port).val :=
      lcEval_shiftedPortTerms assignment canonical (xOutImageAt lane).port
        (xOutImageAt_ready lane)

/-- Ordered decoded XOut values in the exact 32-field lifecycle layout. -/
def decodedXOutValues (assignment : Nat → Nat) : List Nat :=
  List.ofFn fun lane : Fin 32 => assignment (28041899 + lane.val)

private def finalXOutValues
    (assignment : AbsoluteAssignment callPlacement8) : List Nat :=
  List.ofFn fun lane : Fin 32 =>
    (absolutePortAction assignment (xOutImageAt lane).port).val

private theorem finalXOutValues_eq_terminalXOutValues
    (assignment : AbsoluteAssignment callPlacement8) :
    finalXOutValues assignment = terminalXOutValues assignment := by
  rfl

/-- The 32 exact source-binding decoder rows and the final terminal hash use
the same ordered field values on one projected assignment. -/
theorem source_rows_imply_terminal_x_out_values
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : sourceArtifact.Satisfied assignment) :
    decodedXOutValues assignment =
      terminalXOutValues (projectedFinalAssignment assignment canonical) := by
  calc
    decodedXOutValues assignment =
        finalXOutValues (projectedFinalAssignment assignment canonical) := by
      unfold decodedXOutValues finalXOutValues
      apply congrArg List.ofFn
      funext lane
      exact rows_imply_decoded_x_out_value assignment canonical one satisfied
        lane
    _ = terminalXOutValues
          (projectedFinalAssignment assignment canonical) :=
      finalXOutValues_eq_terminalXOutValues _

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge
