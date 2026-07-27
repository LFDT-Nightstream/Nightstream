import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest

/-!
Contract: exact active, honest, and inactive semantics of the typed
fixed-23 canonical Poseidon2 sponge occurrence.

Owns: transport between the typed assignment and the canonical numeric
program, the four activation gates, and a temporary-only honest completion.

Does not own: the optional digest wrapper, F′ preimage serialization, or
collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Numeric

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

def input
    (frame : Frame) (assignment : ColumnId → Field) : Preimage :=
  fun index => (assignment (inputColumn frame index.val)).val

def pulled
    (frame : Frame) (assignment : ColumnId → Field) : Nat → Nat :=
  numericAssignment (columnMap frame) assignment

def semanticLane
    (frame : Frame) (assignment : ColumnId → Field)
    (lane : Fin digestLength) : Field :=
  residue
    (digest
      Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
      (dataChunks (input frame assignment)) lane)

theorem pulled_residues
    (frame : Frame) (assignment : ColumnId → Field) :
    ∀ source, pulled frame assignment source <
      Nightstream.Implementation.R1CS.goldilocksP :=
  numericAssignment_canonical (columnMap frame) assignment

theorem pulled_constant
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1) :
    pulled frame assignment 0 = 1 := by
  change (assignment frame.one).val = 1
  rw [constantOne]
  rfl

theorem pulled_inputsAgree
    (frame : Frame) (assignment : ColumnId → Field) :
    InputsAgree (pulled frame assignment) (input frame assignment) := by
  intro index
  simp only [pulled, numericAssignment, input]
  rw [show
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn
          index.val = 2527 + index.val by
        simp only [
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn,
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq],
    columnMap_input frame index.val (by
      have indexLt := index.isLt
      change index.val < 23 at indexLt
      exact indexLt)]

end Numeric

private theorem satisfies_iff_forall
    (source : List OwnedRow) (assignment : ColumnId → Field) :
    Satisfies source assignment ↔
      ∀ owned, owned ∈ source → owned.row.Holds assignment := by
  induction source with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [satisfies_cons, inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ owned member
        rcases List.mem_cons.mp member with rfl | tailMember
        · exact headHolds
        · exact tailHolds owned tailMember
      · intro all
        exact ⟨
          all head (by simp),
          fun owned member => all owned (by simp [member])
        ⟩

theorem gateRow_active_iff
    (frame : Frame) (assignment : ColumnId → Field)
    (activeOne : assignment frame.active = 1) (lane : Nat) :
    (gateRow frame lane).row.Holds assignment ↔
      assignment (internalOutputColumn frame lane) =
        assignment (outputColumn frame lane) := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton, Goldilocks.difference,
    Goldilocks.LinearCombination.eval, activeOne, Fin.one_mul, Fin.mul_one,
    Fin.add_zero, Lean.Grind.Fin.neg_mul]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
      assignment (internalOutputColumn frame lane) -
            assignment (outputColumn frame lane) = 0 ↔
        assignment (internalOutputColumn frame lane) =
          assignment (outputColumn frame lane))

theorem gateRow_complete_of_inactive
    (frame : Frame) (assignment : ColumnId → Field)
    (activeZero : assignment frame.active = 0) (lane : Nat) :
    (gateRow frame lane).row.Holds assignment := by
  simp only [gateRow, Row.Holds, Goldilocks.singleton, Goldilocks.difference,
    Goldilocks.LinearCombination.eval, activeZero, Fin.one_mul, Fin.add_zero,
    Lean.Grind.Fin.neg_mul]
  exact Fin.zero_mul _

theorem gateRows_active_output_eq
    (frame : Frame) (assignment : ColumnId → Field)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (gateRows frame) assignment)
    (lane : Nat) (laneLt : lane < outputWidth) :
    assignment (internalOutputColumn frame lane) =
      assignment (outputColumn frame lane) := by
  apply (gateRow_active_iff frame assignment activeOne lane).1
  apply (satisfies_iff_forall (gateRows frame) assignment).1 holds
  apply List.mem_map.mpr
  exact ⟨lane, List.mem_range.mpr (by simpa [gateRowCount, outputWidth]
    using laneLt), rfl⟩

theorem gateRows_complete_of_inactive
    (frame : Frame) (assignment : ColumnId → Field)
    (activeZero : assignment frame.active = 0) :
    Satisfies (gateRows frame) assignment := by
  apply (satisfies_iff_forall (gateRows frame) assignment).2
  intro owned member
  rcases List.mem_map.mp member with ⟨lane, _, rfl⟩
  exact gateRow_complete_of_inactive frame assignment activeZero lane

private theorem internalOutputColumn_exact
    (frame : Frame) (lane : Nat) (laneLt : lane < outputWidth) :
    internalOutputColumn frame lane =
      columnMap frame
        ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
          ⟨lane, by
            simp only [outputWidth] at laneLt
            simp only [
              Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width
            ]
            omega⟩) := by
  unfold internalOutputColumn
  congr 3
  simp only [outputWidth] at laneLt
  simp only [Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width]
  exact Nat.mod_eq_of_lt (by omega)

/-- Active typed satisfaction computes the exact selected fixed-23 digest. -/
theorem active_sound
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (rows frame) assignment)
    (lane : Nat) (laneLt : lane < outputWidth) :
    assignment (outputColumn frame lane) =
      Numeric.semanticLane frame assignment
        ⟨lane, by simpa [outputWidth,
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
          using laneLt⟩ := by
  have split :=
    (satisfies_append_iff (coreRows frame) (gateRows frame) assignment).1 holds
  have numericSatisfies :
      Nightstream.Implementation.R1CS.Satisfies Canonical.rows
        (Numeric.pulled frame assignment) :=
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows assignment).1 split.1
  let bounded :
      Fin Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength :=
    ⟨lane, by
      simpa [outputWidth,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
        using laneLt⟩
  have numericSound :=
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program_computes_digest
      Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
      (Numeric.pulled frame assignment)
      (Numeric.input frame assignment)
      (Numeric.pulled_residues frame assignment)
      (Numeric.pulled_constant frame assignment constantOne)
      (Numeric.pulled_inputsAgree frame assignment)
      numericSatisfies bounded
  have internalEqualsSemantic :
      assignment (internalOutputColumn frame lane) =
        Numeric.semanticLane frame assignment bounded := by
    rw [internalOutputColumn_exact frame lane laneLt]
    calc
      assignment
          (columnMap frame
            ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
              ⟨lane, by
                simp only [outputWidth] at laneLt
                simp only [
                  Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width
                ]
                omega⟩)) =
          residue
            (Numeric.pulled frame assignment
              ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
                ⟨lane, by
                  simp only [outputWidth] at laneLt
                  simp only [
                    Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width
                  ]
                  omega⟩)) := by
        simp [Numeric.pulled, numericAssignment]
      _ = Numeric.semanticLane frame assignment bounded := by
        rw [numericSound]
        rfl
  exact
    (gateRows_active_output_eq frame assignment activeOne split.2 lane laneLt).symm.trans
      internalEqualsSemantic

/-! ## Honest temporary-only completion -/

namespace Honest

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership

def numericAssignment
    (frame : Frame) (assignment : ColumnId → Field) : Nat → Nat :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest.assignment
    Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
    (Numeric.input frame assignment)

def temporaryValues
    (frame : Frame) (assignment : ColumnId → Field) : List Field :=
  List.ofFn fun position : TemporaryPosition =>
    residue
      (numericAssignment frame assignment
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position))

@[simp] theorem temporaryValues_length
    (frame : Frame) (assignment : ColumnId → Field) :
    (temporaryValues frame assignment).length = temporaryWidth := by
  change (List.ofFn (fun position : Fin 2464 =>
    residue
      (numericAssignment frame assignment
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
          position)))).length = 2464
  exact List.length_ofFn

def complete
    (frame : Frame) (assignment : ColumnId → Field) : ColumnId → Field :=
  writeColumns assignment frame.temporaries.ids
    (temporaryValues frame assignment)

theorem complete_changesOnly
    (frame : Frame) (assignment : ColumnId → Field) :
    ChangesOnly frame.temporaries.ids assignment (complete frame assignment) :=
  writeColumns_changesOnly assignment frame.temporaries.ids
    (temporaryValues frame assignment)

theorem complete_agrees_visible
    (frame : Frame) (assignment : ColumnId → Field) :
    AgreesOn frame.visibleIds assignment (complete frame assignment) :=
  writeColumns_agreesOn assignment frame.temporaries.ids frame.visibleIds
    (temporaryValues frame assignment) frame.temporariesDisjointVisible

private theorem temporaryValues_getD
    (frame : Frame) (assignment : ColumnId → Field)
    (index : Nat) (indexLt : index < temporaryWidth) (fallback : Field) :
    (temporaryValues frame assignment).getD index fallback =
      residue
        (numericAssignment frame assignment
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
            ⟨index, by
            simpa [temporaryWidth, temporaries] using indexLt⟩)) := by
  have valuesLt : index < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    exact indexLt
  rw [← List.getElem_eq_getD
    (l := temporaryValues frame assignment) (i := index)
    (h := valuesLt) fallback]
  simp [temporaryValues]

theorem complete_temporary
    (frame : Frame) (assignment : ColumnId → Field)
    (position : TemporaryPosition) :
    complete frame assignment (temporaryColumn frame position.val) =
      residue
        (numericAssignment frame assignment
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
            position)) := by
  have recovered :
      frame.temporaries.ids.map (complete frame assignment) =
        temporaryValues frame assignment := by
    apply writeColumns_map_eq
    · rw [Frame.temporary_ids_length, temporaryValues_length]
    · exact frame.temporary_ids_nodup
  have idsLt : position.val < frame.temporaries.ids.length := by
    rw [Frame.temporary_ids_length]
    simpa [temporaryWidth, temporaries] using position.isLt
  have valuesLt :
      position.val < (temporaryValues frame assignment).length := by
    rw [temporaryValues_length]
    simpa [temporaryWidth, temporaries] using position.isLt
  have atIndex := congrArg
    (fun values : List Field =>
      values.getD position.val (complete frame assignment frame.one))
    recovered
  change
    (frame.temporaries.ids.map (complete frame assignment)).getD
          position.val (complete frame assignment frame.one) =
        (temporaryValues frame assignment).getD
          position.val (complete frame assignment frame.one)
    at atIndex
  rw [← List.getElem_eq_getD
      (l := frame.temporaries.ids.map (complete frame assignment))
      (i := position.val) (h := by simpa using idsLt)
      (complete frame assignment frame.one),
    ← List.getElem_eq_getD
      (l := temporaryValues frame assignment)
      (i := position.val) (h := valuesLt)
      (complete frame assignment frame.one)] at atIndex
  simp only [List.getElem_map] at atIndex
  rw [List.getElem_eq_getD
      (l := frame.temporaries.ids) (i := position.val)
      (h := idsLt) frame.one,
    List.getElem_eq_getD
      (l := temporaryValues frame assignment) (i := position.val)
      (h := valuesLt) (complete frame assignment frame.one)] at atIndex
  rw [temporaryValues_getD frame assignment position.val
    (by simpa [temporaryWidth, temporaries] using position.isLt)] at atIndex
  exact atIndex

private theorem inputColumn_mem
    (frame : Frame) (index : Nat) (indexLt : index < inputWidth) :
    inputColumn frame index ∈ frame.input.ids := by
  have idsLt : index < frame.input.ids.length := by
    rw [Frame.input_ids_length]
    exact indexLt
  unfold inputColumn
  rw [← List.getElem_eq_getD
    (l := frame.input.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem residue_val_of_lt
    (value : Nat)
    (valueLt : value < Nightstream.Implementation.R1CS.goldilocksP) :
    (residue value).val = value := by
  change
    value % Nightstream.SuperNeo.Concrete.goldilocksModulus = value
  apply Nat.mod_eq_of_lt
  simpa [
    Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus
  ] using valueLt

theorem numericAssignment_residues
    (frame : Frame) (assignment : ColumnId → Field) :
    ∀ source, numericAssignment frame assignment source <
      Nightstream.Implementation.R1CS.goldilocksP := by
  apply
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest.assignment_residues
  intro index
  exact (assignment (inputColumn frame index.val)).isLt

theorem completedPulled_eq_numeric_of_allocated
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (source : Nat)
    (allocated :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.Allocated
        source) :
    Numeric.pulled frame (complete frame assignment) source =
      numericAssignment frame assignment source := by
  rcases allocated with rfl | inputMember | temporaryMember
  · change (complete frame assignment frame.one).val =
      numericAssignment frame assignment 0
    have oneVisible : frame.one ∈ frame.visibleIds := by
      simp [Frame.visibleIds]
    rw [complete_agrees_visible frame assignment frame.one oneVisible,
      constantOne]
    rfl
  · rcases List.mem_ofFn.mp inputMember with ⟨index, rfl⟩
    have indexLt : index.val < inputWidth := by
      have := index.isLt
      change index.val < 23 at this
      exact this
    change
      (complete frame assignment
        (columnMap frame
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn
            index.val))).val =
        numericAssignment frame assignment
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn
            index.val)
    rw [show
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn
            index.val = 2527 + index.val by
          simp only [
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn,
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq],
      columnMap_input frame index.val indexLt]
    have inputVisible :
        inputColumn frame index.val ∈ frame.visibleIds := by
      simp [Frame.visibleIds, inputColumn_mem frame index.val indexLt]
    rw [complete_agrees_visible frame assignment
      (inputColumn frame index.val) inputVisible]
    symm
    simpa only [
      numericAssignment, Numeric.input,
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn,
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq
    ] using
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest.assignment_input
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (Numeric.input frame assignment) index
  · rcases List.mem_ofFn.mp temporaryMember with ⟨position, rfl⟩
    change
      (complete frame assignment
        (columnMap frame
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
            position))).val =
        numericAssignment frame assignment
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaryColumn
            position)
    rw [columnMap_sourceTemporary frame position,
      complete_temporary frame assignment position,
      residue_val_of_lt _ (numericAssignment_residues frame assignment _)]

end Honest

private theorem numericLcEval_congr
    {left right : Nat → Nat} (terms : List (Nat × Nat))
    (agreement : ∀ term, term ∈ terms → left term.1 = right term.1) :
    Nightstream.Implementation.R1CS.lcEval left terms =
      Nightstream.Implementation.R1CS.lcEval right terms := by
  unfold Nightstream.Implementation.R1CS.lcEval
  have foldAgreement : ∀ initial,
      terms.foldl (fun acc term => acc + term.2 * left term.1) initial =
        terms.foldl (fun acc term => acc + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head (by simp)]
        exact inductionHypothesis
          (fun term member => agreement term (by simp [member])) _
  rw [foldAgreement 0]

private theorem numericRowHolds_of_agreement
    (row : Nightstream.Implementation.R1CS.Row)
    (left right : Nat → Nat)
    (agreement :
      ∀ column,
        Nightstream.Implementation.R1CS.Canonical.LinCombNormal.Mentions
            row.a column ∨
          Nightstream.Implementation.R1CS.Canonical.LinCombNormal.Mentions
            row.b column ∨
          Nightstream.Implementation.R1CS.Canonical.LinCombNormal.Mentions
            row.c column →
        left column = right column)
    (holds : Nightstream.Implementation.R1CS.RowHolds right row) :
    Nightstream.Implementation.R1CS.RowHolds left row := by
  unfold Nightstream.Implementation.R1CS.RowHolds at holds ⊢
  rw [numericLcEval_congr row.a (by
      intro term member
      exact agreement term.1
        (Or.inl (List.mem_map.mpr ⟨term, member, rfl⟩))),
    numericLcEval_congr row.b (by
      intro term member
      exact agreement term.1
        (Or.inr (Or.inl (List.mem_map.mpr ⟨term, member, rfl⟩)))),
    numericLcEval_congr row.c (by
      intro term member
      exact agreement term.1
        (Or.inr (Or.inr (List.mem_map.mpr ⟨term, member, rfl⟩))))]
  exact holds

theorem core_complete
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1) :
    Satisfies (coreRows frame) (Honest.complete frame assignment) := by
  apply
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows
      (Honest.complete frame assignment)).2
  intro row member
  apply numericRowHolds_of_agreement row
    (Numeric.pulled frame (Honest.complete frame assignment))
    (Honest.numericAssignment frame assignment)
  · intro column mentioned
    exact Honest.completedPulled_eq_numeric_of_allocated
      frame assignment constantOne column
      (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.program_conservation
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        row member column mentioned)
  · exact
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest.honest_satisfies
        Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
        (Numeric.input frame assignment)
      (fun index => (assignment (inputColumn frame index.val)).isLt)
        row member

private theorem outputColumn_mem
    (frame : Frame) (lane : Nat) (laneLt : lane < outputWidth) :
    outputColumn frame lane ∈ frame.output.ids := by
  have idsLt : lane < frame.output.ids.length := by
    rw [Frame.output_ids_length]
    exact laneLt
  unfold outputColumn
  rw [← List.getElem_eq_getD
    (l := frame.output.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

theorem Honest.input_complete_eq
    (frame : Frame) (assignment : ColumnId → Field) :
    Numeric.input frame (Honest.complete frame assignment) =
      Numeric.input frame assignment := by
  funext index
  change
    (Honest.complete frame assignment
      (inputColumn frame index.val)).val =
        (assignment (inputColumn frame index.val)).val
  apply congrArg Fin.val
  apply Honest.complete_agrees_visible frame assignment
  have indexLt : index.val < inputWidth := by
    have := index.isLt
    change index.val < 23 at this
    exact this
  simp [Frame.visibleIds, Honest.inputColumn_mem frame index.val indexLt]

private theorem completed_internalOutput_eq_semantic
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (coreHolds :
      Satisfies (coreRows frame) (Honest.complete frame assignment))
    (lane : Nat) (laneLt : lane < outputWidth) :
    Honest.complete frame assignment (internalOutputColumn frame lane) =
      Numeric.semanticLane frame assignment
        ⟨lane, by simpa [outputWidth,
          Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
          using laneLt⟩ := by
  have oneVisible : frame.one ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedOne :
      Honest.complete frame assignment frame.one = 1 :=
    (Honest.complete_agrees_visible frame assignment frame.one oneVisible).trans
      constantOne
  have numericSatisfies :
      Nightstream.Implementation.R1CS.Satisfies Canonical.rows
        (Numeric.pulled frame (Honest.complete frame assignment)) :=
    (ownedRowsFrom_satisfies_iff frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows
      (Honest.complete frame assignment)).1 coreHolds
  let bounded :
      Fin Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength :=
    ⟨lane, by
      simpa [outputWidth,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
        using laneLt⟩
  have numericSound :=
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.program_computes_digest
      Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
      (Numeric.pulled frame (Honest.complete frame assignment))
      (Numeric.input frame (Honest.complete frame assignment))
      (Numeric.pulled_residues frame (Honest.complete frame assignment))
      (Numeric.pulled_constant frame (Honest.complete frame assignment)
        completedOne)
      (Numeric.pulled_inputsAgree frame
        (Honest.complete frame assignment))
      numericSatisfies bounded
  rw [internalOutputColumn_exact frame lane laneLt]
  calc
    Honest.complete frame assignment
          (columnMap frame
            ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
              ⟨lane, by
                simp only [outputWidth] at laneLt
                simp only [
                  Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width
                ]
                omega⟩)) =
        residue
          (Numeric.pulled frame (Honest.complete frame assignment)
            ((Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.layout.call 6).outputPort
              ⟨lane, by
                simp only [outputWidth] at laneLt
                simp only [
                  Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width
                ]
                omega⟩)) := by
      simp [Numeric.pulled, numericAssignment]
    _ = residue
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
            Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
            (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.dataChunks
              (Numeric.input frame (Honest.complete frame assignment)))
            bounded) := by rw [numericSound]
    _ = Numeric.semanticLane frame assignment bounded := by
      rw [Honest.input_complete_eq frame assignment]
      rfl

theorem gateRows_complete_of_active
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane : Fin outputWidth,
        assignment (outputColumn frame lane.val) =
          Numeric.semanticLane frame assignment
            ⟨lane.val, by
              have laneLt := lane.isLt
              simpa [outputWidth,
                Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
                using laneLt⟩)
    (coreHolds :
      Satisfies (coreRows frame) (Honest.complete frame assignment)) :
    Satisfies (gateRows frame) (Honest.complete frame assignment) := by
  apply
    (satisfies_iff_forall (gateRows frame)
      (Honest.complete frame assignment)).2
  intro owned member
  rcases List.mem_map.mp member with ⟨lane, laneMember, rfl⟩
  have laneLt : lane < outputWidth := by
    have := List.mem_range.mp laneMember
    simpa [gateRowCount, outputWidth] using this
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      Honest.complete frame assignment frame.active = 1 :=
    (Honest.complete_agrees_visible frame assignment frame.active
      activeVisible).trans activeOne
  apply
    (gateRow_active_iff frame (Honest.complete frame assignment)
      completedActive lane).2
  have outputVisible :
      outputColumn frame lane ∈ frame.visibleIds := by
    simp [Frame.visibleIds, outputColumn_mem frame lane laneLt]
  calc
    Honest.complete frame assignment (internalOutputColumn frame lane) =
        Numeric.semanticLane frame assignment
          ⟨lane, by simpa [outputWidth,
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
            using laneLt⟩ :=
      completed_internalOutput_eq_semantic
        frame assignment constantOne coreHolds lane laneLt
    _ = assignment (outputColumn frame lane) :=
      (outputsCorrect ⟨lane, laneLt⟩).symm
    _ = Honest.complete frame assignment (outputColumn frame lane) :=
      (Honest.complete_agrees_visible frame assignment
        (outputColumn frame lane) outputVisible).symm

/-- Honest active values extend by changing exactly the receipt temporaries. -/
theorem active_complete
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      ∀ lane : Fin outputWidth,
        assignment (outputColumn frame lane.val) =
          Numeric.semanticLane frame assignment
            ⟨lane.val, by
              have laneLt := lane.isLt
              simpa [outputWidth,
                Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digestLength]
                using laneLt⟩) :
    Satisfies (rows frame) (Honest.complete frame assignment) := by
  have core := core_complete frame assignment constantOne
  exact
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (Honest.complete frame assignment)).2
      ⟨core,
        gateRows_complete_of_active frame assignment constantOne
          activeOne outputsCorrect core⟩

/-- Inactive occurrences retain the deterministic core and vacuous gates. -/
theorem inactive_complete
    (frame : Frame) (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0) :
    Satisfies (rows frame) (Honest.complete frame assignment) := by
  have activeVisible : frame.active ∈ frame.visibleIds := by
    simp [Frame.visibleIds]
  have completedActive :
      Honest.complete frame assignment frame.active = 0 :=
    (Honest.complete_agrees_visible frame assignment frame.active
      activeVisible).trans activeZero
  exact
    (satisfies_append_iff (coreRows frame) (gateRows frame)
      (Honest.complete frame assignment)).2
      ⟨core_complete frame assignment constantOne,
        gateRows_complete_of_inactive frame
          (Honest.complete frame assignment) completedActive⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe
