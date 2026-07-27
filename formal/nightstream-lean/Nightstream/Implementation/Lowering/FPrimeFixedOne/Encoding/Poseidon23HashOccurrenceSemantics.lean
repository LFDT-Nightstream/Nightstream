import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrence

/-!
Contract: exact active semantics of one total fixed-23 binding-hash
occurrence.

Owns: normalization, the selected preimage copy, complete alignment equality,
the optional-result branch, and transport through the canonical Poseidon2
core.

Does not own: typed call operands, application serialization, honest
completion, Rust, generated rows, or collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashOccurrence

private theorem rawSatisfies_member
    {rows : List Row}
    {assignment : ColumnId -> Field}
    (holds : RawSatisfies rows assignment)
    {row : Row}
    (member : row ∈ rows) :
    row.Holds assignment := by
  induction rows with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact holds.1
      · exact inductionHypothesis holds.2 tailMember

private theorem rawSatisfies_mono
    {large small : List Row}
    {assignment : ColumnId -> Field}
    (subset : ∀ row, row ∈ small -> row ∈ large)
    (holds : RawSatisfies large assignment) :
    RawSatisfies small assignment := by
  induction small with
  | nil =>
      trivial
  | cons head tail inductionHypothesis =>
      exact
        ⟨rawSatisfies_member holds (subset head (by simp)),
          inductionHypothesis
            (fun row member => subset row (by simp [member]))⟩

private theorem satisfies_raw_map_iff
    (rows : List OwnedRow)
    (assignment : ColumnId -> Field) :
    Satisfies rows assignment ↔
      RawSatisfies (rows.map fun row => row.row) assignment := by
  induction rows with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simpa only [List.map_cons, satisfies_cons, rawSatisfies_cons,
        inductionHypothesis]

def sourceValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  frame.source.map fun column => assignment column.id

def outputValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  frame.output.map fun column => assignment column.id

@[simp] theorem sourceValues_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) :
    (sourceValues frame assignment).length = sourceWidth := by
  simp [sourceValues, frame.source_length]

@[simp] theorem outputValues_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) :
    (outputValues frame assignment).length = 5 := by
  simp [outputValues, frame.outputLength]

theorem outputValues_getD
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (index : Nat)
    (indexLt : index < 5) :
    (outputValues frame assignment).getD index 0 =
      assignment (frame.outputAt ⟨index, indexLt⟩).id := by
  have valuesLt : index < (outputValues frame assignment).length := by
    rw [outputValues_length]
    exact indexLt
  rw [← List.getElem_eq_getD
    (l := outputValues frame assignment) (i := index)
    (h := valuesLt) 0]
  simp [outputValues, Frame.outputAt]

theorem coreOutputValues_getD
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4) :
    (frame.coreOutput.values assignment).getD lane.val 0 =
      assignment (frame.coreOutputAt lane).id := by
  have valuesLt :
      lane.val < (frame.coreOutput.values assignment).length := by
    rw [ColumnBundle.values_length]
    simpa [auxiliary, auxiliaryLayout, ownedLayout] using lane.isLt
  rw [← List.getElem_eq_getD
    (l := frame.coreOutput.values assignment) (i := lane.val)
    (h := valuesLt) 0]
  simp [ColumnBundle.values, Frame.coreOutputAt]

theorem sourceValues_get
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (index : Fin sourceWidth) :
    (sourceValues frame assignment).get
        ⟨index.val, by
          rw [sourceValues_length]
          exact index.isLt⟩ =
      assignment (frame.sourceAt index).id := by
  simp [sourceValues, Frame.sourceAt]

theorem projected_values
    {sourceWidth alignmentWidth targetWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (projection : Fin targetWidth -> Fin sourceWidth) :
    (frame.projected projection).map
        (fun column => assignment column.id) =
      Poseidon23Hash.select (sourceValues frame assignment) projection := by
  unfold Frame.projected Poseidon23Hash.select
  rw [List.map_ofFn]
  congr 1
  funext index
  have coordinate :=
    sourceValues_get frame assignment (projection index)
  rw [List.get_eq_getElem, List.getElem_eq_getD
      (l := sourceValues frame assignment)
      (i := (projection index).val)
      (h := by
        rw [sourceValues_length]
        exact (projection index).isLt) 0] at coordinate
  exact coordinate.symm

private theorem singleton_eval
    (assignment : ColumnId -> Field)
    (column : ColumnId) :
    (Goldilocks.singleton column 1).eval assignment =
      assignment column := by
  simp only [Goldilocks.singleton, Goldilocks.LinearCombination.eval,
    Fin.one_mul, Fin.add_zero]

private theorem difference_eval
    (assignment : ColumnId -> Field)
    (left right : ColumnId) :
    (difference left right).eval assignment =
      assignment left - assignment right := by
  simp only [difference, Goldilocks.LinearCombination.eval, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem normalizationRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (holds : (normalizationRow frame).Holds assignment) :
    assignment frame.normalizedColumn.id =
      if frame.next then assignment frame.iteration.id + 1
      else assignment frame.iteration.id := by
  cases nextValue : frame.next
  · have equation :
        assignment frame.iteration.id -
            assignment frame.normalizedColumn.id = 0 := by
      simpa [normalizationRow, nextValue, Row.Holds,
        singleton_eval, difference_eval,
        Goldilocks.LinearCombination.eval_nil, constantOne,
        Fin.one_mul] using holds
    exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp equation).symm
  · have equation :
        assignment frame.iteration.id + 1 -
            assignment frame.normalizedColumn.id = 0 := by
      simpa [normalizationRow, nextValue, Row.Holds,
        singleton_eval, Goldilocks.LinearCombination.eval,
        Goldilocks.LinearCombination.eval_nil, constantOne, Fin.one_mul,
        Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg,
        Lean.Grind.Fin.add_assoc] using holds
    exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp equation).symm

private theorem preimageRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (index : Fin 23)
    (holds : (preimageRow frame index).Holds assignment) :
    assignment
        (frame.preimage.columns.get
          ⟨index.val, by
            rw [frame.preimage.length_eq]
            simpa [auxiliary, auxiliaryLayout, ownedLayout]
              using index.isLt⟩).id =
      assignment (frame.sourceAt (frame.plan.preimage index)).id := by
  simp only [preimageRow, Row.Holds, singleton_eval, difference_eval,
    Goldilocks.LinearCombination.eval_nil, constantOne, Fin.one_mul]
    at holds
  exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp holds).symm

private theorem selectedRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (holds : (selectedRow frame).Holds assignment) :
    assignment frame.selectedColumn.id =
      assignment frame.equalityOutputColumn.id := by
  simp only [selectedRow, Row.Holds, singleton_eval, activeOne,
    Fin.one_mul] at holds
  exact holds.symm

private theorem tagRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (holds : (tagRow frame).Holds assignment) :
    assignment (frame.outputAt 0).id =
      assignment frame.selectedColumn.id := by
  simp only [tagRow, Row.Holds, singleton_eval, difference_eval,
    Goldilocks.LinearCombination.eval_nil, activeOne, Fin.one_mul]
    at holds
  exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp holds).symm

private theorem payloadSuccessRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4)
    (selectedOne : assignment frame.selectedColumn.id = 1)
    (holds : (payloadSuccessRow frame lane).Holds assignment) :
    assignment
        (frame.outputAt ⟨lane.val + 1, by omega⟩).id =
      assignment (frame.coreOutputAt lane).id := by
  simp only [payloadSuccessRow, Row.Holds, singleton_eval,
    difference_eval, Goldilocks.LinearCombination.eval_nil, selectedOne,
    Fin.one_mul] at holds
  exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp holds).symm

private theorem payloadAbsentRow_value
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (lane : Fin 4)
    (selectedZero : assignment frame.selectedColumn.id = 0)
    (holds : (payloadAbsentRow frame lane).Holds assignment) :
    assignment
        (frame.outputAt ⟨lane.val + 1, by omega⟩).id = 0 := by
  simp only [payloadAbsentRow, Row.Holds, singleton_eval,
    difference_eval, Goldilocks.LinearCombination.eval_nil, activeOne,
    selectedZero, Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
    Fin.add_zero, Fin.one_mul] at holds
  exact holds

private theorem preimage_values
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (holds : RawSatisfies (preimageRows frame) assignment) :
    frame.preimage.values assignment =
      Poseidon23Hash.select (sourceValues frame assignment)
        frame.plan.preimage := by
  rw [← projected_values frame assignment frame.plan.preimage]
  unfold Frame.projected
  rw [List.map_ofFn]
  apply List.ext_get
  · rw [ColumnBundle.values_length]
    simp [auxiliary, auxiliaryLayout, ownedLayout]
  · intro index leftLt rightLt
    have indexLt : index < 23 := by
      rw [ColumnBundle.values_length] at leftLt
      simpa [auxiliary, auxiliaryLayout, ownedLayout] using leftLt
    let bounded : Fin 23 := ⟨index, indexLt⟩
    have rowHolds :
        (preimageRow frame bounded).Holds assignment :=
      rawSatisfies_member holds
        (by
          unfold preimageRows
          exact List.mem_ofFn.mpr ⟨bounded, rfl⟩)
    simp only [ColumnBundle.values, List.get_eq_getElem,
      List.getElem_map]
    have columnsLt : index < frame.preimage.columns.length := by
      rw [frame.preimage.length_eq]
      simpa [auxiliary, auxiliaryLayout, ownedLayout] using indexLt
    have copied :=
      preimageRow_value frame assignment constantOne bounded rowHolds
    simpa only [List.getElem_ofFn, Function.comp_apply, bounded] using
      copied

theorem core_semanticLane
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (lane : Fin 4) :
    CanonicalPoseidon2Sponge23Recipe.Numeric.semanticLane
        (core frame facts) assignment lane =
      residue
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.dataChunks
            (fun index =>
              ((frame.preimage.values assignment).getD
                index.val 0).val))
          lane) := by
  unfold CanonicalPoseidon2Sponge23Recipe.Numeric.semanticLane
  congr 3
  funext index
  unfold CanonicalPoseidon2Sponge23Recipe.Numeric.input
    CanonicalPoseidon2Sponge23Recipe.inputColumn
  change
    (assignment
      ((core frame facts).input.ids.getD index.val
        (core frame facts).one)).val =
      ((frame.preimage.values assignment).getD index.val 0).val
  have idsLt :
      index.val < frame.preimage.ids.length := by
    rw [ColumnBundle.ids, List.length_map, frame.preimage.length_eq]
    simpa [auxiliary, auxiliaryLayout, ownedLayout,
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23Fields]
      using index.isLt
  have valuesLt :
      index.val < (frame.preimage.values assignment).length := by
    rw [ColumnBundle.values_length]
    simpa [auxiliary, auxiliaryLayout, ownedLayout,
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23Fields]
      using index.isLt
  have fieldExact :
      assignment
          ((core frame facts).input.ids.getD index.val
            (core frame facts).one) =
        (frame.preimage.values assignment).getD index.val 0 := by
    change
      assignment (frame.preimage.ids.getD index.val frame.one) =
        (frame.preimage.values assignment).getD index.val 0
    rw [← List.getElem_eq_getD
        (l := frame.preimage.ids) (i := index.val)
        (h := idsLt) frame.one,
      ← List.getElem_eq_getD
        (l := frame.preimage.values assignment) (i := index.val)
        (h := valuesLt) 0]
    simp [ColumnBundle.ids, ColumnBundle.values]
  exact congrArg Fin.val fieldExact

theorem digestCoordinates_getD
    (preimage : List Field)
    (lane : Fin 4) :
    (Poseidon23Hash.digestCoordinates preimage).getD lane.val 0 =
      residue
        (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
          Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
          (Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.dataChunks
            (fun index => ((preimage.getD index.val 0).val)))
          lane) := by
  have laneLt :
      lane.val < (Poseidon23Hash.digestCoordinates preimage).length := by
    rw [Poseidon23Hash.digestCoordinates_length]
    exact lane.isLt
  rw [← List.getElem_eq_getD
    (l := Poseidon23Hash.digestCoordinates preimage)
    (i := lane.val) (h := laneLt) 0]
  have laneCases :
      lane.val = 0 ∨ lane.val = 1 ∨ lane.val = 2 ∨ lane.val = 3 := by
    omega
  rcases laneCases with first | second | third | fourth
  · have exactLane : lane = (0 : Fin 4) := Fin.ext first
    subst lane
    rfl
  · have exactLane : lane = (1 : Fin 4) := Fin.ext second
    subst lane
    rfl
  · have exactLane : lane = (2 : Fin 4) := Fin.ext third
    subst lane
    rfl
  · have exactLane : lane = (3 : Fin 4) := Fin.ext fourth
    subst lane
    rfl

theorem core_outputColumn
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (lane : Fin 4) :
    CanonicalPoseidon2Sponge23Recipe.outputColumn
        (core frame facts) lane.val =
      (frame.coreOutputAt lane).id := by
  unfold CanonicalPoseidon2Sponge23Recipe.outputColumn
    Frame.coreOutputAt
  change
    (frame.coreOutput.ids.getD lane.val frame.one) =
      (frame.coreOutput.columns.get ⟨lane.val, _⟩).id
  have idsLt : lane.val < frame.coreOutput.ids.length := by
    rw [ColumnBundle.ids, List.length_map, frame.coreOutput.length_eq]
    simpa [auxiliary, auxiliaryLayout, ownedLayout] using lane.isLt
  rw [← List.getElem_eq_getD
    (l := frame.coreOutput.ids) (i := lane.val)
    (h := idsLt) frame.one]
  simp [ColumnBundle.ids]

/-- Active satisfaction enforces the complete total optional digest and
normalizes the iteration coordinate in transcript order. -/
theorem active_sound
    {sourceWidth alignmentWidth : Nat}
    (laws : FieldLaws)
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (holds : Satisfies (rows frame facts) assignment) :
    assignment frame.normalizedColumn.id =
        (if frame.next then assignment frame.iteration.id + 1
          else assignment frame.iteration.id) ∧
      outputValues frame assignment =
        Poseidon23Hash.resultCoordinates frame.plan
          (sourceValues frame assignment) := by
  have raw :
      RawSatisfies (rawRows frame facts) assignment :=
    (satisfies_ownRows_iff frame.owner (rawRows frame facts)
      assignment).mp holds
  have normalizationHolds :
      (normalizationRow frame).Holds assignment :=
    rawSatisfies_member raw (by simp [rawRows, wrapperRawRows])
  have preimageHolds :
      RawSatisfies (preimageRows frame) assignment :=
    rawSatisfies_mono (by
      intro row member
      simp [rawRows, wrapperRawRows, member]) raw
  have equalityHolds :
      Satisfies frame.equality.rows assignment := by
    apply
      (satisfies_ownRows_iff frame.owner frame.equality.rawRows
        assignment).mpr
    exact rawSatisfies_mono (by
      intro row member
      simp [rawRows, wrapperRawRows, member]) raw
  have selectedHolds :
      (selectedRow frame).Holds assignment :=
    rawSatisfies_member raw (by simp [rawRows, wrapperRawRows])
  have tagHolds :
      (tagRow frame).Holds assignment :=
    rawSatisfies_member raw (by simp [rawRows, wrapperRawRows])
  have payloadHolds :
      RawSatisfies (payloadRows frame) assignment :=
    rawSatisfies_mono (by
      intro row member
      simp [rawRows, wrapperRawRows, member]) raw
  have coreHolds :
      Satisfies
        (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
        assignment := by
    apply
      (satisfies_raw_map_iff
        (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
        assignment).mpr
    exact rawSatisfies_mono (by
      intro row member
      simp [rawRows, member]) raw
  have preimageExact :=
    preimage_values frame assignment constantOne preimageHolds
  have alignmentLeft :=
    projected_values frame assignment frame.plan.alignmentLeft
  have alignmentRight :=
    projected_values frame assignment frame.plan.alignmentRight
  have equalityOutput :=
    frame.equality.active_sound laws assignment constantOne activeOne
      equalityHolds
  have equalityOutputExact :
      assignment frame.equalityOutputColumn.id =
        if
          frame.equality.left.map
              (fun column => assignment column.id) =
            frame.equality.right.map
              (fun column => assignment column.id)
        then 1 else 0 := by
    simpa [Frame.equality] using equalityOutput
  have selectedExact :=
    selectedRow_value frame assignment activeOne selectedHolds
  have tagExact :=
    tagRow_value frame assignment activeOne tagHolds
  constructor
  · exact normalizationRow_value frame assignment constantOne
      normalizationHolds
  · unfold Poseidon23Hash.resultCoordinates
    by_cases aligned :
        Poseidon23Hash.select (sourceValues frame assignment)
            frame.plan.alignmentLeft =
          Poseidon23Hash.select (sourceValues frame assignment)
            frame.plan.alignmentRight
    · rw [if_pos aligned]
      have equalityAligned :
          frame.equality.left.map (fun column => assignment column.id) =
            frame.equality.right.map
              (fun column => assignment column.id) := by
        simpa [Frame.equality] using
          alignmentLeft.trans (aligned.trans alignmentRight.symm)
      have equalityOne :
          assignment frame.equalityOutputColumn.id = 1 := by
        rw [equalityOutputExact]
        simp [equalityAligned]
      have selectedOne :
          assignment frame.selectedColumn.id = 1 := by
        rw [selectedExact, equalityOne]
      have tagOne :
          assignment (frame.outputAt 0).id = 1 := by
        rw [tagExact, selectedOne]
      apply List.ext_get
      · simp [outputValues_length]
      · intro index leftLt rightLt
        have indexLt : index < 5 := by
          simpa [outputValues_length] using leftLt
        cases index with
        | zero =>
            simp only [List.get_eq_getElem]
            rw [List.getElem_eq_getD
                (l := outputValues frame assignment)
                (i := 0) (h := leftLt) 0,
              List.getElem_eq_getD
                (l := 1 ::
                  Poseidon23Hash.digestCoordinates
                    (Poseidon23Hash.select
                      (sourceValues frame assignment)
                      frame.plan.preimage))
                (i := 0) (h := rightLt) 0,
              outputValues_getD frame assignment 0 (by decide)]
            simpa using tagOne
        | succ lane =>
            have laneLt : lane < 4 := by omega
            let bounded : Fin 4 := ⟨lane, laneLt⟩
            have successHolds :
                (payloadSuccessRow frame bounded).Holds assignment :=
              rawSatisfies_member payloadHolds (by
                unfold payloadRows
                exact List.mem_append_left _
                  (List.mem_ofFn.mpr ⟨bounded, rfl⟩))
            have payloadExact :=
              payloadSuccessRow_value frame assignment bounded
                selectedOne successHolds
            have coreCoordinate :
                assignment (frame.coreOutputAt bounded).id =
                  (Poseidon23Hash.digestCoordinates
                    (Poseidon23Hash.select
                      (sourceValues frame assignment)
                      frame.plan.preimage)).getD bounded.val 0 := by
              calc
                assignment (frame.coreOutputAt bounded).id =
                    CanonicalPoseidon2Sponge23Recipe.Numeric.semanticLane
                      (core frame facts) assignment bounded := by
                  rw [← core_outputColumn frame facts bounded]
                  exact
                    CanonicalPoseidon2Sponge23Recipe.active_sound
                      (core frame facts) assignment constantOne selectedOne
                      coreHolds bounded.val bounded.isLt
                _ = (Poseidon23Hash.digestCoordinates
                      (frame.preimage.values assignment)).getD
                        bounded.val 0 := by
                  rw [core_semanticLane frame facts assignment bounded,
                    ← digestCoordinates_getD
                      (frame.preimage.values assignment) bounded]
                _ = (Poseidon23Hash.digestCoordinates
                      (Poseidon23Hash.select
                        (sourceValues frame assignment)
                        frame.plan.preimage)).getD bounded.val 0 :=
                  congrArg
                    (fun values =>
                      (Poseidon23Hash.digestCoordinates values).getD
                        bounded.val 0)
                    preimageExact
            simp only [List.get_eq_getElem]
            rw [List.getElem_eq_getD
                (l := outputValues frame assignment)
                (i := lane + 1) (h := leftLt) 0,
              List.getElem_eq_getD
                (l := 1 ::
                  Poseidon23Hash.digestCoordinates
                    (Poseidon23Hash.select
                      (sourceValues frame assignment)
                      frame.plan.preimage))
                (i := lane + 1) (h := rightLt) 0,
              outputValues_getD frame assignment (lane + 1) indexLt,
              payloadExact, coreCoordinate]
            simp [List.getD_eq_getElem?_getD, laneLt, bounded]
    · rw [if_neg aligned]
      have equalityNotAligned :
          ¬frame.equality.left.map
                (fun column => assignment column.id) =
            frame.equality.right.map
              (fun column => assignment column.id) := by
        intro equal
        apply aligned
        have projectedEqual :
            (frame.projected frame.plan.alignmentLeft).map
                (fun column => assignment column.id) =
              (frame.projected frame.plan.alignmentRight).map
                (fun column => assignment column.id) := by
          simpa [Frame.equality] using equal
        exact alignmentLeft.symm.trans
          (projectedEqual.trans alignmentRight)
      have equalityZero :
          assignment frame.equalityOutputColumn.id = 0 := by
        rw [equalityOutputExact]
        simp [equalityNotAligned]
      have selectedZero :
          assignment frame.selectedColumn.id = 0 := by
        rw [selectedExact, equalityZero]
      have tagZero :
          assignment (frame.outputAt 0).id = 0 := by
        rw [tagExact, selectedZero]
      apply List.ext_get
      · simp [outputValues_length]
      · intro index leftLt rightLt
        have indexLt : index < 5 := by
          simpa [outputValues_length] using leftLt
        cases index with
        | zero =>
            simp only [List.get_eq_getElem]
            rw [List.getElem_eq_getD
                (l := outputValues frame assignment)
                (i := 0) (h := leftLt) 0,
              List.getElem_eq_getD
                (l := [0, 0, 0, 0, 0])
                (i := 0) (h := rightLt) 0,
              outputValues_getD frame assignment 0 (by decide)]
            simpa using tagZero
        | succ lane =>
            have laneLt : lane < 4 := by omega
            let bounded : Fin 4 := ⟨lane, laneLt⟩
            have absentHolds :
                (payloadAbsentRow frame bounded).Holds assignment :=
              rawSatisfies_member payloadHolds (by
                unfold payloadRows
                exact List.mem_append_right _
                  (List.mem_ofFn.mpr ⟨bounded, rfl⟩))
            have payloadZero :=
              payloadAbsentRow_value frame assignment activeOne bounded
                selectedZero absentHolds
            simp only [List.get_eq_getElem]
            rw [List.getElem_eq_getD
                (l := outputValues frame assignment)
                (i := lane + 1) (h := leftLt) 0,
              List.getElem_eq_getD
                (l := [0, 0, 0, 0, 0])
                (i := lane + 1) (h := rightLt) 0,
              outputValues_getD frame assignment (lane + 1) indexLt]
            have laneCases :
                lane = 0 ∨ lane = 1 ∨ lane = 2 ∨ lane = 3 := by
              omega
            rcases laneCases with rfl | rfl | rfl | rfl <;>
              simpa [bounded] using payloadZero

end Poseidon23HashOccurrence

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
