import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrenceHonest

/-!
Contract: total honest completion of one fixed-23 binding-hash occurrence.

Owns: active and inactive wrapper satisfaction, preservation across the
canonical sponge witness, and final satisfaction of every emitted row.

Does not own: typed call decoding, application serialization, Rust,
generated rows, collision resistance, or deployment selection.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashOccurrence

namespace Honest

private theorem rawSatisfies_of_forall
    {rows : List Row}
    {assignment : ColumnId -> Field}
    (holds : ∀ row, row ∈ rows -> row.Holds assignment) :
    RawSatisfies rows assignment := by
  induction rows with
  | nil =>
      trivial
  | cons row tail inductionHypothesis =>
      exact
        ⟨holds row (by simp),
          inductionHypothesis (fun item member =>
            holds item (by simp [member]))⟩

private theorem raw_map_of_satisfies
    {rows : List OwnedRow}
    {assignment : ColumnId -> Field}
    (holds : Satisfies rows assignment) :
    RawSatisfies (rows.map fun row => row.row) assignment := by
  induction rows with
  | nil =>
      trivial
  | cons row tail inductionHypothesis =>
      exact ⟨holds.1, inductionHypothesis holds.2⟩

private theorem selectedRow_holds
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (selectedExact :
      assignment frame.selectedColumn.id =
        assignment frame.active *
          assignment frame.equalityOutputColumn.id) :
    (selectedRow frame).Holds assignment := by
  simp only [selectedRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.LinearCombination.eval, Fin.one_mul, Fin.add_zero]
  exact selectedExact.symm

private theorem tagRow_holds_of_active
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (activeOne : assignment frame.active = 1)
    (outputExact :
      assignment (frame.outputAt 0).id =
        assignment frame.selectedColumn.id) :
    (tagRow frame).Holds assignment := by
  simp only [tagRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeOne, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr outputExact.symm)

private theorem tagRow_holds_of_inactive
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (activeZero : assignment frame.active = 0) :
    (tagRow frame).Holds assignment := by
  simp only [tagRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeZero, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.zero_mul]

private theorem payloadSuccessRow_holds_of_selected
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4)
    (selectedOne : assignment frame.selectedColumn.id = 1)
    (outputExact :
      assignment (frame.outputAt ⟨lane.val + 1, by omega⟩).id =
        assignment (frame.coreOutputAt lane).id) :
    (payloadSuccessRow frame lane).Holds assignment := by
  simp only [payloadSuccessRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    selectedOne, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr outputExact.symm)

private theorem payloadSuccessRow_holds_of_unselected
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4)
    (selectedZero : assignment frame.selectedColumn.id = 0) :
    (payloadSuccessRow frame lane).Holds assignment := by
  simp only [payloadSuccessRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    selectedZero, Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.zero_mul]

private theorem payloadAbsentRow_holds_of_equal_flags
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4)
    (flagsEqual :
      assignment frame.active =
        assignment frame.selectedColumn.id) :
    (payloadAbsentRow frame lane).Holds assignment := by
  simp only [payloadAbsentRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg]
  rw [flagsEqual, Lean.Grind.AddCommGroup.add_neg_cancel, Fin.zero_mul]

private theorem payloadAbsentRow_holds_of_active_unselected
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (lane : Fin 4)
    (activeOne : assignment frame.active = 1)
    (selectedZero : assignment frame.selectedColumn.id = 0)
    (outputZero :
      assignment (frame.outputAt ⟨lane.val + 1, by omega⟩).id = 0) :
    (payloadAbsentRow frame lane).Holds assignment := by
  simp only [payloadAbsentRow, Row.Holds, Goldilocks.singleton,
    Goldilocks.difference, Goldilocks.LinearCombination.eval,
    activeOne, selectedZero, outputZero, Fin.one_mul, Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Lean.Grind.AddCommGroup.neg_zero,
    Fin.zero_mul]

private theorem wrapper_preCore_active
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds)
    (outputsCorrect :
      outputValues frame assignment =
        Poseidon23Hash.resultCoordinates frame.plan
          (Honest.sourceValues frame assignment)) :
    RawSatisfies (wrapperRawRows frame)
      (preCore inverseLaw frame assignment) := by
  let completed := preCore inverseLaw frame assignment
  have oneCompleted : completed frame.one = 1 := by
    change preCore inverseLaw frame assignment frame.one = 1
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.one]
    · exact constantOne
    · simp [Frame.visibleIds]
  have activeCompleted : completed frame.active = 1 := by
    change preCore inverseLaw frame assignment frame.active = 1
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.active]
    · exact activeOne
    · simp [Frame.visibleIds]
  have equalityCoordinates :=
    preCore_equality_values inverseLaw frame assignment prefixNodup
  have scalarCoordinates :=
    preCore_scalar_values inverseLaw frame assignment prefixNodup
  have leftCoordinates :=
    preCore_projected_values inverseLaw frame assignment
      frame.plan.alignmentLeft prefixNodup temporariesDisjointVisible
  have rightCoordinates :=
    preCore_projected_values inverseLaw frame assignment
      frame.plan.alignmentRight prefixNodup temporariesDisjointVisible
  have equalityHolds :
      Satisfies frame.equality.rows completed := by
    apply frame.equality.active_complete inverseLaw completed
      oneCompleted activeCompleted
    · change
        frame.inverses.values
            (preCore inverseLaw frame assignment) =
          coordinateInverseValues inverseLaw
            ((frame.projected frame.plan.alignmentLeft).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
            ((frame.projected frame.plan.alignmentRight).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
      rw [leftCoordinates, rightCoordinates]
      exact equalityCoordinates.1
    · change
        frame.equals.values
            (preCore inverseLaw frame assignment) =
          coordinateEqualValues
            ((frame.projected frame.plan.alignmentLeft).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
            ((frame.projected frame.plan.alignmentRight).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
      rw [leftCoordinates, rightCoordinates]
      exact equalityCoordinates.2.1
    · change
        frame.products.values
            (preCore inverseLaw frame assignment) =
          productValues
            (frame.equals.values
              (preCore inverseLaw frame assignment))
      rw [equalityCoordinates.2.1]
      exact equalityCoordinates.2.2
    · change
        preCore inverseLaw frame assignment
            frame.equalityOutputColumn.id =
          if
            (frame.projected frame.plan.alignmentLeft).map
                (fun column =>
                  preCore inverseLaw frame assignment column.id) =
              (frame.projected frame.plan.alignmentRight).map
                (fun column =>
                  preCore inverseLaw frame assignment column.id)
          then 1 else 0
      rw [leftCoordinates, rightCoordinates]
      exact scalarCoordinates.1
  have equalityRaw :
      RawSatisfies frame.equality.rawRows completed :=
    (satisfies_ownRows_iff frame.owner frame.equality.rawRows
      completed).mp equalityHolds
  have outputExact :
      outputValues frame completed =
        Poseidon23Hash.resultCoordinates frame.plan
          (Honest.sourceValues frame assignment) := by
    rw [preCore_output_values inverseLaw frame assignment
      temporariesDisjointVisible]
    exact outputsCorrect
  have coreExact :=
    preCore_digest inverseLaw frame assignment prefixNodup
  have normalization :=
    preCore_normalization_holds inverseLaw frame assignment constantOne
      prefixNodup temporariesDisjointVisible
  have preimage :=
    preCore_preimage_holds inverseLaw frame assignment constantOne
      prefixNodup temporariesDisjointVisible
  by_cases aligned :
      alignmentLeftValues frame assignment =
        alignmentRightValues frame assignment
  · have equalityCompleted :
        completed frame.equalityOutputColumn.id = 1 := by
      change
        preCore inverseLaw frame assignment
            frame.equalityOutputColumn.id = 1
      rw [scalarCoordinates.1, equalityValue, if_pos aligned]
    have selectedCompleted :
        completed frame.selectedColumn.id = 1 := by
      change
        preCore inverseLaw frame assignment frame.selectedColumn.id = 1
      rw [scalarCoordinates.2, selectedValue, activeOne, equalityValue,
        if_pos aligned, Fin.one_mul]
    have resultExact :
        Poseidon23Hash.resultCoordinates frame.plan
            (Honest.sourceValues frame assignment) =
          1 :: digestValues frame assignment := by
      unfold Poseidon23Hash.resultCoordinates
      rw [if_pos (by
        simpa [alignmentLeftValues, alignmentRightValues] using aligned)]
      rfl
    have outputTag :
        completed (frame.outputAt 0).id = 1 := by
      have atZero := outputValues_getD frame completed 0 (by omega)
      rw [outputExact, resultExact] at atZero
      exact atZero.symm
    have outputPayload :
        ∀ lane : Fin 4,
          completed (frame.outputAt ⟨lane.val + 1, by omega⟩).id =
            completed (frame.coreOutputAt lane).id := by
      intro lane
      have outputCore :
          outputValues frame completed =
            1 :: frame.coreOutput.values completed :=
        outputExact.trans
          (resultExact.trans
            (congrArg (List.cons (1 : Field)) coreExact.symm))
      have atLane :=
        congrArg (fun values => values.getD (lane.val + 1) 0) outputCore
      change
        (outputValues frame completed).getD (lane.val + 1) 0 =
          (1 :: frame.coreOutput.values completed).getD
            (lane.val + 1) 0 at atLane
      rw [outputValues_getD frame completed (lane.val + 1) (by omega),
        List.getD_cons_succ,
        coreOutputValues_getD frame completed lane] at atLane
      exact atLane
    have selectedHolds :
        (selectedRow frame).Holds completed := by
      exact selectedRow_holds frame completed (by
        calc
          completed frame.selectedColumn.id = 1 := selectedCompleted
          _ = 1 * 1 := by rw [Fin.one_mul]
          _ =
              completed frame.active *
                completed frame.equalityOutputColumn.id := by
            rw [activeCompleted, equalityCompleted])
    have tagHolds : (tagRow frame).Holds completed := by
      exact tagRow_holds_of_active frame completed activeCompleted
        (outputTag.trans selectedCompleted.symm)
    have successHolds :
        RawSatisfies (List.ofFn (payloadSuccessRow frame)) completed := by
      apply rawSatisfies_of_forall
      intro row member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      exact payloadSuccessRow_holds_of_selected frame completed lane
        selectedCompleted (outputPayload lane)
    have absentHolds :
        RawSatisfies (List.ofFn (payloadAbsentRow frame)) completed := by
      apply rawSatisfies_of_forall
      intro row member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      exact payloadAbsentRow_holds_of_equal_flags frame completed lane
        (activeCompleted.trans selectedCompleted.symm)
    exact
      (rawSatisfies_append_iff _ _ completed).mpr
        ⟨⟨normalization, trivial⟩,
          (rawSatisfies_append_iff _ _ completed).mpr
            ⟨preimage,
              (rawSatisfies_append_iff _ _ completed).mpr
                ⟨equalityRaw,
                  (rawSatisfies_append_iff _ _ completed).mpr
                    ⟨⟨selectedHolds, ⟨tagHolds, trivial⟩⟩,
                      (rawSatisfies_append_iff _ _ completed).mpr
                        ⟨successHolds, absentHolds⟩⟩⟩⟩⟩
  · have equalityCompleted :
        completed frame.equalityOutputColumn.id = 0 := by
      change
        preCore inverseLaw frame assignment
            frame.equalityOutputColumn.id = 0
      rw [scalarCoordinates.1, equalityValue, if_neg aligned]
    have selectedCompleted :
        completed frame.selectedColumn.id = 0 := by
      change
        preCore inverseLaw frame assignment frame.selectedColumn.id = 0
      rw [scalarCoordinates.2, selectedValue, activeOne, equalityValue,
        if_neg aligned, Fin.one_mul]
    have resultExact :
        Poseidon23Hash.resultCoordinates frame.plan
            (Honest.sourceValues frame assignment) =
          [0, 0, 0, 0, 0] := by
      unfold Poseidon23Hash.resultCoordinates
      rw [if_neg (by
        simpa [alignmentLeftValues, alignmentRightValues] using aligned)]
    have outputZero :
        ∀ index : Fin 5,
          completed (frame.outputAt index).id = 0 := by
      intro index
      have atIndex :=
        outputValues_getD frame completed index.val index.isLt
      rw [outputExact, resultExact] at atIndex
      have cases :
          index.val = 0 ∨ index.val = 1 ∨ index.val = 2 ∨
            index.val = 3 ∨ index.val = 4 := by
        omega
      rcases cases with first | second | third | fourth | fifth
      · have indexExact : index = 0 := Fin.ext first
        subst index
        exact atIndex.symm
      · have indexExact : index = 1 := Fin.ext second
        subst index
        exact atIndex.symm
      · have indexExact : index = 2 := Fin.ext third
        subst index
        exact atIndex.symm
      · have indexExact : index = 3 := Fin.ext fourth
        subst index
        exact atIndex.symm
      · have indexExact : index = 4 := Fin.ext fifth
        subst index
        exact atIndex.symm
    have selectedHolds :
        (selectedRow frame).Holds completed := by
      exact selectedRow_holds frame completed (by
        calc
          completed frame.selectedColumn.id = 0 := selectedCompleted
          _ = 1 * 0 := by rw [Fin.mul_zero]
          _ =
              completed frame.active *
                completed frame.equalityOutputColumn.id := by
            rw [activeCompleted, equalityCompleted])
    have tagHolds : (tagRow frame).Holds completed := by
      exact tagRow_holds_of_active frame completed activeCompleted
        ((outputZero 0).trans selectedCompleted.symm)
    have successHolds :
        RawSatisfies (List.ofFn (payloadSuccessRow frame)) completed := by
      apply rawSatisfies_of_forall
      intro row member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      exact payloadSuccessRow_holds_of_unselected frame completed lane
        selectedCompleted
    have absentHolds :
        RawSatisfies (List.ofFn (payloadAbsentRow frame)) completed := by
      apply rawSatisfies_of_forall
      intro row member
      rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
      exact payloadAbsentRow_holds_of_active_unselected frame completed
        lane activeCompleted selectedCompleted
        (outputZero ⟨lane.val + 1, by omega⟩)
    exact
      (rawSatisfies_append_iff _ _ completed).mpr
        ⟨⟨normalization, trivial⟩,
          (rawSatisfies_append_iff _ _ completed).mpr
            ⟨preimage,
              (rawSatisfies_append_iff _ _ completed).mpr
                ⟨equalityRaw,
                  (rawSatisfies_append_iff _ _ completed).mpr
                    ⟨⟨selectedHolds, ⟨tagHolds, trivial⟩⟩,
                      (rawSatisfies_append_iff _ _ completed).mpr
                        ⟨successHolds, absentHolds⟩⟩⟩⟩⟩

private theorem wrapper_preCore_inactive
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    RawSatisfies (wrapperRawRows frame)
      (preCore inverseLaw frame assignment) := by
  let completed := preCore inverseLaw frame assignment
  have oneCompleted : completed frame.one = 1 := by
    change preCore inverseLaw frame assignment frame.one = 1
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.one]
    · exact constantOne
    · simp [Frame.visibleIds]
  have activeCompleted : completed frame.active = 0 := by
    change preCore inverseLaw frame assignment frame.active = 0
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.active]
    · exact activeZero
    · simp [Frame.visibleIds]
  have equalityCoordinates :=
    preCore_equality_values inverseLaw frame assignment prefixNodup
  have scalarCoordinates :=
    preCore_scalar_values inverseLaw frame assignment prefixNodup
  have leftCoordinates :=
    preCore_projected_values inverseLaw frame assignment
      frame.plan.alignmentLeft prefixNodup temporariesDisjointVisible
  have rightCoordinates :=
    preCore_projected_values inverseLaw frame assignment
      frame.plan.alignmentRight prefixNodup temporariesDisjointVisible
  have equalityHolds :
      Satisfies frame.equality.rows completed := by
    apply frame.equality.inactive_complete inverseLaw completed
      oneCompleted activeCompleted
    · change
        frame.inverses.values
            (preCore inverseLaw frame assignment) =
          coordinateInverseValues inverseLaw
            ((frame.projected frame.plan.alignmentLeft).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
            ((frame.projected frame.plan.alignmentRight).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
      rw [leftCoordinates, rightCoordinates]
      exact equalityCoordinates.1
    · change
        frame.equals.values
            (preCore inverseLaw frame assignment) =
          coordinateEqualValues
            ((frame.projected frame.plan.alignmentLeft).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
            ((frame.projected frame.plan.alignmentRight).map
              (fun column =>
                preCore inverseLaw frame assignment column.id))
      rw [leftCoordinates, rightCoordinates]
      exact equalityCoordinates.2.1
    · change
        frame.products.values
            (preCore inverseLaw frame assignment) =
          productValues
            (frame.equals.values
              (preCore inverseLaw frame assignment))
      rw [equalityCoordinates.2.1]
      exact equalityCoordinates.2.2
  have equalityRaw :
      RawSatisfies frame.equality.rawRows completed :=
    (satisfies_ownRows_iff frame.owner frame.equality.rawRows
      completed).mp equalityHolds
  have selectedCompleted :
      completed frame.selectedColumn.id = 0 := by
    change
      preCore inverseLaw frame assignment frame.selectedColumn.id = 0
    rw [scalarCoordinates.2, selectedValue, activeZero, Fin.zero_mul]
  have normalization :=
    preCore_normalization_holds inverseLaw frame assignment constantOne
      prefixNodup temporariesDisjointVisible
  have preimage :=
    preCore_preimage_holds inverseLaw frame assignment constantOne
      prefixNodup temporariesDisjointVisible
  have selectedHolds :
      (selectedRow frame).Holds completed := by
    exact selectedRow_holds frame completed (by
      calc
        completed frame.selectedColumn.id = 0 := selectedCompleted
        _ = 0 *
            completed frame.equalityOutputColumn.id := by
          rw [Fin.zero_mul]
        _ =
            completed frame.active *
              completed frame.equalityOutputColumn.id := by
          rw [activeCompleted])
  have tagHolds : (tagRow frame).Holds completed := by
    exact tagRow_holds_of_inactive frame completed activeCompleted
  have successHolds :
      RawSatisfies (List.ofFn (payloadSuccessRow frame)) completed := by
    apply rawSatisfies_of_forall
    intro row member
    rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
    exact payloadSuccessRow_holds_of_unselected frame completed lane
      selectedCompleted
  have absentHolds :
      RawSatisfies (List.ofFn (payloadAbsentRow frame)) completed := by
    apply rawSatisfies_of_forall
    intro row member
    rcases List.mem_ofFn.mp member with ⟨lane, rfl⟩
    exact payloadAbsentRow_holds_of_equal_flags frame completed lane
      (activeCompleted.trans selectedCompleted.symm)
  exact
    (rawSatisfies_append_iff _ _ completed).mpr
      ⟨⟨normalization, trivial⟩,
        (rawSatisfies_append_iff _ _ completed).mpr
          ⟨preimage,
            (rawSatisfies_append_iff _ _ completed).mpr
              ⟨equalityRaw,
                (rawSatisfies_append_iff _ _ completed).mpr
                  ⟨⟨selectedHolds, ⟨tagHolds, trivial⟩⟩,
                    (rawSatisfies_append_iff _ _ completed).mpr
                      ⟨successHolds, absentHolds⟩⟩⟩⟩⟩

private theorem wrapper_agrees_after_core
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (temporaryNodup : frame.temporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    AgreesOn
      (rowsColumns (ownRows frame.owner (wrapperRawRows frame)))
      (preCore inverseLaw frame assignment)
      (complete inverseLaw frame facts assignment) := by
  have split := List.nodup_append.mp temporaryNodup
  have coreChanges :
      ChangesOnly frame.coreTemporaries.ids
        (preCore inverseLaw frame assignment)
        (complete inverseLaw frame facts assignment) := by
    exact CanonicalPoseidon2Sponge23Recipe.Honest.complete_changesOnly
      (core frame facts) (preCore inverseLaw frame assignment)
  apply agreesOn_of_changesOnly _ coreChanges
  intro id coreMember rowsMember
  rcases List.mem_flatMap.mp rowsMember with
    ⟨owned, ownedMember, columnMember⟩
  have rawMember :=
    ownRows_row_mem frame.owner (wrapperRawRows frame) owned ownedMember
  have columnMember' : id ∈ owned.columnIds := by
    simpa [rowColumns, OwnedRow.columnIds, Row.columnIds,
      List.map_append, List.append_assoc] using columnMember
  have supported :=
    wrapperRawRows_supported frame owned.row rawMember id columnMember'
  rcases List.mem_append.mp supported with visible | prefixMember
  · exact temporariesDisjointVisible id
      (List.mem_append_right frame.prefixTemporaryIds coreMember) visible
  · exact split.2.2 id prefixMember id coreMember rfl

private theorem core_outputs_correct
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    ∀ lane : Fin 4,
      preCore inverseLaw frame assignment
          (CanonicalPoseidon2Sponge23Recipe.outputColumn
            (core frame facts) lane.val) =
        CanonicalPoseidon2Sponge23Recipe.Numeric.semanticLane
          (core frame facts)
          (preCore inverseLaw frame assignment) lane := by
  intro lane
  rw [core_outputColumn frame facts lane,
    ← coreOutputValues_getD frame
      (preCore inverseLaw frame assignment) lane,
    preCore_digest inverseLaw frame assignment prefixNodup,
    digestValues,
    core_semanticLane frame facts
      (preCore inverseLaw frame assignment) lane,
    digestCoordinates_getD]
  rw [preCore_preimage inverseLaw frame assignment prefixNodup]

private theorem assemble
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (wrapperPreCore :
      RawSatisfies (wrapperRawRows frame)
        (preCore inverseLaw frame assignment))
    (coreComplete :
      Satisfies
        (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
        (complete inverseLaw frame facts assignment))
    (temporaryNodup : frame.temporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    Satisfies (rows frame facts)
      (complete inverseLaw frame facts assignment) := by
  have wrapperPre :
      Satisfies (ownRows frame.owner (wrapperRawRows frame))
        (preCore inverseLaw frame assignment) :=
    (satisfies_ownRows_iff frame.owner (wrapperRawRows frame)
      (preCore inverseLaw frame assignment)).mpr wrapperPreCore
  have wrapperFinal :
      Satisfies (ownRows frame.owner (wrapperRawRows frame))
        (complete inverseLaw frame facts assignment) :=
    satisfies_of_agrees
      (ownRows frame.owner (wrapperRawRows frame))
      (preCore inverseLaw frame assignment)
      (complete inverseLaw frame facts assignment)
      (wrapper_agrees_after_core inverseLaw frame facts assignment
        temporaryNodup temporariesDisjointVisible)
      wrapperPre
  apply
    (satisfies_ownRows_iff frame.owner (rawRows frame facts)
      (complete inverseLaw frame facts assignment)).mpr
  rw [rawRows_eq_wrapper_append_core]
  exact
    (rawSatisfies_append_iff _ _
      (complete inverseLaw frame facts assignment)).mpr
      ⟨(satisfies_ownRows_iff frame.owner (wrapperRawRows frame)
          (complete inverseLaw frame facts assignment)).mp wrapperFinal,
        raw_map_of_satisfies coreComplete⟩

/-- Active honest values extend by changing exactly the complete receipt and
satisfy the total optional hash occurrence. -/
theorem active_complete
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (outputsCorrect :
      outputValues frame assignment =
        Poseidon23Hash.resultCoordinates frame.plan
          (Honest.sourceValues frame assignment))
    (temporaryNodup : frame.temporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    Satisfies (rows frame facts)
      (complete inverseLaw frame facts assignment) := by
  have prefixNodup := (List.nodup_append.mp temporaryNodup).1
  have wrapper :=
    wrapper_preCore_active inverseLaw frame assignment constantOne
      activeOne prefixNodup temporariesDisjointVisible outputsCorrect
  by_cases aligned :
      alignmentLeftValues frame assignment =
        alignmentRightValues frame assignment
  · have selectedOne :
        preCore inverseLaw frame assignment frame.selectedColumn.id = 1 := by
      rw [(preCore_scalar_values inverseLaw frame assignment
        prefixNodup).2, selectedValue, activeOne, equalityValue,
        if_pos aligned, Fin.one_mul]
    have coreComplete :
        Satisfies
          (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
          (complete inverseLaw frame facts assignment) := by
      refine CanonicalPoseidon2Sponge23Recipe.active_complete
        (core frame facts) (preCore inverseLaw frame assignment) ?_ ?_ ?_
      · change preCore inverseLaw frame assignment frame.one = 1
        rw [preCore_agrees_visible inverseLaw frame assignment
            temporariesDisjointVisible frame.one]
        · exact constantOne
        · simp [Frame.visibleIds]
      · change
          preCore inverseLaw frame assignment frame.selectedColumn.id = 1
        exact selectedOne
      · intro lane
        simpa only [CanonicalPoseidon2Sponge23Recipe.outputWidth] using
          core_outputs_correct inverseLaw frame facts assignment
            prefixNodup lane
    exact assemble inverseLaw frame facts assignment wrapper coreComplete
      temporaryNodup temporariesDisjointVisible
  · have selectedZero :
        preCore inverseLaw frame assignment frame.selectedColumn.id = 0 := by
      rw [(preCore_scalar_values inverseLaw frame assignment
        prefixNodup).2, selectedValue, activeOne, equalityValue,
        if_neg aligned, Fin.one_mul]
    have coreComplete :
        Satisfies
          (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
          (complete inverseLaw frame facts assignment) := by
      refine CanonicalPoseidon2Sponge23Recipe.inactive_complete
        (core frame facts) (preCore inverseLaw frame assignment) ?_ ?_
      · change preCore inverseLaw frame assignment frame.one = 1
        rw [preCore_agrees_visible inverseLaw frame assignment
            temporariesDisjointVisible frame.one]
        · exact constantOne
        · simp [Frame.visibleIds]
      · change
          preCore inverseLaw frame assignment frame.selectedColumn.id = 0
        exact selectedZero
    exact assemble inverseLaw frame facts assignment wrapper coreComplete
      temporaryNodup temporariesDisjointVisible

/-- Inactive honest values retain deterministic internal witnesses while all
externally visible optional-result gates remain vacuous. -/
theorem inactive_complete
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0)
    (temporaryNodup : frame.temporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    Satisfies (rows frame facts)
      (complete inverseLaw frame facts assignment) := by
  have prefixNodup := (List.nodup_append.mp temporaryNodup).1
  have wrapper :=
    wrapper_preCore_inactive inverseLaw frame assignment constantOne
      activeZero prefixNodup temporariesDisjointVisible
  have selectedZero :
      preCore inverseLaw frame assignment frame.selectedColumn.id = 0 := by
    rw [(preCore_scalar_values inverseLaw frame assignment
      prefixNodup).2, selectedValue, activeZero, Fin.zero_mul]
  have coreComplete :
      Satisfies
        (CanonicalPoseidon2Sponge23Recipe.rows (core frame facts))
        (complete inverseLaw frame facts assignment) := by
    refine CanonicalPoseidon2Sponge23Recipe.inactive_complete
      (core frame facts) (preCore inverseLaw frame assignment) ?_ ?_
    · change preCore inverseLaw frame assignment frame.one = 1
      rw [preCore_agrees_visible inverseLaw frame assignment
          temporariesDisjointVisible frame.one]
      · exact constantOne
      · simp [Frame.visibleIds]
    · change
        preCore inverseLaw frame assignment frame.selectedColumn.id = 0
      exact selectedZero
  exact assemble inverseLaw frame facts assignment wrapper coreComplete
    temporaryNodup temporariesDisjointVisible

end Honest

end Poseidon23HashOccurrence

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
