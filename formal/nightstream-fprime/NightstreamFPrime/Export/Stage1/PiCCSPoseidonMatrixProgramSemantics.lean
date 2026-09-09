import NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

/-!
Proves that the compact PiCCS Poseidon2 matrix program reconstructs the exact
action-driven input states and squeeze-binding rows.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

private theorem previousRule_zero
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin 8) :
    (previousRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val 0 lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [previousRule, PoseidonInput.Region.offsets?]

private theorem previousRule_succ
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocationOffset : Fin 7603) (lane : Fin 8) :
    (previousRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
        (1 + invocationOffset.val) lane.val =
      some (some (PoseidonRetainedFamily.outputState
        (PiCCSPoseidonPlan.schedule program)
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
        ⟨invocationOffset.val, by
          rw [PiCCSPoseidonPlan.invocationCount_eq]
          omega⟩ lane)) := by
  have slotBound : ∀ selected : Fin 8,
      78 + invocationOffset.val * 86 + selected.val <
        (PiCCSPoseidonPlan.schedule program).block.slotCount := by
    intro selected
    rw [(PiCCSPoseidonPlan.schedule program).slotCount_eq,
      PiCCSPoseidonPlan.invocationCount_eq,
      PoseidonRetainedSlots.rows_length]
    omega
  rw [show (previousRule program).form? logicalWidth
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
      (1 + invocationOffset.val) lane.val =
        some (some (SparseLayer.external (fun selected : Fin 8 =>
          (PiCCSPoseidonPlan.schedule program).block.form
            (PiCCSPoseidonPlan.retainedStart program)
            (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
            ⟨78 + invocationOffset.val * 86 + selected.val,
              slotBound selected⟩) lane)) by
    simpa [previousRule] using
      PoseidonInput.Rule.external_form?_ofSemantic
        (region := PoseidonInput.Region.mk 1 7603 0 8)
        invocationOffset lane lane.isLt
        (PiCCSPoseidonPlan.schedule program).block
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val 78 86 slotBound]
  apply congrArg some
  apply congrArg some
  unfold PoseidonRetainedFamily.outputState PoseidonRetainedFamily.form
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  apply congrArg ((PiCCSPoseidonPlan.schedule program).block.form
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)))
  apply Fin.ext
  simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
    PoseidonRetainedSlots.finalRow_val]
  omega

private theorem previousRule_result
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8) :
    (previousRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (if invocation.val = 0 then none else
        some (PiCCSPoseidonPlan.previousOutput
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation lane)) := by
  by_cases first : invocation.val = 0
  · rw [if_pos first]
    have invocationEq : invocation = ⟨0, by omega⟩ := by
      apply Fin.ext
      exact first
    rw [invocationEq]
    exact previousRule_zero geometry lane
  · rw [if_neg first]
    let invocationOffset : Fin 7603 :=
      ⟨invocation.val - 1, by
        have bound : invocation.val < 7604 := by
          simpa only [PiCCSPoseidonPlan.invocationCount_eq] using
            invocation.isLt
        omega⟩
    have invocationEq : invocation.val = 1 + invocationOffset.val := by
      dsimp [invocationOffset]
      omega
    rw [invocationEq]
    have selected := previousRule_succ geometry invocationOffset lane
    rw [selected]
    apply congrArg some
    apply congrArg some
    unfold PiCCSPoseidonPlan.previousOutput
    rw [dif_neg first]

private theorem payloadRule_absorb
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 4)
    (block : List NightstreamFPrime.Circuit.Expr)
    (found : PiCCSActionPayloadBlock.kindAt invocation = .absorb block) :
    (payloadRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (some (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry)
        invocation ⟨lane.val, by change lane.val < 8; omega⟩)) := by
  have selected : tagAt invocation = .absorb := by
    simp [tagAt, invocationTag, found]
  have exactTerm : (payloadRule program).term.form? logicalWidth
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PiCCSPayloadWiring.form geometry (Fin.encodeProd (invocation, lane))) := by
    change (PoseidonInput.Term.taggedAffine
      (PiCCSPayloadMatrix.table ())
      (PiCCSOrdinaryMatrixProgram.substitution program)
      (PoseidonInput.TagTable.ofSemantic tagAt) .absorb 4).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val = _
    rw [PiCCSPayloadMatrix.table_eq_ofSemantic]
    exact (PoseidonInput.Term.taggedAffine_form?_of_eq
      (laneCount := 4)
      PiCCSPayloadMatrix.combination (PiCCSOrdinaryMatrixProgram.substitution program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) tagAt .absorb invocation lane selected).trans
      (PiCCSPayloadMatrix.compileCombination_eq geometry (Fin.encodeProd (invocation, lane)))
  have offsets : (payloadRule program).region.offsets? invocation.val lane.val =
      some (invocation.val, lane.val) := by
    simpa only [payloadRule, Nat.zero_add] using
      (PoseidonInput.Region.offsets?_of_offsets
        (PoseidonInput.Region.mk 0 7604 0 4) invocation lane)
  unfold PoseidonInput.Rule.form?
  rw [offsets]
  simp only
  rw [exactTerm]
  simp [PiCCSPoseidonPlan.payloadForm, Spec.Poseidon2.rate, lane.isLt]

private theorem payloadRule_notAbsorb
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 4)
    (notSelected : tagAt invocation ≠ .absorb) :
    (payloadRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (some .empty) := by
  have exactTerm : (payloadRule program).term.form? logicalWidth
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some .empty := by
    exact PoseidonInput.Term.taggedAffine_form?_of_ne
      (PiCCSPayloadMatrix.table ()) (PiCCSOrdinaryMatrixProgram.substitution program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val 4 lane.val
      tagAt .absorb invocation notSelected
  have offsets : (payloadRule program).region.offsets? invocation.val lane.val =
      some (invocation.val, lane.val) := by
    simpa only [payloadRule, Nat.zero_add] using
      (PoseidonInput.Region.offsets?_of_offsets
        (PoseidonInput.Region.mk 0 7604 0 4) invocation lane)
  unfold PoseidonInput.Rule.form?
  rw [offsets]
  simp only
  rw [exactTerm]
  rfl

private theorem payloadRule_outside
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8)
    (outside : 4 ≤ lane.val) :
    (payloadRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [payloadRule, PoseidonInput.Region.offsets?, outside]

theorem inputProgram_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 8) :
    (inputProgram program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation lane) := by
  have previous := previousRule_result geometry invocation lane
  cases kindFound : PiCCSActionPayloadBlock.kindAt invocation with
  | absorb block =>
      by_cases rateLane : lane.val < 4
      · let selectedLane : Fin 4 := ⟨lane.val, rateLane⟩
        have payload :
            (payloadRule program).form? logicalWidth
                (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val
                lane.val =
              some (some (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry) invocation
                lane)) := by
          simpa [selectedLane] using
            payloadRule_absorb geometry invocation selectedLane block kindFound
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, SparseForm.add,
            SparseForm.empty] using folded
      · have payload := payloadRule_outside geometry invocation lane (by omega)
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, PiCCSPoseidonPlan.payloadForm,
            Spec.Poseidon2.rate, rateLane, SparseForm.add,
            SparseForm.empty] using folded
  | squeezeFirst expected =>
      by_cases rateLane : lane.val < 4
      · let selectedLane : Fin 4 := ⟨lane.val, rateLane⟩
        have payload :
            (payloadRule program).form? logicalWidth
                (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val
                lane.val = some (some .empty) := by
          simpa [selectedLane] using
            payloadRule_notAbsorb geometry invocation selectedLane
              (by simp [tagAt, invocationTag, kindFound])
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, SparseForm.add,
            SparseForm.empty] using folded
      · have payload := payloadRule_outside geometry invocation lane (by omega)
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, SparseForm.add,
            SparseForm.empty] using folded
  | squeezeSecond =>
      by_cases rateLane : lane.val < 4
      · let selectedLane : Fin 4 := ⟨lane.val, rateLane⟩
        have payload :
            (payloadRule program).form? logicalWidth
                (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val
                lane.val = some (some .empty) := by
          simpa [selectedLane] using
            payloadRule_notAbsorb geometry invocation selectedLane
              (by simp [tagAt, invocationTag, kindFound])
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, SparseForm.add,
            SparseForm.empty] using folded
      · have payload := payloadRule_outside geometry invocation lane (by omega)
        have folded := PoseidonInput.Program.two_form?_of_results
          (previousRule program) (payloadRule program)
          (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val
          _ _ previous payload
        by_cases first : invocation.val = 0 <;>
          simpa [inputProgram, PiCCSPoseidonPlan.inputState, kindFound, first,
            PiCCSPoseidonPlan.previousOutput, SparseForm.add,
            SparseForm.empty] using folded

theorem inputProgram_state?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    (inputProgram program).state? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val =
      some (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation) := by
  apply PoseidonInput.Program.state?_eq_some
  · simpa using inputProgram_form? geometry invocation (0 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (1 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (2 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (3 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (4 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (5 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (6 : Fin 8)
  · simpa using inputProgram_form? geometry invocation (7 : Fin 8)

theorem poseidonBlock_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (global : Fin (PiCCSPoseidonPlan.invocationCount * 94)) :
    (poseidonBlock geometry).row? logicalWidth global.val =
      let decoded : Fin PiCCSPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiCCSPoseidonPlan.interface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) decoded.1 decoded.2) := by
  simpa [poseidonBlock, PiCCSPoseidonPlan.interface] using
    Poseidon.Block.row?_ofSemantic (PiCCSPoseidonPlan.schedule program)
      (by rfl) (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)
      (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
      (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
      (inputProgram_state? geometry) global

theorem bindingBlock_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (row : Fin PiCCSPoseidonPlan.bindingRowCount) :
    (bindingBlock geometry).row? logicalWidth row.val =
      some (PinFamilyPlan.forms
        (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) row) := by
  exact Pin.Block.row?_ofSemantic
    (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) row

theorem matrixProgram_poseidon_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiCCSPoseidonPlan.invocationCount * 94)) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      let decoded : Fin PiCCSPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiCCSPoseidonPlan.interface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) decoded.1 decoded.2) := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.poseidon (poseidonBlock geometry), .pin (bindingBlock geometry)] by
    rfl]
  rw [MatrixProgram.Program.two_first_row?
    (.poseidon (poseidonBlock geometry)) (.pin (bindingBlock geometry))
    logicalWidth sourceRow global.val (by exact global.isLt)]
  exact poseidonBlock_row? geometry global

theorem matrixProgram_binding_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiCCSPoseidonPlan.bindingRowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow
        (PiCCSPoseidonPlan.invocationCount * 94 + row.val) =
      some (PinFamilyPlan.forms
        (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) row).meaningfulForm := by
  let left : MatrixProgram.Block := .poseidon (poseidonBlock geometry)
  let right : MatrixProgram.Block := .pin (bindingBlock geometry)
  have leftCount : left.rowCount =
      PiCCSPoseidonPlan.invocationCount * 94 := by
    change (poseidonBlock geometry).rowCount = _
    exact Poseidon.Block.ofSemantic_rowCount
      (PiCCSPoseidonPlan.schedule program)
      (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)
  have rightCount : right.rowCount = PiCCSPoseidonPlan.bindingRowCount := by
    change (bindingBlock geometry).rowCount = _
    exact Pin.Block.ofSemantic_rowCount
      (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
  have leftBound : left.rowCount ≤
      PiCCSPoseidonPlan.invocationCount * 94 + row.val := by
    rw [leftCount]
    omega
  have rightBound :
      PiCCSPoseidonPlan.invocationCount * 94 + row.val - left.rowCount <
        right.rowCount := by
    rw [leftCount, rightCount, Nat.add_sub_cancel_left]
    exact row.isLt
  have selected := MatrixProgram.Program.two_second_row? left right
    logicalWidth sourceRow
    (PiCCSPoseidonPlan.invocationCount * 94 + row.val)
    leftBound rightBound
  rw [leftCount, Nat.add_sub_cancel_left] at selected
  have wrapped : right.row? logicalWidth sourceRow row.val =
      some (PinFamilyPlan.forms
        (PiCCSPoseidonPlan.bindingInterface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) row).meaningfulForm := by
    change (do
      let forms ← (bindingBlock geometry).row? logicalWidth row.val
      pure forms.meaningfulForm) = _
    rw [bindingBlock_row? geometry row]
    rfl
  rw [show matrixProgram geometry = MatrixProgram.Program.mk [left, right] by
    rfl]
  rw [selected, wrapped]

theorem matrixProgram_sbox_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiCCSPoseidonPlan.sboxPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiCCSPoseidonPlan.sboxPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms global) := by
  simpa [PiCCSPoseidonPlan.sboxPlan, PoseidonSboxFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      matrixProgram_poseidon_row? geometry sourceRow global

theorem matrixProgram_binding_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin (PiCCSPoseidonPlan.bindingPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow
        ((PiCCSPoseidonPlan.sboxPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).rowCount + row.val) =
      some ((PiCCSPoseidonPlan.bindingPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms row) := by
  simpa [PiCCSPoseidonPlan.sboxPlan, PiCCSPoseidonPlan.bindingPlan,
    PinFamilyPlan.plan] using
      matrixProgram_binding_row? geometry sourceRow row

/-- Every compact PiCCS Poseidon row is the exact row of the canonical
Poseidon-and-binding plan. -/
theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms global) := by
  let sboxPlan := PiCCSPoseidonPlan.sboxPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)
  let bindingPlan := PiCCSPoseidonPlan.bindingPlan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)
  cases selected : ProductionRelation.Plan.splitIndex
      sboxPlan.rowCount bindingPlan.rowCount global with
  | inl sboxRow =>
      have globalEq := ProductionRelation.Plan.leftIndex_of_splitIndex_eq
        sboxPlan.rowCount bindingPlan.rowCount global sboxRow selected
      rw [← globalEq]
      calc
        (matrixProgram geometry).row? logicalWidth sourceRow
            (ProductionRelation.Plan.leftIndex sboxPlan.rowCount
              bindingPlan.rowCount sboxRow).val =
            some (sboxPlan.forms sboxRow) := by
              simpa only [ProductionRelation.Plan.leftIndex_val,
                sboxPlan] using
                  matrixProgram_sbox_plan_row? geometry sourceRow sboxRow
        _ = some ((PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms
              (ProductionRelation.Plan.leftIndex sboxPlan.rowCount
                bindingPlan.rowCount sboxRow)) := by
              apply congrArg some
              funext port
              simpa [PiCCSPoseidonPlan.plan, sboxPlan, bindingPlan] using
                (ProductionRelation.Plan.append_forms_left sboxPlan bindingPlan
                  (PiCCSPoseidonPlan.combinedRowCount_le (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
                  sboxRow port).symm
  | inr bindingRow =>
      have globalEq := ProductionRelation.Plan.rightIndex_of_splitIndex_eq
        sboxPlan.rowCount bindingPlan.rowCount global bindingRow selected
      rw [← globalEq]
      calc
        (matrixProgram geometry).row? logicalWidth sourceRow
            (ProductionRelation.Plan.rightIndex sboxPlan.rowCount
              bindingPlan.rowCount bindingRow).val =
            some (bindingPlan.forms bindingRow) := by
              simpa only [ProductionRelation.Plan.rightIndex_val,
                sboxPlan, bindingPlan] using
                  matrixProgram_binding_plan_row? geometry sourceRow bindingRow
        _ = some ((PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms
              (ProductionRelation.Plan.rightIndex sboxPlan.rowCount
                bindingPlan.rowCount bindingRow)) := by
              apply congrArg some
              funext port
              simpa [PiCCSPoseidonPlan.plan, sboxPlan, bindingPlan] using
                (ProductionRelation.Plan.append_forms_right sboxPlan
                  bindingPlan (PiCCSPoseidonPlan.combinedRowCount_le (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
                  bindingRow port).symm

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
