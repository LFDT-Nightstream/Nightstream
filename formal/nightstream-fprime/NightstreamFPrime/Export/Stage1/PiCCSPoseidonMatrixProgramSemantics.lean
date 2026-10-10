import NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

/-!
Proves that the compact PiCCS Poseidon2 matrix program reconstructs the exact
action-driven input states.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

private theorem previousRule_zero
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin 16) :
    (previousRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val 0 lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [previousRule, PoseidonInput.Region.offsets?]

private theorem previousRule_succ
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocationOffset : Fin 947) (lane : Fin 16) :
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
  have slotBound : ∀ selected : Fin 16,
      134 + invocationOffset.val * 150 + selected.val <
        (PiCCSPoseidonPlan.schedule program).block.slotCount := by
    intro selected
    rw [(PiCCSPoseidonPlan.schedule program).slotCount_eq,
      PiCCSPoseidonPlan.invocationCount_eq,
      PoseidonRetainedSlots.rows_length]
    omega
  rw [show (previousRule program).form? logicalWidth
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
      (1 + invocationOffset.val) lane.val =
        some (some (SparseLayer.external (fun selected : Fin 16 =>
          (PiCCSPoseidonPlan.schedule program).block.form
            (PiCCSPoseidonPlan.retainedStart program)
            (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
            ⟨134 + invocationOffset.val * 150 + selected.val,
              slotBound selected⟩) lane)) by
    simpa [previousRule] using!
      PoseidonInput.Rule.external_form?_ofSemantic
        (region := PoseidonInput.Region.mk 1 947 0 16)
        invocationOffset lane lane.isLt
        (PiCCSPoseidonPlan.schedule program).block
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val 134 150 slotBound]
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
  simp only [PoseidonRetainedFamily.slot_val, PoseidonRetainedSlots.rows_length,
    PoseidonRetainedSlots.finalRow_val]
  omega

private theorem previousRule_result
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 16) :
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
    let invocationOffset : Fin 947 :=
      ⟨invocation.val - 1, by
        have bound : invocation.val < 948 := by
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

private theorem payloadRule_lane
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 12) :
    (payloadRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (some (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry)
        invocation ⟨lane.val, by change lane.val < 16; omega⟩)) := by
  have exactTerm : (payloadRule program).term.form? logicalWidth
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PiCCSPayloadWiring.form geometry (Fin.encodeProd (invocation, lane))) := by
    change (PoseidonInput.Term.affine
      (PiCCSPayloadMatrix.table ())
      (PiCCSOrdinaryMatrixProgram.substitution program) 12).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val = _
    rw [PiCCSPayloadMatrix.table_eq_ofSemantic]
    exact (PoseidonInput.Term.affine_form?_ofSemantic
      (laneCount := 12)
      PiCCSPayloadMatrix.combination (PiCCSOrdinaryMatrixProgram.substitution program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) invocation lane).trans
      (PiCCSPayloadMatrix.compileCombination_eq geometry (Fin.encodeProd (invocation, lane)))
  have offsets : (payloadRule program).region.offsets? invocation.val lane.val =
      some (invocation.val, lane.val) := by
    simpa only [payloadRule, Nat.zero_add] using
      (PoseidonInput.Region.offsets?_of_offsets
        (PoseidonInput.Region.mk 0 948 0 12) invocation lane)
  unfold PoseidonInput.Rule.form?
  rw [offsets]
  simp only
  rw [exactTerm]
  simp [PiCCSPoseidonPlan.payloadForm, Spec.Poseidon2.rate, lane.isLt]

private theorem payloadRule_outside
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 16)
    (outside : 12 ≤ lane.val) :
    (payloadRule program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [payloadRule, PoseidonInput.Region.offsets?, outside]

theorem inputProgram_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) (lane : Fin 16) :
    (inputProgram program).form? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation lane) := by
  have previous := previousRule_result geometry invocation lane
  cases kindFound : PiCCSActionPayloadBlock.kindAt invocation with
  | absorb block =>
      by_cases rateLane : lane.val < 12
      · let selectedLane : Fin 12 := ⟨lane.val, rateLane⟩
        have payload :
            (payloadRule program).form? logicalWidth
                (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val
                lane.val =
              some (some (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry) invocation
                lane)) := by
          simpa [selectedLane] using
            payloadRule_lane geometry invocation selectedLane
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
theorem inputProgram_state?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (invocation : Fin PiCCSPoseidonPlan.invocationCount) :
    (inputProgram program).state? logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val invocation.val =
      some (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry) invocation) := by
  apply PoseidonInput.Program.state?_eq_some
  · simpa using inputProgram_form? geometry invocation (0 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (1 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (2 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (3 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (4 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (5 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (6 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (7 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (8 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (9 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (10 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (11 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (12 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (13 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (14 : Fin 16)
  · simpa using inputProgram_form? geometry invocation (15 : Fin 16)

theorem poseidonBlock_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (global : Fin (PiCCSPoseidonPlan.invocationCount * 150)) :
    (poseidonBlock geometry).row? logicalWidth global.val =
      let decoded : Fin PiCCSPoseidonPlan.invocationCount × Fin 150 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiCCSPoseidonPlan.interface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) decoded.1 decoded.2) := by
  simpa [poseidonBlock, PiCCSPoseidonPlan.interface] using!
    Poseidon.Block.row?_ofSemantic (PiCCSPoseidonPlan.schedule program)
      (by rfl) (PiCCSPoseidonPlan.retainedStart program)
      (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (inputProgram program)
      (PiCCSPoseidonPlan.retainedFits
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
      (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry))
      (inputProgram_state? geometry) global

theorem matrixProgram_poseidon_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiCCSPoseidonPlan.invocationCount * 150)) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      let decoded : Fin PiCCSPoseidonPlan.invocationCount × Fin 150 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiCCSPoseidonPlan.interface (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)) decoded.1 decoded.2) := by
  rw [show matrixProgram geometry = MatrixProgram.Program.mk
      [.poseidon (poseidonBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos (by
    change global.val < (poseidonBlock geometry).rowCount
    rw [poseidonBlock, Poseidon.Block.ofSemantic_rowCount]
    exact global.isLt)]
  exact poseidonBlock_row? geometry global

/-- Every compact PiCCS Poseidon row is the exact row of the canonical
Poseidon plan. -/
theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form geometry)
          (PiCCSOrdinaryRetainedGeometry.poseidonGeometry geometry)).forms global) := by
  simpa [PiCCSPoseidonPlan.plan, PoseidonSboxFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      matrixProgram_poseidon_row? geometry sourceRow global

end NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgram
