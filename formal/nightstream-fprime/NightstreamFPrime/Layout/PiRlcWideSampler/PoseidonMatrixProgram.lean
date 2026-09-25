import NightstreamFPrime.Layout.PiRlcWideSampler.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.Exact

/-! Compact matrix program for the wide sampler's 34 Poseidon2 permutations.
The eight initial input forms are serialized directly from the proved plan. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.PoseidonMatrix

open MatrixProgram ProductionRelation NightstreamFPrime.Spec

private def firstRule {columns : Nat} (interface : BatchPlan.Interface columns) : PoseidonInput.Rule :=
  ⟨⟨0, 1, 0, 8⟩, .sparse (Array.ofFn fun lane => WireForm.ofSemantic (interface.initialState lane)) 8⟩

private def previousRule {columns : Nat} (interface : BatchPlan.Interface columns) : PoseidonInput.Rule :=
  ⟨⟨1, 33, 0, 8⟩, .external (MatrixRows.poseidonBlock interface) 78 86⟩

private def constantAt (index : Fin (34 * 8)) : Option F :=
  let decoded : Fin 34 × Fin 8 := Fin.decodeProd index
  if decoded.1.val % 2 = 0 then some (BatchPlan.entryWord (decoded.1.val / 2) decoded.2) else none

private def entryRule : PoseidonInput.Rule :=
  ⟨⟨0, 34, 0, 8⟩, .optionalConstant (PoseidonInput.OptionalConstantTable.ofSemantic constantAt) 8⟩

def inputProgram {columns : Nat} (interface : BatchPlan.Interface columns) : PoseidonInput.Program :=
  ⟨[firstRule interface, previousRule interface, entryRule]⟩

private theorem previous {columns : Nat} (interface : BatchPlan.Interface columns)
    (offset : Fin 33) (lane : Fin 8) :
    (previousRule interface).form? columns interface.oneColumn.val (1 + offset.val) lane.val =
      some (some (BatchPlan.outputState interface ⟨offset.val, by omega⟩ lane)) := by
  have slots : ∀ selected : Fin 8, 78 + offset.val * 86 + selected.val < BatchPlan.poseidonBlock.slotCount := by
    intro selected
    change 78 + offset.val * 86 + selected.val < 34 * 86
    omega
  have loaded := PoseidonInput.Rule.external_form?_ofSemantic
    (region := ⟨1, 33, 0, 8⟩) offset lane lane.isLt BatchPlan.poseidonBlock interface.start
    (MatrixRows.poseidonFits interface) interface.oneColumn.val 78 86 slots
  simp only [Nat.zero_add] at loaded
  change (previousRule interface).form? columns interface.oneColumn.val (1 + offset.val) lane.val = _ at loaded
  refine loaded.trans ?_
  apply congrArg some
  apply congrArg some
  unfold BatchPlan.outputState
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  unfold BatchPlan.sbox
  apply congrArg (BatchPlan.poseidonBlock.form interface.start _)
  apply Fin.ext
  simp only [PoseidonRetainedSlots.finalRow_val]
  omega

private theorem entry {columns : Nat} (one : Fin columns) (current : Fin 34) (lane : Fin 8) :
    entryRule.form? columns one.val current.val lane.val =
      some (some (if current.val % 2 = 0 then
        SparseForm.singleton one (BatchPlan.entryWord (current.val / 2) lane) else .empty)) := by
  have indexEq : current.val * 8 + lane.val = (Fin.encodeProd (current, lane)).val := by
    simp [Fin.encodeProd, Nat.mul_comm]
  by_cases even : current.val % 2 = 0
  · have found : constantAt (Fin.encodeProd (current, lane)) =
        some (BatchPlan.entryWord (current.val / 2) lane) := by simp [constantAt, even]
    simpa only [entryRule, if_pos even, Nat.zero_add] using
      PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_some
        (region := ⟨0, 34, 0, 8⟩) current lane one constantAt 8
        (Fin.encodeProd (current, lane)) indexEq _ found
  · have found : constantAt (Fin.encodeProd (current, lane)) = none := by simp [constantAt, even]
    simpa only [entryRule, if_neg even, Nat.zero_add] using
      PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_none
        (region := ⟨0, 34, 0, 8⟩) current lane one.val constantAt 8
        (Fin.encodeProd (current, lane)) indexEq found

theorem input_form {columns : Nat} (interface : BatchPlan.Interface columns)
    (current : Fin 34) (lane : Fin 8) :
    (inputProgram interface).form? columns interface.oneColumn.val current.val lane.val =
      some ((BatchPlan.poseidonInterface interface).input current lane) := by
  have first : (firstRule interface).form? columns interface.oneColumn.val current.val lane.val =
      some (if current.val = 0 then some (BatchPlan.priorState interface current lane) else none) := by
    by_cases zero : current.val = 0
    · simp [firstRule, PoseidonInput.Rule.form?, PoseidonInput.Region.offsets?,
        PoseidonInput.Term.form?, zero, lane.isLt, WireForm.semantic?_ofSemantic, BatchPlan.priorState]
    · rw [if_neg zero]
      apply PoseidonInput.Rule.form?_eq_some_none
      simp [firstRule, PoseidonInput.Region.offsets?, zero]
  have prior : (previousRule interface).form? columns interface.oneColumn.val current.val lane.val =
      some (if current.val = 0 then none else some (BatchPlan.priorState interface current lane)) := by
    by_cases zero : current.val = 0
    · rw [if_pos zero, zero]
      apply PoseidonInput.Rule.form?_eq_some_none
      simp [previousRule, PoseidonInput.Region.offsets?]
    · rw [if_neg zero]
      let offset : Fin 33 := ⟨current.val - 1, by omega⟩
      have position : current.val = 1 + offset.val := by dsimp [offset]; omega
      rw [position]
      simpa only [BatchPlan.priorState, dif_neg zero] using previous interface offset lane
  have folded := PoseidonInput.Program.three_form?_of_results
    (firstRule interface) (previousRule interface) entryRule interface.oneColumn.val current.val lane.val
    _ _ _ first prior (entry interface.oneColumn current lane)
  by_cases zero : current.val = 0 <;> by_cases even : current.val % 2 = 0 <;>
    simpa [inputProgram, BatchPlan.poseidonInterface, zero, even, SparseLayer.addConstant,
      SparseForm.add, SparseForm.empty] using! folded

private def schedule : PoseidonRetainedFamily.Schedule (34 * 86) 34 where
  block := BatchPlan.poseidonBlock
  slotCount_eq := by simp [BatchPlan.poseidonBlock, PoseidonRetainedSlots.rows_length]

def block {columns : Nat} (interface : BatchPlan.Interface columns) : Poseidon.Block :=
  Poseidon.Block.ofSemantic schedule interface.start interface.oneColumn (inputProgram interface)

def program {columns : Nat} (interface : BatchPlan.Interface columns) : MatrixProgram.Program :=
  ⟨[.poseidon (block interface)]⟩

private theorem family_interface {columns : Nat} (interface : BatchPlan.Interface columns) :
    PoseidonRetainedFamily.familyInterface schedule interface.start (MatrixRows.poseidonFits interface)
        interface.oneColumn (BatchPlan.poseidonInterface interface).input = BatchPlan.poseidonInterface interface := by
  unfold PoseidonRetainedFamily.familyInterface BatchPlan.poseidonInterface
  congr 1
  funext invocation row
  unfold PoseidonRetainedFamily.form BatchPlan.sbox
  apply congrArg (BatchPlan.poseidonBlock.form interface.start _)
  apply Fin.ext
  simp [PoseidonRetainedFamily.slot, Fin.encodeProd, PoseidonRetainedSlots.rows_length, Nat.mul_comm]

/-- The serialized program gives every row of the candidate's Poseidon family. -/
theorem exact {columns : Nat} (interface : BatchPlan.Interface columns)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (program interface) (BatchPlan.poseidonPlan interface) sourceRow := by
  refine ⟨rfl, ?_⟩
  intro row
  have inputs : ∀ current : Fin 34,
      (inputProgram interface).state? columns interface.oneColumn.val current.val =
        some ((BatchPlan.poseidonInterface interface).input current) := by
    intro current
    apply PoseidonInput.Program.state?_eq_some
    · exact input_form interface current (0 : Fin 8)
    · exact input_form interface current (1 : Fin 8)
    · exact input_form interface current (2 : Fin 8)
    · exact input_form interface current (3 : Fin 8)
    · exact input_form interface current (4 : Fin 8)
    · exact input_form interface current (5 : Fin 8)
    · exact input_form interface current (6 : Fin 8)
    · exact input_form interface current (7 : Fin 8)
  rw [show program interface = ⟨[.poseidon (block interface)]⟩ by rfl,
    MatrixProgram.Program.singleton_row?, if_pos (show row.val < (MatrixProgram.Block.poseidon (block interface)).rowCount from row.isLt)]
  have result := Poseidon.Block.row?_ofSemantic schedule rfl interface.start interface.oneColumn
    (inputProgram interface) (MatrixRows.poseidonFits interface)
    (BatchPlan.poseidonInterface interface).input inputs row
  rw [family_interface] at result
  exact result

end NightstreamFPrime.Layout.PiRlcWideSampler.PoseidonMatrix
