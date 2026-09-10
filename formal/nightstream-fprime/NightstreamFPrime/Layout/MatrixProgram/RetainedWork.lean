import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.SparseWork

/-!
Counted construction of one retained slot's sparse reconstruction form.
The generator visits only that slot's coordinates. Its value is the existing
block form, including entry order. Work counts named operations and list
traversals; it does not measure machine execution or generate a whole block.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.MatrixProgram.RetainedWork

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Layout
open _root_.NightstreamFPrime.Layout.ProductionRelation
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private def coordinates {columns : Nat} (start : Nat) :
    (count : Nat) → start + count ≤ columns → Result (SparseForm columns)
  | 0, _ =>
      let result := SparseWork.empty ()
      ⟨result.value, result.work + 5⟩
  | count + 1, fits =>
      let tail := coordinates (start + 1) count (by omega)
      let head := SparseWork.singleton ⟨start, by omega⟩ 1
      let scaled := SparseWork.scale (⟨3, by decide⟩ : F) tail.value
      let joined := SparseWork.add head.value scaled.value
      ⟨joined.value, tail.work + head.work + scaled.work + joined.work + 15⟩

private theorem coordinates_value {columns : Nat} (start count : Nat)
    (fits : start + count ≤ columns) :
    (coordinates start count fits).value = RetainedSlot.recomposeForms
      (List.ofFn fun index : Fin count =>
        SparseForm.singleton ⟨start + index.val, by omega⟩ 1) := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
      have three : Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 = (⟨3, by decide⟩ : F) := by
        apply Fin.ext
        exact Nat.mod_eq_of_lt (by decide : 3 < goldilocksModulus)
      simp only [coordinates, SparseWork.add_value, SparseWork.singleton_value,
        SparseWork.scale_value, ih, List.ofFn_succ, RetainedSlot.recomposeForms,
        Fin.val_zero, Nat.add_zero, Fin.val_succ, Nat.add_assoc, Nat.add_comm 1, three]

private theorem coordinates_length {columns : Nat} (start count : Nat)
    (fits : start + count ≤ columns) :
    (coordinates start count fits).value.entries.length = count := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
      simp only [coordinates, SparseWork.add_value, SparseWork.singleton_value,
        SparseWork.scale_value, SparseForm.add, SparseForm.singleton,
        SparseForm.scale, List.length_append, List.length_map,
        List.length_cons, List.length_nil, ih]
      omega

/-- The empty case charges dispatch, Unit construction, the empty call,
its value read and Result construction. The step charges dispatch/predecessor,
start addition, four calls, column/one/three constructors, four value reads,
and Result construction (15). Clock arithmetic is instrumentation. -/
private theorem coordinates_work {columns : Nat} (start count : Nat)
    (fits : start + count ≤ columns) :
    (coordinates start count fits).work = 7 * count ^ 2 + 49 * count + 8 := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
      simp only [coordinates, SparseWork.singleton_work, SparseWork.scale_work,
        SparseWork.add_work, SparseWork.singleton_value, SparseForm.singleton,
        List.length_cons, List.length_nil, coordinates_length, ih]
      ring

/-- Check the whole block geometry before constructing the selected slot.
The successful wrapper charges slot-count read/comparison/branch (3), kind
read/width call/dispatch (3), start read (1), geometry arithmetic/check (4),
slot arithmetic (2), call/value read (2), and Some/Result construction (2).
Rejections charge their executed prefixes and None/Result construction. -/
def form? (block : RetainedBlock) (logicalWidth slot : Nat) :
    Result (Option (SparseForm logicalWidth)) :=
  let slotCount := block.slotCount
  if slotBound : slot < slotCount then
    let width := block.kind.width
    let start := block.start
    if fits : start + slotCount * width ≤ logicalWidth then
      let result := coordinates (start + slot * width) width (by
          have bound := Nat.mul_le_mul_right width
            (show slot + 1 ≤ slotCount by omega)
          nlinarith)
      ⟨some result.value, result.work + 17⟩
    else ⟨none, 13⟩
  else ⟨none, 5⟩

theorem form?_value (block : RetainedBlock) (logicalWidth slot : Nat) :
    (form? block logicalWidth slot).value = block.form? logicalWidth slot := by
  dsimp only [form?, RetainedBlock.form?, RetainedBlock.coordinateCount]
  split
  · split
    · rw [coordinates_value]
      simp only [LowNormBlock.Block.form, LowNormBlock.Block.column,
        LowNormBlock.Block.coordinateOffset, RetainedBlock.semantic, Nat.add_assoc]
    · rfl
  · rfl

theorem form?_length (block : RetainedBlock) (logicalWidth slot : Nat)
    (form : SparseForm logicalWidth)
    (returned : (form? block logicalWidth slot).value = some form) :
    form.entries.length = block.kind.width := by
  dsimp only [form?] at returned
  split at returned
  · split at returned
    · cases Option.some.inj returned
      exact coordinates_length _ _ _
    · contradiction
  · contradiction

theorem form?_work_le (block : RetainedBlock) (logicalWidth slot : Nat) :
    (form? block logicalWidth slot).work ≤
      7 * block.kind.width ^ 2 + 49 * block.kind.width + 25 := by
  dsimp only [form?]
  split
  · split
    · rw [coordinates_work]
    · change (13 : Nat) ≤ _
      omega
  · change (5 : Nat) ≤ _
    omega

end NightstreamFPrime.Layout.MatrixProgram.RetainedWork
