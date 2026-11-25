import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeErase
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! The schedule facts of each script of `layout30` for the two rows of `key30.aks` as theorems
(Akita revision `d50f700`; `Replay/ShapeErase.lean`).

- `skeleton30_w27_level0` to `skeleton30_w26_tensor` are closed computations of the kernel on the
  skeleton of `layout30 r 0 0` for `r = Key30.w27` and `r = Key30.w26`. The folds and the skeleton
  do not depend on `zMax` and `rice` (`skeleton30`, `folds30`).
- `fold30 r ℓ` is the fold of the level `ℓ` of `layout30 r` (`layout30_fold`).
- `script30`: each script of `layout30 r zMax rice` for a row `r` has `endsFit`, the terminal of
  the level 7 as its last operation with its tensor reduction, at each level the shapes at
  `fused.end`, at the last range check and at the tensor reduction of a trace level, and at the
  offloaded root `offloadShape` with the plan of Stage 3 and its rounds. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

theorem skeleton30_w27_level0 : levelShaped (layout30 Key30.w27 0 0) 0 = true := by
  decide +kernel

theorem skeleton30_w27_level1 : levelShaped (layout30 Key30.w27 0 0) 1 = true := by
  decide +kernel

theorem skeleton30_w27_level2 : levelShaped (layout30 Key30.w27 0 0) 2 = true := by
  decide +kernel

theorem skeleton30_w27_level3 : levelShaped (layout30 Key30.w27 0 0) 3 = true := by
  decide +kernel

theorem skeleton30_w27_level4 : levelShaped (layout30 Key30.w27 0 0) 4 = true := by
  decide +kernel

theorem skeleton30_w27_level5 : levelShaped (layout30 Key30.w27 0 0) 5 = true := by
  decide +kernel

theorem skeleton30_w27_level6 : levelShaped (layout30 Key30.w27 0 0) 6 = true := by
  decide +kernel

theorem skeleton30_w27_ends : endsFit (skeleton (layout30 Key30.w27 0 0)).zipIdx = true := by
  decide +kernel

theorem skeleton30_w27_tensor : tensorShapeAt (skeleton (layout30 Key30.w27 0 0)).zipIdx 7
    (layout30 Key30.w27 0 0).terminal.rounds = true := by
  decide +kernel

theorem skeleton30_w26_level0 : levelShaped (layout30 Key30.w26 0 0) 0 = true := by
  decide +kernel

theorem skeleton30_w26_level1 : levelShaped (layout30 Key30.w26 0 0) 1 = true := by
  decide +kernel

theorem skeleton30_w26_level2 : levelShaped (layout30 Key30.w26 0 0) 2 = true := by
  decide +kernel

theorem skeleton30_w26_level3 : levelShaped (layout30 Key30.w26 0 0) 3 = true := by
  decide +kernel

theorem skeleton30_w26_level4 : levelShaped (layout30 Key30.w26 0 0) 4 = true := by
  decide +kernel

theorem skeleton30_w26_level5 : levelShaped (layout30 Key30.w26 0 0) 5 = true := by
  decide +kernel

theorem skeleton30_w26_level6 : levelShaped (layout30 Key30.w26 0 0) 6 = true := by
  decide +kernel

theorem skeleton30_w26_ends : endsFit (skeleton (layout30 Key30.w26 0 0)).zipIdx = true := by
  decide +kernel

theorem skeleton30_w26_tensor : tensorShapeAt (skeleton (layout30 Key30.w26 0 0)).zipIdx 7
    (layout30 Key30.w26 0 0).terminal.rounds = true := by
  decide +kernel

/-- The skeleton of `layout30` does not depend on `zMax` and `rice`. -/
theorem skeleton30 (r : Key30.Row) (zMax rice : ℕ) :
    skeleton (layout30 r zMax rice) = skeleton (layout30 r 0 0) := rfl

theorem folds30 (r : Key30.Row) (zMax rice : ℕ) :
    (layout30 r zMax rice).folds = (layout30 r 0 0).folds := rfl

theorem fusedIndex30 (r : Key30.Row) (zMax rice ℓ : ℕ) :
    fusedIndex (layout30 r zMax rice) ℓ = fusedIndex (layout30 r 0 0) ℓ := rfl

theorem length30 (r : Key30.Row) : (layout30 r 0 0).folds.length = 7 := rfl

/-- The fold of the level `ℓ` of `layout30 r`. -/
def fold30 (r : Key30.Row) (ℓ : Fin 7) : Fold :=
  (layout30 r 0 0).folds[ℓ.val]'(by rw [length30]; exact ℓ.isLt)

theorem layout30_fold (r : Key30.Row) (zMax rice : ℕ) (ℓ : Fin 7) :
    (layout30 r zMax rice).folds[ℓ.val]? = some (fold30 r ℓ) := by
  rw [folds30]
  exact List.getElem?_eq_getElem _

theorem skeleton30_shaped {r : Key30.Row} (member : r ∈ Key30.rows) (ℓ : Fin 7) :
    levelShaped (layout30 r 0 0) ℓ = true ∧ endsFit (skeleton (layout30 r 0 0)).zipIdx = true ∧
      tensorShapeAt (skeleton (layout30 r 0 0)).zipIdx 7 (layout30 r 0 0).terminal.rounds = true := by
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl
  · refine ⟨?_, skeleton30_w26_ends, skeleton30_w26_tensor⟩
    match ℓ with
    | 0 => exact skeleton30_w26_level0
    | 1 => exact skeleton30_w26_level1
    | 2 => exact skeleton30_w26_level2
    | 3 => exact skeleton30_w26_level3
    | 4 => exact skeleton30_w26_level4
    | 5 => exact skeleton30_w26_level5
    | 6 => exact skeleton30_w26_level6
  · refine ⟨?_, skeleton30_w27_ends, skeleton30_w27_tensor⟩
    match ℓ with
    | 0 => exact skeleton30_w27_level0
    | 1 => exact skeleton30_w27_level1
    | 2 => exact skeleton30_w27_level2
    | 3 => exact skeleton30_w27_level3
    | 4 => exact skeleton30_w27_level4
    | 5 => exact skeleton30_w27_level5
    | 6 => exact skeleton30_w27_level6

/-- The root of each row has Stage 3 with `2^23` cells and the plan of the weight of the code. -/
theorem fold30_root {r : Key30.Row} (member : r ∈ Key30.rows) :
    (fold30 r 0).stage3 = some Key30.setupVars ∧
      (fold30 r 0).setup = some (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) (fold30 r 0).low 3) := by
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> exact ⟨rfl, by decide⟩

/-- The schedule facts of each script of `layout30` for a row of `key30.aks`. -/
theorem script30 {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ} {plan : Array Run}
    {ops : Array Op} (scripted : script (layout30 r zMax rice) plan = .ok ops) :
    endsFit ops.zipIdx.toList = true ∧ ops[ops.size - 1]? = some (.terminal 7) ∧
      (∃ c, TensorAt ops 7 (layout30 r zMax rice).terminal.rounds c) ∧
      OffloadAt ops 0 (fold30 r 0) (SetupPlan.ofBatch (Key30.root r) (groupOrder 3)
        (fold30 r 0).low 3) Key30.setupVars (fusedIndex (layout30 r zMax rice) 0) ∧
      ∀ ℓ : Fin 7, FusedAt ops ℓ (fold30 r ℓ) (fusedIndex (layout30 r zMax rice) ℓ) ∧
        (∃ c, RangeAt ops ℓ (fold30 r ℓ) c) ∧
        ∀ n d, (fold30 r ℓ).trace = some (n, d) → ∃ c, TensorAt ops ℓ n c := by
  have erased : ops.toList.map Op.erase = skeleton (layout30 r 0 0) := by
    rw [script_erase scripted, skeleton30]
  have shaped := skeleton30_shaped member
  refine ⟨erased_ends erased (shaped 0).2.1, erased_last erased,
    erased_tensor erased (shaped 0).2.2, ?_, fun ℓ => ?_⟩
  · rw [fusedIndex30]
    exact (erased_level erased (layout30_fold r 0 0 0) (shaped 0).1).2.2.2 _ _
      (fold30_root member).1 (fold30_root member).2
  · rw [fusedIndex30]
    have level := erased_level erased (layout30_fold r 0 0 ℓ) (shaped ℓ).1
    exact ⟨level.1, level.2.1, level.2.2.1⟩

/-- The schedule facts of an accepted proof of `layout30` on `pub.ops`. -/
theorem verifyProof_script30 {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ}
    {pub : Public} {proof : ByteArray} (layout : pub.layout = layout30 r zMax rice)
    (accepted : runProof pub proof = .verdict .accept) :
    script pub.layout pub.plan = .ok pub.ops ∧ endsFit pub.ops.zipIdx.toList = true ∧
      OffloadAt pub.ops 0 (fold30 r 0) (SetupPlan.ofBatch (Key30.root r) (groupOrder 3)
        (fold30 r 0).low 3) Key30.setupVars (fusedIndex pub.layout 0) ∧
      ∀ ℓ : Fin 7, FusedAt pub.ops ℓ (fold30 r ℓ) (fusedIndex pub.layout ℓ) := by
  have scripted := verifyProof_scripted accepted
  have facts := script30 member (layout ▸ scripted)
  refine ⟨scripted, facts.1, ?_, fun ℓ => ?_⟩
  · rw [layout]
    exact facts.2.2.2.1
  · rw [layout]
    exact (facts.2.2.2.2 ℓ).1

end NightstreamFPrime.Spec.Compression.Akita.Replay
