import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeErase
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! The schedule facts of each script of `layout20` as theorems (Akita revision `d50f700`;
`Replay/ShapeErase.lean`).

- `skeleton20_level0` to `skeleton20_level5`, `skeleton20_ends` and `skeleton20_tensor` are closed
  computations of the kernel on the skeleton of `layout20 0 0`. The folds and the skeleton do not
  depend on `zMax` and `rice` (`skeleton20`, `folds20`).
- `fold20 ℓ` is the fold of the level `ℓ` of `layout20` (`layout20_fold`).
- `script20`: each script of `layout20 zMax rice` has `endsFit`, the terminal as its last
  operation with its tensor reduction, and `FusedAt`, `RangeAt` and `TensorAt` at each level with
  the fold `fold20 ℓ`. `verifyProof_script20`: the same for `pub.ops` of an accepted proof. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

theorem skeleton20_level0 : levelShaped (layout20 0 0) 0 = true := by decide +kernel

theorem skeleton20_level1 : levelShaped (layout20 0 0) 1 = true := by decide +kernel

theorem skeleton20_level2 : levelShaped (layout20 0 0) 2 = true := by decide +kernel

theorem skeleton20_level3 : levelShaped (layout20 0 0) 3 = true := by decide +kernel

theorem skeleton20_level4 : levelShaped (layout20 0 0) 4 = true := by decide +kernel

theorem skeleton20_level5 : levelShaped (layout20 0 0) 5 = true := by decide +kernel

theorem skeleton20_ends : endsFit (skeleton (layout20 0 0)).zipIdx = true := by decide +kernel

theorem skeleton20_tensor :
    tensorShapeAt (skeleton (layout20 0 0)).zipIdx 6 (layout20 0 0).terminal.rounds = true := by
  decide +kernel

/-- The skeleton of `layout20` does not depend on `zMax` and `rice`. -/
theorem skeleton20 (zMax rice : ℕ) : skeleton (layout20 zMax rice) = skeleton (layout20 0 0) := rfl

theorem folds20 (zMax rice : ℕ) : (layout20 zMax rice).folds = (layout20 0 0).folds := rfl

theorem fusedIndex20 (zMax rice ℓ : ℕ) :
    fusedIndex (layout20 zMax rice) ℓ = fusedIndex (layout20 0 0) ℓ := rfl

theorem length20 : (layout20 0 0).folds.length = 6 := rfl

/-- The fold of the level `ℓ` of `layout20`. -/
def fold20 (ℓ : Fin 6) : Fold := (layout20 0 0).folds[ℓ.val]'(by rw [length20]; exact ℓ.isLt)

theorem layout20_fold (zMax rice : ℕ) (ℓ : Fin 6) :
    (layout20 zMax rice).folds[ℓ.val]? = some (fold20 ℓ) := by
  rw [folds20]
  exact List.getElem?_eq_getElem _

theorem skeleton20_shaped (ℓ : Fin 6) : levelShaped (layout20 0 0) ℓ = true := by
  match ℓ with
  | 0 => exact skeleton20_level0
  | 1 => exact skeleton20_level1
  | 2 => exact skeleton20_level2
  | 3 => exact skeleton20_level3
  | 4 => exact skeleton20_level4
  | 5 => exact skeleton20_level5

/-- The schedule facts of each script of `layout20`: `endsFit`, the terminal of the level 6 as
the last operation with its tensor reduction, and at each level `ℓ` the shapes at `fused.end` (at
`fusedIndex`), at the last range check and at the tensor reduction of a trace level, with the
fold `fold20 ℓ` of the layout. -/
theorem script20 {zMax rice : ℕ} {plan : Array Run} {ops : Array Op}
    (scripted : script (layout20 zMax rice) plan = .ok ops) :
    endsFit ops.zipIdx.toList = true ∧ ops[ops.size - 1]? = some (.terminal 6) ∧
      (∃ c, TensorAt ops 6 (layout20 zMax rice).terminal.rounds c) ∧
      ∀ ℓ : Fin 6, FusedAt ops ℓ (fold20 ℓ) (fusedIndex (layout20 zMax rice) ℓ) ∧
        (∃ c, RangeAt ops ℓ (fold20 ℓ) c) ∧
        ∀ n d, (fold20 ℓ).trace = some (n, d) → ∃ c, TensorAt ops ℓ n c := by
  have erased : ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, skeleton20]
  refine ⟨erased_ends erased skeleton20_ends, erased_last erased, erased_tensor erased skeleton20_tensor,
    fun ℓ => ?_⟩
  rw [fusedIndex20]
  have level := erased_level erased (layout20_fold 0 0 ℓ) (skeleton20_shaped ℓ)
  exact ⟨level.1, level.2.1, level.2.2.1⟩

/-- The schedule facts of an accepted proof of `layout20` on `pub.ops`. -/
theorem verifyProof_script20 {zMax rice : ℕ} {pub : Public} {proof : ByteArray}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) :
    script pub.layout pub.plan = .ok pub.ops ∧ endsFit pub.ops.zipIdx.toList = true ∧
      ∀ ℓ : Fin 6, FusedAt pub.ops ℓ (fold20 ℓ) (fusedIndex pub.layout ℓ) := by
  have scripted := verifyProof_scripted accepted
  have facts := script20 (layout ▸ scripted)
  refine ⟨scripted, facts.1, fun ℓ => ?_⟩
  rw [layout]
  exact (facts.2.2.2 ℓ).1

end NightstreamFPrime.Spec.Compression.Akita.Replay
