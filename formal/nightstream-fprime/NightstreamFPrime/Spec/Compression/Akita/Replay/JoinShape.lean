import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBits

/-! The schedule facts of the links of `key20.aks` as theorems for each script of `layout20`
(Akita revision `d50f700`; `Replay/ShapeErase.lean`, `Replay/ShapeLayout20.lean`;
`docs/akita-work/w16/design.md` section 4, track A, item 2).

- `isAbsorbPoint`, `isAbsorbValues`, `isAbsorbRecorded`: Bools for the absorbs of the links.
- `RootFrame`, `PackFrame`, `TensorFrame`: the indices and the frames of the absorbs of the root
  link, of the packing link from the level 0 to the level 1, and of the tensor reduction of the
  level `level + 1` for the claim of the level `level`: the hypotheses of
  `verifyProof_key20_rootLink`, `verifyProof_key20_packLink`, `verifyProof_tensored` and
  `verifyProof_key20_traceLink`.
- `rootShape20`, `packShape20`, `tensorShape20` are these facts on the skeleton of `layout20 0 0`
  (Bools, closed computations of the kernel). `script20_frames`: each script of
  `layout20 zMax rice` has the frames, at the `fused.end` of the levels (`fusedIndex`), at the
  tensor ends 385, 649, 908, 1210, 1513 and at the terminal 1521.
- `verifyProof_key20_rootLink20`, `verifyProof_key20_packLink20`, `verifyProof_key20_traceLink20`:
  the links with the layout as the only schedule hypothesis. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Matchers

/-- The absorb of the point of the public group `group` at `key`. -/
def isAbsorbPoint (key : Key) (group : ℕ) : Op → Bool
  | .absorb k (.point g) => k == key && g == group
  | _ => false

/-- The absorb of the values of the public groups at `key`. -/
def isAbsorbValues (key : Key) : Op → Bool
  | .absorb k .values => k == key
  | _ => false

/-- The absorb of the recorded values `keys` at `key`. -/
def isAbsorbRecorded (key : Key) (keys : List Key) : Op → Bool
  | .absorb k (.recorded ks) => k == key && ks == keys
  | _ => false

theorem isAbsorbPoint_sound {key : Key} {group : ℕ} {op : Op}
    (h : isAbsorbPoint key group op = true) : op = .absorb key (.point group) := by
  unfold isAbsorbPoint at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl
  · cases h

theorem isAbsorbValues_sound {key : Key} {op : Op} (h : isAbsorbValues key op = true) :
    op = .absorb key .values := by
  unfold isAbsorbValues at h
  split at h
  · simp only [beq_iff_eq] at h
    rw [h]
  · cases h

theorem isAbsorbRecorded_sound {key : Key} {keys : List Key} {op : Op}
    (h : isAbsorbRecorded key keys op = true) : op = .absorb key (.recorded keys) := by
  unfold isAbsorbRecorded at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl
  · cases h

theorem Op.erase_absorb {op : Op} {key : Key} {value : Value} (h : op.erase = .absorb key value) :
    op = .absorb key value := by
  cases op <;> first | exact h | cases h

end Matchers

section Transfer

variable {L : Layout} {ops : Array Op}

/-- An absorb of the skeleton is an absorb of the operations at the same index. -/
theorem erased_absorb (erased : ops.toList.map Op.erase = skeleton L) {c : ℕ} {key : Key}
    {value : Value} (at_ : (skeleton L)[c]? = some (.absorb key value)) :
    ops[c]? = some (.absorb key value) := by
  rw [← erased, List.getElem?_map] at at_
  obtain ⟨op, found, same⟩ := Option.map_eq_some_iff.mp at_
  rw [← Array.getElem?_toList, found, Op.erase_absorb same]

/-- An operation of the skeleton that a Bool accepts, with a soundness lemma for the Bool. -/
theorem erased_matched {p : Op → Bool} {op : Op} (sound : ∀ op', p op' = true → op' = op)
    {c : ℕ} (found : ((skeleton L)[c]?).any p = true) : (skeleton L)[c]? = some op := by
  rcases at_ : (skeleton L)[c]? with _ | op'
  · rw [at_] at found
    cases found
  · rw [at_] at found
    rw [sound op' found]

theorem joined_frees (erased : ops.toList.map Op.erase = skeleton L) (c k : ℕ) (K R : List Key)
    (shaped : frees (((skeleton L).zipIdx.take c).drop k) K R = true) :
    frees ((ops.zipIdx.toList.take c).drop k) K R = true := by
  rw [← frees_erase, List.map_drop, take_erase erased]
  exact shaped

theorem joined_frees_take (erased : ops.toList.map Op.erase = skeleton L) (c : ℕ) (K R : List Key)
    (shaped : frees ((skeleton L).zipIdx.take c) K R = true) :
    frees (ops.zipIdx.toList.take c) K R = true := by
  rw [← frees_erase, take_erase erased]
  exact shaped

theorem joined_drawnCount (erased : ops.toList.map Op.erase = skeleton L) (c : ℕ) (key : Key)
    {n : ℕ} (shaped : drawnCount key ((skeleton L).zipIdx.take c) = some n) :
    drawnCount key (ops.zipIdx.toList.take c) = some n := by
  rw [← drawnCount_erase, take_erase erased]
  exact shaped

theorem joined_roundsAt (erased : ops.toList.map Op.erase = skeleton L) (c : ℕ) (key : Key)
    (n d : ℕ) (shaped : roundsAt ((skeleton L).zipIdx.take c) key n d = true) :
    roundsAt (ops.zipIdx.toList.take c) key n d = true := by
  rw [← roundsAt_erase, take_erase erased]
  exact shaped

end Transfer

section Frames

/-- The schedule facts of the root link (`verifyProof_key20_rootLink`) before the `fused.end` at
`c`: three weights `row_coefficients`, the absorb of the point of each public group `g` at `3 + g`
and of the values at 6, with no other write of these keys before `c`. -/
structure RootFrame (ops : Array Op) (c : ℕ) : Prop where
  weights : drawnCount ⟨0, "row_coefficients", none⟩ (ops.zipIdx.toList.take c) = some 3
  pointAt : ∀ g : Fin 3,
    ops[3 + g.val]? = some (.absorb ⟨0, s!"root.point.{g.val}", none⟩ (.point g.val))
  pointFresh : ∀ g : Fin 3,
    frees (ops.zipIdx.toList.take (3 + g.val)) [⟨0, s!"root.point.{g.val}", none⟩] [] = true
  pointAfter : ∀ g : Fin 3, 3 + g.val + 1 ≤ c
  pointKept : ∀ g : Fin 3, frees ((ops.zipIdx.toList.take c).drop (3 + g.val + 1))
    [⟨0, s!"root.point.{g.val}", none⟩] [] = true
  openingsAt : ops[6]? = some (.absorb ⟨0, "openings", none⟩ .values)
  openingsFresh : frees (ops.zipIdx.toList.take 6) [⟨0, "openings", none⟩] [] = true
  openingsAfter : 6 + 1 ≤ c
  openingsKept : frees ((ops.zipIdx.toList.take c).drop (6 + 1)) [⟨0, "openings", none⟩] [] =
    true

/-- The schedule facts of the packing link (`verifyProof_key20_packLink`) from the `fused.end` of
the level 0 at `c` to the `fused.end` of the level 1 at `c'`: the absorbs of `point.0` at 166 and
of `openings` at 167 of the level 1, and the frames of their keys. -/
structure PackFrame (ops : Array Op) (c c' : ℕ) : Prop where
  pointAt : ops[166]? = some (.absorb ⟨1, "point.0", none⟩ (.recorded [⟨0, "fused.point", none⟩]))
  openingsAt : ops[167]? = some (.absorb ⟨1, "openings", none⟩
    (.recorded [⟨0, "witness_evaluation", none⟩]))
  pointBefore : c ≤ 166
  openingsBefore : c ≤ 167
  pointAfter : 166 + 1 ≤ c'
  openingsAfter : 167 + 1 ≤ c'
  pointHeld : frees ((ops.zipIdx.toList.take 166).drop c) [⟨0, "fused.point", none⟩] [] = true
  openingsHeld : frees ((ops.zipIdx.toList.take 167).drop c)
    [⟨0, "witness_evaluation", none⟩] [] = true
  pointFresh : frees (ops.zipIdx.toList.take 166) [⟨1, "point.0", none⟩] [] = true
  openingsFresh : frees (ops.zipIdx.toList.take 167) [⟨1, "openings", none⟩] [] = true
  pointKept : frees ((ops.zipIdx.toList.take c').drop (166 + 1)) [⟨1, "point.0", none⟩] [] = true
  openingsKept : frees ((ops.zipIdx.toList.take c').drop (167 + 1)) [⟨1, "openings", none⟩] [] =
    true

/-- The schedule facts of the tensor reduction of the level `level + 1` at `t` with `n` rounds for
the claim of the level `level` whose `fused.end` is at `c`, up to `c'` (`verifyProof_tensored`,
`verifyProof_key20_traceLink`). -/
structure TensorFrame (ops : Array Op) (level n c t c' : ℕ) : Prop where
  at_ : ops[t]? = some (.check ⟨level + 1, "tensor.end", none⟩
    (.tensor (level + 1) n ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩))
  rounds : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ n 2 = true
  before : c ≤ t
  after : t ≤ c'
  held : frees ((ops.zipIdx.toList.take t).drop c)
    [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true
  kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
    [⟨level + 1, "tensor.sumcheck", none⟩] = true

end Frames

section Skeleton

/-- The skeleton of `layout20` with its indices. -/
abbrev sk20 : List (Op × ℕ) := (skeleton (layout20 0 0)).zipIdx

theorem skeleton20_length : (skeleton (layout20 0 0)).length = 1522 := by decide +kernel

theorem fusedIndex20_values : fusedIndex (layout20 0 0) 0 = 165 ∧ fusedIndex (layout20 0 0) 1 = 316 ∧
    fusedIndex (layout20 0 0) 2 = 586 ∧ fusedIndex (layout20 0 0) 3 = 848 ∧
    fusedIndex (layout20 0 0) 4 = 1153 ∧ fusedIndex (layout20 0 0) 5 = 1456 := by
  decide +kernel

/-- The facts of `RootFrame` on the skeleton of `layout20`. -/
def rootShape20 : Bool :=
  drawnCount ⟨0, "row_coefficients", none⟩ (sk20.take 165) == some 3 &&
    (List.finRange 3).all (fun g =>
      ((skeleton (layout20 0 0))[3 + g.val]?).any
          (isAbsorbPoint ⟨0, s!"root.point.{g.val}", none⟩ g.val) &&
        frees (sk20.take (3 + g.val)) [⟨0, s!"root.point.{g.val}", none⟩] [] &&
        frees ((sk20.take 165).drop (3 + g.val + 1)) [⟨0, s!"root.point.{g.val}", none⟩] []) &&
    ((skeleton (layout20 0 0))[6]?).any (isAbsorbValues ⟨0, "openings", none⟩) &&
    frees (sk20.take 6) [⟨0, "openings", none⟩] [] &&
    frees ((sk20.take 165).drop (6 + 1)) [⟨0, "openings", none⟩] []

/-- The facts of `PackFrame` on the skeleton of `layout20`. -/
def packShape20 : Bool :=
  ((skeleton (layout20 0 0))[166]?).any
      (isAbsorbRecorded ⟨1, "point.0", none⟩ [⟨0, "fused.point", none⟩]) &&
    ((skeleton (layout20 0 0))[167]?).any
      (isAbsorbRecorded ⟨1, "openings", none⟩ [⟨0, "witness_evaluation", none⟩]) &&
    frees ((sk20.take 166).drop 165) [⟨0, "fused.point", none⟩] [] &&
    frees ((sk20.take 167).drop 165) [⟨0, "witness_evaluation", none⟩] [] &&
    frees (sk20.take 166) [⟨1, "point.0", none⟩] [] &&
    frees (sk20.take 167) [⟨1, "openings", none⟩] [] &&
    frees ((sk20.take 316).drop (166 + 1)) [⟨1, "point.0", none⟩] [] &&
    frees ((sk20.take 316).drop (167 + 1)) [⟨1, "openings", none⟩] []

/-- The facts of `TensorFrame` on the skeleton of `layout20`. -/
def tensorShape20 (level n c t c' : ℕ) : Bool :=
  ((skeleton (layout20 0 0))[t]?).any (isTensorEnd (level + 1) n) &&
    roundsAt (sk20.take t) ⟨level + 1, "tensor.sumcheck", none⟩ n 2 &&
    frees ((sk20.take t).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] &&
    frees ((sk20.take c').drop t) (tensorKeys level) [⟨level + 1, "tensor.sumcheck", none⟩] &&
    decide (c ≤ t) && decide (t ≤ c')

theorem skeleton20_root : rootShape20 = true := by decide +kernel

theorem skeleton20_pack : packShape20 = true := by decide +kernel

theorem skeleton20_tensor1 : tensorShape20 1 Key20.level₁.tail 316 385 586 = true := by
  decide +kernel

theorem skeleton20_tensor2 : tensorShape20 2 Key20.level₂.tail 586 649 848 = true := by
  decide +kernel

theorem skeleton20_tensor3 : tensorShape20 3 Key20.level₃.tail 848 908 1153 = true := by
  decide +kernel

theorem skeleton20_tensor4 : tensorShape20 4 Key20.level₄.tail 1153 1210 1456 = true := by
  decide +kernel

theorem skeleton20_tensor5 : tensorShape20 5 Key20.level₅.tail 1456 1513 1521 = true := by
  decide +kernel

end Skeleton

section Script

variable {ops : Array Op}

theorem tensorFrame_of (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {level n c t c' : ℕ} (shaped : tensorShape20 level n c t c' = true) :
    TensorFrame ops level n c t c' := by
  unfold tensorShape20 at shaped
  simp only [Bool.and_eq_true, decide_eq_true_eq] at shaped
  obtain ⟨⟨⟨⟨⟨found, rounds⟩, held⟩, kept⟩, before⟩, after⟩ := shaped
  have at_ := erased_check erased (erased_matched (fun _ h => isTensorEnd_sound h) found)
  simp only [Nat.add_sub_cancel] at at_
  exact ⟨at_, joined_roundsAt erased _ _ _ _ rounds, before, after,
    joined_frees erased _ _ _ _ held, joined_frees erased _ _ _ _ kept⟩

theorem rootFrame_of (erased : ops.toList.map Op.erase = skeleton (layout20 0 0)) :
    RootFrame ops 165 := by
  have shaped := skeleton20_root
  unfold rootShape20 at shaped
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_finRange,
    true_implies] at shaped
  obtain ⟨⟨⟨⟨weights, points⟩, openings⟩, fresh⟩, kept⟩ := shaped
  refine ⟨joined_drawnCount erased _ _ weights, fun g => ?_, fun g => ?_, fun g => ?_,
    fun g => ?_, erased_absorb erased (erased_matched (fun _ h => isAbsorbValues_sound h) openings),
    joined_frees_take erased _ _ _ fresh, by decide, joined_frees erased _ _ _ _ kept⟩
  · exact erased_absorb erased (erased_matched (fun _ h => isAbsorbPoint_sound h) (points g).1.1)
  · exact joined_frees_take erased _ _ _ (points g).1.2
  · fin_cases g <;> decide
  · exact joined_frees erased _ _ _ _ (points g).2

theorem packFrame_of (erased : ops.toList.map Op.erase = skeleton (layout20 0 0)) :
    PackFrame ops 165 316 := by
  have shaped := skeleton20_pack
  unfold packShape20 at shaped
  simp only [Bool.and_eq_true] at shaped
  obtain ⟨⟨⟨⟨⟨⟨⟨point, openings⟩, pointHeld⟩, openingsHeld⟩, pointFresh⟩, openingsFresh⟩,
    pointKept⟩, openingsKept⟩ := shaped
  exact ⟨erased_absorb erased (erased_matched (fun _ h => isAbsorbRecorded_sound h) point),
    erased_absorb erased (erased_matched (fun _ h => isAbsorbRecorded_sound h) openings),
    by decide, by decide, by decide, by decide, joined_frees erased _ _ _ _ pointHeld,
    joined_frees erased _ _ _ _ openingsHeld, joined_frees_take erased _ _ _ pointFresh,
    joined_frees_take erased _ _ _ openingsFresh, joined_frees erased _ _ _ _ pointKept,
    joined_frees erased _ _ _ _ openingsKept⟩

/-- The frames of the links of each script of `layout20`: the root link and the packing link at
the `fused.end` of the levels 0 and 1, and the tensor reduction of each level 2 to 6 at its
`tensor.end` (385, 649, 908, 1210, 1513) for the claim of the level before it, up to the
`fused.end` of the receiver or the terminal (the last operation). -/
theorem script20_frames {zMax rice : ℕ} {plan : Array Run}
    (scripted : script (layout20 zMax rice) plan = .ok ops) :
    RootFrame ops (fusedIndex (layout20 zMax rice) 0) ∧
      PackFrame ops (fusedIndex (layout20 zMax rice) 0) (fusedIndex (layout20 zMax rice) 1) ∧
      TensorFrame ops 1 Key20.level₁.tail (fusedIndex (layout20 zMax rice) 1) 385
        (fusedIndex (layout20 zMax rice) 2) ∧
      TensorFrame ops 2 Key20.level₂.tail (fusedIndex (layout20 zMax rice) 2) 649
        (fusedIndex (layout20 zMax rice) 3) ∧
      TensorFrame ops 3 Key20.level₃.tail (fusedIndex (layout20 zMax rice) 3) 908
        (fusedIndex (layout20 zMax rice) 4) ∧
      TensorFrame ops 4 Key20.level₄.tail (fusedIndex (layout20 zMax rice) 4) 1210
        (fusedIndex (layout20 zMax rice) 5) ∧
      TensorFrame ops 5 Key20.level₅.tail (fusedIndex (layout20 zMax rice) 5) 1513
        (ops.size - 1) ∧
      ops[ops.size - 1]? = some (.terminal 6) := by
  have erased : ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, skeleton20]
  have size : ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  obtain ⟨f0, f1, f2, f3, f4, f5⟩ := fusedIndex20_values
  simp only [fusedIndex20 zMax rice, f0, f1, f2, f3, f4, f5, size]
  exact ⟨rootFrame_of erased, packFrame_of erased, tensorFrame_of erased skeleton20_tensor1,
    tensorFrame_of erased skeleton20_tensor2, tensorFrame_of erased skeleton20_tensor3,
    tensorFrame_of erased skeleton20_tensor4, tensorFrame_of erased skeleton20_tensor5,
    size ▸ erased_last erased⟩

/-- The frames of the links of an accepted proof of `layout20` on `pub.ops`. -/
theorem verifyProof_frames20 {zMax rice : ℕ} {pub : Public} {proof : ByteArray}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) :
    RootFrame pub.ops (fusedIndex pub.layout 0) ∧
      PackFrame pub.ops (fusedIndex pub.layout 0) (fusedIndex pub.layout 1) ∧
      TensorFrame pub.ops 1 Key20.level₁.tail (fusedIndex pub.layout 1) 385
        (fusedIndex pub.layout 2) ∧
      TensorFrame pub.ops 2 Key20.level₂.tail (fusedIndex pub.layout 2) 649
        (fusedIndex pub.layout 3) ∧
      TensorFrame pub.ops 3 Key20.level₃.tail (fusedIndex pub.layout 3) 908
        (fusedIndex pub.layout 4) ∧
      TensorFrame pub.ops 4 Key20.level₄.tail (fusedIndex pub.layout 4) 1210
        (fusedIndex pub.layout 5) ∧
      TensorFrame pub.ops 5 Key20.level₅.tail (fusedIndex pub.layout 5) 1513
        (pub.ops.size - 1) ∧
      pub.ops[pub.ops.size - 1]? = some (.terminal 6) := by
  have scripted := verifyProof_scripted accepted
  rw [layout] at scripted ⊢
  exact script20_frames scripted

end Script

end NightstreamFPrime.Spec.Compression.Akita.Replay
