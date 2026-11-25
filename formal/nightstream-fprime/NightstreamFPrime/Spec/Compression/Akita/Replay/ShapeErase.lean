import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeScript
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunRange
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalDrawn

/-! The shapes of the replay read only the erased operations, so they are facts of the skeleton
of the layout (Akita revision `d50f700`; `Replay/ShapeScript.lean`).

- `foldShape_erase`, `offloadShape_erase`, `gameShape_erase`, `drawnShape_erase`,
  `rangeShape_erase`, `roundsAt_erase` and `endsFit_erase`: each shape has the same value on the
  operations with `Op.erase`.
- `zipIdx_erase`: the erased operations of a script with their indices are those of the
  skeleton. `script_check`: a check of the skeleton is a check of the script at the same index.
- `fusedIndex L ℓ` is the index of `fused.end` of the level `ℓ` in the skeleton, and
  `skeleton_fused` the check there. `skeleton_terminal`: the last operation is the terminal.
- `levelShaped L ℓ` is a Bool on the skeleton: the shapes at `fused.end` of the level `ℓ`, at
  its last range check and at its tensor reduction. `erased_level`: on operations whose erased
  forms are the skeleton, it gives `FusedAt`, `RangeAt` and `TensorAt`; `erased_ends` and
  `erased_last` give `endsFit` and the terminal at the end.
- `Public.ops` is the script of a public input; `verifyProof_scripted`: an accepted proof has it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open Replayer

section Erase

theorem Op.writes_erase (op : Op) : op.erase.writes = op.writes := by cases op <;> rfl

theorem Op.ends_erase (op : Op) : op.erase.ends = op.ends := by cases op <;> rfl

theorem Op.drawsInto_erase (key : Key) (op : Op) : op.erase.drawsInto key = op.drawsInto key := by
  cases op <;> rfl

theorem Op.extsInto_erase (key : Key) (op : Op) : op.erase.extsInto key = op.extsInto key := by
  cases op <;> rfl

theorem Op.drawsGroup_erase (level group : ℕ) (op : Op) :
    op.erase.drawsGroup level group = op.drawsGroup level group := by
  cases op <;> rfl

theorem isRangeCheck_erase (level : ℕ) (F : Fold) (op : Op) :
    isRangeCheck level F op.erase = isRangeCheck level F op := by
  cases op <;> rfl

theorem isFusedClaim_erase (level : ℕ) (F : Fold) (op : Op) :
    isFusedClaim level F op.erase = isFusedClaim level F op := by
  cases op <;> rfl

theorem isPlainEnd_erase (key : Key) (n : ℕ) (op : Op) :
    isPlainEnd key n op.erase = isPlainEnd key n op := by
  cases op <;> rfl

theorem isStage3End_erase (level : ℕ) (plan : SetupPlan) (op : Op) :
    isStage3End level plan op.erase = isStage3End level plan op := by
  cases op <;> rfl

theorem isSetupClaim_erase (level : ℕ) (op : Op) :
    isSetupClaim level op.erase = isSetupClaim level op := by
  cases op <;> rfl

theorem isEndOf_erase (key : Key) (n : ℕ) (op : Op) : isEndOf key n op.erase = isEndOf key n op := by
  cases op <;> rfl

/-- A pair of an operation and its index with the operation erased. -/
abbrev eraseAt : Op × ℕ → Op × ℕ := Prod.map Op.erase id

theorem frees_erase (l : List (Op × ℕ)) (K R : List Key) :
    frees (l.map eraseAt) K R = frees l K R := by
  simp only [frees, List.forall_mem_map, Prod.map_fst, Op.writes_erase, Op.ends_erase]

theorem drawnCount_erase (key : Key) : ∀ l : List (Op × ℕ),
    drawnCount key (l.map eraseAt) = drawnCount key l
  | [] => rfl
  | x :: l => by
      simp only [List.map_cons, drawnCount, Prod.map_fst, Op.drawsInto_erase, drawnCount_erase key l]

theorem extsCount_erase (key : Key) : ∀ l : List (Op × ℕ),
    extsCount key (l.map eraseAt) = extsCount key l
  | [] => rfl
  | x :: l => by
      simp only [List.map_cons, extsCount, Prod.map_fst, Op.extsInto_erase, extsCount_erase key l]

theorem squeezeOf_erase (level group : ℕ) : ∀ l : List (Op × ℕ),
    squeezeOf level group (l.map eraseAt) = squeezeOf level group l
  | [] => rfl
  | x :: l => by
      simp only [List.map_cons, squeezeOf, Prod.map_fst, Op.drawsGroup_erase,
        squeezeOf_erase level group l]

theorem roundsShape_erase (l : List (Op × ℕ)) (key : Key) (n d e : ℕ) :
    roundsShape (l.map eraseAt) key n d e = roundsShape l key n d e := by
  simp only [roundsShape, List.getElem?_map, Option.any_map, Prod.map_fst, isEndOf_erase,
    ← List.map_take, ← List.map_drop, frees_erase, drawnCount_erase, extsCount_erase]

theorem roundsAt_erase (l : List (Op × ℕ)) (key : Key) (n d : ℕ) :
    roundsAt (l.map eraseAt) key n d = roundsAt l key n d := by
  simp only [roundsAt, List.findIdx?_map, Function.comp_def, Prod.map_fst, isEndOf_erase,
    roundsShape_erase]

theorem foldShape_erase (l : List (Op × ℕ)) (level : ℕ) (F : Fold) (r a e : ℕ) :
    foldShape (l.map eraseAt) level F r a e = foldShape l level F r a e := by
  simp only [foldShape, List.getElem?_map, Option.any_map, Prod.map_fst, isRangeCheck_erase,
    isFusedClaim_erase, isPlainEnd_erase, ← List.map_take, ← List.map_drop, frees_erase,
    drawnCount_erase]

theorem offloadShape_erase (l : List (Op × ℕ)) (level : ℕ) (F : Fold) (plan : SetupPlan)
    (r a e b t : ℕ) :
    offloadShape (l.map eraseAt) level F plan r a e b t = offloadShape l level F plan r a e b t := by
  simp only [offloadShape, foldShape_erase, List.getElem?_map, Option.any_map, Prod.map_fst,
    isSetupClaim_erase, isStage3End_erase, ← List.map_take, ← List.map_drop, frees_erase]

theorem gameShape_erase (l : List (Op × ℕ)) (level : ℕ) (F : Fold) :
    gameShape (l.map eraseAt) level F = gameShape l level F := by
  simp only [gameShape, drawnCount_erase, roundsAt_erase, extsCount_erase]

theorem rangeShape_erase (l : List (Op × ℕ)) (level : ℕ) (F : Fold) :
    rangeShape (l.map eraseAt) level F = rangeShape l level F := by
  simp only [rangeShape, drawnCount_erase, roundsAt_erase]

theorem drawnShape_erase (l : List (Op × ℕ)) (level : ℕ) (groups : List FoldGroup) :
    drawnShape (l.map eraseAt) level groups = drawnShape l level groups := by
  simp only [drawnShape, squeezeOf_erase]

theorem endFits_erase (pre : List (Op × ℕ)) (op : Op) :
    endFits (pre.map eraseAt) op.erase = endFits pre op := by
  cases op <;> simp only [Op.erase, endFits, drawnCount_erase]

theorem endsFit_erase (l : List (Op × ℕ)) : endsFit (l.map eraseAt) = endsFit l := by
  simp only [endsFit, List.length_map, List.getElem?_map, Option.map_map, Function.comp_def,
    Prod.map_fst, ← List.map_take, endFits_erase]

end Erase

section Script

variable {L : Layout} {ops : Array Op}

theorem zipIdx_erase (erased : ops.toList.map Op.erase = skeleton L) :
    ops.zipIdx.toList.map eraseAt = (skeleton L).zipIdx := by
  rw [Array.toList_zipIdx, ← erased, List.zipIdx_map]

theorem size_skeleton (erased : ops.toList.map Op.erase = skeleton L) :
    ops.size = (skeleton L).length := by
  rw [← erased, List.length_map, Array.length_toList]

theorem Op.erase_check {op : Op} {key : Key} {c : Check} (h : op.erase = .check key c) :
    op = .check key c := by
  cases op <;> first | exact h | cases h

theorem Op.erase_terminal {op : Op} {level : ℕ} (h : op.erase = .terminal level) :
    op = .terminal level := by
  cases op <;> first | exact h | cases h

/-- A check of the skeleton is a check of the operations at the same index. -/
theorem erased_check (erased : ops.toList.map Op.erase = skeleton L) {c : ℕ} {key : Key}
    {check : Check} (at_ : (skeleton L)[c]? = some (.check key check)) :
    ops[c]? = some (.check key check) := by
  rw [← erased, List.getElem?_map] at at_
  obtain ⟨op, found, same⟩ := Option.map_eq_some_iff.mp at_
  rw [← Array.getElem?_toList, found, Op.erase_check same]

theorem erased_terminal (erased : ops.toList.map Op.erase = skeleton L) {c level : ℕ}
    (at_ : (skeleton L)[c]? = some (.terminal level)) : ops[c]? = some (.terminal level) := by
  rw [← erased, List.getElem?_map] at at_
  obtain ⟨op, found, same⟩ := Option.map_eq_some_iff.mp at_
  rw [← Array.getElem?_toList, found, Op.erase_terminal same]

theorem take_erase (erased : ops.toList.map Op.erase = skeleton L) (c : ℕ) :
    (ops.zipIdx.toList.take c).map eraseAt = (skeleton L).zipIdx.take c := by
  rw [List.map_take, zipIdx_erase erased]

end Script

section Index

/-- The levels of the skeleton: each fold with the fold before it and its index. -/
abbrev blocks (L : Layout) : List ((Fold × Option Fold) × ℕ) :=
  (L.folds.zip (none :: L.folds.map some)).zipIdx

/-- The index of `fused.end` of the level `ℓ` in the skeleton of `L`. -/
def fusedIndex (L : Layout) (ℓ : ℕ) : ℕ := (((blocks L).take (ℓ + 1)).flatMap Skeleton.level).length - 1

theorem Skeleton.level_last (x : (Fold × Option Fold) × ℕ) :
    ∃ front, Skeleton.level x =
      front ++ [.check ⟨x.2, "fused.end", none⟩ (.fused x.2 x.1.1)] := by
  refine List.getLast?_eq_some_iff.mp ?_
  simp [Skeleton.level, Skeleton.fold, Skeleton.close, List.getLast?_cons]

theorem blocks_getElem {L : Layout} {ℓ : ℕ} {F : Fold} (fold : L.folds[ℓ]? = some F) :
    ∃ before, (blocks L)[ℓ]? = some ((F, before), ℓ) := by
  have below : ℓ < L.folds.length := (List.getElem?_eq_some_iff.mp fold).1
  have inside : ℓ < (none :: L.folds.map some).length := by simp; omega
  refine ⟨(none :: L.folds.map some)[ℓ], ?_⟩
  have zipped : (L.folds.zip (none :: L.folds.map some))[ℓ]? =
      some (F, (none :: L.folds.map some)[ℓ]) :=
    List.getElem?_zip_eq_some.mpr ⟨fold, List.getElem?_eq_getElem inside⟩
  rw [List.getElem?_zipIdx, zipped]
  simp

/-- The check `fused.end` of the fold `F` of the level `ℓ` in the skeleton. -/
theorem skeleton_fused {L : Layout} {ℓ : ℕ} {F : Fold} (fold : L.folds[ℓ]? = some F) :
    (skeleton L)[fusedIndex L ℓ]? = some (.check ⟨ℓ, "fused.end", none⟩ (.fused ℓ F)) := by
  obtain ⟨before, found⟩ := blocks_getElem fold
  obtain ⟨front, last⟩ := Skeleton.level_last ((F, before), ℓ)
  have split : ((blocks L).take (ℓ + 1)).flatMap Skeleton.level =
      ((blocks L).take ℓ).flatMap Skeleton.level ++ front ++
        [.check ⟨ℓ, "fused.end", none⟩ (.fused ℓ F)] := by
    rw [List.take_add_one, found, Option.toList_some, List.flatMap_append, List.flatMap_singleton,
      last, List.append_assoc]
  have whole : skeleton L = ((blocks L).take (ℓ + 1)).flatMap Skeleton.level ++
      (((blocks L).drop (ℓ + 1)).flatMap Skeleton.level ++
        Skeleton.terminal L.folds.length L.terminal) := by
    rw [← List.append_assoc, ← List.flatMap_append, List.take_append_drop]
    rfl
  have index : fusedIndex L ℓ = (((blocks L).take ℓ).flatMap Skeleton.level ++ front).length := by
    unfold fusedIndex
    rw [split, List.length_append, List.length_singleton, Nat.add_sub_cancel]
  rw [index, whole, split, List.getElem?_append_left (by simp),
    List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]
  rfl

/-- The last operation of the skeleton is the terminal of the level `L.folds.length`. -/
theorem skeleton_terminal (L : Layout) :
    (skeleton L)[(skeleton L).length - 1]? = some (.terminal L.folds.length) := by
  rw [← List.getLast?_eq_getElem?]
  simp [skeleton, Skeleton.terminal, List.getLast?_cons]

end Index

section Shapes

/-- `foldShape` at the first last range check, absorb of `fused.claim` and end of `fused` of the
level. -/
def foldShapeAt (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) : Bool :=
  match pre.findIdx? (fun x => isRangeCheck level F x.1),
    pre.findIdx? (fun x => isFusedClaim level F x.1),
    pre.findIdx? (fun x => isPlainEnd ⟨level, "fused", none⟩ F.vars x.1) with
  | some r, some a, some e => foldShape pre level F r a e
  | _, _, _ => false

theorem foldShapeAt_sound {pre : List (Op × ℕ)} {level : ℕ} {F : Fold}
    (h : foldShapeAt pre level F = true) : ∃ r a e, foldShape pre level F r a e = true := by
  unfold foldShapeAt at h
  split at h
  · exact ⟨_, _, _, h⟩
  · cases h

/-- `offloadShape` at the indices of `foldShapeAt`, the absorb of `stage3.claim` and the check
`stage3.end` of the plan. -/
def offloadShapeAt (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) (plan : SetupPlan) : Bool :=
  match pre.findIdx? (fun x => isRangeCheck level F x.1),
    pre.findIdx? (fun x => isFusedClaim level F x.1),
    pre.findIdx? (fun x => isPlainEnd ⟨level, "fused", none⟩ F.vars x.1),
    pre.findIdx? (fun x => isSetupClaim level x.1),
    pre.findIdx? (fun x => isStage3End level plan x.1) with
  | some r, some a, some e, some b, some t => offloadShape pre level F plan r a e b t
  | _, _, _, _, _ => false

theorem offloadShapeAt_sound {pre : List (Op × ℕ)} {level : ℕ} {F : Fold} {plan : SetupPlan}
    (h : offloadShapeAt pre level F plan = true) :
    ∃ r a e b t, offloadShape pre level F plan r a e b t = true := by
  unfold offloadShapeAt at h
  split at h
  · exact ⟨_, _, _, _, _, h⟩
  · cases h

/-- `rangeShape` before the first last range check of the level. -/
def rangeShapeAt (l : List (Op × ℕ)) (level : ℕ) (F : Fold) : Bool :=
  match l.findIdx? (fun x => isRangeCheck level F x.1) with
  | some c => rangeShape (l.take c) level F
  | none => false

/-- The check `tensor.end` of the level for the claim `(fused.point, witness_evaluation)` of the
level before. -/
def isTensorEnd (level count : ℕ) : Op → Bool
  | .check key (.tensor l n point opening) =>
      key == ⟨level, "tensor.end", none⟩ && l == level && n == count &&
        point == ⟨level - 1, "fused.point", none⟩ && opening == ⟨level - 1, "witness_evaluation", none⟩
  | _ => false

theorem isTensorEnd_sound {level count : ℕ} {op : Op} (h : isTensorEnd level count op = true) :
    op = .check ⟨level, "tensor.end", none⟩ (.tensor level count ⟨level - 1, "fused.point", none⟩
      ⟨level - 1, "witness_evaluation", none⟩) := by
  unfold isTensorEnd at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
    rfl
  · cases h

/-- `roundsAt` of the sum-check of the tensor reduction before the first `tensor.end` of the
level. -/
def tensorShapeAt (l : List (Op × ℕ)) (level count : ℕ) : Bool :=
  match l.findIdx? (fun x => isTensorEnd level count x.1) with
  | some c => roundsAt (l.take c) ⟨level, "tensor.sumcheck", none⟩ count 2
  | none => false

/-- The shapes of the skeleton at the level `ℓ`: at its `fused.end`, at its last range check, at a
level with evaluation trace at its tensor reduction, and at an offloaded level (Stage 3 with the
plan of its weight) `offloadShape` and the rounds of Stage 3. -/
def levelShaped (L : Layout) (ℓ : ℕ) : Bool :=
  match L.folds[ℓ]? with
  | none => false
  | some F =>
      foldShapeAt ((skeleton L).zipIdx.take (fusedIndex L ℓ)) ℓ F &&
        gameShape ((skeleton L).zipIdx.take (fusedIndex L ℓ)) ℓ F &&
        drawnShape ((skeleton L).zipIdx.take (fusedIndex L ℓ)) ℓ F.groups &&
        rangeShapeAt (skeleton L).zipIdx ℓ F &&
        (match F.trace with
          | some (n, _) => tensorShapeAt (skeleton L).zipIdx ℓ n
          | none => true) &&
        match F.stage3, F.setup with
        | some ν, some plan =>
            offloadShapeAt ((skeleton L).zipIdx.take (fusedIndex L ℓ)) ℓ F plan &&
              roundsAt ((skeleton L).zipIdx.take (fusedIndex L ℓ)) ⟨ℓ, "stage3", none⟩ ν 2
        | _, _ => true

/-- The shapes at `fused.end` of the fold `F` of `level` at the index `c` of the operations. -/
structure FusedAt (ops : Array Op) (level : ℕ) (F : Fold) (c : ℕ) : Prop where
  at_ : ops[c]? = some (.check ⟨level, "fused.end", none⟩ (.fused level F))
  shape : ∃ r a e, foldShape (ops.zipIdx.toList.take c) level F r a e = true
  game : gameShape (ops.zipIdx.toList.take c) level F = true
  drawn : drawnShape (ops.zipIdx.toList.take c) level F.groups = true

/-- The shapes at `fused.end` of the offloaded fold `F` of `level` at the index `c`, with the
plan `plan` of the weight of Stage 3 and its `ν` rounds. -/
structure OffloadAt (ops : Array Op) (level : ℕ) (F : Fold) (plan : SetupPlan) (ν c : ℕ) : Prop where
  shape : ∃ r a e b t, offloadShape (ops.zipIdx.toList.take c) level F plan r a e b t = true
  setup : roundsAt (ops.zipIdx.toList.take c) ⟨level, "stage3", none⟩ ν 2 = true

/-- The shape at the last range check of the fold `F` of `level` at the index `c`. -/
structure RangeAt (ops : Array Op) (level : ℕ) (F : Fold) (c : ℕ) : Prop where
  at_ : ∃ key, ops[c]? = some (.check key (rangeCheckOf level F))
  shape : rangeShape (ops.zipIdx.toList.take c) level F = true

/-- The shape at `tensor.end` of `level` with `count` rounds at the index `c`. -/
structure TensorAt (ops : Array Op) (level count : ℕ) (c : ℕ) : Prop where
  at_ : ops[c]? = some (.check ⟨level, "tensor.end", none⟩ (.tensor level count
    ⟨level - 1, "fused.point", none⟩ ⟨level - 1, "witness_evaluation", none⟩))
  shape : roundsAt (ops.zipIdx.toList.take c) ⟨level, "tensor.sumcheck", none⟩ count 2 = true

variable {L : Layout} {ops : Array Op}

theorem erased_tensor (erased : ops.toList.map Op.erase = skeleton L) {level count : ℕ}
    (shaped : tensorShapeAt (skeleton L).zipIdx level count = true) :
    ∃ c, TensorAt ops level count c := by
  unfold tensorShapeAt at shaped
  split at shaped
  · rename_i c found
    obtain ⟨inside, hit, -⟩ := List.findIdx?_eq_some_iff_getElem.mp found
    have below : c < (skeleton L).length := by simpa using inside
    refine ⟨c, erased_check erased ?_, ?_⟩
    · rw [List.getElem?_eq_getElem below,
        isTensorEnd_sound (op := (skeleton L)[c]) (by simpa using hit)]
    · rw [← roundsAt_erase, take_erase erased]
      exact shaped
  · cases shaped

theorem erased_range (erased : ops.toList.map Op.erase = skeleton L) {level : ℕ} {F : Fold}
    (shaped : rangeShapeAt (skeleton L).zipIdx level F = true) : ∃ c, RangeAt ops level F c := by
  unfold rangeShapeAt at shaped
  split at shaped
  · rename_i c found
    obtain ⟨inside, hit, -⟩ := List.findIdx?_eq_some_iff_getElem.mp found
    obtain ⟨key, same⟩ := isRangeCheck_sound (by simpa using hit)
    refine ⟨c, ⟨key, erased_check erased ?_⟩, ?_⟩
    · rw [List.getElem?_eq_getElem (by simpa using inside), ← same]
    · rw [← rangeShape_erase, take_erase erased]
      exact shaped
  · cases shaped

/-- The shapes of the operations at the level `ℓ` from `levelShaped` of the skeleton. -/
theorem erased_level (erased : ops.toList.map Op.erase = skeleton L) {ℓ : ℕ} {F : Fold}
    (fold : L.folds[ℓ]? = some F) (shaped : levelShaped L ℓ = true) :
    FusedAt ops ℓ F (fusedIndex L ℓ) ∧ (∃ c, RangeAt ops ℓ F c) ∧
      (∀ n d, F.trace = some (n, d) → ∃ c, TensorAt ops ℓ n c) ∧
      ∀ ν plan, F.stage3 = some ν → F.setup = some plan → OffloadAt ops ℓ F plan ν (fusedIndex L ℓ) := by
  unfold levelShaped at shaped
  rw [fold] at shaped
  simp only [Bool.and_eq_true] at shaped
  obtain ⟨⟨⟨⟨⟨fused, game⟩, drawn⟩, range⟩, tensor⟩, offload⟩ := shaped
  refine ⟨⟨erased_check erased (skeleton_fused fold), ?_, ?_, ?_⟩, erased_range erased range, ?_,
    ?_⟩
  · obtain ⟨r, a, e, shape⟩ := foldShapeAt_sound fused
    exact ⟨r, a, e, by rw [← foldShape_erase, take_erase erased]; exact shape⟩
  · rw [← gameShape_erase, take_erase erased]
    exact game
  · rw [← drawnShape_erase, take_erase erased]
    exact drawn
  · intro n d traced
    rw [traced] at tensor
    exact erased_tensor erased tensor
  · intro ν plan staged planned
    rw [staged, planned] at offload
    simp only [Bool.and_eq_true] at offload
    obtain ⟨r, a, e, b, t, shape⟩ := offloadShapeAt_sound offload.1
    exact ⟨⟨r, a, e, b, t, by rw [← offloadShape_erase, take_erase erased]; exact shape⟩,
      by rw [← roundsAt_erase, take_erase erased]; exact offload.2⟩

theorem erased_ends (erased : ops.toList.map Op.erase = skeleton L)
    (shaped : endsFit (skeleton L).zipIdx = true) : endsFit ops.zipIdx.toList = true := by
  rw [← endsFit_erase, zipIdx_erase erased]
  exact shaped

/-- The last operation is the terminal. -/
theorem erased_last (erased : ops.toList.map Op.erase = skeleton L) :
    ops[ops.size - 1]? = some (.terminal L.folds.length) := by
  rw [size_skeleton erased]
  exact erased_terminal erased (skeleton_terminal L)

end Shapes

/-- The operations of the script of `pub`, and none when its layout does not take its plan. -/
def Public.ops (pub : Public) : Array Op :=
  match script pub.layout pub.plan with
  | .ok ops => ops
  | .error _ => #[]

/-- An accepted proof has a public input whose layout takes its plan, with the operations
`pub.ops`. -/
theorem verifyProof_scripted {pub : Public} {proof : ByteArray}
    (accepted : runProof pub proof = .verdict .accept) : script pub.layout pub.plan = .ok pub.ops := by
  unfold Public.ops
  cases found : script pub.layout pub.plan with
  | error reason =>
      rw [verifyProof_plan pub proof found] at accepted
      cases accepted
  | ok ops => rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
