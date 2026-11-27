import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingLevel
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Match

/-! The quotient entries of the join of a fold with opening groups (ePrint 2026/1983, (348);
`Batch.flatRelation` of `Extract/BatchRows.lean`).

- `place_symm_code`: the index of row `i` of the group `g` of a family is
  `codeOffset count g + i`. `sum_before_place` splits a sum over the rows before such a row into
  the groups before `g` in the order `groupOrder` and the rows of `g` before `i`.
- `joinEntry_append_left`, `joinEntry_append_right`, `joinEntry_gather` and `joinEntry_plain`:
  `quotientEntry` of `Join.append`, `Join.gather` and `Join.plain`. The entries of the field rows
  of a group in a gather start after the entries of the groups before it in the order
  `groupOrder`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Order

variable {n : ℕ}

/-- The index of the row `i` of the group `g` in a family whose groups are in the order
`groupOrder`. -/
theorem place_symm_code (count : Fin n → ℕ) (g : Fin n) (i : Fin (count g)) :
    ((place count).symm ⟨g, i⟩ : ℕ) = codeOffset count g + i := by
  obtain ⟨k, rfl⟩ := (groupOrder n).surjective g
  rw [place_symm_val, codeOffset, Equiv.symm_apply_apply, offset_rowsBefore]

theorem codeOffset_le_of_lt (count : Fin n → ℕ) {j g : Fin n}
    (before : (groupOrder n).symm j < (groupOrder n).symm g) :
    codeOffset count j + count j ≤ codeOffset count g := by
  have step := offset_mono (fun k => count (groupOrder n k))
    (show ((groupOrder n).symm j : ℕ) + 1 ≤ (groupOrder n).symm g from before)
  rw [offset_succ, Equiv.apply_symm_apply] at step
  exact step

/-- A sum over the rows before the row `k` of the group `g` of a family: the rows of the groups
before `g` in the order `groupOrder`, then the rows of `g` before `k`. -/
theorem sum_before_place {M : Type} [AddCommMonoid M] (count : Fin n → ℕ)
    (F : (Σ g, Fin (count g)) → M) (g : Fin n) (k : Fin (count g)) :
    ∑ K : Fin (∑ h, count h), (if (K : ℕ) < ((place count).symm ⟨g, k⟩ : ℕ) then
        F (place count K) else 0) =
      ∑ j : Fin n, (if (groupOrder n).symm j < (groupOrder n).symm g then ∑ i, F ⟨j, i⟩
        else 0) + ∑ i : Fin (count g), if (i : ℕ) < k then F ⟨g, i⟩ else 0 := by
  rw [← (place count).symm.sum_comp, Fintype.sum_sigma]
  simp only [Equiv.apply_symm_apply, place_symm_code]
  have each : ∀ j : Fin n, (∑ i : Fin (count j), if codeOffset count j + (i : ℕ) <
      codeOffset count g + k then F ⟨j, i⟩ else 0) =
        (if (groupOrder n).symm j < (groupOrder n).symm g then ∑ i, F ⟨j, i⟩ else 0) +
          if j = g then ∑ i : Fin (count j), (if (i : ℕ) < k then F ⟨j, i⟩ else 0) else 0 := by
    intro j
    rcases lt_trichotomy ((groupOrder n).symm j) ((groupOrder n).symm g) with before | same | after
    · have le := codeOffset_le_of_lt count before
      have other : j ≠ g := fun equal => by rw [equal] at before; exact lt_irrefl _ before
      rw [ite_eq_left before, ite_eq_right other, add_zero]
      exact Finset.sum_congr rfl fun i _ => ite_eq_left (by omega)
    · obtain rfl := (groupOrder n).symm.injective same
      rw [ite_eq_right (lt_irrefl _), ite_eq_left rfl, zero_add]
      exact Finset.sum_congr rfl fun i _ => by simp only [Nat.add_lt_add_iff_left]
    · have le := codeOffset_le_of_lt count after
      have other : j ≠ g := fun equal => by rw [equal] at after; exact lt_irrefl _ after
      rw [ite_eq_right (lt_asymm after), ite_eq_right other, add_zero]
      exact Finset.sum_eq_zero fun i _ => ite_eq_right (by have := k.isLt; omega)
  rw [Finset.sum_congr rfl fun j _ => each j, Finset.sum_add_distrib, Finset.sum_ite_eq']
  rw [ite_eq_left (Finset.mem_univ g)]

/-- `codeOffset` as a sum over the groups before `g` in the order `groupOrder`. -/
theorem codeOffset_eq_sum (size : Fin n → ℕ) (g : Fin n) :
    codeOffset size g =
      ∑ j : Fin n, if (groupOrder n).symm j < (groupOrder n).symm g then size j else 0 := by
  unfold codeOffset offset
  exact Fintype.sum_equiv (groupOrder n) _ _ fun k => by
    simp only [Equiv.symm_apply_apply, Fin.lt_def]

end Order

section Entries

variable {digits : ℕ}

theorem castAdd_ne_natAdd {p q : ℕ} (x : Fin p) (y : Fin q) :
    Fin.castAdd q x ≠ Fin.natAdd p y := fun same => by
  have values := congrArg Fin.val same
  simp only [Fin.val_castAdd, Fin.val_natAdd] at values
  omega

section Append

variable {m m' c c' : ℕ} (sa : Fin m → Fin c) (sb : Fin m' → Fin c') (degree : Fin (m + m') → ℕ)

/-- The slots of `Join.append`: the field rows of the left join, then those of the right join. -/
def appendSlot : Fin (m + m') → Fin (c + c') :=
  Fin.addCases (fun i => Fin.castAdd c' (sa i)) fun j => Fin.natAdd c (sb j)

theorem appendSlot_left (r : Fin m) : appendSlot sa sb (Fin.castAdd m' r) = Fin.castAdd c' (sa r) :=
  Fin.addCases_left _

theorem appendSlot_right (r : Fin m') : appendSlot sa sb (Fin.natAdd m r) = Fin.natAdd c (sb r) :=
  Fin.addCases_right _

theorem slotWidth_append_left (k : Fin c) :
    slotWidth (appendSlot sa sb) degree (Fin.castAdd c' k) =
      slotWidth sa (fun r => degree (Fin.castAdd m' r)) k := by
  unfold slotWidth slotDegree
  rw [Fin.sum_univ_add, Finset.sum_eq_zero (s := Finset.univ) fun r _ => ite_eq_right fun same =>
    castAdd_ne_natAdd _ _ ((appendSlot_right sa sb r).symm.trans same).symm, add_zero]
  exact Finset.sum_congr rfl fun r _ => by
    simp only [appendSlot_left, Fin.castAdd_inj]

theorem slotWidth_append_right (k : Fin c') :
    slotWidth (appendSlot sa sb) degree (Fin.natAdd c k) =
      slotWidth sb (fun r => degree (Fin.natAdd m r)) k := by
  unfold slotWidth slotDegree
  rw [Fin.sum_univ_add, Finset.sum_eq_zero (s := Finset.univ) fun r _ => ite_eq_right fun same =>
    castAdd_ne_natAdd _ _ ((appendSlot_left sa sb r).symm.trans same), zero_add]
  exact Finset.sum_congr rfl fun r _ => by
    simp only [appendSlot_right, Fin.natAdd_inj]

/-- The entry of a ring row of the left join of `Join.append` is its entry in the left join. -/
theorem joinEntry_append_left (r : Fin m) (j x : ℕ) :
    quotientEntry (appendSlot sa sb) degree digits (Fin.castAdd m' r) j x =
      quotientEntry sa (fun r => degree (Fin.castAdd m' r)) digits r j x := by
  unfold quotientEntry
  rw [appendSlot_left, slotWidth_append_left, Fin.val_castAdd, Fin.val_castAdd, offset_castAdd,
    offset_castAdd]
  congr 2
  · unfold slotCells
    simp only [slotWidth_append_left]
  · unfold slotDegree
    simp only [appendSlot_left, Fin.castAdd_inj]

/-- The entry of a ring row of the right join of `Join.append` is its entry in the right join
after the entries of the left join. -/
theorem joinEntry_append_right (r : Fin m') (j x : ℕ) :
    quotientEntry (appendSlot sa sb) degree digits (Fin.natAdd m r) j x =
      digits * ∑ i : Fin m, degree (Fin.castAdd m' i) +
        quotientEntry sb (fun r => degree (Fin.natAdd m r)) digits r j x := by
  unfold quotientEntry
  rw [appendSlot_right, slotWidth_append_right, Fin.val_natAdd, Fin.val_natAdd, offset_natAdd,
    offset_natAdd]
  have left : ∑ i : Fin c, slotCells (appendSlot sa sb) degree digits (Fin.castAdd c' i) =
      digits * ∑ i : Fin m, degree (Fin.castAdd m' i) := by
    unfold slotCells
    simp only [slotWidth_append_left]
    rw [← Finset.mul_sum]
    congr 1
    unfold slotWidth slotDegree
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun r _ => by rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]
  have none : ∑ i : Fin m, slotDegree (appendSlot sa sb) degree (Fin.natAdd c (sb r))
      (Fin.castAdd m' i) = 0 :=
    Finset.sum_eq_zero fun i _ => ite_eq_right fun same =>
      castAdd_ne_natAdd _ _ ((appendSlot_left sa sb i).symm.trans same)
  rw [left, none, zero_add, Nat.add_assoc]
  congr 2
  · unfold slotCells
    simp only [slotWidth_append_right]
  · unfold slotDegree
    simp only [appendSlot_right, Fin.natAdd_inj]

end Append

section Plain

variable {m : ℕ} (degree : Fin m → ℕ)

/-- The entry of a ring row of `Join.plain`: the entries of the rows before it, then digit `a`
of coefficient `j`. -/
theorem joinEntry_plain (r : Fin m) (j x : ℕ) :
    quotientEntry (id : Fin m → Fin m) degree digits r j x =
      digits * offset degree r + (x * degree r + j) := by
  have width : ∀ k : Fin m, slotWidth (id : Fin m → Fin m) degree k = degree k := fun k => by
    unfold slotWidth slotDegree
    simp only [id_eq]
    rw [Finset.sum_ite_eq' Finset.univ k, ite_eq_left (Finset.mem_univ _)]
  have none : offset (slotDegree (id : Fin m → Fin m) degree r) r = 0 :=
    Finset.sum_eq_zero fun i _ => by
      unfold slotDegree
      simp only [id_eq]
      by_cases same : i = r
      · subst same
        exact ite_eq_right (lt_irrefl _)
      · rw [ite_eq_right same, ite_self]
  unfold quotientEntry
  simp only [id_eq]
  rw [none, zero_add, width]
  congr 1
  unfold offset slotCells
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by rw [width, mul_ite, mul_zero]

end Plain

section Gather

variable {n : ℕ} {size c : Fin n → ℕ} (slots : ∀ g, Fin (size g) → Fin (c g))
  (degree : Fin (∑ g, size g) → ℕ)

/-- The slots of `Join.gather`: the field rows of the groups in the order `groupOrder`. -/
def gatherSlot (r : Fin (∑ g, size g)) : Fin (∑ g, c g) :=
  (place c).symm ⟨(place size r).1, slots (place size r).1 (place size r).2⟩

theorem gatherSlot_place (g : Fin n) (i : Fin (size g)) :
    gatherSlot slots ((place size).symm ⟨g, i⟩) = (place c).symm ⟨g, slots g i⟩ := by
  unfold gatherSlot
  rw [Equiv.apply_symm_apply]

theorem join_gather_basis {E : Type} [CommRing E] (f : ∀ g, Join E (size g)) (g : Fin n)
    (i : Fin (size g)) : (Join.gather f).basis ((place size).symm ⟨g, i⟩) = (f g).basis i := by
  show (f (place size ((place size).symm ⟨g, i⟩)).1).basis
    (place size ((place size).symm ⟨g, i⟩)).2 = _
  rw [Equiv.apply_symm_apply]

theorem slotWidth_gather (g : Fin n) (k : Fin (c g)) :
    slotWidth (gatherSlot slots) degree ((place c).symm ⟨g, k⟩) =
      slotWidth (slots g) (fun i => degree ((place size).symm ⟨g, i⟩)) k := by
  unfold slotWidth slotDegree
  rw [← (place size).symm.sum_comp, Fintype.sum_sigma]
  simp only [gatherSlot_place, Equiv.apply_eq_iff_eq]
  rw [Finset.sum_eq_single g]
  · exact Finset.sum_congr rfl fun i _ => by simp only [Sigma.mk.inj_iff, heq_eq_eq, true_and]
  · intro h _ other
    exact Finset.sum_eq_zero fun i _ => ite_eq_right fun same => other (Sigma.mk.inj_iff.mp same).1
  · exact fun absent => absurd (Finset.mem_univ g) absent

/-- The entry of the ring row `i` of the group `g` of a gather: the entries of the field rows of
the groups before `g` in the order `groupOrder`, then its entry in the join of `g`. -/
theorem joinEntry_gather (g : Fin n) (i : Fin (size g)) (j x : ℕ) :
    quotientEntry (gatherSlot slots) degree digits ((place size).symm ⟨g, i⟩) j x =
      digits * codeOffset (fun h => ∑ i, degree ((place size).symm ⟨h, i⟩)) g +
        quotientEntry (slots g) (fun i => degree ((place size).symm ⟨g, i⟩)) digits i j x := by
  have cells : offset (slotCells (gatherSlot slots) degree digits)
      ((place c).symm ⟨g, slots g i⟩ : ℕ) =
        digits * codeOffset (fun h => ∑ i, degree ((place size).symm ⟨h, i⟩)) g +
          offset (slotCells (slots g) (fun i => degree ((place size).symm ⟨g, i⟩)) digits)
            (slots g i) := by
    have moved := sum_before_place c
      (fun σ => slotCells (gatherSlot slots) degree digits ((place c).symm σ)) g (slots g i)
    simp only [Equiv.symm_apply_apply] at moved
    unfold offset
    rw [moved]
    congr 1
    · rw [codeOffset_eq_sum, Finset.mul_sum]
      refine Finset.sum_congr rfl fun h _ => ?_
      rw [mul_ite, mul_zero]
      congr 1
      unfold slotCells
      simp only [slotWidth_gather]
      rw [← Finset.mul_sum]
      congr 1
      unfold slotWidth slotDegree
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun r _ => by
        rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]
    · exact Finset.sum_congr rfl fun k _ => by
        unfold slotCells
        rw [slotWidth_gather]
  have degrees : offset (slotDegree (gatherSlot slots) degree ((place c).symm ⟨g, slots g i⟩))
      ((place size).symm ⟨g, i⟩ : ℕ) =
        offset (slotDegree (slots g) (fun i => degree ((place size).symm ⟨g, i⟩)) (slots g i))
          i := by
    have moved := sum_before_place size
      (fun σ => slotDegree (gatherSlot slots) degree ((place c).symm ⟨g, slots g i⟩)
        ((place size).symm σ)) g i
    simp only [Equiv.symm_apply_apply] at moved
    unfold offset
    rw [moved, Finset.sum_eq_zero (s := Finset.univ) fun h _ => ?_, zero_add]
    · exact Finset.sum_congr rfl fun k _ => by
        unfold slotDegree
        simp only [gatherSlot_place, Equiv.apply_eq_iff_eq, Sigma.mk.inj_iff, heq_eq_eq, true_and]
    · refine ite_eq_right_iff.mpr fun before => Finset.sum_eq_zero fun k _ => ?_
      have other : h ≠ g := fun equal => by rw [equal] at before; exact lt_irrefl _ before
      unfold slotDegree
      rw [gatherSlot_place]
      exact ite_eq_right fun same => other (Sigma.mk.inj_iff.mp ((place c).symm.injective same)).1
  unfold quotientEntry
  rw [gatherSlot_place, cells, slotWidth_gather, degrees, Nat.add_assoc]

end Gather

end Entries

section Spec

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E]

theorem sum_groupPlace {M : Type} [AddCommMonoid M] (count : Fin n → ℕ) (F : Fin (∑ g, count g) → M) :
    ∑ r, F r = ∑ g, ∑ i : Fin (count g), F ((place count).symm ⟨g, i⟩) := by
  rw [← (place count).symm.sum_comp, Fintype.sum_sigma]

theorem gather_place {count : Fin n → ℕ} (rows : (g : Fin n) → Fin (count g) → RingRow) (g : Fin n)
    (i : Fin (count g)) : gather rows ((place count).symm ⟨g, i⟩) = rows g i := by
  unfold gather
  rw [Equiv.apply_symm_apply]

/-- The slots of the join of `Batch.flatRelation`: for each group in the order `groupOrder` its
carrier row and its tail rows, then the rows after the groups. -/
def batchJoinSlot : Fin (∑ g, B.groupPlaneRows g + B.restCount) →
    Fin (∑ g, B.groupCount g + B.restCount) :=
  appendSlot (gatherSlot (size := B.groupPlaneRows) (c := B.groupCount) fun g =>
    flatSlot (B.planes g) (B.groupTail g)) (id : Fin B.restCount → Fin B.restCount)

/-- The coefficients of the ring rows of the join of `Batch.flatRelation`: `β_t` for the plane
`t` of a carrier row, and 1 for each other ring row. -/
noncomputable def batchJoinBasis (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) :
    Fin (∑ g, B.groupPlaneRows g + B.restCount) → E :=
  Fin.append (fun q => Fin.append (⇑(β (place B.groupPlaneRows q).1)) (fun _ => (1 : E))
    (place B.groupPlaneRows q).2) fun _ => 1

theorem batchJoinSlot_group (g : Fin n) (i : Fin (B.groupPlaneRows g)) :
    (batchJoinSlot B (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, i⟩)) : ℕ) =
      codeOffset B.groupCount g + flatSlot (B.planes g) (B.groupTail g) i :=
  (congrArg Fin.val ((appendSlot_left _ _ _).trans (congrArg (Fin.castAdd B.restCount)
    (gatherSlot_place (size := B.groupPlaneRows) (c := B.groupCount)
      (fun g => flatSlot (B.planes g) (B.groupTail g)) g i)))).trans
    ((Fin.val_castAdd _ _).trans (place_symm_code _ _ _))

theorem batchJoinSlot_rest (r : Fin B.restCount) :
    (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g) r) : ℕ) = ∑ g, B.groupCount g + r := by
  unfold batchJoinSlot
  rw [appendSlot_right, Fin.val_natAdd]
  rfl

variable (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)

theorem batchJoinBasis_group (g : Fin n) (i : Fin (B.groupPlaneRows g)) :
    batchJoinBasis B β (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, i⟩)) =
      Fin.append (⇑(β g)) (fun _ => (1 : E)) i := by
  unfold batchJoinBasis
  rw [Fin.append_left]
  show Fin.append (⇑(β (place B.groupPlaneRows ((place B.groupPlaneRows).symm ⟨g, i⟩)).1))
    (fun _ => (1 : E)) (place B.groupPlaneRows ((place B.groupPlaneRows).symm ⟨g, i⟩)).2 = _
  rw [Equiv.apply_symm_apply]

theorem batchJoinBasis_rest (r : Fin B.restCount) :
    batchJoinBasis B β (Fin.natAdd (∑ g, B.groupPlaneRows g) r) = 1 :=
  Fin.append_right _ _ r

variable (st : B.Statement E) (challenge : B.Challenge)

theorem batch_flatRows_group (g : Fin n) (i : Fin (B.groupPlaneRows g)) :
    B.flatRows β st challenge (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, i⟩)) =
      B.groupRows β st challenge g i :=
  (Fin.append_left (gather (B.groupRows β st challenge)) _ _).trans (gather_place _ g i)

theorem groupRows_degree_sum (g : Fin n) :
    ∑ i, (B.groupRows β st challenge g i).degree = B.groupDegrees g := by
  unfold Batch.groupRows Batch.groupDegrees Batch.groupPlaneRows Batch.groupTail
  rw [Fin.sum_univ_add, Fin.sum_univ_add]
  simp only [Fin.append_left, Fin.append_right, relocate_degree, evaluationRow_degree,
    innerRow_degree, imageRow_degree, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul]

variable [Fintype E]

/-- `m_{τ_1}(r_2)` of `Batch.liftedReduction` as a sum over the ring rows of `Batch.flatRows`:
each ring row with the coefficient `ẽq(τ_1, slot)·basis`, as its linear form at the cells
`place` minus `(α^{d_r} + 1)` times its quotient digits at the cells `slot` of their entries, and
the scalar row. -/
theorem batchLifted_rows {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ} (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (α : E) {τ r : List E} (length : r.length = μ) (tau : τ.length = batch) :
    mle r (((Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit).rows
      α).batchWeight τ) =
      ∑ q, eqAt τ (batchJoinSlot B q) * (batchJoinBasis B β q *
        ((B.flatRows β st challenge q).scan α
            (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r)) -
          (α ^ (B.flatRows β st challenge q).degree + 1) * quotientScan α r base digits
            (B.flatRows β st challenge q).degree fun a j => (B.gaps digits).slot
              (quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
                digits q j a))) +
      eqAt τ (∑ g, B.groupCount g + B.restCount) * mle r (B.scalarWeight β st μ) := by
  have rowInside : ∀ (q : Fin B.planeRows) c j x, j < (B.flatRows β st challenge q).degree →
      ((B.flatRows β st challenge q).relocate (B.gaps digits).place).address c j = some x →
        x < 2 ^ r.length := by
    intro q c j x _ read
    cases h : (B.flatRows β st challenge q).address c j with
    | none =>
        simp only [RingRow.relocate, h, Option.map_none] at read
        cases read
    | some y =>
        simp only [RingRow.relocate, h, Option.map_some, Option.some.injEq] at read
        rw [← read, length]
        exact (B.place_lt_room digits (B.flatRows_below β st challenge q c j y h)).trans_le room
  have quotientInside : ∀ (q : Fin B.planeRows) j (a : Fin digits),
      j < (B.flatRows β st challenge q).degree →
      (B.gaps digits).slot (quotientEntry (batchJoinSlot B)
        (fun q => (B.flatRows β st challenge q).degree) digits q j a) < 2 ^ r.length := by
    intro q j a small
    rw [length]
    exact ((B.flatRelation β st challenge μ fits).quotientCell_lt _ digits q small a).trans_le
      ((Batch.flatRelation_quotientEnd β st challenge μ fits digits).trans_le room)
  refine (LiftRelation.mle_lift _ _ _ _ _ α τ r tau).trans ?_
  show ∑ q : Fin (∑ g, B.groupPlaneRows g + B.restCount), eqAt τ (batchJoinSlot B q : ℕ) *
      (batchJoinBasis B β q * mle r (((B.flatRows β st challenge q).relocate
        (B.gaps digits).place).liftWeight ⟨digits, fun a => (base : Zq) ^ (a : ℕ),
          fun j a => (B.gaps digits).slot (quotientEntry (batchJoinSlot B)
            (fun q => (B.flatRows β st challenge q).degree) digits q j a)⟩ α)) +
    eqAt τ (∑ g, B.groupCount g + B.restCount) * mle r (B.scalarWeight β st μ) = _
  congr 1
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [RingRow.mle_liftWeight _ _ α r (rowInside q) (fun j a small => quotientInside q j a small),
    RingRow.scan_relocate]
  simp only [map_pow, map_natCast]
  rfl

end Spec

section Terms

variable {E : Type} [CommRing E] [Algebra Zq E]

/-- A ring row at `r_2`: its linear form at the cells of `φ` minus `(α^d + 1)` times its quotient
digits at the cells `cell`. -/
noncomputable def ringTerm (α : E) (r : List E) (base digits : ℕ) (φ : Option ℕ → E) (row : RingRow)
    (cell : ℕ → ℕ → ℕ) : E :=
  row.scan α φ - (α ^ row.degree + 1) * quotientScan α r base digits row.degree cell

theorem entry_below {digits X Y D a y : ℕ} (digit : a < digits) (inside : y < D) (le : X + D ≤ Y) :
    digits * X + (a * D + y) < digits * Y := by
  have step : (a + 1) * D ≤ digits * D := Nat.mul_le_mul_right D digit
  have total : digits * X + digits * D ≤ digits * Y := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left digits le
  rw [Nat.succ_mul] at step
  omega

theorem coefficient_below {P d t j : ℕ} (plane : t < P) (inside : j < d) : t * d + j < P * d := by
  have step : (t + 1) * d ≤ P * d := Nat.mul_le_mul_right d plane
  rw [Nat.succ_mul] at step
  omega

variable (α : E) (r : List E) (b digits d : ℕ) (G : Gaps) {cell cell' : ℕ → ℕ → ℕ}

omit [Algebra Zq E] in
theorem quotientScan_slot_main (inside : ∀ a < digits, ∀ j < d, cell a j < G.mainCells)
    (same : ∀ a < digits, ∀ j < d, G.main + cell a j = cell' a j) :
    quotientScan α r b digits d (fun a j => G.slot (cell a j)) = quotientScan α r b digits d cell' :=
  quotientScan_congr α r b digits d fun a digit j small => by
    rw [Gaps.slot_main G (inside a digit j small), same a digit j small]

omit [Algebra Zq E] in
theorem quotientScan_slot_middle (low : ∀ a < digits, ∀ j < d, G.mainCells ≤ cell a j)
    (inside : ∀ a < digits, ∀ j < d, cell a j < G.mainCells + G.middleCells)
    (same : ∀ a < digits, ∀ j < d, G.middle + cell a j = cell' a j) :
    quotientScan α r b digits d (fun a j => G.slot (cell a j)) = quotientScan α r b digits d cell' :=
  quotientScan_congr α r b digits d fun a digit j small => by
    rw [Gaps.slot_middle G (low a digit j small) (inside a digit j small), same a digit j small]

omit [Algebra Zq E] in
theorem quotientScan_slot_last (low : ∀ a < digits, ∀ j < d, G.mainCells + G.middleCells ≤ cell a j)
    (same : ∀ a < digits, ∀ j < d, G.last + cell a j = cell' a j) :
    quotientScan α r b digits d (fun a j => G.slot (cell a j)) = quotientScan α r b digits d cell' :=
  quotientScan_congr α r b digits d fun a digit j small => by
    rw [Gaps.slot_last G (low a digit j small), same a digit j small]

end Terms

section Groups

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E]
  (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E) (challenge : B.Challenge)
  (base digits : ℕ) (α : E) (τ r : List E)

/-- The ring row `q` of `Batch.flatRows` in `m_{τ_1}(r_2)`, with its coefficient. -/
noncomputable def batchTerm (q : Fin (∑ g, B.groupPlaneRows g + B.restCount)) : E :=
  eqAt τ (batchJoinSlot B q) * (batchJoinBasis B β q * ringTerm α r base digits
    (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r)) (B.flatRows β st challenge q)
      fun a j => (B.gaps digits).slot (quotientEntry (batchJoinSlot B)
        (fun q => (B.flatRows β st challenge q).degree) digits q j a))

/-- The cells of the code of the ring rows of the group `g`: the positions of its flat witness
moved by `Batch.relocate` and `Gaps.place`. -/
def groupCells (g : Fin n) (o : Option ℕ) : Option ℕ :=
  o.map fun x => (B.gaps digits).place (B.relocate g x)

/-- `batchTerm` of a ring row from its slot, its coefficient, its row and the cells of its
quotient digits. -/
theorem batchTerm_of (q : Fin (∑ g, B.groupPlaneRows g + B.restCount)) {S : ℕ} {c : E} {row : RingRow}
    {cell : ℕ → ℕ → ℕ} (slot : (batchJoinSlot B q : ℕ) = S) (basis : batchJoinBasis B β q = c)
    (rowEq : B.flatRows β st challenge q = row)
    (cells : ∀ a < digits, ∀ j < row.degree, (B.gaps digits).slot (quotientEntry (batchJoinSlot B)
      (fun q => (B.flatRows β st challenge q).degree) digits q j a) = cell a j) :
    batchTerm B β st challenge base digits α τ r q = eqAt τ S * (c * ringTerm α r base digits
      (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r)) row cell) := by
  unfold batchTerm ringTerm
  rw [slot, basis, rowEq]
  congr 4
  exact quotientScan_congr α r base digits _ cells

theorem ringTerm_relocate (row : RingRow) (move : ℕ → ℕ) (g : Fin n) (cell : ℕ → ℕ → ℕ)
    (same : ∀ x, (B.gaps digits).place (move x) = (B.gaps digits).place (B.relocate g x)) :
    ringTerm α r base digits (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
        (row.relocate move) cell =
      ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r)) row cell := by
  unfold ringTerm
  rw [RingRow.scan_relocate, relocate_degree]
  congr 3
  funext o
  cases o with
  | none => rfl
  | some x => exact congrArg (eqAt r) (same x)

theorem batchEntry_group (g : Fin n) (i : Fin (B.groupPlaneRows g)) (j a : ℕ) :
    quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, i⟩)) j a =
      digits * codeOffset B.groupDegrees g + quotientEntry (flatSlot (B.planes g) (B.groupTail g))
        (fun i => (B.groupRows β st challenge g i).degree) digits i j a := by
  have glued := (joinEntry_append_left (digits := digits)
    (gatherSlot (size := B.groupPlaneRows) (c := B.groupCount)
      fun g => flatSlot (B.planes g) (B.groupTail g)) (id : Fin B.restCount → Fin B.restCount)
    (fun q => (B.flatRows β st challenge q).degree) ((place B.groupPlaneRows).symm ⟨g, i⟩) j a).trans
    (joinEntry_gather (digits := digits) (size := B.groupPlaneRows) (c := B.groupCount)
      (fun g => flatSlot (B.planes g) (B.groupTail g))
      (fun q => (B.flatRows β st challenge (Fin.castAdd B.restCount q)).degree) g i j a)
  have degrees : ∀ h (i : Fin (B.groupPlaneRows h)),
      (B.flatRows β st challenge (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨h, i⟩))).degree =
        (B.groupRows β st challenge h i).degree := fun h i =>
    congrArg RingRow.degree (batch_flatRows_group B β st challenge h i)
  simp only [degrees, groupRows_degree_sum] at glued
  exact glued

theorem groupCells_map (g : Fin n) (o : Option ℕ) :
    (o.map (B.relocate g)).map (B.gaps digits).place = groupCells B digits g o :=
  Option.map_map _ _ _

theorem groupDegrees_le (g : Fin n) :
    codeOffset B.groupDegrees g + B.groupDegrees g ≤ B.mainDegrees :=
  (codeOffset_add_le B.groupDegrees g).trans (Nat.le_add_right _ _)

/-- Plane `t` of the carrier row (102) of the group `g`. -/
theorem batchTerm_plane (g : Fin n) (t : Fin (B.planes g)) :
    batchTerm B β st challenge base digits α τ r
        (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, Fin.castAdd (B.groupTail g) t⟩)) =
      eqAt τ (codeOffset B.groupCount g) * (β g t * ringTerm α r base digits
        (fun o => (groupCells B digits g o).elim 0 (eqAt r))
        (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t)
        fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
          a * (B.planes g * B.carrierDegree g) + t * B.carrierDegree g + j) := by
  have row : B.flatRows β st challenge (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.castAdd (B.groupTail g) t⟩)) =
        (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t).relocate
          (B.relocate g) :=
    (batch_flatRows_group B β st challenge g _).trans (Fin.append_left _ _ t)
  have slot : (batchJoinSlot B (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.castAdd (B.groupTail g) t⟩)) : ℕ) =
        codeOffset B.groupCount g :=
    (batchJoinSlot_group B g _).trans (by rw [flatSlot_left, Fin.val_castAdd, Fin.val_zero, add_zero])
  have basis : batchJoinBasis B β (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.castAdd (B.groupTail g) t⟩)) = β g t :=
    (batchJoinBasis_group B β g _).trans (Fin.append_left _ _ t)
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.castAdd B.restCount
        ((place B.groupPlaneRows).symm ⟨g, Fin.castAdd (B.groupTail g) t⟩)) j a =
      digits * codeOffset B.groupDegrees g +
        (a * (B.planes g * B.carrierDegree g) + (t * B.carrierDegree g + j)) := fun j a =>
    (batchEntry_group B β st challenge digits g _ j a).trans (by
      rw [flatSlot_entry_plane (p := B.planes g) (m := B.groupTail g)
        (fun i => (B.groupRows β st challenge g i).degree) (B.carrierDegree g)
        digits (fun t => congrArg RingRow.degree (Fin.append_left _ _ t)) t j a])
  unfold batchTerm ringTerm
  rw [slot, basis, row, RingRow.scan_relocate, relocate_degree]
  simp only [entry, groupCells_map]
  congr 4
  refine quotientScan_slot_main α r base digits _ (B.gaps digits) (fun a digit j small => ?_)
    (fun a _ j _ => ?_)
  · have below := coefficient_below t.isLt small
    have le := groupDegrees_le B g
    have plane : B.planes g * B.carrierDegree g ≤ B.groupDegrees g := Nat.le_add_right _ _
    exact entry_below digit below (by omega)
  · show B.chainStart + _ = _
    ring

/-- The rows (104) of `A` of the group `g`. -/
theorem batchTerm_inner (g : Fin n) (x : Fin (B.shape g).innerRank) :
    batchTerm B β st challenge base digits α τ r
        (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
          (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x)⟩)) =
      eqAt τ (codeOffset B.groupCount g + (1 + x)) * ringTerm α r base digits
        (fun o => (groupCells B digits g o).elim 0 (eqAt r))
        (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) x)
        (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
          digits * (B.planes g * B.carrierDegree g) + x * (digits * (B.shape g).innerDegree) +
            a * (B.shape g).innerDegree + j) := by
  have row : B.flatRows β st challenge (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x)⟩)) =
        (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) x).relocate (B.relocate g) :=
    (batch_flatRows_group B β st challenge g _).trans
      ((Fin.append_right _ _ _).trans (Fin.append_left _ _ x))
  have slot : (batchJoinSlot B (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x)⟩)) : ℕ) =
        codeOffset B.groupCount g + (1 + x) :=
    (batchJoinSlot_group B g _).trans (congrArg (codeOffset B.groupCount g + ·)
      (((congrArg Fin.val (flatSlot_right (p := B.planes g) (m := B.groupTail g) _)).trans
        (Fin.val_natAdd _ _)).trans (congrArg (1 + ·) (Fin.val_castAdd _ _))))
  have basis : batchJoinBasis B β (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x)⟩)) = 1 :=
    (batchJoinBasis_group B β g _).trans (Fin.append_right _ _ _)
  have tail : ∀ y : Fin (B.shape g).innerRank,
      (B.groupRows β st challenge g (Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) y))).degree =
          (B.shape g).innerDegree := fun y =>
    congrArg RingRow.degree ((Fin.append_right _ _ _).trans (Fin.append_left _ _ y))
  have offsetInner : offset (fun i : Fin (B.groupTail g) =>
      (B.groupRows β st challenge g (Fin.natAdd (B.planes g) i)).degree)
      ((Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x : Fin (B.groupTail g)) : ℕ) =
        x * (B.shape g).innerDegree := by
    rw [Fin.val_castAdd]
    exact (offset_castAdd (a := (B.shape g).innerRank)
      (b := (B.shape g).slices * (B.shape g).outerRank) _ x).trans (by simp only [tail, offset_const])
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x)⟩)) j a =
      digits * codeOffset B.groupDegrees g + (digits * (B.planes g * B.carrierDegree g) +
        digits * (x * (B.shape g).innerDegree) + (a * (B.shape g).innerDegree + j)) := fun j a =>
    (batchEntry_group B β st challenge digits g _ j a).trans (by
      rw [flatSlot_entry_tail (p := B.planes g) (m := B.groupTail g)
        (fun i => (B.groupRows β st challenge g i).degree) (B.carrierDegree g)
        digits (fun t => congrArg RingRow.degree (Fin.append_left _ _ t))
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) x) j a, offsetInner,
        tail x])
  unfold batchTerm ringTerm
  rw [slot, basis, one_mul, row, RingRow.scan_relocate, relocate_degree]
  simp only [entry, groupCells_map]
  congr 3
  refine quotientScan_slot_main α r base digits _ (B.gaps digits) (fun a digit j small => ?_)
    (fun a _ j _ => ?_)
  · have le := groupDegrees_le B g
    have within : B.planes g * B.carrierDegree g + (x * (B.shape g).innerDegree +
        (B.shape g).innerDegree) ≤ B.groupDegrees g := by
      have step : (x + 1) * (B.shape g).innerDegree ≤
          (B.shape g).innerRank * (B.shape g).innerDegree := Nat.mul_le_mul_right _ x.isLt
      rw [Nat.succ_mul] at step
      unfold Batch.groupDegrees
      omega
    have shape : digits * codeOffset B.groupDegrees g + (digits * (B.planes g * B.carrierDegree g) +
        digits * (x * (B.shape g).innerDegree) + (a * (B.shape g).innerDegree + j)) =
          digits * (codeOffset B.groupDegrees g + B.planes g * B.carrierDegree g +
            x * (B.shape g).innerDegree) + (a * (B.shape g).innerDegree + j) := by ring
    rw [shape]
    exact entry_below digit small (by omega)
  · show B.chainStart + _ = _
    ring

/-- The rows (105) of `B` of the group `g`. -/
theorem batchTerm_image (g : Fin n) (x : Fin ((B.shape g).slices * (B.shape g).outerRank)) :
    batchTerm B β st challenge base digits α τ r
        (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
          (Fin.natAdd (B.shape g).innerRank x)⟩)) =
      eqAt τ (codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + x))) *
        ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
        (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
          ((B.opening g).firstCell (B.fold g)) (st.payload g) x)
        (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
          digits * (B.planes g * B.carrierDegree g) +
            (B.shape g).innerRank * (digits * (B.shape g).innerDegree) +
              x * (digits * (B.shape g).outerDegree) + a * (B.shape g).outerDegree + j) := by
  have row : B.flatRows β st challenge (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.natAdd (B.shape g).innerRank x)⟩)) =
        (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
          ((B.opening g).firstCell (B.fold g)) (st.payload g) x).relocate (B.relocate g) :=
    (batch_flatRows_group B β st challenge g _).trans
      ((Fin.append_right _ _ _).trans (Fin.append_right _ _ x))
  have slot : (batchJoinSlot B (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.natAdd (B.shape g).innerRank x)⟩)) : ℕ) =
        codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + x)) :=
    (batchJoinSlot_group B g _).trans (congrArg (codeOffset B.groupCount g + ·)
      (((congrArg Fin.val (flatSlot_right (p := B.planes g) (m := B.groupTail g) _)).trans
        (Fin.val_natAdd _ _)).trans (congrArg (1 + ·) (Fin.val_natAdd _ _))))
  have basis : batchJoinBasis B β (Fin.castAdd B.restCount
      ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.natAdd (B.shape g).innerRank x)⟩)) = 1 :=
    (batchJoinBasis_group B β g _).trans (Fin.append_right _ _ _)
  have inner : ∀ y : Fin (B.shape g).innerRank,
      (B.groupRows β st challenge g (Fin.natAdd (B.planes g)
        (Fin.castAdd ((B.shape g).slices * (B.shape g).outerRank) y))).degree =
          (B.shape g).innerDegree := fun y =>
    congrArg RingRow.degree ((Fin.append_right _ _ _).trans (Fin.append_left _ _ y))
  have outer : ∀ y : Fin ((B.shape g).slices * (B.shape g).outerRank),
      (B.groupRows β st challenge g (Fin.natAdd (B.planes g)
        (Fin.natAdd (B.shape g).innerRank y))).degree = (B.shape g).outerDegree := fun y =>
    ((congrArg RingRow.degree ((Fin.append_right _ _ _).trans (Fin.append_right _ _ y))).trans
      (imageRow_degree _ _ _ _ y))
  have offsetImage : offset (fun i : Fin (B.groupTail g) =>
      (B.groupRows β st challenge g (Fin.natAdd (B.planes g) i)).degree)
      ((Fin.natAdd (B.shape g).innerRank x : Fin (B.groupTail g)) : ℕ) =
        (B.shape g).innerRank * (B.shape g).innerDegree + x * (B.shape g).outerDegree := by
    rw [Fin.val_natAdd]
    exact (offset_natAdd (a := (B.shape g).innerRank)
      (b := (B.shape g).slices * (B.shape g).outerRank) _ x).trans (by
        simp only [inner, outer, offset_const, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          smul_eq_mul])
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, Fin.natAdd (B.planes g)
        (Fin.natAdd (B.shape g).innerRank x)⟩)) j a =
      digits * codeOffset B.groupDegrees g + (digits * (B.planes g * B.carrierDegree g) +
        digits * ((B.shape g).innerRank * (B.shape g).innerDegree + x * (B.shape g).outerDegree) +
          (a * (B.shape g).outerDegree + j)) := fun j a =>
    (batchEntry_group B β st challenge digits g _ j a).trans (by
      rw [flatSlot_entry_tail (p := B.planes g) (m := B.groupTail g)
        (fun i => (B.groupRows β st challenge g i).degree) (B.carrierDegree g)
        digits (fun t => congrArg RingRow.degree (Fin.append_left _ _ t))
        (Fin.natAdd (B.shape g).innerRank x) j a, offsetImage,
        outer x])
  unfold batchTerm ringTerm
  rw [slot, basis, one_mul, row, RingRow.scan_relocate, relocate_degree, imageRow_degree]
  simp only [entry, groupCells_map]
  congr 3
  refine quotientScan_slot_main α r base digits _ (B.gaps digits) (fun a digit j small => ?_)
    (fun a _ j _ => ?_)
  · have le := groupDegrees_le B g
    have within : B.planes g * B.carrierDegree g + ((B.shape g).innerRank * (B.shape g).innerDegree +
        x * (B.shape g).outerDegree + (B.shape g).outerDegree) ≤ B.groupDegrees g := by
      have step : (x + 1) * (B.shape g).outerDegree ≤
          (B.shape g).slices * (B.shape g).outerRank * (B.shape g).outerDegree :=
        Nat.mul_le_mul_right _ x.isLt
      rw [Nat.succ_mul] at step
      unfold Batch.groupDegrees
      omega
    have shape : digits * codeOffset B.groupDegrees g + (digits * (B.planes g * B.carrierDegree g) +
        digits * ((B.shape g).innerRank * (B.shape g).innerDegree + x * (B.shape g).outerDegree) +
          (a * (B.shape g).outerDegree + j)) =
          digits * (codeOffset B.groupDegrees g + B.planes g * B.carrierDegree g +
            ((B.shape g).innerRank * (B.shape g).innerDegree + x * (B.shape g).outerDegree)) +
              (a * (B.shape g).outerDegree + j) := by ring
    rw [shape]
    exact entry_below digit small (by omega)
  · show B.chainStart + _ = _
    ring

theorem offset_gather (count d : Fin n → ℕ) (g : Fin n) (x : Fin (count g)) :
    offset (fun y : Fin (∑ h, count h) => d (place count y).1) ((place count).symm ⟨g, x⟩ : ℕ) =
      codeOffset (fun h => count h * d h) g + x * d g := by
  unfold offset
  rw [sum_before_place count (fun σ => d σ.1) g x, codeOffset_eq_sum]
  congr 1
  · refine Finset.sum_congr rfl fun h _ => ?_
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  · exact offset_const (d g) x

theorem groupDegrees_sum :
    ∑ q : Fin (∑ g, B.groupPlaneRows g),
      (B.flatRows β st challenge (Fin.castAdd B.restCount q)).degree = ∑ g, B.groupDegrees g := by
  rw [sum_groupPlace]
  exact Finset.sum_congr rfl fun g _ => (Finset.sum_congr rfl fun i _ =>
    congrArg RingRow.degree (batch_flatRows_group B β st challenge g i)).trans
      (groupRows_degree_sum B β st challenge g)

theorem batchEntry_rest (q : Fin B.restCount) (j a : ℕ) :
    quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.natAdd (∑ g, B.groupPlaneRows g) q) j a =
      digits * ∑ g, B.groupDegrees g + (digits * offset (fun q : Fin B.restCount =>
        (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree) q +
          (a * (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree + j)) :=
  (joinEntry_append_right (digits := digits)
    (gatherSlot (size := B.groupPlaneRows) (c := B.groupCount)
      fun g => flatSlot (B.planes g) (B.groupTail g)) (id : Fin B.restCount → Fin B.restCount)
    (fun q => (B.flatRows β st challenge q).degree) q j a).trans (by
      rw [groupDegrees_sum B β st challenge, joinEntry_plain])

theorem degree_append {a b : ℕ} (u : Fin a → RingRow) (v : Fin b → RingRow) :
    (fun q => (Fin.append u v q).degree) = Fin.append (fun x => (u x).degree) (fun y => (v y).degree) :=
  funext fun q => by
    refine Fin.addCases (fun x => ?_) (fun y => ?_) q
    · simp only [Fin.append_left]
    · simp only [Fin.append_right]

/-- The degrees of the ring rows of `Batch.flatRows` after the groups, in their five families. -/
theorem restDegrees :
    (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree) =
      Fin.append (fun _ : Fin (1 * B.openRank) => B.openDegree)
        (Fin.append (fun k : Fin (∑ h, firstRank (B.shape h).chain) =>
            firstDegree (B.shape (place (fun h => firstRank (B.shape h).chain) k).1).chain)
          (Fin.append (fun _ : Fin (firstRank B.chain) => firstDegree B.chain)
            (Fin.append (fun k : Fin (∑ h, secondRank (B.shape h).chain) =>
                secondDegree (B.shape (place (fun h => secondRank (B.shape h).chain) k).1).chain)
              (fun _ : Fin (secondRank B.chain) => secondDegree B.chain)))) := by
  refine (funext fun q => congrArg RingRow.degree (Fin.append_right _ _ q)).trans
    ((degree_append _ _).trans ?_)
  congr 1
  · exact funext fun x => imageRow_degree _ _ _ _ x
  refine (degree_append _ _).trans ?_
  congr 1
  · exact funext fun y => firstRow_degree _ _ _ _ _
  refine (degree_append _ _).trans ?_
  congr 1
  · exact funext fun x => firstRow_degree _ _ _ _ x
  refine (degree_append _ _).trans ?_
  congr 1
  · exact funext fun y => secondRow_degree _ _ _ _ _
  · exact funext fun x => secondRow_degree _ _ _ _ x

theorem sum_groupPlace_const (count d : Fin n → ℕ) :
    ∑ k : Fin (∑ h, count h), d (place count k).1 = ∑ h, count h * d h := by
  rw [sum_groupPlace]
  refine Finset.sum_congr rfl fun h _ => ?_
  simp only [Equiv.apply_symm_apply, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul]

/-- The rows (106) of `D` on the columns of the groups. -/
theorem batchTerm_openImage (x : Fin (1 * B.openRank)) :
    batchTerm B β st challenge base digits α τ r (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
          (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount)) =
      eqAt τ (∑ g, B.groupCount g + x) * ringTerm α r base digits
        (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
        (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload x)
        (fun a j => B.chainStart + digits * ∑ g, B.groupDegrees g + x * (digits * B.openDegree) +
          a * B.openDegree + j) := by
  have row : B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
        (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount)) =
        imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload x :=
    (Fin.append_right _ _ _).trans (Fin.append_left _ _ x)
  have off : offset (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree)
      ((Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
        (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount) : ℕ) =
        x * B.openDegree :=
    (congrArg (fun f => offset f _) (restDegrees B β st challenge)).trans
      ((offset_append_castAdd _ _ x).trans (offset_const _ x))
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
          (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount)) j a =
      digits * ∑ g, B.groupDegrees g + (digits * (x * B.openDegree) + (a * B.openDegree + j)) :=
    fun j a => (batchEntry_rest B β st challenge digits _ j a).trans (by
      rw [off, row, imageRow_degree])
  have slot : (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
        (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount)) : ℕ) =
        ∑ g, B.groupCount g + x :=
    batchJoinSlot_rest B _
  have cells : ∀ a < digits, ∀ j < (imageRow B.chain (B.openForm st.openKey) B.openFirstCell
      st.openPayload x).degree, (B.gaps digits).slot (quotientEntry (batchJoinSlot B)
        (fun q => (B.flatRows β st challenge q).degree) digits (Fin.natAdd (∑ g, B.groupPlaneRows g)
          (Fin.castAdd (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
            (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) x : Fin B.restCount)) j a) =
        B.chainStart + digits * ∑ g, B.groupDegrees g + x * (digits * B.openDegree) +
          a * B.openDegree + j := by
    intro a digit j small
    rw [entry j a]
    rw [imageRow_degree] at small
    have step : (x + 1) * B.openDegree ≤ 1 * B.openRank * B.openDegree :=
      Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    have shape : digits * ∑ g, B.groupDegrees g + (digits * (x * B.openDegree) +
        (a * B.openDegree + j)) = digits * (∑ g, B.groupDegrees g + x * B.openDegree) +
          (a * B.openDegree + j) := by ring
    rw [Gaps.slot_main _ (by
      rw [shape]
      exact entry_below digit small (by unfold Batch.mainDegrees; omega))]
    show B.chainStart + _ = _
    ring
  exact (batchTerm_of B β st challenge base digits α τ r _ slot (batchJoinBasis_rest B β _) row
    cells).trans (congrArg _ (one_mul _))

/-- The rows of `F_1` of the group `g` (Akita (47)). -/
theorem batchTerm_first (g : Fin n) (x : Fin (firstRank (B.shape g).chain)) :
    batchTerm B β st challenge base digits α τ r (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount)) =
      eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (codeOffset (fun h => firstRank (B.shape h).chain) g + x))) * ringTerm α r base digits
        (fun o => (groupCells B digits g o).elim 0 (eqAt r))
        (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g)) ((B.opening g).secondCell (B.fold g)) x)
        (fun a j => B.secondStart + digits * B.mainDegrees + digits * codeOffset (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g + x * (digits * firstDegree (B.shape g).chain) + a * firstDegree (B.shape g).chain + j) := by
  have row : B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount)) = (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g)) ((B.opening g).secondCell (B.fold g)) x).relocate (B.relocate g) :=
    (Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_left _ _ _).trans (gather_place
      (count := fun h => firstRank (B.shape h).chain) (fun g r => (firstRow (B.shape g).chain
        (st.key g).chain ((B.opening g).firstCell (B.fold g)) ((B.opening g).secondCell (B.fold g))
          r).relocate (B.relocate g)) g x)))
  have off : offset (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree)
      ((Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount) : ℕ) = 1 * B.openRank * B.openDegree + (codeOffset (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g + x * firstDegree (B.shape g).chain) :=
    (congrArg (fun f => offset f _) (restDegrees B β st challenge)).trans
      ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
        ((offset_append_castAdd _ _ _).trans (offset_gather (fun h => firstRank (B.shape h).chain)
          (fun h => firstDegree (B.shape h).chain) g x))))
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount)) j a =
      digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree + (codeOffset (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g + x * firstDegree (B.shape g).chain)) + (a * firstDegree (B.shape g).chain + j)) :=
    fun j a => (batchEntry_rest B β st challenge digits _ j a).trans (by
      rw [off, row, relocate_degree, firstRow_degree])
  have slot : (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount)) : ℕ) = ∑ g, B.groupCount g + (1 * B.openRank + (codeOffset (fun h => firstRank (B.shape h).chain) g + x)) :=
    (batchJoinSlot_rest B _).trans (congrArg (∑ g, B.groupCount g + ·)
      (congrArg (1 * B.openRank + ·)
        (place_symm_code (fun h => firstRank (B.shape h).chain) g x)))
  have cells : ∀ a < digits, ∀ j < ((firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g)) ((B.opening g).secondCell (B.fold g)) x).relocate (B.relocate g)).degree, (B.gaps digits).slot
      (quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.castAdd (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) ((place fun h => firstRank (B.shape h).chain).symm ⟨g, x⟩)) : Fin B.restCount)) j a) =
        B.secondStart + digits * B.mainDegrees + digits * codeOffset (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g + x * (digits * firstDegree (B.shape g).chain) + a * firstDegree (B.shape g).chain + j := by
    intro a digit j small
    rw [entry j a]
    rw [relocate_degree, firstRow_degree] at small
    have step : (x + 1) * firstDegree (B.shape g).chain ≤ firstRank (B.shape g).chain * firstDegree (B.shape g).chain := Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    have total := codeOffset_add_le (fun h => firstRank (B.shape h).chain *
      firstDegree (B.shape h).chain) g
    have shape : digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree +
        (codeOffset (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
          x * firstDegree (B.shape g).chain)) + (a * firstDegree (B.shape g).chain + j)) =
        digits * (B.mainDegrees + (codeOffset (fun h => firstRank (B.shape h).chain *
          firstDegree (B.shape h).chain) g + x * firstDegree (B.shape g).chain)) +
            (a * firstDegree (B.shape g).chain + j) := by
      unfold Batch.mainDegrees
      ring
    have low : digits * B.mainDegrees ≤ digits * (B.mainDegrees + (codeOffset (fun h =>
        firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
          x * firstDegree (B.shape g).chain)) := Nat.mul_le_mul_left _ (Nat.le_add_right _ _)
    have high := entry_below digit small (show B.mainDegrees + (codeOffset (fun h =>
        firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
          x * firstDegree (B.shape g).chain) + firstDegree (B.shape g).chain ≤
        B.mainDegrees + B.firstDegrees by unfold Batch.firstDegrees; omega)
    rw [shape, Gaps.slot_middle _ (by show digits * B.mainDegrees ≤ _; omega)
      (by show _ < digits * B.mainDegrees + digits * B.firstDegrees; rw [← Nat.mul_add]; exact high)]
    show B.secondStart + _ = _
    ring
  exact (batchTerm_of B β st challenge base digits α τ r _ slot (batchJoinBasis_rest B β _) row
    cells).trans (congrArg _ ((one_mul _).trans
      (ringTerm_relocate B base digits α r _ _ g _ fun _ => rfl)))

/-- The rows of `H_1` (Akita (47)). -/
theorem batchTerm_openFirst (x : Fin (firstRank B.chain)) :
    batchTerm B β st challenge base digits α τ r (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount)) =
      eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + x))) * ringTerm α r base digits
        (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
        (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x)
        (fun a j => B.secondStart + digits * B.mainDegrees + digits * ∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + x * (digits * firstDegree B.chain) + a * firstDegree B.chain + j) := by
  have row : B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount)) = firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x :=
    (Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans (Fin.append_left _ _ x)))
  have off : offset (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree)
      ((Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount) : ℕ) = 1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + x * firstDegree B.chain) :=
    (congrArg (fun f => offset f _) (restDegrees B β st challenge)).trans
      ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
        ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·)
          (sum_groupPlace_const _ fun h => firstDegree (B.shape h).chain)
          ((offset_append_castAdd _ _ x).trans (offset_const _ x))))))
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount)) j a =
      digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + x * firstDegree B.chain)) + (a * firstDegree B.chain + j)) :=
    fun j a => (batchEntry_rest B β st challenge digits _ j a).trans (by
      rw [off, row, firstRow_degree])
  have slot : (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount)) : ℕ) = ∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + x)) :=
    batchJoinSlot_rest B _
  have cells : ∀ a < digits, ∀ j < (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x).degree, (B.gaps digits).slot
      (quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.castAdd (∑ h, secondRank (B.shape h).chain + secondRank B.chain) x)) : Fin B.restCount)) j a) =
        B.secondStart + digits * B.mainDegrees + digits * ∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + x * (digits * firstDegree B.chain) + a * firstDegree B.chain + j := by
    intro a digit j small
    rw [entry j a]
    rw [firstRow_degree] at small
    have step : (x + 1) * firstDegree B.chain ≤ firstRank B.chain * firstDegree B.chain := Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    have shape : digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree +
        (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
          x * firstDegree B.chain)) + (a * firstDegree B.chain + j)) =
        digits * (B.mainDegrees + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
          x * firstDegree B.chain)) + (a * firstDegree B.chain + j) := by
      unfold Batch.mainDegrees
      ring
    have low : digits * B.mainDegrees ≤ digits * (B.mainDegrees + (∑ h, firstRank (B.shape h).chain *
        firstDegree (B.shape h).chain + x * firstDegree B.chain)) :=
      Nat.mul_le_mul_left _ (Nat.le_add_right _ _)
    have high := entry_below digit small (show B.mainDegrees + (∑ h, firstRank (B.shape h).chain *
        firstDegree (B.shape h).chain + x * firstDegree B.chain) + firstDegree B.chain ≤
        B.mainDegrees + B.firstDegrees by unfold Batch.firstDegrees; omega)
    rw [shape, Gaps.slot_middle _ (by show digits * B.mainDegrees ≤ _; omega)
      (by show _ < digits * B.mainDegrees + digits * B.firstDegrees; rw [← Nat.mul_add]; exact high)]
    show B.secondStart + _ = _
    ring
  exact (batchTerm_of B β st challenge base digits α τ r _ slot (batchJoinBasis_rest B β _) row
    cells).trans (congrArg _ (one_mul _))

/-- The rows of `F_2` of the group `g` (Akita (47)). -/
theorem batchTerm_second (g : Fin n) (x : Fin (secondRank (B.shape g).chain)) :
    batchTerm B β st challenge base digits α τ r (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount)) =
      eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (codeOffset (fun h => secondRank (B.shape h).chain) g + x))))) * ringTerm α r base digits
        (fun o => (groupCells B digits g o).elim 0 (eqAt r))
        (secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g)) (st.payload g) x)
        (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * codeOffset (fun h => secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g + x * (digits * secondDegree (B.shape g).chain) + a * secondDegree (B.shape g).chain + j) := by
  have row : B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount)) = (secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g)) (st.payload g) x).relocate (B.relocate g) :=
    (Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_left _ _ _).trans (gather_place
      (count := fun h => secondRank (B.shape h).chain) (fun g r => (secondRow (B.shape g).chain
        (st.key g).chain ((B.opening g).secondCell (B.fold g)) (st.payload g) r).relocate
          (B.relocate g)) g x)))))
  have off : offset (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree)
      ((Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount) : ℕ) = 1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + (firstRank B.chain * firstDegree B.chain + (codeOffset (fun h => secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g + x * secondDegree (B.shape g).chain))) :=
    (congrArg (fun f => offset f _) (restDegrees B β st challenge)).trans
      ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
        ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·)
          (sum_groupPlace_const _ fun h => firstDegree (B.shape h).chain)
          ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
            ((offset_append_castAdd _ _ _).trans (offset_gather (fun h => secondRank (B.shape h).chain)
              (fun h => secondDegree (B.shape h).chain) g x))))))))
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount)) j a =
      digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + (firstRank B.chain * firstDegree B.chain + (codeOffset (fun h => secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g + x * secondDegree (B.shape g).chain)))) + (a * secondDegree (B.shape g).chain + j)) :=
    fun j a => (batchEntry_rest B β st challenge digits _ j a).trans (by
      rw [off, row, relocate_degree, secondRow_degree])
  have slot : (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount)) : ℕ) = ∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (codeOffset (fun h => secondRank (B.shape h).chain) g + x)))) :=
    (batchJoinSlot_rest B _).trans (congrArg (∑ g, B.groupCount g + ·)
      (congrArg (fun y => 1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + y)))
        (place_symm_code (fun h => secondRank (B.shape h).chain) g x)))
  have cells : ∀ a < digits, ∀ j < ((secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g)) (st.payload g) x).relocate (B.relocate g)).degree, (B.gaps digits).slot
      (quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.castAdd (secondRank B.chain) ((place fun h => secondRank (B.shape h).chain).symm ⟨g, x⟩)))) : Fin B.restCount)) j a) =
        B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * codeOffset (fun h => secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g + x * (digits * secondDegree (B.shape g).chain) + a * secondDegree (B.shape g).chain + j := by
    intro a digit j small
    rw [entry j a]
    rw [relocate_degree, secondRow_degree] at small
    have step : (x + 1) * secondDegree (B.shape g).chain ≤ secondRank (B.shape g).chain * secondDegree (B.shape g).chain := Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    have shape : digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree +
        (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
          (firstRank B.chain * firstDegree B.chain + (codeOffset (fun h => secondRank (B.shape h).chain *
            secondDegree (B.shape h).chain) g + x * secondDegree (B.shape g).chain)))) +
              (a * secondDegree (B.shape g).chain + j)) =
        digits * B.mainDegrees + digits * B.firstDegrees + (digits * (codeOffset (fun h =>
          secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g +
            x * secondDegree (B.shape g).chain) + (a * secondDegree (B.shape g).chain + j)) := by
      unfold Batch.mainDegrees Batch.firstDegrees
      ring
    rw [shape, Gaps.slot_last _ (by show digits * B.mainDegrees + digits * B.firstDegrees ≤ _; omega)]
    show B.cells + _ = _
    ring
  exact (batchTerm_of B β st challenge base digits α τ r _ slot (batchJoinBasis_rest B β _) row
    cells).trans (congrArg _ ((one_mul _).trans
      (ringTerm_relocate B base digits α r _ _ g _ fun _ => rfl)))

/-- The rows of `H_2` (Akita (47)). -/
theorem batchTerm_openSecond (x : Fin (secondRank B.chain)) :
    batchTerm B β st challenge base digits α τ r (Fin.natAdd (∑ g, B.groupPlaneRows g)
        (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount)) =
      eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + x))))) * ringTerm α r base digits
        (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
        (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x)
        (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * ∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain + x * (digits * secondDegree B.chain) + a * secondDegree B.chain + j) := by
  have row : B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount)) = secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x :=
    (Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans ((Fin.append_right _ _ _).trans (Fin.append_right _ _ x))))
  have off : offset (fun q : Fin B.restCount =>
      (B.flatRows β st challenge (Fin.natAdd (∑ g, B.groupPlaneRows g) q)).degree)
      ((Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount) : ℕ) = 1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + (firstRank B.chain * firstDegree B.chain + (∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain + x * secondDegree B.chain))) :=
    (congrArg (fun f => offset f _) (restDegrees B β st challenge)).trans
      ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
        ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·)
          (sum_groupPlace_const _ fun h => firstDegree (B.shape h).chain)
          ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·) (by simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul])
            ((offset_append_natAdd _ _ _).trans (congrArg₂ (· + ·)
              (sum_groupPlace_const _ fun h => secondDegree (B.shape h).chain)
              (offset_const _ x)))))))))
  have entry : ∀ j a, quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree)
      digits (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount)) j a =
      digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree + (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain + (firstRank B.chain * firstDegree B.chain + (∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain + x * secondDegree B.chain)))) + (a * secondDegree B.chain + j)) :=
    fun j a => (batchEntry_rest B β st challenge digits _ j a).trans (by
      rw [off, row, secondRow_degree])
  have slot : (batchJoinSlot B (Fin.natAdd (∑ g, B.groupPlaneRows g)
      (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount)) : ℕ) = ∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + x)))) :=
    batchJoinSlot_rest B _
  have cells : ∀ a < digits, ∀ j < (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x).degree, (B.gaps digits).slot
      (quotientEntry (batchJoinSlot B) (fun q => (B.flatRows β st challenge q).degree) digits
        (Fin.natAdd (∑ g, B.groupPlaneRows g) (Fin.natAdd (1 * B.openRank) (Fin.natAdd (∑ h, firstRank (B.shape h).chain) (Fin.natAdd (firstRank B.chain) (Fin.natAdd (∑ h, secondRank (B.shape h).chain) x))) : Fin B.restCount)) j a) =
        B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * ∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain + x * (digits * secondDegree B.chain) + a * secondDegree B.chain + j := by
    intro a digit j small
    rw [entry j a]
    rw [secondRow_degree] at small
    have step : (x + 1) * secondDegree B.chain ≤ secondRank B.chain * secondDegree B.chain := Nat.mul_le_mul_right _ x.isLt
    rw [Nat.succ_mul] at step
    have shape : digits * ∑ g, B.groupDegrees g + (digits * (1 * B.openRank * B.openDegree +
        (∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
          (firstRank B.chain * firstDegree B.chain + (∑ h, secondRank (B.shape h).chain *
            secondDegree (B.shape h).chain + x * secondDegree B.chain)))) +
              (a * secondDegree B.chain + j)) =
        digits * B.mainDegrees + digits * B.firstDegrees + (digits * (∑ h,
          secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
            x * secondDegree B.chain) + (a * secondDegree B.chain + j)) := by
      unfold Batch.mainDegrees Batch.firstDegrees
      ring
    rw [shape, Gaps.slot_last _ (by show digits * B.mainDegrees + digits * B.firstDegrees ≤ _; omega)]
    show B.cells + _ = _
    ring
  exact (batchTerm_of B β st challenge base digits α τ r _ slot (batchJoinBasis_rest B β _) row
    cells).trans (congrArg _ (one_mul _))

/-- The ring rows of the group `g` before the rows of `D`: its carrier row (102), its rows
(104) and its rows (105). -/
theorem batchGroup_value (g : Fin n) :
    ∑ i : Fin (B.groupPlaneRows g), batchTerm B β st challenge base digits α τ r
        (Fin.castAdd B.restCount ((place B.groupPlaneRows).symm ⟨g, i⟩)) =
      ∑ t : Fin (B.planes g), eqAt τ (codeOffset B.groupCount g) * (β g t * ringTerm α r base digits
          (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t)
          fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            a * (B.planes g * B.carrierDegree g) + t * B.carrierDegree g + j) +
        (∑ x : Fin (B.shape g).innerRank, eqAt τ (codeOffset B.groupCount g + (1 + x)) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) x)
          (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            digits * (B.planes g * B.carrierDegree g) + x * (digits * (B.shape g).innerDegree) +
              a * (B.shape g).innerDegree + j) +
        ∑ x : Fin ((B.shape g).slices * (B.shape g).outerRank),
          eqAt τ (codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + x))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
            ((B.opening g).firstCell (B.fold g)) (st.payload g) x)
          (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            digits * (B.planes g * B.carrierDegree g) +
              (B.shape g).innerRank * (digits * (B.shape g).innerDegree) +
                x * (digits * (B.shape g).outerDegree) + a * (B.shape g).outerDegree + j)) := by
  refine (Fin.sum_univ_add (a := B.planes g) (b := B.groupTail g) _).trans ?_
  congr 1
  · exact Finset.sum_congr rfl fun t _ => batchTerm_plane B β st challenge base digits α τ r g t
  refine (Fin.sum_univ_add (a := (B.shape g).innerRank)
    (b := (B.shape g).slices * (B.shape g).outerRank) _).trans ?_
  congr 1
  · exact Finset.sum_congr rfl fun x _ => batchTerm_inner B β st challenge base digits α τ r g x
  · exact Finset.sum_congr rfl fun x _ => batchTerm_image B β st challenge base digits α τ r g x

/-- The ring rows after the groups: the rows (106) of `D` and the rows (47) of `F_1` of each
group, of `H_1`, of `F_2` of each group and of `H_2`. -/
theorem batchRest_value :
    ∑ q : Fin B.restCount, batchTerm B β st challenge base digits α τ r
        (Fin.natAdd (∑ g, B.groupPlaneRows g) q) =
      ∑ x : Fin (1 * B.openRank), eqAt τ (∑ g, B.groupCount g + x) * ringTerm α r base digits
          (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload x)
          (fun a j => B.chainStart + digits * ∑ g, B.groupDegrees g + x * (digits * B.openDegree) +
            a * B.openDegree + j) +
        (∑ g, ∑ x : Fin (firstRank (B.shape g).chain),
          eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (codeOffset (fun h => firstRank (B.shape h).chain) g + x))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g))
            ((B.opening g).secondCell (B.fold g)) x)
          (fun a j => B.secondStart + digits * B.mainDegrees + digits * codeOffset (fun h =>
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
              x * (digits * firstDegree (B.shape g).chain) + a * firstDegree (B.shape g).chain + j) +
        (∑ x : Fin (firstRank B.chain), eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (∑ h, firstRank (B.shape h).chain + x))) * ringTerm α r base digits (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x)
          (fun a j => B.secondStart + digits * B.mainDegrees + digits * ∑ h,
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
              x * (digits * firstDegree B.chain) + a * firstDegree B.chain + j) +
        (∑ g, ∑ x : Fin (secondRank (B.shape g).chain),
          eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
            (codeOffset (fun h => secondRank (B.shape h).chain) g + x))))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g))
            (st.payload g) x)
          (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees +
            digits * codeOffset (fun h => secondRank (B.shape h).chain *
              secondDegree (B.shape h).chain) g + x * (digits * secondDegree (B.shape g).chain) +
                a * secondDegree (B.shape g).chain + j) +
        ∑ x : Fin (secondRank B.chain), eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + x))))) * ringTerm α r base digits (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x)
          (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * ∑ h,
            secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
              x * (digits * secondDegree B.chain) + a * secondDegree B.chain + j)))) := by
  refine (Fin.sum_univ_add (a := 1 * B.openRank) (b := ∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
    (∑ h, secondRank (B.shape h).chain + secondRank B.chain))) _).trans ?_
  congr 1
  · exact Finset.sum_congr rfl fun x _ => batchTerm_openImage B β st challenge base digits α τ r x
  refine (Fin.sum_univ_add (a := ∑ h, firstRank (B.shape h).chain) (b := firstRank B.chain +
    (∑ h, secondRank (B.shape h).chain + secondRank B.chain)) _).trans ?_
  congr 1
  · refine (sum_groupPlace (fun h => firstRank (B.shape h).chain) _).trans ?_
    exact Finset.sum_congr rfl fun g _ => Finset.sum_congr rfl fun x _ =>
      batchTerm_first B β st challenge base digits α τ r g x
  refine (Fin.sum_univ_add (a := firstRank B.chain) (b := ∑ h, secondRank (B.shape h).chain + secondRank B.chain) _).trans ?_
  congr 1
  · exact Finset.sum_congr rfl fun x _ => batchTerm_openFirst B β st challenge base digits α τ r x
  refine (Fin.sum_univ_add (a := ∑ h, secondRank (B.shape h).chain) (b := secondRank B.chain) _).trans ?_
  congr 1
  · refine (sum_groupPlace (fun h => secondRank (B.shape h).chain) _).trans ?_
    exact Finset.sum_congr rfl fun g _ => Finset.sum_congr rfl fun x _ =>
      batchTerm_second B β st challenge base digits α τ r g x
  · exact Finset.sum_congr rfl fun x _ => batchTerm_openSecond B β st challenge base digits α τ r x

variable [Fintype E]

/-- `m_{τ_1}(r_2)` of `Batch.liftedReduction` (Akita (123) at `r_2`): for each group `g` its
planes with `ẽq(τ_1, row_g)·β_t`, its rows (104) and (105), its rows of `F_1` and `F_2`, then the
rows (106) of `D`, of `H_1` and of `H_2`, and the scalar row (100). Each ring row is its linear
form at the cells `place ∘ relocate` minus `(α^{d_r} + 1)` times its quotient digits at their
cells in the code order (348). -/
theorem batchReduction_weight {μ : ℕ} (fits : B.cells ≤ 2 ^ μ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ} (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (length : r.length = μ) (tau : τ.length = batch) :
    mle r (((Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit).rows
      α).batchWeight τ) =
      ∑ g, (∑ t : Fin (B.planes g), eqAt τ (codeOffset B.groupCount g) * (β g t * ringTerm α r base digits
          (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t)
          fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            a * (B.planes g * B.carrierDegree g) + t * B.carrierDegree g + j) +
        (∑ x : Fin (B.shape g).innerRank, eqAt τ (codeOffset B.groupCount g + (1 + x)) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) x)
          (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            digits * (B.planes g * B.carrierDegree g) + x * (digits * (B.shape g).innerDegree) +
              a * (B.shape g).innerDegree + j) +
        ∑ x : Fin ((B.shape g).slices * (B.shape g).outerRank),
          eqAt τ (codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + x))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
            ((B.opening g).firstCell (B.fold g)) (st.payload g) x)
          (fun a j => B.chainStart + digits * codeOffset B.groupDegrees g +
            digits * (B.planes g * B.carrierDegree g) +
              (B.shape g).innerRank * (digits * (B.shape g).innerDegree) +
                x * (digits * (B.shape g).outerDegree) + a * (B.shape g).outerDegree + j))) + (∑ x : Fin (1 * B.openRank), eqAt τ (∑ g, B.groupCount g + x) * ringTerm α r base digits
          (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload x)
          (fun a j => B.chainStart + digits * ∑ g, B.groupDegrees g + x * (digits * B.openDegree) +
            a * B.openDegree + j) +
        (∑ g, ∑ x : Fin (firstRank (B.shape g).chain),
          eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (codeOffset (fun h => firstRank (B.shape h).chain) g + x))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g))
            ((B.opening g).secondCell (B.fold g)) x)
          (fun a j => B.secondStart + digits * B.mainDegrees + digits * codeOffset (fun h =>
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
              x * (digits * firstDegree (B.shape g).chain) + a * firstDegree (B.shape g).chain + j) +
        (∑ x : Fin (firstRank B.chain), eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (∑ h, firstRank (B.shape h).chain + x))) * ringTerm α r base digits (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x)
          (fun a j => B.secondStart + digits * B.mainDegrees + digits * ∑ h,
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
              x * (digits * firstDegree B.chain) + a * firstDegree B.chain + j) +
        (∑ g, ∑ x : Fin (secondRank (B.shape g).chain),
          eqAt τ (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
            (codeOffset (fun h => secondRank (B.shape h).chain) g + x))))) *
          ringTerm α r base digits (fun o => (groupCells B digits g o).elim 0 (eqAt r))
          (secondRow (B.shape g).chain (st.key g).chain ((B.opening g).secondCell (B.fold g))
            (st.payload g) x)
          (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees +
            digits * codeOffset (fun h => secondRank (B.shape h).chain *
              secondDegree (B.shape h).chain) g + x * (digits * secondDegree (B.shape g).chain) +
                a * secondDegree (B.shape g).chain + j) +
        ∑ x : Fin (secondRank B.chain), eqAt τ (∑ g, B.groupCount g + (1 * B.openRank +
            (∑ h, firstRank (B.shape h).chain + (firstRank B.chain + (∑ h, secondRank (B.shape h).chain + x))))) * ringTerm α r base digits (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))
          (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x)
          (fun a j => B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * ∑ h,
            secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
              x * (digits * secondDegree B.chain) + a * secondDegree B.chain + j))))) +
      eqAt τ (∑ g, B.groupCount g + B.restCount) * mle r (B.scalarWeight β st μ) := by
  rw [batchLifted_rows B β st challenge fits base digits room rowsFit α length tau]
  congr 1
  refine (Fin.sum_univ_add (a := ∑ g, B.groupPlaneRows g) (b := B.restCount)
    (fun q => batchTerm B β st challenge base digits α τ r q)).trans ?_
  congr 1
  · refine (sum_groupPlace B.groupPlaneRows _).trans ?_
    exact Finset.sum_congr rfl fun g _ => batchGroup_value B β st challenge base digits α τ r g
  · exact batchRest_value B β st challenge base digits α τ r

end Groups

end NightstreamFPrime.Spec.Compression.Akita.Replay
