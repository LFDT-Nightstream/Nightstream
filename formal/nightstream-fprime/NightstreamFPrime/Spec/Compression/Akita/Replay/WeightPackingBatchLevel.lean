import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchReplay

/-! The weight at `r_2` of one group of a fold with opening groups (ePrint 2026/1983, (102), (104),
(105), (47), (111) and (348); `batchReduction_weight` of `Replay/WeightPackingBatch.lean`).

- `relocated_shift`, `relocated_first`, `relocated_second` and `relocated_bounded`: the cells
  `Gaps.place ∘ Batch.relocate g` of a group are a shift on each segment of its flat witness and
  stay in the cube.
- `groupWeight_batch`: the parts of the replay of one group (its partials, rows (104) and (105),
  compression rows and contraction, and their quotient digits) are the families of the group in
  `batchReduction_weight`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Cells

variable {n : ℕ} (B : Batch (Fin n)) (digits : ℕ) (g : Fin n)

theorem relocated_shift : ∀ x c, x + c < (B.opening g).chainStart (B.fold g) →
    (B.gaps digits).place (B.relocate g (x + c)) = (B.gaps digits).place (B.relocate g x) + c := by
  intro x c small
  have ends := B.relocate_lt_chainStart g small
  have before := B.relocate_lt_chainStart g (show x < (B.opening g).chainStart (B.fold g) by omega)
  rw [B.relocate_main g small, B.relocate_main g (show x < B.mainCells g by unfold Batch.mainCells; omega)]
  rw [B.relocate_main g small] at ends
  rw [B.relocate_main g (show x < B.mainCells g by unfold Batch.mainCells; omega)] at before
  rw [Gaps.place_of_lt _ ends, Gaps.place_of_lt _ before, Nat.add_assoc]

theorem relocated_first : ∀ x c, (B.opening g).chainStart (B.fold g) ≤ x →
    x + c < (B.opening g).chainStart (B.fold g) + firstCells (B.shape g).chain →
    (B.gaps digits).place (B.relocate g (x + c)) = (B.gaps digits).place (B.relocate g x) + c := by
  intro x c low high
  obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le low
  obtain ⟨-, -, a, b, -, -, -, -⟩ := B.bounds g
  rw [Nat.add_assoc, B.relocate_first g (show y + c < _ by omega), B.relocate_first g (by omega),
    ← Nat.add_assoc]
  exact Gaps.place_shift (B.gaps digits) (lo := B.chainStart) (hi := B.openFirstStart)
    (Or.inr (Or.inl ⟨le_rfl, show B.openFirstStart ≤ B.secondStart from Nat.le_add_right _ _⟩))
    (B.firstStart g + y) c (by omega) (by omega)

theorem relocated_second : ∀ x c, (B.opening g).secondStart (B.fold g) ≤ x →
    x + c < (B.opening g).secondStart (B.fold g) + secondCells (B.shape g).chain →
    (B.gaps digits).place (B.relocate g (x + c)) = (B.gaps digits).place (B.relocate g x) + c := by
  intro x c low high
  obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le low
  obtain ⟨-, -, -, -, -, a, b, -⟩ := B.bounds g
  rw [Nat.add_assoc, B.relocate_second g (show y + c < _ by omega), B.relocate_second g (by omega),
    ← Nat.add_assoc]
  exact Gaps.place_shift (B.gaps digits) (lo := B.secondStart) (hi := B.openSecondStart)
    (Or.inr (Or.inr (show B.secondStart ≤ B.secondStart from le_rfl))) (B.secondStartOf g + y) c
    (by omega) (by omega)

theorem relocated_bounded {μ : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) :
    ∀ x < (B.opening g).cells (B.fold g), (B.gaps digits).place (B.relocate g x) < 2 ^ μ :=
  fun _ small => (B.place_lt_room digits (B.relocate_lt g small)).trans_le room

end Cells

section Group

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The parts of the replay of the group at the index `k` before the rows of `D`, its rows of
`F_1` and `F_2` and their quotient digits are the families of the group `groupOrder n k` in
`batchReduction_weight`. -/
theorem groupWeight_batch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (F : FoldInputs) (st : B.Statement E) (keyed : ∀ g, (st.key g).Shared (flatOf F.setup))
    (challenge : B.Challenge) (k : Fin n)
    (challenged : ∀ i, challenge (groupOrder n k) i = Poly.toRq _ ((cs k).getD i
      (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (ambiented : ∀ i : Fin (B.shape (groupOrder n k)).blocks,
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((amb k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i))
    (based : ∀ t, β (groupOrder n k) t = (extBasis.getD t Ext.zero).into)
    (positioned : ∀ x, (st.point (groupOrder n k)).position x =
      eqAt ((batchGroup B cs amb pts w k).segments.2.2.1.map Ext.into) x)
    (packed : ∀ a, (st.point (groupOrder n k)).pack a =
      eqAt ((batchGroup B cs amb pts w k).segments.1.map Ext.into) a)
    (α : Ext) (r2 : List Ext) (top : ℕ) (short : Nat.log 2 top ≤ r2.length)
    (numbers : (batchGroup B cs amb pts w k).Fits r2.length)
    (within : max B.openDegree (B.shape (groupOrder n k)).innerDegree ≤ top)
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length) (code : F.order = .code)
    (row first second start q1 q2 : ℕ)
    (insideC : start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) ≤
      2 ^ r2.length)
    (insideA : start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
      (B.shape (groupOrder n k)).innerRank * (F.digits * (B.shape (groupOrder n k)).innerDegree) ≤
        2 ^ r2.length)
    (insideB : start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
      (B.shape (groupOrder n k)).innerRank * (F.digits * (B.shape (groupOrder n k)).innerDegree) +
        (B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank *
          (F.digits * (B.shape (groupOrder n k)).outerDegree) ≤ 2 ^ r2.length)
    (inside1 : q1 + firstRank (B.shape (groupOrder n k)).chain *
      (F.digits * firstDegree (B.shape (groupOrder n k)).chain) ≤ 2 ^ r2.length)
    (inside2 : q2 + secondRank (B.shape (groupOrder n k)).chain *
      (F.digits * secondDegree (B.shape (groupOrder n k)).chain) ≤ 2 ^ r2.length) :
    ((partialLifts F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row).map fun l =>
          ((l.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((responseLifts F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row).map fun l =>
          ((l.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((imageLifts F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row).map fun l =>
          ((l.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((outerLifts F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row).map fun l =>
          ((l.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((chainLifts F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row first second).map fun p =>
          ((liftsValue (Tables.of α r2 (Nat.log 2 top) top) p).into : E)).sum +
      ((contractionValue F (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x))
        (batchGroup B cs amb pts w k) row (Tables.of α r2 (Nat.log 2 top) top)).into : E) +
      ((carrierQuotient F (batchGroup B cs amb pts w k) α (eqIndex F.tau1 row) start).map fun l =>
        ((l.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((liftsValue (Tables.of α r2 (Nat.log 2 top) top) (((rowStarts F.digits
        (start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)))
          [((B.shape (groupOrder n k)).innerDegree, (B.shape (groupOrder n k)).innerRank),
            ((B.shape (groupOrder n k)).outerDegree,
              (B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank)]).zipIdx).map
          fun x => quotientLift F.base F.digits x.1.1 α (eqIndex F.tau1 (row + 1 + x.2)) x.1.2)).into :
            E) +
      ∑ x : Fin (firstRank (B.shape (groupOrder n k)).chain),
        (((quotientLift F.base F.digits (firstDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (first + x))
          (q1 + x * (F.digits * firstDegree (B.shape (groupOrder n k)).chain))).value
            (Tables.of α r2 (Nat.log 2 top) top)).into : E) +
      ∑ x : Fin (secondRank (B.shape (groupOrder n k)).chain),
        (((quotientLift F.base F.digits (secondDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (second + x))
          (q2 + x * (F.digits * secondDegree (B.shape (groupOrder n k)).chain))).value
            (Tables.of α r2 (Nat.log 2 top) top)).into : E) =
    ∑ t : Fin (B.planes (groupOrder n k)), eqAt (F.tau1.map Ext.into) row *
        (β (groupOrder n k) t * ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (evaluationRow (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (β (groupOrder n k))
            (st.point (groupOrder n k)) (challenge (groupOrder n k)) t)
          fun a j => start + a * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
            t * B.carrierDegree (groupOrder n k) + j) +
      (∑ x : Fin (B.shape (groupOrder n k)).innerRank, eqAt (F.tau1.map Ext.into) (row + (1 + x)) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (innerRow (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (st.key (groupOrder n k))
            (challenge (groupOrder n k)) x)
          (fun a j => start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
            x * (F.digits * (B.shape (groupOrder n k)).innerDegree) +
              a * (B.shape (groupOrder n k)).innerDegree + j) +
      ∑ x : Fin ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank),
        eqAt (F.tau1.map Ext.into) (row + (1 + ((B.shape (groupOrder n k)).innerRank + x))) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (imageRow (B.shape (groupOrder n k)).chain
            (outerForm (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (st.key (groupOrder n k)))
            ((B.opening (groupOrder n k)).firstCell (B.fold (groupOrder n k)))
            (st.payload (groupOrder n k)) x)
          (fun a j => start + F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
            (B.shape (groupOrder n k)).innerRank * (F.digits * (B.shape (groupOrder n k)).innerDegree) +
              x * (F.digits * (B.shape (groupOrder n k)).outerDegree) +
                a * (B.shape (groupOrder n k)).outerDegree + j)) +
      ∑ x : Fin (firstRank (B.shape (groupOrder n k)).chain), eqAt (F.tau1.map Ext.into) (first + x) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (firstRow (B.shape (groupOrder n k)).chain (st.key (groupOrder n k)).chain
            ((B.opening (groupOrder n k)).firstCell (B.fold (groupOrder n k)))
            ((B.opening (groupOrder n k)).secondCell (B.fold (groupOrder n k))) x)
          (fun a j => q1 + x * (F.digits * firstDegree (B.shape (groupOrder n k)).chain) +
            a * firstDegree (B.shape (groupOrder n k)).chain + j) +
      ∑ x : Fin (secondRank (B.shape (groupOrder n k)).chain), eqAt (F.tau1.map Ext.into) (second + x) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (secondRow (B.shape (groupOrder n k)).chain (st.key (groupOrder n k)).chain
            ((B.opening (groupOrder n k)).secondCell (B.fold (groupOrder n k)))
            (st.payload (groupOrder n k)) x)
          (fun a j => q2 + x * (F.digits * secondDegree (B.shape (groupOrder n k)).chain) +
            a * secondDegree (B.shape (groupOrder n k)).chain + j) := by
  have topInner : (B.shape (groupOrder n k)).innerDegree ≤ top := (le_max_right _ _).trans within
  have topOf : ∀ {d : ℕ}, d ≤ max B.openDegree (B.shape (groupOrder n k)).innerDegree → d ≤ top :=
    fun le => le.trans within
  have shift := relocated_shift B F.digits (groupOrder n k)
  have bounded := relocated_bounded B F.digits (groupOrder n k) room
  have bound : ∀ {lo hi : ℕ}, hi ≤ (B.opening (groupOrder n k)).cells (B.fold (groupOrder n k)) →
      ∀ x, lo ≤ x → x < hi → (B.gaps F.digits).place (B.relocate (groupOrder n k) x) <
        2 ^ r2.length := fun le x _ high => bounded x (by omega)
  have starts := B.starts (groupOrder n k)
  have hP := (partialLifts_value (E := E) F
    (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) (batchGroup B cs amb pts w k)
    row α r2 _ top short (pow_log_top numbers.carrier) (pow_log_top numbers.opened)
    (log_le_top (topOf numbers.carrierTop)) (topOf numbers.carrierTop) shift bounded
    (challenge (groupOrder n k)) challenged)
  have hR := (responseLifts_value (E := E) F
    (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) (batchGroup B cs amb pts w k)
    row α r2 _ top short (pow_log_top numbers.inner) (log_le_top topInner) topInner shift bounded
    (st.key (groupOrder n k)) (keyed (groupOrder n k)).1)
  have hI := (imageLifts_value (E := E) F
    (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) (batchGroup B cs amb pts w k)
    row α r2 _ top short (pow_log_top numbers.outer) (log_le_top (topOf numbers.outerTop)) topInner
    shift bounded (challenge (groupOrder n k)) ambiented)
  have hO := (outerLifts_value (E := E) F
    (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) (batchGroup B cs amb pts w k)
    row α r2 _ top short (pow_log_top numbers.outer) (log_le_top (topOf numbers.outerTop))
    (topOf numbers.outerTop) shift bounded (st.key (groupOrder n k)) (keyed (groupOrder n k)).2.1)
  have hC := contractionValue_into (E := E) F
    (fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) (batchGroup B cs amb pts w k)
    row α r2 _ top short (pow_log_top numbers.inner) (log_le_top topInner) shift bounded
    (β (groupOrder n k)) (st.point (groupOrder n k)) positioned packed
  have hQC := (carrierQuotient_value (E := E) F code (batchGroup B cs amb pts w k)
    α r2 _ top (eqIndex F.tau1 row) start short (pow_log_top numbers.carrier)
    (log_le_top (topOf numbers.carrierTop)) (topOf numbers.carrierTop) insideC)
  have hQA := rowQuotients_two (E := E) α r2 _ top F.base F.digits
    (B.shape (groupOrder n k)).innerDegree (B.shape (groupOrder n k)).innerRank
    (B.shape (groupOrder n k)).outerDegree
    ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank) F.tau1 (row + 1) _
    short (pow_log_top numbers.inner) (log_le_top topInner) topInner (pow_log_top numbers.outer)
    (log_le_top (topOf numbers.outerTop)) (topOf numbers.outerTop) insideA insideB
  have firstPositive : NeZero (firstDegree (B.shape (groupOrder n k)).chain) := by
    obtain ⟨e, he⟩ := numbers.first
    have he' : 2 ^ e = firstDegree (B.shape (groupOrder n k)).chain := he
    exact ⟨by rw [← he']; positivity⟩
  have secondPositive : NeZero (secondDegree (B.shape (groupOrder n k)).chain) := by
    obtain ⟨e, he⟩ := numbers.second
    have he' : 2 ^ e = secondDegree (B.shape (groupOrder n k)).chain := he
    exact ⟨by rw [← he']; positivity⟩
  have hQ1 := quotientRows_value (E := E) α r2 _ top F.base F.digits
    (firstDegree (B.shape (groupOrder n k)).chain) F.tau1 q1 _ (fun x => first + x) short
    (pow_log_top numbers.first) (log_le_top (topOf numbers.firstTop)) (topOf numbers.firstTop)
    inside1
  have hQ2 := quotientRows_value (E := E) α r2 _ top F.base F.digits
    (secondDegree (B.shape (groupOrder n k)).chain) F.tau1 q2 _ (fun x => second + x) short
    (pow_log_top numbers.second) (log_le_top (topOf numbers.secondTop)) (topOf numbers.secondTop)
    inside2
  have hChain := chainLifts_into (E := E) (α := α) (r2 := r2) (bits := Nat.log 2 top) (top := top)
    (move := fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)) F.setup
    (eqIndex F.tau1) (B.shape (groupOrder n k)).chain (st.key (groupOrder n k)).chain
    (keyed (groupOrder n k)).2.2 (d := (B.shape (groupOrder n k)).outerDegree)
    ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank)
    ((B.opening (groupOrder n k)).chainStart (B.fold (groupOrder n k)))
    ((B.opening (groupOrder n k)).secondStart (B.fold (groupOrder n k)))
    (st.payload (groupOrder n k)) (row + 1 + (B.shape (groupOrder n k)).innerRank) first second short
    (pow_log_top numbers.outer) (log_le_top (topOf numbers.outerTop)) (topOf numbers.outerTop)
    (pow_log_top numbers.first) (pow_log_top numbers.second) (log_le_top (topOf numbers.firstTop))
    (log_le_top (topOf numbers.secondTop)) (topOf numbers.firstTop) (topOf numbers.secondTop)
    numbers.image numbers.chained (relocated_first B F.digits (groupOrder n k))
    (relocated_second B F.digits (groupOrder n k))
    (bound (by omega)) (bound (by omega))
  rw [chainLifts, firstCell_eq, secondCell_eq]
  dsimp only [batchGroup] at hP hR hI hO hC hQC hQA hQ1 hQ2 hChain ⊢
  rw [hChain]
  simp only [ringTerm, evaluationRow_degree, innerRow_degree, imageRow_degree, firstRow_degree,
    secondRow_degree]
  simp only [evaluationRow, innerRow, Form.scan_row, Form.scan_sub, imageRow_scan, firstCell_eq,
    secondCell_eq, groupCells, mul_sub, Finset.sum_sub_distrib]
  simp only [eqIndex_into, ← based, zero_add, Nat.add_assoc] at hP hR hI hO hC hQC hQA hQ1 hQ2 ⊢
  replace hP : _ = ∑ t : Fin (B.planes (groupOrder n k)), eqAt (F.tau1.map Ext.into) row *
      (β (groupOrder n k) t * (partialForm (B.opening (groupOrder n k)) (B.fold (groupOrder n k))
        (challenge (groupOrder n k)) t).scan (α.into : E) fun o => (o.map fun x =>
          (B.gaps F.digits).place (B.relocate (groupOrder n k) x)).elim 0 (eqAt (r2.map Ext.into))) :=
    hP
  replace hC : _ = -∑ t : Fin (B.planes (groupOrder n k)), eqAt (F.tau1.map Ext.into) row *
      (β (groupOrder n k) t * (contractForm (B.opening (groupOrder n k)) (B.fold (groupOrder n k))
        (β (groupOrder n k)) (st.point (groupOrder n k)) t).scan (α.into : E) fun o => (o.map fun x =>
          (B.gaps F.digits).place (B.relocate (groupOrder n k) x)).elim 0 (eqAt (r2.map Ext.into))) :=
    hC
  replace hQC : _ = -∑ t : Fin (B.planes (groupOrder n k)), eqAt (F.tau1.map Ext.into) row *
      (β (groupOrder n k) t * (((α.into : E) ^ B.carrierDegree (groupOrder n k) + 1) *
        quotientScan (α.into : E) (r2.map Ext.into) F.base F.digits (B.carrierDegree (groupOrder n k))
          fun a j => start + (a * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
            (t * B.carrierDegree (groupOrder n k) + j)))) :=
    hQC
  replace hR : _ = -∑ a : Fin (B.shape (groupOrder n k)).innerRank,
      eqAt (F.tau1.map Ext.into) (row + (1 + a)) * (responseForm (B.fold (groupOrder n k))
        (st.key (groupOrder n k)) a).scan (α.into : E) fun o => (o.map fun x =>
          (B.gaps F.digits).place (B.relocate (groupOrder n k) x)).elim 0 (eqAt (r2.map Ext.into)) :=
    hR
  replace hI : _ = ∑ a : Fin (B.shape (groupOrder n k)).innerRank,
      eqAt (F.tau1.map Ext.into) (row + (1 + a)) * (imageForm (B.opening (groupOrder n k))
        (B.fold (groupOrder n k)) (challenge (groupOrder n k)) a).scan (α.into : E) fun o =>
          (o.map fun x => (B.gaps F.digits).place (B.relocate (groupOrder n k) x)).elim 0
            (eqAt (r2.map Ext.into)) :=
    hI
  replace hO : _ = ∑ j : Fin ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank),
      eqAt (F.tau1.map Ext.into) (row + (1 + ((B.shape (groupOrder n k)).innerRank + j))) *
        (outerForm (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (st.key (groupOrder n k))
          j).scan (α.into : E) fun o => (o.map fun x =>
            (B.gaps F.digits).place (B.relocate (groupOrder n k) x)).elim 0 (eqAt (r2.map Ext.into)) :=
    hO
  linear_combination hP + hR + hI + hO + hC + hQC + hQA + hQ1 + hQ2

end Group

section Shared

variable {n : ℕ} (B : Batch (Fin n))

theorem openSetupCode_code (g : Fin n) (c : (B.opening g).OpenColumn) :
    (B.openSetupCode ⟨g, c⟩ : ℕ) = codeOffset (fun g => (B.shape g).blocks *
      ((B.opening g).subcolumns * (B.shape g).openingDigits)) g + ((B.opening g).openSetupCode c : ℕ) :=
  place_symm_code (fun g => (B.shape g).blocks * ((B.opening g).subcolumns *
    (B.shape g).openingDigits)) g ((B.opening g).openSetupCode c)

variable {E : Type} [Field E] [Algebra Zq E]

/-- The rows (106) of `D` as the sums over the groups of their columns: the column `c` of the
group `groupOrder n k` is the column `codeOffset + openSetupCode c` of the view of `D`. -/
theorem openRows_batch (setup : Array Fq) (key : B.OpenKey)
    (opened : key.matrix = Akita.Setup.shared (flatOf setup) B.openDegree B.openRank _ B.openSetupCode)
    (α : E) (τ r : List E) (D0 digits : ℕ) :
    ∑ x : Fin (1 * B.openRank), eqAt τ (D0 + x) * (B.openForm key x).scan α
        (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r)) =
      ∑ k : Fin n, ∑ c : (B.opening (groupOrder n k)).OpenColumn, ∑ i : Fin B.openRank,
        eqAt τ (D0 + i) * ∑ j ∈ Finset.range B.openDegree, Polynomial.aeval α
          (rep (Poly.toRq B.openDegree (viewEntry setup B.openDegree
            (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)) i
            (codeOffset (fun g => (B.shape g).blocks * ((B.opening g).subcolumns *
              (B.shape g).openingDigits)) (groupOrder n k) +
                ((B.opening (groupOrder n k)).openSetupCode c : ℕ))))) * α ^ j *
          (((B.opening (groupOrder n k)).partialCell (B.fold (groupOrder n k)) c j).map fun x =>
            (B.gaps digits).place (B.relocate (groupOrder n k) x)).elim 0 (eqAt r) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type, Fin.sum_univ_one]
  simp only [Batch.openForm, Form.scan_ofMatrix, Equiv.symm_apply_apply, finProdFinEquiv_apply_val,
    Fin.val_eq_zero, mul_zero, add_zero, opened, shared_flatOf, Finset.mul_sum]
  rw [Finset.sum_comm, Fintype.sum_sigma, ← Equiv.sum_comp (groupOrder n)]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun c _ =>
    Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [openSetupCode_code, Option.map_map]
  rfl

variable (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E) (digits : ℕ)

theorem scalar_place (g : Fin n) (r : List E) (y : ℕ) :
    eqAt r ((B.gaps digits).place (B.relocate g y)) *
        Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g) y =
      eqAt r (B.relocate g y) * Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g) y := by
  by_cases small : y < (B.opening g).chainStart (B.fold g)
  · rw [Gaps.place_of_lt _ (B.relocate_lt_chainStart g small)]
  · rw [Akita.scalarWeight_of_le _ _ _ _ (Nat.le_of_not_lt small), mul_zero, mul_zero]

end Shared

section Places

variable {n : ℕ} (B : Batch (Fin n)) (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext)
  (w : Fin n → Ext) (digits : ℕ)

/-- The cells of the code of the shared path of `D` and the quotient segments of a fold with
opening groups: `Placement.of` is `Batch.gaps` (Akita (76)). -/
theorem Placement.of_batch :
    (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩ digits).quotientMain =
        B.chainStart ∧
      (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩ digits).openFirst =
        B.openFirstStart + digits * B.mainDegrees ∧
      (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩
          digits).quotientFirst = B.secondStart + digits * B.mainDegrees ∧
      (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩
          digits).openSecond = B.openSecondStart + digits * B.mainDegrees + digits * B.firstDegrees ∧
      (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩
          digits).quotientSecond = B.cells + digits * B.mainDegrees + digits * B.firstDegrees := by
  have total := batchGroups_sum B cs amb pts w (·.mainCells) B.mainCells (fun _ => rfl)
  have degrees := batchGroups_sum B cs amb pts w (·.mainDegrees) B.groupDegrees (fun _ => rfl)
  have firsts := batchGroups_sum B cs amb pts w (fun Q => firstCells Q.shape.chain)
    (fun g => firstCells (B.shape g).chain) (fun _ => rfl)
  have firstRows := batchGroups_sum B cs amb pts w
    (fun Q => firstRank Q.shape.chain * firstDegree Q.shape.chain)
    (fun g => firstRank (B.shape g).chain * firstDegree (B.shape g).chain) (fun _ => rfl)
  have seconds := batchGroups_sum B cs amb pts w (fun Q => secondCells Q.shape.chain)
    (fun g => secondCells (B.shape g).chain) (fun _ => rfl)
  simp only [Placement.of, total, degrees, firsts, firstRows, seconds]
  unfold Batch.mainDegrees Batch.firstDegrees Batch.cells Batch.openSecondStart Batch.secondStart
    Batch.openFirstStart Batch.chainStart
  rw [one_mul]
  refine ⟨rfl, by ring, by ring, by ring, by ring⟩

theorem Placement.of_batch_spans :
    (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩ digits).firstSpan =
        B.chainStart + digits * B.mainDegrees ∧
      (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩
          digits).secondSpan = B.secondStart + digits * B.mainDegrees + digits * B.firstDegrees := by
  have total := batchGroups_sum B cs amb pts w (·.mainCells) B.mainCells (fun _ => rfl)
  have degrees := batchGroups_sum B cs amb pts w (·.mainDegrees) B.groupDegrees (fun _ => rfl)
  have firsts := batchGroups_sum B cs amb pts w (fun Q => firstCells Q.shape.chain)
    (fun g => firstCells (B.shape g).chain) (fun _ => rfl)
  have firstRows := batchGroups_sum B cs amb pts w
    (fun Q => firstRank Q.shape.chain * firstDegree Q.shape.chain)
    (fun g => firstRank (B.shape g).chain * firstDegree (B.shape g).chain) (fun _ => rfl)
  simp only [Placement.of, total, degrees, firsts, firstRows]
  unfold Batch.mainDegrees Batch.firstDegrees Batch.secondStart Batch.openFirstStart Batch.chainStart
  rw [one_mul]
  refine ⟨rfl, by ring⟩

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] in
/-- The binary-support weight of a fold with opening groups: `foldSupport` is
`1_{I_bin}·ẽq(r_virt, ·)` at `r_2` for the support `Gaps.support` of `Batch.gaps` (Akita (76)):
the digits of `F_1` of each group and of `H_1`, and of `F_2` of each group and of `H_2`, at their
cells in the code. -/
theorem foldSupport_batch (virt eq : ℕ → Ext) {μ : ℕ} {r r2 : List E}
    (room : B.quotientRoom digits ≤ 2 ^ μ) (length2 : r2.length = μ)
    (virtual : ∀ u < 2 ^ μ, ((virt u).into : E) = eqAt r u)
    (same : ∀ x < 2 ^ μ, ((eq x).into : E) = eqAt r2 x) :
    ((foldSupport (Placement.of (batchGroups B cs amb pts w) ⟨B.openDegree, B.openRank, B.chain⟩
      digits) virt eq).into : E) =
      mle r2 fun x => if (B.gaps digits).support x then eqAt r x else 0 := by
  have placed := Placement.of_batch B cs amb pts w digits
  have spans := Placement.of_batch_spans B cs amb pts w digits
  have counts := B.gaps_counts digits
  have ordered := (B.gaps digits).ordered
  have room' : B.cells + (B.gaps digits).mainCells + (B.gaps digits).middleCells ≤ 2 ^ μ := by
    unfold Batch.quotientRoom at room
    omega
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length2]
  simp only [mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  have support : (Finset.range (2 ^ μ)).filter (B.gaps digits).support =
      Finset.Ico (B.chainStart + digits * B.mainDegrees) (B.secondStart + digits * B.mainDegrees) ∪
        Finset.Ico (B.secondStart + digits * B.mainDegrees + digits * B.firstDegrees)
          (B.cells + digits * B.mainDegrees + digits * B.firstDegrees) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_union, Finset.mem_Ico, Gaps.support]
    show _ ↔ _ ∨ _
    simp only [Batch.gaps] at room' ordered ⊢
    omega
  have apart : Disjoint (Finset.Ico (B.chainStart + digits * B.mainDegrees)
      (B.secondStart + digits * B.mainDegrees))
        (Finset.Ico (B.secondStart + digits * B.mainDegrees + digits * B.firstDegrees)
          (B.cells + digits * B.mainDegrees + digits * B.firstDegrees)) :=
    Finset.disjoint_left.mpr fun x first second => by
      simp only [Finset.mem_Ico] at first second
      omega
  rw [support, Finset.sum_union apart, Finset.sum_Ico_eq_sum_range, Finset.sum_Ico_eq_sum_range]
  unfold foldSupport
  rw [placed.2.2.1, placed.2.2.2.2, spans.1, spans.2]
  simp only [Ext.into_add, sumList_into, sum_list_range]
  have inside : B.cells + digits * B.mainDegrees + digits * B.firstDegrees ≤ 2 ^ μ := room'
  have below : B.chainStart ≤ B.secondStart ∧ B.secondStart ≤ B.cells := ordered
  congr 1
  · refine Finset.sum_congr rfl fun k member => ?_
    have := Finset.mem_range.mp member
    rw [Ext.into_mul, virtual _ (by omega), same _ (by omega), mul_comm]
  · refine Finset.sum_congr rfl fun k member => ?_
    have := Finset.mem_range.mp member
    rw [Ext.into_mul, virtual _ (by omega), same _ (by omega), mul_comm]

end Places

end NightstreamFPrime.Spec.Compression.Akita.Replay
