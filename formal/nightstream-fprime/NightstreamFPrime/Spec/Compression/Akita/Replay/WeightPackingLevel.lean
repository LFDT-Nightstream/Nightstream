import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingQuotient

/-! The weight at `r_2` of a packing level with one group against the specification (ePrint
2026/1983, (111) to (113) and (123); `liftedReduction` of `Extract/LiftSource.lean`).

- `flatTail` is the tail of `flatRows`: the ring rows after the planes of the carrier row.
- `liftedReduction_weight`: `m_{τ_1}(r_2)` of `liftedReduction` is the planes of the carrier row
  with `ẽq(τ_1, 0)·β_t`, the tail row `i` with `ẽq(τ_1, 1 + i)` and the scalar row with
  `ẽq(τ_1, N)`, each ring row as its linear form at the cells `place` minus `(α^{d_r} + 1)` times
  its quotient digits at the cells `slot` of their entries (348). `flatTail_weight` splits the
  tail rows into their seven families with the offsets of their quotient digits.
- `PackGroup.Fits` are the numbers of one group that the theorem reads (powers of two, degrees at
  most `max(d_D, d_A)`, the runs of the compression digits).
- `foldWeight_one`: the weight of `foldFused` of the replay at `r_2` is `m_{τ_1}(r_2)` of
  `liftedReduction`, for a statement of the views of the setup array and the weights of the
  opening point of the group. `foldSupport_one`: the binary-support weight is the multilinear
  extension of `1_{I_bin}·ẽq(r_virt, ·)` for `Gaps.support`. `packingFused_holds_iff`: the check
  `fused.end` is the final check of `fusedCheck`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Spec

variable {E : Type} [Field E] [Algebra Zq E] {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- The ring rows of `flatRows` after the planes of the carrier row (102): the rows (104), (105),
(106) and (47). -/
noncomputable def flatTail (st : Akita.Statement O E) (challenge : Fin S.blocks → Rq O.carrierDegree) :
    Fin O.tailCount → RingRow :=
  Fin.append (innerRow O fold st.key challenge)
    (Fin.append (imageRow S.chain (outerForm O fold st.key) (O.firstCell fold) st.payload)
      (Fin.append
        (imageRow O.chain (openForm O fold st.openKey) (O.openFirstCell fold) st.openPayload)
        (Fin.append (firstRow S.chain st.key.chain (O.firstCell fold) (O.secondCell fold))
          (Fin.append
            (firstRow O.chain st.openKey.chain (O.openFirstCell fold) (O.openSecondCell fold))
            (Fin.append (secondRow S.chain st.key.chain (O.secondCell fold) st.payload)
              (secondRow O.chain st.openKey.chain (O.openSecondCell fold) st.openPayload))))))

variable {O fold} (β : Module.Basis (Fin O.planes) Zq E) (st : Akita.Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree)

theorem flatRows_castAdd (t : Fin O.planes) :
    flatRows O fold β st challenge (Fin.castAdd O.tailCount t) =
      evaluationRow O fold β st.point challenge t :=
  Fin.append_left _ _ t

theorem evaluationRow_degree (point : Point O E) (t : Fin O.planes) :
    (evaluationRow O fold β point challenge t).degree = O.carrierDegree :=
  rfl

theorem relocate_degree (row : RingRow) (move : ℕ → ℕ) : (row.relocate move).degree = row.degree :=
  rfl

theorem flatRows_natAdd (i : Fin O.tailCount) :
    flatRows O fold β st challenge (Fin.natAdd O.planes i) = flatTail O fold st challenge i :=
  Fin.append_right _ _ i

variable [Fintype E]

/-- `m_{τ_1}(r_2)` of `liftedReduction` (Akita (123) at `r_2`): plane `t` of the carrier row with
`ẽq(τ_1, 0)·β_t`, the tail row `i` with `ẽq(τ_1, 1 + i)` and the scalar row (100) with
`ẽq(τ_1, N)`. Each ring row is its linear form at the cells `place` minus `(α^{d_r} + 1)` times
its quotient digits at the cells `slot` of their entries (348). -/
theorem liftedReduction_weight {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) (digits : ℕ)
    (room : O.quotientRoom fold digits ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
    (α : E) {τ r : List E} (length : r.length = μ) (tau : τ.length = batch) :
    mle r (((liftedReduction β st challenge μ fits digits room batch rowsFit).rows α).batchWeight
      τ) =
      ∑ t : Fin O.planes, eqAt τ 0 * (β t * ((evaluationRow O fold β st.point challenge t).scan α
          (fun o => (o.map (O.gaps fold digits).place).elim 0 (eqAt r)) -
        (α ^ O.carrierDegree + 1) * quotientScan α r S.certifiedBase digits O.carrierDegree
          fun a j => (O.gaps fold digits).slot (a * (O.planes * O.carrierDegree) +
            (t * O.carrierDegree + j)))) +
      ∑ i : Fin O.tailCount, eqAt τ (1 + i) * ((flatTail O fold st challenge i).scan α
          (fun o => (o.map (O.gaps fold digits).place).elim 0 (eqAt r)) -
        (α ^ (flatTail O fold st challenge i).degree + 1) * quotientScan α r S.certifiedBase digits
          (flatTail O fold st challenge i).degree fun a j => (O.gaps fold digits).slot
            (digits * (O.planes * O.carrierDegree) +
              digits * offset (fun i' => (flatTail O fold st challenge i').degree) i +
                (a * (flatTail O fold st challenge i).degree + j))) +
      eqAt τ (1 + O.tailCount) * mle r (scalarWeight O fold β st.point) := by
  have rowInside : ∀ (q : Fin (O.planes + O.tailCount)) c j x,
      j < (flatRows O fold β st challenge q).degree →
      ((flatRows O fold β st challenge q).relocate (O.gaps fold digits).place).address c j =
        some x → x < 2 ^ r.length := by
    intro q c j x _ read
    cases h : (flatRows O fold β st challenge q).address c j with
    | none =>
        simp only [RingRow.relocate, h, Option.map_none] at read
        cases read
    | some y =>
        simp only [RingRow.relocate, h, Option.map_some, Option.some.injEq] at read
        rw [← read, length]
        exact (O.place_lt_room fold digits (flatRows_below O fold β st challenge q c j y h)).trans_le
          room
  have quotientInside : ∀ (q : Fin (O.planes + O.tailCount)) j (a : Fin digits),
      j < (flatRows O fold β st challenge q).degree →
      (O.gaps fold digits).slot (quotientEntry (flatSlot O.planes O.tailCount)
        (fun q : Fin (O.planes + O.tailCount) => (flatRows O fold β st challenge q).degree) digits q j a) < 2 ^ r.length := by
    intro q j a small
    rw [length]
    exact ((flatRelation O fold β st challenge μ fits).quotientCell_lt _ digits q small a).trans_le
      ((flatRelation_quotientEnd β st challenge μ fits digits).trans_le room)
  unfold quotientScan
  refine (LiftRelation.mle_lift _ _ _ _ _ α τ r tau).trans ?_
  show ∑ q : Fin (O.planes + O.tailCount), eqAt τ (flatSlot O.planes O.tailCount q : ℕ) *
      (Fin.append (⇑β) (fun _ => (1 : E)) q * mle r (((flatRows O fold β st challenge q).relocate
        (O.gaps fold digits).place).liftWeight ⟨digits, fun a => (S.certifiedBase : Zq) ^ (a : ℕ),
          fun j a => (O.gaps fold digits).slot (quotientEntry (flatSlot O.planes O.tailCount)
            (fun q : Fin (O.planes + O.tailCount) => (flatRows O fold β st challenge q).degree) digits q j a)⟩ α)) +
    eqAt τ (1 + O.tailCount) * mle r (scalarWeight O fold β st.point) = _
  have planes : ∀ t : Fin O.planes,
      (flatRows O fold β st challenge (Fin.castAdd O.tailCount t)).degree = O.carrierDegree :=
    fun t => by simp only [flatRows_castAdd, evaluationRow_degree]
  rw [Fin.sum_univ_add]
  congr 2
  · refine Finset.sum_congr rfl fun t _ => ?_
    rw [RingRow.mle_liftWeight _ ⟨digits, fun a => (S.certifiedBase : Zq) ^ (a : ℕ), fun j a =>
        (O.gaps fold digits).slot (quotientEntry (flatSlot O.planes O.tailCount)
          (fun q : Fin (O.planes + O.tailCount) => (flatRows O fold β st challenge q).degree) digits (Fin.castAdd O.tailCount t)
            j a)⟩ α r (rowInside _) (fun j a small => quotientInside _ j a small),
      RingRow.scan_relocate, flatSlot_left, Fin.append_left, relocate_degree]
    simp only [flatSlot_entry_plane (fun q : Fin (O.planes + O.tailCount) =>
        (flatRows O fold β st challenge q).degree) O.carrierDegree digits planes, flatRows_castAdd,
      evaluationRow_degree, map_pow, map_natCast]
    rfl
  · refine Finset.sum_congr rfl fun i _ => ?_
    rw [RingRow.mle_liftWeight _ ⟨digits, fun a => (S.certifiedBase : Zq) ^ (a : ℕ), fun j a =>
        (O.gaps fold digits).slot (quotientEntry (flatSlot O.planes O.tailCount)
          (fun q : Fin (O.planes + O.tailCount) => (flatRows O fold β st challenge q).degree) digits (Fin.natAdd O.planes i)
            j a)⟩ α r (rowInside _) (fun j a small => quotientInside _ j a small),
      RingRow.scan_relocate, flatSlot_right, Fin.append_right, one_mul, relocate_degree]
    simp only [flatSlot_entry_tail (fun q : Fin (O.planes + O.tailCount) =>
        (flatRows O fold β st challenge q).degree) O.carrierDegree digits planes, flatRows_natAdd, map_pow,
      map_natCast, Fin.val_natAdd]

theorem innerRow_degree (key : Key S) (a : Fin S.innerRank) :
    (innerRow O fold key challenge a).degree = S.innerDegree :=
  rfl

theorem sum_seven {M : Type} [AddCommMonoid M] {n₁ n₂ n₃ n₄ n₅ n₆ n₇ : ℕ}
    (f : Fin (n₁ + (n₂ + (n₃ + (n₄ + (n₅ + (n₆ + n₇)))))) → M) :
    ∑ i, f i = ∑ x : Fin n₁, f (Fin.castAdd _ x) + (∑ x : Fin n₂, f (Fin.natAdd n₁ (Fin.castAdd _ x)) +
      (∑ x : Fin n₃, f (Fin.natAdd n₁ (Fin.natAdd n₂ (Fin.castAdd _ x))) +
      (∑ x : Fin n₄, f (Fin.natAdd n₁ (Fin.natAdd n₂ (Fin.natAdd n₃ (Fin.castAdd _ x)))) +
      (∑ x : Fin n₅, f (Fin.natAdd n₁ (Fin.natAdd n₂ (Fin.natAdd n₃ (Fin.natAdd n₄
        (Fin.castAdd _ x))))) +
      (∑ x : Fin n₆, f (Fin.natAdd n₁ (Fin.natAdd n₂ (Fin.natAdd n₃ (Fin.natAdd n₄
        (Fin.natAdd n₅ (Fin.castAdd _ x)))))) +
      ∑ x : Fin n₇, f (Fin.natAdd n₁ (Fin.natAdd n₂ (Fin.natAdd n₃ (Fin.natAdd n₄
        (Fin.natAdd n₅ (Fin.natAdd n₆ x))))))))))) := by
  rw [Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add,
    Fin.sum_univ_add]

theorem offset_castAdd {a b : ℕ} (size : Fin (a + b) → ℕ) (i : Fin a) :
    offset size (i : ℕ) = offset (fun j => size (Fin.castAdd b j)) i := by
  unfold offset
  rw [Fin.sum_univ_add, Finset.sum_eq_zero (s := Finset.univ) (f := fun j : Fin b =>
    if ((Fin.natAdd a j : Fin (a + b)) : ℕ) < i then size (Fin.natAdd a j) else 0)
    fun j _ => ite_eq_right fun before => by simp only [Fin.val_natAdd] at before; omega, add_zero]
  rfl

theorem offset_natAdd {a b : ℕ} (size : Fin (a + b) → ℕ) (j : Fin b) :
    offset size (a + (j : ℕ)) =
      (∑ i, size (Fin.castAdd b i)) + offset (fun k => size (Fin.natAdd a k)) j := by
  unfold offset
  rw [Fin.sum_univ_add]
  congr 1
  · exact Finset.sum_congr rfl fun i _ => ite_eq_left (by simp only [Fin.val_castAdd]; omega)
  · exact Finset.sum_congr rfl fun k _ => by simp only [Fin.val_natAdd, Nat.add_lt_add_iff_left]

theorem offset_const {n : ℕ} (d : ℕ) (i : Fin n) : offset (fun _ : Fin n => d) i = i * d := by
  unfold offset
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < (i : ℕ) then d else 0), sum_range_ite_lt,
    min_eq_left i.isLt.le]

omit [Field E] [Algebra Zq E] [Fintype E] in
/-- The size of the tail rows before the tail row `i`, as a sum over `Fin` of the seven
families. -/
theorem flatTail_offset (i : ℕ) :
    offset (fun i' => (flatTail O fold st challenge i').degree) i =
      offset (n := S.innerRank + (S.slices * S.outerRank + (1 * O.openRank + (firstRank S.chain +
        (firstRank O.chain + (secondRank S.chain + secondRank O.chain))))))
        (Fin.append (fun _ => S.innerDegree) (Fin.append (fun _ => S.outerDegree)
          (Fin.append (fun _ => O.openDegree) (Fin.append (fun _ => firstDegree S.chain)
            (Fin.append (fun _ => firstDegree O.chain) (Fin.append (fun _ => secondDegree S.chain)
              (fun _ => secondDegree O.chain))))))) i := by
  unfold offset
  refine Finset.sum_congr rfl fun i' _ => ?_
  congr 1
  refine Fin.addCases (fun x => ?_) (fun i' => ?_) i'
  · simp only [flatTail, Fin.append_left, innerRow_degree]
  refine Fin.addCases (fun x => ?_) (fun i' => ?_) i'
  · simp only [flatTail, Fin.append_left, Fin.append_right, imageRow_degree]
  refine Fin.addCases (fun x => ?_) (fun i' => ?_) i'
  · simp only [flatTail, Fin.append_left, Fin.append_right, imageRow_degree]
  refine Fin.addCases (fun x => ?_) (fun i' => ?_) i'
  · simp only [flatTail, Fin.append_left, Fin.append_right, firstRow_degree]
  refine Fin.addCases (fun x => ?_) (fun i' => ?_) i'
  · simp only [flatTail, Fin.append_left, Fin.append_right, firstRow_degree]
  refine Fin.addCases (fun x => ?_) (fun x => ?_) i'
  · simp only [flatTail, Fin.append_left, Fin.append_right, secondRow_degree]
  · simp only [flatTail, Fin.append_right, secondRow_degree]

theorem offset_append_castAdd {a b : ℕ} (f : Fin a → ℕ) (g : Fin b → ℕ) (i : Fin a) :
    offset (Fin.append f g) (i : ℕ) = offset f i := by
  rw [offset_castAdd]
  simp only [Fin.append_left]

theorem offset_append_natAdd {a b : ℕ} (f : Fin a → ℕ) (g : Fin b → ℕ) (k : ℕ) :
    offset (Fin.append f g) (a + k) = (∑ i, f i) + offset g k := by
  unfold offset
  rw [Fin.sum_univ_add]
  congr 1
  · exact Finset.sum_congr rfl fun i _ => by
      rw [ite_eq_left (by simp only [Fin.val_castAdd]; omega), Fin.append_left]
  · exact Finset.sum_congr rfl fun j _ => by
      simp only [Fin.val_natAdd, Nat.add_lt_add_iff_left, Fin.append_right]

omit [Fintype E] in
/-- The tail rows of `liftedReduction_weight` in their seven families, with the offsets of their
quotient digits. -/
theorem flatTail_weight (α : E) (τ r : List E) (b digits base : ℕ) (φ : Option ℕ → E)
    (slot : ℕ → ℕ) :
    ∑ i : Fin O.tailCount, eqAt τ (1 + i) * ((flatTail O fold st challenge i).scan α φ -
        (α ^ (flatTail O fold st challenge i).degree + 1) * quotientScan α r b digits
          (flatTail O fold st challenge i).degree fun a j => slot (base +
            digits * offset (fun i' => (flatTail O fold st challenge i').degree) i +
              (a * (flatTail O fold st challenge i).degree + j))) =
      ∑ x : Fin S.innerRank, eqAt τ (1 + x) * ((innerRow O fold st.key challenge x).scan α φ -
        (α ^ S.innerDegree + 1) * quotientScan α r b digits S.innerDegree fun a j =>
          slot (base + digits * (x * S.innerDegree) + (a * S.innerDegree + j))) +
      (∑ x : Fin (S.slices * S.outerRank), eqAt τ (1 + (S.innerRank + x)) *
        ((imageRow S.chain (outerForm O fold st.key) (O.firstCell fold) st.payload x).scan α φ -
          (α ^ S.outerDegree + 1) * quotientScan α r b digits S.outerDegree fun a j =>
            slot (base + digits * (S.innerRank * S.innerDegree + x * S.outerDegree) +
              (a * S.outerDegree + j))) +
      (∑ x : Fin (1 * O.openRank), eqAt τ (1 + (S.innerRank + (S.slices * S.outerRank + x))) *
        ((imageRow O.chain (openForm O fold st.openKey) (O.openFirstCell fold) st.openPayload
            x).scan α φ -
          (α ^ O.openDegree + 1) * quotientScan α r b digits O.openDegree fun a j =>
            slot (base + digits * (S.innerRank * S.innerDegree +
              S.slices * S.outerRank * S.outerDegree + x * O.openDegree) +
                (a * O.openDegree + j))) +
      (∑ x : Fin (firstRank S.chain), eqAt τ (1 + (S.innerRank + (S.slices * S.outerRank +
          (1 * O.openRank + x)))) *
        ((firstRow S.chain st.key.chain (O.firstCell fold) (O.secondCell fold) x).scan α φ -
          (α ^ firstDegree S.chain + 1) * quotientScan α r b digits (firstDegree S.chain)
            fun a j => slot (base + digits * (S.innerRank * S.innerDegree +
              S.slices * S.outerRank * S.outerDegree + 1 * O.openRank * O.openDegree +
                x * firstDegree S.chain) + (a * firstDegree S.chain + j))) +
      (∑ x : Fin (firstRank O.chain), eqAt τ (1 + (S.innerRank + (S.slices * S.outerRank +
          (1 * O.openRank + (firstRank S.chain + x))))) *
        ((firstRow O.chain st.openKey.chain (O.openFirstCell fold) (O.openSecondCell fold)
            x).scan α φ -
          (α ^ firstDegree O.chain + 1) * quotientScan α r b digits (firstDegree O.chain)
            fun a j => slot (base + digits * (S.innerRank * S.innerDegree +
              S.slices * S.outerRank * S.outerDegree + 1 * O.openRank * O.openDegree +
                firstRank S.chain * firstDegree S.chain + x * firstDegree O.chain) +
                  (a * firstDegree O.chain + j))) +
      (∑ x : Fin (secondRank S.chain), eqAt τ (1 + (S.innerRank + (S.slices * S.outerRank +
          (1 * O.openRank + (firstRank S.chain + (firstRank O.chain + x)))))) *
        ((secondRow S.chain st.key.chain (O.secondCell fold) st.payload x).scan α φ -
          (α ^ secondDegree S.chain + 1) * quotientScan α r b digits (secondDegree S.chain)
            fun a j => slot (base + digits * (S.innerRank * S.innerDegree +
              S.slices * S.outerRank * S.outerDegree + 1 * O.openRank * O.openDegree +
                firstRank S.chain * firstDegree S.chain + firstRank O.chain * firstDegree O.chain +
                  x * secondDegree S.chain) + (a * secondDegree S.chain + j))) +
      ∑ x : Fin (secondRank O.chain), eqAt τ (1 + (S.innerRank + (S.slices * S.outerRank +
          (1 * O.openRank + (firstRank S.chain + (firstRank O.chain + (secondRank S.chain +
            x))))))) *
        ((secondRow O.chain st.openKey.chain (O.openSecondCell fold) st.openPayload x).scan α φ -
          (α ^ secondDegree O.chain + 1) * quotientScan α r b digits (secondDegree O.chain)
            fun a j => slot (base + digits * (S.innerRank * S.innerDegree +
              S.slices * S.outerRank * S.outerDegree + 1 * O.openRank * O.openDegree +
                firstRank S.chain * firstDegree S.chain + firstRank O.chain * firstDegree O.chain +
                  secondRank S.chain * secondDegree S.chain + x * secondDegree O.chain) +
                    (a * secondDegree O.chain + j)))))))) := by
  simp only [flatTail_offset]
  refine (sum_seven (n₁ := S.innerRank) (n₂ := S.slices * S.outerRank) (n₃ := 1 * O.openRank)
    (n₄ := firstRank S.chain) (n₅ := firstRank O.chain) (n₆ := secondRank S.chain)
    (n₇ := secondRank O.chain) _).trans ?_
  simp only [flatTail, Fin.append_left, Fin.append_right, Fin.val_castAdd, Fin.val_natAdd,
    offset_append_castAdd, offset_append_natAdd, innerRow_degree, imageRow_degree, firstRow_degree,
    secondRow_degree, offset_const, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    smul_eq_mul, Nat.add_assoc]

end Spec

theorem filterMap_some_map {α β : Type} (f : α → β) (l : List α) :
    l.filterMap (fun x => some (f x)) = l.map f := by
  induction l with
  | nil => rfl
  | cons x l ih => simp only [List.filterMap_cons, ih, List.map_cons]

theorem zipIdx_map_fst {α β : Type} (f : α → β) : ∀ (l : List α) (k : ℕ),
    (l.zipIdx k).map (fun x => f x.1) = l.map f
  | [], _ => rfl
  | x :: l, k => by rw [List.zipIdx_cons, List.map_cons, List.map_cons, zipIdx_map_fst f l]


theorem lv_into {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E] (T : Tables) (l : List Lift) :
    ((liftsValue T l).into : E) = (l.map fun x => ((x.value T).into : E)).sum :=
  sumList_into _ _

theorem zipIdx_map_lift {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E] (T : Tables)
    (l : List (List Lift)) (k : ℕ) :
    (l.zipIdx k).map (fun x => ((liftsValue T x.1).into : E)) =
      l.map fun p => ((liftsValue T p).into : E) :=
  zipIdx_map_fst (fun p => ((liftsValue T p).into : E)) l k

theorem segCell_eta (s count : ℕ) :
    (fun position => if position < count then some (s + position) else none) = segCell s count :=
  rfl

section Cells

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

theorem firstCell_eq : O.firstCell fold = segCell (O.chainStart fold) (firstCells S.chain) := rfl

theorem secondCell_eq : O.secondCell fold = segCell (O.secondStart fold) (secondCells S.chain) := rfl

theorem openFirstCell_eq :
    O.openFirstCell fold = segCell (O.openFirstStart fold) (firstCells O.chain) := rfl

theorem openSecondCell_eq :
    O.openSecondCell fold = segCell (O.openSecondStart fold) (secondCells O.chain) := rfl

end Cells

/-- The cells of the code of a fold with one group: the quotient digits of the rows before the
compression rows after `t̂`, then the digits of `F_1` and `H_1` and their quotient digits, then the
digits of `F_2` and `H_2` and their quotient digits. -/
theorem Placement.of_one (P : PackGroup) (digits : ℕ) :
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩ digits).quotientMain =
        P.opening.chainStart P.fold ∧
      (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          digits).openFirst =
        P.opening.chainStart P.fold + digits * P.opening.mainDegrees + firstCells P.shape.chain ∧
      (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          digits).quotientFirst =
        P.opening.secondStart P.fold + digits * P.opening.mainDegrees ∧
      (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          digits).openSecond =
        P.opening.secondStart P.fold + digits * P.opening.mainDegrees +
          digits * P.opening.firstDegrees + secondCells P.shape.chain ∧
      (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          digits).quotientSecond =
        P.opening.cells P.fold + digits * P.opening.mainDegrees + digits * P.opening.firstDegrees := by
  simp only [Placement.of, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    PackGroup.mainCells, PackGroup.mainDegrees, OpeningShape.mainDegrees, OpeningShape.firstDegrees,
    OpeningShape.secondStart, OpeningShape.openFirstStart, OpeningShape.cells,
    OpeningShape.openSecondStart]
  refine ⟨trivial, by ring, by ring, by ring, by ring⟩

section Level

/-- The numbers of one group that the weight theorem of a packing level reads: each ring degree is
a power of two and at most `max(d_D, d_A)`, the degree of `B` and of `D` divides the digits of
their first compression map and the degree of the first map the digits of the second, and the
largest degree is at most `2^μ`. -/
structure PackGroup.Fits (P : PackGroup) (μ : ℕ) : Prop where
  carrier : ∃ k, 2 ^ k = P.opening.carrierDegree
  inner : ∃ k, 2 ^ k = P.shape.innerDegree
  outer : ∃ k, 2 ^ k = P.shape.outerDegree
  opened : ∃ k, 2 ^ k = P.opening.openDegree
  first : ∃ k, 2 ^ k = firstDegree P.shape.chain
  second : ∃ k, 2 ^ k = secondDegree P.shape.chain
  openFirst : ∃ k, 2 ^ k = firstDegree P.opening.chain
  openSecond : ∃ k, 2 ^ k = secondDegree P.opening.chain
  carrierTop : P.opening.carrierDegree ≤ max P.opening.openDegree P.shape.innerDegree
  outerTop : P.shape.outerDegree ≤ max P.opening.openDegree P.shape.innerDegree
  firstTop : firstDegree P.shape.chain ≤ max P.opening.openDegree P.shape.innerDegree
  secondTop : secondDegree P.shape.chain ≤ max P.opening.openDegree P.shape.innerDegree
  openFirstTop : firstDegree P.opening.chain ≤ max P.opening.openDegree P.shape.innerDegree
  openSecondTop : secondDegree P.opening.chain ≤ max P.opening.openDegree P.shape.innerDegree
  image : P.shape.outerDegree ∣ firstCells P.shape.chain
  chained : firstDegree P.shape.chain ∣ secondCells P.shape.chain
  openImage : P.opening.openDegree ∣ firstCells P.opening.chain
  openChained : firstDegree P.opening.chain ∣ secondCells P.opening.chain
  cube : max P.opening.openDegree P.shape.innerDegree ≤ 2 ^ μ

theorem pow_log_top {d : ℕ} (pow : ∃ k, 2 ^ k = d) : 2 ^ Nat.log 2 d = d := by
  obtain ⟨k, rfl⟩ := pow
  rw [Nat.log_pow (by norm_num)]

theorem log_le_top {d top : ℕ} (le : d ≤ top) : Nat.log 2 d ≤ Nat.log 2 top :=
  Nat.log_mono_right le

theorem log_le_cube {top μ : ℕ} (cube : top ≤ 2 ^ μ) : Nat.log 2 top ≤ μ :=
  (Nat.log_mono_right cube).trans (Nat.log_pow (by norm_num) μ).le

/-- The quotient digits of the rows of a fold with one group end in the cube: the bounds of the
quotient terms of `foldParts`. -/
theorem quotient_insides (P : PackGroup) (digits : ℕ) (r2 : List Ext)
    (cube : P.opening.quotientRoom P.fold digits ≤ 2 ^ r2.length) :
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) ≤
        2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) +
        P.shape.innerRank * (digits * P.shape.innerDegree) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) +
        P.shape.innerRank * (digits * P.shape.innerDegree) +
          P.shape.slices * P.shape.outerRank * (digits * P.shape.outerDegree) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * P.mainDegrees +
        1 * P.opening.openRank * (digits * P.opening.openDegree) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientFirst +
        firstRank P.shape.chain * (digits * firstDegree P.shape.chain) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientFirst +
        firstRank P.shape.chain * (digits * firstDegree P.shape.chain) +
          firstRank P.opening.chain * (digits * firstDegree P.opening.chain) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientSecond +
        secondRank P.shape.chain * (digits * secondDegree P.shape.chain) ≤ 2 ^ r2.length ∧
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientSecond +
        secondRank P.shape.chain * (digits * secondDegree P.shape.chain) +
          secondRank P.opening.chain * (digits * secondDegree P.opening.chain) ≤ 2 ^ r2.length := by
  have placed := Placement.of_one P digits
  have below : ∀ {X : ℕ}, X ≤ P.opening.quotientRoom P.fold digits → X ≤ 2 ^ r2.length :=
    fun le => le.trans cube
  have insideCarrier : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) ≤
        2 ^ r2.length := below (by
    rw [placed.1]
    unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
      OpeningShape.openSecondStart OpeningShape.secondStart OpeningShape.openFirstStart
    ring_nf
    omega)
  have insideA : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) +
        P.shape.innerRank * (digits * P.shape.innerDegree) ≤ 2 ^ r2.length := below (by
    rw [placed.1]
    unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
      OpeningShape.openSecondStart OpeningShape.secondStart OpeningShape.openFirstStart
    ring_nf
    omega)
  have insideB : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * (P.opening.planes * P.opening.carrierDegree) +
        P.shape.innerRank * (digits * P.shape.innerDegree) +
          P.shape.slices * P.shape.outerRank * (digits * P.shape.outerDegree) ≤ 2 ^ r2.length :=
    below (by
      rw [placed.1]
      unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
        OpeningShape.openSecondStart OpeningShape.secondStart OpeningShape.openFirstStart
      ring_nf
      omega)
  have insideD : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientMain + digits * P.mainDegrees +
        1 * P.opening.openRank * (digits * P.opening.openDegree) ≤ 2 ^ r2.length := below (by
    rw [placed.1]
    unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
      OpeningShape.openSecondStart OpeningShape.secondStart OpeningShape.openFirstStart
      PackGroup.mainDegrees
    ring_nf
    omega)
  have insideF1 : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientFirst +
        firstRank P.shape.chain * (digits * firstDegree P.shape.chain) ≤ 2 ^ r2.length := below (by
    rw [placed.2.2.1]
    unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
      OpeningShape.openSecondStart OpeningShape.mainDegrees
    ring_nf
    omega)
  have insideH1 : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientFirst +
        firstRank P.shape.chain * (digits * firstDegree P.shape.chain) +
          firstRank P.opening.chain * (digits * firstDegree P.opening.chain) ≤ 2 ^ r2.length :=
    below (by
      rw [placed.2.2.1]
      unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.cells
        OpeningShape.openSecondStart OpeningShape.mainDegrees
      ring_nf
      omega)
  have insideF2 : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientSecond +
        secondRank P.shape.chain * (digits * secondDegree P.shape.chain) ≤ 2 ^ r2.length :=
    below (by
      rw [placed.2.2.2.2]
      unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.mainDegrees
        OpeningShape.firstDegrees
      ring_nf
      omega)
  have insideH2 : (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits).quotientSecond +
        secondRank P.shape.chain * (digits * secondDegree P.shape.chain) +
          secondRank P.opening.chain * (digits * secondDegree P.opening.chain) ≤ 2 ^ r2.length :=
    below (by
      rw [placed.2.2.2.2]
      unfold OpeningShape.quotientRoom OpeningShape.degreeSum OpeningShape.mainDegrees
        OpeningShape.firstDegrees
      ring_nf
      omega)
  exact ⟨insideCarrier, insideA, insideB, insideD, insideF1, insideH1, insideF2, insideH2⟩


/-- The digit spans of the compression maps of a fold with one group: `F_1` and `H_1` after the
quotient digits `Q̂_main`, `F_2` and `H_2` after `Q̂_1`. -/
theorem Placement.of_one_spans (P : PackGroup) (digits : ℕ) :
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
        digits).firstSpan = P.opening.chainStart P.fold + digits * P.opening.mainDegrees ∧
      (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          digits).secondSpan =
        P.opening.secondStart P.fold + digits * P.opening.mainDegrees +
          digits * P.opening.firstDegrees := by
  simp only [Placement.of, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    PackGroup.mainCells, PackGroup.mainDegrees, OpeningShape.mainDegrees, OpeningShape.firstDegrees,
    OpeningShape.secondStart, OpeningShape.openFirstStart]
  refine ⟨by ring, by ring⟩

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] in
/-- The binary-support weight of a packing level with one group: `foldSupport` is
`1_{I_bin}·ẽq(r_virt, ·)` at `r_2` for the support `Gaps.support` of the witness of Akita (76):
the digits of `F_1` and `H_1` and the digits of `F_2` and `H_2` at their cells in the code. -/
theorem foldSupport_one (P : PackGroup) (digits : ℕ) (virt eq : ℕ → Ext) {μ : ℕ} {r r2 : List E}
    (room : P.opening.quotientRoom P.fold digits ≤ 2 ^ μ) (length2 : r2.length = μ)
    (virtual : ∀ u < 2 ^ μ, ((virt u).into : E) = eqAt r u)
    (same : ∀ x < 2 ^ μ, ((eq x).into : E) = eqAt r2 x) :
    ((foldSupport (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank,
      P.opening.chain⟩ digits) virt eq).into : E) =
      mle r2 fun x => if (P.opening.gaps P.fold digits).support x then eqAt r x else 0 := by
  have placed := Placement.of_one P digits
  have spans := Placement.of_one_spans P digits
  have counts := P.opening.gaps_counts P.fold digits
  have ordered := (P.opening.gaps P.fold digits).ordered
  have room' : P.opening.cells P.fold + (P.opening.gaps P.fold digits).mainCells +
      (P.opening.gaps P.fold digits).middleCells ≤ 2 ^ μ := by
    unfold OpeningShape.quotientRoom at room
    omega
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length2]
  simp only [mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  have support : (Finset.range (2 ^ μ)).filter (P.opening.gaps P.fold digits).support =
      Finset.Ico (P.opening.chainStart P.fold + digits * P.opening.mainDegrees)
          (P.opening.secondStart P.fold + digits * P.opening.mainDegrees) ∪
        Finset.Ico (P.opening.secondStart P.fold + digits * P.opening.mainDegrees +
            digits * P.opening.firstDegrees)
          (P.opening.cells P.fold + digits * P.opening.mainDegrees +
            digits * P.opening.firstDegrees) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_union, Finset.mem_Ico, Gaps.support]
    show _ ↔ _ ∨ _
    simp only [OpeningShape.gaps] at room' ordered ⊢
    omega
  have apart : Disjoint (Finset.Ico (P.opening.chainStart P.fold + digits * P.opening.mainDegrees)
      (P.opening.secondStart P.fold + digits * P.opening.mainDegrees))
        (Finset.Ico (P.opening.secondStart P.fold + digits * P.opening.mainDegrees +
            digits * P.opening.firstDegrees)
          (P.opening.cells P.fold + digits * P.opening.mainDegrees +
            digits * P.opening.firstDegrees)) :=
    Finset.disjoint_left.mpr fun x first second => by
      simp only [Finset.mem_Ico] at first second
      omega
  rw [support, Finset.sum_union apart, Finset.sum_Ico_eq_sum_range, Finset.sum_Ico_eq_sum_range]
  unfold foldSupport
  rw [placed.2.2.1, placed.2.2.2.2, spans.1, spans.2]
  simp only [Ext.into_add, sumList_into, sum_list_range]
  have inside : P.opening.cells P.fold + digits * P.opening.mainDegrees +
      digits * P.opening.firstDegrees ≤ 2 ^ μ := room'
  have below : P.opening.chainStart P.fold ≤ P.opening.secondStart P.fold ∧
      P.opening.secondStart P.fold ≤ P.opening.cells P.fold := ordered
  congr 1
  · refine Finset.sum_congr rfl fun k member => ?_
    have := Finset.mem_range.mp member
    rw [Ext.into_mul, virtual _ (by omega), same _ (by omega), mul_comm]
  · refine Finset.sum_congr rfl fun k member => ?_
    have := Finset.mem_range.mp member
    rw [Ext.into_mul, virtual _ (by omega), same _ (by omega), mul_comm]

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] [Fintype E]

/-- The weight theorem of a packing level with one group: the relation weight that the replay
computes at `r_2` (`foldFused`) is `m_{τ_1}(r_2)` of `liftedReduction` for a statement whose
matrices are the views of the setup array and whose point has the weights of the opening point of
the group, the fold challenges `c_i` of the group in `R_{q,d_car}` and `ι(c_i)` in `R_{q,d_A}`, and
the basis `(1, u)`. -/
theorem foldWeight_one (P : PackGroup) (β : Module.Basis (Fin P.opening.planes) Zq E)
    (based : ∀ t, β t = (extBasis.getD t Ext.zero).into) (F : FoldInputs) (V : PackingValues)
    (st : Akita.Statement P.opening E) (keyed : st.key.Shared (flatOf F.setup))
    (opened : st.openKey.Shared (flatOf F.setup))
    (positioned : ∀ x, st.point.position x = eqAt (P.segments.2.2.1.map Ext.into) x)
    (packed : ∀ a, st.point.pack a = eqAt (P.segments.1.map Ext.into) a)
    (blocked : ∀ i, st.point.block i = eqAt (P.segments.2.2.2.map Ext.into) i)
    (tailed : ∀ j, st.point.tail j = eqAt (P.segments.2.1.map Ext.into) j)
    (challenge : Fin P.shape.blocks → Rq P.opening.carrierDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq _ (P.challenges.getD i (Poly.zero P.opening.carrierDegree)))
    (ambiented : ∀ i : Fin P.shape.blocks, Poly.toRq P.shape.innerDegree
      (P.ambient.getD i.val (Poly.zero P.shape.innerDegree)) =
        embed P.opening.stride P.opening.stride_tower (challenge i))
    (claimed : P.claim = Ext.one) {μ : ℕ} (fits : P.opening.cells P.fold ≤ 2 ^ μ)
    (room : P.opening.quotientRoom P.fold F.digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : P.opening.rowCount + 1 ≤ 2 ^ batch) (numbers : P.Fits μ)
    (tau1 : F.tau1 = V.tau1) (base : F.base = P.shape.certifiedBase)
    (path : F.path = ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩)
    (code : F.order = .code) (cells : F.cells = {}) (tau : V.tau1.length = batch)
    (r2 : List Ext) (length : r2.length = μ) :
    ((foldFused F [P] { V with r2 } none).weight.into : E) =
      mle (r2.map Ext.into) (((liftedReduction β st challenge μ fits F.digits room batch
        rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into)) := by
  have top : max P.opening.openDegree P.shape.innerDegree ≤ 2 ^ r2.length := length ▸ numbers.cube
  have short : Nat.log 2 (max P.opening.openDegree P.shape.innerDegree) ≤ r2.length :=
    log_le_cube top
  have topInner : P.shape.innerDegree ≤ max P.opening.openDegree P.shape.innerDegree :=
    le_max_right _ _
  have topOpen : P.opening.openDegree ≤ max P.opening.openDegree P.shape.innerDegree :=
    le_max_left _ _
  have cube : P.opening.quotientRoom P.fold F.digits ≤ 2 ^ r2.length := length ▸ room
  have bounded : ∀ x < P.opening.cells P.fold,
      (P.opening.gaps P.fold F.digits).place x < 2 ^ r2.length := fun x small =>
    (P.opening.place_lt_room P.fold F.digits small).trans_le cube
  have ordered : P.opening.chainStart P.fold + firstCells P.shape.chain ≤
      P.opening.secondStart P.fold ∧ P.opening.openFirstStart P.fold + firstCells P.opening.chain =
        P.opening.secondStart P.fold ∧ P.opening.secondStart P.fold ≤
          P.opening.openSecondStart P.fold ∧ P.opening.openSecondStart P.fold +
            secondCells P.opening.chain = P.opening.cells P.fold ∧
      P.opening.chainStart P.fold ≤ P.opening.openFirstStart P.fold ∧
      P.opening.secondStart P.fold + secondCells P.shape.chain ≤ P.opening.cells P.fold := by
    unfold OpeningShape.cells OpeningShape.openSecondStart OpeningShape.secondStart
      OpeningShape.openFirstStart
    omega
  have shift : ∀ x c, x + c < P.opening.chainStart P.fold →
      (P.opening.gaps P.fold F.digits).place (x + c) =
        (P.opening.gaps P.fold F.digits).place x + c :=
    fun x c high => Gaps.place_shift _ (lo := 0) (Or.inl le_rfl) x c (Nat.zero_le _) high
  have shiftFirst : ∀ x c, P.opening.chainStart P.fold ≤ x →
      x + c < P.opening.chainStart P.fold + firstCells P.shape.chain →
        (P.opening.gaps P.fold F.digits).place (x + c) =
          (P.opening.gaps P.fold F.digits).place x + c :=
    Gaps.place_shift _ (Or.inr (Or.inl ⟨le_rfl, ordered.1⟩))
  have shiftSecond : ∀ x c, P.opening.secondStart P.fold ≤ x →
      x + c < P.opening.secondStart P.fold + secondCells P.shape.chain →
        (P.opening.gaps P.fold F.digits).place (x + c) =
          (P.opening.gaps P.fold F.digits).place x + c :=
    Gaps.place_shift _ (Or.inr (Or.inr le_rfl))
  have shiftOpenFirst : ∀ x c, P.opening.openFirstStart P.fold ≤ x →
      x + c < P.opening.openFirstStart P.fold + firstCells P.opening.chain →
        (P.opening.gaps P.fold F.digits).place (x + c) =
          (P.opening.gaps P.fold F.digits).place x + c :=
    Gaps.place_shift _ (Or.inr (Or.inl ⟨ordered.2.2.2.2.1, ordered.2.1.le⟩))
  have shiftOpenSecond : ∀ x c, P.opening.openSecondStart P.fold ≤ x →
      x + c < P.opening.openSecondStart P.fold + secondCells P.opening.chain →
        (P.opening.gaps P.fold F.digits).place (x + c) =
          (P.opening.gaps P.fold F.digits).place x + c :=
    Gaps.place_shift _ (Or.inr (Or.inr ordered.2.2.1))
  have placed := Placement.of_one P F.digits
  have openFirstMoves : ∀ p < firstCells P.opening.chain,
      (P.opening.gaps P.fold F.digits).place (P.opening.openFirstStart P.fold + p) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).openFirst + p := fun p small => by
    rw [placed.2.1]
    unfold Gaps.place OpeningShape.gaps
    simp only
    have := ordered.2.1
    have := ordered.2.2.2.2.1
    unfold OpeningShape.openFirstStart at *
    split_ifs <;> omega
  have openSecondMoves : ∀ p < secondCells P.opening.chain,
      (P.opening.gaps P.fold F.digits).place (P.opening.openSecondStart P.fold + p) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).openSecond + p := fun p small => by
    rw [placed.2.2.2.1]
    unfold Gaps.place OpeningShape.gaps
    simp only
    have := ordered.2.2.1
    have := ordered.1
    unfold OpeningShape.openSecondStart at *
    split_ifs <;> omega
  have boundFirst : ∀ x, P.opening.chainStart P.fold ≤ x →
      x < P.opening.chainStart P.fold + firstCells P.shape.chain →
        (P.opening.gaps P.fold F.digits).place x < 2 ^ r2.length := fun x _ high =>
    bounded x (by have := ordered.1; have := ordered.2.2.2.2.2; omega)
  have boundSecond : ∀ x, P.opening.secondStart P.fold ≤ x →
      x < P.opening.secondStart P.fold + secondCells P.shape.chain →
        (P.opening.gaps P.fold F.digits).place x < 2 ^ r2.length := fun x _ high =>
    bounded x (by have := ordered.2.2.2.2.2; omega)
  have boundOpenFirst : ∀ x, P.opening.openFirstStart P.fold ≤ x →
      x < P.opening.openFirstStart P.fold + firstCells P.opening.chain →
        (P.opening.gaps P.fold F.digits).place x < 2 ^ r2.length := fun x _ high =>
    bounded x (by have := ordered.2.1; have := ordered.2.2.1; have := ordered.2.2.2.1; omega)
  have boundOpenSecond : ∀ x, P.opening.openSecondStart P.fold ≤ x →
      x < P.opening.openSecondStart P.fold + secondCells P.opening.chain →
        (P.opening.gaps P.fold F.digits).place x < 2 ^ r2.length := fun x _ high =>
    bounded x (by have := ordered.2.2.2.1; omega)
  obtain ⟨insideCarrier, insideA, insideB, insideD, insideF1, insideH1, insideF2, insideH2⟩ :=
    quotient_insides P F.digits r2 cube
  rw [← tau1]
  simp only [foldFused, packingEnd, foldTables, path, List.map_cons, List.map_nil, List.foldl_cons,
    List.foldl_nil, sumList_into]
  simp only [foldParts, path, cells, List.zipIdx_cons, List.zipIdx_nil, List.map_cons,
    List.map_nil, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_append,
    List.sum_append, List.sum_cons, List.sum_nil, groupParts_eq, openParts_eq, Option.isSome_none,
    Bool.false_and, Bool.false_eq_true, ite_false, List.length_singleton, List.range_one,
    List.getD_cons_zero, before, List.take_zero, beforeIn, Placement.move_one,
    List.singleton_append, filterMap_some_map, List.map_map, Function.comp_def, zipIdx_map_lift,
    sumList_into, Ext.into_mul, add_zero, mul_zero, PackGroup.openWidth, Tables.of_alpha,
    segCell_eta, lv_append]
  rw [chainLifts, firstCell_eq, secondCell_eq]
  rw [chainLifts_into (α := V.alpha) (r2 := r2) (chain := P.shape.chain) (key := st.key.chain)
      (shared := keyed.2.2) (d := P.shape.outerDegree) (n := P.shape.slices * P.shape.outerRank)
      (s1 := P.opening.chainStart P.fold) (s2 := P.opening.secondStart P.fold)
      (payload := st.payload) _ _ _ _ _ _ _ short (pow_log_top numbers.outer)
      (log_le_top numbers.outerTop) numbers.outerTop (pow_log_top numbers.first)
      (pow_log_top numbers.second) (log_le_top numbers.firstTop) (log_le_top numbers.secondTop)
      numbers.firstTop numbers.secondTop numbers.image numbers.chained shiftFirst shiftSecond
      boundFirst boundSecond,
    ← moved_segCell openFirstMoves, ← moved_segCell openSecondMoves]
  have openChain := chainLifts_into (E := E) (α := V.alpha) (r2 := r2)
    (bits := Nat.log 2 (max P.opening.openDegree P.shape.innerDegree))
    (top := max P.opening.openDegree P.shape.innerDegree) (setup := F.setup)
    (coef := fun r => eqIndex F.tau1 r) (chain := P.opening.chain) (key := st.openKey.chain)
    (shared := opened.2) (d := P.opening.openDegree) (n := 1 * P.opening.openRank)
    (s1 := P.opening.openFirstStart P.fold) (s2 := P.opening.openSecondStart P.fold)
    (payload := st.openPayload) (image := P.rowCount)
    (first := P.rowCount + P.opening.openRank + firstRank P.shape.chain)
    (second := P.rowCount + P.opening.openRank + firstRank P.shape.chain +
      firstRank P.opening.chain + secondRank P.shape.chain) short
    (pow_log_top numbers.opened) (log_le_top topOpen) topOpen (pow_log_top numbers.openFirst)
    (pow_log_top numbers.openSecond) (log_le_top numbers.openFirstTop)
    (log_le_top numbers.openSecondTop) numbers.openFirstTop numbers.openSecondTop numbers.openImage
    numbers.openChained shiftOpenFirst shiftOpenSecond boundOpenFirst boundOpenSecond
  simp only [List.map_map, Function.comp_def] at openChain
  rw [openChain]
  have firstPositive : NeZero (firstDegree P.shape.chain) :=
    ⟨fun zero => by obtain ⟨k, hk⟩ := numbers.first; rw [zero] at hk; exact absurd hk (by positivity)⟩
  have secondPositive : NeZero (secondDegree P.shape.chain) :=
    ⟨fun zero => by obtain ⟨k, hk⟩ := numbers.second; rw [zero] at hk; exact absurd hk (by positivity)⟩
  have openFirstPositive : NeZero (firstDegree P.opening.chain) :=
    ⟨fun zero => by obtain ⟨k, hk⟩ := numbers.openFirst; rw [zero] at hk; exact absurd hk (by positivity)⟩
  have openSecondPositive : NeZero (secondDegree P.opening.chain) :=
    ⟨fun zero => by obtain ⟨k, hk⟩ := numbers.openSecond; rw [zero] at hk; exact absurd hk (by positivity)⟩
  have hP := (lv_into (E := E) _ _).trans (partialLifts_value F (P.opening.gaps P.fold F.digits).place
    P 0 V.alpha r2 _ _ short (pow_log_top numbers.carrier) (pow_log_top numbers.opened)
    (log_le_top numbers.carrierTop) numbers.carrierTop shift bounded challenge challenged)
  have hR := (lv_into (E := E) _ _).trans (responseLifts_value F
    (P.opening.gaps P.fold F.digits).place P 0 V.alpha r2 _ _ short (pow_log_top numbers.inner)
    (log_le_top topInner) topInner shift bounded st.key keyed.1)
  have hI := (lv_into (E := E) _ _).trans (imageLifts_value F (P.opening.gaps P.fold F.digits).place
    P 0 V.alpha r2 _ _ short (pow_log_top numbers.outer) (log_le_top numbers.outerTop) topInner shift
    bounded challenge ambiented)
  have hO := (lv_into (E := E) _ _).trans (outerLifts_value F (P.opening.gaps P.fold F.digits).place
    P 0 V.alpha r2 _ _ short (pow_log_top numbers.outer) (log_le_top numbers.outerTop)
    numbers.outerTop shift bounded st.key keyed.2.1)
  have hC := contractionValue_into (E := E) F (P.opening.gaps P.fold F.digits).place P 0 V.alpha r2
    _ (max P.opening.openDegree P.shape.innerDegree) short (pow_log_top numbers.inner)
    (log_le_top topInner) shift bounded β st.point positioned packed
  have hOp := (lv_into (E := E) _ _).trans (openLifts_value F P V.alpha r2 _ _ P.rowCount short
    (pow_log_top numbers.opened) (log_le_top topOpen) topOpen shift bounded st.openKey opened.1)
  have hS := scalarPart_into (E := E) (move := (P.opening.gaps P.fold F.digits).place) P V.alpha r2
    _ (max P.opening.openDegree P.shape.innerDegree) short (pow_log_top numbers.carrier)
    (pow_log_top numbers.opened) (log_le_top numbers.carrierTop)
    (fun x small => Gaps.place_of_lt _ small) (by rw [length]; exact fits) β based st.point blocked
    tailed (by rw [claimed, Ext.into_one])
  have hQC := (lv_into (E := E) _ _).trans (carrierQuotient_value F code P V.alpha r2 _
    (max P.opening.openDegree P.shape.innerDegree) (eqIndex F.tau1 0) _ short
    (pow_log_top numbers.carrier) (log_le_top numbers.carrierTop) numbers.carrierTop insideCarrier)
  have hQA := rowQuotients_two (E := E) V.alpha r2 _ (max P.opening.openDegree P.shape.innerDegree)
    F.base F.digits P.shape.innerDegree P.shape.innerRank P.shape.outerDegree
    (P.shape.slices * P.shape.outerRank) F.tau1 (0 + 1) _ short (pow_log_top numbers.inner)
    (log_le_top topInner) topInner (pow_log_top numbers.outer) (log_le_top numbers.outerTop)
    numbers.outerTop insideA insideB
  have hQD := rowQuotients_one (E := E) V.alpha r2 _ (max P.opening.openDegree P.shape.innerDegree)
    F.base F.digits P.opening.openDegree F.tau1 P.rowCount _ P.opening.openRank short
    (pow_log_top numbers.opened) (log_le_top topOpen) topOpen insideD
  have hQ1 := rowQuotients_two (E := E) V.alpha r2 _ (max P.opening.openDegree P.shape.innerDegree)
    F.base F.digits (firstDegree P.shape.chain) (firstRank P.shape.chain)
    (firstDegree P.opening.chain) (firstRank P.opening.chain) F.tau1
    (P.rowCount + P.opening.openRank) _ short (pow_log_top numbers.first)
    (log_le_top numbers.firstTop) numbers.firstTop (pow_log_top numbers.openFirst)
    (log_le_top numbers.openFirstTop) numbers.openFirstTop insideF1 insideH1
  have hQ2 := rowQuotients_two (E := E) V.alpha r2 _ (max P.opening.openDegree P.shape.innerDegree)
    F.base F.digits (secondDegree P.shape.chain) (secondRank P.shape.chain)
    (secondDegree P.opening.chain) (secondRank P.opening.chain) F.tau1
    (P.rowCount + P.opening.openRank + firstRank P.shape.chain + firstRank P.opening.chain) _ short
    (pow_log_top numbers.second) (log_le_top numbers.secondTop) numbers.secondTop
    (pow_log_top numbers.openSecond) (log_le_top numbers.openSecondTop) numbers.openSecondTop
    insideF2 insideH2
  rw [hS]
  have hSpec := liftedReduction_weight β st challenge fits F.digits room rowsFit V.alpha.into
    (τ := F.tau1.map Ext.into) (r := r2.map Ext.into) (by rw [List.length_map, length])
    (by rw [List.length_map, tau1, tau])
  have hTail := flatTail_weight (O := P.opening) (fold := P.fold) st challenge V.alpha.into
    (F.tau1.map Ext.into) (r2.map Ext.into) P.shape.certifiedBase F.digits
    (F.digits * (P.opening.planes * P.opening.carrierDegree))
    (fun o => (o.map (P.opening.gaps P.fold F.digits).place).elim 0 (eqAt (r2.map Ext.into)))
    (P.opening.gaps P.fold F.digits).slot
  have fitsPlane : F.digits * (P.opening.planes * P.opening.carrierDegree) ≤
      (P.opening.gaps P.fold F.digits).mainCells :=
    Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have fitsA : F.digits * (P.opening.planes * P.opening.carrierDegree) +
      F.digits * (P.shape.innerRank * P.shape.innerDegree) ≤
        (P.opening.gaps P.fold F.digits).mainCells := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have fitsB : F.digits * (P.opening.planes * P.opening.carrierDegree) +
      F.digits * (P.shape.innerRank * P.shape.innerDegree +
        P.shape.slices * P.shape.outerRank * P.shape.outerDegree) ≤
          (P.opening.gaps P.fold F.digits).mainCells := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have fitsD : F.digits * (P.opening.planes * P.opening.carrierDegree) +
      F.digits * (P.shape.innerRank * P.shape.innerDegree +
        P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
          1 * P.opening.openRank * P.opening.openDegree) ≤
            (P.opening.gaps P.fold F.digits).mainCells := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have lowF1 : (P.opening.gaps P.fold F.digits).mainCells ≤
      F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree) := by
    show F.digits * _ ≤ _
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have fitsF1 : F.digits * (P.opening.planes * P.opening.carrierDegree) +
      F.digits * (P.shape.innerRank * P.shape.innerDegree +
        P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
          1 * P.opening.openRank * P.opening.openDegree +
            firstRank P.shape.chain * firstDegree P.shape.chain) ≤
              (P.opening.gaps P.fold F.digits).mainCells +
                (P.opening.gaps P.fold F.digits).middleCells := by
    show _ ≤ F.digits * _ + F.digits * _
    rw [← Nat.mul_add, ← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by
      unfold OpeningShape.mainDegrees OpeningShape.firstDegrees; omega)
  have lowH1 : (P.opening.gaps P.fold F.digits).mainCells ≤
      F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree +
              firstRank P.shape.chain * firstDegree P.shape.chain) := by
    show F.digits * _ ≤ _
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by unfold OpeningShape.mainDegrees; omega)
  have fitsH1 : F.digits * (P.opening.planes * P.opening.carrierDegree) +
      F.digits * (P.shape.innerRank * P.shape.innerDegree +
        P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
          1 * P.opening.openRank * P.opening.openDegree +
            firstRank P.shape.chain * firstDegree P.shape.chain +
              firstRank P.opening.chain * firstDegree P.opening.chain) ≤
                (P.opening.gaps P.fold F.digits).mainCells +
                  (P.opening.gaps P.fold F.digits).middleCells := by
    show _ ≤ F.digits * _ + F.digits * _
    rw [← Nat.mul_add, ← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by
      unfold OpeningShape.mainDegrees OpeningShape.firstDegrees; omega)
  have lowF2 : (P.opening.gaps P.fold F.digits).mainCells +
      (P.opening.gaps P.fold F.digits).middleCells ≤
        F.digits * (P.opening.planes * P.opening.carrierDegree) +
          F.digits * (P.shape.innerRank * P.shape.innerDegree +
            P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
              1 * P.opening.openRank * P.opening.openDegree +
                firstRank P.shape.chain * firstDegree P.shape.chain +
                  firstRank P.opening.chain * firstDegree P.opening.chain) := by
    show F.digits * _ + F.digits * _ ≤ _
    rw [← Nat.mul_add, ← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by
      unfold OpeningShape.mainDegrees OpeningShape.firstDegrees; omega)
  have lowH2 : (P.opening.gaps P.fold F.digits).mainCells +
      (P.opening.gaps P.fold F.digits).middleCells ≤
        F.digits * (P.opening.planes * P.opening.carrierDegree) +
          F.digits * (P.shape.innerRank * P.shape.innerDegree +
            P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
              1 * P.opening.openRank * P.opening.openDegree +
                firstRank P.shape.chain * firstDegree P.shape.chain +
                  firstRank P.opening.chain * firstDegree P.opening.chain +
                    secondRank P.shape.chain * secondDegree P.shape.chain) := by
    show F.digits * _ + F.digits * _ ≤ _
    rw [← Nat.mul_add, ← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (by
      unfold OpeningShape.mainDegrees OpeningShape.firstDegrees; omega)
  have startsA : (P.opening.gaps P.fold F.digits).main +
      F.digits * (P.opening.planes * P.opening.carrierDegree) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientMain + F.digits * (P.opening.planes * P.opening.carrierDegree) := by
    rw [placed.1]
    rfl
  have startsB : (P.opening.gaps P.fold F.digits).main +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientMain + F.digits * (P.opening.planes * P.opening.carrierDegree) +
            P.shape.innerRank * (F.digits * P.shape.innerDegree) := by
    rw [placed.1]
    show P.opening.chainStart P.fold + _ = _
    ring
  have startsD : (P.opening.gaps P.fold F.digits).main +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientMain + F.digits * P.mainDegrees := by
    rw [placed.1]
    show P.opening.chainStart P.fold + _ = _
    unfold PackGroup.mainDegrees
    ring
  have startsF1 : (P.opening.gaps P.fold F.digits).middle +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientFirst := by
    rw [placed.2.2.1]
    show P.opening.secondStart P.fold + _ = _
    unfold OpeningShape.mainDegrees
    ring
  have startsH1 : (P.opening.gaps P.fold F.digits).middle +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree +
              firstRank P.shape.chain * firstDegree P.shape.chain)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientFirst + firstRank P.shape.chain * (F.digits * firstDegree P.shape.chain) := by
    rw [placed.2.2.1]
    show P.opening.secondStart P.fold + _ = _
    unfold OpeningShape.mainDegrees
    ring
  have startsF2 : (P.opening.gaps P.fold F.digits).last +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree +
              firstRank P.shape.chain * firstDegree P.shape.chain +
                firstRank P.opening.chain * firstDegree P.opening.chain)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientSecond := by
    rw [placed.2.2.2.2]
    show P.opening.cells P.fold + _ = _
    unfold OpeningShape.mainDegrees OpeningShape.firstDegrees
    ring
  have startsH2 : (P.opening.gaps P.fold F.digits).last +
      (F.digits * (P.opening.planes * P.opening.carrierDegree) +
        F.digits * (P.shape.innerRank * P.shape.innerDegree +
          P.shape.slices * P.shape.outerRank * P.shape.outerDegree +
            1 * P.opening.openRank * P.opening.openDegree +
              firstRank P.shape.chain * firstDegree P.shape.chain +
                firstRank P.opening.chain * firstDegree P.opening.chain +
                  secondRank P.shape.chain * secondDegree P.shape.chain)) =
        (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩
          F.digits).quotientSecond +
            secondRank P.shape.chain * (F.digits * secondDegree P.shape.chain) := by
    rw [placed.2.2.2.2]
    show P.opening.cells P.fold + _ = _
    unfold OpeningShape.mainDegrees OpeningShape.firstDegrees
    ring
  simp only [quotientScan_row_plane _ _ _ _ _ _ _ fitsPlane placed.1.symm,
    quotientScan_row_main₀ _ _ _ _ _ _ _ fitsA startsA,
    quotientScan_row_main _ _ _ _ _ _ _ fitsB startsB,
    quotientScan_row_main _ _ _ _ _ _ _ fitsD startsD,
    quotientScan_row_middle _ _ _ _ _ _ _ lowF1 fitsF1 startsF1,
    quotientScan_row_middle _ _ _ _ _ _ _ lowH1 fitsH1 startsH1,
    quotientScan_row_last _ _ _ _ _ _ _ lowF2 startsF2,
    quotientScan_row_last _ _ _ _ _ _ _ lowH2 startsH2] at hSpec hTail
  simp only [evaluationRow, innerRow, Form.scan_row, Form.scan_sub, imageRow_scan, firstCell_eq,
    secondCell_eq, openFirstCell_eq, openSecondCell_eq, mul_sub, Finset.sum_sub_distrib]
    at hSpec hTail
  simp only [eqIndex_into, ← based, base, zero_add, one_mul, Nat.add_assoc,
    PackGroup.rowCount, OpeningShape.tailCount]
    at hP hR hI hO hC hOp hQC hQA hQD hQ1 hQ2 hSpec hTail ⊢
  linear_combination hP + hR + hI + hO + hC + hOp + hQC + hQA + hQD + hQ1 + hQ2 - hSpec - hTail

/-- The end check `fused.end` of a packing level with one group is the final check of the fused
sum-check of the specification (`fusedCheck`) for `liftedReduction`, the support `Gaps.support` of
the witness and the virtual weight `γ·ẽq(r_virt, ·) + ζ_bin·1_{I_bin}·ẽq(r_virt, ·)`. -/
theorem packingFused_holds_iff (P : PackGroup) (β : Module.Basis (Fin P.opening.planes) Zq E)
    (based : ∀ t, β t = (extBasis.getD t Ext.zero).into) (F : FoldInputs) (V : PackingValues)
    (st : Akita.Statement P.opening E) (keyed : st.key.Shared (flatOf F.setup))
    (opened : st.openKey.Shared (flatOf F.setup))
    (positioned : ∀ x, st.point.position x = eqAt (P.segments.2.2.1.map Ext.into) x)
    (packed : ∀ a, st.point.pack a = eqAt (P.segments.1.map Ext.into) a)
    (blocked : ∀ i, st.point.block i = eqAt (P.segments.2.2.2.map Ext.into) i)
    (tailed : ∀ j, st.point.tail j = eqAt (P.segments.2.1.map Ext.into) j)
    (challenge : Fin P.shape.blocks → Rq P.opening.carrierDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq _ (P.challenges.getD i (Poly.zero P.opening.carrierDegree)))
    (ambiented : ∀ i : Fin P.shape.blocks, Poly.toRq P.shape.innerDegree
      (P.ambient.getD i.val (Poly.zero P.shape.innerDegree)) =
        embed P.opening.stride P.opening.stride_tower (challenge i))
    (claimed : P.claim = Ext.one) {μ : ℕ} (fits : P.opening.cells P.fold ≤ 2 ^ μ)
    (room : P.opening.quotientRoom P.fold F.digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : P.opening.rowCount + 1 ≤ 2 ^ batch) (numbers : P.Fits μ)
    (tau1 : F.tau1 = V.tau1) (base : F.base = P.shape.certifiedBase)
    (path : F.path = ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩)
    (code : F.order = .code) (cells : F.cells = {}) (tau : V.tau1.length = batch)
    (length : V.r2.length = μ) (virtualLength : V.virtual.length = μ)
    (compressed : V.compressed = P.opening.compressed P.fold) :
    (foldFused F [P] V none).holds = true ↔
      (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into)
        (((liftedReduction β st challenge μ fits F.digits room batch rowsFit).rows
          V.alpha.into).batchWeight (V.tau1.map Ext.into)) +
        mle (V.r2.map Ext.into) (virtWeight (P.opening.gaps P.fold F.digits).support
          (V.virtual.map Ext.into) V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1)) := by
  have weighted := foldWeight_one P β based F V st keyed opened positioned packed blocked tailed
    challenge challenged ambiented claimed fits room rowsFit numbers tau1 base path code cells tau
    V.r2 length
  have normed : ((foldFused F [P] V none).normWeight.into : E) =
      mle (V.r2.map Ext.into) (fun _ => 0) := by
    rw [mle_const]
    exact Ext.into_zero
  have virt : ((foldFused F [P] V none).virt.into : E) =
      mle (V.r2.map Ext.into) (eqAt (V.virtual.map Ext.into)) := by
    show ((eqPoints V.virtual V.r2).into : E) = _
    rw [eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map, length,
      virtualLength])]
  have supported : ((foldFused F [P] V none).support.into : E) =
      mle (V.r2.map Ext.into) fun x => if (P.opening.gaps P.fold F.digits).support x then
        eqAt (V.virtual.map Ext.into) x else 0 := by
    show ((if V.compressed then foldSupport (Placement.of [P] F.path F.digits)
      (CellEq.half V.virtual).get (CellEq.half V.r2).get else Ext.zero).into : E) = _
    by_cases binary : V.compressed
    · rw [ite_eq_left binary, path]
      exact foldSupport_one P F.digits _ _ room (by rw [List.length_map, length])
        (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
        (fun x small => CellEq.get_half V.r2 (by rw [length]; exact small))
    · have empty : P.opening.compressed P.fold = false := by
        rw [← compressed]
        simpa using binary
      rw [ite_eq_right binary, Ext.into_zero, show (fun x =>
        if (P.opening.gaps P.fold F.digits).support x then eqAt (V.virtual.map Ext.into) x
          else (0 : E)) = fun _ => 0 from
        funext fun x => ite_eq_right (P.opening.gaps_support_empty P.fold F.digits empty x),
        mle_const]
  rw [FusedEnd.holds_iff_fused _ (P.opening.gaps P.fold F.digits).support _ _
    (V.virtual.map Ext.into) (V.r2.map Ext.into) weighted normed virt supported]
  show (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into) (fun x => _ +
    (Ext.zero.into : E) * 0) + mle (V.r2.map Ext.into) (virtWeight _ _ V.batch.into
      ((if V.compressed then V.binary else Ext.zero).into)) * (V.value.into * (V.value.into + 1)) ↔ _
  by_cases binary : V.compressed <;>
    simp only [binary, Bool.false_eq_true, ite_true, ite_false, Ext.into_zero, mul_zero, add_zero]

end Level

end NightstreamFPrime.Spec.Compression.Akita.Replay
