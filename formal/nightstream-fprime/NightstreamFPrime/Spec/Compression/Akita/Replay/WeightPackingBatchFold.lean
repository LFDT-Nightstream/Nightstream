import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchLevel

/-! The weight theorem of a fold with opening groups (ePrint 2026/1983, (111) to (113) and (123);
`Batch.liftedReduction` of `Extract/LiftSource.lean`; Akita revision `d50f700`,
`RelationMatrixEvaluator::eval_flat_at_point` with the groups in the order `root_group_order`).

- `flat_parts_into` and `getD_range_map`: the named parts of `foldParts` as sums over the groups.
- `le_foldl_max` and `foldl_max_le`: the largest degree `top` of the tables of a fold.
- `quotients_split`: the quotient digits of the rows of `F_1` (`F_2`) of the groups and of `H_1`
  (`H_2`) as a sum over the groups and the rows of the shared path.
- `batchScalar_weight`: the scalar row (100) with the claim coefficient `ρ_g` of each group.
- `batchOpenLifts_value`: the columns of the groups in `D` are the rows (106) of `D` on `ê`;
  `batchOpen_weight`: the rows (106) of `D`, the compression rows of `H_1` and `H_2` and their
  quotient digits are the rows of the shared path in `batchReduction_weight`.
- `groupFamilies_batch` and `groupsWeight_batch`: the parts of each group are its families in
  `batchReduction_weight` (with `groupWeight_batch` of `Replay/WeightPackingBatchLevel.lean`).
- `foldWeight_batch`: the weight of `foldFused` at `r_2` for the groups of `batchGroups` is
  `m_{τ_1}(r_2)` of `Batch.liftedReduction`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Lists

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

theorem flat_parts_into (T : Tables) {γ : Type} (l : List γ) (parts : γ → List (List Lift))
    (con : γ → Ext) (name : γ × ℕ → List Lift × ℕ → String) (cname : γ × ℕ → String) :
    (List.map (fun x => (x.2.into : E)) (List.flatMap (fun x =>
      List.map (fun y => (name x y, liftsValue T y.1)) (parts x.1).zipIdx ++ [(cname x, con x.1)])
        l.zipIdx)).sum =
      (l.map fun x => ((parts x).map fun p => ((liftsValue T p).into : E)).sum +
        ((con x).into : E)).sum := by
  rw [sum_flatMap]
  simp only [List.map_append, List.map_map, Function.comp_def, List.sum_append, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  have inner : ∀ x : γ × ℕ, (List.map (fun y => ((liftsValue T y.1).into : E))
      (parts x.1).zipIdx).sum = ((parts x.1).map fun p => ((liftsValue T p).into : E)).sum :=
    fun x => congrArg List.sum (zipIdx_map_fst (fun p => ((liftsValue T p).into : E)) (parts x.1) 0)
  simp only [inner]
  exact congrArg List.sum (zipIdx_map_fst (fun x => ((parts x).map fun p =>
    ((liftsValue T p).into : E)).sum + ((con x).into : E)) l 0)

theorem getD_range_map {α : Type} (h : ℕ → α) (d : α) {n k : ℕ} (small : k < n) :
    ((List.range n).map h).getD k d = h k := by
  simp [small]

end Lists

theorem le_foldl_max (l : List ℕ) : ∀ init, init ≤ l.foldl max init ∧ ∀ x ∈ l, x ≤ l.foldl max init := by
  induction l with
  | nil => exact fun init => ⟨le_rfl, fun x member => by simp at member⟩
  | cons y l ih =>
      intro init
      obtain ⟨first, rest⟩ := ih (max init y)
      refine ⟨show init ≤ l.foldl max (max init y) from (le_max_left init y).trans first,
        fun x member => show x ≤ l.foldl max (max init y) from ?_⟩
      rcases List.mem_cons.mp member with same | inside
      · exact same ▸ (le_max_right init y).trans first
      · exact rest x inside

theorem foldl_max_le (l : List ℕ) {b : ℕ} : ∀ init, init ≤ b → (∀ x ∈ l, x ≤ b) → l.foldl max init ≤ b := by
  induction l with
  | nil => exact fun _ le _ => le
  | cons y l ih =>
      intro init le all
      exact ih (max init y) (max_le le (all y List.mem_cons_self)) fun x member =>
        all x (List.mem_cons_of_mem _ member)

section Split

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The quotient terms of the rows of `rowStarts` for the rows of the groups and one row family of
the shared path: a sum over the groups and the rows of the shared path. -/
theorem quotients_split (T : Tables) (b digits : ℕ) (α : Ext) (τ : List Ext) (c s : ℕ) {n : ℕ}
    (deg rank : Fin n → ℕ) (dl nl : ℕ) :
    ((liftsValue T (((rowStarts digits s ((List.finRange n).map (fun k => (deg k, rank k)) ++
      [(dl, nl)])).zipIdx).map fun x =>
        quotientLift b digits x.1.1 α (eqIndex τ (c + x.2)) x.1.2)).into : E) =
      ∑ k, ∑ x : Fin (rank k), ((quotientLift b digits (deg k) α
        (eqIndex τ (c + offset rank k + x))
          (s + digits * offset (fun j => rank j * deg j) k + x * (digits * deg k))).value T).into +
      ∑ x : Fin nl, ((quotientLift b digits dl α (eqIndex τ (c + ∑ j, rank j + x))
        (s + digits * ∑ j, rank j * deg j + x * (digits * dl))).value T).into := by
  rw [lv_into, List.map_map, rowStarts_groups]
  simp only [Function.comp_apply, zero_add, Nat.add_assoc]

end Split

/-- The flat witness of one group is not longer than the flat witness of the fold. -/
theorem cells_le_batch {n : ℕ} (B : Batch (Fin n)) (g : Fin n) :
    (B.opening g).cells (B.fold g) ≤ B.cells := by
  obtain ⟨first, second, third, last⟩ := B.starts g
  obtain ⟨main, a, b, c, d, e, f, ends⟩ := B.bounds g
  omega

section Scalar

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The scalar row (100) of a fold with opening groups at `r_2`: the claim coefficient `ρ_g` and the
scalar weight of each group at its cells `Gaps.place ∘ Batch.relocate`. -/
theorem batchScalar_weight (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (based : ∀ g t, β g t = (extBasis.getD t Ext.zero).into)
    (blocked : ∀ k i, (st.point (groupOrder n k)).block i =
      eqAt ((batchGroup B cs amb pts w k).segments.2.2.2.map Ext.into) i)
    (tailed : ∀ k j, (st.point (groupOrder n k)).tail j =
      eqAt ((batchGroup B cs amb pts w k).segments.2.1.map Ext.into) j)
    (weighted : ∀ k, st.weight (groupOrder n k) = (w k).into) (digits : ℕ) (τ : List Ext)
    (α : Ext) (r2 : List Ext) (top : ℕ) (short : Nat.log 2 top ≤ r2.length)
    (numbers : ∀ k, (batchGroup B cs amb pts w k).Fits r2.length)
    (within : ∀ k, max B.openDegree (B.shape (groupOrder n k)).innerDegree ≤ top) {μ : ℕ}
    (fits : B.cells ≤ 2 ^ μ) (room : B.quotientRoom digits ≤ 2 ^ μ) (length : r2.length = μ)
    (row : ℕ) :
    ((eqIndex τ row).into : E) * ∑ k, ((scalarPart (fun y => (B.gaps digits).place
      (B.relocate (groupOrder n k) y)) (batchGroup B cs amb pts w k)
        (Tables.of α r2 (Nat.log 2 top) top)).into : E) =
      eqAt (τ.map Ext.into) row * mle (r2.map Ext.into) (B.scalarWeight β st μ) := by
  subst length
  rw [mle_batchScalar B β st fits (List.length_map _), eqIndex_into]
  conv_rhs => rw [← Equiv.sum_comp (groupOrder n)]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [scalarPart_moved (E := E) (move := fun y => (B.gaps digits).place
      (B.relocate (groupOrder n k) y)) (batchGroup B cs amb pts w k) α r2 _ top short
    (pow_log_top (numbers k).carrier) (pow_log_top (numbers k).opened)
    (log_le_top ((numbers k).carrierTop.trans (within k)))
    (relocated_shift B digits (groupOrder n k)) (relocated_bounded B digits (groupOrder n k) room)
    ((cells_le_batch B (groupOrder n k)).trans fits) (β (groupOrder n k)) (based (groupOrder n k))
    (st.point (groupOrder n k)) (blocked k) (tailed k), weighted k]
  congr 1
  exact Finset.sum_congr rfl fun y _ => scalar_place B β st digits (groupOrder n k) _ y

end Scalar

section Open

variable {n : ℕ} (B : Batch (Fin n))

theorem batchFirstCell_eq : B.openFirstCell = segCell B.openFirstStart (firstCells B.chain) := rfl

theorem batchSecondCell_eq :
    B.openSecondCell = segCell B.openSecondStart (secondCells B.chain) := rfl

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The columns of the groups in the rows (106) of `D` are the rows of `D` on `ê` of the fold. -/
theorem batchOpenLifts_value (st : B.Statement E) (F : FoldInputs)
    (opened : st.openKey.matrix =
      Akita.Setup.shared (flatOf F.setup) B.openDegree B.openRank _ B.openSetupCode) (α : Ext)
    (r2 : List Ext) (top : ℕ) (short : Nat.log 2 top ≤ r2.length)
    (openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree) (topOpen : B.openDegree ≤ top)
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length) :
    ∑ k, ((openLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
        (batchGroup B cs amb pts w k) (∑ g, B.groupCount g)
        (codeOffset (fun g => (B.shape g).blocks * ((B.opening g).subcolumns *
          (B.shape g).openingDigits)) (groupOrder n k))
        (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))).map
          fun x => ((x.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum =
      ∑ x : Fin (1 * B.openRank), eqAt (F.tau1.map Ext.into) (∑ g, B.groupCount g + x) *
        (B.openForm st.openKey x).scan (α.into : E)
          (fun o => (o.map (B.gaps F.digits).place).elim 0 (eqAt (r2.map Ext.into))) := by
  rw [openRows_batch B F.setup st.openKey opened (α.into : E) (F.tau1.map Ext.into)
    (r2.map Ext.into) (∑ g, B.groupCount g) F.digits]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [openLifts_columns F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
    (batchGroup B cs amb pts w k) α r2 (Nat.log 2 top) top _ _ _ short openPow
    (log_le_top topOpen) topOpen (relocated_shift B F.digits (groupOrder n k))
    (relocated_bounded B F.digits (groupOrder n k) room)]
  simp only [eqIndex_into]
  rfl

/-- The rows (106) of `D` with the columns of each group, the compression rows of the shared path
and their quotient digits are the families of `D`, `H_1` and `H_2` in `batchReduction_weight`. -/
theorem batchOpen_weight (st : B.Statement E) (F : FoldInputs)
    (opened : st.openKey.matrix =
      Akita.Setup.shared (flatOf F.setup) B.openDegree B.openRank _ B.openSetupCode)
    (openChain : PathShared (flatOf F.setup) B.chain st.openKey.chain) (α : Ext) (r2 : List Ext)
    (top : ℕ) (short : Nat.log 2 top ≤ r2.length) (k₀ : Fin n)
    (numbers : (batchGroup B cs amb pts w k₀).Fits r2.length)
    (within : max B.openDegree (B.shape (groupOrder n k₀)).innerDegree ≤ top)
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length) :
    ∑ k, ((openLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
        (batchGroup B cs amb pts w k) (∑ g, B.groupCount g)
        (codeOffset (fun g => (B.shape g).blocks * ((B.opening g).subcolumns *
          (B.shape g).openingDigits)) (groupOrder n k))
        (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))).map
          fun x => ((x.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ((chainTerms F.setup (fun r => eqIndex F.tau1 r) B.chain B.openDegree (1 * B.openRank)
          (segCell (B.openFirstStart + F.digits * B.mainDegrees) (firstCells B.chain))
          (segCell (B.openSecondStart + F.digits * B.mainDegrees + F.digits * B.firstDegrees)
            (secondCells B.chain))
          (∑ g, B.groupCount g) (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain)
          (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain + firstRank B.chain +
            ∑ g, secondRank (B.shape g).chain)).map fun x =>
              ((liftsValue (Tables.of α r2 (Nat.log 2 top) top) (x.map Lift.ofTerm)).into : E)).sum +
      ((liftsValue (Tables.of α r2 (Nat.log 2 top) top)
        (((rowStarts F.digits (B.chainStart + F.digits * ∑ g, B.groupDegrees g)
          [(B.openDegree, B.openRank)]).zipIdx).map fun x =>
            quotientLift F.base F.digits x.1.1 α (eqIndex F.tau1 (∑ g, B.groupCount g + x.2))
              x.1.2)).into : E) +
      ∑ x : Fin (firstRank B.chain), ((quotientLift F.base F.digits (firstDegree B.chain) α
        (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank +
          ∑ j, firstRank (B.shape (groupOrder n j)).chain + x))
        (B.secondStart + F.digits * B.mainDegrees + F.digits * ∑ j,
          firstRank (B.shape (groupOrder n j)).chain * firstDegree (B.shape (groupOrder n j)).chain +
            x * (F.digits * firstDegree B.chain))).value
              (Tables.of α r2 (Nat.log 2 top) top)).into +
      ∑ x : Fin (secondRank B.chain), ((quotientLift F.base F.digits (secondDegree B.chain) α
        (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain +
          firstRank B.chain + ∑ j, secondRank (B.shape (groupOrder n j)).chain + x))
        (B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * ∑ j,
          secondRank (B.shape (groupOrder n j)).chain * secondDegree (B.shape (groupOrder n j)).chain +
            x * (F.digits * secondDegree B.chain))).value
              (Tables.of α r2 (Nat.log 2 top) top)).into =
    ∑ x : Fin (1 * B.openRank), eqAt (F.tau1.map Ext.into) (∑ g, B.groupCount g + x) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (o.map (B.gaps F.digits).place).elim 0 (eqAt (r2.map Ext.into)))
          (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload x)
          (fun a j => B.chainStart + F.digits * ∑ g, B.groupDegrees g +
            x * (F.digits * B.openDegree) + a * B.openDegree + j) +
      (∑ x : Fin (firstRank B.chain), eqAt (F.tau1.map Ext.into) (∑ g, B.groupCount g +
          (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + x))) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (o.map (B.gaps F.digits).place).elim 0 (eqAt (r2.map Ext.into)))
          (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell x)
          (fun a j => B.secondStart + F.digits * B.mainDegrees + F.digits * ∑ h,
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain +
              x * (F.digits * firstDegree B.chain) + a * firstDegree B.chain + j) +
      ∑ x : Fin (secondRank B.chain), eqAt (F.tau1.map Ext.into) (∑ g, B.groupCount g +
          (1 * B.openRank + (∑ h, firstRank (B.shape h).chain + (firstRank B.chain +
            (∑ h, secondRank (B.shape h).chain + x))))) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (o.map (B.gaps F.digits).place).elim 0 (eqAt (r2.map Ext.into)))
          (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload x)
          (fun a j => B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * ∑ h,
            secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
              x * (F.digits * secondDegree B.chain) + a * secondDegree B.chain + j)) := by
  have topOpen : B.openDegree ≤ top := (le_max_left _ _).trans within
  have topOf : ∀ {d : ℕ}, d ≤ max B.openDegree (B.shape (groupOrder n k₀)).innerDegree → d ≤ top :=
    fun le => le.trans within
  have openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree := pow_log_top numbers.opened
  have firstPositive : NeZero (firstDegree B.chain) := by
    obtain ⟨e, he⟩ := numbers.openFirst
    have he' : 2 ^ e = firstDegree B.chain := he
    exact ⟨by rw [← he']; positivity⟩
  have secondPositive : NeZero (secondDegree B.chain) := by
    obtain ⟨e, he⟩ := numbers.openSecond
    have he' : 2 ^ e = secondDegree B.chain := he
    exact ⟨by rw [← he']; positivity⟩
  have degrees := B.gaps_counts F.digits
  have ordered : B.chainStart ≤ B.openFirstStart ∧ B.openFirstStart + firstCells B.chain = B.secondStart ∧
      B.secondStart ≤ B.openSecondStart ∧ B.openSecondStart + secondCells B.chain = B.cells := by
    unfold Batch.cells Batch.openSecondStart Batch.secondStart Batch.openFirstStart
    omega
  have inside : ∀ x, x < B.cells → (B.gaps F.digits).place x < 2 ^ r2.length := fun x small =>
    (B.place_lt_room F.digits small).trans_le room
  have openFirstMoves : ∀ p < firstCells B.chain, (B.gaps F.digits).place (B.openFirstStart + p) =
      B.openFirstStart + F.digits * B.mainDegrees + p := fun p small => by
    unfold Gaps.place Batch.gaps
    simp only
    split_ifs <;> omega
  have openSecondMoves : ∀ p < secondCells B.chain, (B.gaps F.digits).place (B.openSecondStart + p) =
      B.openSecondStart + F.digits * B.mainDegrees + F.digits * B.firstDegrees + p := fun p small => by
    unfold Gaps.place Batch.gaps
    simp only
    split_ifs <;> omega
  have hChain := chainLifts_into (E := E) (α := α) (r2 := r2) (bits := Nat.log 2 top) (top := top)
    (move := (B.gaps F.digits).place) F.setup (eqIndex F.tau1) B.chain st.openKey.chain openChain
    (d := B.openDegree) (1 * B.openRank) B.openFirstStart B.openSecondStart st.openPayload
    (∑ g, B.groupCount g) (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain)
    (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain + firstRank B.chain +
      ∑ g, secondRank (B.shape g).chain) short openPow (log_le_top topOpen) topOpen
    (pow_log_top numbers.openFirst) (pow_log_top numbers.openSecond)
    (log_le_top (topOf numbers.openFirstTop)) (log_le_top (topOf numbers.openSecondTop))
    (topOf numbers.openFirstTop) (topOf numbers.openSecondTop) numbers.openImage numbers.openChained
    (Gaps.place_shift _ (Or.inr (Or.inl ⟨ordered.1, ordered.2.1.le⟩)))
    (Gaps.place_shift _ (Or.inr (Or.inr ordered.2.2.1)))
    (fun x _ high => inside x (by omega)) (fun x _ high => inside x (by omega))
  simp only [List.map_map, Function.comp_def] at hChain
  rw [moved_segCell openFirstMoves, moved_segCell openSecondMoves] at hChain
  have sumFirst : ∑ j, firstRank (B.shape (groupOrder n j)).chain = ∑ h, firstRank (B.shape h).chain :=
    Equiv.sum_comp (groupOrder n) fun h => firstRank (B.shape h).chain
  have sumSecond : ∑ j, secondRank (B.shape (groupOrder n j)).chain =
      ∑ h, secondRank (B.shape h).chain :=
    Equiv.sum_comp (groupOrder n) fun h => secondRank (B.shape h).chain
  have sumFirstCells : ∑ j, firstRank (B.shape (groupOrder n j)).chain *
      firstDegree (B.shape (groupOrder n j)).chain =
        ∑ h, firstRank (B.shape h).chain * firstDegree (B.shape h).chain :=
    Equiv.sum_comp (groupOrder n) fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain
  have sumSecondCells : ∑ j, secondRank (B.shape (groupOrder n j)).chain *
      secondDegree (B.shape (groupOrder n j)).chain =
        ∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain :=
    Equiv.sum_comp (groupOrder n) fun h =>
      secondRank (B.shape h).chain * secondDegree (B.shape h).chain
  have degreeSum : B.degreeSum = B.mainDegrees + B.firstDegrees +
      (∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
        secondRank B.chain * secondDegree B.chain) := by
    unfold Batch.degreeSum Batch.mainDegrees Batch.firstDegrees
    ring
  have cellsLe := B.chainStart_le_cells
  unfold Batch.quotientRoom at room
  have hQD := rowQuotients_one (E := E) α r2 _ top F.base F.digits B.openDegree F.tau1
    (∑ g, B.groupCount g) (B.chainStart + F.digits * ∑ g, B.groupDegrees g) B.openRank short
    openPow (log_le_top topOpen) topOpen (by
      calc B.chainStart + F.digits * ∑ g, B.groupDegrees g + 1 * B.openRank * (F.digits * B.openDegree)
          = B.chainStart + F.digits * B.mainDegrees := by unfold Batch.mainDegrees; ring
        _ ≤ 2 ^ r2.length := by rw [degreeSum, Nat.mul_add, Nat.mul_add] at room; omega)
  have hQ1 := quotientRows_value (E := E) α r2 _ top F.base F.digits (firstDegree B.chain) F.tau1
    (B.secondStart + F.digits * B.mainDegrees + F.digits * ∑ j,
      firstRank (B.shape (groupOrder n j)).chain * firstDegree (B.shape (groupOrder n j)).chain)
    (firstRank B.chain) (fun x => ∑ g, B.groupCount g + B.openRank +
          ∑ j, firstRank (B.shape (groupOrder n j)).chain + x) short
    (pow_log_top numbers.openFirst) (log_le_top (topOf numbers.openFirstTop))
    (topOf numbers.openFirstTop) (by
      calc B.secondStart + F.digits * B.mainDegrees + F.digits * ∑ j,
            firstRank (B.shape (groupOrder n j)).chain * firstDegree (B.shape (groupOrder n j)).chain +
              firstRank B.chain * (F.digits * firstDegree B.chain)
          = B.secondStart + F.digits * B.mainDegrees + F.digits * B.firstDegrees := by
            rw [sumFirstCells]; unfold Batch.firstDegrees; ring
        _ ≤ 2 ^ r2.length := by
            rw [degreeSum, Nat.mul_add, Nat.mul_add] at room; omega)
  have hQ2 := quotientRows_value (E := E) α r2 _ top F.base F.digits (secondDegree B.chain) F.tau1
    (B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * ∑ j,
      secondRank (B.shape (groupOrder n j)).chain * secondDegree (B.shape (groupOrder n j)).chain)
    (secondRank B.chain) (fun x => ∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain +
          firstRank B.chain + ∑ j, secondRank (B.shape (groupOrder n j)).chain + x) short
    (pow_log_top numbers.openSecond) (log_le_top (topOf numbers.openSecondTop))
    (topOf numbers.openSecondTop) (by
      calc B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * ∑ j,
            secondRank (B.shape (groupOrder n j)).chain * secondDegree (B.shape (groupOrder n j)).chain +
              secondRank B.chain * (F.digits * secondDegree B.chain)
          = B.cells + F.digits * B.degreeSum := by rw [sumSecondCells, degreeSum]; ring
        _ ≤ 2 ^ r2.length := room)
  rw [batchOpenLifts_value B cs amb pts w st F opened α r2 top short openPow topOpen room, hChain, hQD,
    hQ1, hQ2]
  simp only [ringTerm, imageRow_degree, firstRow_degree, secondRow_degree, imageRow_scan,
    batchFirstCell_eq, batchSecondCell_eq, mul_sub, Finset.sum_sub_distrib, eqIndex_into,
    sumFirst, sumSecond, sumFirstCells, sumSecondCells, zero_add, one_mul, Nat.add_assoc]
  ring

end Open

section Insides

variable {n : ℕ} (B : Batch (Fin n))

/-- The quotient digits of the rows of a group of a fold with opening groups end in the cube. -/
theorem batchGroup_insides (digits : ℕ) {μ : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) (g : Fin n) :
    B.chainStart + digits * codeOffset B.groupDegrees g + digits * (B.planes g * B.carrierDegree g) +
        (B.shape g).innerRank * (digits * (B.shape g).innerDegree) +
          (B.shape g).slices * (B.shape g).outerRank * (digits * (B.shape g).outerDegree) ≤ 2 ^ μ ∧
      B.secondStart + digits * B.mainDegrees + digits * codeOffset (fun h =>
        firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
          firstRank (B.shape g).chain * (digits * firstDegree (B.shape g).chain) ≤ 2 ^ μ ∧
      B.cells + digits * B.mainDegrees + digits * B.firstDegrees + digits * codeOffset (fun h =>
        secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g +
          secondRank (B.shape g).chain * (digits * secondDegree (B.shape g).chain) ≤ 2 ^ μ := by
  have degreeSum : B.degreeSum = B.mainDegrees + B.firstDegrees +
      (∑ h, secondRank (B.shape h).chain * secondDegree (B.shape h).chain +
        secondRank B.chain * secondDegree B.chain) := by
    unfold Batch.degreeSum Batch.mainDegrees Batch.firstDegrees
    ring
  have cellsLe := B.chainStart_le_cells
  have secondLe : B.secondStart ≤ B.cells := by
    unfold Batch.cells Batch.openSecondStart
    omega
  unfold Batch.quotientRoom at room
  rw [degreeSum, Nat.mul_add, Nat.mul_add] at room
  have main := Nat.mul_le_mul_left digits (groupDegrees_le B g)
  have first := Nat.mul_le_mul_left digits ((codeOffset_add_le (fun h =>
    firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g).trans
      (show _ ≤ B.firstDegrees from Nat.le_add_right _ _))
  have second := Nat.mul_le_mul_left digits ((codeOffset_add_le (fun h =>
    secondRank (B.shape h).chain * secondDegree (B.shape h).chain) g).trans
      (Nat.le_add_right _ (secondRank B.chain * secondDegree B.chain)))
  rw [Nat.mul_add] at main first second
  refine ⟨?_, ?_, ?_⟩
  · have : digits * codeOffset B.groupDegrees g + digits * (B.planes g * B.carrierDegree g) +
        (B.shape g).innerRank * (digits * (B.shape g).innerDegree) +
          (B.shape g).slices * (B.shape g).outerRank * (digits * (B.shape g).outerDegree) =
        digits * codeOffset B.groupDegrees g + digits * B.groupDegrees g := by
      unfold Batch.groupDegrees
      ring
    omega
  · have : firstRank (B.shape g).chain * (digits * firstDegree (B.shape g).chain) =
        digits * (firstRank (B.shape g).chain * firstDegree (B.shape g).chain) := by ring
    omega
  · have : secondRank (B.shape g).chain * (digits * secondDegree (B.shape g).chain) =
        digits * (secondRank (B.shape g).chain * secondDegree (B.shape g).chain) := by ring
    omega

end Insides

section Groups

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The parts of the replay of the group at the index `k` (its parts, its quotient digits of the
rows before the rows of `D` and of its rows of `F_1` and `F_2`) are the families of the group
`groupOrder n k` in `batchReduction_weight`. -/
theorem groupFamilies_batch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
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
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length) (code : F.order = .code) :
    ((([partialLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          responseLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          imageLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          outerLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))] ++
        chainLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
          (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))
          (∑ g, B.groupCount g + B.openRank +
            codeOffset (fun g => firstRank (B.shape g).chain) (groupOrder n k))
          (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain + firstRank B.chain +
            codeOffset (fun g => secondRank (B.shape g).chain) (groupOrder n k))).map
          fun p => ((liftsValue (Tables.of α r2 (Nat.log 2 top) top) p).into : E)).sum +
        ((contractionValue F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
          (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))
          (Tables.of α r2 (Nat.log 2 top) top)).into : E)) +
      ((carrierQuotient F (batchGroup B cs amb pts w k) α
          (eqIndex F.tau1 (codeOffset B.groupCount (groupOrder n k)))
          (B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k)) ++
        ((rowStarts F.digits (B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) +
            F.digits * ((batchGroup B cs amb pts w k).opening.planes *
              (batchGroup B cs amb pts w k).opening.carrierDegree))
          [((batchGroup B cs amb pts w k).shape.innerDegree,
              (batchGroup B cs amb pts w k).shape.innerRank),
            ((batchGroup B cs amb pts w k).shape.outerDegree,
              (batchGroup B cs amb pts w k).shape.slices *
                (batchGroup B cs amb pts w k).shape.outerRank)]).zipIdx).map
          fun x => quotientLift F.base F.digits x.1.1 α
            (eqIndex F.tau1 (codeOffset B.groupCount (groupOrder n k) + 1 + x.2)) x.1.2).map
        fun x => ((x.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ∑ x : Fin (firstRank (B.shape (groupOrder n k)).chain),
        ((quotientLift F.base F.digits (firstDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank +
            offset (fun j => firstRank (B.shape (groupOrder n j)).chain) k + x))
          (B.secondStart + F.digits * B.mainDegrees + F.digits * offset (fun j =>
            firstRank (B.shape (groupOrder n j)).chain * firstDegree (B.shape (groupOrder n j)).chain) k +
              x * (F.digits * firstDegree (B.shape (groupOrder n k)).chain))).value
                (Tables.of α r2 (Nat.log 2 top) top)).into +
      ∑ x : Fin (secondRank (B.shape (groupOrder n k)).chain),
        ((quotientLift F.base F.digits (secondDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain +
            firstRank B.chain + offset (fun j => secondRank (B.shape (groupOrder n j)).chain) k + x))
          (B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * offset (fun j =>
            secondRank (B.shape (groupOrder n j)).chain *
              secondDegree (B.shape (groupOrder n j)).chain) k +
                x * (F.digits * secondDegree (B.shape (groupOrder n k)).chain))).value
                  (Tables.of α r2 (Nat.log 2 top) top)).into =
    (∑ t : Fin (B.planes (groupOrder n k)), eqAt (F.tau1.map Ext.into)
        (codeOffset B.groupCount (groupOrder n k)) * (β (groupOrder n k) t *
          ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
            (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
            (evaluationRow (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (β (groupOrder n k))
              (st.point (groupOrder n k)) (challenge (groupOrder n k)) t)
            fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) +
              a * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
                t * B.carrierDegree (groupOrder n k) + j) +
      (∑ x : Fin (B.shape (groupOrder n k)).innerRank, eqAt (F.tau1.map Ext.into)
          (codeOffset B.groupCount (groupOrder n k) + (1 + x)) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
          (innerRow (B.opening (groupOrder n k)) (B.fold (groupOrder n k)) (st.key (groupOrder n k))
            (challenge (groupOrder n k)) x)
          (fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) +
            F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
              x * (F.digits * (B.shape (groupOrder n k)).innerDegree) +
                a * (B.shape (groupOrder n k)).innerDegree + j) +
        ∑ x : Fin ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank),
          eqAt (F.tau1.map Ext.into) (codeOffset B.groupCount (groupOrder n k) +
            (1 + ((B.shape (groupOrder n k)).innerRank + x))) *
          ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
            (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
            (imageRow (B.shape (groupOrder n k)).chain (outerForm (B.opening (groupOrder n k))
              (B.fold (groupOrder n k)) (st.key (groupOrder n k)))
              ((B.opening (groupOrder n k)).firstCell (B.fold (groupOrder n k)))
              (st.payload (groupOrder n k)) x)
            (fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) +
              F.digits * (B.planes (groupOrder n k) * B.carrierDegree (groupOrder n k)) +
                (B.shape (groupOrder n k)).innerRank * (F.digits * (B.shape (groupOrder n k)).innerDegree) +
                  x * (F.digits * (B.shape (groupOrder n k)).outerDegree) +
                    a * (B.shape (groupOrder n k)).outerDegree + j))) +
    ∑ x : Fin (firstRank (B.shape (groupOrder n k)).chain), eqAt (F.tau1.map Ext.into)
        (∑ g, B.groupCount g + (1 * B.openRank +
          (codeOffset (fun h => firstRank (B.shape h).chain) (groupOrder n k) + x))) *
      ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
        (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
        (firstRow (B.shape (groupOrder n k)).chain (st.key (groupOrder n k)).chain
          ((B.opening (groupOrder n k)).firstCell (B.fold (groupOrder n k)))
          ((B.opening (groupOrder n k)).secondCell (B.fold (groupOrder n k))) x)
        (fun a j => B.secondStart + F.digits * B.mainDegrees + F.digits * codeOffset (fun h =>
          firstRank (B.shape h).chain * firstDegree (B.shape h).chain) (groupOrder n k) +
            x * (F.digits * firstDegree (B.shape (groupOrder n k)).chain) +
              a * firstDegree (B.shape (groupOrder n k)).chain + j) +
    ∑ x : Fin (secondRank (B.shape (groupOrder n k)).chain), eqAt (F.tau1.map Ext.into)
        (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain +
          (firstRank B.chain + (codeOffset (fun h => secondRank (B.shape h).chain)
            (groupOrder n k) + x))))) *
      ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
        (fun o => (groupCells B F.digits (groupOrder n k) o).elim 0 (eqAt (r2.map Ext.into)))
        (secondRow (B.shape (groupOrder n k)).chain (st.key (groupOrder n k)).chain
          ((B.opening (groupOrder n k)).secondCell (B.fold (groupOrder n k)))
          (st.payload (groupOrder n k)) x)
        (fun a j => B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees +
          F.digits * codeOffset (fun h => secondRank (B.shape h).chain *
            secondDegree (B.shape h).chain) (groupOrder n k) +
              x * (F.digits * secondDegree (B.shape (groupOrder n k)).chain) +
                a * secondDegree (B.shape (groupOrder n k)).chain + j) := by
  obtain ⟨insideB, inside1, inside2⟩ := batchGroup_insides B F.digits room (groupOrder n k)
  have h := groupWeight_batch B cs amb pts w β F st keyed challenge k challenged ambiented based
    positioned packed α r2 top short numbers within room code
    (codeOffset B.groupCount (groupOrder n k))
    (∑ g, B.groupCount g + B.openRank +
      codeOffset (fun g => firstRank (B.shape g).chain) (groupOrder n k))
    (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain + firstRank B.chain +
      codeOffset (fun g => secondRank (B.shape g).chain) (groupOrder n k))
    (B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k))
    (B.secondStart + F.digits * B.mainDegrees + F.digits * codeOffset (fun h =>
      firstRank (B.shape h).chain * firstDegree (B.shape h).chain) (groupOrder n k))
    (B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * codeOffset (fun h =>
      secondRank (B.shape h).chain * secondDegree (B.shape h).chain) (groupOrder n k))
    (by omega) (by omega) insideB inside1 inside2
  rw [offset_order (fun g => firstRank (B.shape g).chain) k,
    offset_order (fun h => firstRank (B.shape h).chain * firstDegree (B.shape h).chain) k,
    offset_order (fun g => secondRank (B.shape g).chain) k,
    offset_order (fun h => secondRank (B.shape h).chain * secondDegree (B.shape h).chain) k,
    List.map_append, List.sum_append]
  rw [lv_into _ (List.map _ (rowStarts _ _ _).zipIdx)] at h
  simp only [List.map_append, List.map_cons, List.map_nil, List.sum_append, List.sum_cons,
    List.sum_nil, add_zero, lv_into _ (partialLifts _ _ _ _), lv_into _ (responseLifts _ _ _ _),
    lv_into _ (imageLifts _ _ _ _), lv_into _ (outerLifts _ _ _ _)]
  dsimp only [batchGroup] at h ⊢
  simp only [one_mul, Nat.add_assoc] at h ⊢
  linear_combination h

/-- The parts of the replay of the groups and their quotient digits before the rows of `D` and
in the rows of `F_1` and `F_2` are the families of the groups in `batchReduction_weight`. -/
theorem groupsWeight_batch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (F : FoldInputs) (st : B.Statement E) (keyed : ∀ g, (st.key g).Shared (flatOf F.setup))
    (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _ ((cs k).getD i
      (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (ambiented : ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((amb k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i))
    (based : ∀ g t, β g t = (extBasis.getD t Ext.zero).into)
    (positioned : ∀ k x, (st.point (groupOrder n k)).position x =
      eqAt ((batchGroup B cs amb pts w k).segments.2.2.1.map Ext.into) x)
    (packed : ∀ k a, (st.point (groupOrder n k)).pack a =
      eqAt ((batchGroup B cs amb pts w k).segments.1.map Ext.into) a)
    (α : Ext) (r2 : List Ext) (top : ℕ) (short : Nat.log 2 top ≤ r2.length)
    (numbers : ∀ k, (batchGroup B cs amb pts w k).Fits r2.length)
    (within : ∀ k, max B.openDegree (B.shape (groupOrder n k)).innerDegree ≤ top)
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length) (code : F.order = .code) :
    ∑ k, ((([partialLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          responseLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          imageLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k)),
          outerLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
            (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))] ++
        chainLifts F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
          (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))
          (∑ g, B.groupCount g + B.openRank +
            codeOffset (fun g => firstRank (B.shape g).chain) (groupOrder n k))
          (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain + firstRank B.chain +
            codeOffset (fun g => secondRank (B.shape g).chain) (groupOrder n k))).map
          fun p => ((liftsValue (Tables.of α r2 (Nat.log 2 top) top) p).into : E)).sum +
        ((contractionValue F (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y))
          (batchGroup B cs amb pts w k) (codeOffset B.groupCount (groupOrder n k))
          (Tables.of α r2 (Nat.log 2 top) top)).into : E)) +
      ∑ k, ((carrierQuotient F (batchGroup B cs amb pts w k) α
          (eqIndex F.tau1 (codeOffset B.groupCount (groupOrder n k)))
          (B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k)) ++
        ((rowStarts F.digits (B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) +
            F.digits * ((batchGroup B cs amb pts w k).opening.planes *
              (batchGroup B cs amb pts w k).opening.carrierDegree))
          [((batchGroup B cs amb pts w k).shape.innerDegree,
              (batchGroup B cs amb pts w k).shape.innerRank),
            ((batchGroup B cs amb pts w k).shape.outerDegree,
              (batchGroup B cs amb pts w k).shape.slices *
                (batchGroup B cs amb pts w k).shape.outerRank)]).zipIdx).map
          fun x => quotientLift F.base F.digits x.1.1 α
            (eqIndex F.tau1 (codeOffset B.groupCount (groupOrder n k) + 1 + x.2)) x.1.2).map
        fun x => ((x.value (Tables.of α r2 (Nat.log 2 top) top)).into : E)).sum +
      ∑ k, ∑ x : Fin (firstRank (B.shape (groupOrder n k)).chain),
        ((quotientLift F.base F.digits (firstDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank +
            offset (fun j => firstRank (B.shape (groupOrder n j)).chain) k + x))
          (B.secondStart + F.digits * B.mainDegrees + F.digits * offset (fun j =>
            firstRank (B.shape (groupOrder n j)).chain * firstDegree (B.shape (groupOrder n j)).chain) k +
              x * (F.digits * firstDegree (B.shape (groupOrder n k)).chain))).value
                (Tables.of α r2 (Nat.log 2 top) top)).into +
      ∑ k, ∑ x : Fin (secondRank (B.shape (groupOrder n k)).chain),
        ((quotientLift F.base F.digits (secondDegree (B.shape (groupOrder n k)).chain) α
          (eqIndex F.tau1 (∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain +
            firstRank B.chain + offset (fun j => secondRank (B.shape (groupOrder n j)).chain) k + x))
          (B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees + F.digits * offset (fun j =>
            secondRank (B.shape (groupOrder n j)).chain *
              secondDegree (B.shape (groupOrder n j)).chain) k +
                x * (F.digits * secondDegree (B.shape (groupOrder n k)).chain))).value
                  (Tables.of α r2 (Nat.log 2 top) top)).into =
    ∑ g, (∑ t : Fin (B.planes g), eqAt (F.tau1.map Ext.into)
          (codeOffset B.groupCount g) * (β g t *
            ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
              (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))
              (evaluationRow (B.opening g) (B.fold g) (β g)
                (st.point g) (challenge g) t)
              fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees g +
                a * (B.planes g * B.carrierDegree g) +
                  t * B.carrierDegree g + j) +
        (∑ x : Fin (B.shape g).innerRank, eqAt (F.tau1.map Ext.into)
            (codeOffset B.groupCount g + (1 + x)) *
          ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
            (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))
            (innerRow (B.opening g) (B.fold g) (st.key g)
              (challenge g) x)
            (fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees g +
              F.digits * (B.planes g * B.carrierDegree g) +
                x * (F.digits * (B.shape g).innerDegree) +
                  a * (B.shape g).innerDegree + j) +
          ∑ x : Fin ((B.shape g).slices * (B.shape g).outerRank),
            eqAt (F.tau1.map Ext.into) (codeOffset B.groupCount g +
              (1 + ((B.shape g).innerRank + x))) *
            ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
              (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))
              (imageRow (B.shape g).chain (outerForm (B.opening g)
                (B.fold g) (st.key g))
                ((B.opening g).firstCell (B.fold g))
                (st.payload g) x)
              (fun a j => B.chainStart + F.digits * codeOffset B.groupDegrees g +
                F.digits * (B.planes g * B.carrierDegree g) +
                  (B.shape g).innerRank * (F.digits * (B.shape g).innerDegree) +
                    x * (F.digits * (B.shape g).outerDegree) +
                      a * (B.shape g).outerDegree + j))) +
      ∑ g, ∑ x : Fin (firstRank (B.shape g).chain), eqAt (F.tau1.map Ext.into)
          (∑ g, B.groupCount g + (1 * B.openRank +
            (codeOffset (fun h => firstRank (B.shape h).chain) g + x))) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))
          (firstRow (B.shape g).chain (st.key g).chain
            ((B.opening g).firstCell (B.fold g))
            ((B.opening g).secondCell (B.fold g)) x)
          (fun a j => B.secondStart + F.digits * B.mainDegrees + F.digits * codeOffset (fun h =>
            firstRank (B.shape h).chain * firstDegree (B.shape h).chain) g +
              x * (F.digits * firstDegree (B.shape g).chain) +
                a * firstDegree (B.shape g).chain + j) +
      ∑ g, ∑ x : Fin (secondRank (B.shape g).chain), eqAt (F.tau1.map Ext.into)
          (∑ g, B.groupCount g + (1 * B.openRank + (∑ h, firstRank (B.shape h).chain +
            (firstRank B.chain + (codeOffset (fun h => secondRank (B.shape h).chain)
              g + x))))) *
        ringTerm (α.into : E) (r2.map Ext.into) F.base F.digits
          (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))
          (secondRow (B.shape g).chain (st.key g).chain
            ((B.opening g).secondCell (B.fold g))
            (st.payload g) x)
          (fun a j => B.cells + F.digits * B.mainDegrees + F.digits * B.firstDegrees +
            F.digits * codeOffset (fun h => secondRank (B.shape h).chain *
              secondDegree (B.shape h).chain) g +
                x * (F.digits * secondDegree (B.shape g).chain) +
                  a * secondDegree (B.shape g).chain + j) := by
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact (Finset.sum_congr rfl fun k _ => groupFamilies_batch B cs amb pts w β F st keyed challenge k
    (challenged k) (ambiented k) (based (groupOrder n k)) (positioned k) (packed k) α r2 top short
    (numbers k) (within k) room code).trans (Fintype.sum_equiv (groupOrder n) _ _ fun _ => rfl)

end Groups

section Fold

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] [Fintype E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The weight theorem of a fold with opening groups: the relation weight that the replay computes
at `r_2` (`foldFused` for the groups of `batchGroups` in the order of the code) is
`m_{τ_1}(r_2)` of `Batch.liftedReduction` for a statement whose matrices are the views of the
setup array and whose points have the weights of the opening points of the groups, the fold
challenges `c_{g,i}` of each group in `R_{q,d_car,g}` and `ι(c_{g,i})` in `R_{q,d_A,g}`, and the
basis `(1, u)`. -/
theorem foldWeight_batch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (based : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (F : FoldInputs) (V : PackingValues)
    (st : B.Statement E) (keyed : ∀ g, (st.key g).Shared (flatOf F.setup))
    (opened : st.openKey.matrix =
      Akita.Setup.shared (flatOf F.setup) B.openDegree B.openRank _ B.openSetupCode)
    (openChain : PathShared (flatOf F.setup) B.chain st.openKey.chain)
    (positioned : ∀ k x, (st.point (groupOrder n k)).position x =
      eqAt ((batchGroup B cs amb pts w k).segments.2.2.1.map Ext.into) x)
    (packed : ∀ k a, (st.point (groupOrder n k)).pack a =
      eqAt ((batchGroup B cs amb pts w k).segments.1.map Ext.into) a)
    (blocked : ∀ k i, (st.point (groupOrder n k)).block i =
      eqAt ((batchGroup B cs amb pts w k).segments.2.2.2.map Ext.into) i)
    (tailed : ∀ k j, (st.point (groupOrder n k)).tail j =
      eqAt ((batchGroup B cs amb pts w k).segments.2.1.map Ext.into) j)
    (weighted : ∀ k, st.weight (groupOrder n k) = (w k).into) (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _ ((cs k).getD i
      (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (ambiented : ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((amb k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i))
    {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (room : B.quotientRoom F.digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (numbers : ∀ k, (batchGroup B cs amb pts w k).Fits μ)
    (k₀ : Fin n) (tau1 : F.tau1 = V.tau1)
    (path : F.path = ⟨B.openDegree, B.openRank, B.chain⟩) (code : F.order = .code)
    (cells : F.cells = {}) (tau : V.tau1.length = batch) (r2 : List Ext) (length : r2.length = μ) :
    ((foldFused F (batchGroups B cs amb pts w) { V with r2 } none).weight.into : E) =
      mle (r2.map Ext.into) (((Batch.liftedReduction β st challenge μ fits F.base F.digits room batch
        rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into)) := by
  rw [← tau1, batchReduction_weight B β st challenge F.base F.digits V.alpha.into _ _ fits room rowsFit
    (by rw [List.length_map, length]) (by rw [List.length_map, tau1, tau])]
  subst length
  have bound : ∀ k : Fin n, max B.openDegree (B.shape (groupOrder n k)).innerDegree ≤
      List.foldl max B.openDegree ((batchGroups B cs amb pts w).map fun P => P.shape.innerDegree) :=
    fun k => max_le (le_foldl_max _ _).1 ((le_foldl_max _ _).2 _ (List.mem_map_of_mem
      (List.mem_map_of_mem (List.mem_finRange k))))
  have cube : List.foldl max B.openDegree ((batchGroups B cs amb pts w).map fun P =>
      P.shape.innerDegree) ≤ 2 ^ r2.length :=
    foldl_max_le _ _ ((le_max_left _ _).trans (numbers k₀).cube) fun x member => by
      obtain ⟨P, inside, rfl⟩ := List.mem_map.mp member
      obtain ⟨k, -, rfl⟩ := List.mem_map.mp inside
      exact (le_max_right _ _).trans (numbers k).cube
  simp only [foldFused, packingEnd, foldTables, path, sumList_into]
  simp only [foldParts, path, cells, List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, groupParts_eq, openParts_eq, Option.isSome_none, Bool.false_and,
    Bool.false_eq_true, ite_false, beforeIn, filterMap_some_map, Tables.of_alpha, segCell_eta,
    batchGroups_length, add_zero]
  rw [flat_parts_into, lv_into _ (List.flatMap _ _), lv_into _ (List.flatMap _ _), sum_flatMap,
    sum_flatMap, Ext.into_mul, sumList_into]
  generalize List.foldl max B.openDegree (List.map (fun P => P.shape.innerDegree)
    (batchGroups B cs amb pts w)) = top at bound cube
  have zipped : (batchGroups B cs amb pts w).zipIdx =
      (List.finRange n).map fun k => (batchGroup B cs amb pts w k, (k : ℕ)) :=
    zipIdx_finRange _
  have rows : ∀ k : Fin n, ((List.range n).map fun k =>
      before (batchGroups B cs amb pts w) (fun x => x.rowCount) k).getD k 0 =
        codeOffset B.groupCount (groupOrder n k) := fun k =>
    (getD_range_map _ _ k.isLt).trans (batchGroups_before B cs amb pts w _ _
      (fun _ => Nat.add_assoc _ _ _) k)
  have starts : ∀ k : Fin n, ((List.range n).map fun k =>
      B.chainStart + F.digits * before (batchGroups B cs amb pts w) (·.mainDegrees) k).getD
          k 0 = B.chainStart + F.digits * codeOffset B.groupDegrees (groupOrder n k) := fun k =>
    (getD_range_map _ _ k.isLt).trans (by
      rw [batchGroups_before B cs amb pts w _ B.groupDegrees (fun _ => rfl) k])
  have rowSum := batchGroups_sum B cs amb pts w (·.rowCount) B.groupCount
    (fun _ => Nat.add_assoc _ _ _)
  have firstSum := batchGroups_sum B cs amb pts w (fun Q => firstRank Q.shape.chain)
    (fun g => firstRank (B.shape g).chain) (fun _ => rfl)
  have secondSum := batchGroups_sum B cs amb pts w (fun Q => secondRank Q.shape.chain)
    (fun g => secondRank (B.shape g).chain) (fun _ => rfl)
  have widthSum := batchGroups_sum B cs amb pts w (·.openWidth)
    (fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
    (fun _ => rfl)
  have degreeSum := batchGroups_sum B cs amb pts w (·.mainDegrees) B.groupDegrees (fun _ => rfl)
  have firstBefore := batchGroups_before B cs amb pts w (fun Q => firstRank Q.shape.chain)
    (fun g => firstRank (B.shape g).chain) (fun _ => rfl)
  have secondBefore := batchGroups_before B cs amb pts w (fun Q => secondRank Q.shape.chain)
    (fun g => secondRank (B.shape g).chain) (fun _ => rfl)
  have widthBefore := batchGroups_before B cs amb pts w (·.openWidth)
    (fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
    (fun _ => rfl)
  have firstRows : (batchGroups B cs amb pts w).map (fun P =>
      (firstDegree P.shape.chain, firstRank P.shape.chain)) = (List.finRange n).map fun k =>
        (firstDegree (B.shape (groupOrder n k)).chain, firstRank (B.shape (groupOrder n k)).chain) :=
    List.map_map
  have secondRows : (batchGroups B cs amb pts w).map (fun P =>
      (secondDegree P.shape.chain, secondRank P.shape.chain)) = (List.finRange n).map fun k =>
        (secondDegree (B.shape (groupOrder n k)).chain, secondRank (B.shape (groupOrder n k)).chain) :=
    List.map_map
  obtain ⟨quotientMain, openFirst, quotientFirst, openSecond, quotientSecond⟩ :=
    Placement.of_batch B cs amb pts w F.digits
  simp only [quotientMain]
  simp only [zipped, List.map_map, Function.comp_def, sum_finRange, rows, starts, zipIdx_map_lift,
    Placement.move_batch, rowSum, firstSum, secondSum, widthSum, degreeSum, firstBefore,
    secondBefore, widthBefore, firstRows, secondRows, quotientFirst, quotientSecond, openFirst,
    openSecond]
  rw [quotients_split, quotients_split]
  have short : Nat.log 2 top ≤ r2.length := log_le_cube cube
  have scalarRow : ∑ g, B.groupCount g + B.openRank + ∑ g, firstRank (B.shape g).chain +
      firstRank B.chain + ∑ g, secondRank (B.shape g).chain + secondRank B.chain =
        ∑ g, B.groupCount g + B.restCount := by
    unfold Batch.restCount
    ring
  have hG := groupsWeight_batch B cs amb pts w β F st keyed challenge challenged ambiented based
    positioned packed V.alpha r2 top short numbers bound room code
  have hO := batchOpen_weight B cs amb pts w st F opened openChain V.alpha r2 top short k₀
    (numbers k₀) (bound k₀) room
  have hS := batchScalar_weight B cs amb pts w β st based blocked tailed weighted F.digits F.tau1
    V.alpha r2 top short numbers bound fits room rfl (∑ g, B.groupCount g + B.openRank +
      ∑ g, firstRank (B.shape g).chain + firstRank B.chain + ∑ g, secondRank (B.shape g).chain +
        secondRank B.chain)
  conv at hS => rhs; rw [scalarRow]
  linear_combination hG + hO + hS

end Fold

end NightstreamFPrime.Spec.Compression.Akita.Replay
