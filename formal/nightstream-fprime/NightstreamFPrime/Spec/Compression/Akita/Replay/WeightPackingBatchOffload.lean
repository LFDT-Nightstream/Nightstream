import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchFold

/-! The weight theorem of a fold with opening groups whose setup is offloaded to Stage 3, the root
of `key30.aks` (ePrint 2026/1983, (132); Akita revision `d50f700`,
`RelationMatrixEvaluator::eval_flat_at_point_with_deferred_setup` of
`akita-verifier/src/relation/evaluation.rs` and `common_alpha_evaluation`).

- `setupParts` are the parts of the setup matrices `A`, `B` and `D` of `foldParts`;
  `foldParts_offload`: with the claim `σ` of Stage 3, the weight of the replay is the weight
  without it minus `setupParts` plus `Tables.common low·σ` (with `filterMap_parts`, `skip_rest`,
  `skip_parts`, `list_sum_map_sub` and `flat_filter_into` for the parts of each group).
- `setupParts_batch`: `setupParts` of the groups of `batchGroups` are the rows (104) on `ẑ`, (105)
  and (106) of the specification; `batchSetup_weight`: these rows are `m_{τ_1}(r_2)` of
  `Batch.liftedReduction` minus its weight for the statement with the zero setup
  (`Statement.over 0`).
- `common_into`: `Tables.common m` is `powerScale(α, r_2[..m])`.
- `foldWeight_offload`: the weight with `σ` is `m_local(r_2) + powerScale(α, r_2[..low])·σ`.
- `batchFused_holds_iff` and `offloadFused_holds_iff`: the end check `fused.end` of a fold with
  opening groups, without and with the claim `σ` of Stage 3, is the final check of the fused
  sum-check of the specification. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Setup

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The parts of the setup matrices `A`, `B` and `D` of `foldParts`: the rows (104) on `ẑ` and the
rows (105) of each group, and the rows (106) of `D`. With the setup offloaded to Stage 3, the code
replaces them by `powerScale(α, r_2[..low])·σ` (`eval_flat_at_point_with_deferred_setup`). -/
def setupParts (F : FoldInputs) (gs : List PackGroup) (T : Tables) : Ext :=
  let L := Placement.of gs F.path F.digits
  let starts := (List.range gs.length).map fun k => before gs (·.rowCount) k
  sumList gs.zipIdx (fun x => liftsValue T (responseLifts F (L.move gs x.2 F.cells) x.1
      (starts.getD x.2 0)) + liftsValue T (outerLifts F (L.move gs x.2 F.cells) x.1
        (starts.getD x.2 0))) +
    liftsValue T (gs.zipIdx.flatMap fun x => openParts F gs L x.2 x.1 (gs.map (·.rowCount)).sum
      (beforeIn F.cells.columns gs (·.openWidth) x.2) (gs.map (·.openWidth)).sum)

theorem filterMap_parts (T : Tables) {γ : Type} (name : List Lift × ℕ → γ) :
    ∀ (l : List (List Lift)) (k : ℕ), (((l.zipIdx k).filterMap fun y =>
      if (y.2 == 1 || y.2 == 3) = true then none else some (name y, liftsValue T y.1)).map
        fun z => (z.2.into : E)).sum =
      ((l.zipIdx k).map fun y => if (y.2 == 1 || y.2 == 3) = true then 0 else
        (liftsValue T y.1).into).sum
  | [], _ => rfl
  | p :: l, k => by
      rw [List.zipIdx_cons, List.filterMap_cons, List.map_cons, List.sum_cons]
      by_cases skip : (k == 1 || k == 3) = true
      · rw [ite_eq_left skip, ite_eq_left skip, filterMap_parts T name l (k + 1), zero_add]
      · rw [ite_eq_right skip, ite_eq_right skip, List.map_cons, List.sum_cons, filterMap_parts T name l (k + 1)]

theorem skip_rest (T : Tables) : ∀ (l : List (List Lift)) (k : ℕ), 4 ≤ k →
    ((l.zipIdx k).map fun y => if (y.2 == 1 || y.2 == 3) = true then (0 : E) else
      (liftsValue T y.1).into).sum = (l.map fun p => (liftsValue T p).into).sum
  | [], _, _ => rfl
  | p :: l, k, large => by
      rw [List.zipIdx_cons, List.map_cons, List.sum_cons, List.map_cons, List.sum_cons,
        skip_rest T l (k + 1) (by omega)]
      have keep : (k == 1 || k == 3) = false := by
        simp only [Bool.or_eq_false_iff, beq_eq_false_iff_ne]
        omega
      rw [keep]
      rfl

theorem skip_parts (T : Tables) (a b c d : List Lift) (rest : List (List Lift)) :
    (([a, b, c, d] ++ rest).zipIdx.map fun y => if (y.2 == 1 || y.2 == 3) = true then (0 : E) else
      (liftsValue T y.1).into).sum = (([a, b, c, d] ++ rest).map fun p =>
        ((liftsValue T p).into : E)).sum - (liftsValue T b).into - (liftsValue T d).into := by
  simp only [List.cons_append, List.nil_append, List.zipIdx_cons, List.map_cons, List.sum_cons]
  rw [skip_rest T rest _ (by norm_num)]
  simp only [zero_add, Nat.reduceAdd, beq_self_eq_true, Bool.or_true, Bool.true_or, ite_true]
  norm_num
  ring

theorem list_sum_map_sub {α M : Type} [AddCommGroup M] (l : List α) (f g : α → M) :
    (l.map fun x => f x - g x).sum = (l.map f).sum - (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      abel

theorem flat_filter_into (T : Tables) {γ : Type} (l : List γ) (parts : γ → List (List Lift))
    (con : γ → Ext) (name : γ × ℕ → List Lift × ℕ → String) (cname : γ × ℕ → String) :
    (List.map (fun x => (x.2.into : E)) (List.flatMap (fun x =>
      List.filterMap (fun y => if (y.2 == 1 || y.2 == 3) = true then none else
        some (name x y, liftsValue T y.1)) (parts x.1).zipIdx ++ [(cname x, con x.1)])
          l.zipIdx)).sum =
      (l.map fun x => ((parts x).zipIdx.map fun y => if (y.2 == 1 || y.2 == 3) = true then (0 : E)
        else (liftsValue T y.1).into).sum + (con x).into).sum := by
  rw [sum_flatMap]
  simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, add_zero]
  have inner : ∀ x : γ × ℕ, (((parts x.1).zipIdx.filterMap fun y =>
      if (y.2 == 1 || y.2 == 3) = true then none else some (name x y, liftsValue T y.1)).map
        fun z => (z.2.into : E)).sum = ((parts x.1).zipIdx.map fun y =>
          if (y.2 == 1 || y.2 == 3) = true then (0 : E) else (liftsValue T y.1).into).sum :=
    fun x => filterMap_parts T (name x) (parts x.1) 0
  simp only [inner]
  exact congrArg List.sum (zipIdx_map_fst (fun x => ((parts x).zipIdx.map fun y =>
    if (y.2 == 1 || y.2 == 3) = true then (0 : E) else (liftsValue T y.1).into).sum +
      ((con x).into : E)) l 0)

theorem foldParts_offload (F : FoldInputs) (gs : List PackGroup) (T : Tables) (σ : Ext) :
    ((foldParts F gs T (some σ)).map fun x => (x.2.into : E)).sum =
      ((foldParts F gs T none).map fun x => (x.2.into : E)).sum - (setupParts F gs T).into +
        (T.common F.low * σ).into := by
  simp only [foldParts, List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, add_zero, Option.isSome_some, Option.isSome_none, Bool.true_and,
    Bool.false_and, Bool.false_eq_true, ite_false, filterMap_some_map]
  rw [flat_filter_into, flat_parts_into]
  simp only [setupParts, List.map_map, Function.comp_def, groupParts_eq, skip_parts, openParts_eq,
    Ext.into_add, sumList_into]
  simp only [List.sum_map_add, list_sum_map_sub]
  ring

end Setup

section Values

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The parts of the setup matrices of a fold with opening groups at `r_2` (`setupParts`): the rows
(104) of `A` on `ẑ` and the rows (105) of `B` of each group and the rows (106) of `D`, at the cells
of the code. -/
theorem setupParts_batch (st : B.Statement E) (F : FoldInputs)
    (keyed : ∀ g, (st.key g).Shared (flatOf F.setup))
    (opened : st.openKey.matrix =
      Akita.Setup.shared (flatOf F.setup) B.openDegree B.openRank _ B.openSetupCode) (α : Ext)
    (r2 : List Ext) (top : ℕ) (short : Nat.log 2 top ≤ r2.length)
    (numbers : ∀ k, (batchGroup B cs amb pts w k).Fits r2.length)
    (within : ∀ k, max B.openDegree (B.shape (groupOrder n k)).innerDegree ≤ top) (k₀ : Fin n)
    (room : B.quotientRoom F.digits ≤ 2 ^ r2.length)
    (path : F.path = ⟨B.openDegree, B.openRank, B.chain⟩) (cells : F.cells = {}) :
    ((setupParts F (batchGroups B cs amb pts w) (Tables.of α r2 (Nat.log 2 top) top)).into : E) =
      ∑ g, (-∑ a : Fin (B.shape g).innerRank, eqAt (F.tau1.map Ext.into)
            (codeOffset B.groupCount g + (1 + a)) *
          (responseForm (B.fold g) (st.key g) a).scan (α.into : E)
            (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into))) +
        ∑ j : Fin ((B.shape g).slices * (B.shape g).outerRank), eqAt (F.tau1.map Ext.into)
            (codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + j))) *
          (outerForm (B.opening g) (B.fold g) (st.key g) j).scan (α.into : E)
            (fun o => (groupCells B F.digits g o).elim 0 (eqAt (r2.map Ext.into)))) +
      ∑ x : Fin (1 * B.openRank), eqAt (F.tau1.map Ext.into) (∑ g, B.groupCount g + x) *
        (B.openForm st.openKey x).scan (α.into : E)
          (fun o => (o.map (B.gaps F.digits).place).elim 0 (eqAt (r2.map Ext.into))) := by
  have zipped : (batchGroups B cs amb pts w).zipIdx =
      (List.finRange n).map fun k => (batchGroup B cs amb pts w k, (k : ℕ)) :=
    zipIdx_finRange _
  have rows : ∀ k : Fin n, ((List.range (batchGroups B cs amb pts w).length).map fun k =>
      before (batchGroups B cs amb pts w) (fun x => x.rowCount) k).getD k 0 =
        codeOffset B.groupCount (groupOrder n k) := fun k =>
    (getD_range_map _ _ (by rw [batchGroups_length]; exact k.isLt)).trans
      (batchGroups_before B cs amb pts w _ _ (fun _ => Nat.add_assoc _ _ _) k)
  have rowSum := batchGroups_sum B cs amb pts w (·.rowCount) B.groupCount
    (fun _ => Nat.add_assoc _ _ _)
  have widthSum := batchGroups_sum B cs amb pts w (·.openWidth)
    (fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
    (fun _ => rfl)
  have widthBefore := batchGroups_before B cs amb pts w (·.openWidth)
    (fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
    (fun _ => rfl)
  unfold setupParts
  simp only [path, cells, beforeIn, Ext.into_add, sumList_into, lv_into _ (List.flatMap _ _),
    sum_flatMap, zipped, List.map_map, Function.comp_def, sum_finRange, rows,
    Placement.move_batch, rowSum, widthSum, widthBefore, openParts_eq]
  rw [batchOpenLifts_value B cs amb pts w st F opened α r2 top short
    (pow_log_top (numbers k₀).opened) ((le_max_left _ _).trans (within k₀)) room]
  congr 1
  refine (Finset.sum_congr rfl fun k _ => ?_).trans (Fintype.sum_equiv (groupOrder n) _ _ fun _ => rfl)
  have topInner : (B.shape (groupOrder n k)).innerDegree ≤ top := (le_max_right _ _).trans (within k)
  rw [lv_into, lv_into, responseLifts_value (E := E) F
      (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y)) (batchGroup B cs amb pts w k)
      _ α r2 _ top short (pow_log_top (numbers k).inner) (log_le_top topInner) topInner
      (relocated_shift B F.digits (groupOrder n k))
      (relocated_bounded B F.digits (groupOrder n k) room) (st.key (groupOrder n k))
      (keyed (groupOrder n k)).1,
    outerLifts_value (E := E) F
      (fun y => (B.gaps F.digits).place (B.relocate (groupOrder n k) y)) (batchGroup B cs amb pts w k)
      _ α r2 _ top short (pow_log_top (numbers k).outer)
      (log_le_top ((numbers k).outerTop.trans (within k))) ((numbers k).outerTop.trans (within k))
      (relocated_shift B F.digits (groupOrder n k))
      (relocated_bounded B F.digits (groupOrder n k) room) (st.key (groupOrder n k))
      (keyed (groupOrder n k)).2.1]
  simp only [eqIndex_into, groupCells, Nat.add_assoc]
  rfl

end Values

section Spec

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E] [Fintype E]
  (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E) (challenge : B.Challenge)
  (base digits : ℕ) (α : E) (τ r : List E)

/-- `m_{τ_1}(r_2)` of `Batch.liftedReduction` is the weight for the statement with the zero setup
(`Statement.over 0`: the matrices `A`, `B` and `D` are zero) plus the rows (104) on `ẑ` and the
rows (105) of `A` and `B` of each group and the rows (106) of `D` (Akita (132): `m_local` and the
part of the setup). -/
theorem batchSetup_weight {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (length : r.length = μ)
    (tau : τ.length = batch) (N : ℕ) :
    mle r (((Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit).rows
      α).batchWeight τ) =
      mle r (((Batch.liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits base digits
        room batch rowsFit).rows α).batchWeight τ) +
      (∑ g, (-∑ a : Fin (B.shape g).innerRank, eqAt τ (codeOffset B.groupCount g + (1 + a)) *
            (responseForm (B.fold g) (st.key g) a).scan α
              (fun o => (groupCells B digits g o).elim 0 (eqAt r)) +
          ∑ j : Fin ((B.shape g).slices * (B.shape g).outerRank),
            eqAt τ (codeOffset B.groupCount g + (1 + ((B.shape g).innerRank + j))) *
              (outerForm (B.opening g) (B.fold g) (st.key g) j).scan α
                (fun o => (groupCells B digits g o).elim 0 (eqAt r))) +
        ∑ x : Fin (1 * B.openRank), eqAt τ (∑ g, B.groupCount g + x) *
          (B.openForm st.openKey x).scan α (fun o => (o.map (B.gaps digits).place).elim 0 (eqAt r))) := by
  have scalar : B.scalarWeight β (st.over (0 : Fin N → Zq)) μ = B.scalarWeight β st μ := rfl
  rw [batchReduction_weight B β st challenge base digits α τ r fits room rowsFit length tau,
    batchReduction_weight B β (st.over (0 : Fin N → Zq)) challenge base digits α τ r fits room rowsFit
      length tau, scalar]
  simp only [Batch.Statement.over, ringTerm, evaluationRow_degree, innerRow_degree, imageRow_degree,
    firstRow_degree, secondRow_degree]
  simp only [innerRow, imageRow_scan, Form.scan_row, Form.scan_sub,
    Form.scan_response, outerForm, Batch.openForm, Form.scan_ofMatrix, shared_zero, zero_mul,
    aeval_rep_zero, Finset.sum_const_zero, mul_zero, sub_zero, mul_sub, Finset.sum_sub_distrib,
    Finset.sum_add_distrib, Finset.sum_neg_distrib]
  ring

end Spec

section Common

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- `Tables.common` at `m` is `powerScale(α, r_2[..m])` (`common_alpha_evaluation`). -/
theorem common_into (α : Ext) (r2 : List Ext) (bits top m : ℕ) (le : m ≤ bits)
    (short : m ≤ r2.length) (cube : 2 ^ m ≤ top) :
    (((Tables.of α r2 bits top).common m).into : E) =
      powerScale (α.into : E) ((r2.map Ext.into).take m) := by
  have sum := sum_pow_eqAt m (α.into : E) ((r2.map Ext.into).take m) 0
    (by rw [List.length_take, List.length_map]; omega)
  rw [List.take_take, min_self, zero_mul, List.drop_eq_nil_of_le
    (by rw [List.length_take, List.length_map]; omega)] at sum
  simp only [zero_add] at sum
  rw [show eqAt ([] : List E) 0 = 1 from rfl, mul_one] at sum
  rw [← sum]
  show (((lowSums r2 bits (powersOf α top)).getD m #[]).getD 0 Ext.zero).into = _
  have size : (powersOf α top).size = top := by simp [powersOf]
  rw [lowSums_getD r2 _ le, lowLevel_into r2 _ m short 0 (by rw [zero_add, one_mul, size]; exact cube)]
  refine Finset.sum_congr rfl fun c member => ?_
  rw [zero_mul, zero_add, powersOf_getD α ((Finset.mem_range.mp member).trans_le cube),
    List.map_take]

end Common

section Offload

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] [Fintype E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The weight theorem of a fold with opening groups whose setup is offloaded to Stage 3 (the root
of `key30.aks`): the weight of `foldFused` with the claim `σ` of Stage 3 is `m_local(r_2)`, the
weight of `Batch.liftedReduction` for the statement with the zero setup, plus
`powerScale(α, r_2[..low])·σ` (Akita (132); `eval_flat_at_point_with_deferred_setup`). -/
theorem foldWeight_offload (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
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
    (cells : F.cells = {}) (tau : V.tau1.length = batch) (lowFits : 2 ^ F.low ≤ B.openDegree)
    (lowShort : F.low ≤ μ) (N : ℕ) (σ : Ext) (r2 : List Ext) (length : r2.length = μ) :
    ((foldFused F (batchGroups B cs amb pts w) { V with r2 } (some σ)).weight.into : E) =
      mle (r2.map Ext.into) (((Batch.liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits
        F.base F.digits room batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into)) +
        powerScale V.alpha.into ((r2.map Ext.into).take F.low) * σ.into := by
  have hNone := foldWeight_batch B cs amb pts w β based F V st keyed opened openChain positioned
    packed blocked tailed weighted challenge challenged ambiented fits room rowsFit numbers k₀ tau1
    path code cells tau r2 length
  have hSpec := batchSetup_weight B β st challenge F.base F.digits V.alpha.into (F.tau1.map Ext.into)
    (r2.map Ext.into) fits room rowsFit (by rw [List.length_map, length])
    (by rw [List.length_map, tau1, tau]) N
  rw [← tau1] at hNone ⊢
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
  have lowTop : 2 ^ F.low ≤ List.foldl max B.openDegree ((batchGroups B cs amb pts w).map fun P =>
      P.shape.innerDegree) := lowFits.trans (le_foldl_max _ _).1
  simp only [foldFused, packingEnd, foldTables, path, sumList_into] at hNone ⊢
  rw [foldParts_offload, hNone, Ext.into_mul,
    setupParts_batch B cs amb pts w st F keyed opened V.alpha r2 _ (log_le_cube cube) numbers bound k₀
      room path cells,
    common_into V.alpha r2 _ _ F.low (Nat.le_log_of_pow_le (by norm_num) lowTop) lowShort lowTop]
  linear_combination hSpec

end Offload

section Holds

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] [Fintype E] {n : ℕ} (B : Batch (Fin n))
  (cs amb : Fin n → List (Array Fq)) (pts : Fin n → List Ext) (w : Fin n → Ext)

/-- The end check `fused.end` of a fold with opening groups is the final check of the fused
sum-check of the specification (`fusedCheck`) for `Batch.liftedReduction`, the support of
`Batch.gaps` and the virtual weight `γ·ẽq(r_virt, ·) + ζ_bin·1_{I_bin}·ẽq(r_virt, ·)`. -/
theorem batchFused_holds_iff (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
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
    (cells : F.cells = {}) (tau : V.tau1.length = batch) (length : V.r2.length = μ)
    (virtualLength : V.virtual.length = μ) (compressed : V.compressed = B.compressed) :
    (foldFused F (batchGroups B cs amb pts w) V none).holds = true ↔
      (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into)
        (((Batch.liftedReduction β st challenge μ fits F.base F.digits room batch rowsFit).rows
          V.alpha.into).batchWeight (V.tau1.map Ext.into)) +
        mle (V.r2.map Ext.into) (virtWeight (B.gaps F.digits).support (V.virtual.map Ext.into)
          V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1)) := by
  have weighted := foldWeight_batch B cs amb pts w β based F V st keyed opened openChain positioned
    packed blocked tailed weighted challenge challenged ambiented fits room rowsFit numbers k₀ tau1
    path code cells tau V.r2 length
  have normed : ((foldFused F (batchGroups B cs amb pts w) V none).normWeight.into : E) =
      mle (V.r2.map Ext.into) (fun _ => 0) := by
    rw [mle_const]
    exact Ext.into_zero
  have virt : ((foldFused F (batchGroups B cs amb pts w) V none).virt.into : E) =
      mle (V.r2.map Ext.into) (eqAt (V.virtual.map Ext.into)) := by
    show ((eqPoints V.virtual V.r2).into : E) = _
    rw [eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map, length,
      virtualLength])]
  have supported : ((foldFused F (batchGroups B cs amb pts w) V none).support.into : E) =
      mle (V.r2.map Ext.into) fun x => if (B.gaps F.digits).support x then
        eqAt (V.virtual.map Ext.into) x else 0 := by
    show ((if V.compressed then foldSupport (Placement.of (batchGroups B cs amb pts w) F.path
      F.digits) (CellEq.half V.virtual).get (CellEq.half V.r2).get else Ext.zero).into : E) = _
    by_cases binary : V.compressed
    · rw [ite_eq_left binary, path]
      exact foldSupport_batch B cs amb pts w F.digits _ _ room (by rw [List.length_map, length])
        (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
        (fun x small => CellEq.get_half V.r2 (by rw [length]; exact small))
    · have empty : B.compressed = false := by
        rw [← compressed]
        simpa using binary
      rw [ite_eq_right binary, Ext.into_zero, show (fun x =>
        if (B.gaps F.digits).support x then eqAt (V.virtual.map Ext.into) x
          else (0 : E)) = fun _ => 0 from
        funext fun x => ite_eq_right (B.gaps_support_empty F.digits empty x), mle_const]
  rw [FusedEnd.holds_iff_fused _ (B.gaps F.digits).support _ _ (V.virtual.map Ext.into)
    (V.r2.map Ext.into) weighted normed virt supported]
  show (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into) (fun x => _ +
    (Ext.zero.into : E) * 0) + mle (V.r2.map Ext.into) (virtWeight _ _ V.batch.into
      ((if V.compressed then V.binary else Ext.zero).into)) * (V.value.into * (V.value.into + 1)) ↔ _
  by_cases binary : V.compressed <;>
    simp only [binary, Bool.false_eq_true, ite_true, ite_false, Ext.into_zero, mul_zero, add_zero]

/-- The end check `fused.end` of the root of `key30.aks`, a fold with opening groups whose setup is
offloaded to Stage 3: the final check of the fused sum-check of the specification with the weight
`m_local(r_2) + powerScale(α, r_2[..low])·σ` (Akita (132)). -/
theorem offloadFused_holds_iff (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
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
    (cells : F.cells = {}) (tau : V.tau1.length = batch) (lowFits : 2 ^ F.low ≤ B.openDegree)
    (lowShort : F.low ≤ μ) (N : ℕ) (σ : Ext) (length : V.r2.length = μ)
    (virtualLength : V.virtual.length = μ) (compressed : V.compressed = B.compressed) :
    (foldFused F (batchGroups B cs amb pts w) V (some σ)).holds = true ↔
      (V.output.into : E) = V.value.into * (mle (V.r2.map Ext.into)
        (((Batch.liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits F.base F.digits room
          batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into)) +
            powerScale V.alpha.into ((V.r2.map Ext.into).take F.low) * σ.into) +
        mle (V.r2.map Ext.into) (virtWeight (B.gaps F.digits).support (V.virtual.map Ext.into)
          V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1)) := by
  have weighted := foldWeight_offload B cs amb pts w β based F V st keyed opened openChain positioned
    packed blocked tailed weighted challenge challenged ambiented fits room rowsFit numbers k₀ tau1
    path code cells tau lowFits lowShort N σ V.r2 length
  have supported : ((foldFused F (batchGroups B cs amb pts w) V (some σ)).support.into : E) =
      mle (V.r2.map Ext.into) fun x => if (B.gaps F.digits).support x then
        eqAt (V.virtual.map Ext.into) x else 0 := by
    show ((if V.compressed then foldSupport (Placement.of (batchGroups B cs amb pts w) F.path
      F.digits) (CellEq.half V.virtual).get (CellEq.half V.r2).get else Ext.zero).into : E) = _
    by_cases binary : V.compressed
    · rw [ite_eq_left binary, path]
      exact foldSupport_batch B cs amb pts w F.digits _ _ room (by rw [List.length_map, length])
        (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
        (fun x small => CellEq.get_half V.r2 (by rw [length]; exact small))
    · have empty : B.compressed = false := by
        rw [← compressed]
        simpa using binary
      rw [ite_eq_right binary, Ext.into_zero, show (fun x =>
        if (B.gaps F.digits).support x then eqAt (V.virtual.map Ext.into) x
          else (0 : E)) = fun _ => 0 from
        funext fun x => ite_eq_right (B.gaps_support_empty F.digits empty x), mle_const]
  have split : mle (V.r2.map Ext.into) (virtWeight (B.gaps F.digits).support (V.virtual.map Ext.into)
      V.batch.into (if V.compressed then (V.binary.into : E) else 0)) =
      (V.batch.into : E) * mle (V.r2.map Ext.into) (eqAt (V.virtual.map Ext.into)) +
        (if V.compressed then (V.binary.into : E) else 0) * mle (V.r2.map Ext.into)
          (fun x => if (B.gaps F.digits).support x then eqAt (V.virtual.map Ext.into) x else 0) := by
    rw [show virtWeight (B.gaps F.digits).support (V.virtual.map Ext.into) V.batch.into
      (if V.compressed then (V.binary.into : E) else 0) = fun x =>
        (V.batch.into : E) * eqAt (V.virtual.map Ext.into) x +
          (if V.compressed then (V.binary.into : E) else 0) *
            (if (B.gaps F.digits).support x then eqAt (V.virtual.map Ext.into) x else 0) from rfl,
      mle_add, mle_mul_left, mle_mul_left]
  have weighs : ((foldFused F (batchGroups B cs amb pts w) V (some σ)).weight.into : E) =
      mle (V.r2.map Ext.into) (((Batch.liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits
        F.base F.digits room batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into)) +
          powerScale V.alpha.into ((V.r2.map Ext.into).take F.low) * σ.into := weighted
  have virt : ((foldFused F (batchGroups B cs amb pts w) V (some σ)).virt.into : E) =
      mle (V.r2.map Ext.into) (eqAt (V.virtual.map Ext.into)) := by
    show ((eqPoints V.virtual V.r2).into : E) = _
    rw [eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map, length,
      virtualLength])]
  rw [FusedEnd.holds_iff (E := E), weighs, virt, supported, split]
  show (V.output.into : E) = V.value.into * (_ + (Ext.zero.into : E) * Ext.zero.into) +
    (V.batch.into * _ + ((if V.compressed then V.binary else Ext.zero).into) * _) *
      (V.value.into * (V.value.into + 1)) ↔ _
  by_cases binary : V.compressed <;>
    simp only [binary, Bool.false_eq_true, ite_true, ite_false, Ext.into_zero, mul_zero, add_zero,
      zero_mul]

end Holds

end NightstreamFPrime.Spec.Compression.Akita.Replay
