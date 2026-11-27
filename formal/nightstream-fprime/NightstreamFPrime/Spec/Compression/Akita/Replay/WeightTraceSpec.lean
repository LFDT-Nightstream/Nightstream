import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightSpec

/-! The weight at `r_2` of a trace fold against the specification (ePrint 2026/1983, Lemma 7.3,
(117) to (123); `traceRelation` and `traceReduction` of `Extract/TraceSource.lean`).

- `rowScan row α φ` is `Σ_c Σ_j κ_{A_c,α}(j)·φ(addr_c(j))`: for `φ = ẽq(r_2, ·)` the weight of the
  ring row at `r_2` (`mle_freeWeight`). `formScan` is the same sum for a `Form`.
- `traceReduction_weight`: `m_{τ_1}(r_2)` of `traceReduction` is `Σ_k ẽq(τ_1, k)·rowScan` over the
  ring rows plus `ẽq(τ_1, N)` times the scalar row (101) at `r_2`.
- `flatOf setup` is the flat setup vector of an array, and `view_flatOf` states that the view
  (35) of it is `viewEntry`.
- The family lemmas give the value of each part of `traceParts` as these sums of the rows, and
  `traceWeight_into` states that `traceTerms` and `traceScalar` are `m_{τ_1}(r_2)` of
  `traceReduction` for the statement of the setup array and the opening point.
- `normValue_into` and `supportValue_into`: the norm weight `K_{r_virt}` with `Shape.normKernel`
  and the binary-support weight at `r_2`.
- `traceFused_holds_iff`: the check `fused.end` of the code at a trace level is the final check
  of `fusedCheck` (with the weight of `foldCheckSq` at a Euclidean level) for `traceReduction`,
  and `traceFused_holds_iff_of` is it for the statement `traceStatementOf`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Scan

variable {E : Type} [Field E] [Algebra Zq E]

/-- `Σ_c Σ_{j<d_r} κ_{A_c,α}(j)·φ(addr_c(j))` of a ring row. -/
noncomputable def rowScan (row : RingRow) (α : E) (φ : ℕ → E) : E :=
  ∑ c, ∑ j ∈ Finset.range row.degree,
    residueKernel (row.multiplier c) α j * (row.address c j).elim 0 φ

/-- `rowScan` of a sum (110). -/
noncomputable def formScan {d : ℕ} [NeZero d] (f : Akita.Form d) (α : E) (φ : ℕ → E) : E :=
  ∑ c, ∑ j ∈ Finset.range d, residueKernel (f.multiplier c) α j * (f.address c j).elim 0 φ

variable {d : ℕ} [NeZero d] (α : E) (φ : ℕ → E)

theorem rowScan_row (f : Akita.Form d) (target : Rq d) : rowScan (f.row target) α φ = formScan f α φ :=
  (Fintype.equivFin f.Occurrence).symm.sum_comp fun c =>
    ∑ j ∈ Finset.range d, residueKernel (f.multiplier c) α j * (f.address c j).elim 0 φ

theorem residueKernel_neg (A : Rq d) (j : ℕ) : residueKernel (-A) α j = -residueKernel A α j := by
  rw [residueKernel, residueKernel, neg_mul, map_neg]

theorem residueKernel_smul (s : Zq) (A : Rq d) (j : ℕ) :
    residueKernel (s • A) α j = algebraMap Zq E s * residueKernel A α j := by
  rw [residueKernel, residueKernel, smul_mul_assoc, map_smul, Algebra.smul_def]

theorem residueKernel_mul_natCast (A : Rq d) (n j : ℕ) :
    residueKernel (A * (n : Rq d)) α j = (n : E) * residueKernel A α j := by
  rw [residueKernel, residueKernel, mul_comm A (n : Rq d), mul_assoc, ← nsmul_eq_mul, map_nsmul,
    nsmul_eq_mul]

theorem formScan_sub (f g : Akita.Form d) :
    formScan (f.sub g) α φ = formScan f α φ - formScan g α φ := by
  unfold formScan Akita.Form.sub
  rw [Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, residueKernel_neg, neg_mul, Finset.sum_neg_distrib,
    sub_eq_add_neg]

theorem rowScan_congr {row : RingRow} {bound : ℕ} (below : row.Below bound) {φ ψ : ℕ → E}
    (same : ∀ x < bound, φ x = ψ x) : rowScan row α φ = rowScan row α ψ :=
  Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun j _ => by
    cases read : row.address c j with
    | none => rfl
    | some x => exact congrArg (residueKernel (row.multiplier c) α j * ·) (same x (below c j x read))

/-- `Σ_j κ_A(j)·Σ_u s_u·e_u(j) = Σ_u Σ_j κ_{A·s_u}(j)·e_u(j)`: the reads of a term share the
kernel of its multiplier. -/
theorem kernel_reads {ι : Type} (T : Finset ι) (A : Rq d) (s : ι → ℕ) (e : ι → ℕ → E) :
    ∑ j ∈ Finset.range d, residueKernel A α j * ∑ u ∈ T, (s u : E) * e u j =
      ∑ u ∈ T, ∑ j ∈ Finset.range d, residueKernel (A * (s u : Rq d)) α j * e u j := by
  simp only [residueKernel_mul_natCast, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun j _ => by ring

end Scan

section Lists

theorem sum_finRange {M : Type} [AddCommMonoid M] {n : ℕ} (f : Fin n → M) :
    ((List.finRange n).map f).sum = ∑ i, f i :=
  (Fin.sum_univ_def f).symm

theorem sum_flatMap {M : Type} [AddCommMonoid M] {α β : Type} (l : List α) (g : α → List β)
    (f : β → M) : ((l.flatMap g).map f).sum = (l.map fun a => ((g a).map f).sum).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, List.map_append, List.sum_append, ih, List.map_cons,
      List.sum_cons]

variable {E : Type} [CommRing E] [Algebra Zq E]

theorem reads_finRange {n : ℕ} (s : Fin n → ℕ) (addr : Fin n → ℕ → Option ℕ) (φ : ℕ → E)
    (j : ℕ) :
    ((((List.finRange n).map fun u => (natFq (s u), addr u)) : List Read).map fun r =>
      algebraMap Zq E r.1.toZq * (r.2 j).elim 0 φ).sum = ∑ u, (s u : E) * (addr u j).elim 0 φ := by
  rw [List.map_map, sum_finRange]
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [Function.comp, natFq, Fq.toZq_ofNat, map_natCast]

theorem reads_one (addr : ℕ → Option ℕ) (φ : ℕ → E) (j : ℕ) :
    ((([(Fq.one, addr)]) : List Read).map fun r =>
      algebraMap Zq E r.1.toZq * (r.2 j).elim 0 φ).sum = (addr j).elim 0 φ := by
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Fq.toZq_one, map_one,
    one_mul, add_zero]

end Lists

section Setup

/-- The flat setup vector of an array: entry `p` is element `p` of the array. -/
noncomputable def flatOf (setup : Array Fq) : Fin setup.size → Zq := fun p => (Poly.get setup p).toZq

/-- The view (35) of the flat setup vector of an array is `viewEntry`. -/
theorem view_flatOf (setup : Array Fq) (I : Akita.Setup.Instance) (r : Fin I.rows)
    (c : Fin I.cols) :
    Akita.Setup.view I (flatOf setup) r c = Poly.toRq I.degree (viewEntry setup I.degree I.cols r c) := by
  refine ext_coeff fun l small => ?_
  rw [Poly.coeff_toRq _ small, viewEntry, Poly.ring, Poly.get_ofFn _ small]
  simp only [Akita.Setup.view, Matrix.of_apply, coeff_ofCoeffs _ small]
  split
  · rfl
  · rename_i outside
    simp only [Poly.get, Array.getD, dite_eq_right outside, Fq.toZq_zero]

theorem shared_flatOf (setup : Array Fq) (d n width : ℕ) [NeZero d] {cols : Type}
    (code : cols → Fin width) (r : Fin n) (c : cols) :
    Akita.Setup.shared (flatOf setup) d n width code r c =
      Poly.toRq d (viewEntry setup d width r (code c)) :=
  view_flatOf setup ⟨d, n, width, false, 0⟩ r (code c)

end Setup

section Rows

variable {E : Type} [Field E] [Algebra Zq E] [Fintype E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor}

/-- The weight `m_{τ_1}(r_2)` of `traceReduction`: the ring rows with the coefficients
`ẽq(τ_1, k)`, and the scalar row (101) with `ẽq(τ_1, N)`, for each table `φ` that is `ẽq(r_2, ·)`
on the cells. -/
theorem traceReduction_weight (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (α : E) {τ r : List E}
    (length : r.length = μ) (tau : τ.length = batch) (φ : ℕ → E)
    (same : ∀ x < O.cells fold, φ x = eqAt r x) :
    mle r (((traceReduction st challenge μ fits batch rowsFit).rows α).batchWeight τ) =
      ∑ k : Fin O.traceRowCount, eqAt τ k * rowScan (traceRows O fold st challenge k) α φ +
        eqAt τ O.traceRowCount * ∑ entry : Fin S.blocks × Fin S.innerDegree × Fin S.openingDigits,
          (O.partialPosition fold entry.1 entry.2.2 entry.2.1).elim 0 φ *
            (st.scale * (st.block entry.1 * ((S.openingBase : Zq) ^ (entry.2.2 : ℕ) •
              st.functional (AdjoinRoot.root (modulus S.innerDegree) ^ (entry.2.1 : ℕ))))) := by
  have count : ((traceReduction st challenge μ fits batch rowsFit).rows α).count ≤
      2 ^ τ.length := tau ▸ rowsFit
  rw [mle_batchWeight _ τ r count]
  have atRow : ∀ i : Fin ((traceReduction st challenge μ fits batch rowsFit).rows α).count,
      ((traceReduction st challenge μ fits batch rowsFit).rows α).weight i =
        ((traceReduction st challenge μ fits batch rowsFit).rows α).weightAt i := fun i =>
    funext fun u => by rw [FieldRows.weightAt, dite_eq_left i.isLt]
  simp only [atRow]
  rw [Fin.sum_univ_eq_sum_range (fun i =>
    eqAt τ i * mle r (((traceReduction st challenge μ fits batch rowsFit).rows α).weightAt i))]
  show ∑ i ∈ Finset.range (O.traceRowCount + 1), eqAt τ i *
      mle r (((traceRelation O fold st challenge μ fits).freeRows α).weightAt i) = _
  have ring : ∀ k : Fin O.traceRowCount,
      ((traceRelation O fold st challenge μ fits).freeRows α).weightAt k =
        (traceRows O fold st challenge k).freeWeight α := fun k => funext fun u => by
    show (if h : (k : ℕ) < O.traceRowCount + 1 then
      Fin.lastCases (motive := fun _ => ℕ → E) (traceScalarWeight O fold st)
        (fun k u => ∑ q : Fin O.traceRowCount,
          if id q = k then (1 : E) * (traceRows O fold st challenge q).freeWeight α u else 0)
        ⟨k, h⟩ u else 0) = _
    rw [dite_eq_left (Nat.lt_succ_of_lt k.isLt)]
    show Fin.lastCases (motive := fun _ => ℕ → E) (traceScalarWeight O fold st)
        (fun k u => ∑ q : Fin O.traceRowCount,
          if id q = k then (1 : E) * (traceRows O fold st challenge q).freeWeight α u else 0)
        (Fin.castSucc k) u = _
    rw [Fin.lastCases_castSucc]
    simp only [id, one_mul]
    rw [Finset.sum_ite_eq' Finset.univ k, ite_eq_left (Finset.mem_univ k)]
  have scalar : ((traceRelation O fold st challenge μ fits).freeRows α).weightAt O.traceRowCount =
      traceScalarWeight O fold st := funext fun u => by
    show (if h : O.traceRowCount < O.traceRowCount + 1 then
      Fin.lastCases (motive := fun _ => ℕ → E) (traceScalarWeight O fold st)
        (fun k u => ∑ q : Fin O.traceRowCount,
          if id q = k then (1 : E) * (traceRows O fold st challenge q).freeWeight α u else 0)
        ⟨O.traceRowCount, h⟩ u else 0) = _
    rw [dite_eq_left (Nat.lt_succ_self _)]
    show Fin.lastCases (motive := fun _ => ℕ → E) (traceScalarWeight O fold st)
        (fun k u => ∑ q : Fin O.traceRowCount,
          if id q = k then (1 : E) * (traceRows O fold st challenge q).freeWeight α u else 0)
        (Fin.last O.traceRowCount) u = _
    rw [Fin.lastCases_last]
  rw [Finset.sum_range_succ, scalar, ← Fin.sum_univ_eq_sum_range (fun i =>
    eqAt τ i * mle r (((traceRelation O fold st challenge μ fits).freeRows α).weightAt i))]
  congr 1
  · refine Finset.sum_congr rfl fun k _ => ?_
    rw [ring k, mle_freeWeight _ α r fun c j x _ read =>
      length ▸ ((traceRows_below O fold st challenge k c j x read).trans_le fits)]
    exact congrArg _ (rowScan_congr α (traceRows_below O fold st challenge k) same).symm
  · rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length]
    congr 1
    unfold traceScalarWeight
    rw [sum_mul_indicator _ _ _ _ fun entry x read =>
      (O.partialPosition_lt fold _ _ _ read).trans_le fits]
    refine Finset.sum_congr rfl fun entry _ => ?_
    cases read : O.partialPosition fold entry.1 entry.2.2 entry.2.1 with
    | none => rfl
    | some x => exact congrArg (· * _) (same x (O.partialPosition_lt fold _ _ _ read)).symm

end Rows

section Head

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor} (I : TraceInputs) (α : Ext) (eq : ℕ → Ext)

theorem toRq_unitRing (d : ℕ) [NeZero d] : Poly.toRq d (unitRing d) = 1 :=
  ext_coeff fun k small => by
    rw [Poly.coeff_toRq _ small, unitRing, Poly.get_ofFn _ small, coeff_one]
    by_cases zero : k = 0 <;> simp [zero, Fq.toZq_one, Fq.toZq_zero]

theorem toRq_monomial {d j : ℕ} [NeZero d] (small : j < d) :
    Poly.toRq d (monomial d j) = AdjoinRoot.root (modulus d) ^ j :=
  ext_coeff fun k inside => by
    rw [Poly.coeff_toRq _ inside, monomial, Poly.get_ofFn _ inside, coeff_root_pow small inside]
    by_cases same : k = j <;> simp [same, Fq.toZq_one, Fq.toZq_zero]

omit [Algebra Zq E] [Degree2 E] in
/-- `Σ_i Σ_r c_r·X_r(i) = Σ_r c_r·Σ_i X_r(i)`. -/
theorem sum_rows_out {κ ι : Type} [Fintype κ] [Fintype ι] (c : κ → E) (X : κ → ι → E) :
    ∑ i, ∑ r, c r * X r i = ∑ r, c r * ∑ i, X r i := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun r _ => (Finset.mul_sum _ _ _).symm

theorem contract_mul {d : ℕ} [NeZero d] (quarter : 4 * (d / 4) = d) (c : Ext) (n m : ℕ) :
    Poly.toRq d (Poly.scale d (natFq n) (Poly.subMul d (d / 4) c (unitRing d))) * (m : Rq d) =
      subfield (E := E) quarter c.into * ((n : Rq d) * (m : Rq d)) := by
  rw [Poly.toRq_scale, Poly.toRq_subMul (E := E) quarter, toRq_unitRing, mul_one, natFq,
    Fq.toZq_ofNat, Nat.cast_smul_eq_nsmul, nsmul_eq_mul]
  ring

variable (α' : E) (φ : ℕ → E)

omit [Degree2 E] in
theorem formScan_response (key : Key S) (a : Fin S.innerRank) :
    formScan (responseForm fold key a) α' φ =
      ∑ x : Fin S.blockLength, ∑ u : Fin S.sourceDigits, ∑ f : Fin fold.responseDigits,
        ∑ j ∈ Finset.range S.innerDegree,
          residueKernel (key.inner a (x, u) * (fold.responseBase : Rq S.innerDegree) ^ (f : ℕ)) α' j *
            (S.responseCell fold ((x, u), f) j).elim 0 φ := by
  show ∑ c : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits,
    ∑ j ∈ Finset.range S.innerDegree,
      residueKernel (key.inner a c.1 * (fold.responseBase : Rq S.innerDegree) ^ (c.2 : ℕ)) α' j *
        (S.responseCell fold c j).elim 0 φ = _
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]

omit [Degree2 E] in
theorem formScan_contract (subfield' : E →+* Rq S.innerDegree) (position : Fin S.blockLength → E) :
    formScan (traceContractForm fold subfield' position) α' φ =
      ∑ x : Fin S.blockLength, ∑ u : Fin S.sourceDigits, ∑ f : Fin fold.responseDigits,
        ∑ j ∈ Finset.range S.innerDegree,
          residueKernel (subfield' (position x) * ((S.sourceBase : Rq S.innerDegree) ^ (u : ℕ) *
            (fold.responseBase : Rq S.innerDegree) ^ (f : ℕ))) α' j *
            (S.responseCell fold ((x, u), f) j).elim 0 φ := by
  show ∑ c : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits,
    ∑ j ∈ Finset.range S.innerDegree,
      residueKernel (subfield' (position c.1.1) * ((S.sourceBase : Rq S.innerDegree) ^ (c.1.2 : ℕ) *
        (fold.responseBase : Rq S.innerDegree) ^ (c.2 : ℕ))) α' j *
        (S.responseCell fold c j).elim 0 φ = _
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]

omit [Degree2 E] in
theorem formScan_partial (challenge : Fin S.blocks → Rq S.innerDegree) :
    formScan (tracePartialForm O fold challenge) α' φ =
      ∑ i : Fin S.blocks, ∑ u : Fin S.openingDigits, ∑ j ∈ Finset.range S.innerDegree,
        residueKernel (challenge i * (S.openingBase : Rq S.innerDegree) ^ (u : ℕ)) α' j *
          (O.partialPosition fold i u j).elim 0 φ := by
  show ∑ c : Fin S.blocks × Fin S.openingDigits, ∑ j ∈ Finset.range S.innerDegree,
    residueKernel (challenge c.1 * (S.openingBase : Rq S.innerDegree) ^ (c.2 : ℕ)) α' j *
      (O.partialPosition fold c.1 c.2 j).elim 0 φ = _
  rw [Fintype.sum_prod_type]

omit [Degree2 E] in
theorem formScan_image (challenge : Fin S.blocks → Rq S.innerDegree) (a : Fin S.innerRank) :
    formScan (traceImageForm O fold challenge a) α' φ =
      ∑ i : Fin S.blocks, ∑ u : Fin S.openingDigits, ∑ j ∈ Finset.range S.innerDegree,
        residueKernel (challenge i * (S.openingBase : Rq S.innerDegree) ^ (u : ℕ)) α' j *
          (O.innerCell fold i a u j).elim 0 φ := by
  show ∑ c : Fin S.blocks × Fin S.openingDigits, ∑ j ∈ Finset.range S.innerDegree,
    residueKernel (challenge c.1 * (S.openingBase : Rq S.innerDegree) ^ (c.2 : ℕ)) α' j *
      (O.innerCell fold c.1 a c.2 j).elim 0 φ = _
  rw [Fintype.sum_prod_type]

/-- The value of the term of the column `(x, u)` of `ẑ`. -/
theorem responseTerm_value (quarter : 4 * (S.innerDegree / 4) = S.innerDegree) (key : Key S)
    (keyed : key.inner =
      Akita.Setup.shared (flatOf I.setup) S.innerDegree S.innerRank _ S.innerSetupCode)
    (x : Fin S.blockLength) (u : Fin S.sourceDigits) :
    ((termValue α eq (responseTerm (fold := fold) I x u)).into : E) =
      -(∑ a : Fin S.innerRank, ((I.coef (1 + (a : ℕ))).into : E) *
        ∑ f : Fin fold.responseDigits, ∑ j ∈ Finset.range S.innerDegree,
          residueKernel (key.inner a (x, u) * (fold.responseBase : Rq S.innerDegree) ^ (f : ℕ))
            (α.into : E) j * (S.responseCell fold ((x, u), f) j).elim 0 fun x => (eq x).into) -
        ((I.coef 0).into : E) * ∑ f : Fin fold.responseDigits, ∑ j ∈ Finset.range S.innerDegree,
          residueKernel (subfield (E := E) quarter (eqAt ((positionPoint S I.point).map Ext.into) x) *
            ((S.sourceBase : Rq S.innerDegree) ^ (u : ℕ) *
              (fold.responseBase : Rq S.innerDegree) ^ (f : ℕ))) (α.into : E) j *
            (S.responseCell fold ((x, u), f) j).elim 0 fun x => (eq x).into := by
  rw [responseTerm, termValue_into]
  simp only [responseReads, reads_finRange]
  simp only [List.map_append, List.map_map, List.sum_append, sum_finRange, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, add_zero, Function.comp_apply, kernel_reads,
    Ext.into_neg, neg_mul, Finset.sum_neg_distrib, sub_eq_add_neg]
  apply congrArg₂ (· + ·)
  · refine congrArg _ (Finset.sum_congr rfl fun a _ => congrArg _ (Finset.sum_congr rfl fun f _ =>
      Finset.sum_congr rfl fun j _ => ?_))
    rw [keyed, shared_flatOf, Nat.cast_pow]
  · refine congrArg _ (congrArg _ (Finset.sum_congr rfl fun f _ =>
      Finset.sum_congr rfl fun j _ => ?_))
    rw [contract_mul (E := E) quarter, eqIndex_into, Nat.cast_pow, Nat.cast_pow]

/-- The terms of `ẑ` are the rows (104) of `A` on `ẑ` and the contraction of (103), with the sign
of the code. -/
theorem responseTerms_value (quarter : 4 * (S.innerDegree / 4) = S.innerDegree) (key : Key S)
    (keyed : key.inner =
      Akita.Setup.shared (flatOf I.setup) S.innerDegree S.innerRank _ S.innerSetupCode)
    (subfield' : E →+* Rq S.innerDegree)
    (subfielded : subfield' = (subfield (E := E) quarter).toRingHom)
    (position : Fin S.blockLength → E)
    (positioned : ∀ x, position x = eqAt ((positionPoint S I.point).map Ext.into) x) :
    ((responseTerms (S := S) (fold := fold) I).map fun t => ((termValue α eq t).into : E)).sum =
      -(∑ a : Fin S.innerRank, ((I.coef (1 + (a : ℕ))).into : E) *
        formScan (responseForm fold key a) (α.into : E) (fun x => (eq x).into)) -
        ((I.coef 0).into : E) * formScan (traceContractForm fold subfield' position) (α.into : E)
          (fun x => (eq x).into) := by
  rw [responseTerms, sum_flatMap]
  simp only [List.map_map, sum_finRange, Function.comp_apply,
    responseTerm_value (E := E) I α eq quarter key keyed, formScan_response, formScan_contract,
    subfielded, positioned, AlgHom.toRingHom_eq_coe, RingHom.coe_coe]
  simp only [Finset.sum_sub_distrib, Finset.sum_neg_distrib, sum_rows_out, ← Finset.mul_sum]

/-- The terms of the partials are the left side of (103). -/
theorem partialTerms_value (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq S.innerDegree (I.challenge S.innerDegree i)) :
    ((partialTerms (O := O) (fold := fold) I).map fun t => ((termValue α eq t).into : E)).sum =
      ((I.coef 0).into : E) * formScan (tracePartialForm O fold challenge) (α.into : E)
        (fun x => (eq x).into) := by
  rw [partialTerms, List.map_map, sum_finRange, formScan_partial, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Function.comp_apply, partialTerm, termValue_into]
  simp only [reads_finRange, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    kernel_reads, challenged]
  simp only [Nat.cast_pow]

/-- The terms of the images are the left sides of the rows (104). -/
theorem imageTerms_value (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq S.innerDegree (I.challenge S.innerDegree i)) :
    ((imageTerms (O := O) (fold := fold) I).map fun t => ((termValue α eq t).into : E)).sum =
      ∑ a : Fin S.innerRank, ((I.coef (1 + (a : ℕ))).into : E) *
        formScan (traceImageForm O fold challenge a) (α.into : E) (fun x => (eq x).into) := by
  rw [imageTerms, sum_flatMap]
  simp only [List.map_map, sum_finRange, Function.comp_apply, formScan_image]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [imageTerm, termValue_into]
  simp only [reads_finRange, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    kernel_reads, challenged]
  simp only [Nat.cast_pow]

end Head

section Carried

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor} (I : TraceInputs) (α : Ext) (eq : ℕ → Ext) (α' : E) (φ : ℕ → E)

omit [Degree2 E] in
theorem formScan_outer (key : Key S) (j : Fin (S.slices * S.outerRank)) :
    formScan (outerForm O fold key j) α' φ =
      ∑ p : Fin S.sliceLength, ∑ a : Fin S.innerRank, ∑ y : Fin S.ratio,
        ∑ u : Fin S.openingDigits, ∑ k ∈ Finset.range S.outerDegree,
          residueKernel (key.outer (finProdFinEquiv.symm j).2 (p, (a, y, u))) α' k *
            (O.sliceCell fold (finProdFinEquiv.symm j).1 (p, (a, y, u)) k).elim 0 φ := by
  show ∑ c : Fin S.sliceLength × (Fin S.innerRank × Fin S.ratio × Fin S.openingDigits),
    ∑ k ∈ Finset.range S.outerDegree,
      residueKernel (key.outer (finProdFinEquiv.symm j).2 c) α' k *
        (O.sliceCell fold (finProdFinEquiv.symm j).1 c k).elim 0 φ = _
  simp only [Fintype.sum_prod_type]

omit [Degree2 E] in
theorem formScan_open (openKey : OpeningKey O) (j : Fin (1 * O.openRank)) :
    formScan (openForm O fold openKey j) α' φ =
      ∑ i : Fin S.blocks, ∑ y : Fin O.subcolumns, ∑ u : Fin S.openingDigits,
        ∑ k ∈ Finset.range O.openDegree,
          residueKernel (openKey.matrix (finProdFinEquiv.symm j).2 (i, y, u)) α' k *
            (O.partialCell fold (i, y, u) k).elim 0 φ := by
  show ∑ c : Fin S.blocks × Fin O.subcolumns × Fin S.openingDigits,
    ∑ k ∈ Finset.range O.openDegree,
      residueKernel (openKey.matrix (finProdFinEquiv.symm j).2 c) α' k *
        (O.partialCell fold c k).elim 0 φ = _
  simp only [Fintype.sum_prod_type]

/-- The terms of `B` are the rows (105) of `B` on the slices of `t̂`. -/
theorem outerTerms_value (key : Key S)
    (keyed : key.outer =
      Akita.Setup.shared (flatOf I.setup) S.outerDegree S.outerRank _ S.outerSetupCode) :
    ((outerTerms (O := O) (fold := fold) I).map fun t => ((termValue α eq t).into : E)).sum =
      ∑ j : Fin (S.slices * S.outerRank), ((I.coef (1 + S.innerRank + j)).into : E) *
        formScan (outerForm O fold key j) (α.into : E) (fun x => (eq x).into) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  simp only [formScan_outer, Equiv.symm_apply_apply, finProdFinEquiv_apply_val]
  rw [outerTerms, sum_flatMap]
  simp only [sum_flatMap, List.map_map, sum_finRange, Function.comp_apply]
  refine Finset.sum_congr rfl fun s _ => ?_
  have term : ∀ p a y u,
      ((termValue α eq (outerTerm (O := O) (fold := fold) I s p a y u)).into : E) =
        ∑ r : Fin S.outerRank, ((I.coef (1 + S.innerRank + (r + S.outerRank * s))).into : E) *
          ∑ k ∈ Finset.range S.outerDegree,
            residueKernel (key.outer r (p, (a, y, u))) (α.into : E) k *
              (O.sliceCell fold s (p, (a, y, u)) k).elim 0 fun x => (eq x).into := by
    intro p a y u
    rw [outerTerm, termValue_into]
    simp only [List.map_map, sum_finRange, Function.comp_apply, reads_one]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [keyed, shared_flatOf, show (s : ℕ) * S.outerRank + r = r + S.outerRank * s by ring]
  simp only [term, sum_rows_out]

/-- The terms of `D` are the rows (106) of `D` on `ê`. -/
theorem openTerms_value (openKey : OpeningKey O)
    (keyed : openKey.matrix =
      Akita.Setup.shared (flatOf I.setup) O.openDegree O.openRank _ O.openSetupCode) :
    ((openTerms (O := O) (fold := fold) I).map fun t => ((termValue α eq t).into : E)).sum =
      ∑ j : Fin (1 * O.openRank),
        ((I.coef (1 + S.innerRank + S.slices * S.outerRank + j)).into : E) *
          formScan (openForm O fold openKey j) (α.into : E) (fun x => (eq x).into) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type, Fin.sum_univ_one]
  simp only [formScan_open, Equiv.symm_apply_apply, finProdFinEquiv_apply_val, Fin.val_eq_zero,
    mul_zero, add_zero]
  rw [openTerms, sum_flatMap]
  simp only [sum_flatMap, List.map_map, sum_finRange, Function.comp_apply]
  have term : ∀ i y u,
      ((termValue α eq (openTerm (O := O) (fold := fold) I i y u)).into : E) =
        ∑ r : Fin O.openRank, ((I.coef (1 + S.innerRank + S.slices * S.outerRank + r)).into : E) *
          ∑ k ∈ Finset.range O.openDegree,
            residueKernel (openKey.matrix r (i, y, u)) (α.into : E) k *
              (O.partialCell fold (i, y, u) k).elim 0 fun x => (eq x).into := by
    intro i y u
    rw [openTerm, termValue_into]
    simp only [List.map_map, sum_finRange, Function.comp_apply, reads_one]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [keyed, shared_flatOf]
  simp only [term, sum_rows_out]

theorem sum_range_list {M : Type} [AddCommMonoid M] (m : ℕ) (h : ℕ → M) :
    ((List.range m).map h).sum = ∑ i : Fin m, h i := by
  rw [← Finset.sum_range]
  induction m with
  | zero => simp
  | succ m ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ,
      List.map_singleton, List.sum_singleton]

theorem sum_list_range {M : Type} [AddCommMonoid M] (m : ℕ) (h : ℕ → M) :
    ((List.range m).map h).sum = ∑ i ∈ Finset.range m, h i := by
  rw [sum_range_list, Finset.sum_range]

omit [Degree2 E] in
theorem reads_range {m : ℕ} (s : ℕ → ℕ) (addr : ℕ → ℕ → Option ℕ) (j : ℕ) :
    ((((List.range m).map fun u => (natFq (s u), addr u)) : List Read).map fun r =>
      algebraMap Zq E r.1.toZq * (r.2 j).elim 0 φ).sum =
        ∑ u : Fin m, (s u : E) * (addr u j).elim 0 φ := by
  rw [List.map_map, sum_range_list]
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [Function.comp, natFq, Fq.toZq_ofNat, map_natCast]

/-- The recomposition `G·ξ̂` of entry `j` of an image of `n` elements of degree `d` from the
digits at `cell`, for a path with compression maps, and 0 for a path without. -/
noncomputable def recomposeScan (d : ℕ) [NeZero d] (chain : Option ChainShape) (n : ℕ)
    (cell : ℕ → Option ℕ) (j : ℕ) : E :=
  if chain.isSome then
    ∑ u : Fin binaryDigits, ∑ k ∈ Finset.range d,
      residueKernel (((2 : ℕ) : Rq d) ^ (u : ℕ)) α' k * (cell (j * d + k + n * d * u)).elim 0 φ
  else 0

omit [Degree2 E] in
theorem rowScan_imageRow {d n : ℕ} [NeZero d] (chain : Option ChainShape)
    (image : Fin n → Akita.Form d) (cell : ℕ → Option ℕ) (payload : Payload d n chain)
    (j : Fin n) :
    rowScan (imageRow chain image cell payload j) α' φ =
      formScan (image j) α' φ - recomposeScan α' φ d chain n cell j := by
  cases chain with
  | none => rw [imageRow, rowScan_row, recomposeScan, Option.isSome_none, ite_eq_right (by decide),
      sub_zero]
  | some c =>
      rw [imageRow, rowScan_row, formScan_sub, recomposeScan, Option.isSome_some, ite_eq_left rfl]
      rfl

/-- The recomposition terms of a path are `−ẽq(τ_1, row)·G·ξ̂` of each image row. -/
theorem recomposeTerms_value {d : ℕ} [NeZero d] (n : ℕ) (cell : ℕ → Option ℕ) (base : ℕ)
    (coef : ℕ → Ext) :
    (((List.range n).map fun j =>
      ({ degree := d, reads := recomposeReads d n cell j,
         muls := [(-coef (base + j), unitRing d)] } : Term)).map fun t =>
        ((termValue α eq t).into : E)).sum =
      -∑ j : Fin n, ((coef (base + j)).into : E) * ∑ u : Fin binaryDigits,
        ∑ k ∈ Finset.range d, residueKernel (((2 : ℕ) : Rq d) ^ (u : ℕ)) (α.into : E) k *
          (cell (j * d + k + n * d * u)).elim 0 fun x => (eq x).into := by
  rw [List.map_map, sum_range_list, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Function.comp_apply, termValue_into]
  simp only [recomposeReads, reads_range, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero, kernel_reads, toRq_unitRing, one_mul, Ext.into_neg, neg_mul]
  simp only [Nat.cast_pow]

/-- The column terms of a map `K` that is a view of the setup are the rows `K·x` with the
coefficients `ẽq(τ_1, row)`. -/
theorem columnTerms_value {d : ℕ} [NeZero d] (setup : Array Fq) (rank width : ℕ)
    (cell : ℕ → Option ℕ) (base : ℕ) (coef : ℕ → Ext) (K : Matrix (Fin rank) (Fin width) (Rq d))
    (keyed : K = Akita.Setup.view ⟨d, rank, width, false, 1⟩ (flatOf setup)) :
    (((List.range width).map fun col =>
      ({ degree := d, reads := [(Fq.one, fun k => cell (col * d + k))],
         muls := (List.range rank).map fun r =>
           (coef (base + r), viewEntry setup d width r col) } : Term)).map fun t =>
        ((termValue α eq t).into : E)).sum =
      ∑ r : Fin rank, ((coef (base + r)).into : E) *
        formScan (Form.ofMatrix K (fun col k => cell (col * d + k)) r) (α.into : E)
          (fun x => (eq x).into) := by
  rw [List.map_map, sum_range_list]
  have term : ∀ col : Fin width, ((termValue α eq
      ({ degree := d, reads := [(Fq.one, fun k => cell (col * d + k))],
         muls := (List.range rank).map fun r =>
           (coef (base + r), viewEntry setup d width r col) } : Term)).into : E) =
      ∑ r : Fin rank, ((coef (base + r)).into : E) * ∑ k ∈ Finset.range d,
        residueKernel (K r col) (α.into : E) k * (cell (col * d + k)).elim 0 fun x => (eq x).into := by
    intro col
    rw [termValue_into]
    simp only [List.map_map, sum_range_list, Function.comp_apply, reads_one]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [keyed, view_flatOf]
  simp only [Function.comp_apply, term, sum_rows_out]
  rfl

/-- The terms of a compression path (Akita (47)): the recomposition in the image rows, and the
rows of `F_1` and `F_2` with the coefficients `ẽq(τ_1, row)`. -/
theorem chainTerms_value {d n : ℕ} [NeZero d] (setup : Array Fq) (coef : ℕ → Ext)
    (chain : Option ChainShape) (key : PathKey chain) (shared : PathShared (flatOf setup) chain key)
    (firstCell secondCell : ℕ → Option ℕ) (payload : Payload d n chain)
    (image first second : ℕ) :
    (List.map List.sum (List.map (List.map fun t => ((termValue α eq t).into : E))
      (chainTerms setup coef chain d n firstCell secondCell image first second))).sum =
      -(∑ j : Fin n, ((coef (image + j)).into : E) *
        recomposeScan (α.into : E) (fun x => (eq x).into) d chain n firstCell j) +
      (∑ r : Fin (firstRank chain), ((coef (first + r)).into : E) *
        rowScan (firstRow chain key firstCell secondCell r) (α.into : E) (fun x => (eq x).into) +
      ∑ r : Fin (secondRank chain), ((coef (second + r)).into : E) *
        rowScan (secondRow chain key secondCell payload r) (α.into : E) (fun x => (eq x).into)) := by
  cases chain with
  | none =>
      simp only [chainTerms, List.map_nil, List.sum_nil, recomposeScan, Option.isSome_none,
        Bool.false_eq_true, ite_false, mul_zero, Finset.sum_const_zero, neg_zero, zero_add]
      rw [Finset.sum_eq_zero fun r _ => r.elim0, Finset.sum_eq_zero fun r _ => r.elim0, add_zero]
  | some c =>
      obtain ⟨firstShared, secondShared⟩ := shared
      simp only [chainTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
      rw [recomposeTerms_value, columnTerms_value α eq setup _ _ _ _ _ _ firstShared,
        recomposeTerms_value, columnTerms_value α eq setup _ _ _ _ _ _ secondShared]
      simp only [recomposeScan, Option.isSome_some, ite_true, firstRow, secondRow, firstRank,
        secondRank, rowScan_row, formScan_sub, mul_sub, Finset.sum_sub_distrib]
      have recomposed : ∀ r : Fin c.firstRank, formScan (Form.recompose c.firstDegree c.firstRank
          secondCell r) (α.into : E) (fun x => (eq x).into) =
          ∑ u : Fin binaryDigits, ∑ k ∈ Finset.range c.firstDegree,
            residueKernel (((2 : ℕ) : Rq c.firstDegree) ^ (u : ℕ)) (α.into : E) k *
              (secondCell (r * c.firstDegree + k + c.firstRank * c.firstDegree * u)).elim 0
                fun x => (eq x).into := fun _ => rfl
      simp only [recomposed]
      abel

end Carried

section Assemble

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor} (α' : E) (φ : ℕ → E)

omit [Degree2 E] in
/-- The ring rows of a trace fold: the row (103), the rows (104) and the carried rows. -/
theorem traceRows_scan (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (c : ℕ → E) :
    ∑ k : Fin O.traceRowCount, c k * rowScan (traceRows O fold st challenge k) α' φ =
      c 0 * (formScan (tracePartialForm O fold challenge) α' φ -
        formScan (traceContractForm fold st.subfield st.position) α' φ) +
      (∑ a : Fin S.innerRank, c (1 + a) * (formScan (traceImageForm O fold challenge a) α' φ -
        formScan (responseForm fold st.key a) α' φ) +
      ∑ k : Fin O.carriedCount, c (1 + (S.innerRank + k)) *
        rowScan (carriedRows O fold st.key st.openKey st.payload st.openPayload k) α' φ) := by
  show ∑ k : Fin (1 + (S.innerRank + O.carriedCount)), c k * rowScan
    (Fin.append (fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
      (Fin.append (traceInnerRow O fold st.key challenge)
        (carriedRows O fold st.key st.openKey st.payload st.openPayload)) k) α' φ = _
  rw [Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_one]
  simp only [Fin.append_left, Fin.append_right, Fin.val_castAdd, Fin.val_natAdd, Fin.val_eq_zero,
    traceEvaluationRow, traceInnerRow, rowScan_row, formScan_sub]

omit [Degree2 E] in
/-- The carried rows of a fold: the rows (105), (106) and (47). -/
theorem carriedRows_scan (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) (c : ℕ → E) :
    ∑ k : Fin O.carriedCount, c k *
        rowScan (carriedRows O fold key openKey payload openPayload k) α' φ =
      ∑ j : Fin (S.slices * S.outerRank), c j *
        rowScan (imageRow S.chain (outerForm O fold key) (O.firstCell fold) payload j) α' φ +
      (∑ j : Fin (1 * O.openRank), c (S.slices * S.outerRank + j) *
        rowScan (imageRow O.chain (openForm O fold openKey) (O.openFirstCell fold) openPayload j)
          α' φ +
      (∑ r : Fin (firstRank S.chain), c (S.slices * S.outerRank + (1 * O.openRank + r)) *
        rowScan (firstRow S.chain key.chain (O.firstCell fold) (O.secondCell fold) r) α' φ +
      (∑ r : Fin (firstRank O.chain),
        c (S.slices * S.outerRank + (1 * O.openRank + (firstRank S.chain + r))) *
        rowScan (firstRow O.chain openKey.chain (O.openFirstCell fold) (O.openSecondCell fold) r)
          α' φ +
      (∑ r : Fin (secondRank S.chain), c (S.slices * S.outerRank + (1 * O.openRank +
          (firstRank S.chain + (firstRank O.chain + r)))) *
        rowScan (secondRow S.chain key.chain (O.secondCell fold) payload r) α' φ +
      ∑ r : Fin (secondRank O.chain), c (S.slices * S.outerRank + (1 * O.openRank +
          (firstRank S.chain + (firstRank O.chain + (secondRank S.chain + r))))) *
        rowScan (secondRow O.chain openKey.chain (O.openSecondCell fold) openPayload r) α' φ)))) := by
  show ∑ k : Fin (S.slices * S.outerRank + (1 * O.openRank + (firstRank S.chain +
    (firstRank O.chain + (secondRank S.chain + secondRank O.chain))))),
      c k * rowScan (carriedRows O fold key openKey payload openPayload k) α' φ = _
  rw [Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add, Fin.sum_univ_add]
  simp only [carriedRows, Fin.append_left, Fin.append_right, Fin.val_castAdd, Fin.val_natAdd]

/-- The scalar row (101) at `r_2` with its coefficient. -/
theorem traceScalar_into (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (I : TraceInputs) (eq : ℕ → Ext) :
    ((traceScalar (O := O) (fold := fold) I eq).into : E) =
      ((I.coef O.traceRowCount).into : E) *
        ∑ entry : Fin S.blocks × Fin S.innerDegree × Fin S.openingDigits,
          (O.partialPosition fold entry.1 entry.2.2 entry.2.1).elim 0 (fun x => (eq x).into) *
            (I.theta.into * (eqAt ((blockPoint S I.point).map Ext.into) entry.1 *
              ((S.openingBase : Zq) ^ (entry.2.2 : ℕ) •
                packFunctional quarter (eqAt ((packPoint S I.point).map Ext.into))
                  (AdjoinRoot.root (modulus S.innerDegree) ^ (entry.2.1 : ℕ))))) := by
  have entry : ∀ (i : Fin S.blocks) (u : Fin S.openingDigits) (j : Fin S.innerDegree),
      (((O.partialPosition fold i u j).elim Ext.zero fun x =>
        (Array.ofFn (n := S.innerDegree) fun j =>
          functional (S.innerDegree / 4) (packPoint S I.point) (monomial S.innerDegree j.val)).getD
            j Ext.zero * eq x).into : E) =
        (O.partialPosition fold i u j).elim 0 (fun x => (eq x).into) *
          packFunctional quarter (eqAt ((packPoint S I.point).map Ext.into))
            (AdjoinRoot.root (modulus S.innerDegree) ^ (j : ℕ)) := by
    intro i u j
    have got : (Array.ofFn (n := S.innerDegree) fun j =>
        functional (S.innerDegree / 4) (packPoint S I.point) (monomial S.innerDegree j.val)).getD
          j Ext.zero = functional (S.innerDegree / 4) (packPoint S I.point)
            (monomial S.innerDegree j) := by simp [Array.getD, j.isLt]
    cases O.partialPosition fold i u j with
    | none => simp [Ext.into_zero]
    | some x =>
        simp only [Option.elim_some]
        rw [Ext.into_mul, got, functional_into quarter two, toRq_monomial j.isLt, mul_comm]
  simp only [traceScalar, TraceInputs.coef, Ext.into_mul, sumList_into, sum_finRange, eqIndex_into,
    Ext.into_scale, sum_range_list, entry, natFq, Fq.toZq_ofNat, Nat.cast_pow, map_pow,
    map_natCast]
  congr 1
  rw [Finset.mul_sum, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fintype.sum_prod_type, Finset.sum_comm, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Algebra.smul_def, RingHom.map_pow, map_natCast]
  ring

/-- The weight theorem of a trace level: the terms `traceTerms` and the scalar row `traceScalar`
at `r_2` are `m_{τ_1}(r_2)` of `traceReduction` (Akita (123) at `r_2`), for the statement whose
matrices are the views of the setup array (`Setup.shared`), with the subfield, the weights `χ_blk`
and `a`, the functional `T_{ρ_pack}` and the factor `θ` of the opening point and the fold
challenges `c_i` of the inputs. -/
theorem traceWeight_into [Fintype E] (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (I : TraceInputs) (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq S.innerDegree (I.challenge S.innerDegree i))
    (keyed : st.key.Shared (flatOf I.setup)) (opened : st.openKey.Shared (flatOf I.setup))
    (subfielded : st.subfield = (subfield (E := E) quarter).toRingHom)
    (positioned : ∀ x, st.position x = eqAt ((positionPoint S I.point).map Ext.into) x)
    (blocked : ∀ i, st.block i = eqAt ((blockPoint S I.point).map Ext.into) i)
    (functioned : st.functional =
      packFunctional quarter (eqAt ((packPoint S I.point).map Ext.into)))
    (scaled : st.scale = I.theta.into) {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (α : Ext) (r2 : List Ext)
    (length : r2.length = μ) (tau : I.tau1.length = batch) :
    ((termsValue α (CellEq.half r2).get (traceTerms (O := O) (fold := fold) I) +
        traceScalar (O := O) (fold := fold) I (CellEq.half r2).get).into : E) =
      mle (r2.map Ext.into) (((traceReduction st challenge μ fits batch rowsFit).rows
        α.into).batchWeight (I.tau1.map Ext.into)) := by
  have same : ∀ x < O.cells fold, ((((CellEq.half r2).get x).into : E)) =
      eqAt (r2.map Ext.into) x := fun x small =>
    CellEq.get_half r2 (by rw [length]; exact small.trans_le fits)
  rw [traceReduction_weight st challenge fits rowsFit α.into (by rw [List.length_map, length])
    (by rw [List.length_map, tau]) (fun x => ((CellEq.half r2).get x).into) same,
    traceRows_scan _ _ st challenge (eqAt (I.tau1.map Ext.into)),
    carriedRows_scan _ _ st.key st.openKey st.payload st.openPayload
      (fun k => eqAt (I.tau1.map Ext.into) (1 + (S.innerRank + k)))]
  obtain ⟨inner, outer, path⟩ := keyed
  obtain ⟨matrix, openPath⟩ := opened
  rw [Ext.into_add, termsValue, sumList_into, traceTerms, List.map_flatten, List.sum_flatten,
    traceParts]
  simp only [List.map_append, List.map_cons, List.map_nil, List.sum_append, List.sum_cons,
    List.sum_nil, add_zero]
  rw [responseTerms_value I α _ quarter st.key inner st.subfield subfielded st.position positioned,
    partialTerms_value I α _ challenge challenged, imageTerms_value I α _ challenge challenged,
    outerTerms_value I α _ st.key outer, openTerms_value I α _ st.openKey matrix,
    chainTerms_value α _ I.setup I.coef S.chain st.key.chain path _ _ st.payload,
    chainTerms_value α _ I.setup I.coef O.chain st.openKey.chain openPath _ _ st.openPayload,
    traceScalar_into quarter two I]
  simp only [rowScan_imageRow, scaled, blocked, functioned, TraceInputs.coef, eqIndex_into,
    mul_sub, Finset.sum_sub_distrib, Nat.add_assoc, one_mul]
  abel

end Assemble

section End

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor}

theorem responseCode_source (c : Fin (S.blockLength * S.sourceDigits))
    (a : Fin fold.responseDigits) :
    (S.responseCode fold (S.sourceCode.symm c, a) : ℕ) = c * fold.responseDigits + a := by
  simp only [Shape.responseCode, Shape.sourceCode, Equiv.trans_apply, Equiv.prodCongr_apply,
    Prod.map_apply, Equiv.apply_symm_apply, Equiv.refl_apply, finProdFinEquiv_apply_val]
  ring

theorem normKernel_apply {c k : ℕ} (inside : c < S.blockLength * S.sourceDigits)
    (small : k < S.innerDegree) (x : ℕ) :
    S.normKernel fold (c * S.innerDegree + k) x = ∑ a ∈ Finset.range fold.responseDigits,
      if x = (c * fold.responseDigits + a) * S.innerDegree + k then
        (fold.responseBase : Zq) ^ a else 0 := by
  unfold Shape.normKernel
  rw [dite_eq_left (by rw [Akita.Setup.index_div small]; exact inside), Finset.sum_range]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [responseCode_source, Akita.Setup.index_mod small]
  simp only [Akita.Setup.index_div small]

omit [Degree2 E] in
theorem normKernel_zero {u : ℕ} (large : S.blockLength * S.sourceDigits * S.innerDegree ≤ u)
    (x : ℕ) : S.normKernel fold u x = 0 := by
  unfold Shape.normKernel
  rw [dite_eq_right fun inside => absurd ((Nat.div_lt_iff_lt_mul (NeZero.pos _)).mp inside)
    (Nat.not_lt.mpr large)]

/-- `normValue` is `K_{r_virt}` at `r_2`: the norm weight of `foldCheckSq` at a trace level with
the kernel `Shape.normKernel`. -/
theorem normValue_into {μ : ℕ} (cellsRoom : S.responseCells fold ≤ 2 ^ μ)
    (digits : 0 < fold.responseDigits) (virt eq : ℕ → Ext) {r r2 : List E}
    (length2 : r2.length = μ) (virtual : ∀ u < 2 ^ μ, ((virt u).into : E) = eqAt r u)
    (same : ∀ x < 2 ^ μ, ((eq x).into : E) = eqAt r2 x) :
    ((normValue S fold virt eq).into : E) = mle r2 (normWeight μ (S.normKernel fold) r) := by
  have sourceRoom : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ μ :=
    le_trans (Nat.mul_le_mul_right _ (Nat.le_mul_of_pos_right _ digits)) cellsRoom
  have zero : ∀ u ∈ Finset.Ico (S.blockLength * S.sourceDigits * S.innerDegree) (2 ^ μ),
      ∑ x ∈ Finset.range (2 ^ μ), eqAt r2 x * (eqAt r u *
        algebraMap Zq E (S.normKernel fold u x)) = 0 := fun u member =>
    Finset.sum_eq_zero fun x _ => by
      rw [normKernel_zero (Finset.mem_Ico.mp member).1, map_zero, mul_zero, mul_zero]
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length2]
  simp only [normWeight, Finset.mul_sum]
  rw [Finset.sum_comm, ← Finset.sum_range_add_sum_Ico _ sourceRoom, Finset.sum_eq_zero zero,
    add_zero, ← sum_range_mul _ S.innerDegree]
  simp only [normValue, sumList_into, sum_list_range, Ext.into_mul, Ext.into_scale, natFq,
    Fq.toZq_ofNat, map_natCast]
  refine Finset.sum_congr rfl fun c member => Finset.sum_congr rfl fun k inside => ?_
  have c_lt := Finset.mem_range.mp member
  have k_lt := Finset.mem_range.mp inside
  have cell : c * S.innerDegree + k < S.blockLength * S.sourceDigits * S.innerDegree :=
    cell_lt (⟨c, c_lt⟩ : Fin (S.blockLength * S.sourceDigits)) k_lt
  have read : ∀ a < fold.responseDigits,
      (c * fold.responseDigits + a) * S.innerDegree + k < 2 ^ μ := fun a a_lt => by
    refine lt_of_lt_of_le ?_ cellsRoom
    unfold Shape.responseCells
    have row : c * fold.responseDigits + a < S.blockLength * S.sourceDigits * fold.responseDigits :=
      cell_lt (⟨c, c_lt⟩ : Fin (S.blockLength * S.sourceDigits)) a_lt
    exact cell_lt (⟨_, row⟩ : Fin (S.blockLength * S.sourceDigits * fold.responseDigits)) k_lt
  rw [virtual _ (cell.trans_le sourceRoom)]
  simp only [normKernel_apply c_lt k_lt]
  have inner : ∀ x, eqAt r2 x * (eqAt r (c * S.innerDegree + k) * algebraMap Zq E
      (∑ a ∈ Finset.range fold.responseDigits,
        if x = (c * fold.responseDigits + a) * S.innerDegree + k then
          (fold.responseBase : Zq) ^ a else 0)) =
      eqAt r (c * S.innerDegree + k) * ∑ a ∈ Finset.range fold.responseDigits,
        if x = (c * fold.responseDigits + a) * S.innerDegree + k then
          ((fold.responseBase ^ a : ℕ) : E) * eqAt r2 x else 0 := fun x => by
    rw [map_sum, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    split
    · rw [map_pow, map_natCast, Nat.cast_pow]
      ring
    · simp
  rw [Finset.sum_congr rfl fun x _ => inner x, ← Finset.mul_sum, Finset.sum_comm]
  congr 1
  refine Finset.sum_congr rfl fun a member' => ?_
  have a_lt := Finset.mem_range.mp member'
  rw [Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.mpr (read a a_lt)), same _ (read a a_lt)]

/-- `supportValue` is `1_{I_bin}·ẽq(r_virt, ·)` at `r_2`, the binary-support weight of
`fusedCheck`. -/
theorem supportValue_into {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) (virt eq : ℕ → Ext)
    {r r2 : List E} (length2 : r2.length = μ)
    (virtual : ∀ u < 2 ^ μ, ((virt u).into : E) = eqAt r u)
    (same : ∀ x < 2 ^ μ, ((eq x).into : E) = eqAt r2 x) :
    ((supportValue O fold virt eq).into : E) =
      mle r2 fun x => if O.binarySupport fold x then eqAt r x else 0 := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length2]
  simp only [mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  have support : (Finset.range (2 ^ μ)).filter (O.binarySupport fold) =
      Finset.Ico (O.chainStart fold) (O.cells fold) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico, OpeningShape.binarySupport]
    omega
  rw [support, Finset.sum_Ico_eq_sum_range, supportValue, sumList_into, sum_list_range]
  refine Finset.sum_congr rfl fun k member => ?_
  have inside : O.chainStart fold + k < 2 ^ μ := by
    have := Finset.mem_range.mp member
    have := O.chainStart_le_cells fold
    omega
  rw [Ext.into_mul, virtual _ inside, same _ inside, mul_comm]

section Projections

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor) (setup : Array Fq)
  (challenges : List (Array Fq)) (V : TraceValues)

theorem traceFused_weight : (traceFused O fold setup challenges V).weight =
    termsValue V.alpha (CellEq.half V.r2).get (traceTerms (O := O) (fold := fold)
      ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩) +
    traceScalar (O := O) (fold := fold)
      ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩
        (CellEq.half V.r2).get := rfl

theorem traceFused_normWeight : (traceFused O fold setup challenges V).normWeight =
    if V.euclidean then normValue S fold (CellEq.half V.virtual).get (CellEq.half V.r2).get
    else Ext.zero := rfl

theorem traceFused_support : (traceFused O fold setup challenges V).support =
    if V.compressed then supportValue O fold (CellEq.half V.virtual).get (CellEq.half V.r2).get
    else Ext.zero := rfl

theorem traceFused_virt : (traceFused O fold setup challenges V).virt =
    eqPoints V.virtual V.r2 := rfl

theorem traceFused_fields : (traceFused O fold setup challenges V).output = V.output ∧
    (traceFused O fold setup challenges V).value = V.value ∧
    (traceFused O fold setup challenges V).batch = V.batch ∧
    (traceFused O fold setup challenges V).binary =
      (if V.compressed then V.binary else Ext.zero) ∧
    (traceFused O fold setup challenges V).norm = (if V.euclidean then V.norm else Ext.zero) :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

end Projections

/-- The end check `fused.end` of a trace level is the final check of the fused sum-check of the
specification (`fusedCheck`, with the weight `m_{τ_1} + η·K_{r_virt}` of `foldCheckSq` at a
Euclidean level) for `traceReduction` of the statement whose matrices are the views of the setup
array, with no hypothesis on the weights. -/
theorem traceFused_holds_iff [Fintype E] (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree)))
    (keyed : st.key.Shared (flatOf setup)) (opened : st.openKey.Shared (flatOf setup))
    (subfielded : st.subfield = (subfield (E := E) quarter).toRingHom)
    (positioned : ∀ x, st.position x = eqAt ((positionPoint S V.point).map Ext.into) x)
    (blocked : ∀ i, st.block i = eqAt ((blockPoint S V.point).map Ext.into) i)
    (functioned : st.functional =
      packFunctional quarter (eqAt ((packPoint S V.point).map Ext.into)))
    (scaled : st.scale = (tensorFactor V.tail V.eta V.point).into) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (digits : 0 < fold.responseDigits) (length : V.r2.length = μ)
    (virtualLength : V.virtual.length = μ) (tau : V.tau1.length = batch)
    (compressed : V.compressed = O.compressed fold) :
    (traceFused O fold setup challenges V).holds = true ↔
      (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into) (fun x =>
        ((traceReduction st challenge μ fits batch rowsFit).rows V.alpha.into).batchWeight
          (V.tau1.map Ext.into) x + (if V.euclidean then (V.norm.into : E) else 0) *
            normWeight μ (S.normKernel fold) (V.virtual.map Ext.into) x) +
        mle (V.r2.map Ext.into) (virtWeight (O.binarySupport fold) (V.virtual.map Ext.into)
          V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1)) := by
  have same : ∀ x < 2 ^ μ, ((((CellEq.half V.r2).get x).into : E)) =
      eqAt (V.r2.map Ext.into) x := fun x small => CellEq.get_half V.r2 (by rw [length]; exact small)
  have virtual : ∀ x < 2 ^ μ, ((((CellEq.half V.virtual).get x).into : E)) =
      eqAt (V.virtual.map Ext.into) x := fun x small =>
    CellEq.get_half V.virtual (by rw [virtualLength]; exact small)
  have weighted := traceWeight_into (E := E) quarter two
    ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩ st challenge challenged
    keyed opened subfielded positioned blocked functioned scaled fits rowsFit V.alpha V.r2 length tau
  have cellsRoom : S.responseCells fold ≤ 2 ^ μ := (O.responseCells_le_cells fold).trans fits
  have weight : (((traceFused O fold setup challenges V).weight.into : E)) =
      mle (V.r2.map Ext.into) (((traceReduction st challenge μ fits batch rowsFit).rows
        V.alpha.into).batchWeight (V.tau1.map Ext.into)) := by
    rw [traceFused_weight]
    exact weighted
  have normed : (((traceFused O fold setup challenges V).normWeight.into : E)) =
      mle (V.r2.map Ext.into) (if V.euclidean then normWeight μ (S.normKernel fold)
        (V.virtual.map Ext.into) else fun _ => 0) := by
    rw [traceFused_normWeight]
    by_cases euclidean : V.euclidean
    · rw [ite_eq_left euclidean, ite_eq_left euclidean]
      exact normValue_into cellsRoom digits _ _ (by rw [List.length_map, length]) virtual same
    · rw [ite_eq_right euclidean, ite_eq_right euclidean, Ext.into_zero]
      exact (mle_const _ _).symm
  have virt : (((traceFused O fold setup challenges V).virt.into : E)) =
      mle (V.r2.map Ext.into) (eqAt (V.virtual.map Ext.into)) := by
    rw [traceFused_virt, eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map,
      length, virtualLength])]
  have supported : (((traceFused O fold setup challenges V).support.into : E)) =
      mle (V.r2.map Ext.into) fun x =>
        if O.binarySupport fold x then eqAt (V.virtual.map Ext.into) x else 0 := by
    rw [traceFused_support]
    by_cases binary : V.compressed
    · rw [ite_eq_left binary]
      exact supportValue_into fits _ _ (by rw [List.length_map, length]) virtual same
    · have empty : O.compressed fold = false := by rw [← compressed]; simpa using binary
      rw [ite_eq_right binary, Ext.into_zero, show (fun x => if O.binarySupport fold x then
        eqAt (V.virtual.map Ext.into) x else 0) = fun _ => (0 : E) from
          funext fun x => ite_eq_right (O.binarySupport_empty fold empty x), mle_const]
  obtain ⟨output, value, batched, binaried, normed'⟩ := traceFused_fields O fold setup challenges V
  rw [FusedEnd.holds_iff_fused _ (O.binarySupport fold) _ _ (V.virtual.map Ext.into)
    (V.r2.map Ext.into) weight normed virt supported, output, value, batched, binaried, normed']
  by_cases euclidean : V.euclidean <;> by_cases binary : V.compressed <;>
    simp only [euclidean, binary, Bool.false_eq_true, ite_true, ite_false, Ext.into_zero,
      zero_mul, add_zero]

end End

section Instance

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
  {fold : Descriptor}

variable (O) in
/-- The statement of a trace level for a setup array and the values of the replay: the views of
the setup as the matrices, the payloads, and the subfield, the weights `χ_blk` and `a`, the
functional `T_{ρ_pack}` and the factor `θ` of the opening point. -/
noncomputable def traceStatementOf (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (V : TraceValues) (payload : S.Payload) (openPayload : O.Payload)
    (value : E) : TraceStatement O E :=
  ⟨Key.ofSetup (flatOf setup) S, OpeningKey.ofSetup (flatOf setup) O, payload, openPayload,
    (subfield (E := E) quarter).toRingHom, fun i => eqAt ((blockPoint S V.point).map Ext.into) i,
    fun x => eqAt ((positionPoint S V.point).map Ext.into) x,
    packFunctional quarter (eqAt ((packPoint S V.point).map Ext.into)),
    (tensorFactor V.tail V.eta V.point).into, value⟩

/-- `traceFused_holds_iff` for the statement `traceStatementOf`: each hypothesis on the statement
holds. -/
theorem traceFused_holds_iff_of [Fintype E] (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (payload : S.Payload)
    (openPayload : O.Payload) (value : E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (digits : 0 < fold.responseDigits) (length : V.r2.length = μ)
    (virtualLength : V.virtual.length = μ) (tau : V.tau1.length = batch)
    (compressed : V.compressed = O.compressed fold) :
    (traceFused O fold setup challenges V).holds = true ↔
      (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into) (fun x =>
        ((traceReduction (traceStatementOf O quarter setup V payload openPayload value) challenge
          μ fits batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into) x +
          (if V.euclidean then (V.norm.into : E) else 0) *
            normWeight μ (S.normKernel fold) (V.virtual.map Ext.into) x) +
        mle (V.r2.map Ext.into) (virtWeight (O.binarySupport fold) (V.virtual.map Ext.into)
          V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1)) :=
  traceFused_holds_iff quarter two setup challenges V _ challenge challenged
    (Key.ofSetup_shared _ S) (OpeningKey.ofSetup_shared _ O) rfl (fun _ => rfl) (fun _ => rfl) rfl
    rfl fits rowsFit digits length virtualLength tau compressed

end Instance

end NightstreamFPrime.Spec.Compression.Akita.Replay
