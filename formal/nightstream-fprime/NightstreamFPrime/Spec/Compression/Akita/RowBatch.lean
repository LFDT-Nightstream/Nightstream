import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Akita.RingCheck

/-! Row batching of the relation of one fold of Akita (ePrint 2026/1983, revision of 2026-11-23,
Sections 7.2 to 7.4, Theorem 7.4 and Theorem 10.36).

- `FieldRows` is the field relation `M(α)·w = y(α)` of Akita Section 7.2 at one point `α`: for
  each row its weights on the flat witness and its target. `FieldRows.residual` is the residual of
  one row and `FieldRows.Holds` states that each residual is zero.
- `FieldRows.batchWeight` and `FieldRows.batchTarget` are `m_{τ_1}` and `V_{α,τ_1}` of Akita
  (123), and `FieldRows.Batched` is the batched claim (121). `FieldRows.batch_residual` states
  that the batched claim is the multilinear batch of the residuals.
- `FieldRows.batched_of_holds` is the batching part of Akita Theorem 7.4, item 1.
  `FieldRows.batchBad` is the bad set of the batching point `τ_1`, `FieldRows.holds_of_batched`
  gives each row from the batched claim at a point outside this set, and
  `FieldRows.density_batchBad_le` is Akita Theorem 7.4, item 3.
- `Reduction` is a ring reduction of a relation on the flat witness: the field rows at each point
  `α`, a bad set of points `α` for each witness, and the two directions between the relation and
  its field rows. `low` is the count of the last coordinates of the range point `τ_0` that the
  pinned code puts first (`rangePoint` of `FoldCheck.lean`). `Reduction.batched_of_holds`, `Reduction.holds_of_batched` and
  `Reduction.density_accepting_le` are Akita Theorem 7.4, item 1 and the bound (122).
- `RingRow.relocate` is a ring row with each address moved by a map.
- `groupOrder` is the order of the groups in the rows of the pinned code, the last group first,
  and `place` is the group and the index of each row of one family in this order.
- `Join` joins ring rows over `Z_q` into field rows: the planes of the carrier row (102) are one
  row over `E`, with the basis `β_t` as coefficients (Akita Section 7.3, after (109), and (113)).
  `Join.poly` is the polynomial over `E` of one field row, and `Join.poly_ne_zero` states that it
  is not zero when one of its planes is not zero, for coefficients that are independent over
  `Z_q`.
- `FlatRelation` has the ring rows (110) and the scalar row of Akita Figure 5 on the flat
  witness. `FlatRelation.free` is its quotient-free reduction (Akita (120)), with the bad set of
  Akita Theorem 7.4, item 2. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib Polynomial

/-- The field relation `M(α)·w = y(α)` of Akita Section 7.2 at one point `α`: row `i` has the
weight `weight i u` on the entry `u` of the flat witness and the target `target i`. -/
structure FieldRows (E : Type) where
  count : ℕ
  weight : Fin count → ℕ → E
  target : Fin count → E

namespace FieldRows

variable {E : Type} [Field E] [Algebra Zq E] (rows : FieldRows E) (μ : ℕ)

/-- The weight of row `i` on the entry `u`, and zero for an index that is not a row. -/
def weightAt (i u : ℕ) : E := if h : i < rows.count then rows.weight ⟨i, h⟩ u else 0

/-- The target of row `i`, and zero for an index that is not a row. -/
def targetAt (i : ℕ) : E := if h : i < rows.count then rows.target ⟨i, h⟩ else 0

/-- The residual `Σ_u w_u·M(α)_{i,u} − y(α)_i` of row `i` for the witness `w` on the cube of `μ`
variables (Akita Section 7.2). -/
def residual (w : Table) (i : ℕ) : E :=
  ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * rows.weightAt i u - rows.targetAt i

/-- Each row of the field relation holds for the witness `w`. -/
def Holds (w : Table) : Prop := ∀ i, rows.residual μ w i = 0

/-- `m_{τ_1}(u) = Σ_i ẽq(τ_1, i)·M(α)_{i,u}` of Akita (123). -/
def batchWeight (τ : List E) (u : ℕ) : E := mle τ fun i => rows.weightAt i u

/-- `V_{α,τ_1} = Σ_i ẽq(τ_1, i)·y(α)_i` of Akita (123). -/
def batchTarget (τ : List E) : E := mle τ rows.targetAt

/-- The batched claim `Σ_u w_u·m_{τ_1}(u) = V_{α,τ_1}` of Akita (121). -/
def Batched (w : Table) (τ : List E) : Prop :=
  ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * rows.batchWeight τ u = rows.batchTarget τ

theorem residual_of_count_le (w : Table) {i : ℕ} (large : rows.count ≤ i) :
    rows.residual μ w i = 0 := by
  simp [residual, weightAt, targetAt, Nat.not_lt.mpr large]

/-- The two sides of the batched claim (121) differ by the multilinear batch of the residuals of
the rows (Akita (123)). -/
theorem batch_residual (w : Table) (τ : List E) :
    ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * rows.batchWeight τ u - rows.batchTarget τ =
      mle τ (rows.residual μ w) := by
  show _ = mle τ (fun i => ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * rows.weightAt i u -
    rows.targetAt i)
  rw [mle_sub, mle_finset_sum]
  exact congrArg (· - mle τ rows.targetAt)
    (Finset.sum_congr rfl fun u _ => (mle_mul_left τ _ _).symm)

theorem batched_iff (w : Table) (τ : List E) :
    rows.Batched μ w τ ↔ mle τ (rows.residual μ w) = 0 := by
  rw [← batch_residual, sub_eq_zero, Batched]

/-- Akita Theorem 7.4, item 1, for the batching: when each row holds, the batched claim (121)
holds at each point `τ_1`. -/
theorem batched_of_holds {w : Table} (holds : rows.Holds μ w) (τ : List E) :
    rows.Batched μ w τ := by
  rw [batched_iff, show rows.residual μ w = fun _ => 0 from funext holds]
  exact mle_const τ 0

variable [Fintype E] [DecidableEq E]

open Classical in
/-- The bad set of the batching point `τ_1 ∈ E^s` for the witness `w`: the points where the batch
of a nonzero vector of residuals is zero (Akita Theorem 7.4, item 3). -/
noncomputable def batchBad (s : ℕ) (w : Table) : Finset (Fin s → E) :=
  if rows.Holds μ w then ∅
  else Finset.univ.filter fun τ => mle (List.ofFn τ) (rows.residual μ w) = 0

/-- A batched claim (121) that holds at a point `τ_1` outside the bad set gives each row. -/
theorem holds_of_batched {s : ℕ} {w : Table} {τ : Fin s → E}
    (outside : τ ∉ rows.batchBad μ s w) (batched : rows.Batched μ w (List.ofFn τ)) :
    rows.Holds μ w := by
  by_contra fails
  refine outside ?_
  rw [batchBad, ite_eq_right fails, Finset.mem_filter]
  exact ⟨Finset.mem_univ τ, (rows.batched_iff μ w _).mp batched⟩

/-- Akita Theorem 7.4, item 3: for `n_rel ≤ 2^s` rows, the bad set of the batching point has a
density of at most `s / |E|`. -/
theorem density_batchBad_le {s : ℕ} (fits : rows.count ≤ 2 ^ s) (w : Table) :
    density (rows.batchBad μ s w) ≤ s / Fintype.card E := by
  unfold batchBad
  split
  · rw [density_empty]
    positivity
  · rename_i fails
    obtain ⟨i, nonzero⟩ := not_forall.mp fails
    have small : i < rows.count := by
      by_contra large
      exact nonzero (rows.residual_of_count_le μ w (Nat.not_lt.mp large))
    exact density_mle_zero_le s (rows.residual μ w) ⟨i, small.trans_le fits, nonzero⟩

end FieldRows

/-- A ring reduction of a relation `Holds` on the flat witness of `2^vars` entries (Akita
Section 7.3). `rows α` is the field relation `M(α)·w = y(α)` at the point `α`, with at most
`2^batch` rows. `bad w` is a set of at most `badBound` points `α` for the witness `w`. `sound`
states that the field rows at a point outside `bad w` give the relation, and `complete` states that
the relation gives the field rows at each point. -/
structure Reduction (E : Type) [Field E] [Fintype E] [Algebra Zq E] where
  vars : ℕ
  low : ℕ
  batch : ℕ
  Holds : Table → Prop
  rows : E → FieldRows E
  fits : ∀ α, (rows α).count ≤ 2 ^ batch
  bad : Table → Finset E
  badBound : ℕ
  bad_card : ∀ w, (bad w).card ≤ badBound
  sound : ∀ w α, α ∉ bad w → (rows α).Holds vars w → Holds w
  complete : ∀ w α, Holds w → (rows α).Holds vars w

namespace Reduction

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] (R : Reduction E)

/-- Akita Theorem 7.4, item 1: a witness of the relation satisfies the batched claim (121) for
each `α` and each `τ_1`. -/
theorem batched_of_holds {w : Table} (holds : R.Holds w) (α : E) (τ : List E) :
    (R.rows α).Batched R.vars w τ :=
  (R.rows α).batched_of_holds R.vars (R.complete w α holds) τ

variable [DecidableEq E]

/-- A batched claim (121) that holds at a point `α` outside the bad set of the ring reduction and
at a point `τ_1` outside the bad set of the batching gives the relation. -/
theorem holds_of_batched {w : Table} {α : E} {τ : Fin R.batch → E} (reduce : α ∉ R.bad w)
    (outside : τ ∉ (R.rows α).batchBad R.vars R.batch w)
    (batched : (R.rows α).Batched R.vars w (List.ofFn τ)) : R.Holds w :=
  R.sound w α reduce ((R.rows α).holds_of_batched R.vars outside batched)

open Classical in
/-- Akita (122): for a witness that does not satisfy the relation, the batched claim (121) holds
for at most the fraction `(badBound + s_rel) / |E|` of the pairs `(α, τ_1)`. For the
quotient-free reduction, `badBound = d_max − 1`. -/
theorem density_accepting_le {w : Table} (fails : ¬ R.Holds w) :
    density (Finset.univ.filter fun p : E × (Fin R.batch → E) =>
      (R.rows p.1).Batched R.vars w (List.ofFn p.2)) ≤
        (R.badBound + R.batch : ℕ) / Fintype.card E := by
  have batch : (0 : ℝ) ≤ (R.batch : ℝ) / Fintype.card E := by positivity
  refine (density_prod_le' _ (R.bad w) _ batch fun α reduce => ?_).trans ?_
  · refine (density_mono fun τ member => ?_).trans
      ((R.rows α).density_batchBad_le R.vars (R.fits α) w)
    by_contra outside
    exact fails (R.holds_of_batched reduce outside (by simpa using member))
  · have bad : density (R.bad w) ≤ (R.badBound : ℝ) / Fintype.card E :=
      div_le_div_of_nonneg_right (by exact_mod_cast R.bad_card w) (by positivity)
    push_cast
    rw [add_div]
    linarith

end Reduction

/-- The groups of a fold in the order of the rows of the pinned code: the last group, then the
others in their order (`root_group_order` of `crates/akita-params/src/layout/opening_layout.rs`
and `build_relation_rhs_layout` of `relation_rhs_layout.rs` of the Akita code). Akita Figure 5 has
"increasing group"; Akita Theorem 7.4 holds for each fixed order of the rows. -/
def groupOrder (n : ℕ) : Equiv.Perm (Fin n) := (finRotate n).symm

/-- The group and the index in the group of each row of one family with `count g` rows of the
group `g`, for the groups in the order `groupOrder`. -/
def place {n : ℕ} (count : Fin n → ℕ) : Fin (∑ g, count g) ≃ (g : Fin n) × Fin (count g) :=
  (finCongr (Equiv.sum_comp (groupOrder n) count).symm).trans
    ((finSigmaFinEquiv (n := fun k => count (groupOrder n k))).symm.trans
      (Equiv.sigmaCongrLeft (β := fun g => Fin (count g)) (groupOrder n)))

/-- The join of `ring` ring rows over `Z_q` into `count` field rows: the ring row `r` is in the
field row `slot r` with the coefficient `basis r`. The planes of the carrier row (102) are one
field row with the coefficients `β_t` (Akita Section 7.3 and (113)). -/
structure Join (E : Type) (ring : ℕ) where
  count : ℕ
  slot : Fin ring → Fin count
  basis : Fin ring → E

namespace Join

variable {E : Type} [CommRing E] [Algebra Zq E] {ring : ℕ} (J : Join E ring)

/-- The coefficients of the ring rows of each field row are independent over `Z_q`. -/
def Independent : Prop :=
  ∀ (k : Fin J.count) (c : Fin ring → Zq),
    ∑ r, (if J.slot r = k then c r • J.basis r else 0) = 0 → ∀ r, J.slot r = k → c r = 0

/-- The polynomial over `E` of the field row `k` for the polynomials `P r` of the ring rows. -/
noncomputable def poly (k : Fin J.count) (P : Fin ring → Zq[X]) : E[X] :=
  ∑ r, if J.slot r = k then C (J.basis r) * (P r).map (algebraMap Zq E) else 0

theorem eval_poly (k : Fin J.count) (P : Fin ring → Zq[X]) (α : E) :
    (J.poly k P).eval α = ∑ r, if J.slot r = k then J.basis r * aeval α (P r) else 0 := by
  rw [poly, eval_finsetSum]
  refine Finset.sum_congr rfl fun r _ => ?_
  by_cases same : J.slot r = k
  · rw [ite_eq_left same, ite_eq_left same, eval_mul, eval_C, ← eval₂_eq_eval_map, ← aeval_def]
  · rw [ite_eq_right same, ite_eq_right same, eval_zero]

theorem natDegree_poly_le (k : Fin J.count) {P : Fin ring → Zq[X]} {bound : ℕ}
    (degrees : ∀ r, (P r).natDegree ≤ bound) : (J.poly k P).natDegree ≤ bound :=
  natDegree_sum_le_of_forall_le _ _ fun r _ => by
    by_cases same : J.slot r = k
    · rw [ite_eq_left same]
      exact (natDegree_C_mul_le _ _).trans (natDegree_map_le.trans (degrees r))
    · rw [ite_eq_right same, natDegree_zero]
      exact Nat.zero_le _

variable {J}

/-- A field row whose coefficients are independent over `Z_q` has a polynomial that is not zero
when the polynomial of one of its ring rows is not zero. -/
theorem poly_ne_zero (independent : J.Independent) {P : Fin ring → Zq[X]} {r : Fin ring}
    (nonzero : P r ≠ 0) : J.poly (J.slot r) P ≠ 0 := fun zero =>
  nonzero (AddMonoidAlgebra.ext <| Finsupp.ext fun j => by
    rw [Polynomial.coeff_zero]
    refine independent (J.slot r) (fun x => (P x).coeff j) ?_ r rfl
    have coefficient : (J.poly (J.slot r) P).coeff j = 0 := by rw [zero, Polynomial.coeff_zero]
    rw [poly, finsetSum_coeff] at coefficient
    refine Eq.trans (Finset.sum_congr rfl fun x _ => ?_) coefficient
    by_cases same : J.slot x = J.slot r
    · rw [ite_eq_left same, ite_eq_left same, Polynomial.coeff_C_mul, Polynomial.coeff_map,
        Algebra.smul_def, mul_comm]
    · rw [ite_eq_right same, ite_eq_right same, Polynomial.coeff_zero])

/-- Each ring row is one field row. -/
def plain (E : Type) [CommRing E] (ring : ℕ) : Join E ring := ⟨ring, id, fun _ => 1⟩

theorem plain_independent (injective : Function.Injective (algebraMap Zq E)) (ring : ℕ) :
    (plain E ring).Independent := fun k c zero r same => by
  have value : (∑ x, if (plain E ring).slot x = k then c x • (plain E ring).basis x else 0) =
      c k • (1 : E) :=
    (Finset.sum_ite_eq' Finset.univ k fun x => c x • (1 : E)).trans
      (ite_eq_left (Finset.mem_univ k))
  rw [value, Algebra.smul_def, mul_one] at zero
  have row : r = k := same
  rw [row]
  exact injective (zero.trans (map_zero _).symm)

/-- All the ring rows are one field row with the coefficients `β`. -/
def all (β : Fin ring → E) : Join E ring := ⟨1, fun _ => 0, β⟩

theorem all_independent {β : Fin ring → E} (independent : LinearIndependent Zq β) :
    (all β).Independent := fun k c zero r _ => by
  refine Fintype.linearIndependent_iff.mp independent c ?_ r
  rw [← zero]
  exact Finset.sum_congr rfl fun x _ =>
    (ite_eq_left (Subsingleton.elim (α := Fin 1) ((all β).slot x) k)).symm

/-- The field rows of `a`, then the field rows of `b`. -/
def append {m n : ℕ} (a : Join E m) (b : Join E n) : Join E (m + n) where
  count := a.count + b.count
  slot := Fin.addCases (fun i => Fin.castAdd b.count (a.slot i))
    fun j => Fin.natAdd a.count (b.slot j)
  basis := Fin.append a.basis b.basis

theorem append_independent {m n : ℕ} {a : Join E m} {b : Join E n} (left : a.Independent)
    (right : b.Independent) : (a.append b).Independent := by
  have apart : ∀ (x : Fin a.count) (y : Fin b.count),
      Fin.natAdd a.count y ≠ Fin.castAdd b.count x := fun x y same => by
    have values := congrArg Fin.val same
    rw [Fin.val_natAdd, Fin.val_castAdd] at values
    omega
  have raw : ∀ (k : Fin (a.count + b.count)) (c : Fin (m + n) → Zq),
      (∑ r, (if Fin.addCases (motive := fun _ => Fin (a.count + b.count))
          (fun i => Fin.castAdd b.count (a.slot i)) (fun j => Fin.natAdd a.count (b.slot j)) r = k
        then c r • Fin.append a.basis b.basis r else 0)) = 0 →
      ∀ r, Fin.addCases (motive := fun _ => Fin (a.count + b.count))
        (fun i => Fin.castAdd b.count (a.slot i)) (fun j => Fin.natAdd a.count (b.slot j)) r = k →
          c r = 0 := by
    intro k c zero
    rw [Fin.sum_univ_add] at zero
    simp only [Fin.addCases_left, Fin.addCases_right, Fin.append_left, Fin.append_right] at zero
    induction k using Fin.addCases with
    | left k =>
        simp only [apart, ite_false, Finset.sum_const_zero, add_zero, Fin.castAdd_inj] at zero
        intro r
        induction r using Fin.addCases with
        | left i =>
            intro same
            rw [Fin.addCases_left, Fin.castAdd_inj] at same
            exact left k (fun i => c (Fin.castAdd n i)) zero i same
        | right j =>
            intro same
            rw [Fin.addCases_right] at same
            exact absurd same (apart _ _)
    | right k =>
        simp only [fun x y => (apart x y).symm, ite_false, Finset.sum_const_zero, zero_add,
          Fin.natAdd_inj] at zero
        intro r
        induction r using Fin.addCases with
        | left i =>
            intro same
            rw [Fin.addCases_left] at same
            exact absurd same.symm (apart _ _)
        | right j =>
            intro same
            rw [Fin.addCases_right, Fin.natAdd_inj] at same
            exact right k (fun j => c (Fin.natAdd m j)) zero j same
  exact raw

/-- The field rows of each `f g`, with the groups in the order `groupOrder`. -/
def gather {n : ℕ} {size : Fin n → ℕ} (f : ∀ g, Join E (size g)) : Join E (∑ g, size g) where
  count := ∑ g, (f g).count
  slot r := (place fun g => (f g).count).symm
    ⟨(place size r).1, (f (place size r).1).slot (place size r).2⟩
  basis r := (f (place size r).1).basis (place size r).2

theorem gather_independent {n : ℕ} {size : Fin n → ℕ} {f : ∀ g, Join E (size g)}
    (each : ∀ g, (f g).Independent) : (gather f).Independent := by
  have raw : ∀ (k : Fin (∑ g, (f g).count)) (c : Fin (∑ g, size g) → Zq),
      (∑ r, (if ((place fun g => (f g).count).symm ⟨(place size r).1,
          (f (place size r).1).slot (place size r).2⟩ : Fin (∑ g, (f g).count)) = k
        then c r • (f (place size r).1).basis (place size r).2 else 0)) = 0 →
      ∀ r, ((place fun g => (f g).count).symm ⟨(place size r).1,
        (f (place size r).1).slot (place size r).2⟩ : Fin (∑ g, (f g).count)) = k → c r = 0 := by
    intro k c zero r same
    obtain ⟨⟨h, k⟩, rfl⟩ := (place fun g => (f g).count).symm.surjective k
    obtain ⟨⟨g, i⟩, rfl⟩ := (place size).symm.surjective r
    rw [Equiv.apply_symm_apply] at same
    obtain ⟨group, slot⟩ := Sigma.mk.inj_iff.mp ((place _).symm.injective same)
    subst group
    refine each g k (fun i => c ((place size).symm ⟨g, i⟩)) ?_ i (eq_of_heq slot)
    refine Eq.trans ?_ zero
    refine Eq.trans ?_ ((Fintype.sum_sigma _).symm.trans ((place size).symm.sum_comp _))
    refine Eq.trans ?_ (Finset.sum_eq_single g (fun h _ other => Finset.sum_eq_zero fun x _ => ?_)
      fun absent => absurd (Finset.mem_univ g) absent).symm
    · refine Finset.sum_congr rfl fun x _ => ?_
      rw [Equiv.apply_symm_apply]
      exact if_congr ⟨fun same => by rw [same],
        fun same => eq_of_heq (Sigma.mk.inj_iff.mp ((place _).symm.injective same)).2⟩ rfl rfl
    · rw [Equiv.apply_symm_apply]
      exact ite_eq_right fun same =>
        other (Sigma.mk.inj_iff.mp ((place _).symm.injective same)).1
  exact raw

end Join

/-- The relation of one fold on the flat witness of `2^vars` entries (Akita Section 7.3 and
Figure 5): the ring rows in the normal form (110), each with its addresses inside the witness,
the join of the ring rows into the rows over `E`, and one scalar row `Σ_u w_u·weight u = value`
over `E` (family 6 of Figure 5). `low` is `log_2` of the coefficient block of the addresses of the
pinned code (`relation_coefficient_variable_count` of
`crates/akita-params/src/layout/relation_address.rs` of the Akita code). -/
structure FlatRelation (E : Type) [CommRing E] [Algebra Zq E] where
  vars : ℕ
  low : ℕ
  ring : ℕ
  row : Fin ring → RingRow
  inside : ∀ r c j x, j < (row r).degree → (row r).address c j = some x → x < 2 ^ vars
  join : Join E ring
  independent : join.Independent
  weight : ℕ → E
  value : E

namespace RingRow

variable {E : Type} [CommRing E] [Algebra Zq E] (row : RingRow)

/-- The coefficient of the witness entry `u` in `z_r(α)` of Akita (120): the sum of the residue
kernels `κ^{(d_r)}_{A_{r,c},α}(j)` over the positions with `addr_{r,c}(j) = u`. -/
noncomputable def freeWeight (α : E) (u : ℕ) : E :=
  ∑ c, ∑ j ∈ Finset.range row.degree,
    if row.address c j = some u then residueKernel (row.multiplier c) α j else 0

/-- `z_r(α)` of Akita (120) is a linear form of the flat witness with the weights `freeWeight`,
minus the value of the target. -/
theorem reduced_eq_sum (w : Table) (α : E) {μ : ℕ}
    (inside : ∀ c j x, j < row.degree → row.address c j = some x → x < 2 ^ μ) :
    row.reduced w α =
      ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * row.freeWeight α u -
        reducedEval α row.target := by
  rw [reduced]
  refine congrArg (· - reducedEval α row.target) ?_
  simp only [freeWeight, Finset.mul_sum]
  rw [Finset.sum_comm (s := Finset.range (2 ^ μ))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_comm (s := Finset.range (2 ^ μ))]
  refine Finset.sum_congr rfl fun j member => ?_
  simp only [mul_ite, mul_zero]
  cases read : row.address c j with
  | none => simp
  | some x =>
      simp only [Option.some.injEq, Option.elim_some]
      rw [Finset.sum_ite_eq, ite_eq_left
        (Finset.mem_range.mpr (inside c j x (Finset.mem_range.mp member) read))]
      exact mul_comm _ _

end RingRow

theorem elim_map (o : Option ℕ) (move : ℕ → ℕ) (w : Table) :
    (o.map move).elim 0 w = o.elim 0 fun x => w (move x) := by
  cases o <;> rfl

/-- The ring row with each address moved by `move`. -/
def RingRow.relocate (row : RingRow) (move : ℕ → ℕ) : RingRow where
  degree := row.degree
  occurrences := row.occurrences
  multiplier := row.multiplier
  address c j := (row.address c j).map move
  target := row.target

theorem RingRow.relocate_holds (row : RingRow) (move : ℕ → ℕ) (w : Table) :
    (row.relocate move).Holds w ↔ row.Holds fun x => w (move x) := by
  have reads : ∀ c, (row.relocate move).read w c = row.read (fun x => w (move x)) c :=
    fun c => congrArg ofCoeffs (funext fun j => elim_map (row.address c j) move w)
  unfold RingRow.Holds RingRow.residual
  simp only [reads]
  rfl

namespace FlatRelation

variable {E : Type} [Field E] [Algebra Zq E] (rel : FlatRelation E)

/-- The witness `w` satisfies each ring row (110) and the scalar row. -/
def Holds (w : Table) : Prop :=
  (∀ r, (rel.row r).Holds w) ∧
    ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) * rel.weight u = rel.value

/-- The field rows of the quotient-free reduction at the point `α` (Akita Section 7.3): for each
row over `E` the residual `z_r(α)` of (120), which for a joined row is the sum of the residuals of
its planes with the coefficients of the join, then the scalar row. -/
noncomputable def freeRows (α : E) : FieldRows E where
  count := rel.join.count + 1
  weight := Fin.lastCases rel.weight fun k u =>
    ∑ r, if rel.join.slot r = k then rel.join.basis r * (rel.row r).freeWeight α u else 0
  target := Fin.lastCases rel.value fun k =>
    ∑ r, if rel.join.slot r = k then rel.join.basis r * reducedEval α (rel.row r).target else 0

/-- The residual of the row `k` over `E` as a polynomial over `E`: the representatives of the
residuals `Z_r` of its planes with the coefficients of the join. -/
noncomputable def freePoly (w : Table) (k : Fin rel.join.count) : E[X] :=
  rel.join.poly k fun r => rep ((rel.row r).residual w)

theorem freeRows_residual_ring (w : Table) (α : E) (k : Fin rel.join.count) :
    (rel.freeRows α).residual rel.vars w k = (rel.freePoly w k).eval α := by
  have small : (k : ℕ) < rel.join.count + 1 := k.isLt.trans (Nat.lt_succ_self _)
  have cast : (⟨k, small⟩ : Fin (rel.join.count + 1)) = k.castSucc := rfl
  have swap : ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) *
      (∑ r, if rel.join.slot r = k then rel.join.basis r * (rel.row r).freeWeight α u else 0) =
      ∑ r, if rel.join.slot r = k then rel.join.basis r *
        ∑ u ∈ Finset.range (2 ^ rel.vars),
          algebraMap Zq E (w u) * (rel.row r).freeWeight α u else 0 := by
    simp only [Finset.mul_sum, mul_ite, mul_zero]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun r _ => ?_
    by_cases same : rel.join.slot r = k
    · simp only [ite_eq_left same]
      exact Finset.sum_congr rfl fun u _ => mul_left_comm _ _ _
    · simp only [ite_eq_right same, Finset.sum_const_zero]
  rw [freePoly, Join.eval_poly]
  simp only [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, freeRows, small, cast,
    dite_true, Fin.lastCases_castSucc]
  rw [swap, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun r _ => ?_
  by_cases same : rel.join.slot r = k
  · rw [ite_eq_left same, ite_eq_left same, ite_eq_left same, ← mul_sub, ← reducedEval_apply,
      ← RingRow.reduced_eq, (rel.row r).reduced_eq_sum w α (rel.inside r)]
  · rw [ite_eq_right same, ite_eq_right same, ite_eq_right same, sub_zero]

theorem freeRows_residual_scalar (w : Table) (α : E) :
    (rel.freeRows α).residual rel.vars w rel.join.count =
      ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) * rel.weight u - rel.value := by
  have cast : (⟨rel.join.count, Nat.lt_succ_self _⟩ : Fin (rel.join.count + 1)) =
      Fin.last rel.join.count := rfl
  simp [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, freeRows, cast]

/-- Akita Theorem 7.4, item 1, for the ring reduction: a witness of the relation has a zero
residual in each field row at each point `α`. -/
theorem freeRows_holds {w : Table} (holds : rel.Holds w) (α : E) :
    (rel.freeRows α).Holds rel.vars w := by
  intro i
  rcases Nat.lt_trichotomy i rel.join.count with small | rfl | large
  · rw [show i = ((⟨i, small⟩ : Fin rel.join.count) : ℕ) from rfl, freeRows_residual_ring,
      freePoly, Join.eval_poly]
    exact Finset.sum_eq_zero fun r _ => by
      have check : (rel.row r).reduced w α = 0 := (rel.row r).freeCheck_of_holds (holds.1 r) α
      rw [← reducedEval_apply, ← RingRow.reduced_eq, check, mul_zero, ite_self]
  · rw [freeRows_residual_scalar, holds.2, sub_self]
  · exact (rel.freeRows α).residual_of_count_le rel.vars w large

open Classical in
/-- The bad set of the point `α` of the quotient-free reduction for the witness `w`: the roots of
the residual of the row over `E` of one ring row that does not hold (Akita Theorem 7.4,
item 2). -/
noncomputable def freeBad (w : Table) : Finset E :=
  if fails : ∃ r, ¬ (rel.row r).Holds w then
    (rel.freePoly w (rel.join.slot fails.choose)).roots.toFinset
  else ∅

/-- Akita Theorem 7.4, item 2: the bad set of `α` has at most `d_max − 1` points. A joined row
counts once, at the degree of its planes. -/
theorem freeBad_card_le {largest : ℕ} (degrees : ∀ r, (rel.row r).degree ≤ largest) (w : Table) :
    (rel.freeBad w).card ≤ largest - 1 := by
  classical
  unfold freeBad
  split
  · exact (Multiset.toFinset_card_le _).trans ((card_roots' _).trans
      (rel.join.natDegree_poly_le _ fun r => by
        have degree := natDegree_rep_lt ((rel.row r).residual w)
        have bound := degrees r
        omega))
  · simp

/-- Field rows that hold at a point `α` outside the bad set give the relation (Akita
Theorem 7.4, item 2). The planes of a joined row are the coordinates of one polynomial over `E`
in a basis over `Z_q`, so this polynomial is zero only when each plane is zero. -/
theorem holds_of_freeRows {w : Table} {α : E} (outside : α ∉ rel.freeBad w)
    (rows : (rel.freeRows α).Holds rel.vars w) : rel.Holds w := by
  classical
  refine ⟨?_, ?_⟩
  · by_contra fails
    have some : ∃ r, ¬ (rel.row r).Holds w := not_forall.mp fails
    rw [freeBad, dite_eq_left some] at outside
    refine outside (Multiset.mem_toFinset.mpr (mem_roots'.mpr
      ⟨rel.join.poly_ne_zero rel.independent fun zero => some.choose_spec ?_, ?_⟩))
    · rw [RingRow.Holds, ← mk_rep ((rel.row some.choose).residual w), zero, map_zero]
    · rw [IsRoot.def]
      exact (rel.freeRows_residual_ring w α _).symm.trans (rows _)
  · have scalar := rows rel.join.count
    rw [freeRows_residual_scalar] at scalar
    exact sub_eq_zero.mp scalar

variable [Fintype E]

/-- The quotient-free reduction of the relation (Akita Section 7.3 and Theorem 7.4), for ring rows
of the degrees `d_r ≤ d_max` and `n_rel = count + 1 ≤ 2^s_rel` rows over `E`. Its bad set has at
most `d_max − 1` points. -/
noncomputable def free (largest batch : ℕ) (degrees : ∀ r, (rel.row r).degree ≤ largest)
    (fits : rel.join.count + 1 ≤ 2 ^ batch) : Reduction E where
  vars := rel.vars
  low := rel.low
  batch := batch
  Holds := rel.Holds
  rows := rel.freeRows
  fits _ := fits
  bad := rel.freeBad
  badBound := largest - 1
  bad_card := rel.freeBad_card_le degrees
  sound _ _ outside rows := rel.holds_of_freeRows outside rows
  complete _ α holds := rel.freeRows_holds holds α

end FlatRelation

end NightstreamFPrime.Spec.Compression.Akita
