import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Akita.RingCheck

/-! Row batching of the relation of one fold of Akita (ePrint 2026/1983, revision of 2026-10-15,
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
  its field rows. `Reduction.batched_of_holds`, `Reduction.holds_of_batched` and
  `Reduction.density_accepting_le` are Akita Theorem 7.4, item 1 and the bound (122).
- `RingRow.relocate` is a ring row with each address moved by a map.
- `FlatRelation` has the ring rows (110) and the scalar row of Akita Figure 5 on the flat
  witness. `FlatRelation.free` is its quotient-free reduction (Akita (120)), with the bad set of
  Akita Theorem 7.4, item 2. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib

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

/-- The relation of one fold on the flat witness of `2^vars` entries (Akita Section 7.3 and
Figure 5): the ring rows in the normal form (110), each with its addresses inside the witness, and
one scalar row `Σ_u w_u·weight u = value` over `E` (family 6 of Figure 5). -/
structure FlatRelation (E : Type) where
  vars : ℕ
  ring : ℕ
  row : Fin ring → RingRow
  inside : ∀ r c j x, j < (row r).degree → (row r).address c j = some x → x < 2 ^ vars
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

/-- The field rows of the quotient-free reduction at the point `α` (Akita Section 7.3): the
residual `z_r(α)` of (120) for each ring row, then the scalar row. -/
noncomputable def freeRows (α : E) : FieldRows E where
  count := rel.ring + 1
  weight := Fin.lastCases rel.weight fun r => (rel.row r).freeWeight α
  target := Fin.lastCases rel.value fun r => reducedEval α (rel.row r).target

theorem freeRows_residual_ring (w : Table) (α : E) (r : Fin rel.ring) :
    (rel.freeRows α).residual rel.vars w r = (rel.row r).reduced w α := by
  have small : (r : ℕ) < rel.ring + 1 := r.isLt.trans (Nat.lt_succ_self _)
  have cast : (⟨r, small⟩ : Fin (rel.ring + 1)) = r.castSucc := rfl
  rw [(rel.row r).reduced_eq_sum w α (rel.inside r)]
  simp [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, freeRows, small, cast]

theorem freeRows_residual_scalar (w : Table) (α : E) :
    (rel.freeRows α).residual rel.vars w rel.ring =
      ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) * rel.weight u - rel.value := by
  have cast : (⟨rel.ring, Nat.lt_succ_self _⟩ : Fin (rel.ring + 1)) = Fin.last rel.ring := rfl
  simp [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, freeRows, cast]

/-- Akita Theorem 7.4, item 1, for the ring reduction: a witness of the relation has a zero
residual in each field row at each point `α`. -/
theorem freeRows_holds {w : Table} (holds : rel.Holds w) (α : E) :
    (rel.freeRows α).Holds rel.vars w := by
  intro i
  rcases Nat.lt_trichotomy i rel.ring with small | rfl | large
  · rw [show i = ((⟨i, small⟩ : Fin rel.ring) : ℕ) from rfl, freeRows_residual_ring]
    exact (rel.row ⟨i, small⟩).freeCheck_of_holds (holds.1 ⟨i, small⟩) α
  · rw [freeRows_residual_scalar, holds.2, sub_self]
  · exact (rel.freeRows α).residual_of_count_le rel.vars w large

open Classical in
/-- The bad set of the point `α` of the quotient-free reduction for the witness `w`: the roots of
the residual of one ring row that does not hold (Akita Theorem 7.4, item 2). -/
noncomputable def freeBad (w : Table) : Finset E :=
  if fails : ∃ r, ¬ (rel.row r).Holds w then (rel.row fails.choose).freeBad w else ∅

/-- Akita Theorem 7.4, item 2: the bad set of `α` has at most `d_max − 1` points. -/
theorem freeBad_card_le {largest : ℕ} (degrees : ∀ r, (rel.row r).degree ≤ largest) (w : Table) :
    (rel.freeBad w).card ≤ largest - 1 := by
  unfold freeBad
  split
  · rename_i fails
    have degree := degrees fails.choose
    exact ((rel.row fails.choose).freeBad_card_le w).trans (by omega)
  · simp

/-- Field rows that hold at a point `α` outside the bad set give the relation (Akita
Theorem 7.4, item 2). -/
theorem holds_of_freeRows (injective : Function.Injective (algebraMap Zq E)) {w : Table} {α : E}
    (outside : α ∉ rel.freeBad w) (rows : (rel.freeRows α).Holds rel.vars w) :
    rel.Holds w := by
  have checks (r : Fin rel.ring) : (rel.row r).FreeCheck w α := by
    rw [RingRow.FreeCheck, ← freeRows_residual_ring]
    exact rows r
  refine ⟨?_, ?_⟩
  · by_contra fails
    have some : ∃ r, ¬ (rel.row r).Holds w := not_forall.mp fails
    rw [freeBad, dite_eq_left some] at outside
    exact some.choose_spec
      ((rel.row some.choose).holds_of_freeCheck injective outside (checks some.choose))
  · have scalar := rows rel.ring
    rw [freeRows_residual_scalar] at scalar
    exact sub_eq_zero.mp scalar

variable [Fintype E]

/-- The quotient-free reduction of the relation (Akita Section 7.3 and Theorem 7.4), for ring rows
of the degrees `d_r ≤ d_max` and `n_rel = ring + 1 ≤ 2^s_rel` rows. Its bad set has at most
`d_max − 1` points. -/
noncomputable def free (injective : Function.Injective (algebraMap Zq E)) (largest batch : ℕ)
    (degrees : ∀ r, (rel.row r).degree ≤ largest) (fits : rel.ring + 1 ≤ 2 ^ batch) :
    Reduction E where
  vars := rel.vars
  batch := batch
  Holds := rel.Holds
  rows := rel.freeRows
  fits _ := fits
  bad := rel.freeBad
  badBound := largest - 1
  bad_card := rel.freeBad_card_le degrees
  sound _ _ outside rows := rel.holds_of_freeRows injective outside rows
  complete _ α holds := rel.freeRows_holds holds α

end FlatRelation

end NightstreamFPrime.Spec.Compression.Akita
