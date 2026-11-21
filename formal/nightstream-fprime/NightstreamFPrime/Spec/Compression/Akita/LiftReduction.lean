import NightstreamFPrime.Spec.Compression.Akita.RowBatch

/-! The quotient-lift reduction of the relation of one fold of Akita (ePrint 2026/1983, revision
of 2026-10-15, Sections 5.5 and 7.3, Lemma 7.1 and Theorem 10.36).

- `QuotientRead` has the positions of the digits of the quotient `Q_r` of one ring row in the flat
  witness (the segments `Q̂` of Akita (76)) and the public weights that recompose one coefficient.
  `RingRow.quotient` is `Q_r`, of degree below `d_r`.
- `RingRow.liftWeight` is the coefficient of one witness entry in the check (112), and
  `RingRow.liftCheck_iff_sum` states that the check (112) is a linear form of the flat witness.
- `LiftRelation` adds the quotient digits to `FlatRelation`. `LiftRelation.Holds` is the identity
  (111) of each ring row with the quotient of the witness, and the scalar row.
  `LiftRelation.holds_flat` gives the ring rows (110).
- `LiftRelation.lift` is the quotient-lift reduction, with a bad set of at most `2·d_max − 1`
  points (Akita Lemma 7.1, item 2). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

/-- The digits of the quotient of one ring row in the flat witness (Akita Section 5.5):
coefficient `j` of the quotient is `Σ_a scale a · w_{address j a}`. -/
structure QuotientRead where
  digits : ℕ
  scale : Fin digits → Zq
  address : ℕ → Fin digits → ℕ

namespace QuotientRead

variable {E : Type} [CommRing E] [Algebra Zq E] (q : QuotientRead)

/-- The coefficient of the witness entry `u` in the value `Q_r(α)` of a quotient of degree below
`d`. -/
noncomputable def weight (d : ℕ) (α : E) (u : ℕ) : E :=
  ∑ j ∈ Finset.range d, ∑ a, if q.address j a = u then α ^ j * algebraMap Zq E (q.scale a) else 0

end QuotientRead

namespace RingRow

variable {E : Type} [CommRing E] [Algebra Zq E] (row : RingRow) (q : QuotientRead)

/-- The quotient `Q_r` of Akita (111) that the flat witness `w` has at the positions of `q`. -/
noncomputable def quotient (w : Table) : Zq[X] :=
  ∑ j ∈ Finset.range row.degree, C (∑ a, q.scale a * w (q.address j a)) * X ^ j

theorem natDegree_quotient_lt (w : Table) : (row.quotient q w).natDegree < row.degree := by
  have positive := NeZero.pos row.degree
  have bound : (row.quotient q w).natDegree ≤ row.degree - 1 :=
    natDegree_sum_le_of_forall_le _ _ fun j member =>
      (natDegree_C_mul_X_pow_le _ _).trans (by have := Finset.mem_range.mp member; omega)
  omega

/-- The coefficient of the witness entry `u` in the term `Σ_c A_{r,c}(α)·W_{r,c}(α)` of the check
(112). -/
noncomputable def readWeight (α : E) (u : ℕ) : E :=
  ∑ c, ∑ j ∈ Finset.range row.degree,
    if row.address c j = some u then aeval α (rep (row.multiplier c)) * α ^ j else 0

/-- The coefficient of the witness entry `u` in the left side of the check (112). -/
noncomputable def liftWeight (α : E) (u : ℕ) : E :=
  row.readWeight α u - (α ^ row.degree + 1) * q.weight row.degree α u

theorem sum_readWeight (w : Table) (α : E) {μ : ℕ}
    (inside : ∀ c j x, j < row.degree → row.address c j = some x → x < 2 ^ μ) :
    ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * row.readWeight α u =
      ∑ c, aeval α (rep (row.multiplier c)) *
        ∑ j ∈ Finset.range row.degree, α ^ j * algebraMap Zq E ((row.address c j).elim 0 w) := by
  simp only [readWeight, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j member => ?_
  simp only [mul_ite, mul_zero]
  cases read : row.address c j with
  | none => simp
  | some x =>
      simp only [Option.some.injEq, Option.elim_some]
      rw [Finset.sum_ite_eq, ite_eq_left
        (Finset.mem_range.mpr (inside c j x (Finset.mem_range.mp member) read))]
      ring

theorem sum_quotientWeight (w : Table) (α : E) {μ : ℕ}
    (inside : ∀ j a, j < row.degree → q.address j a < 2 ^ μ) :
    ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * q.weight row.degree α u =
      aeval α (row.quotient q w) := by
  simp only [QuotientRead.weight, quotient, Finset.mul_sum, map_sum, map_mul, map_pow, aeval_C,
    aeval_X, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j member => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq,
    ite_eq_left (Finset.mem_range.mpr (inside j a (Finset.mem_range.mp member)))]
  ring

/-- The check (112) of Akita with the quotient of the witness is a linear form of the flat
witness with the weights `liftWeight`, minus the value of the target. -/
theorem liftCheck_iff_sum (w : Table) (α : E) {μ : ℕ}
    (inside : ∀ c j x, j < row.degree → row.address c j = some x → x < 2 ^ μ)
    (quotientInside : ∀ j a, j < row.degree → q.address j a < 2 ^ μ) :
    row.LiftCheck w (row.quotient q w) α ↔
      ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * row.liftWeight q α u -
        aeval α (rep row.target) = 0 := by
  have split : ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * row.liftWeight q α u =
      ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * row.readWeight α u -
        (α ^ row.degree + 1) *
          ∑ u ∈ Finset.range (2 ^ μ), algebraMap Zq E (w u) * q.weight row.degree α u := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun u _ => by rw [liftWeight]; ring
  rw [split, row.sum_readWeight w α inside, row.sum_quotientWeight q w α quotientInside, LiftCheck]
  exact ⟨fun check => by linear_combination check, fun check => by linear_combination check⟩

end RingRow

/-- The relation of one fold on the flat witness on the quotient-lift route (Akita Section 7.3):
the ring rows and the scalar row of `FlatRelation`, and for each ring row the positions of the
digits of its quotient, inside the witness. -/
structure LiftRelation (E : Type) extends FlatRelation E where
  quotient : Fin ring → QuotientRead
  quotientInside : ∀ r j a, j < (row r).degree → (quotient r).address j a < 2 ^ vars

namespace LiftRelation

variable {E : Type} [Field E] [Algebra Zq E] (rel : LiftRelation E)

/-- The identity (111) of the ring row `r` with the quotient of the witness `w`. -/
def Lifts (w : Table) (r : Fin rel.ring) : Prop :=
  (rel.row r).lifted w = modulus (rel.row r).degree * (rel.row r).quotient (rel.quotient r) w

/-- The witness `w` satisfies the identity (111) of each ring row with its quotient, and the
scalar row. -/
def Holds (w : Table) : Prop :=
  (∀ r, rel.Lifts w r) ∧
    ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) * rel.weight u = rel.value

/-- A witness of the identities (111) satisfies the ring rows (110) (Akita Lemma 3.4). -/
theorem holds_flat {w : Table} (holds : rel.Holds w) : rel.toFlatRelation.Holds w :=
  ⟨fun r => ((rel.row r).holds_iff_dvd w).mpr ⟨_, holds.1 r⟩, holds.2⟩

/-- The field rows of the quotient-lift reduction at the point `α` (Akita (112)): the check of
each ring row, then the scalar row. -/
noncomputable def liftRows (α : E) : FieldRows E where
  count := rel.ring + 1
  weight := Fin.lastCases rel.weight fun r => (rel.row r).liftWeight (rel.quotient r) α
  target := Fin.lastCases rel.value fun r => aeval α (rep (rel.row r).target)

theorem liftRows_residual_ring (w : Table) (α : E) (r : Fin rel.ring) :
    (rel.liftRows α).residual rel.vars w r =
      ∑ u ∈ Finset.range (2 ^ rel.vars),
        algebraMap Zq E (w u) * (rel.row r).liftWeight (rel.quotient r) α u -
          aeval α (rep (rel.row r).target) := by
  have small : (r : ℕ) < rel.ring + 1 := r.isLt.trans (Nat.lt_succ_self _)
  have cast : (⟨r, small⟩ : Fin (rel.ring + 1)) = r.castSucc := rfl
  simp [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, liftRows, small, cast]

theorem liftRows_residual_scalar (w : Table) (α : E) :
    (rel.liftRows α).residual rel.vars w rel.ring =
      ∑ u ∈ Finset.range (2 ^ rel.vars), algebraMap Zq E (w u) * rel.weight u - rel.value := by
  have cast : (⟨rel.ring, Nat.lt_succ_self _⟩ : Fin (rel.ring + 1)) = Fin.last rel.ring := rfl
  simp [FieldRows.residual, FieldRows.weightAt, FieldRows.targetAt, liftRows, cast]

theorem liftRows_check (w : Table) (α : E) (r : Fin rel.ring) :
    (rel.liftRows α).residual rel.vars w r = 0 ↔
      (rel.row r).LiftCheck w ((rel.row r).quotient (rel.quotient r) w) α := by
  rw [liftRows_residual_ring, (rel.row r).liftCheck_iff_sum (rel.quotient r) w α (rel.inside r)
    (rel.quotientInside r)]

/-- Akita Lemma 7.1, item 1: a witness of the identities (111) passes the check (112) of each row
at each point `α`. -/
theorem liftRows_holds {w : Table} (holds : rel.Holds w) (α : E) :
    (rel.liftRows α).Holds rel.vars w := by
  intro i
  rcases Nat.lt_trichotomy i rel.ring with small | rfl | large
  · rw [show i = ((⟨i, small⟩ : Fin rel.ring) : ℕ) from rfl, liftRows_check,
      RingRow.liftCheck_iff, holds.1 ⟨i, small⟩, sub_self, map_zero]
  · rw [liftRows_residual_scalar, holds.2, sub_self]
  · exact (rel.liftRows α).residual_of_count_le rel.vars w large

open Classical in
/-- The bad set of the point `α` of the quotient-lift reduction for the witness `w`: the roots of
the difference of the two sides of (111) for one ring row where they differ (Akita Lemma 7.1,
item 2). -/
noncomputable def liftBad (w : Table) : Finset E :=
  if fails : ∃ r, ¬ rel.Lifts w r then
    (rel.row fails.choose).liftBad w ((rel.row fails.choose).quotient (rel.quotient fails.choose) w)
  else ∅

/-- Akita Lemma 7.1, item 2: the bad set of `α` has at most `2·d_max − 1` points. -/
theorem liftBad_card_le {largest : ℕ} (degrees : ∀ r, (rel.row r).degree ≤ largest) (w : Table) :
    (rel.liftBad w).card ≤ 2 * largest - 1 := by
  unfold liftBad
  split
  · rename_i fails
    have degree := degrees fails.choose
    exact ((rel.row fails.choose).liftBad_card_le w
      ((rel.row fails.choose).natDegree_quotient_lt _ w)).trans (by omega)
  · simp

/-- Checks (112) that pass at a point `α` outside the bad set give the identities (111)
(Akita Lemma 7.1, item 2). -/
theorem holds_of_liftRows (injective : Function.Injective (algebraMap Zq E)) {w : Table} {α : E}
    (outside : α ∉ rel.liftBad w) (rows : (rel.liftRows α).Holds rel.vars w) : rel.Holds w := by
  classical
  refine ⟨?_, ?_⟩
  · by_contra fails
    have some : ∃ r, ¬ rel.Lifts w r := not_forall.mp fails
    rw [liftBad, dite_eq_left some] at outside
    have root := ((rel.row some.choose).liftCheck_iff w _ α).mp
      ((rel.liftRows_check w α some.choose).mp (rows some.choose))
    refine outside (Multiset.mem_toFinset.mpr (mem_roots'.mpr
      ⟨fun mapped => some.choose_spec
        (sub_eq_zero.mp ((Polynomial.map_eq_zero_iff injective).mp mapped)), ?_⟩))
    rw [IsRoot.def, ← eval₂_eq_eval_map, ← aeval_def]
    exact root
  · have scalar := rows rel.ring
    rw [liftRows_residual_scalar] at scalar
    exact sub_eq_zero.mp scalar

variable [Fintype E]

/-- The quotient-lift reduction of the relation (Akita Section 7.3 and Lemma 7.1), for ring rows
of the degrees `d_r ≤ d_max` and `n_rel = ring + 1 ≤ 2^s_rel` rows. Its bad set has at most
`2·d_max − 1` points. -/
noncomputable def lift (injective : Function.Injective (algebraMap Zq E)) (largest batch : ℕ)
    (degrees : ∀ r, (rel.row r).degree ≤ largest) (fits : rel.ring + 1 ≤ 2 ^ batch) :
    Reduction E where
  vars := rel.vars
  batch := batch
  Holds := rel.Holds
  rows := rel.liftRows
  fits _ := fits
  bad := rel.liftBad
  badBound := 2 * largest - 1
  bad_card := rel.liftBad_card_le degrees
  sound _ _ outside rows := rel.holds_of_liftRows injective outside rows
  complete _ α holds := rel.liftRows_holds holds α

end LiftRelation

end NightstreamFPrime.Spec.Compression.Akita
