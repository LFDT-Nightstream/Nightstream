import NightstreamFPrime.Spec.Compression.Akita.BatchFlat
import NightstreamFPrime.Spec.Compression.Akita.LiftReduction
import NightstreamFPrime.Spec.Compression.Akita.Prover

/-! The quotient digits of the ring rows of one fold of Akita as segments of the flat witness
(ePrint 2026/1983, revision of 2026-10-15, Section 5.5 with (76), Section 7.3 with (111) and
(112), Lemma 3.4, and Appendix L.2 with (348)).

- `FlatRelation.quotientCell` is the address (348) of one digit of one coefficient of the quotient
  `Q_r` of a ring row: the row `r` has a segment of `δ_quot·d_r` cells, and in it the coefficient
  index changes first, then the digit. The segments of the rows start at the cell `start`, in the
  order of the rows.
- `FlatRelation.withQuotients` is the `LiftRelation` of a relation with these addresses and the
  weights `b^u` of the balanced base `b`.
- `FlatRelation.quotients` has a quotient of each ring row that holds (Akita Lemma 3.4), and
  `FlatRelation.completion` is a table with the digits of these quotients in the segments.
- `FlatRelation.withQuotients_holds` states that the completion of a table with the relation
  satisfies the identities (111): the relation `LiftRelation.Holds` has a witness.
  `FlatRelation.completion_range` is the digit range statement of the completion.

Akita (76) puts the quotient digits of the families 1 to 4 of Figure 5 before the compression
digits, and the quotient digits of each compression stage after its binary digits. Here each
quotient segment is after the cell `start`, which the levels put after the last compression
digit. Each quotient has `δ_quot` digits for each coefficient, with one base for each row. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

namespace FlatRelation

variable {E : Type} (rel : FlatRelation E) (base digits start : ℕ)

/-- The number of cells of the digits of the quotient of the row `r`: `δ_quot·d_r`. -/
def quotientCells (r : Fin rel.ring) : ℕ := digits * (rel.row r).degree

/-- The first cell after the quotient segments. -/
def quotientEnd : ℕ := start + ∑ r, rel.quotientCells digits r

/-- The position of digit `a` of coefficient `j` of the quotient of the row `r`: the local
position `a·d_r + j` of Akita (348), for one base-field coordinate in a coefficient, in the
segment of the row. -/
def quotientCell (r : Fin rel.ring) (j : ℕ) (a : Fin digits) : ℕ :=
  start + (offset (rel.quotientCells digits) r + (a * (rel.row r).degree + j))

theorem quotientCell_lt (r : Fin rel.ring) {j : ℕ} (small : j < (rel.row r).degree)
    (a : Fin digits) : rel.quotientCell digits start r j a < rel.quotientEnd digits start := by
  have cell := cell_lt a small
  have last := offset_add_le (rel.quotientCells digits) r
  have size : rel.quotientCells digits r = digits * (rel.row r).degree := rfl
  unfold quotientCell quotientEnd
  omega

/-- The quotient segments have `δ_quot·Σ_r d_r` cells. -/
theorem quotientEnd_eq : rel.quotientEnd digits start =
    start + digits * ∑ r, (rel.row r).degree := by
  unfold quotientEnd quotientCells
  rw [Finset.mul_sum]

/-- The relation with the quotient digits in the witness (Akita Section 7.3): the quotient of
each ring row has `digits` digits of the base `base` for each coefficient, at the positions
`quotientCell`. -/
def withQuotients (inside : rel.quotientEnd digits start ≤ 2 ^ rel.vars) : LiftRelation E where
  toFlatRelation := rel
  quotient r := ⟨digits, fun a => (base : Zq) ^ (a : ℕ), rel.quotientCell digits start r⟩
  quotientInside r _ a small := (rel.quotientCell_lt digits start r small a).trans_le inside

/-- The table of the digits of the quotients `quotient r`, in the order of the rows. -/
noncomputable def quotientTable (quotient : Fin rel.ring → Zq[X]) : Table :=
  concat (rel.quotientCells digits) fun r y =>
    fieldDigit base digits ((quotient r).coeff (y % (rel.row r).degree))
      (y / (rel.row r).degree)

/-- The table with the cells of `w` before `start`, and then the digits of the quotients. -/
noncomputable def extend (w : Table) (quotient : Fin rel.ring → Zq[X]) : Table :=
  join start w (rel.quotientTable base digits quotient)

theorem extend_left (w : Table) (quotient : Fin rel.ring → Zq[X]) {x : ℕ} (small : x < start) :
    rel.extend base digits start w quotient x = w x :=
  join_left _ _ small

theorem extend_cell (w : Table) (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring) {j : ℕ}
    (small : j < (rel.row r).degree) (a : Fin digits) :
    rel.extend base digits start w quotient (rel.quotientCell digits start r j a) =
      fieldDigit base digits ((quotient r).coeff j) a := by
  have rest : (a * (rel.row r).degree + j) % (rel.row r).degree = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]
  have index : (a * (rel.row r).degree + j) / (rel.row r).degree = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (NeZero.pos _), Nat.div_eq_of_lt small, zero_add]
  unfold extend quotientCell quotientTable
  rw [join_right, concat_apply (rel.quotientCells digits) _ r (cell_lt a small), rest, index]

/-- Each cell of the table of the quotient digits is in the balanced alphabet `A_b`. -/
theorem quotientTable_range (radix : Radix base) (quotient : Fin rel.ring → Zq[X]) (x : ℕ) :
    -((base / 2 : ℕ) : ℤ) ≤ (rel.quotientTable base digits quotient x).valMinAbs ∧
      (rel.quotientTable base digits quotient x).valMinAbs ≤ ((base / 2 : ℕ) : ℤ) - 1 := by
  by_cases inside : x < ∑ r, rel.quotientCells digits r
  · obtain ⟨r, y, small, rfl⟩ := exists_offset (rel.quotientCells digits) inside
    unfold quotientTable
    rw [concat_apply (rel.quotientCells digits) _ r small]
    exact fieldDigit_balanced radix digits _ _
  · unfold quotientTable
    rw [concat_of_le (rel.quotientCells digits) _ (Nat.le_of_not_lt inside), ZMod.valMinAbs_zero]
    have two := radix.two
    constructor <;> omega

variable {base digits start}

/-- The quotient that the extended table has at the positions of a row is the quotient whose
digits it has, when each element of `Z_q` has `digits` balanced digits of the base `base` and the
quotient has a degree below `d_r`. -/
theorem quotient_extend (radix : Radix base)
    (covers : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    (inside : rel.quotientEnd digits start ≤ 2 ^ rel.vars) (w : Table)
    (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring)
    (degree : (quotient r).natDegree < (rel.row r).degree) :
    (rel.row r).quotient ((rel.withQuotients base digits start inside).quotient r)
        (rel.extend base digits start w quotient) = quotient r := by
  have each : ∀ j ∈ Finset.range (rel.row r).degree,
      C (∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
        rel.extend base digits start w quotient (rel.quotientCell digits start r j a)) * X ^ j =
          monomial j ((quotient r).coeff j) := fun j member => by
    have recomposed : ∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
        rel.extend base digits start w quotient (rel.quotientCell digits start r j a) =
          (quotient r).coeff j := by
      rw [Finset.sum_congr rfl fun a _ => by
        rw [rel.extend_cell base digits start w quotient r (Finset.mem_range.mp member) a],
        Fin.sum_univ_eq_sum_range
          (fun u => (base : Zq) ^ u * fieldDigit base digits ((quotient r).coeff j) u)]
      exact fieldDigit_sum_of_range radix (covers _).1 (covers _).2
    rw [recomposed, C_mul_X_pow_eq_monomial]
  show ∑ j ∈ Finset.range (rel.row r).degree,
    C (∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
      rel.extend base digits start w quotient (rel.quotientCell digits start r j a)) * X ^ j = _
  rw [Finset.sum_congr rfl each]
  exact (as_sum_range' (quotient r) _ degree).symm

/-- A row that reads cells before `start` has the same left side of (111) for the extended
table. -/
theorem lifted_extend (w : Table) (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring)
    (below : ∀ c j, (rel.row r).address c j < start) :
    (rel.row r).lifted (rel.extend base digits start w quotient) = (rel.row r).lifted w := by
  have reads : ∀ c, (rel.row r).read (rel.extend base digits start w quotient) c =
      (rel.row r).read w c := fun c =>
    congrArg ofCoeffs (funext fun j => rel.extend_left base digits start w quotient (below c j))
  unfold RingRow.lifted
  simp only [reads]

open Classical in
/-- A quotient of each ring row that holds for the table `w` (Akita Lemma 3.4), and 0 for a row
that does not hold. -/
noncomputable def quotients (w : Table) (r : Fin rel.ring) : Zq[X] :=
  if holds : (rel.row r).Holds w then (((rel.row r).holds_iff_quotient w).mp holds).choose else 0

theorem quotients_spec {w : Table} {r : Fin rel.ring} (holds : (rel.row r).Holds w) :
    (rel.quotients w r).natDegree < (rel.row r).degree ∧
      (rel.row r).lifted w = modulus (rel.row r).degree * rel.quotients w r := by
  have spec := (((rel.row r).holds_iff_quotient w).mp holds).choose_spec
  unfold quotients
  rw [dite_eq_left holds]
  refine ⟨?_, spec.2⟩
  have positive := NeZero.pos (rel.row r).degree
  by_cases zero : (((rel.row r).holds_iff_quotient w).mp holds).choose = 0
  · rw [zero, natDegree_zero]
    exact positive
  · have small := (natDegree_lt_iff_degree_lt zero).mpr spec.1
    omega

variable (base digits start) in
/-- The table `w` before the cell `start`, and then the digits of the quotients of its ring
rows: the witness of the quotient-lift route (Akita (76)). -/
noncomputable def completion (w : Table) : Table :=
  rel.extend base digits start w (rel.quotients w)

theorem completion_left (w : Table) {x : ℕ} (small : x < start) :
    rel.completion base digits start w x = w x :=
  rel.extend_left base digits start w _ small

/-- The digit range statement of Akita Section 6.1 for the completion of a table with the digit
range statement. -/
theorem completion_range (radix : Radix base) {μ : ℕ} {w : Table}
    (range : RangeHolds base μ w) : RangeHolds base μ (rel.completion base digits start w) := by
  refine (rangeHolds_iff radix.small μ _).mpr fun x inside => ?_
  by_cases small : x < start
  · rw [rel.completion_left w small]
    exact (rangeHolds_iff radix.small μ w).mp range x inside
  · obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt small)
    show _ ≤ (join start w _ (start + y)).valMinAbs ∧ (join start w _ (start + y)).valMinAbs ≤ _
    rw [join_right]
    exact rel.quotientTable_range base digits radix _ y

variable [Field E] [Algebra Zq E]

/-- The completion of a table with the relation satisfies the identities (111) and the scalar
row: the quotient-lift relation has a witness (Akita Lemma 7.1, item 1, with Lemma 3.4). `covers`:
each element of `Z_q` has `digits` balanced digits of the base `base`. `below`: each ring row
reads cells before `start`. `weightless`: the scalar row has the weight 0 from the cell
`start`. -/
theorem withQuotients_holds (radix : Radix base)
    (covers : ∀ x : Zq, -((base / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u ≤
        x.valMinAbs ∧
      x.valMinAbs ≤ (((base / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range digits, (base : ℤ) ^ u)
    (inside : rel.quotientEnd digits start ≤ 2 ^ rel.vars)
    (below : ∀ r c j, (rel.row r).address c j < start)
    (weightless : ∀ u, start ≤ u → rel.weight u = 0) {w : Table} (holds : rel.Holds w) :
    (rel.withQuotients base digits start inside).Holds (rel.completion base digits start w) := by
  refine ⟨fun r => ?_, ?_⟩
  · obtain ⟨degree, lifted⟩ := rel.quotients_spec (holds.1 r)
    show (rel.row r).lifted (rel.completion base digits start w) =
      modulus (rel.row r).degree *
        (rel.row r).quotient ((rel.withQuotients base digits start inside).quotient r)
          (rel.completion base digits start w)
    unfold completion
    rw [rel.lifted_extend w _ r (below r), rel.quotient_extend radix covers inside w _ r degree]
    exact lifted
  · show ∑ u ∈ Finset.range (2 ^ rel.vars),
      algebraMap Zq E (rel.completion base digits start w u) * rel.weight u = rel.value
    rw [← holds.2]
    refine Finset.sum_congr rfl fun u _ => ?_
    by_cases small : u < start
    · rw [rel.completion_left w small]
    · rw [weightless u (Nat.le_of_not_lt small), mul_zero, mul_zero]

end FlatRelation

end NightstreamFPrime.Spec.Compression.Akita
