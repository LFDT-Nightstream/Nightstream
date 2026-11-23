import NightstreamFPrime.Spec.Compression.Akita.BatchFlat
import NightstreamFPrime.Spec.Compression.Akita.LiftReduction
import NightstreamFPrime.Spec.Compression.Akita.Prover

/-! The quotient digits of the ring rows of one fold of Akita as segments of the witness of the
next level (ePrint 2026/1983, revision of 2026-11-23, Section 5.5 with (76), Section 7.3 with
(111) and (112), Lemma 3.4, and Appendix L.2 with (348)).

- `Gaps` has the places of the three quotient segments of Akita (76) in the flat witness: `Q̂_main`
  at the flat position `main`, after `t̂`, `Q̂_1` at `middle`, after the digits of `H_1`, and `Q̂_2`
  at `last`, after the digits of `H_2`. `Gaps.place` is the position in the witness of a flat
  position, `Gaps.slot` is the position of an entry of the quotient digits, and `Gaps.merge` is
  the witness of a flat table and a table of quotient digits. `Gaps.support` is the set `I_bin`
  of the witness.
- `FlatRelation.quotientCell` is the address (348) of one digit of one coefficient of the quotient
  `Q_r` of a ring row: the row `r` has `δ_quot·d_r` entries of the quotient digits, in the order
  of the rows, and in them the coefficient index changes first, then the digit.
- `FlatRelation.withQuotients` is the `LiftRelation` on the witness: the rows read the positions
  `Gaps.place`, and the quotient digits have the weights `b^u` of the balanced base `b`.
  `FlatRelation.holds_of_lift` gives the relation of the flat table that the witness has at the
  positions `Gaps.place`.
- `FlatRelation.quotients` has a quotient of each ring row that holds (Akita Lemma 3.4), and
  `FlatRelation.completion` is the witness with the digits of these quotients.
- `FlatRelation.withQuotients_holds` states that the completion of a table with the relation
  satisfies the identities (111): the relation `LiftRelation.Holds` has a witness.
  `FlatRelation.completion_range` is the digit range statement of the completion.

Each quotient has `δ_quot` digits for each coefficient, with one base for each row. The quotient
of the carrier row (102) is `Q_car = Σ_t β_t·Σ_j q_{t,j}·Y^j` of Akita (113): each plane `t` is one
ring row over `Z_q` with its quotient `Σ_j q_{t,j}·Y^j`, and `Join` joins the planes into one row
over `E`. The pinned code has the digits of all planes for each digit index
(`crates/akita-params/src/witness.rs` of the Akita code), and this file has the order plane,
digit, coefficient in the same cells. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

/-- The places of the quotient segments of Akita (76) in the flat witness: `mainCells` entries
of the quotient digits at the flat position `main`, `middleCells` entries at `middle`, and the
other entries at `last`. -/
structure Gaps where
  main : ℕ
  middle : ℕ
  last : ℕ
  mainCells : ℕ
  middleCells : ℕ
  ordered : main ≤ middle ∧ middle ≤ last

namespace Gaps

variable (G : Gaps)

/-- The position in the witness of the flat position `x`, for `x < last`. -/
def place (x : ℕ) : ℕ :=
  x + (if G.main ≤ x then G.mainCells else 0) + (if G.middle ≤ x then G.middleCells else 0)

/-- The position in the witness of entry `y` of the quotient digits. -/
def slot (y : ℕ) : ℕ :=
  if y < G.mainCells then G.main + y
  else if y < G.mainCells + G.middleCells then G.middle + y else G.last + y

/-- The witness with the cells of the flat table `w` at the positions `place` and the cells of
the table `q` of the quotient digits at the positions `slot`. -/
def merge (w q : Table) : Table := fun z =>
  if z < G.main then w z
  else if z < G.main + G.mainCells then q (z - G.main)
  else if z < G.middle + G.mainCells then w (z - G.mainCells)
  else if z < G.middle + G.mainCells + G.middleCells then q (z - G.middle)
  else if z < G.last + G.mainCells + G.middleCells then w (z - G.mainCells - G.middleCells)
  else q (z - G.last)

theorem place_of_lt {x : ℕ} (small : x < G.main) : G.place x = x := by
  have ordered := G.ordered
  unfold place
  split_ifs <;> omega

theorem place_lt {x : ℕ} (small : x < G.last) :
    G.place x < G.last + G.mainCells + G.middleCells := by
  unfold place
  split <;> split <;> omega

theorem merge_place (w q : Table) {x : ℕ} (small : x < G.last) : G.merge w q (G.place x) = w x := by
  have ordered := G.ordered
  unfold merge place
  split_ifs <;> first | omega | exact congrArg _ (by omega)

theorem merge_slot (w q : Table) (y : ℕ) : G.merge w q (G.slot y) = q y := by
  have ordered := G.ordered
  unfold merge slot
  split_ifs <;> first | omega | exact congrArg _ (by omega)

theorem slot_lt {y total : ℕ} (small : y < total) : G.slot y < G.last + total := by
  have ordered := G.ordered
  unfold slot
  split_ifs <;> omega

/-- The witness from the position `last + mainCells + middleCells` is the table of the quotient
digits from its entry `mainCells + middleCells`. -/
theorem merge_of_le (w q : Table) {z : ℕ} (large : G.last + G.mainCells + G.middleCells ≤ z) :
    G.merge w q z = q (z - G.last) := by
  have ordered := G.ordered
  unfold merge
  split_ifs <;> first | omega | rfl

theorem merge_zero : G.merge 0 0 = 0 :=
  funext fun z => by
    unfold merge
    split_ifs <;> rfl

/-- Each cell of the witness is a cell of `w` at a position up to its own, or a cell of `q`. -/
theorem merge_cases (w q : Table) (z : ℕ) :
    (∃ x ≤ z, G.merge w q z = w x) ∨ ∃ y, G.merge w q z = q y := by
  unfold merge
  split_ifs
  · exact Or.inl ⟨z, le_rfl, rfl⟩
  · exact Or.inr ⟨_, rfl⟩
  · exact Or.inl ⟨_, Nat.sub_le _ _, rfl⟩
  · exact Or.inr ⟨_, rfl⟩
  · exact Or.inl ⟨_, by omega, rfl⟩
  · exact Or.inr ⟨_, rfl⟩

/-- The set `I_bin` of the witness: the compression digits, which are the flat positions from
`main` to `last` at the positions `place` (Akita Remark 6.2 and (76)). -/
def support (z : ℕ) : Prop :=
  G.main + G.mainCells ≤ z ∧ z < G.middle + G.mainCells ∨
    G.middle + G.mainCells + G.middleCells ≤ z ∧ z < G.last + G.mainCells + G.middleCells

instance : DecidablePred G.support := fun _ => inferInstanceAs (Decidable (_ ∨ _))

theorem support_place {x : ℕ} (low : G.main ≤ x) (high : x < G.last) : G.support (G.place x) := by
  unfold support place
  split <;> split <;> omega

theorem support_cases {z : ℕ} (member : G.support z) :
    ∃ x, G.main ≤ x ∧ x < G.last ∧ G.place x = z := by
  have ordered := G.ordered
  unfold support at member
  rcases member with ⟨low, high⟩ | ⟨low, high⟩
  · refine ⟨z - G.mainCells, by omega, by omega, ?_⟩
    unfold place
    split_ifs <;> omega
  · refine ⟨z - G.mainCells - G.middleCells, by omega, by omega, ?_⟩
    unfold place
    split_ifs <;> omega

/-- A fold with `last ≤ main` has no compression digits. -/
theorem support_empty (empty : G.last ≤ G.main) (z : ℕ) : ¬ G.support z := by
  have ordered := G.ordered
  unfold support
  omega

end Gaps

namespace RingRow

theorem read_relocate (row : RingRow) (move : ℕ → ℕ) (w : Table) (c : Fin row.occurrences) :
    (row.relocate move).read w c = row.read (fun x => w (move x)) c :=
  congrArg ofCoeffs (funext fun j => elim_map (row.address c j) move w)

theorem lifted_relocate (row : RingRow) (move : ℕ → ℕ) (w : Table) :
    (row.relocate move).lifted w = row.lifted fun x => w (move x) := by
  unfold lifted
  exact congrArg (· - rep row.target) (Finset.sum_congr rfl fun c _ => by
    rw [row.read_relocate move w c]
    rfl)

end RingRow

namespace FlatRelation

variable {E : Type} [Field E] [Algebra Zq E] (rel : FlatRelation E) (G : Gaps)
  (base digits : ℕ)

/-- The number of entries of the digits of the quotient of the row `r`: `δ_quot·d_r`. -/
def quotientCells (r : Fin rel.ring) : ℕ := digits * (rel.row r).degree

/-- The end of the witness: the flat positions before `last` and the quotient digits. -/
def quotientEnd : ℕ := G.last + ∑ r, rel.quotientCells digits r

/-- The position of digit `a` of coefficient `j` of the quotient of the row `r`: the local
position `a·d_r + j` of Akita (348) for one base-field coordinate in a coefficient, in the entries
of the row, at the position `slot` of the witness. -/
def quotientCell (r : Fin rel.ring) (j : ℕ) (a : Fin digits) : ℕ :=
  G.slot (offset (rel.quotientCells digits) r + (a * (rel.row r).degree + j))

theorem quotientCell_lt (r : Fin rel.ring) {j : ℕ} (small : j < (rel.row r).degree)
    (a : Fin digits) : rel.quotientCell G digits r j a < rel.quotientEnd G digits := by
  have cell := cell_lt a small
  have last := offset_add_le (rel.quotientCells digits) r
  have size : rel.quotientCells digits r = digits * (rel.row r).degree := rfl
  exact G.slot_lt (by omega)

/-- The quotient digits have `δ_quot·Σ_r d_r` entries. -/
theorem quotientEnd_eq : rel.quotientEnd G digits = G.last + digits * ∑ r, (rel.row r).degree := by
  unfold quotientEnd quotientCells
  rw [Finset.mul_sum]

/-- The relation on the witness of the quotient-lift route (Akita Section 7.3 and (76)): the ring
rows read the positions `place` of the flat positions, and the quotient of each ring row has
`digits` digits of the base `base` for each coefficient, at the positions `quotientCell`.
`below`: each ring row reads flat positions before `last`. `counts`: the first two quotient
segments have at most the entries of the quotient digits. -/
def withQuotients (below : ∀ r c j x, (rel.row r).address c j = some x → x < G.last)
    (counts : G.mainCells + G.middleCells ≤ ∑ r, rel.quotientCells digits r)
    (inside : rel.quotientEnd G digits ≤ 2 ^ rel.vars) : LiftRelation E where
  vars := rel.vars
  low := rel.low
  ring := rel.ring
  row r := (rel.row r).relocate G.place
  inside r c j x _ read := by
    cases h : (rel.row r).address c j with
    | none => simp [RingRow.relocate, h] at read
    | some y =>
        simp only [RingRow.relocate, h, Option.map_some, Option.some.injEq] at read
        have moved := G.place_lt (below r c j y h)
        unfold quotientEnd at inside
        omega
  join := rel.join
  independent := rel.independent
  weight := rel.weight
  value := rel.value
  quotient r := ⟨digits, fun a => (base : Zq) ^ (a : ℕ), rel.quotientCell G digits r⟩
  quotientInside r _ a small := (rel.quotientCell_lt G digits r small a).trans_le inside

/-- The table of the digits of the quotients `quotient r`, in the order of the rows. -/
noncomputable def quotientTable (quotient : Fin rel.ring → Zq[X]) : Table :=
  concat (rel.quotientCells digits) fun r y =>
    fieldDigit base digits ((quotient r).coeff (y % (rel.row r).degree))
      (y / (rel.row r).degree)

/-- The witness with the flat table `w` and the digits of the quotients. -/
noncomputable def extend (w : Table) (quotient : Fin rel.ring → Zq[X]) : Table :=
  G.merge w (rel.quotientTable base digits quotient)

theorem extend_cell (w : Table) (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring) {j : ℕ}
    (small : j < (rel.row r).degree) (a : Fin digits) :
    rel.extend G base digits w quotient (rel.quotientCell G digits r j a) =
      fieldDigit base digits ((quotient r).coeff j) a := by
  have rest : (a * (rel.row r).degree + j) % (rel.row r).degree = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]
  have index : (a * (rel.row r).degree + j) / (rel.row r).degree = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (NeZero.pos _), Nat.div_eq_of_lt small, zero_add]
  unfold extend quotientCell quotientTable
  rw [Gaps.merge_slot, concat_apply (rel.quotientCells digits) _ r (cell_lt a small), rest, index]

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

variable {G base digits}

/-- The quotient that the witness has at the positions of a row is the quotient whose digits it
has, when `q ≤ base^digits` and the quotient has a degree below `d_r`: the digits of an element
of `Z_q` are those of its lift of Akita Section 3.3 (`lift`, `fieldDigit_sum`), which is the lift
of the pinned code (`decompose_centering_threshold` in
`crates/akita-algebra/src/ring/cyclotomic/decomposition.rs` of the Akita code). -/
theorem quotient_extend (radix : Radix base)
    (covers : fieldModulus ≤ base ^ digits)
    (below : ∀ r c j x, (rel.row r).address c j = some x → x < G.last)
    (counts : G.mainCells + G.middleCells ≤ ∑ r, rel.quotientCells digits r)
    (inside : rel.quotientEnd G digits ≤ 2 ^ rel.vars) (w : Table)
    (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring)
    (degree : (quotient r).natDegree < (rel.row r).degree) :
    (rel.row r).quotient ((rel.withQuotients G base digits below counts inside).quotient r)
        (rel.extend G base digits w quotient) = quotient r := by
  have each : ∀ j ∈ Finset.range (rel.row r).degree,
      C (∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
        rel.extend G base digits w quotient (rel.quotientCell G digits r j a)) * X ^ j =
          monomial j ((quotient r).coeff j) := fun j member => by
    have recomposed : ∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
        rel.extend G base digits w quotient (rel.quotientCell G digits r j a) =
          (quotient r).coeff j := by
      rw [Finset.sum_congr rfl fun a _ => by
        rw [rel.extend_cell G base digits w quotient r (Finset.mem_range.mp member) a],
        Fin.sum_univ_eq_sum_range
          (fun u => (base : Zq) ^ u * fieldDigit base digits ((quotient r).coeff j) u)]
      exact fieldDigit_sum ⟨radix.even, radix.two, radix.small, covers⟩ _
    rw [recomposed, C_mul_X_pow_eq_monomial]
  show ∑ j ∈ Finset.range (rel.row r).degree,
    C (∑ a : Fin digits, (base : Zq) ^ (a : ℕ) *
      rel.extend G base digits w quotient (rel.quotientCell G digits r j a)) * X ^ j = _
  rw [Finset.sum_congr rfl each]
  exact (as_sum_range' (quotient r) _ degree).symm

/-- A row that reads flat positions before `last` has, on the witness at the positions `place`,
the left side of (111) of the flat table. -/
theorem lifted_extend (w : Table) (quotient : Fin rel.ring → Zq[X]) (r : Fin rel.ring)
    (below : ∀ c j x, (rel.row r).address c j = some x → x < G.last) :
    ((rel.row r).relocate G.place).lifted (rel.extend G base digits w quotient) =
      (rel.row r).lifted w := by
  rw [RingRow.lifted_relocate]
  have reads : ∀ c, (rel.row r).read (fun x => rel.extend G base digits w quotient (G.place x)) c =
      (rel.row r).read w c := fun c =>
    congrArg ofCoeffs (funext fun j => by
      cases h : (rel.row r).address c j with
      | none => rfl
      | some x => exact G.merge_place w _ (below c j x h))
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

variable (G base digits) in
/-- The witness of the quotient-lift route for the flat table `w` (Akita (76)): the cells of `w`
at the positions `place`, and the digits of the quotients of its ring rows at the positions
`slot`. -/
noncomputable def completion (w : Table) : Table :=
  rel.extend G base digits w (rel.quotients w)

theorem completion_place (w : Table) {x : ℕ} (small : x < G.last) :
    rel.completion G base digits w (G.place x) = w x :=
  G.merge_place w _ small

/-- Each cell from the end of the witness is 0. -/
theorem completion_of_le (counts : G.mainCells + G.middleCells ≤ ∑ r, rel.quotientCells digits r)
    (w : Table) {x : ℕ} (large : rel.quotientEnd G digits ≤ x) :
    rel.completion G base digits w x = 0 := by
  unfold quotientEnd at large
  show G.merge w _ x = 0
  rw [G.merge_of_le w _ (by omega)]
  exact concat_of_le _ _ (by omega)

/-- The digit range statement of Akita Section 6.1 for the completion of a table with the digit
range statement. -/
theorem completion_range (radix : Radix base) {μ : ℕ} {w : Table}
    (range : RangeHolds base μ w) : RangeHolds base μ (rel.completion G base digits w) := by
  refine (rangeHolds_iff radix.small μ _).mpr fun z inside => ?_
  rcases G.merge_cases w (rel.quotientTable base digits (rel.quotients w)) z with
    ⟨x, le, cell⟩ | ⟨y, cell⟩
  · show _ ≤ (G.merge w _ z).valMinAbs ∧ (G.merge w _ z).valMinAbs ≤ _
    rw [cell]
    exact (rangeHolds_iff radix.small μ w).mp range x (lt_of_le_of_lt le inside)
  · show _ ≤ (G.merge w _ z).valMinAbs ∧ (G.merge w _ z).valMinAbs ≤ _
    rw [cell]
    exact rel.quotientTable_range base digits radix _ y

/-- A witness of the identities (111) of `withQuotients` has the relation for the flat table at
the positions `place` (Akita Lemma 3.4). `weightless`: the scalar row has the weight 0 from the
flat position `main`. -/
theorem holds_of_lift {below : ∀ r c j x, (rel.row r).address c j = some x → x < G.last}
    {counts : G.mainCells + G.middleCells ≤ ∑ r, rel.quotientCells digits r}
    {inside : rel.quotientEnd G digits ≤ 2 ^ rel.vars}
    (weightless : ∀ u, G.main ≤ u → rel.weight u = 0) {T : Table}
    (holds : (rel.withQuotients G base digits below counts inside).Holds T) :
    rel.Holds fun x => T (G.place x) := by
  have flat := LiftRelation.holds_flat _ holds
  refine ⟨fun r => (RingRow.relocate_holds _ _ _).mp (flat.1 r), ?_⟩
  refine Eq.trans (Finset.sum_congr rfl fun u _ => ?_) flat.2
  show algebraMap Zq E (T (G.place u)) * rel.weight u = algebraMap Zq E (T u) * rel.weight u
  by_cases small : u < G.main
  · rw [G.place_of_lt small]
  · rw [weightless u (Nat.le_of_not_lt small), mul_zero, mul_zero]

/-- The completion of a table with the relation satisfies the identities (111) and the scalar
row: the quotient-lift relation has a witness (Akita Lemma 7.1, item 1, with Lemma 3.4). `covers`:
`q ≤ base^digits`, so each element of `Z_q` has `digits` balanced digits of the base `base`. `weightless`: the scalar
row has the weight 0 from the flat position `main`. -/
theorem withQuotients_holds (radix : Radix base)
    (covers : fieldModulus ≤ base ^ digits)
    (below : ∀ r c j x, (rel.row r).address c j = some x → x < G.last)
    (counts : G.mainCells + G.middleCells ≤ ∑ r, rel.quotientCells digits r)
    (inside : rel.quotientEnd G digits ≤ 2 ^ rel.vars)
    (weightless : ∀ u, G.main ≤ u → rel.weight u = 0) {w : Table} (holds : rel.Holds w) :
    (rel.withQuotients G base digits below counts inside).Holds
      (rel.completion G base digits w) := by
  refine ⟨fun r => ?_, ?_⟩
  · obtain ⟨degree, lifted⟩ := rel.quotients_spec (holds.1 r)
    show ((rel.row r).relocate G.place).lifted (rel.completion G base digits w) =
      modulus (rel.row r).degree *
        (rel.row r).quotient ((rel.withQuotients G base digits below counts inside).quotient r)
          (rel.completion G base digits w)
    unfold completion
    rw [rel.lifted_extend w _ r (below r),
      rel.quotient_extend radix covers below counts inside w _ r degree]
    exact lifted
  · show ∑ u ∈ Finset.range (2 ^ rel.vars),
      algebraMap Zq E (rel.completion G base digits w u) * rel.weight u = rel.value
    rw [← holds.2]
    refine Finset.sum_congr rfl fun u _ => ?_
    by_cases small : u < G.main
    · show algebraMap Zq E (G.merge w _ u) * rel.weight u = _
      rw [show G.merge w (rel.quotientTable base digits (rel.quotients w)) u = w u from
        ite_eq_left small]
    · rw [weightless u (Nat.le_of_not_lt small), mul_zero, mul_zero]

end FlatRelation

end NightstreamFPrime.Spec.Compression.Akita
