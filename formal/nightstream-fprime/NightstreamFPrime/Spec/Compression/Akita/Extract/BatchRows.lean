import NightstreamFPrime.Spec.Compression.Akita.BatchFlat
import NightstreamFPrime.Spec.Compression.Akita.Extract.FlatSource

/-! The relation of one fold of Akita with opening groups on the flat witness, for one claim in
each group, and its link to the exact branch relation (ePrint 2026/1983, revision of 2026-11-23,
Section 7.2 with Figure 5, Section 7.3, and the hypothesis of Corollary 10.32).

- `gather` puts the rows of the groups of one family in the order `groupOrder` of the pinned
  code: the last group first.
- `Batch.flatRows` has the ring rows in the order of the pinned code
  (`crates/akita-params/src/layout/relation_rhs_layout.rs` of the Akita code), which is the order
  of the families of Akita Figure 5 with the last group first in each family: for each group the
  planes of the row (102), the rows (104) and the rows (105) of each slice (`Batch.groupRows`),
  then the rows (106) of the shared matrix `D`, and the rows (47) in increasing depth, with the
  rows of `F_a` of each group before the rows of `H_a`. A row of the group `g` is the row of
  `FlatFold.lean` for `Batch.opening g`, with its addresses moved by `Batch.relocate g`.
- `Batch.scalarWeight` has the weights of the scalar row (100) with the weights `ϑ_g`.
- `Batch.flatRelation` is the `FlatRelation` of the fold, and `Batch.flatRelation_holds_iff`
  states that `Batch.Witness.flat` satisfies it when the digit vectors satisfy the rows of
  `Batch.Witness.Satisfies`, and only then.
- `Batch.Witness.Alphabets` is the alphabet part of the branch relation.
  `Batch.Witness.alphabets_of_flat` gives it from the digit range statement and the binary
  statement of the flat witness, `Batch.Witness.satisfies_of_flat` gives the branch relation, and
  `Batch.Witness.Satisfies.flat` is the honest direction.
- `Batch.flatReduction` is the quotient-free reduction, and `Batch.flatFold_satisfies` is
  `foldCheck_avoids` for `Batch.Witness.flat` and `Batch.Witness.Satisfies`.

The one range check of a fold has the common base `b⋆` of the fold (Akita Section 5.5, after
(75)). A `Shape` has its own certified base, so the hypothesis `uniform` states that each group
has the certified base `base` of the checks. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

section Moves

theorem RingRow.Below.relocate {row : RingRow} {bound other : ℕ} {move : ℕ → ℕ}
    (below : row.Below bound) (moves : ∀ x < bound, move x < other) :
    (row.relocate move).Below other := fun c j x read => by
  cases h : row.address c j with
  | none => simp [RingRow.relocate, h] at read
  | some y =>
      simp only [RingRow.relocate, h, Option.map_some, Option.some.injEq] at read
      exact read ▸ moves _ (below c j y h)

variable {n : ℕ} {count : Fin n → ℕ}

/-- The rows of one family for the groups, with the groups in the order `groupOrder` of the pinned
code, the last group first, and increasing slice and matrix-row indices in a group (Akita
Figure 5 has "increasing group, slice, and matrix-row indices"). -/
def gather (rows : (g : Fin n) → Fin (count g) → RingRow) : Fin (∑ g, count g) → RingRow :=
  fun r => rows (place count r).1 (place count r).2

theorem forall_gather (P : RingRow → Prop) (rows : (g : Fin n) → Fin (count g) → RingRow) :
    (∀ r, P (gather rows r)) ↔ ∀ g i, P (rows g i) := by
  constructor
  · intro all g i
    have entry := all ((place count).symm ⟨g, i⟩)
    unfold gather at entry
    rwa [Equiv.apply_symm_apply] at entry
  · exact fun all r => all _ _

/-- A sum over the rows of one family for the groups is the sum over the groups of the sums over
the rows of each group. -/
theorem sum_gather {M : Type} [AddCommMonoid M] (f : RingRow → M)
    (rows : (g : Fin n) → Fin (count g) → RingRow) :
    ∑ r, f (gather rows r) = ∑ g, ∑ i, f (rows g i) :=
  ((place count).sum_comp fun x : (g : Fin n) × Fin (count g) => f (rows x.1 x.2)).trans
    (Fintype.sum_sigma _)

/-- A sum over the cells of a table, with weights that are moved by `move`, is the sum over the
positions before the move. A position with a nonzero weight moves to a cell of the table. -/
theorem sum_mul_relocate {E : Type} [Semiring E] (cells : ℕ) (move : ℕ → ℕ) (weight f : ℕ → E)
    (inside : ∀ y < cells, weight y ≠ 0 → move y < cells) :
    ∑ x ∈ Finset.range cells, f x *
        ∑ y ∈ Finset.range cells, (if move y = x then weight y else 0) =
      ∑ y ∈ Finset.range cells, f (move y) * weight y := by
  simp only [Finset.mul_sum, mul_ite, mul_zero]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y member => ?_
  rw [Finset.sum_ite_eq]
  split
  · rfl
  · rename_i outside
    have zero : weight y = 0 := by
      by_contra nonzero
      exact outside (Finset.mem_range.mpr (inside y (Finset.mem_range.mp member) nonzero))
    rw [zero, mul_zero]

end Moves

namespace Batch

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E]

section Rows

/-- Entry `j` of `D·concat(ê_0, …, ê_{G−1})` (Akita (106)): the column `(g, col)` reads the
element `col` of `ê_g` in the main segment of the group `g`. -/
def openForm (key : B.OpenKey) (j : Fin (1 * B.openRank)) : Form B.openDegree :=
  Form.ofMatrix key.matrix
    (fun column k =>
      ((B.opening column.1).partialCell (B.fold column.1) column.2 k).map (B.relocate column.1))
    (finProdFinEquiv.symm j).2

/-- The weight of the cell `x` in the scalar row (100) of Akita Figure 5 for the groups: the sum
over the groups of `ϑ_g` times the weight `scalarWeight` of the group at the positions that move
to `x`. -/
noncomputable def scalarWeight (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (μ x : ℕ) : E :=
  ∑ g, st.weight g * ∑ y ∈ Finset.range (2 ^ μ),
    if B.relocate g y = x then
      Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g) y
    else 0

/-- `log_2` of the coefficient block of the relation addresses of the pinned code: the least of
the degree of `D` and, for each group, of the degrees of `A` and `B` and of the degree of the
challenge subring (`relation_coefficient_block_len` of `crates/akita-params/src/schedule.rs` of
the Akita code). The range point has its last `low` coordinates first (`rangePoint`). -/
def low : ℕ :=
  Nat.log 2 ((List.ofFn fun g => min (min (B.shape g).innerDegree (B.shape g).outerDegree)
    (B.carrierDegree g)).foldl min B.openDegree)

/-- The number of ring rows of the group `g` after its carrier row and before the rows of
`D`. -/
def groupTail (g : Fin n) : ℕ :=
  (B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank

/-- The number of ring rows over `Z_q` of the group `g` before the rows of `D`: each plane of the
carrier row is one row. -/
def groupPlaneRows (g : Fin n) : ℕ := B.planes g + B.groupTail g

/-- The number of rows over `E` of the group `g` before the rows of `D`: the carrier row (102)
is one row. -/
def groupCount (g : Fin n) : ℕ := 1 + B.groupTail g

/-- The number of ring rows of one fold with opening groups after the rows of the groups. -/
def restCount : ℕ :=
  1 * B.openRank + (∑ g, firstRank (B.shape g).chain + (firstRank B.chain +
    (∑ g, secondRank (B.shape g).chain + secondRank B.chain)))

/-- The number of ring rows over `Z_q` of one fold with opening groups. -/
def planeRows : ℕ := ∑ g, B.groupPlaneRows g + B.restCount

/-- The number of rows over `E` of one fold with opening groups before the scalar row, which is
the number of rows of the relation matrix of the pinned code (`relation_matrix_row_count` in
`crates/akita-params/src/layout/params.rs` of the Akita code). -/
def rowCount : ℕ := ∑ g, B.groupCount g + B.restCount

/-- `d_max`, the largest ring degree of a row (Akita Theorem 7.4). -/
def largestDegree : ℕ :=
  max (Finset.univ.sup fun g => (B.opening g).largestDegree)
    (max B.openDegree (chainDegree B.chain))

theorem group_le (g : Fin n) : (B.opening g).largestDegree ≤ B.largestDegree :=
  le_max_of_le_left (Finset.le_sup (f := fun g => (B.opening g).largestDegree)
    (Finset.mem_univ g))

variable (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
  (challenge : B.Challenge)

/-- The ring rows of the group `g` before the rows of `D`: the planes of the carrier row (102),
the rows (104) of `A_g`, and for each slice the rows (105) of `B_g`, with the addresses moved to
the flat witness of the fold. -/
noncomputable def groupRows (g : Fin n) : Fin (B.groupPlaneRows g) → RingRow :=
  Fin.append
    (fun t => (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g) t).relocate
      (B.relocate g))
    (Fin.append
      (fun a => (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).relocate
        (B.relocate g))
      (fun j => (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
        ((B.opening g).firstCell (B.fold g)) (st.payload g) j).relocate (B.relocate g)))

/-- The ring rows of one fold with opening groups in the order of the pinned code, with the groups
in the order `groupOrder`: for each group the rows `groupRows`, then the rows (106) of `D`, and the rows (47): for each
group the rows of `F_1`, the rows of `H_1`, for each group the rows of `F_2`, and the rows of
`H_2`. -/
noncomputable def flatRows : Fin B.planeRows → RingRow :=
  Fin.append (gather (B.groupRows β st challenge))
    (Fin.append (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload)
      (Fin.append
        (gather (count := fun g => firstRank (B.shape g).chain) fun g r =>
          (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g))
            ((B.opening g).secondCell (B.fold g)) r).relocate (B.relocate g))
        (Fin.append (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell)
          (Fin.append
            (gather (count := fun g => secondRank (B.shape g).chain) fun g r =>
              (secondRow (B.shape g).chain (st.key g).chain
                ((B.opening g).secondCell (B.fold g)) (st.payload g) r).relocate
                  (B.relocate g))
            (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload)))))

theorem openForm_below (key : B.OpenKey) (j : Fin (1 * B.openRank)) :
    (B.openForm key j).Below B.cells := fun column k x read => by
  cases h : (B.opening column.1).partialCell (B.fold column.1) column.2 k with
  | none => simp [openForm, Form.ofMatrix, h] at read
  | some y =>
      simp only [openForm, Form.ofMatrix, h, Option.map_some, Option.some.injEq] at read
      exact read ▸ B.relocate_lt column.1
        ((B.opening column.1).partialCell_lt (B.fold column.1) column.2 k h)

theorem flatRows_below (r : Fin B.planeRows) : (B.flatRows β st challenge r).Below B.cells := by
  have moves : ∀ g, ∀ x < (B.opening g).cells (B.fold g), B.relocate g x < B.cells :=
    fun g _ small => B.relocate_lt g small
  revert r
  refine (forall_append (RingRow.Below · B.cells) _ _).mpr
    ⟨(forall_gather (RingRow.Below · B.cells) _).mpr fun g =>
      (forall_append (RingRow.Below · B.cells) _ _).mpr ⟨fun t => ?_,
        (forall_append (RingRow.Below · B.cells) _ _).mpr ⟨fun a => ?_, fun j => ?_⟩⟩,
    (forall_append (RingRow.Below · B.cells) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · B.cells) _ _).mpr
    ⟨(forall_gather (RingRow.Below · B.cells) _).mpr fun g r => ?_,
    (forall_append (RingRow.Below · B.cells) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · B.cells) _ _).mpr
    ⟨(forall_gather (RingRow.Below · B.cells) _).mpr fun g r => ?_, ?_⟩⟩⟩⟩⟩
  · exact RingRow.Below.relocate ((Form.Below.sub
      (f := partialForm (B.opening g) (B.fold g) (challenge g) t)
      (g := contractForm (B.opening g) (B.fold g) (β g) (st.point g) t)
      (fun _ _ _ => (B.opening g).partialPosition_lt (B.fold g) _ _ _)
      (fun _ _ _ => (B.opening g).responseCell_lt (B.fold g) _ _)).row 0) (moves g)
  · exact RingRow.Below.relocate ((Form.Below.sub
      (f := responseForm (B.fold g) (st.key g) a)
      (g := imageForm (B.opening g) (B.fold g) (challenge g) a)
      (fun _ _ _ => (B.opening g).responseCell_lt (B.fold g) _ _)
      (fun _ _ _ => (B.opening g).innerCell_lt (B.fold g) _ _ _ _)).row 0) (moves g)
  · exact RingRow.Below.relocate (imageRow_below (B.shape g).chain _ _ _
      (fun _ => Form.ofMatrix_below _
        (fun _ _ _ => (B.opening g).sliceCell_lt (B.fold g) _ _ _) _)
      (fun _ _ => (B.opening g).firstCell_lt (B.fold g) _) j) (moves g)
  · exact imageRow_below B.chain _ _ _ (B.openForm_below st.openKey)
      (fun _ _ => B.openFirstCell_lt _)
  · exact RingRow.Below.relocate (firstRow_below (B.shape g).chain _ _ _
      (fun _ _ => (B.opening g).firstCell_lt (B.fold g) _)
      (fun _ _ => (B.opening g).secondCell_lt (B.fold g) _) r) (moves g)
  · exact firstRow_below B.chain _ _ _ (fun _ _ => B.openFirstCell_lt _)
      (fun _ _ => B.openSecondCell_lt _)
  · exact RingRow.Below.relocate (secondRow_below (B.shape g).chain _ _ _
      (fun _ _ => (B.opening g).secondCell_lt (B.fold g) _) r) (moves g)
  · exact secondRow_below B.chain _ _ _ (fun _ _ => B.openSecondCell_lt _)

theorem flatRows_degree_le (r : Fin B.planeRows) :
    (B.flatRows β st challenge r).degree ≤ B.largestDegree := by
  revert r
  refine (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr
    ⟨(forall_gather (RingRow.degree · ≤ B.largestDegree) _).mpr fun g =>
      (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr ⟨fun t => ?_,
        (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr ⟨fun a => ?_, fun j => ?_⟩⟩,
    (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr ⟨fun j => ?_,
    (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr
    ⟨(forall_gather (RingRow.degree · ≤ B.largestDegree) _).mpr fun g r => ?_,
    (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr ⟨fun r => ?_,
    (forall_append (RingRow.degree · ≤ B.largestDegree) _ _).mpr
    ⟨(forall_gather (RingRow.degree · ≤ B.largestDegree) _).mpr fun g r => ?_,
      fun r => ?_⟩⟩⟩⟩⟩
  · exact le_trans (le_max_of_le_left
      (le_max_left (B.opening g).carrierDegree (B.shape g).innerDegree)) (B.group_le g)
  · exact le_trans (le_max_of_le_left
      (le_max_right (B.opening g).carrierDegree (B.shape g).innerDegree)) (B.group_le g)
  · exact le_trans ((imageRow_degree (B.shape g).chain _ _ _ j).le.trans
      (le_max_of_le_right (le_max_of_le_left (le_max_left _ _)))) (B.group_le g)
  · exact (imageRow_degree B.chain _ _ _ j).le.trans (le_max_of_le_right (le_max_left _ _))
  · exact le_trans ((firstRow_degree_le (B.shape g).chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_left _ _)))) (B.group_le g)
  · exact (firstRow_degree_le B.chain _ _ _ r).trans (le_max_of_le_right (le_max_right _ _))
  · exact le_trans ((secondRow_degree_le (B.shape g).chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_left _ _)))) (B.group_le g)
  · exact (secondRow_degree_le B.chain _ _ _ r).trans (le_max_of_le_right (le_max_right _ _))

/-- The sum of the ring degrees of the rows `groupRows` of the group `g`. -/
def groupDegrees (g : Fin n) : ℕ :=
  B.planes g * B.carrierDegree g + ((B.shape g).innerRank * (B.shape g).innerDegree +
    (B.shape g).slices * (B.shape g).outerRank * (B.shape g).outerDegree)

/-- The sum of the ring degrees of the rows of one fold with opening groups, in the order of
`Batch.flatRows`. -/
def degreeSum : ℕ :=
  ∑ g, B.groupDegrees g + (1 * B.openRank * B.openDegree +
    (∑ g, firstRank (B.shape g).chain * firstDegree (B.shape g).chain +
      (firstRank B.chain * firstDegree B.chain +
        (∑ g, secondRank (B.shape g).chain * secondDegree (B.shape g).chain +
          secondRank B.chain * secondDegree B.chain))))

theorem flatRows_degree_sum : ∑ r, (B.flatRows β st challenge r).degree = B.degreeSum := by
  have moved : ∀ (row : RingRow) (move : ℕ → ℕ), (row.relocate move).degree = row.degree :=
    fun _ _ => rfl
  have planes : ∀ g t, (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g)
      t).degree = B.carrierDegree g := fun _ _ => rfl
  have inner : ∀ g a, (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).degree =
      (B.shape g).innerDegree := fun _ _ => rfl
  unfold flatRows groupRows degreeSum groupDegrees planeRows groupPlaneRows groupTail restCount
  simp only [sum_append RingRow.degree, sum_gather RingRow.degree, moved, planes, inner,
    imageRow_degree, firstRow_degree, secondRow_degree, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, smul_eq_mul]

/-- The relation of one fold with opening groups and coefficient packing on the flat witness
`Batch.Witness.flat`, for the fold challenges `c_{g,i}` (Akita Figure 5 for one claim in each
group, on the quotient-free route): the ring rows `flatRows` and the scalar row (100). The cube
has `μ` variables, with `cells < 2^μ`. -/
noncomputable def flatRelation (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) : FlatRelation E where
  vars := μ
  low := B.low
  ring := B.planeRows
  row := B.flatRows β st challenge
  inside r c j _ _ read := (B.flatRows_below β st challenge r c j _ read).trans_le fits
  join := (Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
    (Join.plain E B.restCount)
  independent := Join.append_independent
    (Join.gather_independent fun g => Join.append_independent
      (Join.all_independent (β g).linearIndependent)
      (Join.plain_independent (algebraMap Zq E).injective _))
    (Join.plain_independent (algebraMap Zq E).injective _)
  weight := B.scalarWeight β st μ
  value := st.value

/-- The weight of the scalar row is 0 from the offset `chainStart`. -/
theorem scalarWeight_of_le (μ : ℕ) {x : ℕ} (large : B.chainStart ≤ x) :
    B.scalarWeight β st μ x = 0 :=
  Finset.sum_eq_zero fun g _ => by
    rw [Finset.sum_eq_zero fun y _ => ?_, mul_zero]
    split
    · rename_i moved
      refine Akita.scalarWeight_of_le (B.opening g) (B.fold g) (β g) (st.point g) ?_
      by_contra small
      have below := B.relocate_lt_chainStart g (Nat.lt_of_not_le small)
      omega
    · rfl

end Rows

section Values

variable {B} (w : B.Witness)

theorem openForm_value (key : B.OpenKey) (j : Fin (1 * B.openRank)) :
    (B.openForm key j).value w.flat = stack key.matrix (fun _ : Fin 1 => w.partials) j :=
  Form.value_ofMatrix _ _ _
    (fun column => ofCoeffs_eq fun k small => by
      rw [elim_map, show (fun x => w.flat (B.relocate column.1 x)) = (w.group column.1).flat from
        funext (w.flat_relocate column.1), Akita.Witness.flat_partialCell _ _ small]
      rfl) _

/-- The scalar row of the flat witness is the left side of the scalar row (100) of Akita. -/
theorem scalarWeight_sum (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) :
    ∑ x ∈ Finset.range (2 ^ μ), algebraMap Zq E (w.flat x) * B.scalarWeight β st μ x =
      B.scalar β st w.partials := by
  have each : ∀ g, ∑ x ∈ Finset.range (2 ^ μ), algebraMap Zq E (w.flat x) *
      ∑ y ∈ Finset.range (2 ^ μ), (if B.relocate g y = x then
        Akita.scalarWeight (B.opening g) (B.fold g) (β g) (st.point g) y else 0) =
      Akita.scalar (β g) (st.point g)
        ((B.opening g).partialPlane fun column => w.partials ⟨g, column⟩) := fun g => by
    rw [sum_mul_relocate _ _ _ _ fun y _ nonzero => by
      by_contra outside
      refine nonzero (Akita.scalarWeight_of_le (B.opening g) (B.fold g) (β g) (st.point g) ?_)
      by_contra small
      have below := B.relocate_lt_chainStart g (Nat.lt_of_not_le small)
      have ends : B.chainStart ≤ B.cells := by
        unfold cells openSecondStart secondStart openFirstStart
        omega
      omega]
    simp only [w.flat_relocate g]
    exact Akita.scalarWeight_sum (w.group g) (β g) (st.point g) μ ((B.cells_le g).trans fits)
  unfold Batch.scalarWeight Batch.scalar
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun g _ => ?_
  rw [← each g, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum]
  exact Finset.sum_congr rfl fun y _ => by ring

end Values

section Link

variable {B} (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
  (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)

/-- The flat witness satisfies the relation of one fold with opening groups when the digit
vectors satisfy, for each group, the rows (102), (104) and (105) with their rows (47) or (48),
and the shared row (106) with its rows (47) or (48) and the scalar row (100), of Akita Figure 5,
and only then. -/
theorem flatRelation_holds_iff (w : B.Witness) :
    (B.flatRelation β st challenge μ fits).Holds w.flat ↔
      (∀ g, (∀ t, ∑ i, challenge g i * (B.opening g).partialPlane (w.group g).partials i t =
          contract (β g) (st.point g) t
            (gadget (B.shape g).sourceBase (gadget (B.fold g).responseBase (w.group g).response))) ∧
        (st.key g).inner.mulVec (gadget (B.fold g).responseBase (w.group g).response) =
          ∑ i, embed (B.opening g).stride (B.opening g).stride_tower (challenge g i) •
            gadget (B.shape g).openingBase ((w.group g).inner i) ∧
        Rows (B.shape g).chain (st.key g).chain
          (stack (st.key g).outer ((B.shape g).sliceInput (w.group g).inner)) (w.group g).chain
          (st.payload g)) ∧
      B.OpenRows st.openKey w.partials w.openChain st.openPayload ∧
      B.scalar β st w.partials = st.value := by
  have moved : ∀ g (row : RingRow),
      (row.relocate (B.relocate g)).Holds w.flat ↔ row.Holds (w.group g).flat := fun g row => by
    rw [RingRow.relocate_holds,
      show (fun x => w.flat (B.relocate g x)) = (w.group g).flat from funext (w.flat_relocate g)]
  have outer := fun g => rows_iff (B.shape g).chain (st.key g).chain (w.group g).chain
    (st.payload g) (outerForm (B.opening g) (B.fold g) (st.key g))
    (stack (st.key g).outer ((B.shape g).sliceInput (w.group g).inner))
    ((B.opening g).firstCell (B.fold g)) ((B.opening g).secondCell (B.fold g)) (w.group g).flat
    (outerForm_value (w.group g) (st.key g)) (w.group g).flat_firstCell
    (w.group g).flat_secondCell
  have opening := rows_iff B.chain st.openKey.chain w.openChain st.openPayload
    (B.openForm st.openKey) (stack st.openKey.matrix fun _ : Fin 1 => w.partials)
    B.openFirstCell B.openSecondCell w.flat (openForm_value w st.openKey) w.flat_openFirstCell
    w.flat_openSecondCell
  have rows : (∀ r, (B.flatRows β st challenge r).Holds w.flat) ↔
      (∀ g, (∀ t, (evaluationRow (B.opening g) (B.fold g) (β g) (st.point g) (challenge g)
          t).Holds (w.group g).flat) ∧
        (∀ a, (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).Holds
          (w.group g).flat) ∧
        ∀ j, (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
          ((B.opening g).firstCell (B.fold g)) (st.payload g) j).Holds (w.group g).flat) ∧
      (∀ j, (imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload j).Holds
        w.flat) ∧
      (∀ g r, (firstRow (B.shape g).chain (st.key g).chain ((B.opening g).firstCell (B.fold g))
        ((B.opening g).secondCell (B.fold g)) r).Holds (w.group g).flat) ∧
      (∀ r, (firstRow B.chain st.openKey.chain B.openFirstCell B.openSecondCell r).Holds
        w.flat) ∧
      (∀ g r, (secondRow (B.shape g).chain (st.key g).chain
        ((B.opening g).secondCell (B.fold g)) (st.payload g) r).Holds (w.group g).flat) ∧
      ∀ r, (secondRow B.chain st.openKey.chain B.openSecondCell st.openPayload r).Holds w.flat :=
    (forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr
      ((forall_gather (RingRow.Holds · w.flat) _).trans (forall_congr' fun g =>
        (forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr
          (forall_congr' fun _ => moved g _)
          ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr
            (forall_congr' fun _ => moved g _) (forall_congr' fun _ => moved g _))))))
      ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
        ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr
          ((forall_gather (RingRow.Holds · w.flat) _).trans
            (forall_congr' fun g => forall_congr' fun _ => moved g _))
          ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
            ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_left'
              ((forall_gather (RingRow.Holds · w.flat) _).trans
                (forall_congr' fun g => forall_congr' fun _ => moved g _)))))))))))
  have value := scalarWeight_sum w β st μ fits
  constructor
  · rintro ⟨ring, scalarRow⟩
    obtain ⟨groups, openImage, outerFirst, openFirst, outerSecond, openSecond⟩ := rows.mp ring
    exact ⟨fun g => ⟨fun t => (evaluationRow_holds (w.group g) (β g) (st.point g) (challenge g)
        t).mp ((groups g).1 t),
      funext fun a => (innerRow_holds (w.group g) (st.key g) (challenge g) a).mp
        ((groups g).2.1 a),
      (outer g).mp ⟨(groups g).2.2, outerFirst g, outerSecond g⟩⟩,
      opening.mp ⟨openImage, openFirst, openSecond⟩, value.symm.trans scalarRow⟩
  · rintro ⟨groups, openRows, scalarRow⟩
    obtain ⟨openImage, openFirst, openSecond⟩ := opening.mpr openRows
    exact ⟨rows.mpr ⟨fun g => ⟨fun t => (evaluationRow_holds (w.group g) (β g) (st.point g)
        (challenge g) t).mpr ((groups g).1 t),
      fun a => (innerRow_holds (w.group g) (st.key g) (challenge g) a).mpr
        (congrFun (groups g).2.1 a),
      ((outer g).mpr (groups g).2.2).1⟩, openImage,
      fun g => ((outer g).mpr (groups g).2.2).2.1, openFirst,
      fun g => ((outer g).mpr (groups g).2.2).2.2, openSecond⟩, value.trans scalarRow⟩

end Link

/-- The alphabets of the digit vectors of one fold with opening groups (Akita Definition 10.22
(v) and Section 5.5): the alphabets `Witness.Alphabets` of each group, and `{−1, 0}` for the
digits of `H_1` and `H_2`. -/
structure Witness.Alphabets {B : Batch (Fin n)} (w : B.Witness) : Prop where
  groups : ∀ g, (w.group g).Alphabets
  openDigits : PathDigits.Binary B.chain w.openChain

section Alphabets

variable {B} {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {st : B.Statement E}
  {challenge : B.Challenge} {μ : ℕ} {w : B.Witness}

omit [Field E] [Algebra Zq E] in
/-- The digit range statement and the binary statement of the flat witness give the alphabets of
the digit vectors, when each group has the certified base `base < q`. -/
theorem Witness.alphabets_of_flat {base : ℕ} (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (fits : B.cells ≤ 2 ^ μ)
    (range : RangeHolds base μ w.flat) (binary : BinaryOn B.binarySupport μ w.flat) :
    w.Alphabets where
  groups g := by
    have ranged : RangeHolds (B.shape g).certifiedBase μ (w.group g).flat := by
      rw [uniform g]
      intro x _
      by_cases below : x < (B.opening g).cells (B.fold g)
      · rw [← w.flat_relocate g x]
        exact range _ ((B.relocate_lt g below).trans_le fits)
      · rw [Akita.Witness.flat_of_le _ (Nat.le_of_not_lt below), zero_mul]
        exact range.zero
    have binaried : BinaryOn ((B.opening g).binarySupport (B.fold g)) μ (w.group g).flat :=
      fun x _ member => by
        rw [← w.flat_relocate g x]
        exact binary _ ((B.relocate_lt g member.2).trans_le fits) (B.relocate_support g member)
    exact ((B.opening g).flatLayout (B.fold g) μ ((B.cells_le g).trans fits)).alphabets
      (by rw [uniform g]; exact small) (w.group g) ranged binaried
  openDigits := by
    have ends : B.chainStart ≤ B.openFirstStart ∧ B.secondStart ≤ B.openSecondStart ∧
        B.secondStart = B.openFirstStart + firstCells B.chain ∧
        B.cells = B.openSecondStart + secondCells B.chain :=
      ⟨Nat.le_add_right _ _, Nat.le_add_right _ _, rfl, rfl⟩
    exact (PathDigits.placed_of_tables B.chain w.openChain B.openFirstStart B.openSecondStart
      (fun x small => ⟨⟨by omega, by unfold binarySupport; omega⟩, w.flat_openFirst small⟩)
      (fun x small => ⟨⟨by omega, by unfold binarySupport; omega⟩,
        w.flat_openSecond small⟩)).binary binary

variable (μ) in
/-- The premise `encodes` of the link for `Batch.flatRelation`: a witness with the certified
alphabets whose flat witness satisfies the relation of one fold satisfies the exact branch
relation `Batch.Witness.Satisfies` (the hypothesis of Akita Corollary 10.32). -/
theorem Witness.satisfies_of_flat (fits : B.cells ≤ 2 ^ μ) (alphabets : w.Alphabets)
    (holds : (B.flatRelation β st challenge μ fits).Holds w.flat) :
    w.Satisfies β st challenge := by
  obtain ⟨groups, openRows, value⟩ := (flatRelation_holds_iff β st challenge μ fits w).mp holds
  exact ⟨fun g => ⟨⟨(groups g).2.2, (alphabets.groups g).innerDigits,
      (alphabets.groups g).chainDigits, (alphabets.groups g).partialDigits⟩, (groups g).1,
      (groups g).2.1, (alphabets.groups g).responseDigits⟩, openRows, alphabets.openDigits, value⟩

variable (μ) in
/-- The honest direction: the flat witness of digit vectors that satisfy the exact branch
relation satisfies the relation of one fold (Akita Theorem 7.4, item 1, for the rows). -/
theorem Witness.Satisfies.flat (fits : B.cells ≤ 2 ^ μ)
    (satisfies : w.Satisfies β st challenge) :
    (B.flatRelation β st challenge μ fits).Holds w.flat :=
  (flatRelation_holds_iff β st challenge μ fits w).mpr
    ⟨fun g => ⟨(satisfies.groups g).evaluation, (satisfies.groups g).innerRows,
      (satisfies.groups g).carried.rows⟩, satisfies.openRows, satisfies.value⟩

theorem Witness.Satisfies.alphabets (satisfies : w.Satisfies β st challenge) : w.Alphabets :=
  ⟨fun g => ⟨(satisfies.groups g).responseDigits, (satisfies.groups g).carried.innerDigits,
    (satisfies.groups g).carried.partialDigits, (satisfies.groups g).carried.chainDigits,
    satisfies.openDigits⟩, satisfies.openDigits⟩

end Alphabets

section Check

variable {B} [Fintype E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
  (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)

/-- The quotient-free reduction of the relation of one fold with opening groups (Akita
Section 7.3 and Theorem 7.4), with `2^batch` places for the rows. Its bad set has at most
`d_max − 1` points. -/
noncomputable def flatReduction (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) :
    Reduction E :=
  (B.flatRelation β st challenge μ fits).free B.largestDegree batch
    (B.flatRows_degree_le β st challenge) rowsFit

variable [DecidableEq E]

/-- The link from the checks of one fold with opening groups to its exact branch relation: on a
path of `foldCheck` that avoids the bad sets, an accepted claim that is true for the flat witness
`Batch.Witness.flat` gives `Batch.Witness.Satisfies` for the fold challenges (the hypothesis of
Akita Corollary 10.32 for one child). Each group has the certified base `base`. -/
theorem flatFold_satisfies {base : ℕ} (uniform : ∀ g, (B.shape g).certifiedBase = base)
    (small : base < fieldModulus) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (w : B.Witness) :
    (foldCheck (flatReduction β st challenge μ fits batch rowsFit) base B.binarySupport B.compressed
      w.flat).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w.flat) →
        w.Satisfies β st challenge :=
  (foldCheck_avoids (flatReduction β st challenge μ fits batch rowsFit) base B.binarySupport
    B.compressed B.binarySupport_empty
    w.flat).mono fun _ checked r2 v equal opening =>
      Witness.satisfies_of_flat μ fits
        (Witness.alphabets_of_flat uniform small fits (checked r2 v equal opening).2.1
          (checked r2 v equal opening).2.2)
        (checked r2 v equal opening).1

end Check

end Batch

end NightstreamFPrime.Spec.Compression.Akita
