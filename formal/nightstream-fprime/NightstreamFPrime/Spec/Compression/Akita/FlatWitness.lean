import NightstreamFPrime.Spec.Compression.Akita.Layout

/-! The flat witness of one fold of Akita with coefficient packing, for one claim in one group
(ePrint 2026/1983, revision of 2026-11-23, Section 5.5 with (75) and (76), and Appendix L.2 with
(346) and (347)).

- `Witness.flat` is the flat witness of the quotient-free route: the segments `ẑ`, `ê`, `t̂`, then
  the digits of `F_1`, `H_1`, `F_2` and `H_2`, which is the order of (75) and (76). Each segment
  starts where the segment before it ends. Each cell from `OpeningShape.cells` is 0.
- `Shape.responseCode` is the address (347) of an element of `ẑ`. `OpeningShape.partialCode` and
  `Shape.innerCode` are the address (346) of an element of `ê` in `R_{q,d_D}` and of an element of
  the view of `t̂` in `R_{q,d_B}`.
- `OpeningShape.responseCell`, `OpeningShape.partialCell`, `OpeningShape.outerCell` and the four
  functions for the compression digits give the position of one coefficient in the flat witness.
  A coefficient index that is not in the segment gives `none`: a row has the coefficient 0 there
  and does not read the witness (Akita (109)). `OpeningShape.partialPosition`,
  `OpeningShape.innerCell` and `OpeningShape.sliceCell` are the positions that the rows of Akita
  Figure 5 read.
- `OpeningShape.flatLayout` is the `Layout` of `Witness.flat` on a cube with
  `OpeningShape.cells < 2^μ`, with the set `I_bin = OpeningShape.binarySupport`.

Akita Appendix L.2 puts zeros between the segments for alignment. The levels of the schedule files
have no such zeros before their last segment (`tests/AkitaSchedules.lean`), and this file has
none. -/

namespace NightstreamFPrime.Spec.Compression.Akita

section Tables

/-- The table with the first `n` cells of `a` and then the cells of `b`. -/
def join (n : ℕ) (a b : Table) : Table := fun x => if x < n then a x else b (x - n)

theorem join_left {n x : ℕ} (a b : Table) (small : x < n) : join n a b x = a x :=
  ite_eq_left small

theorem join_right (n : ℕ) (a b : Table) (x : ℕ) : join n a b (n + x) = b x := by
  show (if n + x < n then a (n + x) else b (n + x - n)) = b x
  rw [ite_eq_right (by omega), Nat.add_sub_cancel_left]

theorem cell_lt {m d : ℕ} (j : Fin m) {k : ℕ} (small : k < d) : j * d + k < m * d :=
  calc (j : ℕ) * d + k < (j + 1) * d := by rw [Nat.add_mul, Nat.one_mul]; omega
    _ ≤ m * d := Nat.mul_le_mul_right _ j.isLt

variable {d m : ℕ} [NeZero d] {ι : Type}

theorem sequence_of_le (v : Fin m → Rq d) {x : ℕ} (large : m * d ≤ x) : sequence v x = 0 :=
  dite_eq_right (Nat.not_lt.mpr ((Nat.le_div_iff_mul_le (NeZero.pos d)).mpr large))

theorem ofCoeffs_eq {x : Rq d} {c : ℕ → Zq} (same : ∀ j < d, c j = coeff x j) : ofCoeffs c = x :=
  ext_coeff fun j small => (coeff_ofCoeffs c small).trans (same j small)

/-- The coefficients of the vector `v` in one sequence, with the entries in the order of `code`:
position `code(j)·d + k` has coefficient `k` of entry `j` (Akita Appendix L.2). -/
noncomputable def cellsOf (code : ι ≃ Fin m) (v : ι → Rq d) : Table :=
  sequence fun j => v (code.symm j)

theorem cellsOf_apply (code : ι ≃ Fin m) (v : ι → Rq d) (j : ι) {k : ℕ} (small : k < d) :
    cellsOf code v (code j * d + k) = coeff (v j) k := by
  rw [cellsOf, sequence_apply _ _ small, Equiv.symm_apply_apply]

end Tables

section Chain

/-- The number of coefficients of the digit vector of the first compression map. -/
def firstCells : Option ChainShape → ℕ
  | none => 0
  | some c => c.firstWidth * c.firstDegree

/-- The number of coefficients of the digit vector of the second compression map. -/
def secondCells : Option ChainShape → ℕ
  | none => 0
  | some c => c.secondWidth * c.secondDegree

/-- The coefficients of the digit vector of the first compression map in one sequence, with the
ring-completion zeros of Akita Section 4.5. -/
noncomputable def PathDigits.firstTable : (chain : Option ChainShape) → PathDigits chain → Table
  | none, _ => fun _ => 0
  | some _, digits => sequence (ChainDigits.first digits)

/-- The coefficients of the digit vector of the second compression map in one sequence. -/
noncomputable def PathDigits.secondTable : (chain : Option ChainShape) → PathDigits chain → Table
  | none, _ => fun _ => 0
  | some _, digits => sequence (ChainDigits.second digits)

theorem PathDigits.firstTable_of_le : (chain : Option ChainShape) → (digits : PathDigits chain) →
    (x : ℕ) → firstCells chain ≤ x → PathDigits.firstTable chain digits x = 0
  | none, _, _, _ => rfl
  | some _, digits, _, large => sequence_of_le (ChainDigits.first digits) large

theorem PathDigits.secondTable_of_le : (chain : Option ChainShape) → (digits : PathDigits chain) →
    (x : ℕ) → secondCells chain ≤ x → PathDigits.secondTable chain digits x = 0
  | none, _, _, _ => rfl
  | some _, digits, _, large => sequence_of_le (ChainDigits.second digits) large

/-- The digit vectors of a compression path are placed when the flat witness has their two
sequences at the offsets `first` and `second`, at positions of `positions`. -/
theorem PathDigits.placed_of_tables {flat : Table} {positions : ℕ → Prop} :
    (chain : Option ChainShape) → (digits : PathDigits chain) → (first second : ℕ) →
    (∀ x < firstCells chain, positions (first + x) ∧
      flat (first + x) = PathDigits.firstTable chain digits x) →
    (∀ x < secondCells chain, positions (second + x) ∧
      flat (second + x) = PathDigits.secondTable chain digits x) →
    PathDigits.Placed flat positions chain digits
  | none, _, _, _, _, _ => trivial
  | some c, _, first, second, firsts, seconds =>
      ⟨fun j i small => ⟨first + (j * c.firstDegree + i), (firsts _ (cell_lt j small)).1,
          (firsts _ (cell_lt j small)).2.trans (sequence_apply _ j small)⟩,
        fun j i small => ⟨second + (j * c.secondDegree + i), (seconds _ (cell_lt j small)).1,
          (seconds _ (cell_lt j small)).2.trans (sequence_apply _ j small)⟩⟩

end Chain

namespace Shape

variable (S : Shape) (fold : Descriptor)

/-- The index `δ_f·(δ_com·x + u_com) + u_fold` of the element `(x, u_com, u_fold)` of `ẑ`
(Akita (347)). -/
def responseCode : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits ≃
    Fin (S.blockLength * S.sourceDigits * fold.responseDigits) :=
  (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv

/-- The number of cells of `ẑ`. -/
def responseCells : ℕ :=
  S.blockLength * S.sourceDigits * fold.responseDigits * S.innerDegree

/-- The position of coefficient `k` of an element of `ẑ` (Akita (347)), and `none` for a
coefficient index outside the element. -/
def responseCell (col : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits)
    (k : ℕ) : Option ℕ :=
  if k < S.innerDegree then some (S.responseCode fold col * S.innerDegree + k) else none

/-- The index `((i·n_A + a)·(d_A/d_B) + y)·δ_open + u` of the element `(a, y, u)` of the view of
`t̂_i` in `R_{q,d_B}` (Akita (346) with `(D, d) = (d_A, d_B)`). -/
def innerCode : Fin S.blocks × S.OuterColumn ≃
    Fin (S.blocks * S.innerRank * S.ratio * S.openingDigits) :=
  ((Equiv.prodAssoc (Fin S.blocks) (Fin S.innerRank) (Fin S.ratio × Fin S.openingDigits)).symm.trans
    (Equiv.prodAssoc (Fin S.blocks × Fin S.innerRank) (Fin S.ratio)
      (Fin S.openingDigits)).symm).trans
    (((Equiv.prodCongr (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)) (Equiv.refl _)).trans
      (Equiv.prodCongr finProdFinEquiv (Equiv.refl _))).trans finProdFinEquiv)

/-- The number of cells of `t̂`. -/
def innerCells : ℕ :=
  S.blocks * S.innerRank * S.ratio * S.openingDigits * S.outerDegree

end Shape

namespace OpeningShape

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- The index `(i·(d_op/d_D) + y)·δ_open + u` of the element `(i, y, u)` of `ê`
(Akita (346) with `(D, d) = (k·d_car, d_D)`). -/
def partialCode : O.OpenColumn ≃ Fin (S.blocks * O.subcolumns * S.openingDigits) :=
  (Equiv.prodAssoc (Fin S.blocks) (Fin O.subcolumns) (Fin S.openingDigits)).symm.trans
    ((Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv)

/-- The number of cells of `ê`. -/
def partialCells : ℕ := S.blocks * O.subcolumns * S.openingDigits * O.openDegree

/-- The offset of `t̂`, after `ẑ` and `ê` (Akita (75)). -/
def innerStart : ℕ := S.responseCells fold + O.partialCells

/-- The offset of the digits of `F_1`, which is the first position of `I_bin` (Akita (76)). -/
def chainStart : ℕ := O.innerStart fold + S.innerCells

/-- The offset of the digits of `H_1`. -/
def openFirstStart : ℕ := O.chainStart fold + firstCells S.chain

/-- The offset of the digits of `F_2`. -/
def secondStart : ℕ := O.openFirstStart fold + firstCells O.chain

/-- The offset of the digits of `H_2`. -/
def openSecondStart : ℕ := O.secondStart fold + secondCells S.chain

/-- The number of cells of the flat witness before the zero suffix. -/
def cells : ℕ := O.openSecondStart fold + secondCells O.chain

theorem chainStart_le_cells : O.chainStart fold ≤ O.cells fold := by
  unfold cells openSecondStart secondStart openFirstStart
  omega

/-- The position of coefficient `k` of an element of `ê` (Akita (346)). -/
def partialCell (col : O.OpenColumn) (k : ℕ) : Option ℕ :=
  if k < O.openDegree then some (S.responseCells fold + (O.partialCode col * O.openDegree + k))
  else none

/-- The position of digit `u` of coordinate `position` of partial `i`: coefficient
`position mod d_D` of the subcolumn `position / d_D`, and `none` after the last subcolumn. -/
def partialPosition (i : Fin S.blocks) (u : Fin S.openingDigits) (position : ℕ) : Option ℕ :=
  if inside : position / O.openDegree < O.subcolumns then
    O.partialCell fold (i, ⟨position / O.openDegree, inside⟩, u) (position % O.openDegree)
  else none

/-- The position of coefficient `k` of an element of the view of `t̂_i` in `R_{q,d_B}`
(Akita (346)). -/
def outerCell (i : Fin S.blocks) (col : S.OuterColumn) (k : ℕ) : Option ℕ :=
  if k < S.outerDegree then some (O.innerStart fold + (S.innerCode (i, col) * S.outerDegree + k))
  else none

/-- The position of coefficient `k` of digit `u` of row `a` of `t̂_i` in `R_{q,d_A}`: coefficient
`k mod d_B` of the subcolumn `k / d_B`. -/
def innerCell (i : Fin S.blocks) (a : Fin S.innerRank) (u : Fin S.openingDigits) (k : ℕ) :
    Option ℕ :=
  if inside : k / S.outerDegree < S.ratio then
    O.outerCell fold i (a, ⟨k / S.outerDegree, inside⟩, u) (k % S.outerDegree)
  else none

/-- The position of coefficient `k` of an element of the input of slice `s` of `B` (Akita (44)),
and `none` for an element after the last block of the slice. -/
def sliceCell (s : Fin S.slices) (entry : Fin S.sliceLength × S.OuterColumn) (k : ℕ) :
    Option ℕ :=
  if inside : S.sliceStart s + entry.1 < S.sliceStart ((s : ℕ) + 1) then
    O.outerCell fold ⟨S.sliceStart s + entry.1,
      inside.trans_le (S.sliceStart_le (Nat.succ_le_of_lt s.isLt))⟩ entry.2 k
  else none

/-- The position of entry `position` of the digit sequence of `F_1`. -/
def firstCell (position : ℕ) : Option ℕ :=
  if position < firstCells S.chain then some (O.chainStart fold + position) else none

/-- The position of entry `position` of the digit sequence of `H_1`. -/
def openFirstCell (position : ℕ) : Option ℕ :=
  if position < firstCells O.chain then some (O.openFirstStart fold + position) else none

/-- The position of entry `position` of the digit sequence of `F_2`. -/
def secondCell (position : ℕ) : Option ℕ :=
  if position < secondCells S.chain then some (O.secondStart fold + position) else none

/-- The position of entry `position` of the digit sequence of `H_2`. -/
def openSecondCell (position : ℕ) : Option ℕ :=
  if position < secondCells O.chain then some (O.openSecondStart fold + position) else none

theorem responseCell_lt
    (col : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits) (k : ℕ) {x : ℕ}
    (read : S.responseCell fold col k = some x) : x < O.cells fold := by
  unfold Shape.responseCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (S.responseCode fold col) small
    unfold cells openSecondStart secondStart openFirstStart chainStart innerStart
      Shape.responseCells
    omega
  · cases read

theorem partialCell_lt (col : O.OpenColumn) (k : ℕ) {x : ℕ}
    (read : O.partialCell fold col k = some x) : x < O.cells fold := by
  unfold partialCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (O.partialCode col) small
    unfold cells openSecondStart secondStart openFirstStart chainStart innerStart partialCells
    omega
  · cases read

theorem partialPosition_lt (i : Fin S.blocks) (u : Fin S.openingDigits) (position : ℕ) {x : ℕ}
    (read : O.partialPosition fold i u position = some x) : x < O.cells fold := by
  unfold partialPosition at read
  split at read
  · exact O.partialCell_lt fold _ _ read
  · cases read

/-- A coordinate of a partial is in the segment `ê`, before the offset `innerStart`. -/
theorem partialPosition_lt_innerStart (i : Fin S.blocks) (u : Fin S.openingDigits)
    (position : ℕ) {x : ℕ} (read : O.partialPosition fold i u position = some x) :
    x < O.innerStart fold := by
  unfold partialPosition partialCell at read
  split at read
  · split at read
    · rename_i inside small
      cases read
      have cell := cell_lt (O.partialCode (i, ⟨position / O.openDegree, inside⟩, u)) small
      unfold innerStart partialCells
      omega
    · cases read
  · cases read

theorem outerCell_lt (i : Fin S.blocks) (col : S.OuterColumn) (k : ℕ) {x : ℕ}
    (read : O.outerCell fold i col k = some x) : x < O.cells fold := by
  unfold outerCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (S.innerCode (i, col)) small
    unfold cells openSecondStart secondStart openFirstStart chainStart Shape.innerCells
    omega
  · cases read

theorem innerCell_lt (i : Fin S.blocks) (a : Fin S.innerRank) (u : Fin S.openingDigits) (k : ℕ)
    {x : ℕ} (read : O.innerCell fold i a u k = some x) : x < O.cells fold := by
  unfold innerCell at read
  split at read
  · exact O.outerCell_lt fold _ _ _ read
  · cases read

theorem sliceCell_lt (s : Fin S.slices) (entry : Fin S.sliceLength × S.OuterColumn) (k : ℕ)
    {x : ℕ} (read : O.sliceCell fold s entry k = some x) : x < O.cells fold := by
  unfold sliceCell at read
  split at read
  · exact O.outerCell_lt fold _ _ _ read
  · cases read

theorem firstCell_lt (position : ℕ) {x : ℕ} (read : O.firstCell fold position = some x) :
    x < O.cells fold := by
  unfold firstCell at read
  split at read
  · cases read
    unfold cells openSecondStart secondStart openFirstStart
    omega
  · cases read

theorem openFirstCell_lt (position : ℕ) {x : ℕ} (read : O.openFirstCell fold position = some x) :
    x < O.cells fold := by
  unfold openFirstCell at read
  split at read
  · cases read
    unfold cells openSecondStart secondStart
    omega
  · cases read

theorem secondCell_lt (position : ℕ) {x : ℕ} (read : O.secondCell fold position = some x) :
    x < O.cells fold := by
  unfold secondCell at read
  split at read
  · cases read
    unfold cells openSecondStart
    omega
  · cases read

theorem openSecondCell_lt (position : ℕ) {x : ℕ}
    (read : O.openSecondCell fold position = some x) : x < O.cells fold := by
  unfold openSecondCell at read
  split at read
  · cases read
    unfold cells
    omega
  · cases read

/-- The set `I_bin` of the positions of the compression digits (Akita Remark 6.2): the positions
from the first digit of `F_1` to the last digit of `H_2`. -/
def binarySupport (x : ℕ) : Prop := O.chainStart fold ≤ x ∧ x < O.cells fold

instance : DecidablePred (O.binarySupport fold) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- The level has compression digits: the set `I_bin` is not empty. A level with the payload mode
`Raw` in the schedule file has no compression digits. -/
def compressed : Bool := decide (O.chainStart fold < O.cells fold)

theorem binarySupport_empty (raw : O.compressed fold = false) (x : ℕ) :
    ¬ O.binarySupport fold x := by
  have none := of_decide_eq_false raw
  unfold binarySupport
  omega

end OpeningShape

namespace Witness

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} (w : Witness O fold)

/-- The flat witness of one fold on the quotient-free route (Akita (75) and (76) for one group
and one claim): `ẑ`, `ê`, the view of `t̂` in `R_{q,d_B}`, then the digits of `F_1`, `H_1`, `F_2`
and `H_2`, and zeros. Inside a segment the coefficient index changes first (Akita
Appendix L.2). -/
noncomputable def flat : Table :=
  join (S.responseCells fold) (cellsOf (S.responseCode fold) w.response)
    (join O.partialCells (cellsOf O.partialCode w.partials)
      (join S.innerCells (cellsOf S.innerCode fun entry => S.outerView (w.inner entry.1) entry.2)
        (join (firstCells S.chain) (PathDigits.firstTable S.chain w.chain)
          (join (firstCells O.chain) (PathDigits.firstTable O.chain w.openChain)
            (join (secondCells S.chain) (PathDigits.secondTable S.chain w.chain)
              (join (secondCells O.chain) (PathDigits.secondTable O.chain w.openChain)
                fun _ => 0))))))

theorem flat_response {x : ℕ} (small : x < S.responseCells fold) :
    w.flat x = cellsOf (S.responseCode fold) w.response x :=
  join_left _ _ small

theorem flat_partial {x : ℕ} (small : x < O.partialCells) :
    w.flat (S.responseCells fold + x) = cellsOf O.partialCode w.partials x := by
  unfold flat
  rw [join_right, join_left _ _ small]

theorem flat_inner {x : ℕ} (small : x < S.innerCells) :
    w.flat (O.innerStart fold + x) =
      cellsOf S.innerCode (fun entry => S.outerView (w.inner entry.1) entry.2) x := by
  simp only [OpeningShape.innerStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_left _ _ small]

theorem flat_first {x : ℕ} (small : x < firstCells S.chain) :
    w.flat (O.chainStart fold + x) = PathDigits.firstTable S.chain w.chain x := by
  simp only [OpeningShape.chainStart, OpeningShape.innerStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_left _ _ small]

theorem flat_openFirst {x : ℕ} (small : x < firstCells O.chain) :
    w.flat (O.openFirstStart fold + x) = PathDigits.firstTable O.chain w.openChain x := by
  simp only [OpeningShape.openFirstStart, OpeningShape.chainStart, OpeningShape.innerStart,
    Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_left _ _ small]

theorem flat_second {x : ℕ} (small : x < secondCells S.chain) :
    w.flat (O.secondStart fold + x) = PathDigits.secondTable S.chain w.chain x := by
  simp only [OpeningShape.secondStart, OpeningShape.openFirstStart, OpeningShape.chainStart,
    OpeningShape.innerStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_right, join_left _ _ small]

theorem flat_openSecond {x : ℕ} (small : x < secondCells O.chain) :
    w.flat (O.openSecondStart fold + x) = PathDigits.secondTable O.chain w.openChain x := by
  simp only [OpeningShape.openSecondStart, OpeningShape.secondStart, OpeningShape.openFirstStart,
    OpeningShape.chainStart, OpeningShape.innerStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_right, join_right,
    join_left _ _ small]

/-- Each cell from `OpeningShape.cells` is 0. -/
theorem flat_of_le {x : ℕ} (large : O.cells fold ≤ x) : w.flat x = 0 := by
  obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le large
  simp only [OpeningShape.cells, OpeningShape.openSecondStart, OpeningShape.secondStart,
    OpeningShape.openFirstStart, OpeningShape.chainStart, OpeningShape.innerStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_right, join_right, join_right]

theorem flat_responseCell
    (col : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits) {k : ℕ}
    (small : k < S.innerDegree) :
    (S.responseCell fold col k).elim 0 w.flat = coeff (w.response col) k := by
  rw [Shape.responseCell, ite_eq_left small, Option.elim_some,
    w.flat_response (cell_lt _ small), cellsOf_apply _ _ _ small]

theorem flat_partialCell (col : O.OpenColumn) {k : ℕ} (small : k < O.openDegree) :
    (O.partialCell fold col k).elim 0 w.flat = coeff (w.partials col) k := by
  rw [OpeningShape.partialCell, ite_eq_left small, Option.elim_some,
    w.flat_partial (cell_lt _ small), cellsOf_apply _ _ _ small]

theorem flat_outerCell (i : Fin S.blocks) (col : S.OuterColumn) {k : ℕ}
    (small : k < S.outerDegree) :
    (O.outerCell fold i col k).elim 0 w.flat = coeff (S.outerView (w.inner i) col) k := by
  rw [OpeningShape.outerCell, ite_eq_left small, Option.elim_some, w.flat_inner (cell_lt _ small),
    cellsOf_apply _ _ _ small]

theorem flat_innerCell (i : Fin S.blocks) (a : Fin S.innerRank) (u : Fin S.openingDigits) {k : ℕ}
    (small : k < S.innerDegree) :
    (O.innerCell fold i a u k).elim 0 w.flat = coeff (w.inner i (a, u)) k := by
  have positive := NeZero.pos S.outerDegree
  rw [OpeningShape.innerCell, dite_eq_left (show k / S.outerDegree < S.ratio from
      Nat.div_lt_div_of_lt_of_dvd S.divides small),
    w.flat_outerCell _ _ (Nat.mod_lt k positive)]
  simp only [Shape.outerView, coeff_ofCoeffs _ (Nat.mod_lt k positive), Nat.div_add_mod']

/-- The cells of an element of the input of a slice of `B` give this element (Akita (44)). -/
theorem read_sliceCell (s : Fin S.slices) (entry : Fin S.sliceLength × S.OuterColumn) :
    ofCoeffs (fun k => (O.sliceCell fold s entry k).elim 0 w.flat) =
      S.sliceInput w.inner s entry := by
  by_cases inside : S.sliceStart s + entry.1 < S.sliceStart ((s : ℕ) + 1)
  · simp only [OpeningShape.sliceCell, Shape.sliceInput, inside, ↓reduceDIte]
    exact ofCoeffs_eq fun k small => w.flat_outerCell _ _ small
  · simp only [OpeningShape.sliceCell, Shape.sliceInput, inside, ↓reduceDIte, Option.elim_none]
    exact ofCoeffs_eq fun k _ => (coeff_zero k).symm

/-- Coordinate `position` of partial `i` is the recomposition in base `b_1` of its digits in the
flat witness (Akita (58) and (346)). -/
theorem sum_flat_partialPosition (i : Fin S.blocks) (position : ℕ) :
    ∑ u : Fin S.openingDigits,
        (S.openingBase : Zq) ^ (u : ℕ) * (O.partialPosition fold i u position).elim 0 w.flat =
      sequence (gadget S.openingBase fun entry : Fin O.subcolumns × Fin S.openingDigits =>
        w.partials (i, entry)) position := by
  have positive := NeZero.pos O.openDegree
  by_cases inside : position / O.openDegree < O.subcolumns
  · simp only [sequence, OpeningShape.partialPosition, inside, ↓reduceDIte, gadget, coeff_sum,
      coeff_natCast_pow_mul, w.flat_partialCell _ (Nat.mod_lt position positive)]
  · simp only [sequence, OpeningShape.partialPosition, inside, ↓reduceDIte, Option.elim_none,
      mul_zero, Finset.sum_const_zero]

theorem coeff_partialPlane (i : Fin S.blocks) (t : Fin O.planes) (j : Fin O.carrierDegree) :
    coeff (O.partialPlane w.partials i t) j = ∑ u : Fin S.openingDigits,
      (S.openingBase : Zq) ^ (u : ℕ) *
        (O.partialPosition fold i u (t * O.carrierDegree + j)).elim 0 w.flat := by
  rw [OpeningShape.partialPlane, coeff_ofCoeffs _ j.isLt, w.sum_flat_partialPosition]

theorem flat_firstCell (position : ℕ) :
    (O.firstCell fold position).elim 0 w.flat = PathDigits.firstTable S.chain w.chain position := by
  unfold OpeningShape.firstCell
  split
  · rename_i small
    exact w.flat_first small
  · rename_i large
    exact (PathDigits.firstTable_of_le _ _ _ (Nat.le_of_not_lt large)).symm

theorem flat_openFirstCell (position : ℕ) :
    (O.openFirstCell fold position).elim 0 w.flat =
      PathDigits.firstTable O.chain w.openChain position := by
  unfold OpeningShape.openFirstCell
  split
  · rename_i small
    exact w.flat_openFirst small
  · rename_i large
    exact (PathDigits.firstTable_of_le _ _ _ (Nat.le_of_not_lt large)).symm

theorem flat_secondCell (position : ℕ) :
    (O.secondCell fold position).elim 0 w.flat =
      PathDigits.secondTable S.chain w.chain position := by
  unfold OpeningShape.secondCell
  split
  · rename_i small
    exact w.flat_second small
  · rename_i large
    exact (PathDigits.secondTable_of_le _ _ _ (Nat.le_of_not_lt large)).symm

theorem flat_openSecondCell (position : ℕ) :
    (O.openSecondCell fold position).elim 0 w.flat =
      PathDigits.secondTable O.chain w.openChain position := by
  unfold OpeningShape.openSecondCell
  split
  · rename_i small
    exact w.flat_openSecond small
  · rename_i large
    exact (PathDigits.secondTable_of_le _ _ _ (Nat.le_of_not_lt large)).symm

end Witness

namespace OpeningShape

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- The layout of `Witness.flat` on the cube of `μ` variables, for `cells < 2^μ` (Akita (75) and
(76)). The set `I_bin` is `binarySupport`. -/
@[reducible] noncomputable def flatLayout (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) :
    Layout O fold μ where
  flat := Witness.flat
  support := O.binarySupport fold
  compressed := O.compressed fold
  raw := O.binarySupport_empty fold
  response w col k small := by
    obtain ⟨x, read⟩ := Option.isSome_iff_exists.mp
      (show (S.responseCell fold col k).isSome by simp [Shape.responseCell, small])
    exact ⟨x, (O.responseCell_lt fold col k read).trans_le fits,
      (congrArg (Option.elim · 0 w.flat) read).symm.trans (w.flat_responseCell col small)⟩
  inner w i entry k small := by
    have positive := NeZero.pos S.outerDegree
    have inside : k / S.outerDegree < S.ratio := Nat.div_lt_div_of_lt_of_dvd S.divides small
    obtain ⟨x, read⟩ := Option.isSome_iff_exists.mp
      (show (O.innerCell fold i entry.1 entry.2 k).isSome by
        simp [innerCell, inside, outerCell, Nat.mod_lt k positive])
    exact ⟨x, (O.innerCell_lt fold i entry.1 entry.2 k read).trans_le fits,
      (congrArg (Option.elim · 0 w.flat) read).symm.trans (w.flat_innerCell i entry.1 entry.2
        small)⟩
  partials w col k small := by
    obtain ⟨x, read⟩ := Option.isSome_iff_exists.mp
      (show (O.partialCell fold col k).isSome by simp [partialCell, small])
    exact ⟨x, (O.partialCell_lt fold col k read).trans_le fits,
      (congrArg (Option.elim · 0 w.flat) read).symm.trans (w.flat_partialCell col small)⟩
  chain w := PathDigits.placed_of_tables S.chain w.chain (O.chainStart fold) (O.secondStart fold)
    (fun x small => ⟨by
      unfold binarySupport cells openSecondStart secondStart openFirstStart at *
      omega, w.flat_first small⟩)
    (fun x small => ⟨by
      unfold binarySupport cells openSecondStart secondStart openFirstStart at *
      omega, w.flat_second small⟩)
  openChain w := PathDigits.placed_of_tables O.chain w.openChain (O.openFirstStart fold)
    (O.openSecondStart fold)
    (fun x small => ⟨by
      unfold binarySupport cells openSecondStart secondStart openFirstStart at *
      omega, w.flat_openFirst small⟩)
    (fun x small => ⟨by
      unfold binarySupport cells openSecondStart secondStart openFirstStart at *
      omega, w.flat_openSecond small⟩)

end OpeningShape

end NightstreamFPrime.Spec.Compression.Akita
