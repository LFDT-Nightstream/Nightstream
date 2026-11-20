import NightstreamFPrime.Spec.Compression.Akita.Batch
import NightstreamFPrime.Spec.Compression.Akita.FlatWitness

/-! The flat witness of one fold of Akita with opening groups and coefficient packing, for one
claim in each group (ePrint 2026/1983, revision of 2026-10-15, Section 5.5 with (75) and (76)).

- `offset` and `concat` put the tables of a family of groups in one table, in the order of the
  groups, with no padding.
- `Batch.Witness.flat` is the flat witness for the groups `g : Fin n`: the main segments
  `[ẑ_g | ê_g | t̂_g]` of the groups (75), then the digits of `F_1` of each group, the digits of
  `H_1`, the digits of `F_2` of each group and the digits of `H_2` (76). Each cell from
  `Batch.cells` is 0.
- `Batch.relocate g` maps a position of the flat witness `Witness.flat` of the group `g` with the
  shared `D/H` path to its position in the flat witness of the fold, and
  `Batch.Witness.flat_relocate` states that the two cells are equal. A row of one group on the
  flat witness of the fold is the row of `FlatFold.lean` with its addresses moved by this map.
- `Batch.binarySupport` is the set `I_bin` (Akita Remark 6.2).

- `Batch.relocate_one` and `Batch.Witness.flat_one` state that a fold with one group has the
  positions and the flat witness of `FlatWitness.lean`.

Akita (75) has the order `[ẑ_g | ê_g | t̂_g]` in a group, which is the order of `Witness.flat`.
As in `FlatWitness.lean`, there are no alignment zeros, and the cell `Batch.cells` is a zero
cell. -/

namespace NightstreamFPrime.Spec.Compression.Akita

section Concat

variable {n : ℕ} (size : Fin n → ℕ)

/-- The number of cells of the groups before the group `k`. -/
def offset (k : ℕ) : ℕ := ∑ g : Fin n, if (g : ℕ) < k then size g else 0

theorem offset_succ (g : Fin n) : offset size (g + 1) = offset size g + size g := by
  have each : ∀ h : Fin n, (if (h : ℕ) < g + 1 then size h else 0) =
      (if (h : ℕ) < g then size h else 0) + if h = g then size h else 0 := fun h => by
    by_cases before : (h : ℕ) < g
    · rw [ite_eq_left before, ite_eq_left (by omega),
        ite_eq_right (fun same => by subst same; omega), add_zero]
    · by_cases same : h = g
      · subst same
        rw [ite_eq_left (by omega), ite_eq_right before, ite_eq_left rfl, zero_add]
      · rw [ite_eq_right (fun below => same (Fin.ext (by omega))), ite_eq_right before,
          ite_eq_right same]
  unfold offset
  rw [Finset.sum_congr rfl fun h _ => each h, Finset.sum_add_distrib, Finset.sum_ite_eq',
    ite_eq_left (Finset.mem_univ g)]

theorem offset_zero : offset size 0 = 0 :=
  Finset.sum_eq_zero fun _ _ => ite_eq_right (Nat.not_lt_zero _)

theorem offset_mono {j k : ℕ} (le : j ≤ k) : offset size j ≤ offset size k :=
  Finset.sum_le_sum fun h _ => by
    by_cases before : (h : ℕ) < j
    · rw [ite_eq_left before, ite_eq_left (by omega)]
    · rw [ite_eq_right before]
      exact Nat.zero_le _

theorem offset_all : offset size n = ∑ g, size g :=
  Finset.sum_congr rfl fun g _ => ite_eq_left g.isLt

theorem offset_add_le (g : Fin n) : offset size g + size g ≤ ∑ h, size h := by
  rw [← offset_succ, ← offset_all]
  exact offset_mono size g.isLt

/-- The table with the first `size g` cells of each table `table g`, in the order of the groups
(Akita (75): the intervals are joined "without padding to a common geometry"). -/
noncomputable def concat (table : Fin n → Table) : Table := fun x =>
  ∑ g : Fin n, if offset size g ≤ x ∧ x < offset size g + size g then table g (x - offset size g)
    else 0

theorem concat_apply (table : Fin n → Table) (g : Fin n) {y : ℕ} (small : y < size g) :
    concat size table (offset size g + y) = table g y := by
  unfold concat
  rw [Finset.sum_eq_single g, ite_eq_left ⟨Nat.le_add_right _ _, by omega⟩, Nat.add_sub_cancel_left]
  · intro h _ other
    refine ite_eq_right fun inside => ?_
    rcases Nat.lt_or_gt_of_ne (Fin.val_ne_of_ne other) with before | after
    · have step := offset_mono size (show (h : ℕ) + 1 ≤ g from before)
      rw [offset_succ] at step
      omega
    · have step := offset_mono size (show (g : ℕ) + 1 ≤ h from after)
      rw [offset_succ] at step
      omega
  · exact fun absent => absurd (Finset.mem_univ g) absent

theorem concat_of_le (table : Fin n → Table) {x : ℕ} (large : ∑ g, size g ≤ x) :
    concat size table x = 0 :=
  Finset.sum_eq_zero fun g _ => ite_eq_right fun inside => by
    have last := offset_add_le size g
    omega

/-- Each position before the last cell is a position of one group. -/
theorem exists_offset {x : ℕ} (small : x < ∑ g, size g) :
    ∃ (g : Fin n) (y : ℕ), y < size g ∧ x = offset size g + y := by
  have step : ∀ k, k ≤ n → x < offset size k →
      ∃ (g : Fin n) (y : ℕ), y < size g ∧ x = offset size g + y := by
    intro k
    induction k with
    | zero =>
        intro _ below
        simp [offset] at below
    | succ k induction =>
        intro le below
        by_cases before : x < offset size k
        · exact induction (by omega) before
        · have next : offset size (k + 1) = offset size k + size ⟨k, le⟩ :=
            offset_succ size ⟨k, le⟩
          exact ⟨⟨k, le⟩, x - offset size k, by omega,
            (Nat.add_sub_cancel' (Nat.le_of_not_lt before)).symm⟩
  exact step n le_rfl (by rwa [offset_all])

end Concat

namespace Batch

variable {n : ℕ} (B : Batch (Fin n))

/-- The number of cells of the main segment `[ẑ_g | ê_g | t̂_g]` of the group `g` (Akita (75)). -/
def mainCells (g : Fin n) : ℕ := (B.opening g).chainStart (B.fold g)

/-- The offset of the digits of `F_1` of the first group, which is the first position of `I_bin`:
the number of cells of the main segments (Akita (76)). -/
def chainStart : ℕ := ∑ g, B.mainCells g

/-- The offset of the digits of `H_1`, after the digits of `F_1` of each group. -/
def openFirstStart : ℕ := B.chainStart + ∑ g, firstCells (B.shape g).chain

/-- The offset of the digits of `F_2` of the first group. -/
def secondStart : ℕ := B.openFirstStart + firstCells B.chain

/-- The offset of the digits of `H_2`, after the digits of `F_2` of each group. -/
def openSecondStart : ℕ := B.secondStart + ∑ g, secondCells (B.shape g).chain

/-- The number of cells of the flat witness before the zero suffix. -/
def cells : ℕ := B.openSecondStart + secondCells B.chain

/-- The offset of the main segment of the group `g`. -/
def mainStart (g : Fin n) : ℕ := offset B.mainCells g

/-- The offset of the digits of `F_1` of the group `g`. -/
def firstStart (g : Fin n) : ℕ := B.chainStart + offset (fun h => firstCells (B.shape h).chain) g

/-- The offset of the digits of `F_2` of the group `g`. -/
def secondStartOf (g : Fin n) : ℕ :=
  B.secondStart + offset (fun h => secondCells (B.shape h).chain) g

/-- The position in the flat witness of the fold of the position `x` of the flat witness
`Witness.flat` of the group `g`: the main segment, the digits of `F_1`, of `H_1`, of `F_2` and of
`H_2` of the group move to their segments of the fold, and the zero suffix moves to the zero
suffix. -/
def relocate (g : Fin n) (x : ℕ) : ℕ :=
  if x < (B.opening g).chainStart (B.fold g) then B.mainStart g + x
  else if x < (B.opening g).openFirstStart (B.fold g) then
    B.firstStart g + (x - (B.opening g).chainStart (B.fold g))
  else if x < (B.opening g).secondStart (B.fold g) then
    B.openFirstStart + (x - (B.opening g).openFirstStart (B.fold g))
  else if x < (B.opening g).openSecondStart (B.fold g) then
    B.secondStartOf g + (x - (B.opening g).secondStart (B.fold g))
  else B.openSecondStart + (x - (B.opening g).openSecondStart (B.fold g))

/-- The set `I_bin` of the positions of the compression digits (Akita Remark 6.2): the positions
from the first digit of `F_1` to the last digit of `H_2`. -/
def binarySupport (x : ℕ) : Prop := B.chainStart ≤ x ∧ x < B.cells

instance : DecidablePred B.binarySupport := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- The position of entry `position` of the digit sequence of `H_1`. -/
def openFirstCell (position : ℕ) : ℕ :=
  if position < firstCells B.chain then B.openFirstStart + position else B.cells

/-- The position of entry `position` of the digit sequence of `H_2`. -/
def openSecondCell (position : ℕ) : ℕ :=
  if position < secondCells B.chain then B.openSecondStart + position else B.cells

theorem openFirstCell_le (position : ℕ) : B.openFirstCell position ≤ B.cells := by
  unfold openFirstCell cells openSecondStart secondStart
  split <;> omega

theorem openSecondCell_le (position : ℕ) : B.openSecondCell position ≤ B.cells := by
  unfold openSecondCell cells
  split <;> omega

/-- The offsets of the flat witness of the group `g`, each from the one before it. -/
theorem starts (g : Fin n) :
    (B.opening g).openFirstStart (B.fold g) =
        (B.opening g).chainStart (B.fold g) + firstCells (B.shape g).chain ∧
      (B.opening g).secondStart (B.fold g) =
        (B.opening g).openFirstStart (B.fold g) + firstCells B.chain ∧
      (B.opening g).openSecondStart (B.fold g) =
        (B.opening g).secondStart (B.fold g) + secondCells (B.shape g).chain ∧
      (B.opening g).cells (B.fold g) =
        (B.opening g).openSecondStart (B.fold g) + secondCells B.chain :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- The order of the segments of the flat witness of the fold, for the group `g`. -/
theorem bounds (g : Fin n) :
    B.mainCells g = (B.opening g).chainStart (B.fold g) ∧
      B.mainStart g + B.mainCells g ≤ B.chainStart ∧ B.chainStart ≤ B.firstStart g ∧
      B.firstStart g + firstCells (B.shape g).chain ≤ B.openFirstStart ∧
      B.secondStart = B.openFirstStart + firstCells B.chain ∧
      B.secondStart ≤ B.secondStartOf g ∧
      B.secondStartOf g + secondCells (B.shape g).chain ≤ B.openSecondStart ∧
      B.cells = B.openSecondStart + secondCells B.chain := by
  have one := offset_add_le (fun h => firstCells (B.shape h).chain) g
  have two := offset_add_le (fun h => secondCells (B.shape h).chain) g
  refine ⟨rfl, offset_add_le B.mainCells g, Nat.le_add_right _ _, ?_, rfl, Nat.le_add_right _ _,
    ?_, rfl⟩
  · unfold firstStart openFirstStart
    omega
  · unfold secondStartOf openSecondStart
    omega

theorem relocate_main (g : Fin n) {x : ℕ} (small : x < B.mainCells g) :
    B.relocate g x = B.mainStart g + x :=
  ite_eq_left small

theorem relocate_first (g : Fin n) {y : ℕ} (small : y < firstCells (B.shape g).chain) :
    B.relocate g ((B.opening g).chainStart (B.fold g) + y) = B.firstStart g + y := by
  obtain ⟨first, -, -, -⟩ := B.starts g
  unfold relocate
  rw [ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left]

theorem relocate_openFirst (g : Fin n) {y : ℕ} (small : y < firstCells B.chain) :
    B.relocate g ((B.opening g).openFirstStart (B.fold g) + y) = B.openFirstStart + y := by
  obtain ⟨first, second, -, -⟩ := B.starts g
  unfold relocate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    Nat.add_sub_cancel_left]

theorem relocate_second (g : Fin n) {y : ℕ} (small : y < secondCells (B.shape g).chain) :
    B.relocate g ((B.opening g).secondStart (B.fold g) + y) = B.secondStartOf g + y := by
  obtain ⟨first, second, third, -⟩ := B.starts g
  unfold relocate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega), Nat.add_sub_cancel_left]

theorem relocate_openSecond (g : Fin n) (y : ℕ) :
    B.relocate g ((B.opening g).openSecondStart (B.fold g) + y) = B.openSecondStart + y := by
  obtain ⟨first, second, third, -⟩ := B.starts g
  unfold relocate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), Nat.add_sub_cancel_left]

/-- Each position of the flat witness of a group is one of its five segments or its zero
suffix. -/
theorem relocate_cases (g : Fin n) (x : ℕ) :
    (x < B.mainCells g ∧ B.relocate g x = B.mainStart g + x) ∨
      (∃ y, y < firstCells (B.shape g).chain ∧
        x = (B.opening g).chainStart (B.fold g) + y ∧ B.relocate g x = B.firstStart g + y) ∨
      (∃ y, y < firstCells B.chain ∧
        x = (B.opening g).openFirstStart (B.fold g) + y ∧
          B.relocate g x = B.openFirstStart + y) ∨
      (∃ y, y < secondCells (B.shape g).chain ∧
        x = (B.opening g).secondStart (B.fold g) + y ∧ B.relocate g x = B.secondStartOf g + y) ∨
      ∃ y, x = (B.opening g).openSecondStart (B.fold g) + y ∧
        B.relocate g x = B.openSecondStart + y := by
  obtain ⟨first, second, third, -⟩ := B.starts g
  by_cases main : x < B.mainCells g
  · exact Or.inl ⟨main, B.relocate_main g main⟩
  have main : (B.opening g).chainStart (B.fold g) ≤ x := Nat.le_of_not_lt main
  by_cases one : x < (B.opening g).openFirstStart (B.fold g)
  · obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le main
    exact Or.inr (Or.inl ⟨y, by omega, rfl, B.relocate_first g (by omega)⟩)
  by_cases two : x < (B.opening g).secondStart (B.fold g)
  · obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt one)
    exact Or.inr (Or.inr (Or.inl ⟨y, by omega, rfl, B.relocate_openFirst g (by omega)⟩))
  by_cases three : x < (B.opening g).openSecondStart (B.fold g)
  · obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt two)
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨y, by omega, rfl, B.relocate_second g (by omega)⟩)))
  · obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_not_lt three)
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨y, rfl, B.relocate_openSecond g y⟩)))

/-- A position of a group up to its zero cell moves to a position up to the zero cell of the
fold. -/
theorem relocate_le (g : Fin n) {x : ℕ} (le : x ≤ (B.opening g).cells (B.fold g)) :
    B.relocate g x ≤ B.cells := by
  obtain ⟨first, second, third, last⟩ := B.starts g
  obtain ⟨main, a, b, c, d, e, f, ends⟩ := B.bounds g
  rcases B.relocate_cases g x with ⟨small, moved⟩ | ⟨y, small, rfl, moved⟩ |
    ⟨y, small, rfl, moved⟩ | ⟨y, small, rfl, moved⟩ | ⟨y, rfl, moved⟩ <;> omega

/-- A position of a group after its zero cell moves to a position after the zero cell of the
fold. -/
theorem relocate_lt (g : Fin n) {x : ℕ} (large : (B.opening g).cells (B.fold g) < x) :
    B.cells < B.relocate g x := by
  obtain ⟨first, second, third, last⟩ := B.starts g
  obtain ⟨main, a, b, c, d, e, f, ends⟩ := B.bounds g
  rcases B.relocate_cases g x with ⟨small, moved⟩ | ⟨y, small, rfl, moved⟩ |
    ⟨y, small, rfl, moved⟩ | ⟨y, small, rfl, moved⟩ | ⟨y, rfl, moved⟩ <;> omega

/-- A position of `I_bin` of a group moves to a position of `I_bin` of the fold. -/
theorem relocate_support (g : Fin n) {x : ℕ}
    (member : (B.opening g).binarySupport (B.fold g) x) : B.binarySupport (B.relocate g x) := by
  obtain ⟨first, second, third, last⟩ := B.starts g
  obtain ⟨main, a, b, c, d, e, f, ends⟩ := B.bounds g
  obtain ⟨low, high⟩ := member
  unfold binarySupport
  rcases B.relocate_cases g x with ⟨small, moved⟩ | ⟨y, small, rfl, moved⟩ |
    ⟨y, small, rfl, moved⟩ | ⟨y, small, rfl, moved⟩ | ⟨y, rfl, moved⟩ <;> omega

/-- A fold with one group has the positions of the flat witness of this group. -/
theorem relocate_one (B : Batch (Fin 1)) (x : ℕ) : B.relocate 0 x = x := by
  obtain ⟨first, second, third, last⟩ := B.starts 0
  have main : B.mainStart 0 = 0 := offset_zero B.mainCells
  have chain : B.chainStart = (B.opening 0).chainStart (B.fold 0) := Fin.sum_univ_one _
  have one : B.firstStart 0 = B.chainStart :=
    congrArg (B.chainStart + ·) (offset_zero fun h => firstCells (B.shape h).chain)
  have openFirst : B.openFirstStart = B.chainStart + firstCells (B.shape 0).chain :=
    congrArg (B.chainStart + ·) (Fin.sum_univ_one fun h => firstCells (B.shape h).chain)
  have middle : B.secondStart = B.openFirstStart + firstCells B.chain := rfl
  have two : B.secondStartOf 0 = B.secondStart :=
    congrArg (B.secondStart + ·) (offset_zero fun h => secondCells (B.shape h).chain)
  have openSecond : B.openSecondStart = B.secondStart + secondCells (B.shape 0).chain :=
    congrArg (B.secondStart + ·) (Fin.sum_univ_one fun h => secondCells (B.shape h).chain)
  rcases B.relocate_cases 0 x with ⟨small, moved⟩ | ⟨y, small, rfl, moved⟩ |
    ⟨y, small, rfl, moved⟩ | ⟨y, small, rfl, moved⟩ | ⟨y, rfl, moved⟩ <;> omega

/-- The flat witness of a group has at most the cells of the flat witness of the fold. -/
theorem cells_le (g : Fin n) : (B.opening g).cells (B.fold g) ≤ B.cells := by
  obtain ⟨first, second, third, last⟩ := B.starts g
  have main : (B.opening g).chainStart (B.fold g) ≤ B.chainStart :=
    Finset.single_le_sum (f := B.mainCells) (fun _ _ => Nat.zero_le _) (Finset.mem_univ g)
  have one : firstCells (B.shape g).chain ≤ ∑ h, firstCells (B.shape h).chain :=
    Finset.single_le_sum (f := fun h => firstCells (B.shape h).chain) (fun _ _ => Nat.zero_le _)
      (Finset.mem_univ g)
  have two : secondCells (B.shape g).chain ≤ ∑ h, secondCells (B.shape h).chain :=
    Finset.single_le_sum (f := fun h => secondCells (B.shape h).chain) (fun _ _ => Nat.zero_le _)
      (Finset.mem_univ g)
  unfold cells openSecondStart secondStart openFirstStart
  omega

namespace Witness

variable {B} (w : B.Witness)

/-- The flat witness of one fold with opening groups on the quotient-free route (Akita (75) and
(76) for one claim in each group): for each group `[ẑ_g | ê_g | t̂_g]`, then the digits of `F_1` of
each group, the digits of `H_1`, the digits of `F_2` of each group, the digits of `H_2`, and
zeros. -/
noncomputable def flat : Table :=
  join B.chainStart (concat B.mainCells fun g => (w.group g).flat)
    (join (∑ g, firstCells (B.shape g).chain)
      (concat (fun g => firstCells (B.shape g).chain) fun g =>
        PathDigits.firstTable (B.shape g).chain (w.chain g))
      (join (firstCells B.chain) (PathDigits.firstTable B.chain w.openChain)
        (join (∑ g, secondCells (B.shape g).chain)
          (concat (fun g => secondCells (B.shape g).chain) fun g =>
            PathDigits.secondTable (B.shape g).chain (w.chain g))
          (join (secondCells B.chain) (PathDigits.secondTable B.chain w.openChain)
            fun _ => 0))))

theorem flat_main (g : Fin n) {y : ℕ} (small : y < B.mainCells g) :
    w.flat (B.mainStart g + y) = (w.group g).flat y := by
  have inside := offset_add_le B.mainCells g
  unfold flat
  rw [join_left _ _ (show B.mainStart g + y < B.chainStart from
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left small _) inside)]
  exact concat_apply B.mainCells _ g small

theorem flat_first (g : Fin n) {y : ℕ} (small : y < firstCells (B.shape g).chain) :
    w.flat (B.firstStart g + y) = PathDigits.firstTable (B.shape g).chain (w.chain g) y := by
  have inside := offset_add_le (fun h => firstCells (B.shape h).chain) g
  simp only [firstStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_left _ _ (by omega)]
  exact concat_apply (fun h => firstCells (B.shape h).chain) _ g small

theorem flat_openFirst {y : ℕ} (small : y < firstCells B.chain) :
    w.flat (B.openFirstStart + y) = PathDigits.firstTable B.chain w.openChain y := by
  simp only [openFirstStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_left _ _ small]

theorem flat_second (g : Fin n) {y : ℕ} (small : y < secondCells (B.shape g).chain) :
    w.flat (B.secondStartOf g + y) = PathDigits.secondTable (B.shape g).chain (w.chain g) y := by
  have inside := offset_add_le (fun h => secondCells (B.shape h).chain) g
  simp only [secondStartOf, secondStart, openFirstStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_left _ _ (by omega)]
  exact concat_apply (fun h => secondCells (B.shape h).chain) _ g small

theorem flat_openSecond {y : ℕ} (small : y < secondCells B.chain) :
    w.flat (B.openSecondStart + y) = PathDigits.secondTable B.chain w.openChain y := by
  simp only [openSecondStart, secondStart, openFirstStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_left _ _ small]

/-- Each cell from `Batch.cells` is 0. -/
theorem flat_of_le {x : ℕ} (large : B.cells ≤ x) : w.flat x = 0 := by
  obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le large
  simp only [cells, openSecondStart, secondStart, openFirstStart, Nat.add_assoc]
  unfold flat
  rw [join_right, join_right, join_right, join_right, join_right]

/-- The cell of the flat witness of the fold at the moved position is the cell of the flat
witness of the group. -/
theorem flat_relocate (g : Fin n) (x : ℕ) : w.flat (B.relocate g x) = (w.group g).flat x := by
  obtain ⟨-, -, -, last⟩ := B.starts g
  rcases B.relocate_cases g x with ⟨small, moved⟩ | ⟨y, small, rfl, moved⟩ |
    ⟨y, small, rfl, moved⟩ | ⟨y, small, rfl, moved⟩ | ⟨y, rfl, moved⟩
  · rw [moved, w.flat_main g small]
  · rw [moved, w.flat_first g small, Akita.Witness.flat_first _ small]
    rfl
  · rw [moved, w.flat_openFirst small, Akita.Witness.flat_openFirst _ small]
    rfl
  · rw [moved, w.flat_second g small, Akita.Witness.flat_second _ small]
    rfl
  · rw [moved]
    by_cases small : y < secondCells B.chain
    · rw [w.flat_openSecond small, Akita.Witness.flat_openSecond _ small]
      rfl
    · rw [w.flat_of_le (show B.cells ≤ B.openSecondStart + y by
          unfold cells
          omega),
        Akita.Witness.flat_of_le _ (by omega)]

/-- The flat witness of a fold with one group is the flat witness `Witness.flat` of this group
(Akita Section 7.2: "For `G = 1` and one claim, the rows specialize to the local formulas"). -/
theorem flat_one {B : Batch (Fin 1)} (w : B.Witness) : w.flat = (w.group 0).flat :=
  funext fun x => by rw [← w.flat_relocate 0 x, B.relocate_one x]

theorem flat_openFirstCell (position : ℕ) :
    w.flat (B.openFirstCell position) = PathDigits.firstTable B.chain w.openChain position := by
  unfold openFirstCell
  split
  · rename_i small
    exact w.flat_openFirst small
  · rename_i large
    rw [w.flat_of_le le_rfl, PathDigits.firstTable_of_le _ _ _ (Nat.le_of_not_lt large)]

theorem flat_openSecondCell (position : ℕ) :
    w.flat (B.openSecondCell position) = PathDigits.secondTable B.chain w.openChain position := by
  unfold openSecondCell
  split
  · rename_i small
    exact w.flat_openSecond small
  · rename_i large
    rw [w.flat_of_le le_rfl, PathDigits.secondTable_of_le _ _ _ (Nat.le_of_not_lt large)]

end Witness

end Batch

end NightstreamFPrime.Spec.Compression.Akita
