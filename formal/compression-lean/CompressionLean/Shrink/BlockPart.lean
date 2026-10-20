import CompressionLean.Shrink.Diagonal
import CompressionLean.Shrink.Layout

/-!
Owns the closed form `block_part` (`crates/neo-spartan/src/shrink/layout.rs:172-227`)
and its value (`blockPart_eq`): the entry sum `matrixValue` over every
block's template entries (`Layout.blockEntries`).

Block `b` puts template index `k` at `start + b·stride + k'` of its part
(`slot`). The stride is `2^σ` and the start is a multiple of it, so the eq
weight splits into the low `σ` coordinates at `k'` and the high ones at
`start / stride + b` (`eqAt_split`). The low factor depends on the entry
alone: `block_part` sums the entries per row part and cell part, and
`diagonal` sums the high factors over the blocks.

`block_part` builds the low tables of both parts. A kind whose rows (or
cells) are a power of two has an empty narrow part, of stride `0`. Then
`bits(0) = 64`, and the slice `[..64]` of `rx` (or `ry`) panics
(`layout.rs:180`, `188-189`) whenever the kind has blocks, so Rust rejects
every proof of that shape. The empty part holds no entry, so the value here
does not depend on it, and `blockPart_eq` needs no premise for it.
-/

namespace CompressionLean.Shrink

open CompressionLean.Layer1

/-- One part of a kind's slots: its start and its stride, one slot per
block (`Part`, `layout.rs:49-54`). -/
structure Part where
  start : ℕ
  stride : ℕ

/-- Part `p` of `size` template indices among `count` blocks (`place`,
`layout.rs:64-81`, one kind): part `0` is the wide one at `0`, part `1` the
narrow one after all wide slots; the stable sort keeps this order. Rust
leaves an empty narrow part at `Part::default()`; its start is never read. -/
def part (size count p : ℕ) : Part :=
  if p = 0 then ⟨0, wide size⟩ else ⟨wide size * count, narrow size⟩

/-- `bits` (`layout.rs:180`): `stride.trailing_zeros()`, which is
`log2 stride` for the power-of-two strides of `split`. -/
def strideBits (stride : ℕ) : ℕ := Nat.log 2 stride

/-- The part of template index `k` and its index there (`layout.rs:193-194`,
`198-199`). -/
def partIndex (size k : ℕ) : ℕ × ℕ :=
  ((if wide size ≤ k then 1 else 0), k - (if wide size ≤ k then 1 else 0) * wide size)

section Split

variable {K : Type} [CommRing K]

/-- The eq weight of `lo + 2^k·hi` with `lo < 2^k`: the low `k` coordinates
at `lo` times the rest at `hi`. -/
theorem eqAt_split : ∀ (point : List K) (k lo hi : ℕ), lo < 2 ^ k →
    eqAt point (lo + 2 ^ k * hi) = eqAt (point.take k) lo * eqAt (point.drop k) hi
  | point, 0, lo, hi, lt => by
      obtain rfl : lo = 0 := by simpa using lt
      simp [eqAt]
  | [], k + 1, lo, hi, _ => by simp [eqAt]
  | x :: rest, k + 1, lo, hi, lt => by
      have double : 2 ^ (k + 1) * hi = 2 * (2 ^ k * hi) := by ring
      have small : lo / 2 < 2 ^ k := by rw [pow_succ] at lt; omega
      have low : (lo + 2 * (2 ^ k * hi)) % 2 = lo % 2 := by omega
      have half : (lo + 2 * (2 ^ k * hi)) / 2 = lo / 2 + 2 ^ k * hi := by omega
      rw [double]
      simp only [eqAt, List.take_succ_cons, List.drop_succ_cons, low, half, eqAt_split rest k (lo / 2) hi small]
      ring

/-- The eq weight of index `0`: `Π (1 - x_t)`. -/
theorem eqAt_zero : ∀ point : List K, eqAt point 0 = (point.map fun x => 1 - x).prod
  | [] => rfl
  | x :: rest => by simp [eqAt, eqAt_zero rest]

end Split

section Parts

theorem wide_pos (size : ℕ) : 0 < wide size := pow_pos (by norm_num) _

theorem size_lt_wide (size : ℕ) : size < 2 * wide size := by
  have := Nat.lt_pow_succ_log_self (by norm_num : 1 < 2) size
  unfold wide
  rw [Nat.succ_eq_add_one, pow_succ] at this
  omega

/-- A nonempty narrow part is `2^c` with `c` at most the wide part's exponent. -/
theorem narrow_spec (size : ℕ) (nonempty : narrow size ≠ 0) :
    narrow size = 2 ^ Nat.clog 2 (size - wide size) ∧ Nat.clog 2 (size - wide size) ≤ Nat.log 2 size := by
  unfold narrow at nonempty ⊢
  split_ifs at nonempty ⊢ with empty
  · exact absurd rfl nonempty
  · refine ⟨rfl, Nat.clog_le_of_le_pow ?_⟩
    have := size_lt_wide size
    unfold wide at this ⊢
    omega

/-- Where template index `k < size` sits (`slot`): at `k'` past
`stride·(start / stride + b)` in its part, with `k' < stride = 2^σ`. -/
theorem slot_split (size count b k : ℕ) (inside : k < size) :
    (part size count (partIndex size k).1).stride = 2 ^ strideBits (part size count (partIndex size k).1).stride ∧
      (partIndex size k).2 < (part size count (partIndex size k).1).stride ∧
      slot size count b k = (partIndex size k).2 + (part size count (partIndex size k).1).stride *
        ((part size count (partIndex size k).1).start / (part size count (partIndex size k).1).stride + b) := by
  unfold partIndex slot
  by_cases wideK : wide size ≤ k
  · have rest := rest_le_narrow size
    have nonempty : narrow size ≠ 0 := by omega
    obtain ⟨narrowEq, le⟩ := narrow_spec size nonempty
    simp only [if_pos wideK, part, one_ne_zero, if_false, one_mul, if_neg (show ¬ k < wide size by omega)]
    have divides : narrow size ∣ wide size * count := by
      rw [narrowEq]
      exact Dvd.dvd.mul_right (pow_dvd_pow 2 le) count
    refine ⟨?_, by omega, ?_⟩
    · rw [strideBits, narrowEq, Nat.log_pow (by norm_num)]
    · rw [mul_add, Nat.mul_div_cancel' divides]
      ring
  · simp only [if_neg wideK, part, if_true, zero_mul, Nat.sub_zero, Nat.zero_div, zero_add,
      if_pos (show k < wide size by omega)]
    refine ⟨by rw [strideBits, wide, Nat.log_pow (by norm_num)], by omega, by ring⟩

/-- From `2^σ·a < 2^m`, `a` fits the top `m - σ` coordinates. -/
theorem lt_pow_sub {a σ m : ℕ} (lt : 2 ^ σ * a < 2 ^ m) : a < 2 ^ (m - σ) := by
  by_cases le : σ ≤ m
  · rw [← Nat.pow_sub_mul_pow 2 le, mul_comm] at lt
    exact Nat.lt_of_mul_lt_mul_right lt
  · have : 2 ^ m < 2 ^ σ := Nat.pow_lt_pow_right (by norm_num) (by omega)
    rw [Nat.sub_eq_zero_of_le (by omega), pow_zero]
    by_contra big
    have : 2 ^ σ ≤ 2 ^ σ * a := Nat.le_mul_of_pos_right _ (by omega)
    omega

/-- A part with blocks fits inside the parts' end. -/
theorem part_stride_le (size count p : ℕ) (blocks : count ≠ 0) :
    (part size count p).stride ≤ partsEnd size count := by
  unfold part partsEnd
  split_ifs
  · exact (Nat.le_mul_of_pos_right _ (by omega)).trans (Nat.le_add_right _ _)
  · exact (Nat.le_mul_of_pos_right _ (by omega)).trans (Nat.le_add_left _ _)

/-- The eq weight at a slot: the part's low table at the local index times
the high coordinates at `start / stride + b`. -/
theorem eqAt_slot {K : Type} [CommRing K] (point : List K) (size count b k : ℕ) (inside : k < size) :
    eqAt point (slot size count b k) =
      eqAt (point.take (strideBits (part size count (partIndex size k).1).stride)) (partIndex size k).2 *
        eqAt (point.drop (strideBits (part size count (partIndex size k).1).stride))
          ((part size count (partIndex size k).1).start / (part size count (partIndex size k).1).stride + b) := by
  obtain ⟨stride, low, place⟩ := slot_split size count b k inside
  rw [place]
  generalize part size count (partIndex size k).1 = P at stride low ⊢
  generalize strideBits P.stride = σ at stride ⊢
  have := eqAt_split point σ (partIndex size k).2 (P.start / P.stride + b) (stride ▸ low)
  rwa [← stride] at this

/-- The high index of a slot fits the coordinates above the part's low
ones. -/
theorem slot_fits (size count b k m : ℕ) (inside : k < size) (lt : slot size count b k < 2 ^ m) :
    (part size count (partIndex size k).1).start / (part size count (partIndex size k).1).stride + b <
      2 ^ (m - strideBits (part size count (partIndex size k).1).stride) := by
  obtain ⟨stride, -, place⟩ := slot_split size count b k inside
  apply lt_pow_sub
  rw [← stride]
  rw [place] at lt
  exact lt_of_le_of_lt (Nat.le_add_left _ _) lt

end Parts

section Halves

variable {K : Type} [CommRing K]

theorem drop_last (ry : List K) (n : ℕ) (length : ry.length = n + 1) : ry.drop n = [ry.getD n 0] := by
  rw [List.drop_eq_getElem_cons (by omega), List.drop_eq_nil_of_le (by omega)]
  simp [List.getD_eq_getElem?_getD, show n < ry.length by omega]

/-- A column of `w`: the eq weight is the low `n` coordinates' times `1 - ry_n`. -/
theorem eqAt_low_half (ry : List K) (n col : ℕ) (length : ry.length = n + 1) (lt : col < 2 ^ n) :
    eqAt ry col = eqAt (ry.take n) col * (1 - ry.getD n 0) := by
  have := eqAt_split ry n col 0 lt
  rw [mul_zero, add_zero, drop_last ry n length] at this
  rw [this]
  simp [eqAt]

/-- The constant one's column `2^n`: `Π_{t<n} (1 - ry_t)·ry_n`. -/
theorem eqAt_one (ry : List K) (n : ℕ) (length : ry.length = n + 1) :
    eqAt ry (2 ^ n) = ((ry.take n).map fun y => 1 - y).prod * ry.getD n 0 := by
  have := eqAt_split ry n 0 1 (pow_pos (by norm_num) _)
  rw [mul_one, zero_add, drop_last ry n length] at this
  rw [this, eqAt_zero]
  simp [eqAt]

end Halves

section Closed

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- The low table of a part, `eq_table(point[..bits(stride)])`, at `i`
(`row_low`, `cell_low`, `layout.rs:188-189`). -/
def lowEq (point : List K) (stride i : ℕ) : K := eqAt (point.take (strideBits stride)) i

/-- An entry's row part, cell part and term in `sums` (`layout.rs:192-204`).
Cell part `2` is the constant one, with value `1`. -/
def entryTerm (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (e : TemplateEntry F) : ℕ × ℕ × K :=
  match e.cell with
  | none =>
      ((partIndex T.rows e.row).1, 2,
        ρ ^ (e.matrix : ℕ) * lowEq rx (part T.rows count (partIndex T.rows e.row).1).stride (partIndex T.rows e.row).2 *
          1 * ι e.coef)
  | some k =>
      ((partIndex T.rows e.row).1, (partIndex T.cells k).1,
        ρ ^ (e.matrix : ℕ) * lowEq rx (part T.rows count (partIndex T.rows e.row).1).stride (partIndex T.rows e.row).2 *
          lowEq ry (part T.cells count (partIndex T.cells k).1).stride (partIndex T.cells k).2 * ι e.coef)

/-- `sums[row part][cell part]` (`layout.rs:190-204`). -/
def partSum (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (rp cp : ℕ) : K :=
  (T.entries.map fun e =>
    if (entryTerm T count rx ry ρ e).1 = rp ∧ (entryTerm T count rx ry ρ e).2.1 = cp then
      (entryTerm T count rx ry ρ e).2.2
    else 0).sum

/-- The factor of row part `rp` and cell part `cp` (`layout.rs:210-221`):
`one_column·diagonal([x])` for the constant one, `w_half·diagonal([x, y])`
for a cell part, with `w_half = 1 - ry_n` and
`one_column = Π_{t<n} (1 - ry_t)·ry_n` (`layout.rs:178-179`). -/
def partDiagonal (T : Template F) (s : Counts) (rx ry : List K) (rp cp : ℕ) : K :=
  if cp = 2 then
    ((ry.take (Layout.cellVars T s)).map fun y => 1 - y).prod * ry.getD (Layout.cellVars T s) 0 *
      diagonal [(rx.drop (strideBits (part T.rows s.blocks rp).stride),
        (part T.rows s.blocks rp).start / (part T.rows s.blocks rp).stride)] s.blocks
  else
    (1 - ry.getD (Layout.cellVars T s) 0) *
      diagonal [(rx.drop (strideBits (part T.rows s.blocks rp).stride),
          (part T.rows s.blocks rp).start / (part T.rows s.blocks rp).stride),
        ((ry.take (Layout.cellVars T s)).drop (strideBits (part T.cells s.blocks cp).stride),
          (part T.cells s.blocks cp).start / (part T.cells s.blocks cp).stride)] s.blocks

/-- `block_part` (`layout.rs:172-227`) for the one block kind: no blocks
give `0`; otherwise each row part with slots adds `sum·diagonal` for each
nonzero `sum`. -/
def blockPart [DecidableEq K] (T : Template F) (s : Counts) (rx ry : List K) (ρ : K) : K :=
  if s.blocks = 0 then 0
  else
    ∑ rp ∈ Finset.range 2, if (part T.rows s.blocks rp).stride = 0 then 0
      else
        ∑ cp ∈ Finset.range 3, if partSum T s.blocks rx ry ρ rp cp = 0 then 0
          else partSum T s.blocks rx ry ρ rp cp * partDiagonal T s rx ry rp cp

/-- A sum over the blocks' copies of a list, one list entry at a time. -/
theorem sum_flatMap_range {α β : Type} (l : List α) (g : ℕ → α → β) (h : β → K) :
    ∀ count, (((List.range count).flatMap fun b => l.map (g b)).map h).sum =
      (l.map fun a => ∑ b ∈ Finset.range count, h (g b a)).sum
  | 0 => by simp
  | count + 1 => by
      rw [List.range_succ, List.flatMap_append, List.map_append, List.sum_append, sum_flatMap_range l g h count]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_map, Finset.sum_range_succ]
      rw [← List.sum_map_add]
      rfl

/-- The entry sum over the blocks, one template entry at a time. -/
theorem matrixValue_blockEntries (T : Template F) (s : Counts) (rx ry : List K) (ρ : K) :
    matrixValue (Layout.blockEntries T s) rx ry ρ =
      (T.entries.map fun e => ∑ b ∈ Finset.range s.blocks,
        ρ ^ (e.matrix : ℕ) * eqAt rx (Layout.blockRow T s b e.row) * eqAt ry (Layout.entryColumn T s b e.cell) *
          ι e.coef).sum := by
  unfold matrixValue Layout.blockEntries
  exact sum_flatMap_range _ _ _ _

/-- One entry over every block is its term in `sums` times its parts'
diagonal factor. -/
theorem entry_blocks (T : Template F) (s : Counts) (rx ry : List K) (ρ : K) (wf : T.WellFormed)
    (rxLength : rx.length = Layout.rowVars T s) (ryLength : ry.length = Layout.cellVars T s + 1)
    (blocks : s.blocks ≠ 0) (e : TemplateEntry F) (member : e ∈ T.entries) :
    ∑ b ∈ Finset.range s.blocks,
        ρ ^ (e.matrix : ℕ) * eqAt rx (Layout.blockRow T s b e.row) * eqAt ry (Layout.entryColumn T s b e.cell) *
          ι e.coef =
      (entryTerm T s.blocks rx ry ρ e).2.2 *
        partDiagonal T s rx ry (entryTerm T s.blocks rx ry ρ e).1 (entryTerm T s.blocks rx ry ρ e).2.1 := by
  have rowIn := wf.row_lt e member
  have rowsEnd : partsEnd T.rows s.blocks ≤ 2 ^ Layout.rowVars T s :=
    (Nat.le_add_right _ _).trans (le_pow_variablesFor _)
  have cellsEnd : partsEnd T.cells s.blocks ≤ 2 ^ Layout.cellVars T s :=
    ((Nat.le_add_right _ _).trans (le_max_left _ _)).trans (le_pow_variablesFor _)
  have rowFit : ∀ b < s.blocks,
      (part T.rows s.blocks (partIndex T.rows e.row).1).start /
          (part T.rows s.blocks (partIndex T.rows e.row).1).stride + b <
        2 ^ (rx.drop (strideBits (part T.rows s.blocks (partIndex T.rows e.row).1).stride)).length := by
    intro b lt
    rw [List.length_drop, rxLength]
    exact slot_fits _ _ _ _ _ rowIn ((slot_lt lt rowIn).trans_le rowsEnd)
  have rowEq (b : ℕ) := eqAt_slot rx T.rows s.blocks b e.row rowIn
  cases cell : e.cell with
  | none =>
      have xEq := diagonal_eq [(rx.drop (strideBits (part T.rows s.blocks (partIndex T.rows e.row).1).stride),
        (part T.rows s.blocks (partIndex T.rows e.row).1).start /
          (part T.rows s.blocks (partIndex T.rows e.row).1).stride)] s.blocks
        (by simpa using rowFit)
      simp only [entryTerm, cell, partDiagonal, if_true, Layout.entryColumn, Layout.one, Layout.blockRow,
        eqAt_one ry _ ryLength, rowEq, xEq, Finset.mul_sum, lowEq]
      apply Finset.sum_congr rfl
      intro b _
      simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil]
      ring
  | some k =>
      have cellIn := wf.cell_lt e member k cell
      have narrowSide : (partIndex T.cells k).1 ≠ 2 := by
        unfold partIndex
        split_ifs <;> simp
      have strideLe : strideBits (part T.cells s.blocks (partIndex T.cells k).1).stride ≤ Layout.cellVars T s := by
        have stride := (slot_split T.cells s.blocks 0 k cellIn).1
        have le := (part_stride_le T.cells s.blocks (partIndex T.cells k).1 blocks).trans cellsEnd
        rw [stride] at le
        exact (Nat.pow_le_pow_iff_right (by norm_num)).mp le
      have takeLength : (ry.take (Layout.cellVars T s)).length = Layout.cellVars T s := by
        rw [List.length_take, ryLength]
        omega
      have colEq : ∀ b < s.blocks, eqAt ry (Layout.blockCell T s b k) =
          (1 - ry.getD (Layout.cellVars T s) 0) *
            (lowEq ry (part T.cells s.blocks (partIndex T.cells k).1).stride (partIndex T.cells k).2 *
              eqAt ((ry.take (Layout.cellVars T s)).drop
                  (strideBits (part T.cells s.blocks (partIndex T.cells k).1).stride))
                ((part T.cells s.blocks (partIndex T.cells k).1).start /
                  (part T.cells s.blocks (partIndex T.cells k).1).stride + b)) := by
        intro b lt
        have inside := (slot_lt lt cellIn).trans_le cellsEnd
        rw [Layout.blockCell, eqAt_low_half ry _ _ ryLength inside, eqAt_slot _ T.cells s.blocks b k cellIn,
          List.take_take, min_eq_left strideLe, lowEq]
        ring
      have yFit : ∀ b < s.blocks,
          (part T.cells s.blocks (partIndex T.cells k).1).start /
              (part T.cells s.blocks (partIndex T.cells k).1).stride + b <
            2 ^ ((ry.take (Layout.cellVars T s)).drop
              (strideBits (part T.cells s.blocks (partIndex T.cells k).1).stride)).length := by
        intro b lt
        rw [List.length_drop, takeLength]
        exact slot_fits _ _ _ _ _ cellIn ((slot_lt lt cellIn).trans_le cellsEnd)
      have xyEq := diagonal_eq
        [(rx.drop (strideBits (part T.rows s.blocks (partIndex T.rows e.row).1).stride),
          (part T.rows s.blocks (partIndex T.rows e.row).1).start /
            (part T.rows s.blocks (partIndex T.rows e.row).1).stride),
        ((ry.take (Layout.cellVars T s)).drop (strideBits (part T.cells s.blocks (partIndex T.cells k).1).stride),
          (part T.cells s.blocks (partIndex T.cells k).1).start /
            (part T.cells s.blocks (partIndex T.cells k).1).stride)] s.blocks
        (by
          intro track member b lt
          simp only [List.mem_cons, List.not_mem_nil, or_false] at member
          rcases member with rfl | rfl
          · exact rowFit b lt
          · exact yFit b lt)
      simp only [entryTerm, cell, partDiagonal, if_neg narrowSide, Layout.entryColumn, Layout.blockRow, rowEq,
        xyEq, Finset.mul_sum, lowEq]
      apply Finset.sum_congr rfl
      intro b member
      rw [colEq b (Finset.mem_range.mp member)]
      simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, lowEq]
      ring

/-- An entry's row part is `0` or `1`, and its cell part `0`, `1` or `2`. -/
theorem entryTerm_parts (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (e : TemplateEntry F) :
    (entryTerm T count rx ry ρ e).1 < 2 ∧ (entryTerm T count rx ry ρ e).2.1 < 3 := by
  unfold entryTerm partIndex
  cases e.cell <;> dsimp only <;> constructor <;> (try split_ifs) <;> norm_num

/-- An entry's row part is its row's part. -/
theorem entryTerm_row (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (e : TemplateEntry F) :
    (entryTerm T count rx ry ρ e).1 = (partIndex T.rows e.row).1 := by
  unfold entryTerm
  split <;> rfl

/-- Entries grouped by their row part and cell part. -/
theorem sum_by_parts {α : Type} (l : List α) (key : α → ℕ × ℕ × K) (D : ℕ → ℕ → K)
    (inside : ∀ a ∈ l, (key a).1 < 2 ∧ (key a).2.1 < 3) :
    (l.map fun a => (key a).2.2 * D (key a).1 (key a).2.1).sum =
      ∑ rp ∈ Finset.range 2, ∑ cp ∈ Finset.range 3,
        (l.map fun a => if (key a).1 = rp ∧ (key a).2.1 = cp then (key a).2.2 else 0).sum * D rp cp := by
  induction l with
  | nil => simp
  | cons a l ih =>
      simp only [List.map_cons, List.sum_cons, add_mul, Finset.sum_add_distrib]
      rw [ih (fun a' member => inside a' (List.mem_cons_of_mem _ member))]
      congr 1
      obtain ⟨rowIn, cellIn⟩ := inside a List.mem_cons_self
      rw [Finset.sum_eq_single_of_mem (key a).1 (Finset.mem_range.mpr rowIn)
          (fun rp _ ne => Finset.sum_eq_zero fun cp _ => by rw [if_neg (fun h => ne h.1.symm), zero_mul]),
        Finset.sum_eq_single_of_mem (key a).2.1 (Finset.mem_range.mpr cellIn)
          (fun cp _ ne => by rw [if_neg (fun h => ne h.2.symm), zero_mul]),
        if_pos ⟨rfl, rfl⟩]

/-- A row part without slots holds no entry. -/
theorem partSum_empty (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (wf : T.WellFormed) (rp cp : ℕ)
    (small : rp < 2) (empty : (part T.rows count rp).stride = 0) : partSum T count rx ry ρ rp cp = 0 := by
  have one : rp = 1 := by
    by_contra other
    have zero : rp = 0 := by omega
    rw [zero, part, if_pos rfl] at empty
    exact (wide_pos T.rows).ne' empty
  rw [one, part, if_neg one_ne_zero] at empty
  have short : T.rows ≤ wide T.rows := by
    unfold narrow at empty
    split_ifs at empty with rest
    · omega
    · exact absurd empty (pow_pos (by norm_num) _).ne'
  unfold partSum
  apply List.sum_eq_zero
  intro x member
  obtain ⟨e, entry, rfl⟩ := List.mem_map.mp member
  rw [if_neg]
  rintro ⟨rowPart, -⟩
  have below := wf.row_lt e entry
  have notWide : ¬ wide T.rows ≤ e.row := by omega
  rw [entryTerm_row, partIndex, if_neg notWide] at rowPart
  omega

/-- `block_part` with its skips taken out: every row part and cell part
adds `sum·diagonal`. -/
theorem blockPart_eq_sum [DecidableEq K] (T : Template F) (s : Counts) (rx ry : List K) (ρ : K)
    (wf : T.WellFormed) (blocks : s.blocks ≠ 0) :
    blockPart T s rx ry ρ = ∑ rp ∈ Finset.range 2, ∑ cp ∈ Finset.range 3,
      partSum T s.blocks rx ry ρ rp cp * partDiagonal T s rx ry rp cp := by
  unfold blockPart
  rw [if_neg blocks]
  apply Finset.sum_congr rfl
  intro rp member
  split_ifs with empty
  · symm
    apply Finset.sum_eq_zero
    intro cp _
    rw [partSum_empty T s.blocks rx ry ρ wf rp cp (Finset.mem_range.mp member) empty, zero_mul]
  · apply Finset.sum_congr rfl
    intro cp _
    split_ifs with zero
    · rw [zero, zero_mul]
    · rfl

/-- **The block part.** `block_part` is the verifier's entry sum over every
block's template entries. -/
theorem blockPart_eq [DecidableEq K] (T : Template F) (s : Counts) (rx ry : List K) (ρ : K) (wf : T.WellFormed)
    (rxLength : rx.length = Layout.rowVars T s) (ryLength : ry.length = Layout.cellVars T s + 1) :
    blockPart T s rx ry ρ = matrixValue (Layout.blockEntries T s) rx ry ρ := by
  by_cases blocks : s.blocks = 0
  · unfold blockPart Layout.blockEntries
    simp [blocks, matrixValue]
  · rw [blockPart_eq_sum T s rx ry ρ wf blocks, matrixValue_blockEntries,
      List.map_congr_left (fun e member => entry_blocks T s rx ry ρ wf rxLength ryLength blocks e member),
      sum_by_parts T.entries (entryTerm T s.blocks rx ry ρ) (partDiagonal T s rx ry)
        (fun e _ => entryTerm_parts T s.blocks rx ry ρ e)]
    rfl

end Closed

end CompressionLean.Shrink
