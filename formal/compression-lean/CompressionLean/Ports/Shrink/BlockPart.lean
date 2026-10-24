import CompressionLean.Ports.Shrink.Diagonal
import CompressionLean.Ports.Shrink.Layout

/-!
Owns the port of the closed form `block_part`
(`crates/neo-spartan/src/shrink/layout.rs:172-227`).

Block `b` puts template index `k` at `start + b·stride + k'` of its part
(`slot`). The stride is `2^σ` and the start is a multiple of it, so the eq
weight splits into the low `σ` coordinates at `k'` and the high ones at
`start / stride + b`. The low factor depends on the entry alone:
`block_part` sums the entries per row part and cell part, and `diagonal`
sums the high factors over the blocks.

`block_part` builds the low tables of both parts. A kind whose rows (or
cells) are a power of two has an empty narrow part, of stride `0`; Rust
gives it no variables (`layout.rs`, `bits`).

`Shrink/BlockPart.lean` proves its value.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib

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

end Closed

end CompressionLean.Shrink
