import CompressionLean.Ports.Shrink.Diagonal
import CompressionLean.Ports.Shrink.Layout

namespace CompressionLean.Shrink

open CompressionLean.Lib

structure Part where
  start : ℕ
  stride : ℕ

def part (size count p : ℕ) : Part :=
  if p = 0 then ⟨0, wide size⟩ else ⟨wide size * count, narrow size⟩

def strideBits (stride : ℕ) : ℕ := Nat.log 2 stride

def partIndex (size k : ℕ) : ℕ × ℕ :=
  ((if wide size ≤ k then 1 else 0), k - (if wide size ≤ k then 1 else 0) * wide size)

section Closed

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

def lowEq (point : List K) (stride i : ℕ) : K := eqAt (point.take (strideBits stride)) i

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

def partSum (T : Template F) (count : ℕ) (rx ry : List K) (ρ : K) (rp cp : ℕ) : K :=
  (T.entries.map fun e =>
    if (entryTerm T count rx ry ρ e).1 = rp ∧ (entryTerm T count rx ry ρ e).2.1 = cp then
      (entryTerm T count rx ry ρ e).2.2
    else 0).sum

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

def blockPart [DecidableEq K] (T : Template F) (s : Counts) (rx ry : List K) (ρ : K) : K :=
  if s.blocks = 0 then 0
  else
    ∑ rp ∈ Finset.range 2, if (part T.rows s.blocks rp).stride = 0 then 0
      else
        ∑ cp ∈ Finset.range 3, if partSum T s.blocks rx ry ρ rp cp = 0 then 0
          else partSum T s.blocks rx ry ρ rp cp * partDiagonal T s rx ry rp cp

end Closed

end CompressionLean.Shrink
