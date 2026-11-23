import NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.Tensor

/-! The relation weight of one fold at the point `r_2` of the fused sum-check, without the table
of the witness cells (Akita revision `d50f700`: `RelationMatrixEvaluator::eval_flat_at_point` of
`akita-verifier/src/relation/evaluation.rs` and the setup scan of `setup_contribution/`; ePrint
2026/1983, Lemma 7.3 and (117) to (123)).

- `eqCells r` is the table `ẽq(r, i)` for `i < 2^|r|`, and `CellEq` is `ẽq(r_2, ·)` as the
  product of two such tables, so that a cell costs one product.
- A `Term` is a set of occurrences `(c·s, A, addr)` of the ring rows (110) that share one read
  vector: the multipliers `(c, A)` of the rows with the row coefficient `c = ẽq(τ_1, row)`, and
  the reads `(s, addr)` with a scalar `s ∈ Z_q`. `termValue` is
  `Σ_{(c,A)} c·Σ_{(s,addr)} s·Σ_j κ_{A,α}(j)·ẽq(r_2, addr j)`, with the residue kernels `κ` of
  Akita (117). It reads each coefficient of each multiplier once: by Lemma 7.3 the inner sum is
  `Σ_i A_i·K_i` with the kernels `K_i = Σ_j e_j·σ(i + j)` of the read vector `e`, which
  `kernel` computes with `K_{i+1} = α·K_i − e_{d−1−i}·(α^d + 1)`. This is the setup scan of the
  code: one product for each element of the setup matrices. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- `s·z` for `s ∈ Z_q`. -/
def Ext.scale (s : Fq) (z : Ext) : Ext := ⟨s.mul z.c0, s.mul z.c1⟩

/-- `z^n` by squares. -/
def Ext.pow (z : Ext) (n : ℕ) : Ext :=
  if n = 0 then Ext.one else
    if n % 2 = 1 then Ext.pow (z * z) (n / 2) * z else Ext.pow (z * z) (n / 2)
termination_by n
decreasing_by all_goals omega

/-- `Σ_{x ∈ l} f x` in `E`. -/
def sumList {α : Type} (l : List α) (f : α → Ext) : Ext := l.foldr (fun x acc => f x + acc) Ext.zero

/-- The table `ẽq(r, i)` for `i < 2^|r|`, with the first coordinate on the lowest bit of `i`. -/
def eqCells : List Ext → Array Ext
  | [] => #[Ext.one]
  | r :: rest =>
      let t := eqCells rest
      Array.ofFn (n := 2 * t.size) fun i =>
        (if i.val % 2 = 1 then r else Ext.one - r) * t.getD (i.val / 2) Ext.zero

/-- `ẽq(r, ·)` as the product of the table of the first `bits` coordinates and the table of the
other coordinates. -/
structure CellEq where
  bits : ℕ
  low : Array Ext
  high : Array Ext

def CellEq.of (r : List Ext) (bits : ℕ) : CellEq :=
  ⟨bits, eqCells (r.take bits), eqCells (r.drop bits)⟩

/-- `ẽq(r, x)` for a cell `x < 2^|r|`. -/
def CellEq.get (t : CellEq) (x : ℕ) : Ext :=
  t.low.getD (x % 2 ^ t.bits) Ext.zero * t.high.getD (x / 2 ^ t.bits) Ext.zero

/-- The split of a point of `μ` coordinates with the low table of about `μ/2` coordinates. -/
def CellEq.half (r : List Ext) : CellEq := CellEq.of r (r.length / 2)

/-- One read of a term: the scalar and the address map of the coefficients. -/
abbrev Read := Fq × (ℕ → Option ℕ)

/-- A set of occurrences of ring rows of degree `degree` that share the read vector `reads`. -/
structure Term where
  degree : ℕ
  muls : List (Ext × Array Fq)
  reads : List Read

/-- Entry `j` of the read vector: `e_j = Σ_{(s, addr)} s·ẽq(r_2, addr j)`; the address `none`
reads 0. -/
def readAt (eq : ℕ → Ext) (reads : List Read) (j : ℕ) : Ext :=
  reads.foldr (fun r acc => match r.2 j with
    | some x => Ext.scale r.1 (eq x) + acc
    | none => acc) Ext.zero

/-- `Σ_{k<n} e_{i+k}·α^k` by Horner's rule. -/
def hornerFrom (α : Ext) (e : Array Ext) : ℕ → ℕ → Ext
  | _, 0 => Ext.zero
  | i, n + 1 => e.getD i Ext.zero + α * hornerFrom α e (i + 1) n

/-- The kernels `K_i, …, K_{i+n−1}` from `K_i = k` by `K_{i+1} = α·K_i − e_{d−1−i}·(α^d + 1)`. -/
def kernelFrom (d : ℕ) (α αd1 : Ext) (e : Array Ext) : ℕ → ℕ → Ext → List Ext
  | _, 0, _ => []
  | i, n + 1, k => k :: kernelFrom d α αd1 e (i + 1) n (α * k - e.getD (d - 1 - i) Ext.zero * αd1)

/-- The kernels `K_i = Σ_{j<d} e_j·σ(i + j)` for `i < d`, with `σ(k) = α^k` for `k < d` and
`σ(k) = −α^{k−d}` for `d ≤ k`: the value at `α` of `X^i·e(X) mod (X^d + 1)`. -/
def kernel (d : ℕ) (α αd1 : Ext) (e : Array Ext) : Array Ext :=
  (kernelFrom d α αd1 e 0 d (hornerFrom α e 0 d)).toArray

/-- `Σ_{i<d} A_i·K_i`. -/
def dotKernel (d : ℕ) (a : Array Fq) (K : Array Ext) : Ext :=
  sumExt (fun i => Ext.scale (Poly.get a i) (K.getD i Ext.zero)) d

/-- `Σ_{(c,A)} c·Σ_{(s,addr)} s·Σ_j κ_{A,α}(j)·ẽq(r_2, addr j)`. -/
def termValue (α : Ext) (eq : ℕ → Ext) (t : Term) : Ext :=
  let e := Array.ofFn (n := t.degree) fun j => readAt eq t.reads j.val
  let K := kernel t.degree α (Ext.pow α t.degree + Ext.one) e
  sumList t.muls fun m => m.1 * dotKernel t.degree m.2 K

/-- The sum of the values of the terms. -/
def termsValue (α : Ext) (eq : ℕ → Ext) (terms : List Term) : Ext :=
  sumList terms (termValue α eq)

/-- `Π_k (a_k·b_k + (1 − a_k)·(1 − b_k))`: `ẽq(a, b)` of two points of the same length. -/
def eqPoints : List Ext → List Ext → Ext
  | a :: as, b :: bs => (a * b + (Ext.one - a) * (Ext.one - b)) * eqPoints as bs
  | _, _ => Ext.one

end NightstreamFPrime.Spec.Compression.Akita.Replay
