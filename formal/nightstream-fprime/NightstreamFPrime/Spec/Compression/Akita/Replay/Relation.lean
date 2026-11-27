import NightstreamFPrime.Spec.Compression.Akita.Replay.Layout
import NightstreamFPrime.Spec.Compression.Akita.Replay.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.RowBatch

/-! The claims of Stage 2 of one fold of the pinned Akita verifier (Akita revision `d50f700`:
`fold/mod.rs`, `stages/stage2.rs`, `akita-types/src/proof/relation.rs`,
`akita-params/src/layout/opening_layout.rs`; the glue of
`crates/ns-compress/tests/unit/akita_vectors/replay.rs`, `fold`).

- `eqIndex τ i` is `ẽq(τ, i)` with the coordinate `t` of `τ` at the bit `t` of `i`
  (`eq_eval_at_index`), and `ringEval α c` is `Σ_j α^j·c_j` (`scalar_powers`, `dot_base`).
- `rhsEntries` gives each public row `i` of the right-hand side `y` of the relation with its value
  `y(α)_i`, and `batchEntries τ` is `Σ_i ẽq(τ, i)·y(α)_i` over these rows: the relation claim of
  `relation_claim_from_compressed_rhs_extension`, whose other rows are zero.
- `stage2` gives `opening_target = Σ_i ρ_i·c_i` of the row coefficients `ρ` and the claims `c`
  (`batched_eval_target`; at a level with evaluation trace the final claims of the tensor
  reduction), `relation_claim`, `opening_claim = ẽq(τ_1, N)·target` of the opening row `N`
  (`relation_row_weight`), and the input claim of the fused sum-check: the sum of the two claims
  and `rest = β·image + η·virtual`.

The specification of a row is `reducedEval` (`ringEval_into`); the specification of the batch is
`FieldRows.batchTarget` (`batchEntries_batchTarget`), for field rows whose targets are the public
rows: `stage2_fused` gives the fused claim as `batchTarget` of the rows and the opening row plus
`rest`. `Replay/Rows.lean` proves that the rows of the reductions of the claims have these
targets. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

def eqIndex : List Ext → ℕ → Ext
  | [], _ => Ext.one
  | r :: rest, i => (if i % 2 = 1 then r else Ext.one - r) * eqIndex rest (i / 2)

def ringEval (α : Ext) : List Fq → Ext
  | [] => Ext.zero
  | c :: rest => Ext.ofBase c + α * ringEval α rest

def batchEntries (τ : List Ext) : List (ℕ × Ext) → Ext
  | [] => Ext.zero
  | (i, v) :: rest => eqIndex τ i * v + batchEntries τ rest

/-- The coefficients of the element `r` of degree `degree` of `data`. -/
def chunk (degree : ℕ) (data : Array Fq) (r : ℕ) : List Fq :=
  (data.extract (degree * r) (degree * (r + 1))).toList

/-- The rows of the runs with the coefficients `data run` of each run: the index and `y(α)_i` of
each row. -/
def rhsEntries (α : Ext) (data : RhsRun → Array Fq) (runs : List RhsRun) : List (ℕ × Ext) :=
  runs.flatMap fun run => (List.range run.count).map fun r =>
    (run.row + r, ringEval α (chunk run.degree (data run) r))

/-- `Σ_i a_i·b_i`. -/
def dot (a b : Array Ext) : Ext := (a.zip b).foldl (fun sum (x, y) => sum + x * y) Ext.zero

structure Stage2 where
  target : Ext
  relation : Ext
  opening : Ext
  fused : Ext

def stage2 (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (openingRow : ℕ) (rows claims : Array Ext) (rest : Ext) : Stage2 :=
  let target := dot rows claims
  let relation := batchEntries τ (rhsEntries α data runs)
  let opening := eqIndex τ openingRow * target
  ⟨target, relation, opening, relation + opening + rest⟩

section Specification

theorem mle_delta_at {K : Type} [CommRing K] (p : List K) {j : ℕ} (below : j < 2 ^ p.length)
    (c : K) : mle p (fun i => if j = i then c else 0) = eqAt p j * c := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_range, below, ite_true]

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

/-- The element of `E` with the coordinates of `z` in the basis `(1, u)`. -/
def Ext.into (z : Ext) : E := Degree2.coords.symm z.toE

theorem Ext.into_add (a b : Ext) : (a + b).into = (a.into + b.into : E) := by
  simp [Ext.into, Ext.toE_add]

theorem Ext.into_sub (a b : Ext) : (a - b).into = (a.into - b.into : E) := by
  simp [Ext.into, Ext.toE_sub]

theorem Ext.into_mul (a b : Ext) : (a * b).into = (a.into * b.into : E) := by
  simp [Ext.into, Ext.toE_mul]

theorem Ext.into_zero : (Ext.zero.into : E) = 0 := by
  simp [Ext.into, Ext.toE_zero]

theorem Ext.into_one : (Ext.one.into : E) = 1 := by
  simp [Ext.into, Ext.toE_one]

theorem Ext.into_ofBase (a : Fq) : ((Ext.ofBase a).into : E) = algebraMap Zq E a.toZq := by
  simp [Ext.into, Ext.toE_ofBase]

theorem re_into (z : Ext) : Degree2.re (z.into : E) = z.c0.toZq := by
  simp [Ext.into, Degree2.re, Ext.toE]

theorem im_into (z : Ext) : Degree2.im (z.into : E) = z.c1.toZq := by
  simp [Ext.into, Degree2.im, Ext.toE]

/-- `Ext.into` maps the operations of `Ext` to those of `E`. -/
theorem degree2_extHom : ExtHom (Ext.into (E := E)) :=
  ⟨Ext.into_zero, Ext.into_one, Ext.into_add, Ext.into_sub, Ext.into_mul⟩

theorem eqIndex_into (τ : List Ext) (i : ℕ) :
    ((eqIndex τ i).into : E) = eqAt (τ.map Ext.into) i := by
  induction τ generalizing i with
  | nil => simp [eqIndex, eqAt, Ext.into_one]
  | cons r rest ih =>
      simp only [eqIndex, eqAt, List.map_cons]
      rw [Ext.into_mul, ih]
      split
      · rfl
      · rw [Ext.into_sub, Ext.into_one]

/-- `ringEval` is `Σ_j c_j·α^j`. -/
theorem ringEval_into_sum (α : Ext) (c : List Fq) :
    ((ringEval α c).into : E) =
      ∑ j ∈ Finset.range c.length, (c.getD j Fq.zero).toZq • (α.into : E) ^ j := by
  induction c with
  | nil => simp [ringEval, Ext.into_zero]
  | cons a rest ih =>
      rw [ringEval, Ext.into_add, Ext.into_mul, Ext.into_ofBase, ih, List.length_cons,
        Finset.sum_range_succ', Finset.mul_sum]
      simp only [List.getD_cons_succ, List.getD_cons_zero, pow_zero, Algebra.smul_def, mul_one,
        pow_succ]
      rw [add_comm]
      congr 1
      refine Finset.sum_congr rfl fun j _ => ?_
      ring

/-- A row of at most `d` coefficients has the value `reducedEval α` of its ring element in
`R_{q,d}` (Akita Section 7.3). -/
theorem ringEval_into (α : Ext) (c : List Fq) {d : ℕ} [NeZero d] (length : c.length ≤ d) :
    ((ringEval α c).into : E) =
      reducedEval (d := d) (α.into : E) (ofCoeffs fun j => (c.getD j Fq.zero).toZq) := by
  rw [ringEval_into_sum, reducedEval_apply, aeval_rep]
  refine Finset.sum_subset (Finset.range_subset_range.mpr length) (fun j _ outside => ?_) |>.trans
    (Finset.sum_congr rfl fun j member => ?_)
  · rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_none (Nat.le_of_not_lt (outside ∘ Finset.mem_range.mpr)), Option.getD_none,
      Fq.toZq_zero, zero_smul]
  · rw [coeff_ofCoeffs _ (Finset.mem_range.mp member)]

/-- `batchEntries` is the multilinear batch of the table whose entry `i` is the sum of the values
listed at `i`. -/
theorem batchEntries_into (τ : List Ext) (entries : List (ℕ × Ext))
    (below : ∀ e ∈ entries, e.1 < 2 ^ τ.length) :
    ((batchEntries τ entries).into : E) =
      mle (τ.map Ext.into) fun i => ((entries.filter (·.1 == i)).map (·.2.into)).sum := by
  induction entries with
  | nil => simp [batchEntries, Ext.into_zero, mle_const]
  | cons e rest ih =>
      obtain ⟨j, v⟩ := e
      have split : (fun i => (((((j, v) :: rest).filter (·.1 == i)).map (·.2.into)).sum : E)) =
          fun i => (if j = i then (v.into : E) else 0) +
            ((rest.filter (·.1 == i)).map (·.2.into)).sum := by
        funext i
        by_cases same : j = i
        · simp [same]
        · simp [same]
      rw [split, mle_add, mle_delta_at _ (by simpa using below (j, v) (by simp)),
        ← ih fun e member => below e (List.mem_cons_of_mem _ member), ← eqIndex_into,
        ← Ext.into_mul, ← Ext.into_add]
      rfl

/-- `batchEntries` is `V_{α,τ_1} = FieldRows.batchTarget` of Akita (123) for field rows whose
target at each row is the sum of the values listed at it. -/
theorem batchEntries_batchTarget {F : Type} [Field F] [Algebra Zq F] [Degree2 F]
    (rows : FieldRows F) (τ : List Ext) (entries : List (ℕ × Ext))
    (below : ∀ e ∈ entries, e.1 < 2 ^ τ.length)
    (targets : ∀ i, rows.targetAt i = ((entries.filter (·.1 == i)).map (·.2.into)).sum) :
    rows.batchTarget (τ.map Ext.into) = (batchEntries τ entries).into := by
  rw [batchEntries_into τ entries below, FieldRows.batchTarget]
  exact congrArg _ (funext targets)

theorem batchEntries_append (τ : List Ext) (entries more : List (ℕ × Ext)) :
    ((batchEntries τ (entries ++ more)).into : E) =
      (batchEntries τ entries).into + (batchEntries τ more).into := by
  induction entries with
  | nil => simp [batchEntries, Ext.into_zero]
  | cons e rest ih =>
      obtain ⟨j, v⟩ := e
      simp only [List.cons_append, batchEntries]
      rw [Ext.into_add, Ext.into_add, ih]
      ring

/-- The fused claim of `stage2` is the batch of the public rows of `y` and the opening row with
the target, plus `rest`. -/
theorem stage2_fused (τ : List Ext) (α : Ext) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (openingRow : ℕ) (rows claims : Array Ext) (rest : Ext) :
    (((stage2 τ α data runs openingRow rows claims rest).fused).into : E) =
      (batchEntries τ (rhsEntries α data runs ++ [(openingRow, dot rows claims)])).into +
        rest.into := by
  rw [batchEntries_append]
  simp only [stage2, batchEntries, Ext.into_add, Ext.into_mul, Ext.into_zero, add_zero]

end Specification

end NightstreamFPrime.Spec.Compression.Akita.Replay
