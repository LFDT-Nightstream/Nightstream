import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Verifier.Run
import Mathlib.Data.Rat.Cast.Lemmas

/-! The error sum of Rust `error_bits` (`crates/neo-compress/src/lib.rs`) as an exact rational
(`runError`), from the counts of a layout that it reads. Rust does not compute this sum. It counts
each lookup term as `1/q²` (`runCount`) and returns `runBits`, the largest `k` with
`2^k · runCount ≤ q²`, in integers. The gate of `compression_layout` compares `runBits` with the
caller's security minimum, so no rounding passes a minimum that the sum does not meet.

- `runError_le_runCount`: when `(2^v + 2·bound + 1)² ≤ q` for each table, `runError ≤ runCount/q²`.
- `runBits_spec`: `2^runBits · runCount ≤ q² < 2^(runBits + 1) · runCount`.

At the production parameters, for each layout with `WellFormed`:

- `runCost_le_runError`: `runError` bounds `runCost`, the sum of the bad-set densities of one
  path, so `params_run_soundness` bounds one interactive run by `runError`, and
  `params_run_soundness_bits` by `2^−runBits`. This is what the gate certifies.
- `error_le_runError`: `runError` also bounds `error`, the largest density of one draw, which
  `soundness` multiplies by the query count `q` of the state-restoration experiment. When
  `runBits` is positive, `error_le_runBits` bounds `error` by `2^−runBits`.

Neither bound has a term for the Akita openings or for the Π_CCS run of the fresh claim. -/

namespace NightstreamFPrime.Spec.Compression.Rust

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Verifier NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.Compression.Game

/-- What Rust `error_bits` reads: the witness variables and the bound of each claim with a table,
in claim order, the claims, the public cells, the matrices, the variables of `W` and the block
variables of the key polynomial. -/
structure RunCounts where
  tables : List (ℕ × ℕ)
  claims : ℕ
  publicCells : ℕ
  matrices : ℕ
  witnessVars : ℕ
  keyBlockVars : ℕ

/-- The terms of `error_bits` over `|K| = q²` with `κ = 18` and `D = 64`: the batch of the
equations, the batch of the inverse checks and the sums, the first sum-check, the column point of
`V` and the key sum-check. -/
def RunCounts.extensionTerms (c : RunCounts) : ℕ :=
  18 + 1 + c.matrices + max 63 (c.publicCells - 1) + c.claims - 1 + 2 + c.witnessVars +
    3 * c.witnessVars + 2 + 2 * (6 + c.keyBlockVars)

/-- Rust `error_bits` before `−log₂`: `Σ_tables ((2^v + 2·bound + 1)/q)^4 + extensionTerms/q²`. -/
def runError (c : RunCounts) : ℚ :=
  (c.tables.map fun t => (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ) / fieldOrder) ^ 4).sum +
    (c.extensionTerms : ℚ) / (fieldOrder ^ 2 : ℕ)

/-- Each table has `(2^v + 2·bound + 1)² ≤ q`, so its lookup term is at most `1/q²`. -/
def RunCounts.tablesFit (c : RunCounts) : Bool :=
  c.tables.all fun t => (2 ^ t.1 + (2 * t.2 + 1)) ^ 2 ≤ fieldOrder

/-- The numerator over `q²` of the bound of Rust `error_bits`: `extensionTerms`, and 1 for each
table. -/
def runCount (c : RunCounts) : ℕ :=
  c.extensionTerms + c.tables.length

/-- Rust `error_bits`: the largest `k` with `2^k · runCount ≤ q²`. It is 0 when a table breaks
`tablesFit` or when `runCount > q²`. -/
def runBits (c : RunCounts) : ℕ :=
  if c.tablesFit then Nat.log 2 (fieldOrder ^ 2 / runCount c) else 0

theorem runCount_pos (c : RunCounts) : 0 < runCount c := by
  unfold runCount RunCounts.extensionTerms
  omega

/-- With `tablesFit`, each lookup term is at most `1/q²`, so `runError ≤ runCount/q²`. -/
theorem runError_le_runCount (c : RunCounts) (fit : c.tablesFit = true) :
    runError c ≤ (runCount c : ℚ) / (fieldOrder ^ 2 : ℕ) := by
  have q_pos : (0 : ℚ) < fieldOrder := by norm_num [fieldOrder]
  have each : ∀ x ∈ c.tables.map fun t => (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ) / fieldOrder) ^ 4,
      x ≤ 1 / (fieldOrder ^ 2 : ℕ) := by
    simp only [List.mem_map]
    rintro _ ⟨t, mem, rfl⟩
    have square : (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ)) ^ 2 ≤ fieldOrder := by
      exact_mod_cast of_decide_eq_true (List.all_eq_true.mp fit t mem)
    have fourth : (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ)) ^ 4 ≤ (fieldOrder : ℚ) ^ 2 := by
      calc (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ)) ^ 4 = ((((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ)) ^ 2) ^ 2 := by
            ring
        _ ≤ (fieldOrder : ℚ) ^ 2 := pow_le_pow_left₀ (by positivity) square 2
    rw [div_pow, Nat.cast_pow, div_le_div_iff₀ (by positivity) (by positivity)]
    calc (((2 ^ t.1 + (2 * t.2 + 1) : ℕ) : ℚ)) ^ 4 * (fieldOrder : ℚ) ^ 2 ≤
          (fieldOrder : ℚ) ^ 2 * (fieldOrder : ℚ) ^ 2 := by gcongr
      _ = 1 * (fieldOrder : ℚ) ^ 4 := by ring
  have sum := List.sum_le_length_nsmul _ _ each
  rw [List.length_map, nsmul_eq_mul] at sum
  have split : (runCount c : ℚ) / (fieldOrder ^ 2 : ℕ) =
      c.tables.length * (1 / (fieldOrder ^ 2 : ℕ)) + (c.extensionTerms : ℚ) / (fieldOrder ^ 2 : ℕ) := by
    unfold runCount
    push_cast
    ring
  rw [split]
  unfold runError
  linarith

/-- `runBits` is the largest `k` with `2^k · runCount ≤ q²`. -/
theorem runBits_spec (c : RunCounts) (fit : c.tablesFit = true) (le : runCount c ≤ fieldOrder ^ 2) :
    2 ^ runBits c * runCount c ≤ fieldOrder ^ 2 ∧
      fieldOrder ^ 2 < 2 ^ (runBits c + 1) * runCount c := by
  have pos := runCount_pos c
  have bits : runBits c = Nat.log 2 (fieldOrder ^ 2 / runCount c) := ite_eq_left fit
  rw [bits]
  exact ⟨(Nat.le_div_iff_mul_le pos).mp (Nat.pow_log_le_self 2 (Nat.div_pos le pos).ne'),
    (Nat.div_lt_iff_lt_mul pos).mp (Nat.lt_pow_succ_log_self (by norm_num) _)⟩

/-- When `runBits` is positive, `runError ≤ 2^−runBits`. -/
theorem runError_le_runBits (c : RunCounts) (pos : 0 < runBits c) :
    (runError c : ℝ) ≤ 1 / 2 ^ runBits c := by
  have fit : c.tablesFit = true := by
    by_contra broken
    rw [runBits, ite_eq_right broken] at pos
    exact lt_irrefl _ pos
  have le : runCount c ≤ fieldOrder ^ 2 := by
    by_contra above
    rw [runBits, ite_eq_left fit, Nat.div_eq_of_lt (not_le.mp above), Nat.log_zero_right] at pos
    exact lt_irrefl _ pos
  have spec : runCount c * 2 ^ runBits c ≤ fieldOrder ^ 2 := by
    rw [mul_comm]
    exact (runBits_spec c fit le).1
  have bound := Rat.cast_le (K := ℝ) |>.mpr (runError_le_runCount c fit)
  push_cast at bound
  refine bound.trans ?_
  rw [div_le_div_iff₀ (by norm_num [fieldOrder]) (by positivity), one_mul]
  exact_mod_cast spec

/-- The counts of a production layout: the tables of the two slots, three claims, `m = 320` and
`t = 4`. -/
def productionCounts (layout : Layout) (depths : Fin 2 → ℕ) : RunCounts where
  tables := [((layout.witness 0).vars, bound depths 0), ((layout.witness 1).vars, bound depths 1)]
  claims := 3
  publicCells := 320
  matrices := 4
  witnessVars := layout.witnessVars
  keyBlockVars := layout.keyBlockVars

section Production

variable (key : ℕ → ℕ → ℕ → F) (rel : ℕ → Factored F) (which : Fin 3 → ℕ) (depths : Fin 2 → ℕ)
  (layout : Layout) (zeros : ℕ)

private theorem production_widths :
    productionProfile.commitmentWidth = 18 ∧ productionProfile.ccsMatrices = 4 := ⟨rfl, rfl⟩

private theorem fieldOrder_cast : ((fieldOrder : ℕ) : ℝ) = ((fieldModulus : ℕ) : ℝ) := rfl

private theorem maxPublic_params :
    maxPublic (params key (fun c => (rel (which c)).matrices) depths layout) = 320 :=
  Finset.sup_const ⟨⟨0, Nat.succ_pos 2⟩, Finset.mem_univ _⟩ 320

private theorem tauError_params :
    tauError (params key (fun c => (rel (which c)).matrices) depths layout) =
      ((((2 ^ (layout.witness 0).vars + (2 * bound depths 0 + 1) : ℕ) : ℝ) / fieldOrder) ^ 4 +
        (((2 ^ (layout.witness 1).vars + (2 * bound depths 1 + 1) : ℕ) : ℝ) / fieldOrder) ^ 4) := by
  unfold tauError
  rw [card_F]
  change (∑ c : Fin 3, if decide ((c : ℕ) < 2) = true then
    (((2 ^ (layout.witness c).vars + (2 * bound depths c + 1) : ℕ) : ℝ) / fieldOrder) ^ 4
    else 0) = _
  simp [Fin.sum_univ_three]

private theorem runError_production :
    ((runError (productionCounts layout depths) : ℚ) : ℝ) =
      ((((2 ^ (layout.witness 0).vars + (2 * bound depths 0 + 1) : ℕ) : ℝ) / fieldOrder) ^ 4 +
        (((2 ^ (layout.witness 1).vars + (2 * bound depths 1 + 1) : ℕ) : ℝ) / fieldOrder) ^ 4) +
      ((360 + 4 * layout.witnessVars + 2 * layout.keyBlockVars : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ) := by
  have terms : (productionCounts layout depths).extensionTerms =
      360 + 4 * layout.witnessVars + 2 * layout.keyBlockVars := by
    simp only [RunCounts.extensionTerms, productionCounts]
    omega
  simp only [runError, terms]
  simp only [productionCounts, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  push_cast
  ring

/-- At the production parameters, `runError` bounds the bad-set densities of one path. -/
theorem runCost_le_runError
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout)) :
    runCost (verifierData key rel which depths layout zeros) ≤
      (runError (productionCounts layout depths) : ℝ) := by
  have span : (layout.spanVars : ℝ) ≤ layout.witnessVars := by exact_mod_cast wf.span_le
  rw [runError_production]
  unfold runCost challengeError
  rw [tauError_params, card_K, maxPublic_params]
  simp only [params, Params.equations, Params.lanes, verifierData, production_widths.1, production_widths.2]
  push_cast
  rw [fieldOrder_cast, show max (63 : ℝ) 319 = 319 by norm_num]
  have inv : 0 ≤ ((fieldModulus : ℝ) ^ 2)⁻¹ := by positivity
  have := mul_le_mul_of_nonneg_right span inv
  simp only [div_eq_mul_inv]
  nlinarith

/-- At the production parameters, `runError` bounds the largest density of one draw. -/
theorem error_le_runError
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout)) :
    error (verifierData key rel which depths layout zeros) ≤
      (runError (productionCounts layout depths) : ℝ) := by
  have span : (layout.spanVars : ℝ) ≤ layout.witnessVars := by exact_mod_cast wf.span_le
  rw [runError_production]
  unfold error challengeError
  rw [tauError_params, card_K, maxPublic_params]
  simp only [params, Params.equations, Params.lanes, verifierData, production_widths.1, production_widths.2]
  push_cast
  rw [fieldOrder_cast, show max (63 : ℝ) 319 = 319 by norm_num]
  have inv : 0 ≤ ((fieldModulus : ℝ) ^ 2)⁻¹ := by positivity
  have := mul_le_mul_of_nonneg_right span inv
  have witness : 0 ≤ (layout.witnessVars : ℝ) * ((fieldModulus : ℝ) ^ 2)⁻¹ := by positivity
  have blocks : 0 ≤ (layout.keyBlockVars : ℝ) * ((fieldModulus : ℝ) ^ 2)⁻¹ := by positivity
  simp only [div_eq_mul_inv]
  nlinarith

/-- One interactive run at the production parameters: the bound that the Rust gate compares with
the security minimum. -/
theorem params_run_soundness
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout))
    (slots_le : ∀ c : Fin 3, (c : ℕ) < 2 → (layout.witness c).vars ≤ 25)
    (key_fits : ∀ c : Fin 3, (layout.witness c).vars - 6 ≤ layout.keyBlockVars)
    (s : Strategy (Option (Statement (params key (fun c => (rel (which c)).matrices) depths
      layout) × (ℕ → F)))) :
    prob s (game (verifierData key rel which depths layout zeros))
        (fun r => ¬ ∀ o, r = some o → Relation _ o.1 o.2) ≤
      (runError (productionCounts layout depths) : ℝ) :=
  (run_soundness _ wf (params_countsNonzero key _ depths layout slots_le)
    (verifierData_matrixSpec key rel which depths layout zeros) (le_of_eq rfl)
    (Nat.le_of_ble_eq_true rfl) key_fits s).trans
    (runCost_le_runError key rel which depths layout zeros wf)

/-- One interactive run at the production parameters is at most `2^−runBits`: the gate of
`compression_layout` passes a minimum `m` only when `m ≤ runBits`. -/
theorem params_run_soundness_bits
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout))
    (slots_le : ∀ c : Fin 3, (c : ℕ) < 2 → (layout.witness c).vars ≤ 25)
    (key_fits : ∀ c : Fin 3, (layout.witness c).vars - 6 ≤ layout.keyBlockVars)
    (s : Strategy (Option (Statement (params key (fun c => (rel (which c)).matrices) depths
      layout) × (ℕ → F)))) :
    prob s (game (verifierData key rel which depths layout zeros))
        (fun r => ¬ ∀ o, r = some o → Relation _ o.1 o.2) ≤
      1 / 2 ^ runBits (productionCounts layout depths) := by
  rcases Nat.eq_zero_or_pos (runBits (productionCounts layout depths)) with zero | pos
  · rw [zero, pow_zero, div_one]
    exact prob_le_one _ _ _
  · exact (params_run_soundness key rel which depths layout zeros wf slots_le key_fits s).trans
      (runError_le_runBits _ pos)

/-- When `runBits` is positive, the largest density of one draw is at most `2^−runBits`. -/
theorem error_le_runBits
    (wf : WellFormed (params key (fun c => (rel (which c)).matrices) depths layout))
    (pos : 0 < runBits (productionCounts layout depths)) :
    error (verifierData key rel which depths layout zeros) ≤
      1 / 2 ^ runBits (productionCounts layout depths) :=
  (error_le_runError key rel which depths layout zeros wf).trans (runError_le_runBits _ pos)

end Production

end NightstreamFPrime.Spec.Compression.Rust
