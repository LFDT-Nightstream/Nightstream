import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Verifier.Run
import Mathlib.Data.Rat.Cast.Lemmas

/-! Rust `error_bits` (`crates/neo-compress/src/lib.rs`) as an exact rational (`runError`), from the
counts of a layout that it reads. Rust computes `−log₂ runError` in `f64`, and the gate of
`compression_layout` compares that number with the caller's security minimum.

At the production parameters, for each layout with `WellFormed`:

- `runCost_le_runError`: `runError` bounds `runCost`, the sum of the bad-set densities of one
  path, so `params_run_soundness` bounds one interactive run by `runError`. This is what the gate
  certifies.
- `error_le_runError`: `runError` also bounds `error`, the largest density of one draw, which
  `soundness` multiplies by the query count `q` of the state-restoration experiment.

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

end Production

end NightstreamFPrime.Spec.Compression.Rust
