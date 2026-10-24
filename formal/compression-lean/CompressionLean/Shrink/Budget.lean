import CompressionLean.Ports.Shrink.Budget
import CompressionLean.Shrink.Cost
import CompressionLean.Game.Restoration

/-!
Owns the shrink error at Rust's charge (`shrink_budget`) and the shrink
layer against a Fiat–Shamir prover (`shrink_restore_sound`,
`shrink_restore_budget`).

Rust charges the shrink layer's draws over WHIR's candidates (`terms`,
`mod.rs:129-140`): four degrees over `|Ext|`, taken as `2^191`
(`Ext::bits() - 1`). They are the zero-check point `m`, the outer sum-check
`DEGREE·m = 8·m`, the matrix batching `MATRICES - 1 = 7` and the inner
sum-check `2·(n + 1)`. Plonky3 charges each term over the `L` candidates
(`over_candidates`, p3-security `report.rs:100-111`) and adds it to the WHIR
terms (`error`, p3-sumcheck `prescribed_pcs.rs:91-110`). `Pcs::new` accepts
the configuration only when this sum is at most `2^-t` for the layer bits
`t` (`pcs.rs:196-200`; `Shrink::new` with `layer_bits`, `finish.rs:143`).
The Lean count is exactly `L` times Rust's degrees (`shrinkCount_eq_rust`),
so the Lean error is at most what Rust charges.

Each state-restoration query costs at most the whole interactive error
(`Restoration.lean`), so `calls` queries cost at most `calls · shrinkError`.
-/

namespace CompressionLean.Shrink

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

/-- The Lean count is `L` times Rust's degrees. -/
theorem shrinkCount_eq_rust (m n L : ℕ) : shrinkCount m n L = L * rustDegrees m n := by
  unfold shrinkCount rustDegrees
  ring

/-- **The shrink budget at Rust's charge.** If the commitment's composed
error, as Plonky3 charges it, is at most `2^-t`, the WHIR error `whir` of the
idealized opening (`Opens`) plus the shrink error is at most `2^-t`. -/
theorem shrink_budget (m n L size t : ℕ) (big : 2 ^ 191 ≤ size) (whir : ℝ)
    (rust : whir + L * rustDegrees m n / 2 ^ 191 ≤ 1 / 2 ^ t) :
    whir + shrinkError m n L size ≤ 1 / 2 ^ t := by
  unfold shrinkError
  rw [shrinkCount_eq_rust]
  have positive : (0 : ℝ) < 2 ^ 191 := by positivity
  have sizeBig : (2 : ℝ) ^ 191 ≤ size := by exact_mod_cast big
  have field : ((L * rustDegrees m n : ℕ) : ℝ) / size ≤ L * rustDegrees m n / 2 ^ 191 := by
    push_cast
    exact div_le_div₀ (by positivity) le_rfl positive sizeBig
  linarith

/-- An accepted result names a candidate that satisfies its program's CCS. -/
def Extracts {I : Type} (T : Template F) (s : Counts) (prog : I → Run F)
    (r : Option (Transcript I F K)) : Prop :=
  ∀ tr, r = some tr → ∃ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w

/-- Shrink soundness against a state-restoration prover with `calls` queries. -/
theorem shrink_restore_sound {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T)
    (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b) (calls : ℕ)
    (prover : Game.Strategy (Option (Option (Transcript I F K)))) :
    Game.prob prover (Game.restore (shrinkGame T s prog L) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts T s prog a) ≤
      calls * shrinkError (Layout.rowVars T s) (Layout.cellVars T s) L (Fintype.card K) :=
  Game.restore_sound prover _ _ _
    (shrinkGame_avoids T s prog L wf valid (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num)))
    (shrinkGame_cost T s prog L (shrinkNodes cast 7 le_rfl) (shrinkNodes cast 1 (by norm_num))) calls

/-- At Rust's charge, a prover with `calls` queries breaks the shrink layer
with probability at most `calls / 2^t`. -/
theorem shrink_restore_budget {I : Type} [Nonempty I] (T : Template F) (s : Counts) (prog : I → Run F)
    (L : ℕ) (wf : T.WellFormed) (valid : ∀ i, (prog i).Valid T)
    (cast : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b) (big : 2 ^ 191 ≤ Fintype.card K) (t : ℕ)
    (rust : (L : ℝ) * rustDegrees (Layout.rowVars T s) (Layout.cellVars T s) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ) (prover : Game.Strategy (Option (Option (Transcript I F K)))) :
    Game.prob prover (Game.restore (shrinkGame T s prog L) calls [])
        (fun r => ∃ a, r = some a ∧ ¬ Extracts T s prog a) ≤ calls / 2 ^ t := by
  refine (shrink_restore_sound T s prog L wf valid cast calls prover).trans ?_
  have budget := shrink_budget _ _ L (Fintype.card K) t big 0 (by simpa using rust)
  rw [zero_add] at budget
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left budget (Nat.cast_nonneg _)

end CompressionLean.Shrink
