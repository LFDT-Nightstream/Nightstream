import CompressionLean.Ports.Shrink.Budget
import CompressionLean.Shrink.Cost

/-!
Owns the shrink error at Rust's charge (`shrink_budget`).

Rust charges the shrink layer's draws over WHIR's candidates (`terms`,
`mod.rs::terms`): four degrees over `|Ext|`, taken as `2^191`
(`Ext::bits() - 1`). They are the zero-check point `m`, the outer sum-check
`DEGREE·m = 8·m`, the matrix batching `MATRICES - 1 = 7` and the inner
sum-check `2·(n + 1)`. Plonky3 charges each term over the `L` candidates
(`over_candidates`, p3-security `report.rs:100-111`) and adds it to the WHIR
terms (`error`, p3-sumcheck `prescribed_pcs.rs:91-110`). `Pcs::new` accepts
the configuration only when this sum is at most `2^-t` for the layer bits
`t` (`pcs.rs::Pcs::new`; `Shrink::new` with `layer_bits`, `finish.rs::CompressionKey::verify`).
The Lean count is exactly `L` times Rust's degrees (`shrinkCount_eq_rust`),
so the Lean error is at most what Rust charges.
-/

namespace CompressionLean.Shrink

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

end CompressionLean.Shrink
