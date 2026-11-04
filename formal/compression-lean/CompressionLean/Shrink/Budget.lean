import CompressionLean.Ports.Shrink.Budget
import CompressionLean.Shrink.Cost

/-! The budget hypothesis is Plonky3's `Pcs::new` check (p3-security `report.rs`, p3-sumcheck `prescribed_pcs.rs`): the composed error is at most `2^-t`. -/

namespace CompressionLean.Shrink

theorem shrinkCount_eq_rust (m n L : ℕ) : shrinkCount m n L = L * rustDegrees m n := by
  unfold shrinkCount rustDegrees
  ring

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
