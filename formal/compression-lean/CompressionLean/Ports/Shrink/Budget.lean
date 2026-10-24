import Mathlib.Data.Nat.Notation

/-!
Owns the port of the shrink layer's charged degrees (`rustDegrees`,
`terms`, `mod.rs::terms`): the zero-check point `m`, the outer sum-check
`DEGREE·m = 8·m`, the matrix batching `MATRICES - 1 = 7` and the inner
sum-check `2·(n + 1)`. `Shrink/Budget.lean` bounds the shrink error with them.
-/

namespace CompressionLean.Shrink

/-- The degrees of `terms` (`mod.rs::terms`). -/
def rustDegrees (m n : ℕ) : ℕ := m + 8 * m + (8 - 1) + 2 * (n + 1)

end CompressionLean.Shrink
