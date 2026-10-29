import Mathlib.Data.Nat.Notation

namespace CompressionLean.Shrink

def rustDegrees (m n : ℕ) : ℕ := m + 8 * m + (8 - 1) + 2 * (n + 1)

end CompressionLean.Shrink
