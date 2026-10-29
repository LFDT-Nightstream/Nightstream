import CompressionLean.Layer1.Cost

namespace CompressionLean.Layer1

def rustGkr (s : GkrShape) : ℕ :=
  ∑ k ∈ Finset.Ico 1 (s.layers + 1), (4 * k + (s.active k).length)

def rustEarly (s : Shape) : ℕ :=
  (2 ^ s.cubeVars + (2 * s.normBound + 1)) + 3 * (2 ^ s.runVars + 2 ^ s.rowVars) +
    (2 ^ s.runVars + 2 ^ (s.slotVars + s.matrixVars) + 1) + rustGkr (earlyShape s) + s.runVars

def rustLinear (s : Shape) : ℕ := (s.rows - 1) + (2 * s.D - 2) + 2 * s.cubeVars

def rustLate (s : Shape) : ℕ := (2 ^ (s.slotVars + 1) + 2 ^ s.blockVars) + rustGkr (lateShape s)

def rustDegrees (s : Shape) : ℕ := rustEarly s + rustLinear s + 9 + rustLate s

end CompressionLean.Layer1
