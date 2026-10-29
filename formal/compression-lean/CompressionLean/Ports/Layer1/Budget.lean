import CompressionLean.Layer1.Cost

/-!
Owns the port of `outer_terms` (`crates/neo-spartan/src/relation.rs`): the
degrees of the layer-1 draws that Rust charges P0, each
`degree · L1 / 2^191`, written term by term (`rustDegrees`).
`Final/Budget.lean` bounds the layer-1 error with them.
-/

namespace CompressionLean.Layer1

/-- `outer_terms`' degree of a GKR: per layer `k`, four per round and one
per tree that reaches `k`. -/
def rustGkr (s : GkrShape) : ℕ :=
  ∑ k ∈ Finset.Ico 1 (s.layers + 1), (4 * k + (s.active k).length)

/-- The early terms of `outer_terms`: norm logUp, row lookup, slot scatter,
early GKR and the slot-block test. -/
def rustEarly (s : Shape) : ℕ :=
  (2 ^ s.cubeVars + (2 * s.normBound + 1)) + 3 * (2 ^ s.runVars + 2 ^ s.rowVars) +
    (2 ^ s.runVars + 2 ^ (s.slotVars + s.matrixVars) + 1) + rustGkr (earlyShape s) + s.runVars

/-- The linear terms: ring-row batching, the quotient at `ζ` and the linear
sum-check. `s.rows` is `ring::row_count` (`ring.rs`). -/
def rustLinear (s : Shape) : ℕ := (s.rows - 1) + (2 * s.D - 2) + 2 * s.cubeVars

/-- The late terms: block scatter and late GKR. -/
def rustLate (s : Shape) : ℕ := (2 ^ (s.slotVars + 1) + 2 ^ s.blockVars) + rustGkr (lateShape s)

/-- Every degree of `outer_terms`; the run group (3) and the honest-fold
partials (6) are the `9`. -/
def rustDegrees (s : Shape) : ℕ := rustEarly s + rustLinear s + 9 + rustLate s

end CompressionLean.Layer1
