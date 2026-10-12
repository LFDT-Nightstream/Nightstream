import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Spec.Nebula.Game

/-! Owns the per-slot bound of the game that motivates security note A6, for
the concrete verifier context: over `K` with the Poseidon2 chains. Each retry
disagreement is a Poseidon2 transcript collision. A6 itself is a premise of
the end-to-end bound (`NebulaMemorySecurity.MemoryRound`); this bound only
motivates the value of its error. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Plan} {σ Coins : Type} [Fintype Coins] (app : Application p σ)
  (g : Game K Digest σ Coins p.sMax)

theorem context_hash (p : Plan) : (context p).hash = hash := rfl

/-- Security note Lemma 3 for segment slot `k` over `K`: the slot is bad at
most `2·m_mem/q²` plus its retry term of the time. -/
theorem badAt_bound (valid : p.Valid) (k : Fin p.sMax) :
    freq (g.BadAt (context p) app k) ≤
      2 * (p.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ) + g.retryTerm (context p) app k := by
  have bound := g.badAt_frequency (context p) app valid k
  rwa [GoldilocksExtensionRing.card_K] at bound

/-- A chain collision between two closing segment views is a Poseidon2
transcript collision. With `Game.disagreement_collision`, every retry
disagreement is one. -/
theorem closing_collision_transcript (valid : p.Valid) {η η' : K × K}
    {v w : SegmentView Digest} (closesV : v.ClosesAt (context p) η)
    (closesW : w.ClosesAt (context p) η')
    (collision : CollisionIn (context p).hash (v.chainInputs (context p))
      (w.chainInputs (context p))) :
    ∃ a ∈ v.chainInputs (context p), ∃ b ∈ w.chainInputs (context p),
      TranscriptCollision (blocks a) (blocks b) :=
  collision_transcript (context_hash p) valid closesV.chainInputs_canonical
    closesW.chainInputs_canonical collision

end NightstreamFPrime.Lifecycle.Nebula
