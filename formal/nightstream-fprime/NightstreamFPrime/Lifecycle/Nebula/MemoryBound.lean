import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Spec.Nebula.Game

/-! Owns the memory terms of security note §5 for the concrete verifier
context: the interactive game of A6 over `K` with the Poseidon2 chains. Both
collision terms are Poseidon2 transcript collisions. The A6 transfer from the
real protocol to this game is not owned here. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint

variable {p : Plan} {σ Coins : Type} [Fintype Coins] (app : Application p σ)
  (g : Game K Digest σ Coins p.sMax)

theorem context_hash (p : Plan) : (context p).hash = hash := rfl

/-- The memory terms of §5 over `K`: an accepted, consistent run that is not an
execution is at most as frequent as a run collision, plus
`S_max · 2·m_mem/q²`, plus the retry terms. -/
theorem memory_bound (valid : p.Valid) :
    freq (g.Fails (context p) app) ≤ freq (g.Collides (context p) app) +
      p.sMax * (2 * (p.maxTuples : ℚ≥0) / (goldilocksModulus ^ 2 : ℕ)) +
      ∑ k, g.retryTerm (context p) app k :=
  calc freq (g.Fails (context p) app)
      ≤ freq (g.Collides (context p) app) +
          p.sMax * (2 * (p.maxTuples : ℚ≥0) / Fintype.card K) +
          ∑ k, g.retryTerm (context p) app k := g.fails_frequency (context p) app valid
    _ = _ := by rw [GoldilocksFingerprint.card_K]

/-- A run collision of an accepted outcome is a Poseidon2 transcript
collision. -/
theorem collides_transcript (valid : p.Valid) {q : Coins × (Fin p.sMax → K × K)}
    (collides : g.Collides (context p) app q) :
    ∃ a ∈ runInputs (context p) (g.run q.1 q.2) (g.statement q.1 q.2).segments,
      ∃ b ∈ runInputs (context p) (g.run q.1 q.2) (g.statement q.1 q.2).segments,
        TranscriptCollision (blocks a) (blocks b) := by
  obtain ⟨accepted, collision⟩ := collides
  have canonical := runInputs_canonical (ctx := (context p).withChallenges q.2) valid accepted
  rw [runInputs_withChallenges, withChallenges_plan] at canonical
  exact collision_transcript (context_hash p) valid canonical canonical collision

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
