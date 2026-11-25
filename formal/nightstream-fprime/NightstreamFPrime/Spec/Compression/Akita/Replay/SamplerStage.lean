import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerCount
import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript

/-! The stage `fold_challenges` of the replay: the
draw of a group gives no challenge (`VerifierFoldDraw`: the operator-norm pool runs out, and the
verifier rejects with `InvalidProof`), and the outcome is `noWork` at the squeeze of the group.

- `step_squeeze`: the step of a squeeze squeezes the message of the group from the sponge and
  draws (`Replayer.draw`). `draw_run`: the draw records the message and `foldChallenges` of it,
  or it stops with `noWork` at the site of the group when `foldChallenges` gives none. So each
  recorded challenge of a group is a challenge of the sampler of the code on the recorded
  message (`drawn_sample`), a member of the family of the group in the ring of `A`
  (`drawn_mem`).
- `foldChallenges_isSome`: an unfiltered group gives its challenges for each message of its
  length, so the stage is only at a filtered group (`draw_filtered`).
- `groupAt` is the family of a group of a layout. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

def groupAt (layout : Layout) (level group : ℕ) : Option FoldGroup :=
  if level < layout.folds.length then layout.folds[level]?.bind (·.groups[group]?)
  else if level = layout.folds.length ∧ group = 0 then some layout.terminal.group else none

/-- An unfiltered group gives its challenges for each message of its length. -/
theorem foldChallenges_isSome {G : FoldGroup} {bytes : ByteArray} (unfiltered : G.filtered = false)
    (positive : 0 < G.rankSpace) (size : bytes.size = G.messageBytes) :
    (foldChallenges G bytes).isSome := by
  rw [foldChallenges, Option.isSome_map]
  exact sample_isSome unfiltered positive size

namespace Replayer

/-- The state after the message `bytes` of the group `group` of `level`. -/
def messaged (s : Replayer) (level group : ℕ) (bytes : ByteArray) : Replayer :=
  { s with derived := { s.derived with
      foldMessages := s.derived.foldMessages.push (level, group, bytes) } }

/-- The state after the challenges `cs` of the group `group` of `level`. -/
def withDrawn (s : Replayer) (level group : ℕ) (cs : List Sparse) : Replayer :=
  { s with derived := { s.derived with drawn := s.derived.drawn.push (level, group, cs) } }

theorem step_squeeze (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (level group : ℕ) (G : FoldGroup) :
    step pub proof messages i (.squeeze level group G) =
      (do draw i level group G (← squeeze G.messageBytes)) :=
  rfl

/-- The draw records the message, and then the challenges of the sampler on it, or it stops with
`noWork` at the site of the group when the sampler gives none. -/
theorem draw_run (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) (s : Replayer) :
    ((draw i level group G bytes).run).run s =
      match foldChallenges G bytes with
      | some cs => (.ok (), withDrawn (messaged s level group bytes) level group cs)
      | none => (.error (.noWork i (Site.foldChallengeGroup level group).text),
          messaged s level group bytes) := by
  cases found : foldChallenges G bytes with
  | none => simp only [draw, found]; rfl
  | some cs => simp only [draw, found]; rfl

/-- Each recorded challenge list of a draw is the sampler of the code on the recorded message,
with each position in the ring of `A`. -/
theorem drawn_sample {i level group : ℕ} {G : FoldGroup} {bytes : ByteArray} {s s' : Replayer}
    (ran : ((draw i level group G bytes).run).run s = (.ok (), s')) :
    ∃ cs, sample G bytes = some cs ∧ s' = withDrawn (messaged s level group bytes) level group
      (cs.map (ambient G)) := by
  rw [draw_run] at ran
  cases found : foldChallenges G bytes with
  | none => rw [found] at ran; cases ran
  | some cs =>
      rw [found] at ran
      obtain ⟨cs₀, sampled, rfl⟩ := Option.map_eq_some_iff.mp found
      exact ⟨cs₀, sampled, (Prod.mk.inj ran).2.symm⟩

/-- Each recorded challenge of a draw is a member of the family of the group in the ring of `A`
(`sample_mem`, `ring_ambient`), and at a filtered group it passes the filter of the code; there
are `G.coordinates` of them. -/
theorem drawn_mem {i level group : ℕ} {G : FoldGroup} [NeZero G.degree] [NeZero G.ring]
    (tower : G.ring / G.degree * G.degree = G.ring) {bytes : ByteArray} {s s' : Replayer}
    (ran : ((draw i level group G bytes).run).run s = (.ok (), s')) :
    ∃ cs, s' = withDrawn (messaged s level group bytes) level group cs ∧
      cs.length = G.coordinates ∧ ∀ c ∈ cs, ∃ c₀, Sparse.ring G.ring c =
        embed (G.ring / G.degree) tower (Sparse.ring G.degree c₀) ∧
        Sparse.ring G.degree c₀ ∈ SparseFamily G.degree G.pm1 G.pm2 ∧
        (G.filtered → TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree)
          (Sparse.ring G.degree c₀)) := by
  obtain ⟨cs, sampled, rfl⟩ := drawn_sample ran
  obtain ⟨length, each⟩ := sample_mem sampled
  refine ⟨_, rfl, by rw [List.length_map, length], fun c member => ?_⟩
  obtain ⟨c₀, inside, rfl⟩ := List.mem_map.mp member
  exact ⟨c₀, ring_ambient tower c₀, each c₀ inside⟩

/-- A draw that stops is at a group whose message has the wrong length, whose rank space is
empty, or which is filtered. -/
theorem draw_filtered {i level group : ℕ} {G : FoldGroup} {bytes : ByteArray} {s : Replayer}
    {stop : Stop} {s' : Replayer} (ran : ((draw i level group G bytes).run).run s = (.error stop, s')) :
    foldChallenges G bytes = none ∧
      stop = .noWork i (Site.foldChallengeGroup level group).text ∧
      (bytes.size = G.messageBytes → 0 < G.rankSpace → G.filtered = true) := by
  rw [draw_run] at ran
  cases found : foldChallenges G bytes with
  | some cs => rw [found] at ran; cases ran
  | none =>
      rw [found] at ran
      refine ⟨rfl, (Except.error.inj (Prod.mk.inj ran).1).symm, fun size positive => ?_⟩
      cases filtered : G.filtered with
      | true => rfl
      | false =>
          have some := foldChallenges_isSome filtered positive size
          rw [found] at some
          cases some

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
