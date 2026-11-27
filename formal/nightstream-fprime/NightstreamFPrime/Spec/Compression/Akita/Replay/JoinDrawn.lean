import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinShape

/-! The drawn fold challenges of a group are the challenges of the sampler of the code on its
message (Akita revision `d50f700`, `VerifierFoldDraw`; `Replay/FinalDrawn.lean`), and the hypothesis
`drawn` of `verifyProof_key20_terminal` as a theorem for `layout20`.

- `bind_draw_sampled`: a draw without a stop after a read of its message that keeps the drawn
  challenges appends `foldChallenges G bytes` of the message `bytes`.
- `drawnIn_sampled`: on the state of a run without a stop from a state with no drawn challenge of
  the group, when the first squeeze of the group has the family `G`, the drawn challenges of the
  group are `foldChallenges G bytes` of a message `bytes`.
- `squeeze20_terminal`: the first squeeze of the terminal group of `layout20` before the terminal
  has the family of the terminal (a closed computation of the kernel on the skeleton).
- `drawn20_before`, `verifyProof_drawn20`: after a run without a stop before the terminal (for an
  accepted proof of `layout20`), the fold challenges of the terminal group on the state before the
  terminal are `foldChallenges` of a message. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A draw without a stop after a read of its message that keeps the drawn challenges appends
the challenges `foldChallenges G bytes` of the message. -/
theorem bind_draw_sampled (i level group : ℕ) (G : FoldGroup) {m : ReplayM ByteArray}
    (keep : KeepsDrawn m) (s : Replayer)
    (ran : (((m >>= fun bytes => draw i level group G bytes).run).run s).1 = .ok ()) :
    ∃ bytes cs, foldChallenges G bytes = some cs ∧
      (((m >>= fun bytes => draw i level group G bytes).run).run s).2.derived.drawn =
        s.derived.drawn.push (level, group, cs) := by
  rw [run_bind] at ran ⊢
  rcases found : (m.run).run s with ⟨_ | bytes, s₁⟩
  · rw [found] at ran
    cases ran
  · have kept : s₁.derived.drawn = s.derived.drawn := by
      have kept := keep s
      rw [found] at kept
      exact kept
    rw [found] at ran
    have drawOk : (((draw i level group G bytes).run).run s₁).1 = .ok () := ran
    rw [draw_run] at drawOk
    rcases challenged : foldChallenges G bytes with _ | cs
    · rw [challenged] at drawOk
      cases drawOk
    · refine ⟨bytes, cs, challenged, ?_⟩
      show (((draw i level group G bytes).run).run s₁).2.derived.drawn = _
      rw [draw_run, challenged, ← kept]
      rfl

/-- On the state of a run without a stop from a state with no drawn challenge of the group `group`
of `level`, when the first squeeze of this group in the operations has the family `G`, the drawn
challenges of the group are the challenges `foldChallenges G bytes` of a message `bytes`. -/
theorem drawnIn_sampled (level group : ℕ) (G : FoldGroup) :
    ∀ (l : List (Op × ℕ)) (s : Replayer),
      s.derived.drawn.find? (fun x => x.1 == level && x.2.1 == group) = none →
      squeezeOf level group l = some G → (runFrom pub proof messages l s).1 = .ok () →
      ∃ bytes, foldChallenges G bytes = some (drawnIn (runFrom pub proof messages l s).2 level group)
  | [], _, _, first, _ => by cases first
  | (op, i) :: l, s, missing, first, ran => by
      obtain ⟨stepped, rest⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped]
      simp only [squeezeOf] at first
      by_cases squeezed : ∃ l' g' G', op = .squeeze l' g' G'
      · obtain ⟨l', g', G', rfl⟩ := squeezed
        have stepped' := stepped
        rw [step_squeeze] at stepped'
        obtain ⟨bytes, cs, sampled, pushed⟩ :=
          bind_draw_sampled i l' g' G' (keepsDrawn_squeeze _) s stepped'
        rw [← step_squeeze] at pushed
        by_cases here : l' = level ∧ g' = group
        · obtain ⟨atLevel, atGroup⟩ := here
          subst l' g'
          simp only [Op.drawsGroup, and_self, ↓reduceIte, Option.some_or,
            Option.some.injEq] at first
          subst G'
          have found : (((step pub proof messages i (.squeeze level group G)).run).run
              s).2.derived.drawn.find? (fun x => x.1 == level && x.2.1 == group) =
                some (level, group, cs) := by
            rw [pushed, Array.find?_push, missing]
            simp
          refine ⟨bytes, ?_⟩
          unfold drawnIn
          rw [find_drawn_grows (runFrom_grows pub proof messages l _).drawn found]
          exact sampled
        · simp only [Op.drawsGroup, here, ↓reduceIte, Option.none_or] at first
          refine drawnIn_sampled level group G l _ ?_ first rest
          rw [pushed, Array.find?_push, missing]
          have differ : (l' == level && g' == group) = false := by
            simpa [Bool.and_eq_true, beq_iff_eq] using here
          simp [differ]
      · have other : ∀ l' g' G', op ≠ .squeeze l' g' G' :=
          fun l' g' G' same => squeezed ⟨l', g', G', same⟩
        have quiet : op.drawsGroup level group = none := by
          cases op <;> first | rfl | exact absurd rfl (other _ _ _)
        rw [quiet, Option.none_or] at first
        refine drawnIn_sampled level group G l _ ?_ first rest
        rw [step_drawn pub proof messages i op other s]
        exact missing

end Replayer

open Replayer

/-- The first squeeze of the terminal group of `layout20` before the terminal has the family of
the terminal. -/
theorem squeeze20_terminal :
    squeezeOf 6 0 (sk20.take 1521) == some (layout20 0 0).terminal.group := by
  decide +kernel

/-- For a script of `layout20`, after a run without a stop before the terminal, the fold
challenges of the terminal group on the state before the terminal are the challenges of the
sampler of the code on a message. -/
theorem drawn20_before {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {messages : Array Message}
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (pub.ops.size - 1))
      (channelStart pub)).1 = .ok ()) :
    ∃ bytes, foldChallenges (layout20 zMax rice).terminal.group bytes =
      some (drawnIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1)) 6 0) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  have size : pub.ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  have first : squeezeOf 6 0 (pub.ops.zipIdx.toList.take (pub.ops.size - 1)) =
      some (layout20 zMax rice).terminal.group := by
    rw [← squeezeOf_erase, take_erase erased, size]
    exact beq_iff_eq.mp squeeze20_terminal
  exact drawnIn_sampled pub proof messages 6 0 _ _ _ (by simp [channelStart]) first before

/-- For an accepted proof of `layout20`, the fold challenges of the terminal group on the state
before the terminal are the challenges of the sampler of the code on a message (the hypothesis
`drawn` of `verifyProof_key20_terminal`). -/
theorem verifyProof_drawn20 {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest)) :
    ∃ bytes, foldChallenges (layout20 zMax rice).terminal.group bytes =
      some (drawnIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1)) 6 0) :=
  let ⟨scripted, ends, _⟩ := verifyProof_script20 layout accepted
  drawn20_before layout scripted (before_of_replay pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 _)

end NightstreamFPrime.Spec.Compression.Akita.Replay
