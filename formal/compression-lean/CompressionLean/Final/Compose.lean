import CompressionLean.Layer1.Restoration

/-!
Owns the composition of two public-coin verifier games into one experiment.

- In sequence (`pair`, `pair_sound`): the first game, then the second. Along
  every path that avoids both games' bad sets, both results hold their
  claims. So a prover breaks either claim with probability at most the sum
  of the two errors.
- Side by side under Fiat–Shamir (`restorePair`, `restorePair_sound`): one
  state-restoration experiment (`Layer1/Restoration.lean`) with one log per
  game. Each query names a node of either game. The prover outputs one path
  of the first game. The result is its walk and the walk of every path of
  the second game under that game's log. Each query costs at most `ε`.

One log per game models two transcripts that share no prefix: an oracle
answer to one never answers the other.

Does not own a protocol or an error term.
-/

namespace CompressionLean.Final

open CompressionLean.Layer1 CompressionLean.Layer1.Game

variable {α β : Type}

/-- A larger event has a larger probability. -/
theorem prob_mono (s : Strategy α) (g : Game α) {E E' : α → Prop} (imp : ∀ a, E a → E' a) :
    prob s g E ≤ prob s g E' := by
  classical
  induction g with
  | pure a =>
      simp only [prob]
      by_cases h : E a
      · rw [if_pos h, if_pos (imp a h)]
      · rw [if_neg h]
        split_ifs <;> norm_num
  | draw C bad next ih => exact Finset.expect_le_expect fun c _ => ih c
  | recv M next ih => exact ih _

/-- Two verifiers in sequence: `g`, then `h`. -/
def pair (g : Game α) (h : Game β) : Game (α × β) :=
  g.bind fun a => h.bind fun b => .pure (a, b)

/-- **Sequential composition.** If `g` and `h` are sound with errors `εg`
and `εh`, a prover breaks the claim of either result with probability at
most `εg + εh`. -/
theorem pair_sound (s : Strategy (α × β)) (g : Game α) (h : Game β) (Q : α → Prop) (R : β → Prop)
    (εg εh : ℝ) (avoidsG : g.Avoids Q) (avoidsH : h.Avoids R) (costG : g.CostLe εg)
    (costH : h.CostLe εh) :
    prob s (pair g h) (fun r => ¬ (Q r.1 ∧ R r.2)) ≤ εg + εh := by
  have avoids : (pair g h).Avoids fun r => Q r.1 ∧ R r.2 :=
    (avoids_bind g _ _).mpr (avoidsG.mono fun a qa =>
      (avoids_bind h _ _).mpr (avoidsH.mono fun b rb => ⟨qa, rb⟩))
  have cost : (pair g h).CostLe (εg + (εh + 0)) :=
    costLe_bind costG fun a => costLe_bind costH fun b => le_refl (0 : ℝ)
  rw [add_zero] at cost
  exact prob_not_le s _ _ _ avoids cost

/-- A walk that accepts ends at the leaf it returns. -/
theorem node_of_walk (log : Log α) (g : Game α) :
    ∀ (p : Pos g) (a : α), walk log g p = some a → Pos.node g p = .pure a := by
  induction g with
  | pure b =>
      intro p a equal
      simp only [walk, Option.some.injEq] at equal
      rw [← equal]
      rfl
  | draw C bad next ih =>
      intro p a equal
      match p with
      | none => simp [walk] at equal
      | some ⟨c, p⟩ =>
          simp only [walk] at equal
          split_ifs at equal
          exact ih c p a equal
  | recv M next ih =>
      intro p a equal
      match p with
      | none => simp [walk] at equal
      | some ⟨m, p⟩ => exact ih m p a equal

instance (g : Game α) (h : Game β) : Nonempty (Pos g ⊕ Pos h) :=
  ⟨.inl (Classical.arbitrary _)⟩

/-- The state-restoration experiment over two games with one log each: `q`
queries, each at a node of `g` or of `h`, then one output path of `g`. The
result is the walk of that path and the walk of every path of `h` under the
log of `h`. -/
noncomputable def restorePair (g : Game α) (h : Game β) :
    ℕ → Log α → Log β → Game (Option α × (Pos h → Option β))
  | 0, lg, lh => .recv (Pos g) fun p => .pure (walk lg g p, walk lh h)
  | q + 1, lg, lh => .recv (Pos g ⊕ Pos h) fun
      | .inl p => answer (fun lg' => restorePair g h q lg' lh) lg (Pos.node g p)
      | .inr p => answer (fun lh' => restorePair g h q lg lh') lh (Pos.node h p)

theorem restorePair_avoids (g : Game α) (h : Game β) (Q : α → Prop) (R : β → Prop)
    (avoidsG : g.Avoids Q) (avoidsH : h.Avoids R) :
    ∀ (q : ℕ) (lg : Log α) (lh : Log β), Keeps Q lg → Keeps R lh →
      (restorePair g h q lg lh).Avoids fun r =>
        (∀ a, r.1 = some a → Q a) ∧ ∀ p b, r.2 p = some b → Pos.node h p = .pure b ∧ R b := by
  intro q
  induction q with
  | zero =>
      intro lg lh keepsG keepsH p
      exact ⟨fun a equal => walk_sound keepsG g p a avoidsG equal,
        fun p b equal => ⟨node_of_walk lh h p b equal, walk_sound keepsH h p b avoidsH equal⟩⟩
  | succ q ih =>
      intro lg lh keepsG keepsH choice
      rcases choice with p | p
      · show (answer (fun lg' => restorePair g h q lg' lh) lg (Pos.node g p)).Avoids _
        generalize Pos.node g p = n
        cases n with
        | draw C bad next =>
            intro c out
            refine ih _ lh (fun e member => ?_) keepsH
            rcases List.mem_cons.mp member with rfl | inside
            · exact fun node => node c out
            · exact keepsG e inside
        | pure b => exact ih lg lh keepsG keepsH
        | recv M next => exact ih lg lh keepsG keepsH
      · show (answer (fun lh' => restorePair g h q lg lh') lh (Pos.node h p)).Avoids _
        generalize Pos.node h p = n
        cases n with
        | draw C bad next =>
            intro c out
            refine ih lg _ keepsG fun e member => ?_
            rcases List.mem_cons.mp member with rfl | inside
            · exact fun node => node c out
            · exact keepsH e inside
        | pure b => exact ih lg lh keepsG keepsH
        | recv M next => exact ih lg lh keepsG keepsH

theorem restorePair_cost (g : Game α) (h : Game β) (ε : ℝ) (densityG : g.DensityLe ε)
    (densityH : h.DensityLe ε) (nonneg : 0 ≤ ε) :
    ∀ (q : ℕ) (lg : Log α) (lh : Log β), (restorePair g h q lg lh).CostLe (q * ε) := by
  intro q
  induction q with
  | zero => intro lg lh p; show (0 : ℝ) ≤ _; simp
  | succ q ih =>
      intro lg lh choice
      rcases choice with p | p
      · show (answer (fun lg' => restorePair g h q lg' lh) lg (Pos.node g p)).CostLe _
        have node := densityG.node p
        generalize Pos.node g p = n at node
        cases n with
        | draw C bad next =>
            refine ⟨q * ε, ?_, fun c => ih _ lh⟩
            push_cast
            linarith [node.1]
        | pure b => exact (ih lg lh).mono (by push_cast; linarith)
        | recv M next => exact (ih lg lh).mono (by push_cast; linarith)
      · show (answer (fun lh' => restorePair g h q lg lh') lh (Pos.node h p)).CostLe _
        have node := densityH.node p
        generalize Pos.node h p = n at node
        cases n with
        | draw C bad next =>
            refine ⟨q * ε, ?_, fun c => ih lg _⟩
            push_cast
            linarith [node.1]
        | pure b => exact (ih lg lh).mono (by push_cast; linarith)
        | recv M next => exact (ih lg lh).mono (by push_cast; linarith)

/-- **State-restoration soundness of two games.** If every avoiding path of
`g` ends in `Q`, every avoiding path of `h` ends in `R`, and every bad set of
both games has density at most `ε`, a prover with `q` queries ends with an
accepted path of either game outside its claim with probability at most
`q·ε`. An accepted path of `h` ends at the leaf it returns. -/
theorem restorePair_sound (g : Game α) (h : Game β) (s : Strategy (Option α × (Pos h → Option β)))
    (Q : α → Prop) (R : β → Prop) (ε : ℝ) (avoidsG : g.Avoids Q) (avoidsH : h.Avoids R)
    (densityG : g.DensityLe ε) (densityH : h.DensityLe ε) (nonneg : 0 ≤ ε) (q : ℕ) :
    prob s (restorePair g h q [] [])
        (fun r => ¬ ((∀ a, r.1 = some a → Q a) ∧
          ∀ p b, r.2 p = some b → Pos.node h p = .pure b ∧ R b)) ≤ q * ε :=
  prob_not_le s _ _ _
    (restorePair_avoids g h Q R avoidsG avoidsH q [] [] (by simp [Keeps]) (by simp [Keeps]))
    (restorePair_cost g h ε densityG densityH nonneg q [] [])

end CompressionLean.Final
