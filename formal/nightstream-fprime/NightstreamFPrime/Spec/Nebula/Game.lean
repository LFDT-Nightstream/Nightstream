import NightstreamFPrime.Spec.Nebula.Soundness
import NightstreamFPrime.Spec.Nebula.RetryAveraging

/-! Owns the translated interactive memory game of security note A6 and the
memory terms of the §5 bound in that game. An outcome is the prover's coins
and one fresh uniform challenge pair per segment slot. Before the challenge of
segment `k`, the prover commits the transcript input of segment `k`; the
commitment depends only on the coins and the earlier challenges. After the
last challenge, the extractor of A1 and A4 returns a statement and a run.

The bound: an accepted, consistent run that is not an execution is at most as
frequent as a collision among the run's own chain inputs, plus
`S_max · 2·m_mem/|E|`, plus one retry-disagreement term per segment slot, and
each disagreement exhibits a collision. A run whose segment transcript inputs
differ from the committed ones is an extraction failure (A1, A4); it is a
separate event. The transfer from the real protocol (A6) is not owned here. -/

namespace NightstreamFPrime.Spec.Nebula

open Classical

/-- The frequency of an event over a finite outcome space. -/
noncomputable def freq {Ω : Type} [Fintype Ω] (event : Ω → Prop) : ℚ≥0 :=
  ((Finset.univ.filter event).card : ℚ≥0) / Fintype.card Ω

/-- The challenges `e` below slot `k`, `x` at slot `k`, and `e'` above it. -/
def splice {A : Type} {S : ℕ} (k : Fin S) (e : Fin S → A) (x : A) (e' : Fin S → A) :
    Fin S → A :=
  fun j => if j.val < k.val then e j else if j = k then x else e' j

/-- Splicing is a bijection up to the unused coordinates. -/
def spliceEquiv {A : Type} {S : ℕ} (k : Fin S) :
    (Fin S → A) × A × (Fin S → A) ≃ (Fin S → A) × (Fin S → A) × A where
  toFun q := (splice k q.1 q.2.1 q.2.2,
    fun j => if j.val < k.val then q.2.2 j else q.1 j, q.2.2 k)
  invFun q := (fun j => if j.val < k.val then q.1 j else q.2.1 j, q.1 k,
    fun j => if j.val < k.val then q.2.1 j else if j = k then q.2.2 else q.1 j)
  left_inv q := by
    obtain ⟨e, x, e'⟩ := q
    refine Prod.ext (funext fun j => ?_) (Prod.ext ?_ (funext fun j => ?_))
    · simp only [splice]; split_ifs <;> rfl
    · simp [splice]
    · simp only [splice]
      split_ifs with h₁ h₂ <;> first | rfl | (subst h₂; rfl)
  right_inv q := by
    obtain ⟨e, r, y⟩ := q
    refine Prod.ext (funext fun j => ?_) (Prod.ext (funext fun j => ?_) ?_)
    · simp only [splice]
      split_ifs with h₁ h₂ <;> first | rfl | (subst h₂; rfl)
    · simp only; split_ifs <;> rfl
    · simp

/-- A spliced outcome is uniform: each outcome has the same number of
preimages. -/
theorem splice_card {A : Type} [Fintype A] [DecidableEq A] {S : ℕ} (k : Fin S)
    (event : (Fin S → A) → Prop) :
    (Finset.univ.filter fun q : (Fin S → A) × A × (Fin S → A) =>
        event (splice k q.1 q.2.1 q.2.2)).card =
      (Finset.univ.filter event).card * (Fintype.card A ^ S * Fintype.card A) := by
  rw [← Finset.card_map (spliceEquiv k).toEmbedding]
  have image : (Finset.univ.filter fun q : (Fin S → A) × A × (Fin S → A) =>
      event (splice k q.1 q.2.1 q.2.2)).map (spliceEquiv k).toEmbedding =
        Finset.univ.filter fun q : (Fin S → A) × (Fin S → A) × A => event q.1 := by
    ext q
    have first := congrArg Prod.fst ((spliceEquiv k).apply_symm_apply q)
    simp only [Finset.mem_map_equiv, Finset.mem_filter, Finset.mem_univ, true_and]
    exact Iff.of_eq (congrArg event first)
  rw [image, show (Finset.univ.filter fun q : (Fin S → A) × (Fin S → A) × A => event q.1) =
      (Finset.univ.filter event) ×ˢ Finset.univ by ext; simp, Finset.card_product,
    Finset.card_univ, Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]

variable {E Digest σ Coins : Type} [CommRing E]

/-- The prover's commitments and the extracted outcome of the interactive
memory game, as functions of the coins and the challenges. -/
structure Game (E Digest σ Coins : Type) (S : ℕ) where
  commit : Coins → (Fin S → E × E) → ℕ → EtaInput Digest
  causal : ∀ c e e' k, (∀ j : Fin S, j.val < k → e j = e' j) → commit c e k = commit c e' k
  statement : Coins → (Fin S → E × E) → Statement σ Digest
  run : Coins → (Fin S → E × E) → List (Invocation σ Digest)

/-- The challenge pair of segment `k`. -/
def challengeAt {S : ℕ} (e : Fin S → E × E) (k : ℕ) : E × E :=
  if h : k < S then e ⟨k, h⟩ else 0

/-- The context of one outcome: segment `k` uses the `k`-th sampled pair. -/
def Context.withChallenges (ctx : Context E Digest) {S : ℕ} (e : Fin S → E × E) :
    Context E Digest :=
  { ctx with eta := fun k _ => challengeAt e k }

theorem withChallenges_plan (ctx : Context E Digest) {S : ℕ} (e : Fin S → E × E) :
    (ctx.withChallenges e).plan = ctx.plan := rfl

/-- The hash inputs of a run do not depend on the challenge function. -/
theorem runInputs_withChallenges (ctx : Context E Digest) {S : ℕ} (e : Fin S → E × E)
    {σ : Type} (run : List (Invocation σ Digest)) (T : ℕ) :
    runInputs (ctx.withChallenges e) run T = runInputs ctx run T := rfl

@[simp] theorem segmentView_withChallenges (ctx : Context E Digest) {S : ℕ}
    (e : Fin S → E × E) {σ : Type} (run : List (Invocation σ Digest)) (k : ℕ) :
    segmentView (ctx.withChallenges e) run k = segmentView ctx run k := rfl

/-- The close checks do not read the challenge function. -/
theorem SegmentView.ClosesAt.of_withChallenges {ctx : Context E Digest} {S : ℕ}
    {e : Fin S → E × E} {η : E × E} {v : SegmentView Digest}
    (closes : v.ClosesAt (ctx.withChallenges e) η) : v.ClosesAt ctx η :=
  ⟨closes.length, closes.rows, closes.opsRoot, closes.initialRoot, closes.finalRoot,
    closes.products⟩

namespace Game

variable (ctx : Context E Digest) (app : Application ctx.plan σ)
  (g : Game E Digest σ Coins ctx.plan.sMax)

/-- The extracted run is accepted with the sampled challenges. -/
def Accepted (q : Coins × (Fin ctx.plan.sMax → E × E)) : Prop :=
  Accepts (ctx.withChallenges q.2) app (g.statement q.1 q.2) (g.run q.1 q.2)

/-- Each segment's transcript input is the one committed before its
challenge. -/
def Consistent (q : Coins × (Fin ctx.plan.sMax → E × E)) : Prop :=
  ∀ k < (g.statement q.1 q.2).segments,
    (segmentView ctx (g.run q.1 q.2) k).etaInput ctx = g.commit q.1 q.2 k

/-- An accepted, consistent run that is not an execution. -/
def Fails (q : Coins × (Fin ctx.plan.sMax → E × E)) : Prop :=
  g.Accepted ctx app q ∧ g.Consistent ctx q ∧
    ¬ Attests (ctx.withChallenges q.2) app (g.statement q.1 q.2) (g.run q.1 q.2)

/-- A collision among the hash inputs of an accepted run. -/
def Collides (q : Coins × (Fin ctx.plan.sMax → E × E)) : Prop :=
  g.Accepted ctx app q ∧ RunCollision ctx (g.run q.1 q.2) (g.statement q.1 q.2).segments

/-- Segment `k` of an accepted, consistent run passes the product test with
unbalanced multisets. -/
def BadAt (k : Fin ctx.plan.sMax) (q : Coins × (Fin ctx.plan.sMax → E × E)) : Prop :=
  g.Accepted ctx app q ∧ g.Consistent ctx q ∧ k.val < (g.statement q.1 q.2).segments ∧
    BadChallenge (ctx.withChallenges q.2) (segmentView ctx (g.run q.1 q.2) k)

/-- One call of the Lemma 3 game for segment `k` from the prefix `(c, e)`:
the challenge of slot `k` and fresh challenges for the later slots. It
returns the segment's records when the run is accepted and consistent. -/
noncomputable def play (k : Fin ctx.plan.sMax) (c : Coins) (e : Fin ctx.plan.sMax → E × E) :
    (E × E) × (Fin ctx.plan.sMax → E × E) → Option (List StepRecords) := fun x =>
  if g.Accepted ctx app (c, splice k e x.1 x.2) ∧ g.Consistent ctx (c, splice k e x.1 x.2) ∧
      k.val < (g.statement c (splice k e x.1 x.2)).segments then
    some (segmentView ctx (g.run c (splice k e x.1 x.2)) k).records
  else none

/-- The Lemma 3 error predicate of a call: the multisets of the records,
read against the committed transcript input, are not balanced. -/
def Unbalanced (k : Fin ctx.plan.sMax) (c : Coins) (e : Fin ctx.plan.sMax → E × E)
    (z : List StepRecords) : Prop :=
  ¬ (((g.commit c e k).view k z).multisets ctx.plan).Balanced

/-- The retry-disagreement term of Lemma 3 part 2 for segment `k` from the
prefix `(c, e)`. -/
noncomputable def disagreement [Fintype E] (k : Fin ctx.plan.sMax) (c : Coins)
    (e : Fin ctx.plan.sMax → E × E) : ℚ≥0 :=
  ((disagreements (g.play ctx app k c e) (g.Unbalanced ctx k c e)).card : ℚ≥0) /
    ((Fintype.card ((E × E) × (Fin ctx.plan.sMax → E × E)) : ℚ≥0) *
      (successes (g.play ctx app k c e)).card)

/-- The average retry-disagreement term of segment `k`. -/
noncomputable def retryTerm [Fintype E] [Fintype Coins] (k : Fin ctx.plan.sMax) : ℚ≥0 :=
  (∑ q : Coins × (Fin ctx.plan.sMax → E × E), g.disagreement ctx app k q.1 q.2) /
    Fintype.card (Coins × (Fin ctx.plan.sMax → E × E))

omit [CommRing E] in
private theorem view_etaInput (v : SegmentView Digest) :
    (v.etaInput ctx).view v.index v.records = v := by
  cases v
  rfl

omit [CommRing E] in
/-- In an accepted, consistent outcome, segment `k` is the view of the
committed transcript input. -/
theorem view_eq {k : Fin ctx.plan.sMax} {c : Coins} {e : Fin ctx.plan.sMax → E × E}
    {x : (E × E) × (Fin ctx.plan.sMax → E × E)}
    (consistent : g.Consistent ctx (c, splice k e x.1 x.2))
    (hk : k.val < (g.statement c (splice k e x.1 x.2)).segments) :
    (g.commit c e k).view k (segmentView ctx (g.run c (splice k e x.1 x.2)) k).records =
      segmentView ctx (g.run c (splice k e x.1 x.2)) k := by
  have before : g.commit c (splice k e x.1 x.2) k = g.commit c e k :=
    g.causal c _ _ k fun j hj => by simp [splice, hj]
  rw [← before, ← consistent k hk]
  exact view_etaInput ctx (segmentView ctx (g.run c (splice k e x.1 x.2)) k)

/-- Every call that returns records closes against the committed transcript
input with the sampled challenge of slot `k`. -/
theorem play_closes (valid : ctx.plan.Valid) (k : Fin ctx.plan.sMax) (c : Coins)
    (e : Fin ctx.plan.sMax → E × E) :
    ∀ x z, g.play ctx app k c e x = some z → ((g.commit c e k).view k z).ClosesAt ctx x.1 := by
  intro x z returns
  unfold play at returns
  split_ifs at returns with checks
  obtain ⟨accepted, consistent, hk⟩ := checks
  cases returns
  have count := accepted.steps.trans accepted.stepCount
  obtain ⟨final, finalEq, -⟩ := accepted.terminal
  have closes := ((finalCarry_isSome_iff (ctx := ctx.withChallenges (splice k e x.1 x.2))
    valid count).1 ⟨final, finalEq⟩).2 k hk
  have challenge : (segmentView (ctx.withChallenges (splice k e x.1 x.2))
      (g.run c (splice k e x.1 x.2)) k).eta (ctx.withChallenges (splice k e x.1 x.2)) = x.1 := by
    simp [SegmentView.eta, Context.withChallenges, challengeAt, segmentView, splice]
  rw [challenge, segmentView_withChallenges] at closes
  rw [view_eq ctx g consistent hk]
  exact closes.of_withChallenges

/-- A bad segment `k` in a spliced outcome is an erroneous call. -/
theorem badAt_error [Fintype E] {k : Fin ctx.plan.sMax} {c : Coins} {e : Fin ctx.plan.sMax → E × E}
    {x : (E × E) × (Fin ctx.plan.sMax → E × E)}
    (bad : g.BadAt ctx app k (c, splice k e x.1 x.2)) :
    x ∈ errors (g.play ctx app k c e) (g.Unbalanced ctx k c e) := by
  obtain ⟨accepted, consistent, hk, unbalanced, -⟩ := bad
  simp only [errors, Finset.mem_filter, Finset.mem_univ, true_and]
  refine ⟨_, if_pos ⟨accepted, consistent, hk⟩, ?_⟩
  rw [Unbalanced, view_eq ctx g consistent hk]
  exact unbalanced

/-- Every retry disagreement of segment `k` exhibits a collision among the
chain inputs of two closing record sets for the same committed input. -/
theorem disagreement_collision [IsDomain E] [Fintype E] [DecidableEq E]
    [CharP E goldilocksModulus] (valid : ctx.plan.Valid) (k : Fin ctx.plan.sMax) (c : Coins)
    (e : Fin ctx.plan.sMax → E × E) :
    ∀ q ∈ disagreements (g.play ctx app k c e) (g.Unbalanced ctx k c e),
      ∃ z z', g.play ctx app k c e q.1 = some z ∧ g.play ctx app k c e q.2 = some z' ∧
        CollisionIn ctx.hash (((g.commit c e k).view k z).chainInputs ctx)
          (((g.commit c e k).view k z').chainInputs ctx) :=
  (segment_fingerprint_bound valid (g.commit c e k) k (g.play ctx app k c e)
    (g.play_closes ctx app valid k c e)).2

/-- An outcome that fails is a run collision or a bad segment slot. -/
theorem fails_cases (valid : ctx.plan.Valid) {q : Coins × (Fin ctx.plan.sMax → E × E)}
    (fails : g.Fails ctx app q) : g.Collides ctx app q ∨ ∃ k, g.BadAt ctx app k q := by
  obtain ⟨accepted, consistent, notAttests⟩ := fails
  rcases soundness (ctx := ctx.withChallenges q.2) valid accepted with
    attests | collision | ⟨k, hk, bad⟩
  · exact absurd attests notAttests
  · exact Or.inl ⟨accepted, collision⟩
  · exact Or.inr ⟨⟨k, lt_of_lt_of_le hk accepted.segmentsRange.2⟩, accepted, consistent, hk, bad⟩

omit [CommRing E] in
private theorem card_filter_prod {α β : Type} [Fintype α] [Fintype β] (p : α × β → Prop) :
    (Finset.univ.filter p).card = ∑ a, (Finset.univ.filter fun b => p (a, b)).card := by
  simp only [Finset.card_filter, Fintype.sum_prod_type]

/-- Security note Lemma 3 for segment slot `k` of the game, averaged over the
prefix: `Pr[bad slot k] ≤ 2·m_mem/|E| + E[retry disagreement]`. -/
theorem badAt_frequency [IsDomain E] [Fintype E] [DecidableEq E] [CharP E goldilocksModulus]
    [Fintype Coins] (valid : ctx.plan.Valid) (k : Fin ctx.plan.sMax) :
    freq (g.BadAt ctx app k) ≤
      2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E + g.retryTerm ctx app k := by
  set N := Fintype.card ((E × E) × (Fin ctx.plan.sMax → E × E)) with hN
  have spliced : N = Fintype.card (E × E) ^ ctx.plan.sMax * Fintype.card (E × E) := by
    rw [hN, Fintype.card_prod, Fintype.card_fun, Fintype.card_fin, Nat.mul_comm]
  have positive : (0 : ℚ≥0) < N := by exact_mod_cast Fintype.card_pos
  -- Each bad outcome has `N` spliced preimages.
  have total : ((Finset.univ.filter (g.BadAt ctx app k)).card : ℚ≥0) * N =
      ∑ q : Coins × (Fin ctx.plan.sMax → E × E),
        ((Finset.univ.filter fun x : (E × E) × (Fin ctx.plan.sMax → E × E) =>
          g.BadAt ctx app k (q.1, splice k q.2 x.1 x.2)).card : ℚ≥0) := by
    norm_cast
    rw [card_filter_prod, Finset.sum_mul, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [spliced, ← splice_card k (fun e => g.BadAt ctx app k (c, e)), card_filter_prod]
  -- Each spliced bad outcome is an erroneous call of Lemma 3.
  have perPrefix : ∀ q : Coins × (Fin ctx.plan.sMax → E × E),
      ((Finset.univ.filter fun x : (E × E) × (Fin ctx.plan.sMax → E × E) =>
          g.BadAt ctx app k (q.1, splice k q.2 x.1 x.2)).card : ℚ≥0) ≤
        (2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E + g.disagreement ctx app k q.1 q.2) *
          N := by
    intro q
    have errorsBound := (segment_fingerprint_bound valid (g.commit q.1 q.2 k) k
      (g.play ctx app k q.1 q.2) (g.play_closes ctx app valid k q.1 q.2)).1
    rw [div_le_iff₀ positive] at errorsBound
    refine le_trans ?_ errorsBound
    exact_mod_cast Finset.card_le_card fun x hx =>
      g.badAt_error ctx app (Finset.mem_filter.1 hx).2
  have count : ((Finset.univ.filter (g.BadAt ctx app k)).card : ℚ≥0) ≤
      ∑ q : Coins × (Fin ctx.plan.sMax → E × E),
        (2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E + g.disagreement ctx app k q.1 q.2) := by
    refine le_of_mul_le_mul_right ?_ positive
    rw [total, Finset.sum_mul]
    exact Finset.sum_le_sum fun q _ => perPrefix q
  unfold freq retryTerm
  rcases Nat.eq_zero_or_pos (Fintype.card (Coins × (Fin ctx.plan.sMax → E × E))) with empty | full
  · simp [empty]
  have full : (0 : ℚ≥0) < Fintype.card (Coins × (Fin ctx.plan.sMax → E × E)) := by
    exact_mod_cast full
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at count
  calc ((Finset.univ.filter (g.BadAt ctx app k)).card : ℚ≥0) /
        Fintype.card (Coins × (Fin ctx.plan.sMax → E × E))
      ≤ (Fintype.card (Coins × (Fin ctx.plan.sMax → E × E)) *
          (2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E) +
          ∑ q : Coins × (Fin ctx.plan.sMax → E × E), g.disagreement ctx app k q.1 q.2) /
        Fintype.card (Coins × (Fin ctx.plan.sMax → E × E)) :=
      div_le_div_of_nonneg_right count zero_le
    _ = _ := by rw [add_div, mul_div_cancel_left₀ _ full.ne']

/-- The memory terms of security note §5 in the game: an accepted, consistent
run that is not an execution is at most as frequent as a run collision, plus
`S_max · 2·m_mem/|E|`, plus the retry terms. Each retry disagreement exhibits a
collision (`disagreement_collision`). -/
theorem fails_frequency [IsDomain E] [Fintype E] [DecidableEq E] [CharP E goldilocksModulus]
    [Fintype Coins] (valid : ctx.plan.Valid) :
    freq (g.Fails ctx app) ≤ freq (g.Collides ctx app) +
      ctx.plan.sMax * (2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E) +
      ∑ k, g.retryTerm ctx app k := by
  have cover : Finset.univ.filter (g.Fails ctx app) ⊆
      Finset.univ.filter (g.Collides ctx app) ∪
        Finset.univ.biUnion fun k => Finset.univ.filter (g.BadAt ctx app k) := by
    intro q hq
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union,
      Finset.mem_biUnion] at hq ⊢
    exact g.fails_cases ctx app valid hq
  have count : ((Finset.univ.filter (g.Fails ctx app)).card : ℚ≥0) ≤
      (Finset.univ.filter (g.Collides ctx app)).card +
        ∑ k, ((Finset.univ.filter (g.BadAt ctx app k)).card : ℚ≥0) := by
    exact_mod_cast (Finset.card_le_card cover).trans
      ((Finset.card_union_le _ _).trans (Nat.add_le_add_left Finset.card_biUnion_le _))
  have step : freq (g.Fails ctx app) ≤
      freq (g.Collides ctx app) + ∑ k, freq (g.BadAt ctx app k) := by
    unfold freq
    rw [← Finset.sum_div, ← add_div]
    exact div_le_div_of_nonneg_right count zero_le
  calc freq (g.Fails ctx app)
      ≤ freq (g.Collides ctx app) + ∑ k, freq (g.BadAt ctx app k) := step
    _ ≤ freq (g.Collides ctx app) + ∑ k : Fin ctx.plan.sMax,
          (2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E + g.retryTerm ctx app k) :=
      by gcongr with k; exact g.badAt_frequency ctx app valid k
    _ = _ := by
      rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, add_assoc]

end Game

end NightstreamFPrime.Spec.Nebula
