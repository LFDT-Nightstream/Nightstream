import Mathlib.Data.Fintype.Prod
import Mathlib.Data.NNRat.Lemmas
import NightstreamFPrime.Spec.Nebula.Binding

/-! Owns security note Lemma 3 part 2: the records come after `η`, so they can
depend on it. One call of the translated A6 game runs from a fixed transcript
input with fresh uniform coins `(η, rest)` and returns closing records or
nothing. The uniqueness adversary of SuperNeo v1.2 Appendix B repeats calls
after an erroneous one until one succeeds. The averaging is a count: agreeing
retries lie in the fiber of the erroneous records, which part 1 bounds, and
disagreeing retries give a collision among the two calls' own chain inputs.
The game itself, its uniformity, and the A6 transfer stay premises. -/

namespace NightstreamFPrime.Spec.Nebula

section Count

open Classical

variable {C R : Type} [Fintype C] (play : C → Option R) (bad : R → Prop)

/-- Calls that return a result (`Succ`). -/
noncomputable def successes : Finset C := Finset.univ.filter fun c => (play c).isSome

/-- Calls that return a bad result (`Err`). -/
noncomputable def errors : Finset C :=
  Finset.univ.filter fun c => ∃ r, play c = some r ∧ bad r

/-- An erroneous first call and a successful retry with a different result. -/
noncomputable def disagreements : Finset (C × C) :=
  (errors play bad ×ˢ successes play).filter fun q => play q.1 ≠ play q.2

theorem errors_subset_successes : errors play bad ⊆ successes play := by
  intro c hc
  obtain ⟨r, hr, -⟩ : ∃ r, play c = some r ∧ bad r := by simpa [errors] using hc
  simp [successes, hr]

/-- The averaging step of SuperNeo v1.2 Appendix B, count form. If every bad
result has at most `bound` calls that return it, then
`#Err · #Succ ≤ #Err · bound + #disagree`. -/
theorem retry_count {bound : ℕ}
    (fiber : ∀ r, bad r → (Finset.univ.filter fun c => play c = some r).card ≤ bound) :
    (errors play bad).card * (successes play).card ≤
      (errors play bad).card * bound + (disagreements play bad).card := by
  have agree : ((errors play bad ×ˢ successes play).filter fun q => play q.1 = play q.2).card ≤
      (errors play bad).card * bound := by
    rw [Finset.card_filter, Finset.sum_product]
    calc _ ≤ ∑ _c ∈ errors play bad, bound := Finset.sum_le_sum fun c hc => ?_
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
    obtain ⟨r, hr, hbad⟩ : ∃ r, play c = some r ∧ bad r := by simpa [errors] using hc
    rw [← Finset.card_filter]
    refine le_trans (Finset.card_le_card fun c' hc' => ?_) (fiber r hbad)
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hc' ⊢
    rw [← hc'.2, hr]
  have cover : errors play bad ×ˢ successes play ⊆
      (errors play bad ×ˢ successes play).filter (fun q => play q.1 = play q.2) ∪
        disagreements play bad := by
    intro q hq
    by_cases same : play q.1 = play q.2
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hq, same⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hq, same⟩)
  calc (errors play bad).card * (successes play).card
      = (errors play bad ×ˢ successes play).card := (Finset.card_product _ _).symm
    _ ≤ _ := (Finset.card_le_card cover).trans (Finset.card_union_le _ _)
    _ ≤ _ := Nat.add_le_add_right agree _

/-- The same bound as frequencies:
`Pr[Err] ≤ bound / |C| + Pr[Err and the retry disagrees]`. -/
theorem retry_frequency {bound : ℕ}
    (fiber : ∀ r, bad r → (Finset.univ.filter fun c => play c = some r).card ≤ bound) :
    ((errors play bad).card : ℚ≥0) / Fintype.card C ≤
      (bound : ℚ≥0) / Fintype.card C +
        ((disagreements play bad).card : ℚ≥0) /
          ((Fintype.card C : ℚ≥0) * (successes play).card) := by
  have count := retry_count play bad fiber
  have subset := Finset.card_le_card (errors_subset_successes play bad)
  generalize (errors play bad).card = e at count subset ⊢
  generalize (successes play).card = s at count subset ⊢
  generalize (disagreements play bad).card = d at count ⊢
  rcases Nat.eq_zero_or_pos s with hs | hs
  · obtain rfl : e = 0 := by omega
    simp
  rcases Nat.eq_zero_or_pos (Fintype.card C) with hC | hC
  · simp [hC]
  have key : e * s ≤ bound * s + d :=
    calc e * s ≤ e * bound + d := count
      _ ≤ s * bound + d := by gcongr
      _ = bound * s + d := by rw [Nat.mul_comm]
  have hs' : (s : ℚ≥0) ≠ 0 := by exact_mod_cast hs.ne'
  rw [← mul_div_mul_right (e : ℚ≥0) _ hs', ← mul_div_mul_right (bound : ℚ≥0) _ hs', ← add_div]
  exact div_le_div_of_nonneg_right (by exact_mod_cast key) zero_le

/-- The uniqueness adversary of SuperNeo v1.2 Appendix B makes one call and,
after an erroneous call, repeats calls until one succeeds. Its expected number
of calls is `1 + #Err/#Succ`, which is at most 2: `t_1 ≤ 2·t_E`, counted in
calls. -/
theorem retry_expected_calls :
    (1 : ℚ≥0) + ((errors play bad).card : ℚ≥0) / (successes play).card ≤ 2 := by
  have subset := Finset.card_le_card (errors_subset_successes play bad)
  rcases Nat.eq_zero_or_pos (successes play).card with none | some
  · simp [none]
  have ratio : ((errors play bad).card : ℚ≥0) / (successes play).card ≤ 1 := by
    rw [div_le_one₀ (by exact_mod_cast some)]
    exact_mod_cast subset
  calc (1 : ℚ≥0) + ((errors play bad).card : ℚ≥0) / (successes play).card ≤ 1 + 1 := by
        gcongr
    _ = 2 := by norm_num

end Count

section Segment

variable {E Digest : Type} [CommRing E]

/-- The segment view of records closed against a fixed transcript input. -/
def EtaInput.view (inp : EtaInput Digest) (k : ℕ) (records : List StepRecords) :
    SegmentView Digest :=
  ⟨k, inp.ts, inp.memRoot, (inp.opsRoot, inp.finalRoot), records⟩

private theorem scanTuplesFrom_length (base : ℕ) (cs : List ScanSlot) :
    (scanTuplesFrom base cs).length = cs.length := by
  induction cs generalizing base with
  | nil => rfl
  | cons c cs ih => simp [scanTuplesFrom, ih]

private theorem chunkTuples_card (p : Plan) (idx : ℕ) (cs : List (List ScanSlot))
    (shaped : ∀ c ∈ cs, c.length = p.bScan) :
    Multiset.card (chunkTuples p idx cs) = cs.length * p.bScan := by
  induction cs generalizing idx with
  | nil => simp [chunkTuples]
  | cons c cs ih =>
    rw [chunkTuples, Multiset.card_add, Multiset.coe_card, scanTuples, scanTuplesFrom_length,
      shaped c (by simp), ih _ fun c' hc' => shaped c' (by simp [hc']), List.length_cons]
    ring

/-- Each side of a closing segment's check has at most `R + M + N · B_ops`
tuples. -/
theorem segment_multisets_card {ctx : Context E Digest} {η : E × E} {v : SegmentView Digest}
    (valid : ctx.plan.Valid) (closes : v.ClosesAt ctx η) :
    Multiset.card ((v.multisets ctx.plan).initial + (v.multisets ctx.plan).write) ≤
        ctx.plan.maxTuples ∧
      Multiset.card ((v.multisets ctx.plan).read + (v.multisets ctx.plan).final) ≤
        ctx.plan.maxTuples := by
  have scans (cs : List (List ScanSlot)) (shaped : ScansShaped ctx.plan cs)
      (length : cs.length = ctx.plan.n) : Multiset.card (chunkTuples ctx.plan 0 cs) =
        ctx.plan.cells := by
    rw [chunkTuples_card _ _ _ fun c hc => (shaped c hc).1, length, valid.exactCover]
  have initial : Multiset.card (v.multisets ctx.plan).initial = ctx.plan.cells := by
    rw [SegmentView.multisets, segmentMultisets_initial]
    exact scans _ closes.initialShaped (by rw [List.length_map, closes.length])
  have final : Multiset.card (v.multisets ctx.plan).final = ctx.plan.cells := by
    rw [SegmentView.multisets, segmentMultisets_final]
    exact scans _ closes.finalShaped (by rw [List.length_map, closes.length])
  have read : Multiset.card (v.multisets ctx.plan).read =
      (segmentOps v.openTs v.records).length := by
    rw [SegmentView.multisets, segmentMultisets_read, Multiset.coe_card, List.length_map]
  have write : Multiset.card (v.multisets ctx.plan).write =
      (segmentOps v.openTs v.records).length := by
    rw [SegmentView.multisets, segmentMultisets_write, Multiset.coe_card, List.length_map]
  have steps : (v.records.map activeCount).length = ctx.plan.n := by
    rw [List.length_map, closes.length]
  have ops : (segmentOps v.openTs v.records).length ≤ ctx.plan.n * ctx.plan.bOps := by
    rw [segmentOps_length, ← steps, ← smul_eq_mul]
    refine List.sum_le_card_nsmul _ _ fun x hx => ?_
    obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hx
    exact (List.length_filterMap_le _ _).trans
      (closes.opsShaped z.ops (List.mem_map.2 ⟨z, hz, rfl⟩)).1.le
  unfold Plan.maxTuples
  rw [Multiset.card_add, Multiset.card_add, initial, final, read, write]
  omega

private theorem all_add (a b : Multisets) : (a + b).all = a.all + b.all := by
  change a.read + b.read + (a.write + b.write) + (a.initial + b.initial) + (a.final + b.final) =
    a.read + a.write + a.initial + a.final + (b.read + b.write + b.initial + b.final)
  abel

private theorem segmentMultisets_small {p : Plan} (valid : p.Valid) {ts idx : ℕ}
    {rs : List StepRecords} (rows : SegmentRows p ts rs) (fits : idx + rs.length ≤ p.n) :
    ∀ τ ∈ (segmentMultisets p ts idx rs).all, τ.Small := by
  induction rs generalizing ts idx with
  | nil =>
    intro τ hτ
    change τ ∈ (0 : Multiset Tuple) + 0 + 0 + 0 at hτ
    simp at hτ
  | cons z rest ih =>
    obtain ⟨step, rows⟩ := rows
    rw [List.length_cons] at fits
    intro τ hτ
    rw [segmentMultisets, all_add, Multiset.mem_add] at hτ
    rcases hτ with hτ | hτ
    · exact (stepMultisets_small valid step (by omega) τ hτ).1
    · exact ih rows (by omega) τ hτ

/-- Every tuple of a closing segment is small. -/
theorem segment_multisets_small {ctx : Context E Digest} {η : E × E} {v : SegmentView Digest}
    (valid : ctx.plan.Valid) (closes : v.ClosesAt ctx η) :
    ∀ τ ∈ ((v.multisets ctx.plan).initial + (v.multisets ctx.plan).write) +
      ((v.multisets ctx.plan).read + (v.multisets ctx.plan).final), τ.Small := by
  intro τ hτ
  have all : (v.multisets ctx.plan).initial + (v.multisets ctx.plan).write +
      ((v.multisets ctx.plan).read + (v.multisets ctx.plan).final) =
        (v.multisets ctx.plan).all := by
    unfold Multisets.all
    abel
  rw [all] at hτ
  exact segmentMultisets_small valid closes.rows (by simp [closes.length]) τ hτ

/-- `Finset.mem_filter` for any decision instance: the filters of the retry
count use classical decisions that instance search does not rebuild. -/
private theorem mem_filter_any {α : Type} {p : α → Prop} {inst : DecidablePred p}
    {s : Finset α} {a : α} : a ∈ @Finset.filter α p inst s ↔ a ∈ s ∧ p a :=
  Finset.mem_filter

variable [IsDomain E] [Fintype E] [DecidableEq E] [CharP E goldilocksModulus]

/-- Security note Lemma 3 for one segment of the translated game:
`Pr[Err] ≤ 2·m_mem/|E| + Pr[Err and the retry disagrees]`, and every
disagreeing pair gives a collision among the two calls' chain inputs. -/
theorem segment_fingerprint_bound {ctx : Context E Digest} (valid : ctx.plan.Valid)
    (inp : EtaInput Digest) (k : ℕ) {Coins : Type} [Fintype Coins]
    (play : (E × E) × Coins → Option (List StepRecords))
    (closes : ∀ c z, play c = some z → (inp.view k z).ClosesAt ctx c.1) :
    ((errors play fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced).card : ℚ≥0) /
        Fintype.card ((E × E) × Coins) ≤
      2 * (ctx.plan.maxTuples : ℚ≥0) / Fintype.card E +
        ((disagreements play fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced).card :
            ℚ≥0) /
          ((Fintype.card ((E × E) × Coins) : ℚ≥0) * (successes play).card) ∧
      ∀ q ∈ disagreements play (fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced),
        ∃ z z', play q.1 = some z ∧ play q.2 = some z' ∧
          CollisionIn ctx.hash ((inp.view k z).chainInputs ctx) ((inp.view k z').chainInputs ctx) := by
  refine ⟨(retry_frequency play (fun z => ¬ ((inp.view k z).multisets ctx.plan).Balanced)
    (bound := 2 * ctx.plan.maxTuples * Fintype.card E * Fintype.card Coins)
    fun z bad => ?_).trans (add_le_add ?_ le_rfl), fun q hq => ?_⟩
  · by_cases hit : ∃ c, play c = some z
    · obtain ⟨c₀, hc₀⟩ := hit
      have size := segment_multisets_card valid (closes c₀ z hc₀)
      have small := segment_multisets_small valid (closes c₀ z hc₀)
      refine (Finset.card_le_card (t := BadChallenges (E := E)
          (((inp.view k z).multisets ctx.plan).initial + ((inp.view k z).multisets ctx.plan).write)
          (((inp.view k z).multisets ctx.plan).read + ((inp.view k z).multisets ctx.plan).final) ×ˢ
          (Finset.univ : Finset Coins)) fun c hc => ?_).trans ?_
      · have product := (productEq_iff c.1 _).1 (closes c z (mem_filter_any.1 hc).2).products
        exact Finset.mem_product.2
          ⟨Finset.mem_filter.2 ⟨Finset.mem_univ _, product⟩, Finset.mem_univ _⟩
      rw [Finset.card_product, Finset.card_univ]
      exact Nat.mul_le_mul_right _ (badChallenges_card bad small size.1 size.2)
    · refine (Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun c hc => ?_)).trans_le
        (Nat.zero_le _)
      exact hit ⟨c, (mem_filter_any.1 hc).2⟩
  · rw [Fintype.card_prod, Fintype.card_prod]
    push_cast
    rcases Nat.eq_zero_or_pos (Fintype.card Coins) with empty | hCoins
    · simp [empty]
    have hE : (0 : ℚ≥0) < Fintype.card E := by exact_mod_cast Fintype.card_pos
    have hCoins : (0 : ℚ≥0) < Fintype.card Coins := by exact_mod_cast hCoins
    rw [div_le_div_iff₀ (mul_pos (mul_pos hE hE) hCoins) hE]
    exact le_of_eq (by ring)
  · obtain ⟨pair, differ⟩ := mem_filter_any.1 hq
    obtain ⟨err, succ⟩ := Finset.mem_product.1 pair
    obtain ⟨-, z, hz, -⟩ := mem_filter_any.1 err
    obtain ⟨z', hz'⟩ := Option.isSome_iff_exists.1 (mem_filter_any.1 succ).2
    refine ⟨z, z', hz, hz', ?_⟩
    refine (records_eq_or_collision (closes _ _ hz) (closes _ _ hz') rfl).resolve_left
      fun same => ?_
    exact differ (by rw [hz, hz']; exact congrArg some same)

end Segment

end NightstreamFPrime.Spec.Nebula
