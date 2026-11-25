import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalSetup
import NightstreamFPrime.Spec.Compression.Akita.Setup

/-! The flat setup vector of the pinned Akita code in pages, against the setup expansion of the
specification (Akita revision `d50f700`, `akita-types/src/proof/setup.rs`, `setup/range.rs`,
`PublicMatrixDerivation::Shake256PagedV1`; ePrint 2026/1983, (36) and the term `Δ_setup` of
Theorem 10.46; `decisions/akita-compression.md`, difference (c)).

The specification expands the flat vector from one stream of words with exact rejection
(`Akita.Setup.expand`) and bounds the event of a short stream (`Akita.Setup.expand_le`). The code
expands it in pages of 4096 elements: page `j` is the expansion of its own SHAKE256 stream after
a header with `j` (`Replay.Setup.page`).

- `pageStream seed j count` is the stream of page `j`: the first `count` words of 64 bits of its
  SHAKE256 output. This is a DEFINITION of the concrete instance (SHAKE256 and the header).
- `page_expand`: when the loop of `Replay.Setup.page` keeps `count` words of the first
  `count + 64`, the page is `Akita.Setup.expand` of `pageStream seed j (count + 64)`
  (`forIn_kept`, `kept_run`, `page_toList`: the loop keeps the first `count` words below `q`).
- `paged streams` is the flat vector of the code for one stream of each page: entry `p` is entry
  `p mod 4096` of `Akita.Setup.expand` of the stream of page `⌊p / 4096⌋` (`paged_apply`).
- `sum_paged`: on the page streams that each keep 4096 words or more, the flat vector is uniform.
  `paged_le` is the setup hybrid of the code: for independent uniform page streams, the
  probability of an event is at most its probability for a uniform flat vector plus, for each
  page, the chance of a short stream. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.AjtaiSetupV1.Rejection
open NightstreamFPrime.Spec.AjtaiSetupV1 (keptWords keptCoefficientNat keptCoefficientNat_of_kept)
open scoped BigOperators

/-- The first `count` words of the SHAKE256 stream of page `j` of the seed. -/
def pageStream (seed : ByteArray) (j count : ℕ) : Fin count → Word :=
  let bytes := Setup.shake256 (Setup.pageHeader seed j) (8 * count)
  fun i => ⟨(Setup.word bytes (8 * i)).toNat, UInt64.toNat_lt _⟩

/-- The flat vector of `pages` blocks of `size` entries, block `j` at the entries from
`j·size`. -/
def flatten (pages size : ℕ) : (Fin pages → Fin size → Zq) ≃ (Fin (pages * size) → Zq) :=
  (Equiv.curry (Fin pages) (Fin size) Zq).symm.trans
    (Equiv.arrowCongr finProdFinEquiv (Equiv.refl Zq))

variable {pages count : ℕ}

/-- The flat setup vector of the code for the stream `streams j` of each page `j`. -/
def paged (streams : Fin pages → Fin count → Word) : Fin (pages * Setup.pageSize) → Zq :=
  flatten pages Setup.pageSize fun j => Akita.Setup.expand (streams j)

theorem paged_apply (streams : Fin pages → Fin count → Word) (p : Fin (pages * Setup.pageSize)) :
    paged streams p = Akita.Setup.expand (streams p.divNat) p.modNat := by
  simp [paged, flatten, finProdFinEquiv_symm_apply]

/-- A loop that keeps the elements `g i` that are present until it has `count` of them. -/
theorem forIn_kept (g : ℕ → Option Fq) (count : ℕ)
    (f : ℕ → Array Fq → Id (ForInStep (Array Fq)))
    (step : ∀ i r, f i r = if r.size = count then ForInStep.done r
      else ForInStep.yield ((g i).elim r r.push)) :
    ∀ (l : List ℕ) (r : Array Fq), r.size ≤ count →
      (Id.run (forIn l r f)).toList = r.toList ++ (l.filterMap g).take (count - r.size)
  | [], r, _ => by
      simp [Id.run]
      rfl
  | i :: l, r, small => by
      rw [List.forIn_cons, step]
      by_cases full : r.size = count
      · rw [ite_eq_left full]
        show r.toList = _
        simp [full]
      · rw [ite_eq_right full]
        show (Id.run (forIn l ((g i).elim r r.push) f)).toList = _
        cases present : g i with
        | none =>
            simpa [present] using forIn_kept g count f step l r small
        | some a =>
            have below : r.size < count := lt_of_le_of_ne small full
            have next := forIn_kept g count f step l (r.push a) (by simp; omega)
            simp only [Array.size_push, Array.toList_push] at next
            simp only [Option.elim]
            rw [next, List.filterMap_cons_some present,
              show count - r.size = (count - (r.size + 1)) + 1 by omega, List.take_succ_cons,
              List.append_assoc, List.singleton_append]

/-- The loop and the result of `Setup.page`. -/
theorem kept_run (g : ℕ → Option Fq) (count : ℕ) (l : List ℕ)
    (f : ℕ → Array Fq → Id (ForInStep (Array Fq)))
    (step : ∀ i r, f i r = if r.size = count then ForInStep.done r
      else ForInStep.yield ((g i).elim r r.push)) {kept : Array Fq}
    (found : (forIn l (Array.emptyWithCapacity count) f >>= fun s =>
      pure (if s.size = count then some s else none) : Id (Option (Array Fq))).run = some kept) :
    kept.toList = (l.filterMap g).take count ∧ kept.size = count := by
  have loop := forIn_kept g count f step l (Array.emptyWithCapacity count) (by simp)
  change (if (Id.run (forIn l (Array.emptyWithCapacity count) f)).size = count then
    some (Id.run (forIn l (Array.emptyWithCapacity count) f)) else none) = some kept at found
  by_cases full : (Id.run (forIn l (Array.emptyWithCapacity count) f)).size = count
  · rw [ite_eq_left full, Option.some.injEq] at found
    subst found
    refine ⟨?_, full⟩
    simpa using loop
  · rw [ite_eq_right full] at found
    cases found

/-- The elements of a page of the code are the kept words of its first `count + 64` words, in
order, when `count` of them are kept. -/
theorem page_toList (seed : ByteArray) (j count : ℕ) {kept : Array Fq}
    (found : Setup.page seed j count = some kept) :
    kept.toList = ((List.range' 0 (count + 64)).filterMap fun i => Fq.ofWord? (Setup.word
      (Setup.shake256 (Setup.pageHeader seed j) (8 * (count + 64))) (8 * i))).take count ∧
      kept.size = count := by
  unfold Setup.page at found
  simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at found
  have run := kept_run (fun i => Fq.ofWord? (Setup.word
    (Setup.shake256 (Setup.pageHeader seed j) (8 * (count + 64))) (8 * i))) count _ _
    (fun i r => by
      by_cases full : r.size = count
      · simp [full]
        rfl
      · simp only [full, ite_false]
        cases Fq.ofWord? (Setup.word (Setup.shake256 (Setup.pageHeader seed j) (8 * (count + 64)))
          (8 * i)) <;> rfl) found
  simpa using run

theorem kept_words (w : ℕ → UInt64) : ∀ l : List ℕ,
    (l.filterMap fun i => Fq.ofWord? (w i)).map (fun a => a.word.toNat) =
      (l.map fun i => (w i).toNat).filter (· < fieldModulus)
  | [] => rfl
  | i :: l => by
      have ih := kept_words w l
      simp only [Fq.ofWord?] at ih
      by_cases kept : (w i).toNat < fieldModulus
      · simp [Fq.ofWord?, kept, ih]
      · simp [Fq.ofWord?, kept, ih]

theorem finEquiv_mk {n : ℕ} (small : n < fieldModulus) :
    ZMod.finEquiv fieldModulus ⟨n, small⟩ = (n : Zq) := by
  rw [← ZMod.natCast_zmod_val (ZMod.finEquiv fieldModulus ⟨n, small⟩)]
  rfl

/-- A page of the code is the expansion `Akita.Setup.expand` of the specification of the first
`count + 64` words of its stream (`pageStream`), when the code keeps `count` of them. -/
theorem page_expand (seed : ByteArray) (j count : ℕ) {kept : Array Fq}
    (found : Setup.page seed j count = some kept) :
    kept.toList.map Fq.toZq =
      List.ofFn (Akita.Setup.expand (N := count) (pageStream seed j (count + 64))) := by
  obtain ⟨list, size⟩ := page_toList seed j count found
  have words_eq : words (pageStream seed j (count + 64)) = (List.range' 0 (count + 64)).map
      fun i => (Setup.word (Setup.shake256 (Setup.pageHeader seed j) (8 * (count + 64)))
        (8 * i)).toNat := by
    unfold words
    rw [List.ofFn_eq_map, ← List.range_eq_range', ← List.map_coe_finRange_eq_range, List.map_map]
    rfl
  have kept_eq := kept_words (fun i => Setup.word (Setup.shake256 (Setup.pageHeader seed j)
    (8 * (count + 64))) (8 * i)) (List.range' 0 (count + 64))
  rw [← words_eq] at kept_eq
  have long : count ≤ (keptWords (words (pageStream seed j (count + 64)))).length := by
    have lengths := congrArg List.length list
    rw [Array.length_toList, size, List.length_take] at lengths
    rw [keptWords, ← kept_eq, List.length_map]
    omega
  refine List.ext_getElem (by simp [size]) fun p left right => ?_
  have inside : p < count := by simpa [size] using left
  rw [List.getElem_map, List.getElem_ofFn]
  show (((kept.toList[p]'(by simpa [size] using inside)).word.toNat : ℕ) : Zq) =
    ZMod.finEquiv fieldModulus ⟨_, _⟩
  rw [finEquiv_mk, keptCoefficientNat_of_kept _ _ (by omega)]
  congr 1
  have kw : keptWords (words (pageStream seed j (count + 64))) = _ := kept_eq.symm
  rw [List.getElem_of_eq list, List.getElem_take, List.getElem_of_eq kw, List.getElem_map]

/-- The streams of one page that keep `4096` words or more and expand to `t`: as many as for
each `t`. -/
theorem card_page (t : Fin Setup.pageSize → Zq) :
    (Finset.univ.filter fun s : Fin count → Word =>
      ¬ StreamFallback Setup.pageSize s ∧ Akita.Setup.expand s = t).card =
      startCount wordCount count Setup.pageSize := by
  let lift : (Fin Setup.pageSize → F) ≃ (Fin Setup.pageSize → Zq) :=
    Equiv.piCongrRight fun _ => (ZMod.finEquiv fieldModulus).toEquiv
  rw [← card_streamElement (count := count) fieldModulus_le (lift.symm t)]
  refine congrArg Finset.card (Finset.filter_congr fun s _ => and_congr_right fun _ => ?_)
  show lift (streamElement s) = t ↔ _
  exact (Equiv.eq_symm_apply lift).symm

/-- On the page streams that each keep `4096` words or more, the flat vector of the code is
uniform: each flat vector is the vector of `startCount^pages` of them. -/
theorem sum_paged (value : (Fin (pages * Setup.pageSize) → Zq) → ℝ) :
    ∑ streams : Fin pages → Fin count → Word,
        (if ∃ j, StreamFallback Setup.pageSize (streams j) then 0 else value (paged streams)) =
      (startCount wordCount count Setup.pageSize : ℝ) ^ pages * ∑ S, value S := by
  have fibers : ∀ streams : Fin pages → Fin count → Word,
      (if ∃ j, StreamFallback Setup.pageSize (streams j) then 0 else value (paged streams)) =
        ∑ T : Fin pages → Fin Setup.pageSize → Zq,
          if ∀ j, ¬ StreamFallback Setup.pageSize (streams j) ∧
            Akita.Setup.expand (streams j) = T j then value (flatten _ _ T) else 0 := by
    intro streams
    by_cases bad : ∃ j, StreamFallback Setup.pageSize (streams j)
    · rw [ite_eq_left bad]
      refine (Finset.sum_eq_zero fun T _ => ite_eq_right fun all => ?_).symm
      obtain ⟨j, short⟩ := bad
      exact (all j).1 short
    · rw [ite_eq_right bad, Finset.sum_eq_single (fun j => Akita.Setup.expand (streams j))]
      · rw [ite_eq_left fun j => ⟨fun short => bad ⟨j, short⟩, rfl⟩]
        rfl
      · exact fun T _ other => ite_eq_right fun all => other (funext fun j => ((all j).2).symm)
      · exact fun absent => absurd (Finset.mem_univ _) absent
  rw [Finset.sum_congr rfl fun streams _ => fibers streams, Finset.sum_comm,
    ← (flatten pages Setup.pageSize).sum_comp value, Finset.mul_sum]
  refine Finset.sum_congr rfl fun T _ => ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
  have product : (Finset.univ.filter fun streams : Fin pages → Fin count → Word =>
      ∀ j, ¬ StreamFallback Setup.pageSize (streams j) ∧ Akita.Setup.expand (streams j) = T j) =
      Fintype.piFinset fun j => Finset.univ.filter fun s : Fin count → Word =>
        ¬ StreamFallback Setup.pageSize s ∧ Akita.Setup.expand s = T j := by
    ext streams
    simp [Fintype.mem_piFinset]
  rw [product, Fintype.card_piFinset]
  simp only [card_page, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Nat.cast_pow]

/-- The streams whose page `j` is short, as a fraction of all page streams: the fraction of the
short streams of one page. -/
theorem short_fraction (j : Fin pages) :
    ((Finset.univ.filter fun streams : Fin pages → Fin count → Word =>
        StreamFallback Setup.pageSize (streams j)).card : ℝ) /
        Fintype.card (Fin pages → Fin count → Word) =
      ((Finset.univ.filter (StreamFallback Setup.pageSize (count := count)
        (bound := wordCount))).card : ℝ) / Fintype.card (Fin count → Word) := by
  let split := Equiv.funSplitAt j (Fin count → Word)
  have counted : (Finset.univ.filter fun streams : Fin pages → Fin count → Word =>
      StreamFallback Setup.pageSize (streams j)).card =
      (Finset.univ.filter (StreamFallback Setup.pageSize (count := count)
        (bound := wordCount))).card * Fintype.card ({k // k ≠ j} → Fin count → Word) := by
    rw [← Finset.card_univ (α := {k // k ≠ j} → Fin count → Word), ← Finset.card_product]
    refine Finset.card_bij (fun streams _ => split streams) (fun streams member => ?_)
      (fun a _ b _ same => split.injective same) (fun pair member => ?_)
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member
      simp only [Finset.mem_product, Finset.mem_filter, Finset.mem_univ, true_and, and_true]
      exact member
    · refine ⟨split.symm pair, ?_, split.apply_symm_apply pair⟩
      simp only [Finset.mem_product, Finset.mem_filter, Finset.mem_univ, true_and,
        and_true] at member
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have first : (split.symm pair) j = pair.1 := by
        have := congrArg Prod.fst (split.apply_symm_apply pair)
        exact this
      rw [first]
      exact member
  have total : Fintype.card (Fin pages → Fin count → Word) =
      Fintype.card (Fin count → Word) * Fintype.card ({k // k ≠ j} → Fin count → Word) := by
    rw [Fintype.card_congr split, Fintype.card_prod]
  have positive : (0 : ℝ) < Fintype.card ({k // k ≠ j} → Fin count → Word) := by
    exact_mod_cast Fintype.card_pos
  rw [counted, total]
  push_cast
  rw [mul_div_mul_right _ _ positive.ne']

/-- The setup hybrid of the code (the term `Δ_setup` of Akita Theorem 10.46 for the paged form):
for independent uniform streams of `count` words for each page, the probability of an event of the
flat vector is at most its probability for a uniform flat vector plus, for each page, the chance
`C(count, r)·(59/2^64)^r` of a short stream, `r = count − 4096 + 1`. -/
theorem paged_le (enough : Setup.pageSize ≤ count)
    (event : (Fin (pages * Setup.pageSize) → Zq) → Prop) [DecidablePred event] :
    𝔼 streams : Fin pages → Fin count → Word, (if event (paged streams) then (1 : ℝ) else 0) ≤
      (𝔼 S : Fin (pages * Setup.pageSize) → Zq, if event S then (1 : ℝ) else 0) +
        pages * ((count.choose (count - Setup.pageSize + 1) : ℝ) *
          ((wordCount - fieldModulus : ℕ) : ℝ) ^ (count - Setup.pageSize + 1) /
          (wordCount : ℝ) ^ (count - Setup.pageSize + 1)) := by
  have streams : (0 : ℝ) < Fintype.card (Fin pages → Fin count → Word) := by
    exact_mod_cast Fintype.card_pos
  have split : ∀ streams : Fin pages → Fin count → Word,
      (if event (paged streams) then (1 : ℝ) else 0) ≤
        (if ∃ j, StreamFallback Setup.pageSize (streams j) then 0
          else if event (paged streams) then (1 : ℝ) else 0) +
          ∑ j, if StreamFallback Setup.pageSize (streams j) then 1 else 0 := by
    intro streams
    have each : 0 ≤ ∑ j, (if StreamFallback Setup.pageSize (streams j) then (1 : ℝ) else 0) :=
      Finset.sum_nonneg fun j _ => by split <;> norm_num
    by_cases bad : ∃ j, StreamFallback Setup.pageSize (streams j)
    · obtain ⟨j, short⟩ := bad
      rw [ite_eq_left (show ∃ k, StreamFallback Setup.pageSize (streams k) from ⟨j, short⟩),
        zero_add]
      have one : (1 : ℝ) ≤ ∑ k, if StreamFallback Setup.pageSize (streams k) then 1 else 0 :=
        (show (1 : ℝ) = if StreamFallback Setup.pageSize (streams j) then 1 else 0 by
          rw [ite_eq_left short]).le.trans (Finset.single_le_sum (f := fun k =>
            if StreamFallback Setup.pageSize (streams k) then (1 : ℝ) else 0)
            (fun k _ => by split <;> norm_num) (Finset.mem_univ j))
      split <;> linarith
    · rw [ite_eq_right bad]
      linarith
  have kept := sum_paged (count := count) fun S => if event S then (1 : ℝ) else 0
  have total := sum_paged (count := count) fun _ : Fin (pages * Setup.pageSize) → Zq => (1 : ℝ)
  have fewer : (startCount wordCount count Setup.pageSize : ℝ) ^ pages *
      Fintype.card (Fin (pages * Setup.pageSize) → Zq) ≤
      Fintype.card (Fin pages → Fin count → Word) := by
    have ones : ∑ _S : Fin (pages * Setup.pageSize) → Zq, (1 : ℝ) =
        Fintype.card (Fin (pages * Setup.pageSize) → Zq) := by simp
    have all : ∑ _streams : Fin pages → Fin count → Word, (1 : ℝ) =
        Fintype.card (Fin pages → Fin count → Word) := by simp
    rw [← ones, ← total, ← all]
    exact Finset.sum_le_sum fun streams _ => by split <;> norm_num
  have chance := streamFallback_chance_le (bound := wordCount) (count := count)
    (degree := Setup.pageSize) fieldModulus_le (by decide) enough
  have chanceReal : ((Finset.univ.filter (StreamFallback Setup.pageSize (count := count)
      (bound := wordCount))).card : ℝ) / Fintype.card (Fin count → Word) ≤
      (count.choose (count - Setup.pageSize + 1) : ℝ) *
        ((wordCount - fieldModulus : ℕ) : ℝ) ^ (count - Setup.pageSize + 1) /
        (wordCount : ℝ) ^ (count - Setup.pageSize + 1) := by
    have cast := (Rat.cast_le (K := ℝ)).mpr chance
    simp only [Rat.cast_div, Rat.cast_mul, Rat.cast_pow, Rat.cast_natCast] at cast
    exact cast
  refine (Finset.expect_le_expect fun streams _ => split streams).trans ?_
  rw [Finset.expect_add_distrib]
  refine add_le_add ?_ ?_
  · rw [Finset.expect_eq_sum_div_card, kept, Finset.expect_eq_sum_div_card, Finset.card_univ,
      Finset.card_univ, div_le_div_iff₀ streams (by exact_mod_cast Fintype.card_pos)]
    have nonnegative : (0 : ℝ) ≤ ∑ S : Fin (pages * Setup.pageSize) → Zq,
        if event S then (1 : ℝ) else 0 :=
      Finset.sum_nonneg fun S _ => by split <;> norm_num
    rw [mul_right_comm, mul_comm _ (∑ S, _)]
    exact mul_le_mul_of_nonneg_left fewer nonnegative
  · calc 𝔼 streams : Fin pages → Fin count → Word, ∑ j, (if StreamFallback Setup.pageSize
            (streams j) then (1 : ℝ) else 0)
        = ∑ j : Fin pages, ((Finset.univ.filter fun streams : Fin pages → Fin count → Word =>
            StreamFallback Setup.pageSize (streams j)).card : ℝ) /
            Fintype.card (Fin pages → Fin count → Word) := by
          rw [Finset.expect_sum_comm]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.expect_eq_sum_div_card, Finset.card_univ, ← Finset.sum_filter,
            Finset.sum_const, nsmul_eq_mul, mul_one]
      _ ≤ ∑ _j : Fin pages, (count.choose (count - Setup.pageSize + 1) : ℝ) *
            ((wordCount - fieldModulus : ℕ) : ℝ) ^ (count - Setup.pageSize + 1) /
            (wordCount : ℝ) ^ (count - Setup.pageSize + 1) :=
          Finset.sum_le_sum fun j _ => (short_fraction j).le.trans chanceReal
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end NightstreamFPrime.Spec.Compression.Akita.Replay
