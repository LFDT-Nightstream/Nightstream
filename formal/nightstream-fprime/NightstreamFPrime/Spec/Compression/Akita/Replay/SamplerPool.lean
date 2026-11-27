import Mathlib.Data.Fintype.Fin
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerCount

/-! The shared pool of a filtered fold group of the pinned Akita code (revision `d50f700`,
`crates/akita-challenges/src/sampler/stream.rs`, `pool_len`, `first_accepted`: "Whether a
candidate is accepted does not depend on the accepted values") as a function of independent
slots (Akita, ePrint 2026/1983, revision of 2026-11-23, Appendix E.1, "Representatives").

- `rank accepts u i` is the count of the accepted positions of `u` before `i`. `pool accepts u a`
  is `u` with the accepted word of rank `r < W` replaced by `a r`. The pool has the acceptance
  pattern and the ranks of `u` (`pool_accepts`, `rank_pool`), and its accepted word of rank `r`
  is `a r` (`pool_rank`).
- `expect_pool`: for a uniform `u` and an independent uniform `a`, the pool is uniform. The map
  `(u, a) ↦ (pool u a, ranked u a)` is an involution (`swapPool_involutive`), and
  `expect_pool_pair` is the statement on the pairs.
- `getElem?_filterMap_rank`, `exists_rank`, `length_filterMap_ofFn`: the accepted words of a
  list, in order, are the words of rank 0, 1, ….
- `sample_pool`: at a filtered group the challenges are `slotDecode G` of the first `coordinates`
  accepted words of the message (`words`). `sample_slots`: for a message `pool u a` the challenge
  `j` is `slotDecode G (a j)`. `sample_isSome_iff`: a filtered group has its challenges exactly
  when the message has `coordinates` accepted words or more. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open scoped BigOperators

section Pool

variable {Word : Type} (accepts : Word → Prop) [DecidablePred accepts] {B W : ℕ}

/-- The count of the accepted positions of `u` before `i`. -/
def rank (u : Fin B → Word) (i : Fin B) : ℕ :=
  (Finset.univ.filter fun j => j < i ∧ accepts (u j)).card

/-- `u` with the accepted word of rank `r < W` replaced by `a r`. -/
def pool (u : Fin B → Word) (a : Fin W → {w // accepts w}) (i : Fin B) : Word :=
  if h : accepts (u i) ∧ rank accepts u i < W then a ⟨rank accepts u i, h.2⟩ else u i

/-- The pool has the acceptance pattern of `u`. -/
theorem pool_accepts (u : Fin B → Word) (a : Fin W → {w // accepts w}) (i : Fin B) :
    accepts (pool accepts u a i) ↔ accepts (u i) := by
  unfold pool
  split_ifs with h
  · exact ⟨fun _ => h.1, fun _ => (a _).2⟩
  · exact Iff.rfl

/-- The ranks depend only on the acceptance pattern. -/
theorem rank_congr {u v : Fin B → Word} (same : ∀ i, accepts (u i) ↔ accepts (v i)) :
    rank accepts u = rank accepts v := by
  funext i
  unfold rank
  rw [Finset.filter_congr fun j _ => and_congr_right' (same j)]

/-- The pool has the ranks of `u`. -/
theorem rank_pool (u : Fin B → Word) (a : Fin W → {w // accepts w}) :
    rank accepts (pool accepts u a) = rank accepts u :=
  rank_congr accepts (pool_accepts accepts u a)

/-- The accepted word of rank `r < W` of the pool is `a r`. -/
theorem pool_rank {u : Fin B → Word} (a : Fin W → {w // accepts w}) {i : Fin B}
    (accepted : accepts (u i)) (r : rank accepts u i < W) :
    pool accepts u a i = (a ⟨rank accepts u i, r⟩ : Word) := by
  unfold pool
  rw [dite_eq_left ⟨accepted, r⟩]

/-- The pool keeps each word that is rejected or of rank `W` or more. -/
theorem pool_of_not {u : Fin B → Word} (a : Fin W → {w // accepts w}) {i : Fin B}
    (outside : ¬ (accepts (u i) ∧ rank accepts u i < W)) : pool accepts u a i = u i := by
  unfold pool
  rw [dite_eq_right outside]

/-- An accepted position has a smaller rank than each later position. -/
theorem rank_lt_rank {u : Fin B → Word} {i j : Fin B} (accepted : accepts (u i))
    (lt : i < j) :
    rank accepts u i < rank accepts u j := by
  unfold rank
  refine Finset.card_lt_card ((Finset.ssubset_iff_of_subset fun k member => ?_).mpr
    ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, lt, accepted⟩, fun member => ?_⟩)
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member ⊢
    exact ⟨member.1.trans lt, member.2⟩
  · simp at member

/-- Two accepted positions with one rank are equal. -/
theorem rank_inj {u : Fin B → Word} {i j : Fin B} (left : accepts (u i)) (right : accepts (u j))
    (same : rank accepts u i = rank accepts u j) : i = j := by
  rcases lt_trichotomy i j with lt | eq | gt
  · exact absurd same (rank_lt_rank accepts left lt).ne
  · exact eq
  · exact absurd same (rank_lt_rank accepts right gt).ne'

/-- The accepted word of `u` of rank `r`, or `a r` when `u` has none. -/
noncomputable def ranked (u : Fin B → Word) (a : Fin W → {w // accepts w}) (r : Fin W) :
    {w // accepts w} :=
  if h : ∃ i, accepts (u i) ∧ rank accepts u i = r then ⟨u h.choose, h.choose_spec.1⟩
  else a r

/-- `ranked` gives the accepted word of the rank. -/
theorem ranked_eq {u : Fin B → Word} (a : Fin W → {w // accepts w}) {i : Fin B} {r : Fin W}
    (accepted : accepts (u i)) (same : rank accepts u i = r) :
    ranked accepts u a r = ⟨u i, accepted⟩ := by
  have found : ∃ i, accepts (u i) ∧ rank accepts u i = r := ⟨i, accepted, same⟩
  rw [ranked, dite_eq_left found]
  exact Subtype.ext (congrArg u (rank_inj accepts found.choose_spec.1 accepted
    (found.choose_spec.2.trans same.symm)))

/-- `ranked` gives `a r` when no accepted word has the rank `r`. -/
theorem ranked_of_not {u : Fin B → Word} (a : Fin W → {w // accepts w}) {r : Fin W}
    (none : ¬ ∃ i, accepts (u i) ∧ rank accepts u i = r) : ranked accepts u a r = a r := by
  rw [ranked, dite_eq_right none]

/-- `(u, a) ↦ (pool u a, ranked u a)`. -/
noncomputable def swapPool (x : (Fin B → Word) × (Fin W → {w // accepts w})) :
    (Fin B → Word) × (Fin W → {w // accepts w}) :=
  (pool accepts x.1 x.2, ranked accepts x.1 x.2)

/-- `swapPool` is an involution. -/
theorem swapPool_involutive : Function.Involutive (swapPool (B := B) (W := W) accepts) := by
  rintro ⟨u, a⟩
  refine Prod.ext (funext fun i => ?_) (funext fun r => ?_)
  · show pool accepts (pool accepts u a) (ranked accepts u a) i = u i
    by_cases inside : accepts (u i) ∧ rank accepts u i < W
    · have back (r : Fin W) (same : (r : ℕ) = rank accepts u i) :
          (ranked accepts u a r : Word) = u i := by
        rw [ranked_eq accepts a inside.1 same.symm]
      rw [pool_rank accepts _ ((pool_accepts accepts u a i).mpr inside.1)
        (by rw [rank_pool]; exact inside.2)]
      exact back _ (congrFun (rank_pool accepts u a) i)
    · rw [pool_of_not accepts _ (by rwa [pool_accepts accepts u a, rank_pool]),
        pool_of_not accepts a inside]
  · show ranked accepts (pool accepts u a) (ranked accepts u a) r = a r
    by_cases found : ∃ i, accepts (u i) ∧ rank accepts u i = r
    · obtain ⟨i, accepted, same⟩ := found
      rw [ranked_eq accepts _ ((pool_accepts accepts u a i).mpr accepted)
        (by rw [rank_pool]; exact same)]
      refine Subtype.ext (show pool accepts u a i = (a r : Word) from ?_)
      rw [pool_rank accepts a accepted (same ▸ r.isLt)]
      exact congrArg (fun r => (a r : Word)) (Fin.ext same)
    · rw [ranked_of_not accepts _ (by simpa only [pool_accepts, rank_pool] using found),
        ranked_of_not accepts a found]

variable [Fintype Word]

/-- The pool of a uniform pair `(u, a)` is uniform, as a statement on the pairs. -/
theorem expect_pool_pair (f : (Fin B → Word) → ℝ) :
    𝔼 x : (Fin B → Word) × (Fin W → {w // accepts w}), f (pool accepts x.1 x.2) =
      𝔼 x : (Fin B → Word) × (Fin W → {w // accepts w}), f x.1 :=
  Fintype.expect_equiv (Function.Involutive.toPerm _ (swapPool_involutive accepts)) _ _
    fun _ => rfl

/-- The pool of a uniform `u` and an independent uniform `a` is uniform. -/
theorem expect_pool [Nonempty {w // accepts w}] (f : (Fin B → Word) → ℝ) :
    𝔼 u, 𝔼 a : Fin W → {w // accepts w}, f (pool accepts u a) = 𝔼 p, f p :=
  calc 𝔼 u, 𝔼 a : Fin W → {w // accepts w}, f (pool accepts u a)
      = 𝔼 x ∈ Finset.univ ×ˢ Finset.univ, f (pool accepts x.1 x.2) :=
        (Finset.expect_product' _ _ fun u a => f (pool accepts u a)).symm
    _ = 𝔼 x ∈ Finset.univ ×ˢ Finset.univ, f x.1 := by
        rw [Finset.univ_product_univ]
        exact expect_pool_pair accepts f
    _ = 𝔼 p, 𝔼 _a : Fin W → {w // accepts w}, f p := Finset.expect_product' _ _ fun p _ => f p
    _ = 𝔼 p, f p := by simp only [Fintype.expect_const]

end Pool

section Lists

variable {Word α : Type} (accepts : Word → Prop) [DecidablePred accepts] {g : Word → Option α}

/-- The first position has the rank 0. -/
theorem rank_zero {B : ℕ} (u : Fin (B + 1) → Word) : rank accepts u 0 = 0 := by
  simp [rank]

/-- The rank of a later position counts the first position and the ranks of the tail. -/
theorem rank_succ {B : ℕ} (u : Fin (B + 1) → Word) (i : Fin B) :
    rank accepts u i.succ =
      (if accepts (u 0) then 1 else 0) + rank accepts (fun j => u j.succ) i := by
  unfold rank
  rw [Fin.card_filter_univ_succ']
  simp [Fin.succ_pos]

omit [DecidablePred accepts] in
/-- A rejected word has no value. -/
theorem none_of_rejected {w : Word} (some_iff : ∀ w, (g w).isSome ↔ accepts w)
    (rejected : ¬ accepts w) : g w = none :=
  Option.not_isSome_iff_eq_none.mp fun found => rejected ((some_iff w).mp found)

/-- The accepted word of rank `r` of a list is the element `r` of its accepted words. -/
theorem getElem?_filterMap_rank (some_iff : ∀ w, (g w).isSome ↔ accepts w) :
    ∀ {B : ℕ} (p : Fin B → Word) (i : Fin B), accepts (p i) →
      ((List.ofFn p).filterMap g)[rank accepts p i]? = g (p i)
  | 0, _, i, _ => i.elim0
  | B + 1, p, i, accepted => by
      rw [List.ofFn_succ]
      cases i using Fin.cases with
      | zero =>
          obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp ((some_iff _).mpr accepted)
          rw [rank_zero, List.filterMap_cons_some found, found]
          rfl
      | succ i =>
          have next := getElem?_filterMap_rank some_iff (fun j => p j.succ) i accepted
          rw [rank_succ]
          by_cases first : accepts (p 0)
          · obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp ((some_iff _).mpr first)
            rw [List.filterMap_cons_some found, ite_eq_left first, Nat.add_comm,
              List.getElem?_cons_succ]
            exact next
          · rw [List.filterMap_cons_none (none_of_rejected accepts some_iff first),
              ite_eq_right first, Nat.zero_add]
            exact next

/-- Each element of the accepted words of a list is the word of one rank. -/
theorem exists_rank (some_iff : ∀ w, (g w).isSome ↔ accepts w) :
    ∀ {B : ℕ} (p : Fin B → Word) (r : ℕ), r < ((List.ofFn p).filterMap g).length →
      ∃ i, accepts (p i) ∧ rank accepts p i = r
  | 0, _, r, small => by simp at small
  | B + 1, p, r, small => by
      rw [List.ofFn_succ] at small
      by_cases first : accepts (p 0)
      · obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp ((some_iff _).mpr first)
        rw [List.filterMap_cons_some found, List.length_cons] at small
        cases r with
        | zero => exact ⟨0, first, rank_zero accepts p⟩
        | succ r =>
            obtain ⟨i, accepted, same⟩ :=
              exists_rank some_iff (fun j => p j.succ) r (by omega)
            exact ⟨i.succ, accepted, by rw [rank_succ, ite_eq_left first, same, Nat.add_comm]⟩
      · rw [List.filterMap_cons_none (none_of_rejected accepts some_iff first)] at small
        obtain ⟨i, accepted, same⟩ := exists_rank some_iff (fun j => p j.succ) r small
        exact ⟨i.succ, accepted, by rw [rank_succ, ite_eq_right first, same, Nat.zero_add]⟩

/-- The accepted words of a list are as many as its accepted positions. -/
theorem length_filterMap_ofFn (some_iff : ∀ w, (g w).isSome ↔ accepts w) :
    ∀ {B : ℕ} (p : Fin B → Word),
      ((List.ofFn p).filterMap g).length = (Finset.univ.filter fun i => accepts (p i)).card
  | 0, _ => by simp
  | B + 1, p => by
      rw [List.ofFn_succ, Fin.card_filter_univ_succ',
        ← length_filterMap_ofFn some_iff (fun j => p j.succ)]
      by_cases first : accepts (p 0)
      · obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp ((some_iff _).mpr first)
        rw [List.filterMap_cons_some found, List.length_cons, ite_eq_left first, Nat.add_comm]
      · rw [List.filterMap_cons_none (none_of_rejected accepts some_iff first),
          ite_eq_right first, Nat.zero_add]

/-- The element `j < W` of the accepted words of the pool is `g (a j)`. -/
theorem filterMap_pool {B W : ℕ} (some_iff : ∀ w, (g w).isSome ↔ accepts w)
    (u : Fin B → Word) (a : Fin W → {w // accepts w}) (j : Fin W)
    (enough : (j : ℕ) < ((List.ofFn (pool accepts u a)).filterMap g).length) :
    ((List.ofFn (pool accepts u a)).filterMap g)[(j : ℕ)]? = g (a j) := by
  obtain ⟨i, accepted, same⟩ := exists_rank accepts some_iff _ _ enough
  rw [← same, getElem?_filterMap_rank accepts some_iff _ i accepted]
  rw [pool_accepts accepts u a] at accepted
  rw [rank_pool] at same
  rw [pool_rank accepts a accepted (same ▸ j.isLt)]
  exact congrArg (fun r => g (a r : Word)) (Fin.ext same)

end Lists

section Code

/-- The candidate words of the message of a group, as slots of `slotDecode`. -/
def words (G : FoldGroup) (bytes : ByteArray) (j : Fin G.candidates) :
    Fin (2 ^ G.candidateBits) :=
  ⟨readBits bytes (j * G.candidateBits) G.candidateBits, readBits_lt _ _ _⟩

/-- The accepted candidates of the pool of `sample` are `slotDecode G` of the words. -/
theorem filterMap_words {G : FoldGroup} [NeZero G.degree] (filtered : G.filtered = true)
    (bytes : ByteArray) :
    ((List.range G.candidates).filterMap fun j =>
        (candidate G Nat.choose (readBits bytes (j * G.candidateBits) G.candidateBits)).filter
          (Filter.of G.degree).accepts).map (Sparse.ring G.degree) =
      (List.ofFn (words G bytes)).filterMap (slotDecode G) := by
  rw [List.map_filterMap, List.ofFn_eq_map, List.filterMap_map, ← List.map_coe_finRange_eq_range,
    List.filterMap_map]
  congr 1
  funext j
  simp [slotDecode, words, filtered]

/-- At a filtered group the challenges are `slotDecode G` of the first `coordinates` accepted
words of the message, in order. -/
theorem sample_pool {G : FoldGroup} [NeZero G.degree] {bytes : ByteArray} {cs : List Sparse}
    (filtered : G.filtered = true) (sampled : sample G bytes = some cs) :
    cs.map (Sparse.ring G.degree) =
      ((List.ofFn (words G bytes)).filterMap (slotDecode G)).take G.coordinates := by
  obtain ⟨rfl, -⟩ := sample_filtered filtered sampled
  rw [List.map_take, filterMap_words filtered]

/-- For a message whose words are the pool `pool u a` of a filtered group, the challenge `j` is
`slotDecode G (a j)`. -/
theorem sample_slots {G : FoldGroup} [NeZero G.degree] {bytes : ByteArray} {cs : List Sparse}
    (filtered : G.filtered = true) {u : Fin G.candidates → Fin (2 ^ G.candidateBits)}
    {a : Fin G.coordinates → {w // (slotDecode G w).isSome}}
    (message : words G bytes = pool (fun w => (slotDecode G w).isSome) u a)
    (sampled : sample G bytes = some cs) (j : Fin G.coordinates) :
    cs[(j : ℕ)]?.map (Sparse.ring G.degree) = slotDecode G (a j) := by
  have listed := sample_pool filtered sampled
  have length := (sample_filtered filtered sampled).2
  rw [message] at listed
  rw [← List.getElem?_map, listed, List.getElem?_take, ite_eq_left j.isLt]
  refine filterMap_pool _ (fun _ => Iff.rfl) u a j ?_
  have counted := congrArg List.length listed
  rw [List.length_map, List.length_take, length] at counted
  omega

/-- A filtered group has its challenges for a message of its length exactly when the message
has `coordinates` accepted words or more. -/
theorem sample_isSome_iff {G : FoldGroup} [NeZero G.degree] {bytes : ByteArray}
    (filtered : G.filtered = true) (size : bytes.size = G.messageBytes) :
    (sample G bytes).isSome ↔
      G.coordinates ≤
        (Finset.univ.filter fun j => (slotDecode G (words G bytes j)).isSome).card := by
  rw [← length_filterMap_ofFn (fun w => (slotDecode G w).isSome = true) (fun _ => Iff.rfl),
    ← filterMap_words filtered, List.length_map]
  simp only [sample, filtered, ite_true, candidate_table, size, ne_eq, not_true_eq_false,
    ite_false]
  rw [firstAccepted_eq _ _ 0 _ [] (Nat.zero_le _), List.nil_append, List.length_nil, Nat.sub_zero,
    ← List.range_eq_range', List.length_take]
  split <;> simp_all

end Code

end NightstreamFPrime.Spec.Compression.Akita.Replay
