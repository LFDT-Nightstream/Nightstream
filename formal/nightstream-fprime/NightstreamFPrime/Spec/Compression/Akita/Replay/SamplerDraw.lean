import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFamily
import NightstreamFPrime.Spec.Compression.Akita.Extract.Draw

/-! The candidates of the sampler of the pinned Akita code (`Replay/Sampler.lean`) as a decoder
of one slot, in the form of the decoders of the game (`Extract/Draw.lean`, `FitChance.lean`;
Akita, ePrint 2026/1983, revision of 2026-11-23, Section 3.2 and (23)), and the stream of one
group.

- `candidate_mem`, `candidate_injective`, `candidate_isSome`: the decoder of one candidate gives
  members of `SparseFamily d pm1 pm2`, two candidates with one challenge are equal, and each rank
  below `rankSpace` decodes.
- `slotDecode G`: the decoder of one slot `Fin (2^K)` of the code, with the filter of the code
  at a filtered group. `slotDecode_mem`: its challenges are in the family and, at a filtered
  group, pass `TrigTable.codeAccepts`. `slotDecode_injective` is the hypothesis of
  `Extract.slotChance_le_of_injective`, so its slot probability is at most `2^-K`
  (`slotChance_slotDecode`). `slotDecode_isSome`: without a filter each slot decodes, which is the
  hypothesis `total` of `Key20.level₁_search_total` and `Key20.root_search_total`.
- `sample_filtered`: at a filtered group the challenges are the first `coordinates` accepted
  candidates of the pool, in order. `sample_mem`: each challenge of a group is in the family and,
  at a filtered group, passes the filter. `sample_isSome`: an unfiltered group always has its
  challenges. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open Sparse

theorem candidate_table (G : FoldGroup) (value : ℕ) :
    candidate G (chooseAt (binomials G.degree (max G.pm1 G.pm2))) value =
      candidate G Nat.choose value := by
  have table {n k : ℕ} (small : n ≤ G.degree) (low : k ≤ max G.pm1 G.pm2) :
      chooseAt (binomials G.degree (max G.pm1 G.pm2)) n k = n.choose k :=
    chooseAt_binomials small low
  simp only [candidate, table le_rfl (le_max_left _ _),
    unrank_congr (n := G.degree) (k := G.pm1) fun c small i low =>
      table small.le (low.trans (le_max_left _ _)),
    unrank_congr (n := G.degree - G.pm1) (k := G.pm2) fun c small i low =>
      table (by omega) (low.trans (le_max_right _ _))]

theorem candidate_some {G : FoldGroup} {value : ℕ} {c : Sparse}
    (found : candidate G Nat.choose value = some c) :
    value / 2 ^ (G.pm1 + G.pm2) < G.rankSpace ∧ ∃ ones slots twos,
      unrank Nat.choose G.degree G.pm1
        (value / 2 ^ (G.pm1 + G.pm2) % G.degree.choose G.pm1) = some ones ∧
      unrank Nat.choose (G.degree - G.pm1) G.pm2
        (value / 2 ^ (G.pm1 + G.pm2) / G.degree.choose G.pm1) = some slots ∧
      complement G.degree ones slots = some twos ∧
      c = pairs G.pm1 (value % 2 ^ (G.pm1 + G.pm2)) (ones ++ twos) := by
  unfold candidate at found
  by_cases range : G.rankSpace ≤ value / 2 ^ (G.pm1 + G.pm2)
  · simp [range] at found
  · simp only [range, ite_false, Option.bind_eq_some_iff, Option.map_eq_some_iff] at found
    obtain ⟨ones, unranked, slots, unranked', twos, complemented, rfl⟩ := found
    exact ⟨Nat.lt_of_not_le range, ones, slots, twos, unranked, unranked', complemented, rfl⟩

theorem map_fst_pairs (ones signs : ℕ) (positions : List ℕ) :
    (pairs ones signs positions).map Prod.fst = positions :=
  List.ext_getElem (by simp [pairs]) fun i _ _ => by simp [pairs]

theorem mem_pairs {ones signs : ℕ} {positions : List ℕ} {x : ℕ × ℤ} :
    x ∈ pairs ones signs positions ↔
      ∃ (i : ℕ) (small : i < positions.length), (positions[i], coefficient ones signs i) = x := by
  simp [pairs]

theorem value_pairs {ones signs : ℕ} {positions : List ℕ} (distinct : positions.Nodup) {i : ℕ}
    (small : i < positions.length) :
    value (pairs ones signs positions) positions[i] = coefficient ones signs i :=
  value_mem (by rw [map_fst_pairs]; exact distinct) (x := (positions[i], coefficient ones signs i))
    (mem_pairs.mpr ⟨i, small, rfl⟩)

theorem natAbs_coefficient (ones signs i : ℕ) :
    (coefficient ones signs i).natAbs = if i < ones then 1 else 2 := by
  unfold coefficient
  split_ifs <;> rfl

theorem coefficient_neg_iff (ones signs i : ℕ) : coefficient ones signs i < 0 ↔ signs.testBit i := by
  unfold coefficient
  split_ifs <;> simp_all

/-- The facts of the positions of a candidate. -/
structure Positions (d ones : ℕ) (first second : List ℕ) : Prop where
  length : first.length = ones
  ordered : first.Pairwise (· < ·)
  ordered' : second.Pairwise (· < ·)
  bounded : ∀ x ∈ first, x < d
  outside : ∀ x ∈ second, x < d ∧ x ∉ first

theorem Positions.distinct {d ones : ℕ} {first second : List ℕ}
    (P : Positions d ones first second) : (first ++ second).Nodup :=
  List.nodup_append.mpr ⟨P.ordered.imp Nat.ne_of_lt, P.ordered'.imp Nat.ne_of_lt,
    fun _ inside b inside' same => (P.outside b inside').2 (same ▸ inside)⟩

theorem positions_of {G : FoldGroup} {rank : ℕ} {ones slots twos : List ℕ}
    (unranked : unrank Nat.choose G.degree G.pm1 (rank % G.degree.choose G.pm1) = some ones)
    (unranked' : unrank Nat.choose (G.degree - G.pm1) G.pm2 (rank / G.degree.choose G.pm1) =
      some slots)
    (complemented : complement G.degree ones slots = some twos) :
    Positions G.degree G.pm1 ones twos ∧ twos.length = G.pm2 := by
  obtain ⟨length, bounded, ordered, -⟩ := unrank_some unranked
  obtain ⟨length', -, ordered', -⟩ := unrank_some unranked'
  obtain ⟨ordered'', outside⟩ := complement_ordered complemented ordered'
  refine ⟨⟨length, ordered, ordered'', bounded, outside⟩, ?_⟩
  rw [(complement_some complemented).1, List.length_map, length']

theorem Positions.valid {d ones : ℕ} {first second : List ℕ} (P : Positions d ones first second)
    (signs : ℕ) : Valid d (pairs ones signs (first ++ second)) := by
  refine ⟨by rw [map_fst_pairs]; exact P.distinct, fun x member => ?_, fun x member => ?_⟩
  · obtain ⟨i, small, rfl⟩ := mem_pairs.mp member
    rcases List.mem_append.mp (List.getElem_mem small) with inside | inside
    exacts [P.bounded _ inside, (P.outside _ inside).1]
  · obtain ⟨i, small, rfl⟩ := mem_pairs.mp member
    rw [natAbs_coefficient]
    split_ifs <;> omega

/-- The magnitude of the coefficient of a candidate at `j`: 1 at the first positions, 2 at the
second positions, and 0 elsewhere. -/
theorem Positions.natAbs_value {d ones : ℕ} {first second : List ℕ}
    (P : Positions d ones first second) (signs j : ℕ) :
    (value (pairs ones signs (first ++ second)) j).natAbs =
      if j ∈ first then 1 else if j ∈ second then 2 else 0 := by
  by_cases present : j ∈ first ++ second
  · obtain ⟨i, small, rfl⟩ := List.getElem_of_mem present
    rw [value_pairs P.distinct small, natAbs_coefficient]
    by_cases low : i < ones
    · have head : (first ++ second)[i] = first[i]'(by rw [P.length]; exact low) :=
        List.getElem_append_left (by rw [P.length]; exact low)
      rw [ite_eq_left low, head, ite_eq_left (List.getElem_mem _)]
    · have inside : (first ++ second)[i] ∈ second := by
        rw [List.getElem_append_right (by rw [P.length]; omega)]
        exact List.getElem_mem _
      rw [ite_eq_right low, ite_eq_right (P.outside _ inside).2, ite_eq_left inside]
  · rw [value_eq_zero (by rw [map_fst_pairs]; exact present)]
    simp only [List.mem_append, not_or] at present
    rw [ite_eq_right present.1, ite_eq_right present.2]
    rfl

theorem candidate_shape {G : FoldGroup} {value : ℕ} {c : Sparse}
    (found : candidate G Nat.choose value = some c) :
    ∃ ones twos, c = pairs G.pm1 (value % 2 ^ (G.pm1 + G.pm2)) (ones ++ twos) ∧
      Positions G.degree G.pm1 ones twos ∧ twos.length = G.pm2 := by
  obtain ⟨-, ones, slots, twos, unranked, unranked', complemented, rfl⟩ := candidate_some found
  exact ⟨ones, twos, rfl, positions_of unranked unranked' complemented⟩

theorem candidate_valid {G : FoldGroup} {value : ℕ} {c : Sparse}
    (found : candidate G Nat.choose value = some c) : Valid G.degree c := by
  obtain ⟨ones, twos, rfl, P, -⟩ := candidate_shape found
  exact P.valid _

theorem card_filter_eq {d ones m : ℕ} [NeZero d] {first second : List ℕ}
    (P : Positions d ones first second) (signs : ℕ) (l : List ℕ) (distinct : l.Nodup)
    (bounded : ∀ x ∈ l, x < d)
    (same : ∀ j, (if j ∈ first then 1 else if j ∈ second then 2 else 0) = m ↔ j ∈ l) :
    ((Finset.range d).filter fun j =>
      magnitude (coeff (ring d (pairs ones signs (first ++ second))) j) = m).card = l.length := by
  rw [← List.toFinset_card_of_nodup distinct]
  congr 1
  ext j
  simp only [Finset.mem_filter, Finset.mem_range, List.mem_toFinset]
  constructor
  · rintro ⟨small, equal⟩
    rw [magnitude_coeff (P.valid signs) small, P.natAbs_value] at equal
    exact (same j).mp equal
  · intro member
    have small := bounded j member
    exact ⟨small, by rw [magnitude_coeff (P.valid signs) small, P.natAbs_value]; exact (same j).mpr member⟩

/-- Each challenge of a candidate is a member of the family `SparseFamily d pm1 pm2`. -/
theorem candidate_mem {G : FoldGroup} [NeZero G.degree] {value : ℕ} {c : Sparse}
    (found : candidate G Nat.choose value = some c) :
    ring G.degree c ∈ SparseFamily G.degree G.pm1 G.pm2 := by
  obtain ⟨ones, twos, rfl, P, length'⟩ := candidate_shape found
  refine ⟨(normInf_le_iff _ _).mpr fun j small => ?_, ?_, ?_⟩
  · rw [magnitude_coeff (P.valid _) small]
    exact natAbs_value_le (P.valid _) j
  · rw [card_filter_eq P _ ones (P.ordered.imp Nat.ne_of_lt) P.bounded fun j => ?_, P.length]
    by_cases first : j ∈ ones
    · simp [first]
    · by_cases second : j ∈ twos <;> simp [first, second]
  · rw [card_filter_eq P _ twos (P.ordered'.imp Nat.ne_of_lt) (fun x member => (P.outside x member).1)
      fun j => ?_, length']
    by_cases first : j ∈ ones
    · have second : j ∉ twos := fun inside => (P.outside j inside).2 first
      simp [first, second]
    · by_cases second : j ∈ twos <;> simp [first, second]

theorem mem_iff_of_natAbs {d ones ones' : ℕ} {first second first' second' : List ℕ}
    (P : Positions d ones first second) (P' : Positions d ones' first' second') {signs signs' : ℕ}
    (values : ∀ j, value (pairs ones signs (first ++ second)) j =
      value (pairs ones' signs' (first' ++ second')) j) (j : ℕ) :
    (j ∈ first ↔ j ∈ first') ∧ (j ∈ second ↔ j ∈ second') := by
  have equal := congrArg Int.natAbs (values j)
  rw [P.natAbs_value, P'.natAbs_value] at equal
  have outside : j ∈ second → j ∉ first := fun inside => (P.outside j inside).2
  have outside' : j ∈ second' → j ∉ first' := fun inside => (P'.outside j inside).2
  constructor <;> constructor <;> intro member <;> split_ifs at equal <;> simp_all

theorem eq_of_mem_iff {l l' : List ℕ} (ordered : l.Pairwise (· < ·)) (ordered' : l'.Pairwise (· < ·))
    (same : ∀ j, j ∈ l ↔ j ∈ l') : l = l' :=
  List.Perm.eq_of_pairwise (fun _ _ _ _ low high => absurd (low.trans high) (lt_irrefl _))
    ordered ordered' (List.perm_of_nodup_nodup_toFinset_eq (ordered.imp Nat.ne_of_lt)
      (ordered'.imp Nat.ne_of_lt) (by ext j; simp [same]))

/-- Two candidates with the same element of `R_{q,d}` are equal: the decoder is injective on the
candidates that it decodes. -/
theorem candidate_injective {G : FoldGroup} [NeZero G.degree] {value value' : ℕ} {c c' : Sparse}
    (left : candidate G Nat.choose value = some c) (right : candidate G Nat.choose value' = some c')
    (same : ring G.degree c = ring G.degree c') : value = value' := by
  have values := value_eq_of_ring (candidate_valid left) (candidate_valid right) same
  obtain ⟨-, ones, slots, twos, unranked, unranked', complemented, rfl⟩ := candidate_some left
  obtain ⟨-, ones', slots', twos', other, other', complemented', rfl⟩ := candidate_some right
  obtain ⟨P, length⟩ := positions_of unranked unranked' complemented
  obtain ⟨P', length''⟩ := positions_of other other' complemented'
  have members := mem_iff_of_natAbs P P' values
  have first : ones = ones' := eq_of_mem_iff P.ordered P'.ordered fun j => (members j).1
  have second : twos = twos' := eq_of_mem_iff P.ordered' P'.ordered' fun j => (members j).2
  subst first second
  have slotsSame := complement_injective complemented complemented'
  subst slotsSame
  set w := G.pm1 + G.pm2
  have low := unrank_injective unranked other
  have high := unrank_injective unranked' other'
  have rank : value / 2 ^ w = value' / 2 ^ w := by
    rw [← Nat.div_add_mod (value / 2 ^ w) (G.degree.choose G.pm1),
      ← Nat.div_add_mod (value' / 2 ^ w) (G.degree.choose G.pm1), low, high]
  have signs : value % 2 ^ w = value' % 2 ^ w := by
    refine Nat.eq_of_testBit_eq fun i => ?_
    by_cases small : i < (ones ++ twos).length
    · have equal := values (ones ++ twos)[i]
      rw [value_pairs P.distinct small, value_pairs P.distinct small] at equal
      have negative := (coefficient_neg_iff G.pm1 (value % 2 ^ w) i).symm.trans
        (equal ▸ coefficient_neg_iff G.pm1 (value' % 2 ^ w) i)
      exact Bool.eq_iff_iff.mpr negative
    · have large : w ≤ i := by rw [List.length_append, P.length, length] at small; omega
      have below (n : ℕ) : n % 2 ^ w < 2 ^ i :=
        (Nat.mod_lt _ (Nat.two_pow_pos w)).trans_le (Nat.pow_le_pow_right (by decide) large)
      rw [Nat.testBit_lt_two_pow (below value), Nat.testBit_lt_two_pow (below value')]
  rw [← Nat.mod_add_div value (2 ^ w), ← Nat.mod_add_div value' (2 ^ w), signs, rank]

/-- A candidate whose rank is below `rankSpace` has a challenge. -/
theorem candidate_isSome {G : FoldGroup} {value : ℕ}
    (range : value / 2 ^ (G.pm1 + G.pm2) < G.rankSpace) :
    (candidate G Nat.choose value).isSome := by
  have space : G.rankSpace = G.degree.choose G.pm1 * (G.degree - G.pm1).choose G.pm2 := by
    rw [FoldGroup.rankSpace, FoldGroup.binomial_eq_choose, FoldGroup.binomial_eq_choose]
  have range' : value / 2 ^ (G.pm1 + G.pm2) <
      G.degree.choose G.pm1 * (G.degree - G.pm1).choose G.pm2 := space ▸ range
  have positive : 0 < G.degree.choose G.pm1 := Nat.pos_of_ne_zero fun zero => by
    rw [zero, Nat.zero_mul] at range'
    exact Nat.not_lt_zero _ range'
  obtain ⟨ones, unranked⟩ := Option.isSome_iff_exists.mp
    (unrank_isSome (n := G.degree) (k := G.pm1)
      (Nat.mod_lt (value / 2 ^ (G.pm1 + G.pm2)) positive))
  obtain ⟨slots, unranked'⟩ := Option.isSome_iff_exists.mp
    (unrank_isSome (n := G.degree - G.pm1) (k := G.pm2)
      ((Nat.div_lt_iff_lt_mul positive).mpr (by rw [Nat.mul_comm]; exact range')))
  obtain ⟨length, bounded, ordered, -⟩ := unrank_some unranked
  have fits : ∀ s ∈ slots, s < (free G.degree ones).length := fun s member => by
    rw [length_free (ordered.imp Nat.ne_of_lt) bounded, length]
    exact (unrank_some unranked').2.1 s member
  obtain ⟨twos, complemented⟩ := Option.isSome_iff_exists.mp (complement_isSome fits)
  simp only [candidate]
  rw [ite_eq_right (Nat.not_le.mpr range)]
  simp [unranked, unranked', complemented]

/-- The decoder of one slot of the code: the challenge of the candidate `slot`, when it passes
the filter of the code at a filtered group. -/
noncomputable def slotDecode (G : FoldGroup) [NeZero G.degree] (slot : Fin (2 ^ G.candidateBits)) :
    Option (Rq G.degree) :=
  ((candidate G Nat.choose slot).filter fun c => !G.filtered || (Filter.of G.degree).accepts c).map
    (ring G.degree)

theorem slotDecode_some {G : FoldGroup} [NeZero G.degree] {slot : Fin (2 ^ G.candidateBits)}
    {x : Rq G.degree} (decoded : slotDecode G slot = some x) :
    ∃ c, candidate G Nat.choose slot = some c ∧
      (!G.filtered || (Filter.of G.degree).accepts c) = true ∧ ring G.degree c = x := by
  simp only [slotDecode, Option.map_eq_some_iff, Option.filter_eq_some_iff] at decoded
  obtain ⟨c, ⟨found, kept⟩, rfl⟩ := decoded
  exact ⟨c, found, kept, rfl⟩

/-- Each challenge of `slotDecode` is in the family, and at a filtered group it passes the filter
of the code. -/
theorem slotDecode_mem {G : FoldGroup} [NeZero G.degree] {slot : Fin (2 ^ G.candidateBits)}
    {x : Rq G.degree} (decoded : slotDecode G slot = some x) :
    x ∈ SparseFamily G.degree G.pm1 G.pm2 ∧
      (G.filtered → TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x) := by
  obtain ⟨c, found, kept, rfl⟩ := slotDecode_some decoded
  refine ⟨candidate_mem found, fun filtered => (accepts_iff (candidate_valid found)).mp ?_⟩
  simpa [filtered] using kept

/-- `slotDecode` gives each challenge for at most one slot. -/
theorem slotDecode_injective (G : FoldGroup) [NeZero G.degree] (a b : Fin (2 ^ G.candidateBits))
    (x : Rq G.degree) (left : slotDecode G a = some x) (right : slotDecode G b = some x) :
    a = b := by
  obtain ⟨c, found, -, rfl⟩ := slotDecode_some left
  obtain ⟨c', found', -, same⟩ := slotDecode_some right
  exact Fin.ext (candidate_injective found found' same.symm)

/-- The slot probability of the decoder of the code is at most `2^-K`. -/
theorem slotChance_slotDecode (G : FoldGroup) [NeZero G.degree] :
    Extract.slotChance (slotDecode G) ≤ 1 / 2 ^ G.candidateBits := by
  have bound := Extract.slotChance_le_of_injective (slotDecode G) (slotDecode_injective G)
  rwa [Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat] at bound

/-- Without a filter each slot decodes. -/
theorem slotDecode_isSome {G : FoldGroup} [NeZero G.degree] (unfiltered : G.filtered = false)
    (positive : 0 < G.rankSpace) (slot : Fin (2 ^ G.candidateBits)) :
    (slotDecode G slot).isSome := by
  have bits : G.candidateBits = Nat.log 2 G.rankSpace + (G.pm1 + G.pm2) := by
    rw [FoldGroup.candidateBits_eq, FoldGroup.rankBits, unfiltered]
    rfl
  have range : (slot : ℕ) / 2 ^ (G.pm1 + G.pm2) < G.rankSpace := by
    refine lt_of_lt_of_le ((Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr ?_)
      (Nat.pow_log_le_self 2 positive.ne')
    rw [← Nat.pow_add, ← bits]
    exact slot.isLt
  obtain ⟨c, found⟩ := Option.isSome_iff_exists.mp (candidate_isSome range)
  simp [slotDecode, found, unfiltered]

theorem readBits_lt (bytes : ByteArray) (offset width : ℕ) :
    readBits bytes offset width < 2 ^ width :=
  Nat.mod_lt _ (Nat.two_pow_pos width)

/-- Without a filter the rank of each `K`-bit candidate is below `rankSpace`. -/
theorem rank_lt {G : FoldGroup} (unfiltered : G.filtered = false) (positive : 0 < G.rankSpace)
    {value : ℕ} (bits : value < 2 ^ G.candidateBits) :
    value / 2 ^ (G.pm1 + G.pm2) < G.rankSpace := by
  have width : G.candidateBits = Nat.log 2 G.rankSpace + (G.pm1 + G.pm2) := by
    rw [FoldGroup.candidateBits_eq, FoldGroup.rankBits, unfiltered]
    rfl
  refine lt_of_lt_of_le ((Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr ?_)
    (Nat.pow_log_le_self 2 positive.ne')
  rw [← Nat.pow_add, ← width]
  exact bits

theorem firstAccepted_eq {α : Type} (f : ℕ → Option α) (count : ℕ) :
    ∀ (index fuel : ℕ) (acc : List α), acc.length ≤ count →
      firstAccepted f count index fuel acc =
        acc ++ ((List.range' index fuel).filterMap f).take (count - acc.length)
  | index, 0, acc, _ => by simp [firstAccepted]
  | index, fuel + 1, acc, room => by
      unfold firstAccepted
      by_cases full : acc.length = count
      · rw [ite_eq_left full, full, Nat.sub_self, List.take_zero, List.append_nil]
      · rw [ite_eq_right full]
        cases found : f index with
        | none =>
            rw [firstAccepted_eq f count (index + 1) fuel _ (by simpa using room),
              List.range'_succ, List.filterMap_cons, found]
            simp
        | some a =>
            rw [firstAccepted_eq f count (index + 1) fuel _ (by simp; omega),
              List.range'_succ, List.filterMap_cons, found]
            simp only [Option.toList_some, List.append_assoc, List.singleton_append,
              List.length_append, List.length_singleton]
            rw [show count - acc.length = count - (acc.length + 1) + 1 by omega, List.take_succ_cons]

theorem length_filterMap {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).isSome) → (l.filterMap f).length = l.length
  | [], _ => rfl
  | a :: l, each => by
      obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp (each a List.mem_cons_self)
      rw [List.filterMap_cons, found, List.length_cons, List.length_cons,
        length_filterMap fun x member => each x (List.mem_cons_of_mem _ member)]

/-- At a filtered group the challenges are the first `coordinates` candidates of the pool that
decode and pass the filter of the code, in order. -/
theorem sample_filtered {G : FoldGroup} {bytes : ByteArray} {cs : List Sparse}
    (filtered : G.filtered = true) (sampled : sample G bytes = some cs) :
    cs = ((List.range G.candidates).filterMap fun j =>
        (candidate G Nat.choose (readBits bytes (j * G.candidateBits) G.candidateBits)).filter
          (Filter.of G.degree).accepts).take G.coordinates ∧ cs.length = G.coordinates := by
  simp only [sample, filtered, ite_true, candidate_table] at sampled
  split_ifs at sampled with size full
  cases sampled
  rw [firstAccepted_eq _ _ 0 _ [] (Nat.zero_le _), List.nil_append, List.length_nil,
    Nat.sub_zero, ← List.range_eq_range'] at full ⊢
  exact ⟨rfl, full⟩

/-- Each challenge of a group is a member of its family, and at a filtered group it passes the
filter of the code. -/
theorem sample_mem {G : FoldGroup} [NeZero G.degree] {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample G bytes = some cs) :
    cs.length = G.coordinates ∧ ∀ c ∈ cs, ring G.degree c ∈ SparseFamily G.degree G.pm1 G.pm2 ∧
      (G.filtered → TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree)
        (ring G.degree c)) := by
  cases filtered : G.filtered with
  | true =>
      obtain ⟨rfl, length⟩ := sample_filtered filtered sampled
      refine ⟨length, fun c member => ?_⟩
      obtain ⟨j, -, decoded⟩ := List.mem_filterMap.mp (List.mem_of_mem_take member)
      obtain ⟨found, kept⟩ := Option.filter_eq_some_iff.mp decoded
      exact ⟨candidate_mem found, fun _ => (accepts_iff (candidate_valid found)).mp kept⟩
  | false =>
      simp only [sample, filtered, Bool.false_eq_true, ite_false, candidate_table] at sampled
      split_ifs at sampled with size full
      cases sampled
      refine ⟨full, fun c member => ?_⟩
      obtain ⟨j, -, found⟩ := List.mem_filterMap.mp member
      exact ⟨candidate_mem found, fun no => absurd no (by simp)⟩

/-- An unfiltered group has its challenges for each message of its length. -/
theorem sample_isSome {G : FoldGroup} {bytes : ByteArray} (unfiltered : G.filtered = false)
    (positive : 0 < G.rankSpace) (size : bytes.size = G.messageBytes) :
    (sample G bytes).isSome := by
  have each : ∀ j ∈ List.range G.coordinates,
      (candidate G Nat.choose (readBits bytes (j * G.candidateBits) G.candidateBits)).isSome :=
    fun j _ => candidate_isSome (rank_lt unfiltered positive (readBits_lt _ _ _))
  simp only [sample, unfiltered, candidate_table, size, ne_eq, not_true_eq_false, ite_false,
    Bool.false_eq_true, length_filterMap each, List.length_range, ite_true, Option.isSome_some]

theorem value_ambient {s : ℕ} (positive : 0 < s) (c : Sparse) (i : ℕ) :
    value (c.map fun x => (x.1 * s, x.2)) i = if s ∣ i then value c (i / s) else 0 := by
  induction c with
  | nil => simp [value]
  | cons x c ih =>
      rw [List.map_cons, value_cons, ih, value_cons]
      by_cases divides : s ∣ i
      · obtain ⟨m, rfl⟩ := divides
        rw [ite_eq_left (Dvd.intro m rfl), ite_eq_left (Dvd.intro m rfl),
          Nat.mul_div_cancel_left m positive, Nat.mul_comm s m]
        by_cases same : x.1 = m
        · rw [ite_eq_left (by rw [same]), ite_eq_left same]
        · rw [ite_eq_right fun equal => same (Nat.eq_of_mul_eq_mul_right positive equal),
            ite_eq_right same]
      · rw [ite_eq_right divides, ite_eq_right divides,
          ite_eq_right fun equal => divides ⟨x.1, by rw [← equal, Nat.mul_comm]⟩, add_zero]

/-- T11: the challenge of a packing group in the ring of `A` is the embedding `X ↦ X^stride` of
the challenge in the ring of the carrier (Akita (38)). -/
theorem ring_ambient {G : FoldGroup} [NeZero G.degree] [NeZero G.ring]
    (tower : G.ring / G.degree * G.degree = G.ring) (c : Sparse) :
    ring G.ring (ambient G c) = embed (G.ring / G.degree) tower (ring G.degree c) := by
  have positive := stride_pos tower
  refine ext_coeff fun i small => ?_
  rw [coeff_ring _ small, coeff_embed, ambient, value_ambient positive]
  split_ifs with divides
  · rw [coeff_ring _ ((Nat.div_lt_iff_lt_mul positive).mpr (by rw [Nat.mul_comm, tower]; exact small))]
  · rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
