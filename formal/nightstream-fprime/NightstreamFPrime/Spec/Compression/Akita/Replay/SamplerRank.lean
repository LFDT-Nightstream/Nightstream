import NightstreamFPrime.Spec.Compression.Akita.Replay.Sampler

/-! The rank decoder of the sampler of the pinned Akita code (`Replay/Sampler.lean`,
`crates/akita-challenges/src/sampler/stream.rs`) against the binomial coefficients.

- `chooseAt_binomials`: an entry of the table is `Nat.choose`, so `candidate` with the table is
  `candidate` with `Nat.choose` (`candidate_table`).
- `unrank_some`: an unrank of `k` elements below `n` is a strictly increasing list of `k`
  elements below `n` whose colexicographic rank (`colexRank`) is the rank. So the rank is a
  function of the list (`unrank_injective`). `unrank_isSome`: each rank below `C(n, k)` has an
  unrank (the combinatorial number system).
- `complement_some`, `complement_isSome`, `complement_injective`: the slots below `d − |taken|`
  map to a strictly increasing list of elements below `d` outside `taken`, and different slots to
  different elements. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

theorem chooseAt_binomials {maxN maxK n k : ℕ} (small : n ≤ maxN) (low : k ≤ maxK) :
    chooseAt (binomials maxN maxK) n k = n.choose k := by
  simp [chooseAt, binomials, Nat.lt_succ_of_le small, Nat.lt_succ_of_le low,
    FoldGroup.binomial_eq_choose]

theorem scan_congr {B : ℕ → ℕ → ℕ} {i rank : ℕ} :
    ∀ {upper : ℕ}, (∀ c < upper, B c i = c.choose i) →
      scan B i rank upper = scan Nat.choose i rank upper
  | 0, _ => rfl
  | u + 1, same => by
      simp only [scan, same u (Nat.lt_succ_self u),
        scan_congr (upper := u) fun c small => same c (Nat.lt_succ_of_lt small)]

theorem scan_some {i rank : ℕ} : ∀ {upper c : ℕ}, scan Nat.choose i rank upper = some c →
    c < upper ∧ c.choose i ≤ rank ∧ ∀ c', c < c' → c' < upper → rank < c'.choose i
  | 0, _, found => by simp [scan] at found
  | u + 1, c, found => by
      unfold scan at found
      split_ifs at found with fits
      · cases found
        exact ⟨Nat.lt_succ_self _, fits, fun c' above below => by omega⟩
      · obtain ⟨below, fits', above⟩ := scan_some found
        refine ⟨Nat.lt_succ_of_lt below, fits', fun c' larger smaller => ?_⟩
        rcases Nat.lt_succ_iff_lt_or_eq.mp smaller with inside | rfl
        · exact above c' larger inside
        · exact Nat.lt_of_not_le fits

theorem scan_isSome {i rank : ℕ} : ∀ {upper c : ℕ}, c < upper → c.choose i ≤ rank →
    (scan Nat.choose i rank upper).isSome
  | 0, _, below, _ => absurd below (Nat.not_lt_zero _)
  | u + 1, c, below, fits => by
      unfold scan
      split_ifs with fits'
      · rfl
      · rcases Nat.lt_succ_iff_lt_or_eq.mp below with inside | rfl
        · exact scan_isSome inside fits
        · exact absurd fits fits'

/-- The colexicographic rank of a list with its largest element first: `Σ_j C(c_j, j)` for the
element `c_j` with `j − 1` elements after it. -/
def colexRank : List ℕ → ℕ
  | [] => 0
  | c :: rest => c.choose (rest.length + 1) + colexRank rest

theorem unrankDown_congr {B : ℕ → ℕ → ℕ} :
    ∀ {k rank upper : ℕ}, (∀ c < upper, ∀ i ≤ k, B c i = c.choose i) →
      unrankDown B k rank upper = unrankDown Nat.choose k rank upper
  | 0, _, _, _ => rfl
  | k + 1, rank, upper, same => by
      rw [unrankDown, unrankDown, scan_congr fun c small => same c small (k + 1) le_rfl]
      cases found : scan Nat.choose (k + 1) rank upper with
      | none => rfl
      | some c =>
          have below := (scan_some found).1
          simp only [Option.bind_some, same c below (k + 1) le_rfl]
          rw [unrankDown_congr fun c' small i low =>
            same c' (small.trans below) i (Nat.le_succ_of_le low)]

theorem unrankDown_some : ∀ {k rank upper : ℕ} {l : List ℕ},
    unrankDown Nat.choose k rank upper = some l →
      l.length = k ∧ (∀ x ∈ l, x < upper) ∧ l.Pairwise (· > ·) ∧ colexRank l = rank
  | 0, rank, upper, l, found => by
      simp only [unrankDown] at found
      split_ifs at found with zero
      cases found
      simp [colexRank, zero]
  | k + 1, rank, upper, l, found => by
      simp only [unrankDown, Option.bind_eq_some_iff, Option.map_eq_some_iff] at found
      obtain ⟨c, scanned, rest, unranked, rfl⟩ := found
      obtain ⟨below, fits, -⟩ := scan_some scanned
      obtain ⟨length, bounded, ordered, ranked⟩ := unrankDown_some unranked
      refine ⟨by simp [length], fun x member => ?_, List.pairwise_cons.mpr
        ⟨fun x member => bounded x member, ordered⟩, ?_⟩
      · rcases List.mem_cons.mp member with rfl | member
        · exact below
        · exact (bounded x member).trans below
      · simp only [colexRank, length, ranked]
        omega

theorem unrankDown_isSome : ∀ {k rank upper : ℕ}, rank < upper.choose k →
    (unrankDown Nat.choose k rank upper).isSome
  | 0, rank, upper, below => by
      rw [Nat.choose_zero_right, Nat.lt_one_iff] at below
      simp [unrankDown, below]
  | k + 1, rank, upper, below => by
      have room : k < upper := by
        by_contra small
        rw [Nat.choose_eq_zero_of_lt (by omega)] at below
        omega
      obtain ⟨c, scanned⟩ := Option.isSome_iff_exists.mp
        (scan_isSome (i := k + 1) (rank := rank) room (by rw [Nat.choose_eq_zero_of_lt (by omega)]; omega))
      obtain ⟨inside, fits, above⟩ := scan_some scanned
      have next : rank < (c + 1).choose (k + 1) := by
        rcases Nat.lt_or_ge (c + 1) upper with smaller | edge
        · exact above (c + 1) (Nat.lt_succ_self c) smaller
        · rwa [show c + 1 = upper by omega]
      rw [Nat.choose_succ_succ'] at next
      obtain ⟨rest, unranked⟩ := Option.isSome_iff_exists.mp
        (unrankDown_isSome (k := k) (rank := rank - c.choose (k + 1)) (upper := c) (by omega))
      simp [unrankDown, scanned, unranked]

theorem unrank_congr {B : ℕ → ℕ → ℕ} {n k rank : ℕ} (same : ∀ c < n, ∀ i ≤ k, B c i = c.choose i) :
    unrank B n k rank = unrank Nat.choose n k rank := by
  rw [unrank, unrank, unrankDown_congr same]

/-- An unrank is a strictly increasing list of `k` elements below `n`, with the rank as its
colexicographic rank. -/
theorem unrank_some {n k rank : ℕ} {l : List ℕ} (found : unrank Nat.choose n k rank = some l) :
    l.length = k ∧ (∀ x ∈ l, x < n) ∧ l.Pairwise (· < ·) ∧ colexRank l.reverse = rank := by
  simp only [unrank, Option.map_eq_some_iff] at found
  obtain ⟨down, unranked, rfl⟩ := found
  obtain ⟨length, bounded, ordered, ranked⟩ := unrankDown_some unranked
  refine ⟨by rw [List.length_reverse, length], fun x member => bounded x (List.mem_reverse.mp member),
    List.pairwise_reverse.mpr ordered, by rw [List.reverse_reverse, ranked]⟩

/-- Each rank below `C(n, k)` has an unrank. -/
theorem unrank_isSome {n k rank : ℕ} (below : rank < n.choose k) :
    (unrank Nat.choose n k rank).isSome := by
  rw [unrank, Option.isSome_map]
  exact unrankDown_isSome below

/-- The rank is a function of the unrank. -/
theorem unrank_injective {n k rank rank' : ℕ} {l : List ℕ}
    (left : unrank Nat.choose n k rank = some l) (right : unrank Nat.choose n k rank' = some l) :
    rank = rank' :=
  (unrank_some left).2.2.2.symm.trans (unrank_some right).2.2.2

theorem mem_free {d x : ℕ} {taken : List ℕ} : x ∈ free d taken ↔ x < d ∧ x ∉ taken := by
  simp [free]

theorem free_ordered (d : ℕ) (taken : List ℕ) : (free d taken).Pairwise (· < ·) :=
  List.pairwise_lt_range.filter _

theorem length_free {d : ℕ} {taken : List ℕ} (distinct : taken.Nodup)
    (bounded : ∀ x ∈ taken, x < d) : (free d taken).length = d - taken.length := by
  have same : (free d taken).toFinset = Finset.range d \ taken.toFinset := by
    ext x
    simp [mem_free]
  rw [← List.toFinset_card_of_nodup (free_ordered d taken).nodup, same,
    Finset.card_sdiff_of_subset (fun x member => Finset.mem_range.mpr
      (bounded x (List.mem_toFinset.mp member))), Finset.card_range,
    List.toFinset_card_of_nodup distinct]

theorem getD_of_lt {l : List ℕ} {s : ℕ} (small : s < l.length) : l.getD s 0 = l[s] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem small, Option.getD_some]

theorem getD_free_mem {d s : ℕ} {taken : List ℕ} (small : s < (free d taken).length) :
    (free d taken).getD s 0 ∈ free d taken := by
  rw [getD_of_lt small]
  exact List.getElem_mem small

theorem getD_free_lt {d s s' : ℕ} {taken : List ℕ} (low : s < s')
    (small : s' < (free d taken).length) :
    (free d taken).getD s 0 < (free d taken).getD s' 0 := by
  rw [getD_of_lt (low.trans small), getD_of_lt small]
  exact List.pairwise_iff_getElem.mp (free_ordered d taken) s s' (low.trans small) small low

theorem complement_some {d : ℕ} {taken slots l : List ℕ}
    (found : complement d taken slots = some l) :
    l = slots.map (fun s => (free d taken).getD s 0) ∧ ∀ s ∈ slots, s < (free d taken).length := by
  unfold complement at found
  split_ifs at found with fits
  cases found
  exact ⟨rfl, fun s member => by simpa using List.all_eq_true.mp fits s member⟩

/-- The slots below the length of `free d taken` have a complement. -/
theorem complement_isSome {d : ℕ} {taken slots : List ℕ}
    (fits : ∀ s ∈ slots, s < (free d taken).length) : (complement d taken slots).isSome := by
  unfold complement
  rw [ite_eq_left (List.all_eq_true.mpr fun s member => by simpa using fits s member)]
  rfl

/-- A complement of strictly increasing slots is a strictly increasing list of elements below
`d` outside `taken`. -/
theorem complement_ordered {d : ℕ} {taken slots l : List ℕ}
    (found : complement d taken slots = some l) (ordered : slots.Pairwise (· < ·)) :
    l.Pairwise (· < ·) ∧ ∀ x ∈ l, x < d ∧ x ∉ taken := by
  obtain ⟨rfl, fits⟩ := complement_some found
  refine ⟨List.pairwise_map.mpr (ordered.imp_of_mem fun {s s'} _ member' low =>
    getD_free_lt low (fits s' member')), fun x member => ?_⟩
  obtain ⟨s, inside, rfl⟩ := List.mem_map.mp member
  exact mem_free.mp (getD_free_mem (fits s inside))

/-- Different slots have different complements. -/
theorem complement_injective {d : ℕ} {taken : List ℕ} :
    ∀ {slots slots' l : List ℕ}, complement d taken slots = some l →
      complement d taken slots' = some l → slots = slots'
  | [], [], _, _, _ => rfl
  | [], _ :: _, l, left, right => by
      obtain ⟨rfl, -⟩ := complement_some left
      obtain ⟨same, -⟩ := complement_some right
      simp at same
  | _ :: _, [], l, left, right => by
      obtain ⟨rfl, -⟩ := complement_some left
      obtain ⟨same, -⟩ := complement_some right
      simp at same
  | s :: rest, s' :: rest', l, left, right => by
      obtain ⟨rfl, fits⟩ := complement_some left
      obtain ⟨same, fits'⟩ := complement_some right
      simp only [List.map_cons, List.cons.injEq] at same
      have small := fits s List.mem_cons_self
      have small' := fits' s' List.mem_cons_self
      rw [getD_of_lt small, getD_of_lt small'] at same
      have head := (List.Nodup.getElem_inj_iff (free_ordered d taken).nodup).mp same.1
      have tail := complement_injective (slots := rest) (slots' := rest')
        (l := rest.map fun s => (free d taken).getD s 0)
        (by unfold complement; rw [ite_eq_left (List.all_eq_true.mpr fun x member => by
          simpa using fits x (List.mem_cons_of_mem _ member))])
        (by unfold complement; rw [ite_eq_left (List.all_eq_true.mpr fun x member => by
          simpa using fits' x (List.mem_cons_of_mem _ member)), same.2])
      rw [head, tail]

end NightstreamFPrime.Spec.Compression.Akita.Replay
