import Mathlib.Data.Nat.Size
import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerDraw
import NightstreamFPrime.Spec.Compression.Akita.Setup

/-! The image of the decoder of one candidate of the pinned Akita code (`Replay/Sampler.lean`,
`crates/akita-challenges/src/sampler/stream.rs`: "On `[0, |S|)` this is a bijection onto the
shell `S`"), and the slots of an unfiltered group.

- `candidate_surjective`: the candidates below `2^w · rankSpace` give each member of
  `SparseFamily d pm1 pm2` (an injection between two sets of `C(d, pm1)·C(d − pm1, pm2)·2^w`
  elements, `card_sparseFamily`).
- `slotDecode_surjective`, `card_accepted`: at a filtered group the slots that decode are in
  bijection with the members of the family that pass the filter of the code, the set of
  `Key20.Kept`.
- `kept_slots`: with `Key20.Kept`, `2^128` slots or more of each filtered family of the code
  decode.
- `slotChance_le_field`: for `K ≥ 128` the slot probability of the decoder of the code is at
  most `1/q²`.
- `sample_unfiltered`: at an unfiltered group the challenge `j` is the decoded candidate `j` of
  the message. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open Sparse

theorem rankSpace_eq (G : FoldGroup) :
    G.rankSpace = G.degree.choose G.pm1 * (G.degree - G.pm1).choose G.pm2 := by
  rw [FoldGroup.rankSpace, FoldGroup.binomial_eq_choose, FoldGroup.binomial_eq_choose]

/-- The candidates below `2^w · rankSpace` give each member of the family. -/
theorem candidate_surjective {G : FoldGroup} [NeZero G.degree] {x : Rq G.degree}
    (inside : x ∈ SparseFamily G.degree G.pm1 G.pm2) :
    ∃ value < 2 ^ (G.pm1 + G.pm2) * G.rankSpace, ∃ c,
      candidate G Nat.choose value = some c ∧ ring G.degree c = x := by
  set n := 2 ^ (G.pm1 + G.pm2) * G.rankSpace with size
  have decodes (v : Fin n) : (candidate G Nat.choose v).isSome :=
    candidate_isSome ((Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
      (by rw [Nat.mul_comm]; exact v.isLt))
  let f (v : Fin n) : SparseFamily G.degree G.pm1 G.pm2 :=
    ⟨ring G.degree ((candidate G Nat.choose v).get (decodes v)), candidate_mem (Option.some_get (decodes v)).symm⟩
  have injective : Function.Injective f := fun a b same => Fin.ext
    (candidate_injective (Option.some_get (decodes a)).symm (Option.some_get (decodes b)).symm
      (congrArg Subtype.val same))
  have count : Nat.card (Fin n) = Nat.card (SparseFamily G.degree G.pm1 G.pm2) := by
    rw [Nat.card_fin, card_sparseFamily, size, rankSpace_eq]
    ring
  obtain ⟨v, image⟩ := ((Nat.bijective_iff_injective_and_card f).mpr ⟨injective, count⟩).2
    ⟨x, inside⟩
  exact ⟨v, v.isLt, _, (Option.some_get (decodes v)).symm, congrArg Subtype.val image⟩

theorem pow_le_candidates {G : FoldGroup} (filtered : G.filtered = true) :
    2 ^ (G.pm1 + G.pm2) * G.rankSpace ≤ 2 ^ G.candidateBits := by
  rw [FoldGroup.candidateBits_eq, FoldGroup.rankBits, filtered, ite_eq_left rfl,
    show (G.rankSpace - 1).size + (G.pm1 + G.pm2) = G.pm1 + G.pm2 + (G.rankSpace - 1).size by omega,
    Nat.pow_add 2 (G.pm1 + G.pm2) (G.rankSpace - 1).size]
  have below := Nat.lt_size_self (G.rankSpace - 1)
  exact Nat.mul_le_mul_left (2 ^ (G.pm1 + G.pm2))
    (show G.rankSpace ≤ 2 ^ (G.rankSpace - 1).size by omega)

/-- At a filtered group each member of the family that passes the filter of the code is the
challenge of a slot. -/
theorem slotDecode_surjective {G : FoldGroup} [NeZero G.degree] (filtered : G.filtered = true)
    {x : Rq G.degree} (inside : x ∈ SparseFamily G.degree G.pm1 G.pm2)
    (kept : TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x) :
    ∃ slot, slotDecode G slot = some x := by
  obtain ⟨value, small, c, found, rfl⟩ := candidate_surjective inside
  refine ⟨⟨value, small.trans_le (pow_le_candidates filtered)⟩, ?_⟩
  have accepted := (accepts_iff (candidate_valid found)).mpr kept
  simp [slotDecode, found, filtered, accepted]

/-- At a filtered group the slots that decode are as many as the members of the family that pass
the filter of the code. -/
theorem card_accepted {G : FoldGroup} [NeZero G.degree] (filtered : G.filtered = true) :
    Nat.card {slot : Fin (2 ^ G.candidateBits) // (slotDecode G slot).isSome} =
      Nat.card {x ∈ SparseFamily G.degree G.pm1 G.pm2 |
        TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x} := by
  have mem (slot : {slot : Fin (2 ^ G.candidateBits) // (slotDecode G slot).isSome}) :=
    slotDecode_mem (Option.some_get slot.2).symm
  refine Nat.card_congr (Equiv.ofBijective (fun slot => ⟨(slotDecode G slot.1).get slot.2,
    (mem slot).1, (mem slot).2 filtered⟩) ⟨fun a b same => ?_, fun x => ?_⟩)
  · exact Subtype.ext (slotDecode_injective G a.1 b.1 _ (Option.some_get a.2).symm
      ((Option.some_get b.2).symm.trans (congrArg some (congrArg Subtype.val same).symm)))
  · obtain ⟨slot, decoded⟩ := slotDecode_surjective filtered x.2.1 x.2.2
    exact ⟨⟨slot, by rw [decoded]; rfl⟩, Subtype.ext (by simp [decoded])⟩

/-- With `Key20.Kept`, `2^128` slots or more of the filtered families `(64, 31, 11)` and
`(128, 31, 0)` of the code decode. -/
theorem kept_slots (kept : Key20.Kept) (coordinates ring : ℕ) :
    2 ^ 128 ≤ Nat.card {slot // (slotDecode ⟨coordinates, ring, 64, 31, 11, true⟩ slot).isSome} ∧
      2 ^ 128 ≤ Nat.card {slot // (slotDecode ⟨coordinates, ring, 128, 31, 0, true⟩ slot).isSome} := by
  rw [card_accepted rfl, card_accepted rfl]
  exact kept

/-- For `K ≥ 128` the slot probability of the decoder of the code is at most `1/q²`. -/
theorem slotChance_le_field (G : FoldGroup) [NeZero G.degree] (wide : 128 ≤ G.candidateBits) :
    Extract.slotChance (slotDecode G) ≤ 1 / (fieldModulus : ℝ) ^ 2 := by
  refine (slotChance_slotDecode G).trans
    (one_div_le_one_div_of_le (pow_pos (Nat.cast_pos.mpr (by decide)) 2) ?_)
  have bound := (by decide : fieldModulus ^ 2 ≤ 2 ^ 128).trans
    (Nat.pow_le_pow_right (by decide) wide)
  exact_mod_cast bound

theorem getElem?_filterMap {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, (f a).isSome) → ∀ i : ℕ, (l.filterMap f)[i]? = l[i]?.bind f
  | [], _, i => by simp
  | a :: l, each, i => by
      obtain ⟨b, found⟩ := Option.isSome_iff_exists.mp (each a List.mem_cons_self)
      simp only [List.filterMap_cons, found]
      cases i with
      | zero => simp [found]
      | succ i =>
          simp only [List.getElem?_cons_succ]
          exact getElem?_filterMap (fun x member => each x (List.mem_cons_of_mem _ member)) i

/-- At an unfiltered group the challenge `j` is the decoded candidate `j` of the message. -/
theorem sample_unfiltered {G : FoldGroup} {bytes : ByteArray} {cs : List Sparse}
    (unfiltered : G.filtered = false) (positive : 0 < G.rankSpace)
    (sampled : sample G bytes = some cs) (j : ℕ) :
    cs[j]? = if j < G.coordinates then
      candidate G Nat.choose (readBits bytes (j * G.candidateBits) G.candidateBits) else none := by
  simp only [sample, unfiltered, Bool.false_eq_true, ite_false, candidate_table] at sampled
  split_ifs at sampled with size full
  cases sampled
  rw [getElem?_filterMap fun j _ => candidate_isSome (rank_lt unfiltered positive (readBits_lt _ _ _))]
  by_cases small : j < G.coordinates
  · rw [List.getElem?_range small, ite_eq_left small]
    rfl
  · rw [List.getElem?_eq_none (by simpa using small), ite_eq_right small]
    rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
