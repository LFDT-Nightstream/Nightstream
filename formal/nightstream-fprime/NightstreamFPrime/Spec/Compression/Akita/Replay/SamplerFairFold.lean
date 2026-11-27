import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFairField
import NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot

/-! The fold challenge sampler of the pinned Akita code (revision `d50f700`,
`crates/akita-challenges/src/sampler/stream.rs`) is fair on the squeezed message of a group
(Akita, ePrint 2026/1983, revision of 2026-11-23, Section 3.2, Section 8.2 `[KECCAK]` and
Appendix E.1, "Representatives").

- `words_array`: the candidate words of a string are the windows `Bytes.digits` of its number.
- `sample_first`: a draw of a group reads the first `coordinates` accepted windows
  (`sample_pool` with a filter, the definition of `sample` without a filter).
- `card_sample_ring_le`: each list of challenges in the ring has at most
  `256^messageBytes / |Kept G|^coordinates` messages (`card_first_le`, `card_digits`).
  `card_sample_le` is this statement for a list of sparse challenges, and `card_sample_open_le`
  is the bound `256^messageBytes / 2^(K·coordinates)` of a group without a filter
  (`card_kept_open`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open Bytes

/-- The candidate words of a string are the windows of `K` bits of its number. -/
theorem words_array {G : FoldGroup} (u : Bytes G.messageBytes) :
    words G (array u) = digits G.candidates G.candidateBits u :=
  funext fun _ => Fin.ext (readBits_array u _ _)

/-- The candidates of a group are inside its message. -/
theorem digits_fit (G : FoldGroup) : G.candidates * G.candidateBits ≤ 8 * G.messageBytes := by
  unfold FoldGroup.messageBytes
  omega

/-- A draw of a group on a string reads the first `coordinates` accepted windows. -/
theorem sample_first {G : FoldGroup} [NeZero G.degree] {u : Bytes G.messageBytes}
    {cs : List Sparse} (sampled : sample G (array u) = some cs) :
    firstOf (slotDecode G) G.coordinates (digits G.candidates G.candidateBits u) =
      cs.map (Sparse.ring G.degree) := by
  cases filtered : G.filtered with
  | true =>
      rw [← words_array]
      exact (sample_pool filtered sampled).symm
  | false =>
      have same : G.candidates = G.coordinates := by simp [FoldGroup.candidates, filtered]
      rw [same]
      simp only [sample, filtered, Bool.false_eq_true, ite_false, candidate_table] at sampled
      split_ifs at sampled with size full
      cases sampled
      rw [firstOf, List.take_of_length_le ((List.length_filterMap_le _ _).trans (by simp)),
        List.map_filterMap, List.ofFn_eq_map, List.filterMap_map,
        ← List.map_coe_finRange_eq_range, List.filterMap_map]
      congr 1
      funext j
      have keep : ∀ found : Option Sparse, found.filter (fun _ => true) = found := fun found => by
        cases found <;> rfl
      simp [slotDecode, digits, readBits_array, filtered, keep]

/-- The fold draw is fair: each list of challenges in the ring has at most
`256^messageBytes / |Kept G|^coordinates` messages. -/
theorem card_sample_ring_le (G : FoldGroup) [NeZero G.degree] (t : List (Rq G.degree)) :
    Nat.card {u : Bytes G.messageBytes //
        (sample G (array u)).map (List.map (Sparse.ring G.degree)) = some t} *
      Nat.card (Kept G) ^ G.coordinates ≤ 256 ^ G.messageBytes := by
  have first (u : Bytes G.messageBytes)
      (read : (sample G (array u)).map (List.map (Sparse.ring G.degree)) = some t) :
      firstOf (slotDecode G) G.coordinates (digits G.candidates G.candidateBits u) = t ∧
        t.length = G.coordinates := by
    obtain ⟨cs, sampled, rfl⟩ := Option.map_eq_some_iff.mp read
    exact ⟨sample_first sampled, by rw [List.length_map]; exact (sample_mem sampled).1⟩
  rcases isEmpty_or_nonempty {u : Bytes G.messageBytes //
      (sample G (array u)).map (List.map (Sparse.ring G.degree)) = some t} with
    empty | ⟨⟨u₀, read₀⟩⟩
  · rw [Nat.card_of_isEmpty, Nat.zero_mul]
    exact Nat.zero_le _
  · have pools := card_first_le (B := G.candidates)
      (fun w : Fin (2 ^ G.candidateBits) => (slotDecode G w).isSome = true) (g := slotDecode G)
      (fun _ => Iff.rfl) (fun w w' accepted same => by
        obtain ⟨x, found⟩ := Option.isSome_iff_exists.mp accepted
        exact slotDecode_injective G w w' x found (same.symm.trans found)) t (first u₀ read₀).2
    rw [Nat.card_eq_fintype_card (α := Fin (2 ^ G.candidateBits)), Fintype.card_fin] at pools
    calc Nat.card {u : Bytes G.messageBytes //
            (sample G (array u)).map (List.map (Sparse.ring G.degree)) = some t} *
          Nat.card (Kept G) ^ G.coordinates
        ≤ Nat.card {u : Bytes G.messageBytes // firstOf (slotDecode G) G.coordinates
            (digits G.candidates G.candidateBits u) = t} * Nat.card (Kept G) ^ G.coordinates :=
          Nat.mul_le_mul_right _ (Nat.card_le_card_of_injective
            (fun u => ⟨u.1, (first u.1 u.2).1⟩) fun _ _ same => Subtype.ext (Subtype.mk.inj same))
      _ = 2 ^ (8 * G.messageBytes - G.candidates * G.candidateBits) *
            (Nat.card {p : Fin G.candidates → Fin (2 ^ G.candidateBits) //
              firstOf (slotDecode G) G.coordinates p = t} * Nat.card (Kept G) ^ G.coordinates) := by
          rw [card_preimage (digits G.candidates G.candidateBits) (card_digits (digits_fit G))
            fun p => firstOf (slotDecode G) G.coordinates p = t, Nat.mul_assoc]
      _ ≤ 2 ^ (8 * G.messageBytes - G.candidates * G.candidateBits) *
            (2 ^ G.candidateBits) ^ G.candidates := Nat.mul_le_mul_left _ pools
      _ = 256 ^ G.messageBytes := by
          rw [← pow_mul, ← pow_add, show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_mul,
            Nat.mul_comm G.candidateBits, Nat.sub_add_cancel (digits_fit G)]

/-- The fold draw is fair, for a list of sparse challenges. -/
theorem card_sample_le (G : FoldGroup) [NeZero G.degree] (cs : List Sparse) :
    Nat.card {u : Bytes G.messageBytes // sample G (array u) = some cs} *
      Nat.card (Kept G) ^ G.coordinates ≤ 256 ^ G.messageBytes :=
  (Nat.mul_le_mul_right _ (Nat.card_le_card_of_injective
    (fun u : {u : Bytes G.messageBytes // sample G (array u) = some cs} =>
      (⟨u.1, by rw [u.2]; rfl⟩ : {u : Bytes G.messageBytes //
        (sample G (array u)).map (List.map (Sparse.ring G.degree)) =
          some (cs.map (Sparse.ring G.degree))}))
    fun _ _ same => Subtype.ext (Subtype.mk.inj same))).trans (card_sample_ring_le G _)

/-- Without a filter each word of `K` bits is accepted. -/
theorem card_kept_open {G : FoldGroup} [NeZero G.degree] (unfiltered : G.filtered = false)
    (positive : 0 < G.rankSpace) : Nat.card (Kept G) = 2 ^ G.candidateBits :=
  (Nat.card_congr (Equiv.subtypeUnivEquiv fun w => slotDecode_isSome unfiltered positive w)).trans
    (Nat.card_eq_fintype_card.trans (Fintype.card_fin _))

/-- The fold draw of a group without a filter is fair: each list of challenges has at most
`256^messageBytes / 2^(K·coordinates)` messages. -/
theorem card_sample_open_le {G : FoldGroup} [NeZero G.degree] (unfiltered : G.filtered = false)
    (positive : 0 < G.rankSpace) (cs : List Sparse) :
    Nat.card {u : Bytes G.messageBytes // sample G (array u) = some cs} *
      2 ^ (G.candidateBits * G.coordinates) ≤ 256 ^ G.messageBytes := by
  have bound := card_sample_le G cs
  rwa [card_kept_open unfiltered positive, ← pow_mul] at bound

end NightstreamFPrime.Spec.Compression.Akita.Replay
