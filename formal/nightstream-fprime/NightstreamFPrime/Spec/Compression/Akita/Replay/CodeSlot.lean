import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerPool
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerSigns
import NightstreamFPrime.Spec.Compression.Akita.Extract.Decoders

/-! The slot of an answer of the oracle with the words of the fold challenge sampler of the
pinned Akita code (revision `d50f700`, `crates/akita-challenges/src/sampler/stream.rs`; Akita,
ePrint 2026/1983, revision of 2026-11-23, Appendix E.1, "Representatives"), and the fold
decoders of the code on it.

- `Kept G` is the set of the words that `slotDecode G` accepts. `Fill G` is `Kept G` with one
  more value when `Kept G` is empty, so it is never empty. `CodeSlot E` has an element of `E`, a
  word of 131 bits, and an element of `Fill` for each filtered family of the code,
  `(64, 31, 11)` and `(128, 31, 0)`. It is a nonempty finite type, and `FieldSlot (CodeSlot E) E`
  holds.
- `CodeSlot.open G fits` is `slotDecode G` of the low `K` bits of the word, for a group without a
  filter. Each slot decodes (`open_isSome`), its challenges are the challenges of `slotDecode G`
  (`open_iff`) and are in the family (`open_mem`), it has `SignFair` (`signFair_open`), and its
  slot probability is at most `2^-K` (`slotChance_open`), and at most `1/q²` for `K ≥ 128`.
- `CodeSlot.kept64` and `CodeSlot.kept128` are `slotDecode` of the element of `Fill` of a filtered
  family. Their challenges are the members of the family that pass the filter of the code
  (`kept64_mem`, `kept64_surjective`). The slot probability is at most `1/|Kept G|`, and with
  `Key20.Kept` at most `2^-128` and `1/q²`, and each slot decodes.
- `slotDecode_mk`: `slotDecode` does not read the fields `coordinates` and `ring` of a group. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle (mass)
open NightstreamFPrime.Spec.Compression.Akita.Extract (slotChance FieldSlot)

attribute [local instance low] Classical.propDecidable

/-- `slotDecode` does not read the fields `coordinates` and `ring` of a group. -/
theorem slotDecode_mk (coordinates ring coordinates' ring' degree pm1 pm2 : ℕ) (filtered : Bool)
    [NeZero degree] :
    slotDecode ⟨coordinates, ring, degree, pm1, pm2, filtered⟩ =
      slotDecode ⟨coordinates', ring', degree, pm1, pm2, filtered⟩ :=
  rfl

/-- The filtered family `(64, 31, 11)` of the code. -/
abbrev filtered64 : FoldGroup := ⟨0, 64, 64, 31, 11, true⟩

/-- The filtered family `(128, 31, 0)` of the code. -/
abbrev filtered128 : FoldGroup := ⟨0, 128, 128, 31, 0, true⟩

/-- The words of a group that the decoder of the code accepts. -/
def Kept (G : FoldGroup) [NeZero G.degree] : Type :=
  {w : Fin (2 ^ G.candidateBits) // (slotDecode G w).isSome}

/-- An accepted word of `G`, or one more value when `G` has no accepted word. -/
def Fill (G : FoldGroup) [NeZero G.degree] : Type := Kept G ⊕ {_u : Unit // IsEmpty (Kept G)}

section Instances

variable (G : FoldGroup) [NeZero G.degree]

noncomputable instance Kept.fintype : Fintype (Kept G) :=
  inferInstanceAs (Fintype {w : Fin (2 ^ G.candidateBits) // (slotDecode G w).isSome})

instance Kept.decidableEq : DecidableEq (Kept G) :=
  inferInstanceAs (DecidableEq {w : Fin (2 ^ G.candidateBits) // (slotDecode G w).isSome})

noncomputable instance Fill.fintype : Fintype (Fill G) :=
  inferInstanceAs (Fintype (Kept G ⊕ {_u : Unit // IsEmpty (Kept G)}))

instance Fill.decidableEq : DecidableEq (Fill G) :=
  inferInstanceAs (DecidableEq (Kept G ⊕ {_u : Unit // IsEmpty (Kept G)}))

instance Fill.nonempty : Nonempty (Fill G) := by
  rcases isEmpty_or_nonempty (Kept G) with empty | ⟨⟨w⟩⟩
  · exact ⟨Sum.inr ⟨(), empty⟩⟩
  · exact ⟨Sum.inl w⟩

end Instances

/-- One slot of an answer: an element of `E`, a word of 131 bits, and an element of `Fill` for
each filtered family of the code. -/
abbrev CodeSlot (E : Type) := E × Fin (2 ^ 131) × Fill filtered64 × Fill filtered128

/-- `CodeSlot E` is nonempty with no premise on the filters. -/
instance CodeSlot.nonempty (E : Type) [Nonempty E] : Nonempty (CodeSlot E) := inferInstance

/-- The element of `E` of a uniform slot is uniform. -/
instance CodeSlot.fieldSlot (E : Type) [Fintype E] : FieldSlot (CodeSlot E) E :=
  ⟨⟨Prod.fst, Nat.card (Fin (2 ^ 131) × Fill filtered64 × Fill filtered128), fun e =>
    Nat.card_congr
      { toFun := fun s => s.1.2
        invFun := fun rest => ⟨(e, rest), rfl⟩
        left_inv := fun ⟨⟨_, _⟩, same⟩ => by subst same; rfl
        right_inv := fun _ => rfl }⟩⟩

/-- The word of 131 bits as its high bits and its low `K` bits. -/
def splitWord {K : ℕ} (fits : K ≤ 131) : Fin (2 ^ 131) ≃ Fin (2 ^ (131 - K)) × Fin (2 ^ K) :=
  (finCongr (by rw [← Nat.pow_add, Nat.sub_add_cancel fits])).trans finProdFinEquiv.symm

/-- The low part of `splitWord` is the word modulo `2^K`. -/
theorem splitWord_snd {K : ℕ} (fits : K ≤ 131) (w : Fin (2 ^ 131)) :
    ((splitWord fits w).2 : ℕ) = w % 2 ^ K := by
  simp [splitWord]

/-- The challenge of an element of `Fill G`: `slotDecode G` of an accepted word, and none for the
extra value. -/
noncomputable def Fill.decode (G : FoldGroup) [NeZero G.degree] : Fill G → Option (Rq G.degree)
  | Sum.inl w => slotDecode G w.1
  | Sum.inr _ => none

namespace CodeSlot

variable {E : Type}

/-- The decoder of the code of a group on the low `K` bits of the word of a slot. -/
noncomputable def «open» (G : FoldGroup) [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    (s : CodeSlot E) : Option (Rq G.degree) :=
  slotDecode G (splitWord fits s.2.1).2

/-- `slotDecode` of the element of `Fill` of the family `(64, 31, 11)` of a slot. -/
noncomputable def kept64 (s : CodeSlot E) : Option (Rq 64) := Fill.decode filtered64 s.2.2.1

/-- `slotDecode` of the element of `Fill` of the family `(128, 31, 0)` of a slot. -/
noncomputable def kept128 (s : CodeSlot E) : Option (Rq 128) := Fill.decode filtered128 s.2.2.2

end CodeSlot

section Laws

/-- A decoder after a map whose fibres have one size has at most the slot probability of the
decoder. -/
theorem slotChance_comp_le {S T X : Type} [Fintype S] [Nonempty S] [Fintype T] [Nonempty T]
    (decode : T → Option X) (read : S → T) {size : ℕ}
    (fibres : ∀ t, Nat.card {s // read s = t} = size) :
    slotChance (fun s => decode (read s)) ≤ slotChance decode := by
  refine Finset.sup'_le _ _ fun s _ => ?_
  split
  · next decoded =>
    obtain ⟨value, found⟩ := Option.isSome_iff_exists.mp decoded
    have : Nonempty {other // read other = read s} := ⟨⟨s, rfl⟩⟩
    have positive : 0 < size := (fibres (read s)) ▸ Nat.card_pos
    calc mass {other | decode (read other) = decode (read s)}
        = mass {t | decode t = some value} := by
          rw [← Extract.mass_read read positive fibres,
            show decode (read s) = some value from found]
          rfl
      _ ≤ slotChance decode := Extract.mass_slot_le decode value
  · exact Extract.slotChance_nonnegative decode

variable {E : Type}

/-- Each value of the low `K` bits has one count of slots. -/
theorem card_open_fibre {K : ℕ} (fits : K ≤ 131) (t : Fin (2 ^ K)) :
    Nat.card {s : CodeSlot E // (splitWord fits s.2.1).2 = t} =
      Nat.card (E × Fin (2 ^ (131 - K)) × Fill filtered64 × Fill filtered128) :=
  Nat.card_congr
    { toFun := fun s => (s.1.1, (splitWord fits s.1.2.1).1, s.1.2.2)
      invFun := fun x => ⟨(x.1, (splitWord fits).symm (x.2.1, t), x.2.2), by simp⟩
      left_inv := fun ⟨⟨_, w, _⟩, same⟩ => by subst same; simp
      right_inv := fun _ => by simp }

/-- Each element of `Fill (64, 31, 11)` has one count of slots. -/
theorem card_kept64_fibre (f : Fill filtered64) :
    Nat.card {s : CodeSlot E // s.2.2.1 = f} = Nat.card (E × Fin (2 ^ 131) × Fill filtered128) :=
  Nat.card_congr
    { toFun := fun s => (s.1.1, s.1.2.1, s.1.2.2.2)
      invFun := fun x => ⟨(x.1, x.2.1, f, x.2.2), rfl⟩
      left_inv := fun ⟨⟨_, _, _, _⟩, same⟩ => by subst same; rfl
      right_inv := fun _ => rfl }

/-- Each element of `Fill (128, 31, 0)` has one count of slots. -/
theorem card_kept128_fibre (f : Fill filtered128) :
    Nat.card {s : CodeSlot E // s.2.2.2 = f} = Nat.card (E × Fin (2 ^ 131) × Fill filtered64) :=
  Nat.card_congr
    { toFun := fun s => (s.1.1, s.1.2.1, s.1.2.2.1)
      invFun := fun x => ⟨(x.1, x.2.1, x.2.2, f), rfl⟩
      left_inv := fun ⟨⟨_, _, _, _⟩, same⟩ => by subst same; rfl
      right_inv := fun _ => rfl }

/-- `1/2^128 ≤ 1/q²`. -/
theorem pow_le_field : 1 / (2 : ℝ) ^ 128 ≤ 1 / (fieldModulus : ℝ) ^ 2 :=
  one_div_le_one_div_of_le (pow_pos (Nat.cast_pos.mpr (by decide)) 2)
    (by exact_mod_cast (by decide : fieldModulus ^ 2 ≤ 2 ^ 128))

/-- Without a filter each slot decodes. -/
theorem open_isSome {G : FoldGroup} [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    (unfiltered : G.filtered = false) (positive : 0 < G.rankSpace) (s : CodeSlot E) :
    (CodeSlot.open G fits s).isSome :=
  slotDecode_isSome unfiltered positive _

/-- The challenges of `CodeSlot.open G` are the challenges of `slotDecode G`. -/
theorem open_iff [Nonempty E] {G : FoldGroup} [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    {x : Rq G.degree} :
    (∃ s : CodeSlot E, CodeSlot.open G fits s = some x) ↔ ∃ w, slotDecode G w = some x := by
  constructor
  · rintro ⟨s, found⟩
    exact ⟨_, found⟩
  · rintro ⟨w, found⟩
    refine ⟨(Classical.arbitrary E, (splitWord fits).symm (⟨0, Nat.two_pow_pos _⟩, w),
      Classical.arbitrary _, Classical.arbitrary _), ?_⟩
    simpa [CodeSlot.open] using found

/-- Each challenge of `CodeSlot.open G` is in the family, and at a filtered group it passes the
filter of the code. -/
theorem open_mem {G : FoldGroup} [NeZero G.degree] {fits : G.candidateBits ≤ 131}
    {s : CodeSlot E} {x : Rq G.degree} (decoded : CodeSlot.open G fits s = some x) :
    x ∈ SparseFamily G.degree G.pm1 G.pm2 ∧
      (G.filtered → TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x) :=
  slotDecode_mem decoded

/-- Without a filter `CodeSlot.open G` has `SignFair`: the flip of `signFair_slotDecode` on the low
`K` bits of the word. -/
theorem signFair_open {G : FoldGroup} [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    (unfiltered : G.filtered = false) : SignFair (CodeSlot.open (E := E) G fits) := by
  obtain ⟨flip, flips⟩ := signFair_slotDecode unfiltered
  refine ⟨fun ε => (Equiv.refl E).prodCongr (((splitWord fits).trans
    (((Equiv.refl _).prodCongr (flip ε)).trans (splitWord fits).symm)).prodCongr (Equiv.refl _)),
    fun ε s => ?_⟩
  simpa [CodeSlot.open] using flips ε (splitWord fits s.2.1).2

variable [Fintype E] [Nonempty E]

/-- The slot probability of `CodeSlot.open G` is at most `2^-K`. -/
theorem slotChance_open (G : FoldGroup) [NeZero G.degree] (fits : G.candidateBits ≤ 131) :
    slotChance (CodeSlot.open (E := E) G fits) ≤ 1 / 2 ^ G.candidateBits :=
  (slotChance_comp_le (slotDecode G) (fun s : CodeSlot E => (splitWord fits s.2.1).2)
    (card_open_fibre fits)).trans (slotChance_slotDecode G)

/-- For `K ≥ 128` the slot probability of `CodeSlot.open G` is at most `1/q²`. -/
theorem slotChance_open_field (G : FoldGroup) [NeZero G.degree] (fits : G.candidateBits ≤ 131)
    (wide : 128 ≤ G.candidateBits) :
    slotChance (CodeSlot.open (E := E) G fits) ≤ 1 / (fieldModulus : ℝ) ^ 2 :=
  (slotChance_comp_le (slotDecode G) (fun s : CodeSlot E => (splitWord fits s.2.1).2)
    (card_open_fibre fits)).trans (slotChance_le_field G wide)

end Laws

section Fill

variable {G : FoldGroup} [NeZero G.degree]

/-- When `G` has an accepted word, `Fill G` is `Kept G`. -/
theorem card_fill [Nonempty (Kept G)] : Nat.card (Fill G) = Nat.card (Kept G) := by
  have : IsEmpty {_u : Unit // IsEmpty (Kept G)} :=
    ⟨fun u => u.2.false (Classical.arbitrary _)⟩
  rw [show Nat.card (Fill G) = Nat.card (Kept G ⊕ {_u : Unit // IsEmpty (Kept G)}) from rfl,
    Nat.card_sum, Nat.card_of_isEmpty (α := {_u : Unit // IsEmpty (Kept G)}), Nat.add_zero]

/-- When `G` has an accepted word, each element of `Fill G` decodes. -/
theorem Fill.decode_isSome [Nonempty (Kept G)] (f : Fill G) : (Fill.decode G f).isSome := by
  rcases f with w | ⟨_, empty⟩
  · exact w.2
  · exact (empty.false (Classical.arbitrary _)).elim

/-- Each challenge of `Fill.decode G` is in the family, and at a filtered group it passes the
filter of the code. -/
theorem Fill.decode_mem {f : Fill G} {x : Rq G.degree} (decoded : Fill.decode G f = some x) :
    x ∈ SparseFamily G.degree G.pm1 G.pm2 ∧
      (G.filtered → TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x) := by
  rcases f with w | _
  · exact slotDecode_mem decoded
  · exact absurd decoded (by simp [Fill.decode])

/-- At a filtered group each member of the family that passes the filter of the code is a
challenge of `Fill.decode G`. -/
theorem Fill.decode_surjective (filtered : G.filtered = true) {x : Rq G.degree}
    (inside : x ∈ SparseFamily G.degree G.pm1 G.pm2)
    (kept : TrigTable.codeAccepts G.degree (FoldGroup.threshold G.degree) x) :
    ∃ f, Fill.decode G f = some x := by
  obtain ⟨w, found⟩ := slotDecode_surjective filtered inside kept
  exact ⟨Sum.inl ⟨w, by rw [found]; rfl⟩, found⟩

/-- The slot probability of `Fill.decode G` is at most `1/|Kept G|`. -/
theorem slotChance_fill : slotChance (Fill.decode G) ≤ 1 / Nat.card (Kept G) := by
  rcases isEmpty_or_nonempty (Kept G) with empty | nonempty
  · refine Finset.sup'_le _ _ fun f _ => ?_
    rcases f with w | _
    · exact (empty.false w).elim
    · simp [Fill.decode]
  · have bound := Extract.slotChance_le_of_injective (Fill.decode G) fun a b x left right => by
      rcases a with a | _
      · rcases b with b | _
        · exact congrArg Sum.inl (Subtype.ext (slotDecode_injective G a.1 b.1 x left right))
        · exact absurd right (by simp [Fill.decode])
      · exact absurd left (by simp [Fill.decode])
    rwa [← Nat.card_eq_fintype_card, card_fill] at bound

end Fill

section Kept

variable {E : Type}

/-- With `Key20.Kept` both filtered families of the code have an accepted word. -/
theorem kept_nonempty (kept : Key20.Kept) :
    Nonempty (Kept filtered64) ∧ Nonempty (Kept filtered128) :=
  ⟨Nat.card_pos_iff.mp ((Nat.two_pow_pos 128).trans_le (kept_slots kept 0 64).1) |>.1,
    Nat.card_pos_iff.mp ((Nat.two_pow_pos 128).trans_le (kept_slots kept 0 128).2) |>.1⟩

/-- Each challenge of `CodeSlot.kept64` is in the family `(64, 31, 11)` and passes the filter of
the code. -/
theorem kept64_mem {s : CodeSlot E} {x : Rq 64} (decoded : CodeSlot.kept64 s = some x) :
    x ∈ SparseFamily 64 31 11 ∧ TrigTable.codeAccepts 64 18 x :=
  ⟨(Fill.decode_mem decoded).1, (Fill.decode_mem decoded).2 rfl⟩

/-- Each challenge of `CodeSlot.kept128` is in the family `(128, 31, 0)` and passes the filter
of the code. -/
theorem kept128_mem {s : CodeSlot E} {x : Rq 128} (decoded : CodeSlot.kept128 s = some x) :
    x ∈ SparseFamily 128 31 0 ∧ TrigTable.codeAccepts 128 13 x :=
  ⟨(Fill.decode_mem decoded).1, (Fill.decode_mem decoded).2 rfl⟩

/-- Each member of the family `(64, 31, 11)` that passes the filter of the code is a challenge
of `CodeSlot.kept64`. -/
theorem kept64_surjective [Nonempty E] {x : Rq 64} (inside : x ∈ SparseFamily 64 31 11)
    (kept : TrigTable.codeAccepts 64 18 x) : ∃ s : CodeSlot E, CodeSlot.kept64 s = some x := by
  obtain ⟨f, found⟩ := Fill.decode_surjective (G := filtered64) rfl inside kept
  exact ⟨(Classical.arbitrary E, Classical.arbitrary _, f, Classical.arbitrary _), found⟩

/-- Each member of the family `(128, 31, 0)` that passes the filter of the code is a challenge
of `CodeSlot.kept128`. -/
theorem kept128_surjective [Nonempty E] {x : Rq 128} (inside : x ∈ SparseFamily 128 31 0)
    (kept : TrigTable.codeAccepts 128 13 x) : ∃ s : CodeSlot E, CodeSlot.kept128 s = some x := by
  obtain ⟨f, found⟩ := Fill.decode_surjective (G := filtered128) rfl inside kept
  exact ⟨(Classical.arbitrary E, Classical.arbitrary _, Classical.arbitrary _, f), found⟩

/-- With `Key20.Kept` each slot decodes by `CodeSlot.kept64`. -/
theorem kept64_isSome (kept : Key20.Kept) (s : CodeSlot E) : (CodeSlot.kept64 s).isSome :=
  have := (kept_nonempty kept).1
  Fill.decode_isSome _

/-- With `Key20.Kept` each slot decodes by `CodeSlot.kept128`. -/
theorem kept128_isSome (kept : Key20.Kept) (s : CodeSlot E) : (CodeSlot.kept128 s).isSome :=
  have := (kept_nonempty kept).2
  Fill.decode_isSome _

variable [Fintype E] [Nonempty E]

/-- The slot probability of `CodeSlot.kept64` is at most `1/|Kept (64, 31, 11)|`. -/
theorem slotChance_kept64 :
    slotChance (CodeSlot.kept64 (E := E)) ≤ 1 / Nat.card (Kept filtered64) :=
  (slotChance_comp_le (Fill.decode filtered64) (fun s : CodeSlot E => s.2.2.1)
    card_kept64_fibre).trans slotChance_fill

/-- The slot probability of `CodeSlot.kept128` is at most `1/|Kept (128, 31, 0)|`. -/
theorem slotChance_kept128 :
    slotChance (CodeSlot.kept128 (E := E)) ≤ 1 / Nat.card (Kept filtered128) :=
  (slotChance_comp_le (Fill.decode filtered128) (fun s : CodeSlot E => s.2.2.2)
    card_kept128_fibre).trans slotChance_fill

/-- With `Key20.Kept` the slot probability of `CodeSlot.kept64` is at most `2^-128`. -/
theorem slotChance_kept64_le (kept : Key20.Kept) :
    slotChance (CodeSlot.kept64 (E := E)) ≤ 1 / 2 ^ 128 :=
  slotChance_kept64.trans (one_div_le_one_div_of_le (by positivity)
    (by exact_mod_cast (kept_slots kept 0 64).1))

/-- With `Key20.Kept` the slot probability of `CodeSlot.kept128` is at most `2^-128`. -/
theorem slotChance_kept128_le (kept : Key20.Kept) :
    slotChance (CodeSlot.kept128 (E := E)) ≤ 1 / 2 ^ 128 :=
  slotChance_kept128.trans (one_div_le_one_div_of_le (by positivity)
    (by exact_mod_cast (kept_slots kept 0 128).2))

/-- With `Key20.Kept` the slot probability of `CodeSlot.kept64` is at most `1/q²`. -/
theorem slotChance_kept64_field (kept : Key20.Kept) :
    slotChance (CodeSlot.kept64 (E := E)) ≤ 1 / (fieldModulus : ℝ) ^ 2 :=
  (slotChance_kept64_le kept).trans pow_le_field

/-- With `Key20.Kept` the slot probability of `CodeSlot.kept128` is at most `1/q²`. -/
theorem slotChance_kept128_field (kept : Key20.Kept) :
    slotChance (CodeSlot.kept128 (E := E)) ≤ 1 / (fieldModulus : ℝ) ^ 2 :=
  (slotChance_kept128_le kept).trans pow_le_field

end Kept

end NightstreamFPrime.Spec.Compression.Akita.Replay
