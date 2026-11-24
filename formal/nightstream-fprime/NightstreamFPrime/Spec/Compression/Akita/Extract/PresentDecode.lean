import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge

/-! The fold challenges of the code and the decoders of the soundness claims (Akita, ePrint
2026/1983, revision of 2026-11-23, Section 3.2 and (23); the sampler
`crates/akita-challenges/src/sampler/stream.rs` of the Akita code, `Replay/SamplerDraw.lean`).

- The claims read each fold challenge from one slot of the extension field `E`, and `|E| = q²` is
  below `2^K` for `K ≥ 128` (`card_lt_words`). The code reads it from a word of `K` bits, so no
  injection of the words into one slot of `E` exists (`not_injective_words`).
- `exists_not_decoded`: a decoder from a slot type misses a member of each set with more members
  than slots. `exists_word_missed`: for an injective decoder of words with more decoded words than
  slots, a word decodes to a member that the decoder from the slots never gives. `draw_misses` is
  `exists_not_decoded` for `draw`, and `draw_misses_field` for one slot of `E` and a family with
  `2^128` members or more that pass the filter.
- `Tier.Decodes.challenge_mem`: a transcript with `Tier.Decodes` has each fold challenge in the
  family of its decoder, so a member that the decoder never gives is never a fold challenge of
  such a transcript (`Tier.Decodes.challenge_ne`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

section Count

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]

/-- The extension field has fewer elements than the words of `K ≥ 128` bits. -/
theorem card_lt_words {K : ℕ} (wide : 128 ≤ K) : Fintype.card E < 2 ^ K := by
  rw [Key20.card_field]
  exact lt_of_lt_of_le (by decide : fieldModulus ^ 2 < 2 ^ 128)
    (Nat.pow_le_pow_right (by decide) wide)

/-- No map of the words of `K ≥ 128` bits into one slot of `E` is injective. -/
theorem not_injective_words {K : ℕ} (wide : 128 ≤ K) (into : Fin (2 ^ K) → E) :
    ¬ Function.Injective into := fun injective =>
  absurd (Fintype.card_le_of_injective into injective)
    (by rw [Fintype.card_fin]; exact Nat.not_le.mpr (card_lt_words wide))

end Count

section Missed

variable {Slot Ring : Type} [Fintype Slot]

/-- A decoder from `Slot` misses a member of each set with more members than `Slot`. -/
theorem exists_not_decoded (decode : Slot → Option Ring) {members : Set Ring}
    (large : Fintype.card Slot < Nat.card members) :
    ∃ c ∈ members, ∀ slot, decode slot ≠ some c := by
  by_contra reached
  have each : ∀ c ∈ members, ∃ slot, decode slot = some c := fun c inside => by
    by_contra none
    exact reached ⟨c, inside, fun slot found => none ⟨slot, found⟩⟩
  choose pick picked using each
  have injective : Function.Injective fun c : members => pick c.1 c.2 := fun a b same =>
    Subtype.ext (Option.some.inj ((picked a.1 a.2).symm.trans
      ((congrArg decode same).trans (picked b.1 b.2))))
  have bound := Nat.card_le_card_of_injective _ injective
  rw [Nat.card_eq_fintype_card (α := Slot)] at bound
  omega

/-- For a decoder `word` of words that gives each challenge for at most one word, with more
decoded words than slots, a word decodes to a challenge that the decoder from the slots never
gives. -/
theorem exists_word_missed {Word : Type} (word : Word → Option Ring)
    (injective : ∀ a b c, word a = some c → word b = some c → a = b)
    (decode : Slot → Option Ring)
    (large : Fintype.card Slot < Nat.card {w // (word w).isSome}) :
    ∃ w c, word w = some c ∧ ∀ slot, decode slot ≠ some c := by
  have same : Nat.card {w // (word w).isSome} = Nat.card {c | ∃ w, word w = some c} :=
    Nat.card_congr (Equiv.ofBijective
      (fun w => ⟨(word w.1).get w.2, w.1, (Option.some_get w.2).symm⟩)
      ⟨fun a b equal => Subtype.ext (injective a.1 b.1 _ (Option.some_get a.2).symm
          ((Option.some_get b.2).symm.trans (congrArg some (congrArg Subtype.val equal).symm))),
        fun c => by
          obtain ⟨w, found⟩ := c.2
          exact ⟨⟨w, by rw [found]; rfl⟩, Subtype.ext (by simp [found])⟩⟩)
  obtain ⟨c, ⟨w, found⟩, missed⟩ := exists_not_decoded decode (same ▸ large)
  exact ⟨w, c, found, missed⟩

variable {family : Set Ring} {keep : Ring → Prop}

/-- `draw` misses a member of the family that passes the filter when more members than slots
pass it. -/
theorem draw_misses (large : Fintype.card Slot < Nat.card {c ∈ family | keep c}) :
    ∃ c, c ∈ family ∧ keep c ∧ ∀ slot : Slot, draw family keep slot ≠ some c := by
  obtain ⟨c, ⟨inside, kept⟩, missed⟩ := exists_not_decoded (draw family keep) large
  exact ⟨c, inside, kept, missed⟩

/-- `draw` from one slot of `E` misses a member of a family with `2^128` members or more that
pass the filter. -/
theorem draw_misses_field {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
    (large : 2 ^ 128 ≤ Nat.card {c ∈ family | keep c}) :
    ∃ c, c ∈ family ∧ keep c ∧ ∀ slot : E, draw family keep slot ≠ some c :=
  draw_misses ((card_lt_words le_rfl).trans_le large)

end Missed

namespace Tier

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [∀ b, Zero (Ring b)] [Fintype Slot]
  {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source} {total : ℕ}
  {point : Prefix Wire (Option Coord → Slot) total}

/-- A transcript with `Tier.Decodes` has each fold challenge in the family of its decoder. -/
theorem Decodes.challenge_mem (decodes : T.Decodes point) (b : Block) :
    T.challenge point b ∈ T.family b := by
  obtain ⟨reply, answered, -, -⟩ := decodes.1
  exact ⟨_, decodes.challenge answered b⟩

/-- A member that the decoder of a coordinate never gives is not the fold challenge of a
transcript with `Tier.Decodes`. -/
theorem Decodes.challenge_ne (decodes : T.Decodes point) {b : Block} {c : Ring b}
    (missed : ∀ slot, T.decode b slot ≠ some c) : T.challenge point b ≠ c := fun same => by
  obtain ⟨slot, decoded⟩ := decodes.challenge_mem b
  exact missed slot (decoded.trans (congrArg some same))

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
