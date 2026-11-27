import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier

/-! The fold challenges of a transcript and the decoders of the soundness claims (Akita, ePrint
2026/1983, revision of 2026-11-23, Section 3.2 and (23)).

- `Tier.Decodes.challenge_mem`: a transcript with `Tier.Decodes` has each fold challenge in the
  family of its decoder. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

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

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
