import DirectCcsFPrime.Core.FoldedFPrimeAuthority

/-!
Typed interface for proof-carrying folded F' prior authority.

Spec: `specs/Core/FoldedFPrimeAuthority.spec.md`
-/

namespace DirectCcsFPrime

namespace FoldedFPrimeAuthorityInterface

abbrev Authority :=
  @FoldedFPrimeAuthority.Authority

abbrev Accepts :=
  @FoldedFPrimeAuthority.Accepts

abbrev accepts_sound :=
  @FoldedFPrimeAuthority.accepts_sound

abbrev base :=
  @FoldedFPrimeAuthority.base

abbrev base_accepts :=
  @FoldedFPrimeAuthority.base_accepts

abbrev extend :=
  @FoldedFPrimeAuthority.extend

abbrev extend_accepts :=
  @FoldedFPrimeAuthority.extend_accepts

abbrev terminal_compression_reaches_final :=
  @FoldedFPrimeAuthority.terminal_compression_reaches_final

abbrev construction2_terminal_reaches_final :=
  @FoldedFPrimeAuthority.construction2_terminal_reaches_final

abbrev accepted_unreachable_is_not_sound_authority :=
  @FoldedFPrimeAuthority.accepted_unreachable_is_not_sound_authority

end FoldedFPrimeAuthorityInterface

end DirectCcsFPrime
