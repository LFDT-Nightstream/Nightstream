import DirectCcsFPrime.Core.Construction2DirectFPrime

/-!
Typed interface for the direct CCS Construction-2 F' image.

Spec: `specs/Core/Construction2DirectFPrime.spec.md`
-/

namespace DirectCcsFPrime

namespace Construction2DirectFPrimeInterface

abbrev PublicImage :=
  Construction2DirectFPrime.PublicImage

abbrev WellFormed :=
  @Construction2DirectFPrime.WellFormed

abbrev Transition :=
  @Construction2DirectFPrime.Transition

abbrev VerifyLatestStep :=
  @Construction2DirectFPrime.VerifyLatestStep

abbrev latest_step_sound :=
  @Construction2DirectFPrime.latest_step_sound

abbrev terminal_direct_fprime_reaches_final :=
  @Construction2DirectFPrime.terminal_direct_fprime_reaches_final

abbrev transition_next_step :=
  @Construction2DirectFPrime.transition_next_step

abbrev transition_preserves_vkDigest :=
  @Construction2DirectFPrime.transition_preserves_vkDigest

abbrev transition_preserves_initialBoundary :=
  @Construction2DirectFPrime.transition_preserves_initialBoundary

abbrev transition_pc_fixed :=
  @Construction2DirectFPrime.transition_pc_fixed

abbrev reachable_step_counter :=
  @Construction2DirectFPrime.reachable_step_counter

abbrev reachable_preserves_vkDigest :=
  @Construction2DirectFPrime.reachable_preserves_vkDigest

abbrev reachable_preserves_initialBoundary :=
  @Construction2DirectFPrime.reachable_preserves_initialBoundary

abbrev reachable_wellFormed_of_initial :=
  @Construction2DirectFPrime.reachable_wellFormed_of_initial

abbrev terminal_direct_fprime_public_image_invariants :=
  @Construction2DirectFPrime.terminal_direct_fprime_public_image_invariants

end Construction2DirectFPrimeInterface

end DirectCcsFPrime
