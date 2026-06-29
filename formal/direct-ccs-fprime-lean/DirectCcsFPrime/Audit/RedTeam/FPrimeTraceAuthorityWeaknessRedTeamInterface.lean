import DirectCcsFPrime.Audit.RedTeam.FPrimeTraceAuthorityWeaknessRedTeam

/-!
Typed interface for trace-authority weakness red-team checks.

Spec: `specs/Audit/RedTeam/FPrimeTraceAuthorityWeaknessRedTeam.spec.md`
-/

namespace DirectCcsFPrime

namespace FPrimeTraceAuthorityWeaknessRedTeamInterface

abbrev BitImage :=
  FPrimeTraceAuthorityWeaknessRedTeam.BitImage

abbrev AnyTransition :=
  FPrimeTraceAuthorityWeaknessRedTeam.AnyTransition

abbrev VerifyStepOneAnyImage :=
  FPrimeTraceAuthorityWeaknessRedTeam.VerifyStepOneAnyImage

abbrev trace_soundness_does_not_imply_same_proof_functionality :=
  FPrimeTraceAuthorityWeaknessRedTeam.trace_soundness_does_not_imply_same_proof_functionality

abbrev universal_transition_authorizes_any_one_step_image :=
  FPrimeTraceAuthorityWeaknessRedTeam.universal_transition_authorizes_any_one_step_image

abbrev ConstantParentHash :=
  FPrimeTraceAuthorityWeaknessRedTeam.ConstantParentHash

abbrev ToyParentHashBinding :=
  FPrimeTraceAuthorityWeaknessRedTeam.ToyParentHashBinding

abbrev constant_parent_hash_cannot_bind_parent_handles :=
  FPrimeTraceAuthorityWeaknessRedTeam.constant_parent_hash_cannot_bind_parent_handles

end FPrimeTraceAuthorityWeaknessRedTeamInterface

end DirectCcsFPrime
