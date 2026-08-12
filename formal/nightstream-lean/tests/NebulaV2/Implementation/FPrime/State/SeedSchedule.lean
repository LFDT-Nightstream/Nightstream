import Nightstream.Implementation.NebulaV2.FPrime.State.SeedSchedule

set_option autoImplicit false

namespace tests.NebulaV2SeedSchedule

open Nightstream.Implementation.NebulaV2.SeedSchedule

variable (geometry : Geometry)

theorem fixed_role_geometry_is_exact :
    Role.bundleFull.rows = 18 ∧
      Role.bundleOperations.rows = 18 ∧
      Role.bundleSnapshot.rows = 18 ∧
      Role.tokenPrimaryOperations.rows = 2 ∧
      Role.tokenPrimaryMemory.rows = 2 ∧
      Role.tokenShortOperations.rows = 1 ∧
      Role.tokenShortMemory.rows = 1 ∧
      Role.tokenPrimaryOperations.columns geometry = 738 ∧
      Role.tokenPrimaryMemory.columns geometry = 738 ∧
      Role.tokenShortOperations.columns geometry = 82 ∧
      Role.tokenShortMemory.columns geometry = 82 :=
  exact_fixed_role_geometry geometry

/-- Distinct role names alone do not make two seeds independent. The exact
manifest carries seed injectivity, and the security layer must separately
price ChaCha8 pseudorandomness. -/
theorem role_names_without_a_manifest_do_not_fix_seeds :
    ∃ seed : Role → Nat,
      seed .tokenPrimaryOperations = seed .tokenPrimaryMemory :=
  ⟨fun _ => 0, rfl⟩

end tests.NebulaV2SeedSchedule
