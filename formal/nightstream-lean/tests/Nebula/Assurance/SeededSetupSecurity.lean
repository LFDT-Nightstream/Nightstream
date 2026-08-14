import Nightstream.Assurance.Nebula.SeededSetupSecurity

set_option autoImplicit false

namespace tests.NebulaSeededSetupSecurity

open Nightstream.Assurance.Nebula.SeededSetupSecurity

example :
    (Finset.univ.filter fun role :
      Nightstream.Implementation.Nebula.SeedSchedule.Role =>
        role.rows = 18).card = 3 :=
  exact_role_partition.1

example
    {manifest :
      Nightstream.Implementation.Nebula.SeedSchedule.Manifest}
    (assumption : HybridAssumption manifest) :
    assumption.totalSeededAdvantage < dyadic postUnionBits :=
  assumption.total_lt_post_union

end tests.NebulaSeededSetupSecurity
