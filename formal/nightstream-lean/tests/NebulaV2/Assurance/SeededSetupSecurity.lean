import Nightstream.Assurance.NebulaV2.SeededSetupSecurity

set_option autoImplicit false

namespace tests.NebulaV2SeededSetupSecurity

open Nightstream.Assurance.NebulaV2.SeededSetupSecurity

example :
    (Finset.univ.filter fun role :
      Nightstream.Implementation.NebulaV2.SeedSchedule.Role =>
        role.rows = 18).card = 3 :=
  exact_role_partition.1

example
    {manifest :
      Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest}
    (assumption : HybridAssumption manifest) :
    assumption.totalSeededAdvantage < dyadic postUnionBits :=
  assumption.total_lt_post_union

end tests.NebulaV2SeededSetupSecurity
