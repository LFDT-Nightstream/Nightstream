import Nightstream.Implementation.Nebula.Commitment.Compact.TokenRows

set_option autoImplicit false

namespace tests.NebulaCompactTokenRows

open Nightstream.Implementation.Nebula.CompactTokenRows
open Nightstream.Protocol.Nebula.CompactCommit

example (manifest : Nightstream.Implementation.Nebula.SeedSchedule.Manifest)
    (role : Role) (layout : Layout) :
    (rows manifest role layout).length = 134082 :=
  rows_length_exact manifest role layout

example (manifest : Nightstream.Implementation.Nebula.SeedSchedule.Manifest) :
    (key manifest).profile = manifest.profile :=
  key_profile manifest

example (manifest : Nightstream.Implementation.Nebula.SeedSchedule.Manifest) :
    Nightstream.Protocol.Nebula.ProductionProfileCandidates.SupportedIdentity
      (key manifest).profile := by
  simpa using manifest.profileSupported

example (manifest : Nightstream.Implementation.Nebula.SeedSchedule.Manifest) :
    (key manifest).plan = manifest.plan := key_plan manifest

end tests.NebulaCompactTokenRows
