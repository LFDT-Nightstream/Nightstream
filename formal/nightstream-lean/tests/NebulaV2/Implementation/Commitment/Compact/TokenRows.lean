import Nightstream.Implementation.NebulaV2.Commitment.Compact.TokenRows

set_option autoImplicit false

namespace tests.NebulaV2CompactTokenRows

open Nightstream.Implementation.NebulaV2.CompactTokenRows
open Nightstream.Protocol.NebulaV2.CompactCommit

example (manifest : Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest)
    (role : Role) (layout : Layout) :
    (rows manifest role layout).length = 134082 :=
  rows_length_exact manifest role layout

example (manifest : Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest) :
    (key manifest).profile = manifest.profile :=
  key_profile manifest

example (manifest : Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest) :
    Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.SupportedIdentity
      (key manifest).profile := by
  simpa using manifest.profileSupported

example (manifest : Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest) :
    (key manifest).plan = manifest.plan := key_plan manifest

end tests.NebulaV2CompactTokenRows
