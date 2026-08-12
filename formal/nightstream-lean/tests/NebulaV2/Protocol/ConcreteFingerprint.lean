import Nightstream.Assurance.NebulaV2

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2ConcreteFingerprint

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Assurance.NebulaV2.FingerprintSecurity
open Nightstream.Protocol.NebulaV2.Fingerprint

example : Fintype.card ChallengeField =
    goldilocksModulus * goldilocksModulus := by
  simpa [goldilocksModulus,
    Nightstream.Implementation.R1CS.goldilocksP] using
    challengeField_cardinality

example :
    maxSegmentFactors ^ 2 * planningLoss * 2 ^ 186 ≤
      goldilocksModulus ^ 4 :=
  planning_fingerprint_bits_at_least_186

example :
    ¬ maxSegmentFactors ^ 2 * planningLoss * 2 ^ 187 ≤
      goldilocksModulus ^ 4 :=
  planning_fingerprint_bits_not_187

end Nightstream.Tests.NebulaV2ConcreteFingerprint
