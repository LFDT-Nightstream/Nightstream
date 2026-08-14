import Nightstream.Assurance.Nebula

set_option autoImplicit false

namespace Nightstream.Tests.NebulaConcreteFingerprint

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Protocol.Nebula.Fingerprint

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

end Nightstream.Tests.NebulaConcreteFingerprint
