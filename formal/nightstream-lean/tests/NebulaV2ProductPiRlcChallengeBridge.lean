import Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge

/-! Regression surface for the exact selector-to-PiRLC challenge bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPiRlcChallengeBridge

#check Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.challengeSymbol_range
#check Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse

example :
    Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.scalarCount *
        Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows.coefficientCount =
      810 := by
  decide

end tests.NebulaV2ProductPiRlcChallengeBridge
