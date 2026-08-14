import Nightstream.Implementation.Nebula.NIFS.PiRLC.ChallengeBridge

/-! Regression surface for the exact selector-to-PiRLC challenge bridge. -/

set_option autoImplicit false

namespace tests.NebulaProductPiRlcChallengeBridge

#check Nightstream.Implementation.Nebula.ProductPiRlcChallengeBridge.challengeSymbol_range
#check Nightstream.Implementation.Nebula.ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse

example :
    Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.scalarCount *
        Nightstream.Implementation.Nebula.ProductPiRlcTranscriptRows.coefficientCount =
      810 := by
  decide

end tests.NebulaProductPiRlcChallengeBridge
