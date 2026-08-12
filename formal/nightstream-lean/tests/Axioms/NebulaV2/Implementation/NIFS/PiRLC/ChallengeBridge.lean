import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.ChallengeBridge
import tests.Axioms.Support

/-! Dependency audit for the exact selector-to-PiRLC challenge bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.challengeSymbol_range' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.challengeSymbol_range

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiRlcChallengeBridge.decodeChallenges_eq_piRlcResponse
