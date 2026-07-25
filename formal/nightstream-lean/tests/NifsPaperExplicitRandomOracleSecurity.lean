import Nightstream.Protocol.FPrime.Frozen

/-!
Focused type-level regressions for the complete model-level paper NIFS
security theorem.

The headline combines the deterministic verifier theorem with the
quantitative explicit-random-oracle theorem. Poseidon2 realization is not
part of this test.
-/

#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.paperNifsSoundCompleteAndNonInteractive
#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsExecution_coins_eq_replayInput
#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.piCcsExecution_outgoingState_eq_postOutput
#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.piRlcChallenge_eq_response_after_piCcsOutput
#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleMixtureExplicitRandomOracleContract
#check Nightstream.Protocol.FPrime.Frozen.SuperNeo.fullOracleMixtureNifsNonInteractiveSound
