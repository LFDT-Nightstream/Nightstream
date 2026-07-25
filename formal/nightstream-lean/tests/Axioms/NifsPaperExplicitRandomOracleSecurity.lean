import Nightstream.Protocol.FPrime.Frozen
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency guard for the complete model-level paper NIFS
security theorem.
-/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.NifsNonInteractiveBridge.paperNifsSoundCompleteAndNonInteractive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.SuperNeo.paperNifsSoundCompleteAndNonInteractive
