import Nightstream.Implementation.NebulaV2.ProductConcreteNifs
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact V2 paper-NIFS key. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_arity_total' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_arity_total

/-- info: 'Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_initialTranscriptState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_initialTranscriptState

/-- info: 'Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_publicInputState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_publicInputState

/-- info: 'Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_matrixSource' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_matrixSource

/-- info: 'Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_commitment_map' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductConcreteNifs.key_commitment_map
