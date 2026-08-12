import Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
import tests.Axioms.Support

/-! Fail-closed dependency gate for the production low-norm encoding. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.encodeLogical_private_word_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.encodeLogical_private_word_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.decode_privateTritWord' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.decode_privateTritWord

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.encodeCarrier_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.encodeCarrier_norm

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.projectPublicInput_encodeCarrier' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.projectPublicInput_encodeCarrier
