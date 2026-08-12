import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsPublicTranscript
import tests.Axioms.Support

/-! Dependency audit for the field-native production paper-NIFS transcript. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.fixedPrefix_candidate_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.fixedPrefix_candidate_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicInputOf_native_encoding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicInputOf_native_encoding

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicNifsFields_of_value' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicNifsFields_of_value

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicNifsFields_ne_of_candidate_ne' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.publicNifsFields_ne_of_candidate_ne

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frame_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frame_length

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frame_eq_recovers_direct_authority_or_memory_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frame_eq_recovers_direct_authority_or_memory_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frames_ne_of_candidate_ne' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.frames_ne_of_candidate_ne

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.equal_publicState_recovers_authority_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionProductNifsPublicTranscript.equal_publicState_recovers_authority_or_named_failure
