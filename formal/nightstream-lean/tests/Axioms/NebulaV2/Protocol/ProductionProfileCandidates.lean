import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
import tests.Axioms.Support

/-! Dependency audit for successor field-native profile candidates. -/

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.identities_pairwise_distinct' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.identities_pairwise_distinct

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.exact_segment_partition' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.exact_segment_partition

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.local_batch_end_le_segment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.local_batch_end_le_segment

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.candidate_count_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.candidate_count_table

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.memorySuffixCoordinate_split_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.memorySuffixCoordinate_split_exact

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.fieldNativeEnvelopeCoordinate_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionProfileCandidates.fieldNativeEnvelopeCoordinate_table
