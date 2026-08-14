import Nightstream.Protocol.Nebula.ProductionProfileCandidates
import tests.Axioms.Support

/-! Dependency audit for successor field-native profile candidates. -/

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.identities_pairwise_distinct' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.identities_pairwise_distinct

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.exact_segment_partition' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.exact_segment_partition

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.local_batch_end_le_segment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.local_batch_end_le_segment

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.candidate_count_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.candidate_count_table

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.memorySuffixCoordinate_split_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.memorySuffixCoordinate_split_exact

/-- info: 'Nightstream.Protocol.Nebula.ProductionProfileCandidates.fieldNativeEnvelopeCoordinate_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionProfileCandidates.fieldNativeEnvelopeCoordinate_table
