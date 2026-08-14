import Nightstream.Protocol.Nebula.ProductionBatchGeometry
import tests.Axioms.Support

/-! Dependency audit for exact successor batch indexing and lane geometry. -/

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchGeometry.encode_bijective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchGeometry.encode_bijective

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchGeometry.candidate_geometry_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchGeometry.candidate_geometry_table
