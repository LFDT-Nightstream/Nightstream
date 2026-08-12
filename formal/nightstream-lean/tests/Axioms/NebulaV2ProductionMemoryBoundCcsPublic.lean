import Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic
import tests.Axioms.Support

/-! Dependency audit for the successor batch CCS public carrier. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.word_memoryMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.word_memoryMatches

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.authority_eq_or_memory_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic.authority_eq_or_memory_collision
