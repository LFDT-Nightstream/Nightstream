import Nightstream.Implementation.Nebula.Production.Memory.BoundCcsPublic
import tests.Axioms.Support

/-! Dependency audit for the successor batch CCS public carrier. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.word_memoryMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.word_memoryMatches

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.authority_eq_or_memory_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic.authority_eq_or_memory_collision
