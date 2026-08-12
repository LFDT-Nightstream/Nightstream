import Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier
import tests.Axioms.Support

/-! Dependency audit for the mixed successor memory-suffix carrier. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.stepImage_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.stepImage_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.batchImage_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.batchImage_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.batch_coordinate_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier.batch_coordinate_count
