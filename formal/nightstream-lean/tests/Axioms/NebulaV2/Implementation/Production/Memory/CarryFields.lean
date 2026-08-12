import Nightstream.Implementation.NebulaV2.Production.Memory.CarryFields
import tests.Axioms.Support

/-! Dependency audit for the field-native production memory carry. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields.encode_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields.encode_fields_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields.encode_fields_canonical
