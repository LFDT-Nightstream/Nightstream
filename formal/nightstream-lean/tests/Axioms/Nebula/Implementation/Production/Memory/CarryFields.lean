import Nightstream.Implementation.Nebula.Production.Memory.CarryFields
import tests.Axioms.Support

/-! Dependency audit for the field-native production memory carry. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCarryFields.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCarryFields.encode_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryCarryFields.encode_fields_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryCarryFields.encode_fields_canonical
