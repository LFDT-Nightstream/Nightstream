import Nightstream.Implementation.Nebula.Production.Memory.TranscriptHashFrame
import tests.Axioms.Support

/-! Dependency audit for the successor memory challenge frame. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_joint_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_joint_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_fields_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryTranscriptHashFrame.encode_fields_canonical
