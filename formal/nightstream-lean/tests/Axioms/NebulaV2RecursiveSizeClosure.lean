import Nightstream.Implementation.NebulaV2.RecursiveSizeClosure
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.RecursiveSizeClosure

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.payloadCodec_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms payloadCodec_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.requiredWords_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms requiredWords_exact

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.finiteArtifactCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteArtifactCapacity

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.capacityHoldsForMatchingLayout' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms capacityHoldsForMatchingLayout

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.completeCompilerFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeCompilerFit

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.rowFitOnly_does_not_imply_fullCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowFitOnly_does_not_imply_fullCapacity

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.finiteCapacity_does_not_imply_requiredRowsPresent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteCapacity_does_not_imply_requiredRowsPresent
