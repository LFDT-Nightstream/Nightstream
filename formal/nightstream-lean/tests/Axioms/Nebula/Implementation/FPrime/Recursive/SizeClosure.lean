import Nightstream.Implementation.Nebula.FPrime.Recursive.SizeClosure
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.RecursiveSizeClosure

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.payloadCodec_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms payloadCodec_canonical

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.requiredWords_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms requiredWords_exact

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.finiteArtifactCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteArtifactCapacity

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.capacityHoldsForMatchingLayout' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms capacityHoldsForMatchingLayout

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.completeCompilerFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeCompilerFit

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.rowFitOnly_does_not_imply_fullCapacity' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowFitOnly_does_not_imply_fullCapacity

/-- info: 'Nightstream.Implementation.Nebula.RecursiveSizeClosure.finiteCapacity_does_not_imply_requiredRowsPresent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteCapacity_does_not_imply_requiredRowsPresent
