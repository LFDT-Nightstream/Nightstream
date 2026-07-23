import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the paper-level strong `Pi_CCS` reduction.

This is a model-level public-coin theorem. It does not assert Fiat--Shamir,
production decoding, Rust/R1CS conformance, or a cryptographic reduction.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.Statement.sourceProtocolData_toVerifierInput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Statement.sourceProtocolData_toVerifierInput

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.repeatedPublicOutputs_same_phi' does not depend on any axioms -/
#guard_msgs in
#audit_axioms repeatedPublicOutputs_same_phi

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.ambientShape_sourceCount' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ambientShape_sourceCount

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.fullOutput_eq_honestAt_of_ambientOutputHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullOutput_eq_honestAt_of_ambientOutputHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.projectedOutput_eq_messageAt_of_ambientOutputHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms projectedOutput_eq_messageAt_of_ambientOutputHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction.acceptedProbe_extracts_source_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedProbe_extracts_source_or_badEvent
