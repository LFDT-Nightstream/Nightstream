import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the paper-level complete `Pi_CCS` output
family and its projection to the canonical executable output message.

These theorems are model-level. They do not assert SumCheck acceptance,
production decoding, Rust/R1CS conformance, or security reduction.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.matrixVectorAt_identityRow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PaperLinearAlgebra.matrixVectorAt_identityRow

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates.IdentityFirstMatrix.matrixVectorAt_first_eq_assignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms IdentityFirstMatrix.matrixVectorAt_first_eq_assignment

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates.FullOutput.honestAt_freshMatrixImage_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FullOutput.honestAt_freshMatrixImage_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates.FullOutput.honestAt_sourceAssignment_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FullOutput.honestAt_sourceAssignment_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates.FullOutput.honestAt_carriedImage_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FullOutput.honestAt_carriedImage_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates.FullOutput.honestAt_toOutputMessage_eq_messageAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FullOutput.honestAt_toOutputMessage_eq_messageAt
