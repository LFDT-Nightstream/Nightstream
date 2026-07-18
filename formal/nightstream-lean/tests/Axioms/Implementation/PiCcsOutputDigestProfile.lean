import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Profile
import tests.Axioms.Support

/-! Fail-closed dependency checks for artifact-independent output profiles. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.fieldCount_ofSemanticShape' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.fieldCount_ofSemanticShape

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_fieldCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_fieldCount

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.steadyFixedPointThirteenMatrix_fieldCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.steadyFixedPointThirteenMatrix_fieldCount

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_ne_steadyFixedPointThirteenMatrix' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_ne_steadyFixedPointThirteenMatrix

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_fieldCount_ne_steadyFixedPointThirteenMatrix' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Profile.diagnosticThreeMatrix_fieldCount_ne_steadyFixedPointThirteenMatrix
