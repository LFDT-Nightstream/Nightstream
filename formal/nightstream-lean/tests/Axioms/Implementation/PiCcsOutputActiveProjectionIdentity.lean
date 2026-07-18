import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity
import tests.Axioms.Support

/-! Fail-closed kernel dependency expectation for the artifact-independent
active PiCCS `y_zcol` source to PiRLC projection boundary. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.rows_decodedOutput_eq_messageAggregate_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.rows_decodedOutput_eq_messageAggregate_or_badRoot
