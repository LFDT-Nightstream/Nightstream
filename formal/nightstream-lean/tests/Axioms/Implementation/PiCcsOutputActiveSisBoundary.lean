import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary
import tests.Axioms.Support

/-! Fail-closed kernel dependency expectation for the active PiCCS
source-to-SIS boundary. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary.outputs_eq_apply_of_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.SisBoundary.outputs_eq_apply_of_bound
