import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer
import tests.Axioms.Support

/-! Fail-closed kernel dependency expectation for the PiCCS → PiRLC
`y_zcol` consumer bridge. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer.decodedInputs_eq_yZcol_of_bound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.YZcolConsumer.decodedInputs_eq_yZcol_of_bound
