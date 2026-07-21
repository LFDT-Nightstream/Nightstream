import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge
import tests.Axioms.Support

/-! Fail-closed dependencies for the model-level active `PiDEC` result seam. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge.decodedFoldResult_eq_resultOf' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedFoldResult_eq_resultOf

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge.claimsAccepted_decodedFoldResult_eq_resultOf' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms claimsAccepted_decodedFoldResult_eq_resultOf

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge.claimsAccepted_outgoingState_rewrite' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms claimsAccepted_outgoingState_rewrite
