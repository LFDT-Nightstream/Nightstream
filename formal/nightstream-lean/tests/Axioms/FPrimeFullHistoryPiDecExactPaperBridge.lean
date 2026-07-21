import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.ExactPaperBridge
import tests.Axioms.Support

/-! Fail-closed dependency gate for the conditional exact-paper bridge. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.PaperPremises.matrixCount_eq_three' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.PaperPremises.matrixCount_eq_three

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_typedCommitmentEquation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_typedCommitmentEquation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_decodedEvaluationsEquation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_decodedEvaluationsEquation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_refines_outputAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge.strictAccepted_refines_outputAccepted
