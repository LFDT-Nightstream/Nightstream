import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.SelectiveCarrier
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the shape-indexed evaluation carrier.

| Guard | Audited boundary |
|---|---|
| Checked indexing | a proved in-range physical column agrees with the legacy total read |
| Decoder compatibility | no-default extraction and the legacy decoder read the same aligned claim |
| Selective count | the independent selective port vocabulary fixes thirteen evaluations |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier.column_eq_getD' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier.column_eq_getD

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier.decode_fromClaim_eq_decodedEvaluations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimEvaluationCarrier.decode_fromClaim_eq_decodedEvaluations

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.SelectiveCarrier.decode_size' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.SelectiveCarrier.decode_size
