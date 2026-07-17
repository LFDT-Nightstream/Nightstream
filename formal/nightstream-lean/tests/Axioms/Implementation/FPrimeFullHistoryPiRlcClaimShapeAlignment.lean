import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ClaimShapeAlignment
import tests.Axioms.Support

/-!
Fail-closed dependency guards for physical CE-claim shape alignment.

| Guard | Audited boundary |
|---|---|
| Generic mismatch | a three-row physical claim cannot inhabit a non-three-matrix semantic shape |
| Recursive artifact | the recursive diagnostic artifact supplies only its checked three-row fact |
| Terminal artifact | the terminal diagnostic artifact supplies only its checked three-row fact |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape.not_aligned_of_threeRows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShape.not_aligned_of_threeRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment.recursiveArtifact_not_selectiveAligned' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment.recursiveArtifact_not_selectiveAligned

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment.terminalArtifact_not_selectiveAligned' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ClaimShapeAlignment.terminalArtifact_not_selectiveAligned
