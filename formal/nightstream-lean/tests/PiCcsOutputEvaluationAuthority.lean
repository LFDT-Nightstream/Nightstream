import Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority

/-!
Kernel-facing regression for the abstract `PiCCS` output-evaluation authority
gap. The theorem is intentionally negative: accepted FE/NC chains and shape
checks do not determine the output CE evaluation arrays.
-/

namespace tests.PiCcsOutputEvaluationAuthority

open Nightstream.SuperNeo.Folding.PiCCS

#check OutputEvaluationAuthority.accepted_does_not_determine_output_evaluations
#check OutputEvaluationAuthority.accepted_replaceEvaluations_iff
#check OutputEvaluationAuthority.accepted_does_not_determine_common_output_point

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_output_evaluations' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms OutputEvaluationAuthority.accepted_does_not_determine_output_evaluations

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_replaceEvaluations_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms OutputEvaluationAuthority.accepted_replaceEvaluations_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_common_output_point' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms OutputEvaluationAuthority.accepted_does_not_determine_common_output_point

end tests.PiCcsOutputEvaluationAuthority
