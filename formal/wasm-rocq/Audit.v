Require Import ExecutionRefinement GateExclusions.

Check run_refinement.
Check gate_exclusions_sound.
Definition auditedResults := (@run_refinement, @gate_exclusions_sound).
Print Assumptions auditedResults.
