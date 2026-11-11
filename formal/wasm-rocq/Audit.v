Require Import ExecutionRefinement GateExclusions RomSharing.

Check run_refinement.
Check gate_exclusions_sound.
Check shared_rom_replay.
Definition auditedResults := (@run_refinement, @gate_exclusions_sound, @shared_rom_replay).
Print Assumptions auditedResults.
