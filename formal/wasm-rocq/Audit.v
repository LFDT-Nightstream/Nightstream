Require Import ExecutionRefinement GateExclusions RomSharing StackPorts.

Check run_refinement.
Check gate_exclusions_sound.
Check shared_rom_replay.
Check full_stack_port_gates.
Definition auditedResults :=
  (@run_refinement, @gate_exclusions_sound, @shared_rom_replay, @full_stack_port_gates).
Print Assumptions auditedResults.
