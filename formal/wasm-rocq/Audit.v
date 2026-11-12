Require Import ExecutionRefinement GateExclusions RomSharing StackPorts ProgramBinding.

Check run_refinement.
Check gate_exclusions_sound.
Check shared_rom_replay.
Check full_stack_port_gates.
Check full_host_slot_descriptor.
Check full_host_context_refs.
Definition auditedResults :=
  (@run_refinement, @gate_exclusions_sound, @shared_rom_replay, @full_stack_port_gates, @full_host_slot_descriptor, @full_host_context_refs).
Print Assumptions auditedResults.
