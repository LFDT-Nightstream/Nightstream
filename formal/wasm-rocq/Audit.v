Require Import ExecutionRefinement GateExclusions RomSharing StackPorts ProgramBinding.

Check run_refinement.
Check gate_exclusions_sound.
Check shared_rom_replay.
Check full_stack_port_gates.
Check full_host_slot_descriptor.
Check full_host_context_refs.
Check full_host_context_args_depth.
Definition auditedResults :=
  (@run_refinement, @gate_exclusions_sound, @shared_rom_replay, @full_stack_port_gates, @full_host_slot_descriptor, @full_host_context_refs, @full_host_context_args_depth).
Print Assumptions auditedResults.
