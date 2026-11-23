Require Import ExecutionRefinement GateExclusions RomSharing StackPorts ProgramBinding CallInitialization RangeObligations MemoryRanges.

Check run_refinement.
Check gate_exclusions_sound.
Check shared_rom_replay.
Check full_stack_port_gates.
Check full_host_slot_descriptor.
Check full_host_context_refs.
Check full_host_context_args_depth.
Check full_trap_unwind_entry.
Check range_obligations_sound.
Check memory_range_obligations_run.

Definition auditedResults :=
  (@run_refinement, @gate_exclusions_sound, @shared_rom_replay, @full_stack_port_gates,
   @full_host_slot_descriptor, @full_host_context_refs, @full_host_context_args_depth, @full_trap_unwind_entry,
   @range_obligations_sound, @memory_range_obligations_run).
Print Assumptions auditedResults.
