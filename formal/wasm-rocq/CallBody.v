(* Reuse the instruction proofs inside a function whose code has an arbitrary entry PC. *)
From Stdlib Require Import List ZArith Lia.
Require Import CallSpec R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub Locals
  StraightLineSpec StackSimulation LocalSimulation LocalBindings StraightLineInstructions SpecTecCalls CallScope.
Import ListNotations.
Open Scope Z_scope.

Lemma instruction_row_next_pc : forall initial final w instruction,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories w) final ->
  instructionImage initial (w COL_PC_BEFORE) instruction ->
  w COL_PC_AFTER = w COL_PC_BEFORE + 1.
Proof.
  intros initial final w instruction Hprime Hfull Hprogram Hreplay [_ [Hkind [Hnext _]]].
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hkind in Hfetch.
  pose proof (full_static_pc_rom_active w Hprime Hfull Hprogram ltac:(congruence)) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hactive) as Hread.
  rewrite Hnext in Hread. destruct (w COL_CONTROL_CHOICE =? 0); congruence.
Qed.
