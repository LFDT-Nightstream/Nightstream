(* Reuse the instruction proofs inside a function whose code has an arbitrary entry PC. *)
From Stdlib Require Import List ZArith Lia.
Require Import CallSpec R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Selectors CircuitFacts TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub Locals
  StraightLineSpec StackSimulation LocalSimulation LocalBindings StraightLineInstructions SpecTecCalls CallScope.
Import ListNotations.
Open Scope Z_scope.

Lemma instruction_row_next_pc : forall initial final w instruction,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories w) final ->
  instructionImage initial (w COL_PC_BEFORE) instruction ->
  w COL_PC_AFTER = w COL_PC_BEFORE + 1.
Proof.
  intros initial final w instruction Hprime Hfull Hprogram Hreplay [Hop _].
  pose proof (circuit_row_opcode_fetch _ _ _ Hprime Hfull Hreplay Hprogram) as Hcode. rewrite Hop in Hcode.
  apply (full_fallthrough_pc w Hprime Hfull).
  destruct (encoded_opcode_cases _ _ Hcode) as [[_ Hc] | [[_ Hc] | [[x [_ Hc]] | [x [_ Hc]]]]].
  - exact (full_fallthrough_flag w COL_SEL_I32_ADD CODE_I32_ADD Hprime Hfull ltac:(vm_compute; tauto)
      (full_opcode_selects w COL_SEL_I32_ADD CODE_I32_ADD Hprime Hfull Hprogram ltac:(vm_compute; reflexivity) Hc)).
  - exact (full_fallthrough_flag w COL_SEL_I32_SUB CODE_I32_SUB Hprime Hfull ltac:(vm_compute; tauto)
      (full_opcode_selects w COL_SEL_I32_SUB CODE_I32_SUB Hprime Hfull Hprogram ltac:(vm_compute; reflexivity) Hc)).
  - exact (full_fallthrough_flag w COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime Hfull ltac:(vm_compute; tauto)
      (full_opcode_selects w COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime Hfull Hprogram ltac:(vm_compute; reflexivity) Hc)).
  - exact (full_fallthrough_flag w COL_SEL_LOCAL_SET CODE_LOCAL_SET Hprime Hfull ltac:(vm_compute; tauto)
      (full_opcode_selects w COL_SEL_LOCAL_SET CODE_LOCAL_SET Hprime Hfull Hprogram ltac:(vm_compute; reflexivity) Hc)).
Qed.
