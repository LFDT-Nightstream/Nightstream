(* ROM-bound block entry/exit rows and branch depth preservation. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement Wasm2 StraightLineSpec StackSimulation LocalBindings ProgramBinding CallSpec CallBody CallContext
  CallerRefinement ProgramSpec BlockSpec BranchBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma control_row : forall initial final row pc opcode chosen next,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc ->
  filter (fun entry => Z.eqb (snd entry) opcode) selectorCodes = [(chosen, opcode)] ->
  controlImage initial pc opcode next -> row chosen = 1 /\ row COL_PC_AFTER = next.
Proof.
  intros initial final row pc opcode chosen next Hprime Hfull Hprogram Hreplay Hpc Hunique [Hop [Hkind Hnext]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hedge.
  rewrite Hpc, Hkind in Hedge.
  pose proof (full_static_pc_rom_active row Hprime Hfull Hprogram ltac:(congruence)) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hactive) as Htarget.
  rewrite Hpc, Hnext in Htarget. destruct (row COL_CONTROL_CHOICE =? 0); [ | discriminate].
  split; [exact (full_opcode_selects row chosen opcode Hprime Hfull Hprogram Hunique ltac:(congruence)) | congruence].
Qed.

Lemma control_row_static_edge : forall initial final (row : Witness) pc opcode next,
  row COL_IS_PROGRAM_ROW = 1 -> Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc ->
  controlImage initial pc opcode next -> row COL_PC_EDGE_KIND = 0.
Proof.
  intros initial final row pc opcode next Hprogram Hreplay Hpc [_ [Hkind _]].
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hedge.
  rewrite Hpc, Hkind in Hedge. congruence.
Qed.

Lemma br_if_depth_preserved : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_IF_EQZ = 1 ->
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE.
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_BR_IF_EQZ selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { selected_row COL_SEL_BR_IF_EQZ callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { selected_row COL_SEL_BR_IF_EQZ callStackPopSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  pose proof (full_call_depth_balance w Hprime Hfull). lia.
Qed.
