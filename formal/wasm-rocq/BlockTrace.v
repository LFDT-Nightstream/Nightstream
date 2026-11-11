(* ROM-bound block entry/exit rows and branch depth preservation. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement Wasm2 StraightLineSpec StackSimulation LocalBindings ProgramBinding CallSpec CallBody CallContext
  CallerRefinement ProgramSpec BlockSpec BranchBindings CallInitialization.
Import ListNotations.
Open Scope Z_scope.

Lemma control_row : forall initial final row pc opcode chosen next,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc ->
  filter (fun entry => Z.eqb (snd entry) opcode) selectorCodes = [(chosen, opcode)] ->
  ~ In chosen [COL_SEL_BR_IF_EQZ; COL_SEL_IF; COL_SEL_BR_TABLE] ->
  controlImage initial pc opcode next -> row chosen = 1 /\ row COL_PC_AFTER = next.
Proof.
  intros initial final row pc opcode chosen next Hprime Hfull Hprogram Hreplay Hpc Hunique Hstatic [Hop [Hkind Hnext]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hedge.
  rewrite Hpc, Hkind in Hedge.
  pose proof (full_static_pc_rom_active row Hprime Hfull Hprogram ltac:(congruence)) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hprime Hfull Hreplay Hactive) as Htarget.
  assert (Hchosen : row chosen = 1)
    by exact (full_opcode_selects row chosen opcode Hprime Hfull Hprogram Hunique ltac:(congruence)).
  assert (Hin : In chosen selectorColumns).
  { assert (Hentry : In (chosen, opcode) selectorCodes)
      by (apply (proj1 (filter_In (fun entry => Z.eqb (snd entry) opcode) _ _)); rewrite Hunique; left; reflexivity).
    apply (in_map fst), opcode_selector_columns in Hentry. exact Hentry. }
  rewrite (full_selected_choice_pinned row chosen Hprime Hfull Hin Hchosen Hstatic), Z.add_0_r, Hpc, Hnext in Htarget. split; [exact Hchosen | congruence].
Qed.

Lemma control_row_static_edge : forall initial final (row : Witness) pc opcode next,
  row COL_IS_PROGRAM_ROW = 1 -> Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc ->
  controlImage initial pc opcode next -> row COL_PC_EDGE_KIND = 0.
Proof.
  intros initial final row pc opcode next Hprogram Hreplay Hpc [_ [Hkind _]].
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hedge.
  rewrite Hpc, Hkind in Hedge. congruence.
Qed.

Lemma branch_depth_preserved : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> In chosen [COL_SEL_BR; COL_SEL_BR_IF_EQZ; COL_SEL_BR_TABLE] -> w chosen = 1 ->
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE.
Proof.
  intros w chosen Hprime Hfull Hbranch Hchosen.
  assert (Hin : In chosen selectorColumns) by (destruct Hbranch as [<- | [<- | [<- | []]]]; vm_compute; tauto).
  destruct (full_branch_frame w chosen Hprime Hfull Hin Hchosen Hbranch) as [Hpop Hpush _ _ _].
  pose proof (full_call_depth_balance w chosen Hprime Hfull Hin Hchosen). lia.
Qed.
