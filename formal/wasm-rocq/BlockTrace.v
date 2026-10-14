(* Locate the fallthrough tail and its END from ROM fetches. An endpoint outside
   the block prevents an accepted prefix from stopping before that END. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement Wasm2 StraightLineSpec StackSimulation LocalBindings ProgramBinding CallSpec CallBody CallContext
  CallerRefinement ProgramSpec TraceSegmentation BlockSpec BranchBindings.
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

Lemma scan_block_tail : forall body rows previous initial final entry depth start finish,
  fieldPrime -> validCircuitRun initial rows final -> Chain continuityLinks (previous :: rows) ->
  Forall (callerRow depth) rows -> previous COL_PC_AFTER = entry ->
  previous COL_CALL_STACK_DEPTH_AFTER = depth ->
  start < entry -> entry + Z.of_nat (length body) < finish ->
  ~ (start < boundaryPC (rowsFinalBoundary (afterBoundary previous) rows) < finish) ->
  loadedBody initial entry body -> controlImage initial (entry + Z.of_nat (length body)) CODE_END finish ->
  exists middle endRow suffix,
    rows = middle ++ endRow :: suffix /\ length middle = length body /\
    Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) middle /\
    endRow COL_SEL_END = 1.
Proof.
  induction body as [ | i body IH]; intros rows previous initial final entry depth start finish
    Hprime Hrun Hchain Hkinds Hpc Hdepth Hstart Hfinish Houtside Hloaded Hend;
    destruct rows as [ | row rows].
  - exfalso. apply Houtside. cbn [rowsFinalBoundary afterBoundary boundaryPC]. lia.
  - destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (HrowDepth : row COL_CALL_STACK_DEPTH_BEFORE = depth).
    { pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)); lia. }
    destruct (proj2 (Forall_inv Hkinds) HrowDepth) as [Hprogram _].
    assert (HrowPC : row COL_PC_BEFORE = entry).
    { pose proof (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); lia. }
    cbn [length] in Hend. rewrite Z.add_0_r in Hend.
    destruct (control_row _ _ _ _ CODE_END COL_SEL_END _ Hprime Hfull Hprogram Hreplay HrowPC
      ltac:(vm_compute; reflexivity) Hend) as [Hselected _].
    exists [], row, rows. repeat split; auto; constructor.
  - exfalso. apply Houtside. cbn [rowsFinalBoundary afterBoundary boundaryPC]. lia.
  - destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (Hchain' : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
    assert (HrowDepth : row COL_CALL_STACK_DEPTH_BEFORE = depth).
    { pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)); lia. }
    destruct (proj2 (Forall_inv Hkinds) HrowDepth) as [Hprogram _].
    assert (HrowPC : row COL_PC_BEFORE = entry).
    { pose proof (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); lia. }
    destruct Hloaded as [Himage Hloaded]. rewrite <- HrowPC in Himage.
    destruct (fetched_straight_line_selector _ _ _ _ Hprime Hfull Hprogram Hreplay Himage) as [chosen [Hchoice Hchosen]].
    pose proof (instruction_row_next_pc _ _ _ _ Hprime Hfull Hprogram Hreplay Himage) as Hnext.
    pose proof (proj1 (proj2 (proj2 (full_body_context_quiet row chosen Hprime Hfull Hchoice Hchosen)))) as Hsame.
    assert (Hrom : sameRoms initial between).
    { intros memory Hro address. eapply replay_preserves; [exact Hreplay | apply read_only_row; exact Hro]. }
    assert (Hend' : controlImage between (entry + 1 + Z.of_nat (length body)) CODE_END finish).
    { unfold controlImage in *. preserve_roms Hrom.
      replace (entry + 1 + Z.of_nat (length body)) with (entry + Z.of_nat (length (i :: body))) by (cbn; lia).
      exact Hend. }
    destruct (IH rows row between final (entry + 1) depth start finish Hprime Htail Hchain'
      (Forall_inv_tail Hkinds) ltac:(lia) ltac:(lia) ltac:(lia) ltac:(cbn in Hfinish; lia) Houtside
      (loaded_body_same_roms _ _ _ _ Hrom Hloaded) Hend')
      as [middle [endRow [suffix [Hsplit [Hlength [Hprograms Hselected]]]]]].
    exists (row :: middle), endRow, suffix. split; [cbn; now rewrite Hsplit | ].
    split; [cbn; congruence | ]. split; [constructor; assumption | exact Hselected].
Qed.
