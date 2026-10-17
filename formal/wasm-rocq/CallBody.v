(* Reuse the instruction proofs inside a function whose code has an arbitrary entry PC. *)
From Stdlib Require Import List ZArith Lia.
Require Import CallSpec R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub Locals
  StraightLineSpec StackSimulation LocalSimulation LocalBindings StraightLineInstructions SpecTecCalls CallScope.
Import ListNotations.
Open Scope Z_scope.

Lemma instruction_image_preserved : forall before after events pc instruction,
  instructionImage before pc instruction -> Replay before events after ->
  (forall memory address, readOnlyMemory memory memories = true -> Forall (NoWriteTo memory address) events) ->
  instructionImage after pc instruction.
Proof.
  intros before after events pc instruction [Hop [Hkind [Hnext Hlocal]]] Hreplay Hreadonly.
  split; [ | split; [ | split]].
  - rewrite (replay_preserves _ _ _ _ _ Hreplay (Hreadonly _ _ opcode_memory_read_only)). exact Hop.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay (Hreadonly _ _ edge_kind_memory_read_only)). exact Hkind.
  - intros choice. rewrite (replay_preserves _ _ _ _ _ Hreplay (Hreadonly _ _ pc_memory_read_only)). apply Hnext.
  - destruct instruction; try exact I;
      rewrite (replay_preserves _ _ _ _ _ Hreplay (Hreadonly _ _ local_index_memory_read_only)); exact Hlocal.
Qed.

Lemma loaded_body_preserved : forall before after events body entry,
  loadedBody before entry body -> Replay before events after ->
  (forall memory address, readOnlyMemory memory memories = true -> Forall (NoWriteTo memory address) events) ->
  loadedBody after entry body.
Proof.
  intros before after events body. induction body as [ | instruction rest IH];
    intros entry Hloaded Hreplay Hreadonly; cbn in *; [exact I | ].
  destruct Hloaded as [Hfirst Hrest]. split.
  - eapply instruction_image_preserved; eassumption.
  - eapply IH; eassumption.
Qed.

Lemma frame_row_refinement : forall instruction rest z initial final row c0,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  instructionImage initial (row COL_PC_BEFORE) instruction ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (instruction :: rest) /\
    represents z initial (beforeBoundary row) (instruction :: rest) c0) ->
  exists z' c1, Step c0 c1 /\ (programTyped (length (stateLocals z')) (stackHeight (afterBoundary row)) rest /\
    represents z' final (afterBoundary row) rest c1) /\
    row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
    frameRegistersPreserved row /\
    instructionStackDelta instruction = Some (row COL_SP_AFTER - row COL_SP_BEFORE) /\
    fun_store z' = fun_store z /\ fun_moduleinst z' = fun_moduleinst z /\
    length (stateLocals z') = length (stateLocals z).
Proof.
  assert (Hstore : forall z x value, fun_store (with_local z x value) = fun_store z).
  { intros [s f] x value. reflexivity. }
  intros instruction rest z initial final row c0 Hprime Hfull Hprogram Hreplay
    [Hop [Hkind [Hnext Hlocal]]] Hframe.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch.
  pose proof (frame_row_stack_delta instruction rest z initial row c0 Hprime Hfull Hprogram Hfetch Hframe) as Hdelta.
  pose proof (frame_row_scope instruction rest z initial final row c0
    Hprime Hfull Hprogram Hreplay Hfetch Hframe) as Hscope.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindRead.
  rewrite Hkind in HkindRead.
  assert (Hstatic : row COL_PC_EDGE_KIND = 0) by congruence.
  pose proof (full_static_pc_rom_active row Hprime Hfull Hprogram Hstatic) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hactive) as HpcRead.
  rewrite Hnext in HpcRead. destruct (row COL_CONTROL_CHOICE =? 0); [ | discriminate].
  assert (Hpc : row COL_PC_AFTER = row COL_PC_BEFORE + 1) by congruence.
  destruct (encoded_opcode_cases instruction _ Hfetch)
    as [[-> Hcode] | [[-> Hcode] | [[x [-> Hcode]] | [x [-> Hcode]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1).
    { eapply full_opcode_selects with (code := CODE_I32_ADD); try eassumption; reflexivity. }
    destruct (binary_frame_refinement z initial final row c0 _ rest COL_SEL_I32_ADD
      Hprime Hfull Hreplay ltac:(left; reflexivity) Hchosen
      (Add.full_relation_implies_spectec_add row Hprime Hchosen Hfull) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_I32_SUB = 1).
    { eapply full_opcode_selects with (code := CODE_I32_SUB); try eassumption; reflexivity. }
    destruct (binary_frame_refinement z initial final row c0 _ rest COL_SEL_I32_SUB
      Hprime Hfull Hreplay ltac:(right; reflexivity) Hchosen
      (Sub.full_relation_implies_spectec_sub row Hprime Hchosen Hfull) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1).
    { eapply full_opcode_selects with (code := CODE_LOCAL_GET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_GET) by (cbn; auto).
    destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [_ [_ Hgate]].
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as HindexRead.
    rewrite Hlocal in HindexRead.
    destruct (local_get_frame_refinement z initial final row c0 x rest Hprime Hfull Hreplay
      Hchosen ltac:(congruence) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1).
    { eapply full_opcode_selects with (code := CODE_LOCAL_SET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_SET) by (cbn; auto).
    destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [_ [_ Hgate]].
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as HindexRead.
    rewrite Hlocal in HindexRead.
    destruct (local_set_frame_refinement z initial final row c0 x rest Hprime Hfull Hreplay
      Hchosen ltac:(congruence) Hframe) as [c1 [Hstep Hafter]].
    exists (with_local z x (i32Value (row COL_LOCAL_VALUE))), c1.
    split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | ]. split; [apply Hstore | ]. split.
    + destruct z as [s f]. reflexivity.
    + apply locals_length_after_set.
Qed.

Theorem body_run_refinement : forall body rows previous initial final entry z c0,
  fieldPrime -> validCircuitRun initial rows final ->
  Chain continuityLinks (previous :: rows) ->
  Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) rows ->
  length rows = length body -> loadedBody initial entry body ->
  previous COL_PC_AFTER = entry ->
  (programTyped (length (stateLocals z)) (stackHeight (afterBoundary previous)) body /\
    represents z initial (afterBoundary previous) body c0) ->
  exists z' c1, Steps c0 c1 /\
    (programTyped (length (stateLocals z')) (stackHeight (afterBoundary (finalRow previous rows))) [] /\
    represents z' final (afterBoundary (finalRow previous rows)) [] c1) /\
    (finalRow previous rows) COL_PC_AFTER = entry + Z.of_nat (length body) /\
    FrameEffect (previous COL_STACK_FRAME_BASE_AFTER) (previous COL_LOCALS_FBP_AFTER) initial final /\
    (finalRow previous rows) COL_CALL_STACK_DEPTH_AFTER = previous COL_CALL_STACK_DEPTH_AFTER /\
    (finalRow previous rows) COL_STACK_FRAME_BASE_AFTER = previous COL_STACK_FRAME_BASE_AFTER /\
    (finalRow previous rows) COL_LOCALS_FBP_AFTER = previous COL_LOCALS_FBP_AFTER /\
    bodyStackDelta body = Some ((finalRow previous rows) COL_SP_AFTER - previous COL_SP_AFTER) /\
    fun_store z' = fun_store z.
Proof.
  induction body as [ | instruction rest IH];
    intros rows previous initial final entry z c0 Hprime Hrun Hchain Hprogram Hlength Hloaded Hpc Hframe.
  - destruct rows; [ | discriminate]. destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst.
    exists z, c0. split.
    + pose proof (represents_wf _ _ _ _ _ (proj2 Hframe)) as Hwf. destruct c0. apply Steps__refl. exact Hwf.
    + split; [exact Hframe | ]. split; [cbn; lia | ].
      split; [apply frame_effect_refl | repeat split; cbn [finalRow bodyStackDelta]; reflexivity || (f_equal; lia)].
  - destruct rows as [ | row rows]; [discriminate | ].
    destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (HtailChain : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
    destruct Hloaded as [Himage Hloaded].
    assert (HrowPC : row COL_PC_BEFORE = entry).
    { rewrite <- (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)). exact Hpc. }
    pose proof Hframe as Hbefore. rewrite (follows_boundary _ _ Hlink) in Hframe.
    rewrite <- HrowPC in Himage.
    destruct (frame_row_refinement instruction rest z initial middle row c0 Hprime Hfull
      (Forall_inv Hprogram) Hreplay Himage Hframe) as [z1 [c1 [Hstep [Hafter [Hnext [Heffect [Hregisters [Hdelta [Hstore _]]]]]]]]].
    assert (HloadedMiddle : loadedBody middle (entry + 1) rest).
    { eapply loaded_body_preserved; [exact Hloaded | exact Hreplay | ].
      intros memory address Hreadonly. apply read_only_row. exact Hreadonly. }
    destruct (IH rows row middle final (entry + 1) z1 c1 Hprime Htail HtailChain
      (Forall_inv_tail Hprogram) ltac:(cbn in Hlength; lia) HloadedMiddle ltac:(lia) Hafter)
      as [z2 [c2 [Hsteps [Hfinal [HfinalPC [HrestEffect [Hdepth [Hbase [Hfbp [Hsp HstoreFinal]]]]]]]]]].
    exists z2, c2. split.
    + pose proof (represents_wf _ _ _ _ _ (proj2 Hbefore)) as Hwf0.
      pose proof (represents_wf _ _ _ _ _ (proj2 Hafter)) as Hwf1.
      pose proof (represents_wf _ _ _ _ _ (proj2 Hfinal)) as Hwf2.
      destruct c0, c1, c2. eapply trans; eassumption.
    + split; [exact Hfinal | ]. split.
      * cbn [finalRow length]; rewrite Nat2Z.inj_succ; lia.
      * destruct Hregisters as [HrowDepth [HrowBase HrowFbp]].
        pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)) as HdLink.
        pose proof (Hlink COL_STACK_FRAME_BASE_AFTER COL_STACK_FRAME_BASE_BEFORE ltac:(vm_compute; tauto)) as HbLink.
        pose proof (Hlink COL_LOCALS_FBP_AFTER COL_LOCALS_FBP_BEFORE ltac:(vm_compute; tauto)) as HfLink.
        pose proof (Hlink COL_SP_AFTER COL_SP_BEFORE ltac:(vm_compute; tauto)) as HsLink.
        split.
        -- rewrite HbLink, HfLink. rewrite HrowBase, HrowFbp in HrestEffect.
           eapply frame_effect_trans; eassumption.
        -- cbn [finalRow]. repeat split; try congruence; try lia.
           cbn [bodyStackDelta]. rewrite Hdelta, Hsp. f_equal; lia.
Qed.

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

Lemma supported_opcode_is_not_end : forall instruction,
  encodedOpcode instruction <> Some CODE_END.
Proof.
  intros instruction Heq. destruct (encoded_opcode_cases instruction CODE_END Heq)
    as [[_ H] | [[_ H] | [[x [_ H]] | [x [_ H]]]]]; discriminate.
Qed.

(* The body length is derived from ROM fetches and the ending END, rather than
   assumed as a segmentation premise. *)
Lemma body_rows_length : forall body rows previous endRow initial final entry,
  fieldPrime -> validCircuitRun initial (rows ++ [endRow]) final ->
  Chain continuityLinks (previous :: rows ++ [endRow]) ->
  Forall (fun w : Witness => exists chosen, straightLineSelector chosen /\ w chosen = 1) rows ->
  endRow COL_SEL_END = 1 -> previous COL_PC_AFTER = entry ->
  loadedBody initial entry body ->
  initial programOpcodeId [entry + Z.of_nat (length body)] = Some CODE_END ->
  length rows = length body.
Proof.
  induction body as [ | instruction rest IH];
    intros rows previous endRow initial final entry Hprime Hrun Hchain Hchoices Hend Hpc Hloaded HendImage.
  - destruct rows as [ | row rows]; [reflexivity | ].
    destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    destruct (Forall_inv Hchoices) as [chosen [Hchoice Hselected]].
    pose proof (straight_line_selector_in chosen Hchoice) as Hin.
    pose proof (full_selected_is_program_row row chosen Hprime Hfull Hin Hselected) as Hprogram.
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (HrowPC : row COL_PC_BEFORE = entry)
      by (rewrite <- (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); exact Hpc).
    pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
    replace (entry + Z.of_nat (length ([] : list instr))) with entry in HendImage by (cbn; lia).
    rewrite HrowPC, HendImage in Hfetch.
    assert (HselectedEnd : row COL_SEL_END = 1).
    { eapply full_opcode_selects with (code := CODE_END); try eassumption; [reflexivity | congruence]. }
    pose proof (full_opcode_selector_exclusive row chosen COL_SEL_END Hprime Hfull Hselected Hin
      ltac:(vm_compute; tauto) ltac:(destruct Hchoice as [<- | [<- | [<- | [<- | []]]]]; discriminate)) as Hzero.
    congruence.
  - destruct rows as [ | row rows].
    + destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
      pose proof (full_selected_is_program_row endRow COL_SEL_END Hprime Hfull
        ltac:(vm_compute; tauto) Hend) as Hprogram.
      pose proof (CallContext.full_end_opcode endRow Hprime Hfull Hend) as Hcode.
      assert (Hlink : Follows continuityLinks previous endRow) by (inversion Hchain; subst; assumption).
      assert (HendPC : endRow COL_PC_BEFORE = entry)
        by (rewrite <- (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); exact Hpc).
      pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
      destruct Hloaded as [[Hop _] _]. rewrite HendPC, Hop, Hcode in Hfetch.
      exfalso. exact (supported_opcode_is_not_end instruction Hfetch).
    + destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
      destruct (Forall_inv Hchoices) as [chosen [Hchoice Hselected]].
      pose proof (full_selected_is_program_row row chosen Hprime Hfull
        (straight_line_selector_in chosen Hchoice) Hselected) as Hprogram.
      assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
      assert (HtailChain : Chain continuityLinks (row :: rows ++ [endRow])) by (inversion Hchain; subst; assumption).
      assert (HrowPC : row COL_PC_BEFORE = entry)
        by (rewrite <- (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); exact Hpc).
      destruct Hloaded as [Himage Hloaded].
      rewrite <- HrowPC in Himage.
      pose proof (instruction_row_next_pc initial middle row instruction Hprime Hfull Hprogram Hreplay Himage) as Hnext.
      cbn [length]. f_equal.
      eapply (IH rows row endRow middle final (entry + 1)); try eassumption.
      * exact (Forall_inv_tail Hchoices).
      * lia.
      * eapply loaded_body_preserved; [exact Hloaded | exact Hreplay | ].
        intros memory address Hreadonly. apply read_only_row. exact Hreadonly.
      * rewrite (replay_preserves _ _ _ _ _ Hreplay (read_only_row _ _ _ _ opcode_memory_read_only)).
        replace (entry + 1 + Z.of_nat (length rest)) with (entry + Z.of_nat (length (instruction :: rest)))
          by (cbn [length]; rewrite Nat2Z.inj_succ; lia). exact HendImage.
Qed.
