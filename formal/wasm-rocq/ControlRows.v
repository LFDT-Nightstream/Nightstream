(* Each row inside a block either advances its body or exits its label. Block rows
   keep the frame registers and touch no memory below the active frame, so they
   run in any frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches CallerSpec CallerRefinement
  ProgramSpec ProgramEncoding ProgramValidation CodeLayout BlockSpec BlockValidation BlockBindings BlockEncoding
  BlockInstructions BlockTrace BranchBindings BranchSpec BranchRefinement ControlDecode ControlInvariant ControlState SourceInstructions
  CallInitialization Selectors CallScope CallContext CallMemory LocalBindings StraightLineInstructions StackBindings ProgramBinding.
Import ListNotations.
Open Scope Z_scope.

(* A row that writes no stack cell, no local and no call context. *)
Lemma quiet_row_frame_effect : forall initial final row base fbp,
  Replay initial (eventsForRow memories row) final ->
  row COL_STACK_WRITE0_ACTIVE = 0 -> row COL_STACK_WRITE0_HI_ACTIVE = 0 ->
  filter isLocalWrite (eventsForRow memories row) = [] -> row COL_CALL_STACK_PUSH_PRESENT = 0 ->
  FrameEffect base fbp initial final.
Proof.
  intros initial final row base fbp Hreplay Hlo Hhi Hlocal Hpush. constructor.
  - intros index _. eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption].
  - intros other index _. eapply no_local_writes_preserve; eassumption.
  - intros memory column address Hin. eapply context_preserved_without_push; eassumption.
Qed.

(* An ordinary row inside a block: the label's prefix is just more of the frame's stack. *)
Lemma framed_row_scope : forall prefix values i z initial final row,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> encodedOpcode i = Some (row COL_OPCODE_CODE) ->
  wf_state z -> wf_instr i -> programTyped (length (stateLocals z)) (length values) [i] ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
  frameRegistersPreserved row.
Proof.
  intros prefix values i z initial final row Hprime Hfull Hprogram Hreplay Hcode Hz Hi Htyped Hstack Hlocals.
  eapply frame_row_scope with (z := z) (rest := []) (c0 := stackConfig z (prefix ++ values) [i]); try eassumption.
  split.
  - rewrite (stack_matches_height _ _ _ Hstack), length_app, Nat.add_comm.
    destruct (encoded_opcode_cases _ _ Hcode) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
      cbn [programTyped instructionStackHeight] in Htyped |- *;
      destruct (length values) as [ | [ | n]]; cbn in Htyped |- *; tauto.
  - exists (prefix ++ values). split; [exact Hstack | ]. split; [reflexivity | ]. split; [ | exact Hlocals].
    apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) | constructor; [exact Hi | constructor]].
Qed.

Theorem block_control_row : forall program position offset prefix z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final ->
  row COL_IS_PROGRAM_ROW = 1 -> row COL_HALTED = 0 ->
  ActiveState program (inBlock position offset prefix) z initial (beforeBoundary row) c0 ->
  exists cursor z' c1, activeFunction program cursor = functionAt program (sourceFunction position) /\
    fun_store z' = fun_store z /\ Steps c0 c1 /\
    ActiveState program cursor z' final (afterBoundary row) c1 /\ frameRegistersPreserved row /\
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros program position offset prefix z initial final row c0 Hprime Hloaded Hfull Hreplay Hprogram Hhalt
    [f [body [continuation [values [Hf [Houter [Hcode [Hframe [Hvalid [Hsupport [Hbound [Htyped [Hbalanced [Hz [Hstack [Hlocals [Hpc [Hidle ->]]]]]]]]]]]]]]]]]].
  pose proof (supported_branch_suffix body offset Hsupport) as HsupportRest.
  destruct (validated_code_module _ _ _ _ _ Hvalid) as [C HC].
  pose proof (block_rest_typed program f z _ _ Hframe HsupportRest Htyped) as Htyped'.
  clear Htyped. rename Htyped' into Htyped.
  pose proof (validated_code_wf _ _ _ Htyped) as HwfRest.
  pose proof (validated_instructions_wf _ _ _ _ _ Hvalid) as HwfCont.
  pose proof (proj1 (proj2 (proj2 Hstack))) as Hbounds. apply Forall_app in Hbounds.
  destruct (skipn offset body) as [ | i rest] eqn:Hrest.
  - assert (HendOffset : offset = length body).
    { pose proof (f_equal (@length instr) Hrest). rewrite length_skipn in H. cbn in H; lia. }
    assert (Hempty : values = []).
    { destruct (branch_validation_heights _ _ _ Htyped HsupportRest) as [H _]. specialize (H 0%nat).
      rewrite !repeat_length in H. cbn [branchHeights Nat.add] in H.
      apply length_zero_iff_nil. congruence. }
    subst values. rewrite app_nil_r in Hstack. subst offset.
    pose proof (encoded_block_end _ _ _ _ _ _ Hloaded Hf Houter) as Himage.
    pose proof (control_row_static_edge _ _ row _ _ _ Hprogram Hreplay Hpc Himage) as Hedge.
    destruct (control_row _ _ row _ CODE_END COL_SEL_END _ Hprime Hfull Hprogram Hreplay Hpc
      ltac:(vm_compute; reflexivity) Himage) as [Hchosen Hnext].
    destruct (full_block_end_quiet row Hprime Hfull Hchosen Hedge Hhalt) as [Hpush Hpop].
    destruct (block_memory_preserved initial final row COL_SEL_END Hprime Hfull ltac:(right; reflexivity)
      Hchosen Hpop Hhalt Hreplay) as [HstackAfter HlocalsAfter].
    pose proof (full_block_context row COL_SEL_END Hprime Hfull ltac:(right; reflexivity) Hchosen Hpop Hhalt) as Hregs.
    pose proof (program_row_keeps_idle row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) Hchosen
      ltac:(not_call_selector) Hidle) as HidleAfter.
    exists (atInstruction (advanceSource position 1)), z, (stackConfig z prefix continuation).
    split; [reflexivity | ].
    split; [reflexivity | ]. split.
    + pose proof (steps_with_stack_prefix prefix _ _ (proj1 Hbounds)
        (steps_with_instruction_suffix continuation _ _ HwfCont (exit_empty_block z Hz))) as Hsteps.
      exact Hsteps.
    + destruct Hregs as [_ [Hbase' [Hfbp' [Hdepth' [Hw0 [Hw1 Hlw]]]]]].
      split; [eapply block_continuation_state; try eassumption; auto | ].
      split; [split; [exact Hdepth' | split; assumption] | eapply quiet_row_frame_effect; eassumption].
  - destruct (skipn_head_lookup _ _ _ _ Hrest) as [Hi HrestNext].
    assert (HboundNext : (S offset <= length body)%nat) by
      (assert (offset < length body)%nat by (apply nth_error_Some; rewrite Hi; discriminate); lia).
    destruct (Forall_inv HsupportRest) as [Hbranch | Hordinary].
    + subst i. cbn [branchSitesBalanced] in Hbalanced. destruct Hbalanced as [Hheight Hbalanced].
      destruct values as [ | condition [ | extra values]]; cbn [length] in Hheight; try discriminate.
      pose proof (encoded_block_branch _ _ _ _ _ _ _ Hloaded Hf Houter Hi) as Himage.
      assert (Himage' : branchImage initial (row COL_PC_BEFORE) (row COL_PC_BEFORE + 1)
        (row COL_PC_BEFORE + 2 + Z.of_nat (length rest))).
      { pose proof (f_equal (@length instr) Hrest) as Hlength. rewrite length_skipn in Hlength. cbn [length] in Hlength.
        cbn [beforeBoundary boundaryPC] in Hpc. rewrite Hpc.
        replace (guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position)) + 1 + Z.of_nat offset + 1)
          with (guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position)) + 2 + Z.of_nat offset) by lia.
        replace (guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position)) + 1 + Z.of_nat offset + 2 + Z.of_nat (length rest))
          with (guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position)) + 2 + Z.of_nat (length body)) by lia.
        exact Himage. }
      assert (Hrep : blockBranchRepresents z initial (beforeBoundary row) (row COL_PC_BEFORE)
        prefix rest continuation BeforeBranch (blockBranchConfig z prefix condition rest continuation)).
      { split; [exact Hz | ]. split; [exact (Forall_inv_tail HwfRest) | ].
        split; [exact HwfCont | ]. split; [exact Hlocals | ]. split; [reflexivity | ].
        exists condition. split; [exact Hstack | reflexivity]. }
      destruct (block_branch_row_step _ _ _ _ _ _ _ _ _ Hprime Hfull Hprogram Hreplay Himage' Hrep)
        as [choice [c1 [Hnot [Hsteps Hafter]]]].
      pose proof (proj1 (circuit_br_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram eq_refl Himage')) as Hchosen.
      pose proof (br_if_depth_preserved row Hprime Hfull Hchosen) as Hdepth.
      destruct (full_block_selector_quiet row COL_SEL_BR_IF_EQZ Hprime Hfull (or_intror eq_refl) Hchosen)
        as [Hpush _].
      assert (Hshape : UnaryStackShape row true).
      { apply full_br_if_stack_shape; [exact Hprime | exact Hfull | exact Hchosen | ].
        pose proof (stack_matches_height _ _ _ Hstack) as Hh. destruct Hstack as [Hb0 [Hsp _]].
        cbn [beforeBoundary boundarySP boundaryStackBase] in Hb0, Hsp.
        rewrite length_app in Hsp. cbn [length] in Hsp. lia. }
      destruct (full_br_if_locals_unchanged row Hprime Hfull Hchosen) as [Hfbp' Hlw].
      assert (Hscope : frameRegistersPreserved row /\
          FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final).
      { split; [split; [exact Hdepth | split; [exact (local_stack_base_after _ _ Hshape) | exact Hfbp']] | ].
        eapply quiet_row_frame_effect; [exact Hreplay | exact (local_write_active _ _ Hshape) |
          exact (local_write_hi_active _ _ Hshape) | exact Hlw | exact Hpush]. }
      pose proof (program_row_keeps_idle row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen
        ltac:(not_call_selector) Hidle) as HidleAfter.
      destruct choice; [contradiction | | ].
      * destruct Hafter as [_ [_ [_ [HlocalsAfter [HpcAfter [HstackAfter ->]]]]]].
        exists (inBlock position (S offset) prefix), z,
          (stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z [] rest)))).
        split; [reflexivity | ].
        split; [reflexivity | ].
        split; [exact Hsteps | ]. split; [ | exact Hscope].
        exists f, body, continuation, []. rewrite HrestNext. cbn [length].
        assert (Ht : Instrs_ok (blockTypingContext (length (stateLocals z))) rest (i32StackType 0 0))
          by (eapply branch_validation_tail; [exact Htyped | exact HsupportRest | reflexivity]).
        pose proof (block_rest_validated program f z _ _ C Hframe (Forall_inv_tail HsupportRest) HC Ht) as HrestValid.
        rewrite app_nil_r. repeat first [assumption | reflexivity | split].
        cbn [beforeBoundary afterBoundary boundaryPC blockBranchPC] in *. lia.
      * destruct Hafter as [_ [_ [_ [HlocalsAfter [HpcAfter [HstackAfter ->]]]]]].
        exists (atInstruction (advanceSource position 1)), z, (stackConfig z prefix continuation).
        split; [reflexivity | ].
        split; [reflexivity | ]. split; [exact Hsteps | ]. split; [ | exact Hscope].
        eapply block_continuation_state; try eassumption.
        pose proof (f_equal (@length instr) Hrest) as Hlength. rewrite length_skipn in Hlength. cbn [length] in Hlength.
        cbn [beforeBoundary afterBoundary boundaryPC blockBranchPC] in *. lia.
    + pose proof (encoded_block_ordinary _ _ _ _ _ _ _ _ Hloaded Hf Houter Hi Hordinary) as Himage.
      cbn [beforeBoundary boundaryPC] in Hpc. rewrite <- Hpc in Himage.
      pose proof (fetched_ordinary_keeps_idle _ _ _ _ Hprime Hfull Hprogram Hreplay (proj1 Himage) Hordinary Hidle)
        as HidleAfter.
      destruct (branch_validation_heights _ _ _ Htyped HsupportRest) as [Hheights Hindices].
      specialize (Hheights 0%nat). rewrite !repeat_length in Hheights.
      cbn [Nat.add branchHeights] in Hheights. rewrite (branch_height_ordinary _ _ Hordinary) in Hheights.
      destruct (instructionStackHeight i (length values)) as [next | ] eqn:Hheight; [ | discriminate].
      cbn [blockTypingContext context_LOCALS] in Hindices. rewrite repeat_length in Hindices.
      destruct (framed_instruction prefix values i z initial final row Hprime Hfull Hprogram Hreplay Himage Hz
        (Forall_inv HwfRest) (branch_singleton_typed _ _ _ _ Hordinary (Forall_inv Hindices) Hheight) Hstack Hlocals)
        as [z' [values' [[Hstep [Hz' [Hstack' [Hlocals' [Hsame Hheight']]]]] [Hnext Hregs]]]].
      pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
      rewrite (proj1 Himage) in Hfetch.
      destruct (framed_row_scope prefix values i z initial final row Hprime Hfull Hprogram Hreplay Hfetch Hz
        (Forall_inv HwfRest) (branch_singleton_typed _ _ _ _ Hordinary (Forall_inv Hindices) Hheight) Hstack Hlocals)
        as [Heffect _].
      assert (Ht : Instrs_ok (blockTypingContext (length (stateLocals z'))) rest (i32StackType (length values') 0)).
      { rewrite (proj2 (proj2 Hsame)). eapply branch_validation_tail; [exact Htyped | exact HsupportRest | ].
        rewrite branch_height_ordinary by exact Hordinary. exact Hheight'. }
      assert (Hb : branchSitesBalanced rest (length values')).
      { eapply balanced_tail; [exact Hbalanced | ]. rewrite branch_height_ordinary by exact Hordinary. exact Hheight'. }
      assert (Hvalid' : validatedCode program f z' (length prefix) continuation)
        by (eapply validated_code_preserved; eassumption).
      assert (Hframe' : frameMatches program f z') by (eapply frame_matches_preserved; eassumption).
      destruct (validated_code_module _ _ _ _ _ Hvalid') as [C' HC'].
      pose proof (block_rest_validated program f z' _ _ C' Hframe' (Forall_inv_tail HsupportRest) HC' Ht) as HrestValid.
      assert (Hv' : Forall (fun v => 0 <= v < WasmIntegers.i32Modulus) values').
      { pose proof (proj1 (proj2 (proj2 Hstack'))) as H. apply Forall_app in H; tauto. }
      pose proof (one_spec_step _ _ ltac:(apply stack_config_wf; [exact Hz | exact (proj2 Hbounds) | constructor; [exact (Forall_inv HwfRest) | constructor]])
        ltac:(apply stack_config_wf; [exact Hz' | exact Hv' | constructor]) Hstep) as Hone.
      pose proof (stack_suffix_steps z z' values values' [i] rest Hone (Forall_inv_tail HwfRest)) as Hinner.
      exists (inBlock position (S offset) prefix), z',
        (stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z' values' rest)))).
      split; [reflexivity | ].
      split; [exact (proj1 Hsame) | ]. split.
      * apply steps_with_stack_prefix; [exact (proj1 Hbounds) | ].
        apply steps_with_instruction_suffix; [exact HwfCont | ]. apply steps_in_block. exact Hinner.
      * split; [ | exact (conj Hregs Heffect)]. exists f, body, continuation, values'. rewrite HrestNext.
        repeat first [assumption | reflexivity | split]. cbn [beforeBoundary afterBoundary boundaryPC] in *. lia.
Qed.

Theorem block_enter_row : forall program position bt body continuation z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final ->
  row COL_IS_PROGRAM_ROW = 1 -> row COL_HALTED = 0 ->
  SourceState program position z initial (beforeBoundary row) (BLOCK bt body :: continuation) c0 ->
  initializationIdle (beforeBoundary row) ->
  exists prefix c1, Steps c0 c1 /\
    ActiveState program (inBlock position 0 prefix) z final (afterBoundary row) c1 /\ frameRegistersPreserved row /\
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros program position bt body continuation z initial final row c0 Hprime Hloaded Hfull Hreplay Hprogram Hhalt Hsource Hidle.
  unfold SourceState in Hsource. destruct (functionAt program (sourceFunction position)) as [f | ] eqn:Hf; [ | contradiction].
  destruct Hsource as [Hframe [Hcode [Hvalid [Hrep Hpc]]]].
  pose proof (validated_code_facts _ _ _ _ _ Hframe (represented_state_wf _ _ _ _ _ Hrep) Hvalid) as Hfacts.
  destruct Hfacts as [[Hs [Hw Htyped]] Hrest]. destruct Hs as [-> [Hsupport Hbalanced]].
  destruct Hrep as [prefix [Hstack [-> [Hwf Hlocals]]]].
  assert (Hz : wf_state z) by (inversion Hwf; assumption).
  pose proof (stack_matches_height _ _ _ Hstack) as Hheight.
  rewrite Hheight in Hrest.
  assert (Hlookup : nth_error (guestBody f) (sourceOffset position) = Some (BLOCK (_RESULT None) body)).
  { pose proof (f_equal (fun xs => nth_error xs 0) (proj2 Hcode)) as H.
    cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
  pose proof (encoded_block_entry _ _ _ _ _ _ Hloaded Hf Hlookup) as Himage.
  destruct (control_row _ _ row _ CODE_BLOCK COL_SEL_BLOCK _ Hprime Hfull Hprogram Hreplay Hpc
    ltac:(vm_compute; reflexivity) Himage) as [Hchosen Hnext].
  destruct (full_block_selector_quiet row COL_SEL_BLOCK Hprime Hfull (or_introl eq_refl) Hchosen) as [Hpush Hpop].
  destruct (block_memory_preserved initial final row COL_SEL_BLOCK Hprime Hfull ltac:(left; reflexivity)
    Hchosen Hpop Hhalt Hreplay) as [HstackAfter HlocalsAfter].
  pose proof (full_block_context row COL_SEL_BLOCK Hprime Hfull ltac:(left; reflexivity) Hchosen Hpop Hhalt) as Hregs.
  pose proof (program_row_keeps_idle row COL_SEL_BLOCK Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hidle) as HidleAfter.
  assert (Hcont : validatedCode program f z (length prefix) continuation).
  { eapply validated_code_remaining with (executed := [BLOCK (_RESULT None) body]); try eassumption.
    all: first [repeat split; reflexivity | cbn [sourceHeights sourceHeight]; now rewrite Hheight]. }
  assert (HcodeNext : codeAt f (S (sourceOffset position)) continuation).
  { replace (S (sourceOffset position)) with (sourceOffset position + length [BLOCK (_RESULT None) body])%nat by (cbn; lia).
    eapply code_at_advance; exact Hcode. }
  destruct (validated_code_module _ _ _ _ _ Hvalid) as [C HC].
  pose proof (block_rest_validated program f z 0 body C Hframe Hsupport HC Htyped) as HbodyValid.
  exists prefix, (stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z [] body)))).
  split.
  - exact (steps_with_stack_prefix prefix _ _ (proj1 (proj2 (proj2 Hstack)))
      (steps_with_instruction_suffix continuation _ _ (validated_instructions_wf _ _ _ _ _ Hcont)
        (enter_empty_block z body Hz (validated_code_wf _ _ _ Htyped)))).
  - destruct Hregs as [_ [Hbase' [Hfbp' [Hdepth' [Hw0 [Hw1 Hlw]]]]]].
    split; [ | split; [split; [exact Hdepth' | split; assumption] | eapply quiet_row_frame_effect; eassumption]].
    exists f, body, continuation, []. cbn [skipn length]. rewrite app_nil_r.
    specialize (HstackAfter prefix Hstack). specialize (HlocalsAfter z Hlocals).
    repeat first [assumption | reflexivity | split]; try lia.
    cbn [beforeBoundary afterBoundary boundaryPC] in *. lia.
Qed.
