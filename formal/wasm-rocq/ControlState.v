(* Preservation and well-formedness of the source/control representation. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 R1CS MemoryTrace StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches
  CallerSpec CallerRefinement ProgramSpec ProgramValidation SourceValidation CodeLayout BlockSpec BlockValidation
  FunctionValidation ControlDecode ControlInvariant Columns Range FullRelation MemoryCatalog OpcodeCatalog Selectors ProgramBinding
  CallInitialization StraightLineInstructions.
Import ListNotations.
Open Scope Z_scope.

Ltac not_call_selector :=
  let H := fresh in intro H; vm_compute in H; repeat (destruct H as [H | H]; [discriminate H | ]); exact H.

(* An ordinary instruction's row selects its own, non-call opcode. *)
Lemma fetched_ordinary_keeps_idle : forall initial final (w : Witness) i,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories w) final ->
  initial programOpcodeId [w COL_PC_BEFORE] = encodedOpcode i -> (exists op, encodedOpcode i = Some op) ->
  initializationIdle (beforeBoundary w) -> initializationIdle (afterBoundary w).
Proof.
  intros initial final w i Hprime Hfull Hprogram Hreplay Hrom [op Hop] Hidle.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hrom, Hop in Hfetch. injection Hfetch as Hcode.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> ->] | [[-> ->] | [[x [-> ->]] | [x [-> ->]]]]].
  - apply (program_row_keeps_idle w COL_SEL_I32_ADD); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_I32_ADD CODE_I32_ADD Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_I32_SUB); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_I32_SUB CODE_I32_SUB Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_LOCAL_GET); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_LOCAL_SET); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_LOCAL_SET CODE_LOCAL_SET Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
Qed.

Lemma code_at_advance : forall f offset executed remaining,
  codeAt f offset (executed ++ remaining) -> codeAt f (offset + length executed) remaining.
Proof.
  intros f offset executed remaining [Hbound Hcode]. split.
  - pose proof (f_equal (@length instr) Hcode) as Hlength.
    rewrite length_skipn, length_app in Hlength. lia.
  - replace (offset + length executed)%nat with (length executed + offset)%nat by lia.
    rewrite <- skipn_skipn, Hcode, skipn_app, skipn_all, Nat.sub_diag. reflexivity.
Qed.

Lemma frame_matches_preserved : forall program f before after,
  sameCaller before after -> frameMatches program f before -> frameMatches program f after.
Proof.
  intros program f before after [Hs [Ha Hl]] [Hfunctions [Haddresses Hlocals]].
  unfold frameMatches. rewrite Ha, Hl. split; [ | auto].
  destruct before, after. cbn [fun_store fun_funcinst] in *. congruence.
Qed.

Lemma validated_code_preserved : forall program f z z' height code,
  sameCaller z z' -> wf_state z -> wf_state z' -> frameMatches program f z ->
  validatedCode program f z height code -> validatedCode program f z' height code.
Proof.
  intros program f z z' height code Hsame Hz Hz' Hframe Hvalid.
  pose proof (validated_code_facts _ _ _ _ _ Hframe Hz Hvalid) as Hfacts.
  eapply validated_code_remaining with (executed := []) (before := height); try eassumption.
  - eapply frame_matches_preserved; eassumption.
  - reflexivity.
  - eapply source_facts_preserved; eassumption.
Qed.

Lemma validated_instructions_wf : forall program f z height code,
  validatedCode program f z height code -> Forall wf_instr code.
Proof. intros program f z height code [_ [_ [C [_ H]]]]. eapply validated_code_wf; exact H. Qed.

Lemma supported_branch_suffix : forall body offset,
  supportedBranchBody body -> supportedBranchBody (skipn offset body).
Proof.
  intros body offset H. revert offset. induction H; intros [ | offset]; cbn; try constructor; auto.
Qed.

Lemma validated_code_module : forall program f z height code,
  validatedCode program f z height code -> exists C, Moduleinst_ok (fun_store z) (guestModule f) C.
Proof. intros program f z height code [_ [_ [C [H _]]]]. exists C. exact H. Qed.

(* The row proofs type block bodies in blockTypingContext. It agrees with SpecTec's
   block context on the locals and the innermost label, the only parts consulted
   by a supported block body. *)
Lemma block_rest_typed : forall program f z height code,
  frameMatches program f z -> supportedBranchBody code -> validatedBlockRest f z height code ->
  Instrs_ok (blockTypingContext (length (stateLocals z))) code (i32StackType height 0).
Proof.
  intros program f z height code [_ [_ Hcount]] Hsupport [C [Hmodule Hvalid]].
  eapply branch_validation_context;
    [exact Hvalid | exact Hsupport | apply label_context_label | repeat constructor | | reflexivity].
  rewrite label_context_locals, (function_context_locals _ _ _ _ _ _ _ (module_context_locals _ _ _ Hmodule)),
    Hcount. reflexivity.
Qed.

Lemma block_rest_validated : forall program f z height code C,
  frameMatches program f z -> supportedBranchBody code -> Moduleinst_ok (fun_store z) (guestModule f) C ->
  Instrs_ok (blockTypingContext (length (stateLocals z))) code (i32StackType height 0) ->
  validatedBlockRest f z height code.
Proof.
  intros program f z height code C [_ [_ Hcount]] Hsupport Hmodule Htyped.
  exists C. split; [exact Hmodule | ].
  eapply branch_validation_context; [exact Htyped | exact Hsupport | reflexivity | | | apply label_context_label].
  - apply label_context_wf. apply function_context_wf. exact (module_context_wf _ _ _ Hmodule).
  - rewrite label_context_locals, (function_context_locals _ _ _ _ _ _ _ (module_context_locals _ _ _ Hmodule)),
      Hcount. reflexivity.
Qed.

Lemma active_state_wf : forall program cursor z memory boundary c,
  ActiveState program cursor z memory boundary c -> wf_config c.
Proof.
  intros program [position | position offset prefix | function arguments] z memory boundary c H.
  - destruct H as [[code H] _]. unfold SourceState in H.
    destruct (functionAt program (sourceFunction position)); [ | contradiction].
    eapply represents_wf. exact (proj1 (proj2 (proj2 (proj2 H)))).
  - destruct H as [f [body [continuation [values [Hf [Hi [Hcode [Hframe [Hvalid [Hs [Hbound [[C [_ Ht]] [Hb [Hz [Hstack [Hlocals [Hpc [_ ->]]]]]]]]]]]]]]]]]].
    pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues. apply Forall_app in Hvalues.
    apply stack_prefix_wf; [exact (proj1 Hvalues) | ].
    apply instruction_suffix_wf; [eapply validated_instructions_wf; exact Hvalid | ].
    apply block_context_wf. apply stack_config_wf;
      [exact Hz | exact (proj2 Hvalues) | eapply validated_code_wf; exact Ht].
  - destruct H as [f [_ [_ [_ [_ [_ [_ [_ [Hwf [_ [_ [_ [_ ->]]]]]]]]]]]]]. exact Hwf.
Qed.

Lemma block_exit_pc : forall f outer body,
  nth_error (guestBody f) outer = Some (BLOCK (_RESULT None) body) ->
  guestEntry f + Z.of_nat (codeOffset (guestBody f) (S outer)) =
    guestEntry f + Z.of_nat (codeOffset (guestBody f) outer) + 2 + Z.of_nat (length body).
Proof.
  intros f outer body Hi. rewrite (code_offset_next _ _ _ Hi), Nat2Z.inj_add.
  unfold instructionSize, flattenInstruction. cbn [length]. rewrite length_app, block_code_length. cbn [length]. lia.
Qed.

Lemma skipn_head_lookup : forall (body : list instr) offset i rest,
  skipn offset body = i :: rest -> nth_error body offset = Some i /\ skipn (S offset) body = rest.
Proof.
  induction body as [ | j body IH]; intros [ | offset] i rest H; try discriminate.
  - injection H as <- <-. split; reflexivity.
  - apply IH. exact H.
Qed.

Lemma block_continuation_state : forall program position f body continuation prefix z memory boundary,
  functionAt program (sourceFunction position) = Some f ->
  nth_error (guestBody f) (sourceOffset position) = Some (BLOCK (_RESULT None) body) ->
  codeAt f (S (sourceOffset position)) continuation -> frameMatches program f z ->
  validatedCode program f z (length prefix) continuation -> wf_state z ->
  stackMatches memory boundary prefix -> localsMatch memory boundary z ->
  boundaryPC boundary = guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position)) + 2 + Z.of_nat (length body) ->
  initializationIdle boundary ->
  ActiveState program (atInstruction (advanceSource position 1)) z memory boundary (stackConfig z prefix continuation).
Proof.
  intros program position f body continuation prefix z memory boundary Hf Hi Hcode Hframe Hv Hz Hstack Hl Hpc Hidle.
  split; [ | exact Hidle]. exists continuation. unfold SourceState, advanceSource. cbn [sourceFunction sourceOffset]. rewrite Hf.
  replace (sourceOffset position + 1)%nat with (S (sourceOffset position)) by lia.
  split; [exact Hframe | ]. split; [exact Hcode | ]. split.
  - rewrite (stack_matches_height _ _ _ Hstack). exact Hv.
  - split.
    + exists prefix. split; [exact Hstack | ]. split; [reflexivity | ]. split; [ | exact Hl].
      apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) |
        eapply validated_instructions_wf; exact Hv].
    + rewrite (block_exit_pc f (sourceOffset position) body Hi). exact Hpc.
Qed.
