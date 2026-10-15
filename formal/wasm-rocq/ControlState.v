(* Preservation and well-formedness of the source/control representation. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 R1CS MemoryTrace StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches
  CallerSpec CallerRefinement ProgramSpec ProgramValidation SourceValidation CodeLayout BlockSpec BlockValidation ControlSpec.
Import ListNotations.
Open Scope Z_scope.

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
  - eapply source_facts_preserved; eassumption.
Qed.

Lemma validated_instructions_wf : forall program f z height code,
  validatedCode program f z height code -> Forall wf_instr code.
Proof. intros program f z height code [_ [_ [C [outputs [_ H]]]]]. eapply validated_code_wf; exact H. Qed.

Lemma control_state_wf : forall program cursor z memory boundary c,
  ControlState program cursor z memory boundary c -> wf_config c.
Proof.
  intros program [position | position offset prefix] z memory boundary c H.
  - destruct H as [code H]. unfold SourceState in H.
    destruct (functionAt program (sourceFunction position)); [ | contradiction].
    eapply represents_wf. exact (proj1 (proj2 (proj2 (proj2 H)))).
  - destruct H as [f [body [continuation [values [Hf [Hi [Hcode [Hframe [Hvalid [Hs [Hbound [Ht [Hb [Hz [Hstack [Hlocals [Hpc ->]]]]]]]]]]]]]]]]].
    pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues. apply Forall_app in Hvalues.
    apply stack_prefix_wf; [exact (proj1 Hvalues) | ].
    apply instruction_suffix_wf; [eapply validated_instructions_wf; exact Hvalid | ].
    apply block_context_wf. apply stack_config_wf;
      [exact Hz | exact (proj2 Hvalues) | eapply validated_code_wf; exact Ht].
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
  ControlState program (atInstruction (advanceSource position 1)) z memory boundary (stackConfig z prefix continuation).
Proof.
  intros program position f body continuation prefix z memory boundary Hf Hi Hcode Hframe Hv Hz Hstack Hl Hpc.
  exists continuation. unfold SourceState, advanceSource. cbn [sourceFunction sourceOffset]. rewrite Hf.
  replace (sourceOffset position + 1)%nat with (S (sourceOffset position)) by lia.
  split; [exact Hframe | ]. split; [exact Hcode | ]. split.
  - rewrite (stack_matches_height _ _ _ Hstack). exact Hv.
  - split.
    + exists prefix. split; [exact Hstack | ]. split; [reflexivity | ]. split; [ | exact Hl].
      apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) |
        eapply validated_instructions_wf; exact Hv].
    + rewrite (block_exit_pc f (sourceOffset position) body Hi). exact Hpc.
Qed.
