(* SpecTec validation supplies the numerical facts for ordinary rows, calls,
   and zero-result blocks. No new typing judgement is a public premise. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StraightLineValidation SpecTecCalls
  CallSpec CallerSpec CallerRefinement CallerValidation BlockSpec BlockValidation SourceInstructions.
Import ListNotations.
Open Scope Z_scope.

Lemma source_heights_app : forall callees a b h,
  sourceHeights callees (a ++ b) h =
    match sourceHeights callees a h with Some next => sourceHeights callees b next | None => None end.
Proof. intros callees a; induction a; intros; cbn; [reflexivity | ]. destruct (sourceHeight callees a h); auto. Qed.

Lemma source_instruction_height : forall C i inputs outputs callees z,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  context_LOCALS C = repeat valtype_I32 (length (stateLocals z)) -> calleeTypes C callees ->
  supportedSourceInstruction callees i ->
  forall extra, sourceHeight callees i (extra + length inputs) = Some (extra + length outputs)%nat.
Proof.
  intros C i inputs outputs callees z Hi Hl Htypes Hsupport extra.
  assert (Hcases : (exists bt body, i = BLOCK bt body) \/ (exists x, i = CALL x) \/ (exists op, encodedOpcode i = Some op)).
  { destruct i; try (right; right; exact (Forall_inv Hsupport)); eauto. }
  destruct Hcases as [[bt [body ->]] | [[x ->] | Hordinary]].
  - destruct (block_validation_facts _ _ _ _ _ _ Hi Hl Hsupport) as [-> [-> _]]. reflexivity.
  - destruct (Forall_inv Hsupport) as [f [Hf [Hp Hr]]]. inversion Hi; subst.
    pose proof (Htypes _ _ Hf) as Htype. apply (elimT eqP) in H4.
    match type of H4 with ?lhs = _ =>
      assert (Hlookup : lhs = i32StackType (guestParams f) (guestResults f))
        by (apply spec_lookup_from_nth_error; exact Htype); rewrite Hlookup in H4 end.
    inversion H4; subst. cbn [sourceHeight]. rewrite Hf, !repeat_length. f_equal; lia.
  - pose proof (proj1 (branch_instruction_height _ _ _ _ Hi (or_intror Hordinary)) extra) as Hheight.
    destruct Hordinary as [op Hop].
    destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; exact Hheight.
Qed.

Lemma source_validation_facts : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
  forall callees z,
    context_LOCALS C = repeat valtype_I32 (length (stateLocals z)) -> calleeTypes C callees ->
    (forall x f, callees (x :> N) = Some f -> wf_uN 32%N x ->
      (guestParams f < 256)%nat -> (guestResults f < 256)%nat ->
      typedCall z x (guestModule f) (guestTypeIndex f) (guestParams f) (guestResults f) (guestLocals f) (guestBody f)) ->
    supportedSourceCode callees code ->
    (forall extra suffix, sourceFacts callees z (extra + length outputs) suffix ->
      sourceFacts callees z (extra + length inputs) (code ++ suffix)) /\
    (forall extra, sourceHeights callees code (extra + length inputs) = Some (extra + length outputs)%nat)
  end.
Proof.
  apply Instrs_ok_ind.
  - intros; split; intros; assumption || reflexivity.
  - intros C i inputs outputs Hi HC Hw callees z Hl Htypes Hcalls Hsupport.
    pose proof (Forall_inv Hsupport) as Hsupported.
    pose proof (source_instruction_height _ _ _ _ _ _ Hi Hl Htypes Hsupported) as Hheight.
    split.
    + intros extra suffix Hsuffix. cbn [app sourceFacts]. rewrite Hheight. split; [ | exact Hsuffix].
      assert (Hcases : (exists bt body, i = BLOCK bt body) \/ (forall bt body, i <> BLOCK bt body)).
      { destruct i; try (right; intros; discriminate). left; eauto. }
      destruct Hcases as [[bt [body ->]] | Hordinary].
      * split; [exact Hsupported | ]. split; [exact Hw | ].
        exact (proj2 (proj2 (block_validation_facts _ _ _ _ _ _ Hi Hl Hsupported))).
      * assert (Hs : supportedCaller callees [i]).
        { destruct i; try exact Hsupported. exfalso; eapply Hordinary; reflexivity. }
        assert (Hv : Instrs_ok C [i] (mk_functype (mk_list _ inputs) (mk_list _ outputs)))
          by (apply Instrs_ok__instr; assumption).
        pose proof (caller_validation_facts _ _ _ Hv callees z ltac:(rewrite Hl, repeat_length; reflexivity)
          Htypes Hcalls Hs extra [] I) as Hfacts.
        rewrite app_nil_r in Hfacts. destruct i; try exact Hfacts. exfalso; eapply Hordinary; reflexivity.
    + intro extra. cbn [sourceHeights]. now rewrite Hheight.
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb callees z Hl Htypes Hcalls Hsupport.
    apply Forall_app in Hsupport.
    destruct (IHa callees z Hl Htypes Hcalls (proj1 Hsupport)) as [Hfa Hha].
    destruct (IHb callees z Hl Htypes Hcalls (proj2 Hsupport)) as [Hfb Hhb]. split.
    + intros extra suffix Hsuffix. rewrite <- app_assoc. apply Hfa. now apply Hfb.
    + intro extra. rewrite source_heights_app, Hha. apply Hhb.
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw callees z Hl Htypes Hcalls Hsupport.
    rewrite (result_sub_length _ _ Hin), <- (result_sub_length _ _ Hout). apply (IH callees z); assumption.
  - intros C code prefix inputs outputs Hv IH HC Hw callees z Hl Htypes Hcalls Hsupport.
    destruct (IH callees z Hl Htypes Hcalls Hsupport) as [Hfacts Hheight]. rewrite !length_app. split.
    + intros extra suffix Hsuffix. rewrite Nat.add_assoc in Hsuffix |- *. now apply Hfacts.
    + intro extra. rewrite !Nat.add_assoc. apply Hheight.
Qed.

Lemma source_instruction_wf : forall callees z height i,
  sourceInstructionFacts callees z height i -> wf_instr i.
Proof.
  intros callees z height i H. destruct i; try exact (proj1 (proj2 H));
    apply (Forall_inv (caller_program_wf _ _ _ _ H)).
Qed.

Lemma source_facts_wf : forall code callees z height,
  sourceFacts callees z height code -> Forall wf_instr code.
Proof.
  induction code as [ | i rest IH]; intros callees z height H; [constructor | ].
  destruct H as [Hi Hr]. constructor; [eapply source_instruction_wf; eassumption | ].
  destruct (sourceHeight callees i height) as [next | ]; [eapply IH; exact Hr | contradiction].
Qed.

Lemma source_validation_height : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) => forall callees z,
    context_LOCALS C = repeat valtype_I32 (length (stateLocals z)) -> calleeTypes C callees ->
    supportedSourceCode callees code ->
    forall extra, sourceHeights callees code (extra + length inputs) = Some (extra + length outputs)%nat
  end.
Proof.
  apply Instrs_ok_ind.
  - intros; reflexivity.
  - intros C i inputs outputs Hi HC Hw callees z Hl Htypes Hsupport extra.
    cbn [sourceHeights]. rewrite (source_instruction_height _ _ _ _ _ _ Hi Hl Htypes (Forall_inv Hsupport) extra). reflexivity.
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb callees z Hl Htypes Hsupport extra.
    apply Forall_app in Hsupport. rewrite source_heights_app, (IHa callees z Hl Htypes (proj1 Hsupport) extra).
    apply (IHb callees z); tauto.
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw callees z Hl Htypes Hsupport extra.
    rewrite (result_sub_length _ _ Hin), <- (result_sub_length _ _ Hout). apply (IH callees z); assumption.
  - intros C code prefix inputs outputs Hv IH HC Hw callees z Hl Htypes Hsupport extra.
    rewrite !length_app, !Nat.add_assoc. apply (IH callees z); assumption.
Qed.

Lemma caller_singleton_supported : forall callees z height i,
  callerFacts callees z height [i] -> supportedCaller callees [i].
Proof.
  intros callees z height i H. constructor; [ | constructor].
  destruct i; try exact (Forall_inv (proj1 H)).
  cbn [callerFacts] in H. destruct (callees (v_funcidx :> N)) as [f | ] eqn:Hf; [ | contradiction].
  exists f. split; [reflexivity | ]. destruct H as [[_ [_ [_ [Hp Hr]]]] _]. auto.
Qed.

Lemma source_facts_supported : forall code callees z height,
  sourceFacts callees z height code -> supportedSourceCode callees code.
Proof.
  induction code as [ | i rest IH]; intros callees z height H; [constructor | ].
  destruct H as [Hi Hr]. constructor.
  - destruct i; try (apply (caller_singleton_supported _ _ _ _ Hi)); exact (proj1 Hi).
  - destruct (sourceHeight callees i height) as [next | ]; [eapply IH; exact Hr | contradiction].
Qed.

Lemma source_facts_validate : forall code callees z C height,
  context_LOCALS C = repeat valtype_I32 (length (stateLocals z)) ->
  wf_context C -> calleeTypes C callees -> sourceFacts callees z height code ->
  exists outputs, Instrs_ok C code (i32StackType height outputs).
Proof.
  induction code as [ | i rest IH]; intros callees z C height Hl HC Htypes Hfacts.
  - exists height. replace height with (height + 0)%nat by lia.
    unfold i32StackType. rewrite repeat_app. apply Instrs_ok__frame; [constructor | | ]; assumption || constructor.
  - pose proof (source_facts_wf _ _ _ _ Hfacts) as Hwf.
    pose proof (source_facts_supported _ _ _ _ Hfacts) as Hsupport.
    destruct Hfacts as [Hi Hr].
    assert (Hsingle : exists next, Instrs_ok C [i] (i32StackType height next)).
    { destruct i; try (eapply caller_facts_validate; eassumption).
      destruct Hi as [[-> Hshape] [Hw Htyped]]. exists height.
      replace height with (height + 0)%nat at 1 2 by lia.
      apply i32_validation_frame; [ | exact HC | exact Hw].
      eapply block_validates_context; eauto. split; [reflexivity | exact Hshape]. }
    destruct Hsingle as [next Hvalid].
    assert (Hnext : sourceHeight callees i height = Some next).
    { pose proof (source_validation_height _ _ _ Hvalid callees z Hl Htypes
        ltac:(constructor; [exact (Forall_inv Hsupport) | constructor]) 0%nat) as H.
      rewrite !repeat_length in H. cbn [sourceHeights Nat.add] in H.
      destruct (sourceHeight callees i height); [exact H | discriminate]. }
    rewrite Hnext in Hr.
    destruct (IH _ _ _ _ Hl HC Htypes Hr) as [outputs Htail]. exists outputs.
    change (Instrs_ok C ([i] ++ rest) (i32StackType height outputs)).
    eapply res_seq; [exact Hvalid | exact Htail | exact HC |
      constructor; [exact (Forall_inv Hwf) | constructor] | exact (Forall_inv_tail Hwf)].
Qed.

Lemma source_facts_preserved : forall code callees z z' height,
  sameCaller z z' -> wf_state z' -> sourceFacts callees z height code -> sourceFacts callees z' height code.
Proof.
  induction code as [ | i rest IH]; intros callees z z' height Hsame Hz Hfacts; [exact I | ].
  destruct Hfacts as [Hi Hr]. split.
  - destruct i; try (eapply caller_typed_preserved; [exact (proj2 (proj2 Hsame)) | exact Hz | exact Hi]).
    destruct Hi as [Hs [Hw Ht]]. split; [exact Hs | ]. split; [exact Hw | ].
    now rewrite (proj2 (proj2 Hsame)).
  - destruct (sourceHeight callees i height); [eapply IH; eassumption | contradiction].
Qed.
