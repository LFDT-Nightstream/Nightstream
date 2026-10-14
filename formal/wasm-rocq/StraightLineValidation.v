(* SpecTec validation supplies the stack-height and local-index facts consumed
   by the circuit proofs. Opcode support remains a separate restriction. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import Wasm2 StraightLineSpec StraightLineInstructions.
Import ListNotations.
Open Scope Z_scope.

Lemma result_sub_length : forall inputs outputs,
  Resulttype_sub (mk_list _ inputs) (mk_list _ outputs) -> length inputs = length outputs.
Proof.
  intros inputs outputs H. inversion H as [xs ys Hlen Hpair]; subst.
  clear H Hlen. induction Hpair; cbn; congruence.
Qed.

Lemma body_delta_app : forall prefix suffix left right,
  bodyStackDelta prefix = Some left -> bodyStackDelta suffix = Some right ->
  bodyStackDelta (prefix ++ suffix) = Some (left + right).
Proof.
  induction prefix as [ | instruction prefix IH]; intros suffix left right Hp Hs.
  - cbn [bodyStackDelta] in Hp. injection Hp as <-. exact Hs.
  - cbn [app bodyStackDelta] in Hp |- *.
    destruct (instructionStackDelta instruction) as [d | ]; [ | discriminate].
    destruct (bodyStackDelta prefix) as [ds | ] eqn:Hds; [ | discriminate].
    injection Hp as <-. rewrite (IH suffix ds right eq_refl Hs). f_equal. lia.
Qed.

Lemma validated_instruction_height : forall C instruction inputs outputs,
  Instr_ok C instruction (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  (exists code, encodedOpcode instruction = Some code) ->
  (forall extra suffix,
    programTyped (length (context_LOCALS C)) (extra + length outputs) suffix ->
    programTyped (length (context_LOCALS C)) (extra + length inputs) (instruction :: suffix)) /\
  instructionStackDelta instruction = Some (Z.of_nat (length outputs) - Z.of_nat (length inputs)).
Proof.
  intros C instruction inputs outputs Hvalid [code Hcode].
  destruct (encoded_opcode_cases _ _ Hcode) as [[-> _] | [[-> _] |
    [[x [-> _]] | [x [-> _]]]]]; inversion Hvalid; subst;
    cbn [instructionStackDelta length]; split; try reflexivity.
  - intros extra suffix Hsuffix.
    replace (extra + 2)%nat with (S (S extra)) by lia.
    cbn [programTyped instructionStackHeight]. replace (S extra) with (extra + 1)%nat by lia. exact Hsuffix.
  - intros extra suffix Hsuffix.
    replace (extra + 2)%nat with (S (S extra)) by lia.
    cbn [programTyped instructionStackHeight]. replace (S extra) with (extra + 1)%nat by lia. exact Hsuffix.
  - intros extra suffix Hsuffix. cbn [programTyped instructionStackHeight]. split.
    + match goal with Hbound : is_true (N.ltb _ _) |- _ =>
        apply N.ltb_lt in Hbound;
        change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N in Hbound; lia end.
    + replace (S (extra + 0)) with (extra + 1)%nat by lia. exact Hsuffix.
  - intros extra suffix Hsuffix.
    replace (extra + 1)%nat with (S extra) by lia. cbn [programTyped instructionStackHeight]. split.
    + match goal with Hbound : is_true (N.ltb _ _) |- _ =>
        apply N.ltb_lt in Hbound;
        change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N in Hbound; lia end.
    + now rewrite Nat.add_0_r in Hsuffix.
Qed.

(* Quantifying over a preserved stack prefix accounts for Instrs_ok__frame;
   quantifying over the continuation makes sequence composition direct. *)
Lemma validation_stack_effect : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
    supportedStraightLineProgram code ->
    (forall extra suffix,
      programTyped (length (context_LOCALS C)) (extra + length outputs) suffix ->
      programTyped (length (context_LOCALS C)) (extra + length inputs) (code ++ suffix)) /\
    bodyStackDelta code = Some (Z.of_nat (length outputs) - Z.of_nat (length inputs))
  end.
Proof.
  apply Instrs_ok_ind.
  - intros C Hwf Hsupported. split; [intros; assumption | reflexivity].
  - intros C instruction inputs outputs Hvalid Hcontext Hwf Hsupported.
    destruct (validated_instruction_height _ _ _ _ Hvalid (Forall_inv Hsupported)) as [Hheight Hdelta].
    split; [exact Hheight | ]. cbn [bodyStackDelta]. rewrite Hdelta. f_equal. lia.
  - intros C prefix suffix inputs outputs middle Hp IHp Hs IHs Hcontext Hwp Hws Hsupported.
    apply Forall_app in Hsupported. destruct Hsupported as [HpSupport HsSupport].
    destruct (IHp HpSupport) as [HpHeight HpDelta], (IHs HsSupport) as [HsHeight HsDelta].
    split.
    + intros extra rest Hrest. rewrite <- app_assoc. apply HpHeight. now apply HsHeight.
    + replace (Z.of_nat (length outputs) - Z.of_nat (length inputs)) with
        ((Z.of_nat (length middle) - Z.of_nat (length inputs)) +
         (Z.of_nat (length outputs) - Z.of_nat (length middle))) by lia.
      eapply body_delta_app; eassumption.
  - intros C code inputs' outputs' inputs outputs Hvalid IH Hin Hout Hcontext Hwf Hsupported.
    pose proof (result_sub_length _ _ Hin) as Hi. pose proof (result_sub_length _ _ Hout) as Ho.
    rewrite Hi, <- Ho. exact (IH Hsupported).
  - intros C code prefix inputs outputs Hvalid IH Hcontext Hwf Hsupported.
    destruct (IH Hsupported) as [Hheight Hdelta]. rewrite !length_app.
    split.
    + intros extra suffix Hsuffix. rewrite Nat.add_assoc in Hsuffix |- *. now apply Hheight.
    + rewrite Hdelta, !Nat2Z.inj_add. f_equal. lia.
Qed.

Theorem validation_implies_program_typed : forall C code inputs outputs,
  Instrs_ok C code (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  supportedStraightLineProgram code ->
  programTyped (length (context_LOCALS C)) (length inputs) code /\
  bodyStackDelta code = Some (Z.of_nat (length outputs) - Z.of_nat (length inputs)).
Proof.
  intros C code inputs outputs Hvalid Hsupported.
  destruct (validation_stack_effect _ _ _ Hvalid Hsupported) as [Hheight Hdelta].
  split; [ | exact Hdelta]. specialize (Hheight 0%nat [] I). now rewrite app_nil_r in Hheight.
Qed.

Corollary i32_validation_facts : forall localCount inputs outputs code,
  Instrs_ok (i32TypingContext localCount) code (i32StackType inputs outputs) ->
  supportedStraightLineProgram code ->
  programTyped localCount inputs code /\
  bodyStackDelta code = Some (Z.of_nat outputs - Z.of_nat inputs).
Proof.
  intros localCount inputs outputs code Hvalid Hsupported.
  destruct (validation_implies_program_typed _ _ _ _ Hvalid Hsupported) as [Htyped Hdelta].
  cbn [i32TypingContext context_LOCALS] in Htyped. rewrite !repeat_length in Htyped, Hdelta.
  auto.
Qed.

(* Supported straight-line instructions consult only LOCALS. This transports
   a body validated in its real function context to the internal i32 context. *)
Lemma straight_line_validation_context : forall C code ft,
  Instrs_ok C code ft -> supportedStraightLineProgram code ->
  forall D, wf_context D -> context_LOCALS D = context_LOCALS C -> Instrs_ok D code ft.
Proof.
  refine (Instrs_ok_ind (fun C code ft => supportedStraightLineProgram code ->
    forall D, wf_context D -> context_LOCALS D = context_LOCALS C -> Instrs_ok D code ft)
    _ _ _ _ _).
  - intros. constructor; assumption.
  - intros C i inputs outputs Hi HC Hwi Hsupport D HD Hlocals.
    apply Instrs_ok__instr; try assumption.
    destruct (Forall_inv Hsupport) as [opcode Hopcode].
    destruct (encoded_opcode_cases _ _ Hopcode) as [[-> _] | [[-> _] |
      [[x [-> _]] | [x [-> _]]]]]; inversion Hi; subst;
      econstructor; try eassumption;
      rewrite Hlocals; assumption.
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb Hsupport D HD Hl.
    apply Forall_app in Hsupport. eapply res_seq; try eassumption.
    + apply IHa; tauto.
    + apply IHb; tauto.
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw Hsupport D HD Hl.
    eapply sub; try eassumption. apply IH; assumption.
  - intros C code prefix inputs outputs Hv IH HC Hw Hsupport D HD Hl.
    eapply Instrs_ok__frame; try eassumption. apply IH; assumption.
Qed.
