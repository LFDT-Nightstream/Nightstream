(* Bridge between generated caller validation and the numerical facts used by
   row simulation. Function types must be bound to the resolved callees. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StraightLineValidation
  SpecTecCalls CallSpec CallerSpec CallerRefinement FunctionValidation.
Import ListNotations.
Open Scope Z_scope.

Definition calleeTypes (C : context) (callees : CalleeEnvironment) : Prop :=
  forall x f, callees x = Some f ->
    nth_error (context_FUNCS C) (N.to_nat x) = Some (i32StackType (guestParams f) (guestResults f)).

Lemma ordinary_caller_prepend : forall i callees z height rest,
  supportedStraightLineProgram [i] -> wf_instr i ->
  programTyped (length (stateLocals z)) height [i] ->
  (match instructionStackHeight i height with
   | Some next => callerFacts callees z next rest | None => False end) ->
  callerFacts callees z height (i :: rest).
Proof.
  intros i callees z height rest Hs Hw Ht Hr.
  destruct (Forall_inv Hs) as [op Hop].
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] |
    [[x [-> _]] | [x [-> _]]]]]; exact (conj Hs (conj Hw (conj Ht Hr))).
Qed.

Lemma caller_validation_facts : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
  forall callees z,
    length (context_LOCALS C) = length (stateLocals z) -> calleeTypes C callees ->
    (forall x f, callees (x :> N) = Some f -> wf_uN 32%N x ->
      (guestParams f < 256)%nat -> (guestResults f < 256)%nat ->
      typedCall z x (guestModule f) (guestTypeIndex f) (guestParams f)
        (guestResults f) (guestLocals f) (guestBody f)) ->
    supportedCaller callees code ->
    forall extra suffix, callerFacts callees z (extra + length outputs) suffix ->
      callerFacts callees z (extra + length inputs) (code ++ suffix)
  end.
Proof.
  apply Instrs_ok_ind.
  - intros; assumption.
  - intros C i inputs outputs Hi HC Hw callees z Hl Htypes Hcalls Hsupport extra suffix Hsuffix.
    pose proof (Forall_inv Hsupport) as Hsupported.
    assert (Hcases : (exists x, i = CALL x) \/ (exists op, encodedOpcode i = Some op)).
    { destruct i; try (right; exact Hsupported). left; eexists; reflexivity. }
    destruct Hcases as [[x ->] | Hordinary].
    2: {
      destruct (validated_instruction_height _ _ _ _ Hi Hordinary) as [Hheight _].
      pose proof (Hheight extra [] I) as Htyped. rewrite Hl in Htyped.
      eapply ordinary_caller_prepend; [constructor; [exact Hordinary | constructor] | exact Hw | exact Htyped | ].
      destruct Hordinary as [op Hop].
      destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] |
        [[x [-> _]] | [x [-> _]]]]]; inversion Hi; subst;
        cbn [instructionStackHeight length] in *.
      - replace (extra + 2)%nat with (S (S extra)) by lia.
        cbn [instructionStackHeight]. replace (S extra) with (extra + 1)%nat by lia. exact Hsuffix.
      - replace (extra + 2)%nat with (S (S extra)) by lia.
        cbn [instructionStackHeight]. replace (S extra) with (extra + 1)%nat by lia. exact Hsuffix.
      - replace (S (extra + 0)) with (extra + 1)%nat by lia. exact Hsuffix.
      - replace (extra + 1)%nat with (S extra) by lia.
        cbn [instructionStackHeight]. now rewrite Nat.add_0_r in Hsuffix.
    }
    destruct Hsupported as [f [Hf [Hp Hr]]]. inversion Hi; subst.
    pose proof (Htypes _ _ Hf) as Htype.
    apply (elimT eqP) in H4.
    match type of H4 with ?lhs = _ =>
      assert (Hlookup : lhs = i32StackType (guestParams f) (guestResults f))
        by (apply spec_lookup_from_nth_error; exact Htype);
      rewrite Hlookup in H4 end.
    inversion H4; subst.
    cbn [app callerFacts]. rewrite Hf. split.
    + apply Hcalls; try assumption. inversion Hw; assumption.
    + rewrite !repeat_length in *. split; [lia | ].
      replace (extra + guestParams f - guestParams f + guestResults f)%nat with
        (extra + guestResults f)%nat by lia. exact Hsuffix.
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb callees z Hl Htypes Hcalls Hsupport extra suffix Hsuffix.
    apply Forall_app in Hsupport. rewrite <- app_assoc.
    eapply IHa; try eassumption; [exact (proj1 Hsupport) | ].
    eapply IHb; try eassumption. exact (proj2 Hsupport).
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw callees z Hl Htypes Hcalls Hsupport extra suffix Hsuffix.
    pose proof (result_sub_length _ _ Hin) as Hi. pose proof (result_sub_length _ _ Hout) as Ho.
    rewrite Hi. rewrite <- Ho in Hsuffix. eapply IH; eassumption.
  - intros C code prefix inputs outputs Hv IH HC Hw callees z Hl Htypes Hcalls Hsupport extra suffix Hsuffix.
    rewrite !length_app, Nat.add_assoc in Hsuffix |- *. eapply IH; eassumption.
Qed.

Lemma i32_validation_frame : forall C i inputs outputs extra,
  Instr_ok C i (i32StackType inputs outputs) -> wf_context C -> wf_instr i ->
  Instrs_ok C [i] (i32StackType (extra + inputs) (extra + outputs)).
Proof.
  intros. unfold i32StackType. rewrite !repeat_app.
  apply Instrs_ok__frame; [apply Instrs_ok__instr; assumption | assumption | constructor; [assumption | constructor]].
Qed.

Lemma ordinary_instruction_validates : forall C localCount i height,
  context_LOCALS C = repeat valtype_I32 localCount -> wf_context C -> wf_instr i ->
  supportedStraightLineProgram [i] -> programTyped localCount height [i] ->
  exists next, instructionStackHeight i height = Some next /\
    Instrs_ok C [i] (i32StackType height next).
Proof.
  intros C localCount i height Hl HC Hw Hs Ht.
  destruct (Forall_inv Hs) as [op Hop].
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] |
    [[x [-> _]] | [x [-> _]]]]].
  - destruct height as [ | [ | n]]; try contradiction.
    exists (S n). split; [reflexivity | ].
    replace (S (S n)) with (n + 2)%nat by lia. replace (S n) with (n + 1)%nat by lia.
    apply i32_validation_frame; [apply binop | | ]; assumption.
  - destruct height as [ | [ | n]]; try contradiction.
    exists (S n). split; [reflexivity | ].
    replace (S (S n)) with (n + 2)%nat by lia. replace (S n) with (n + 1)%nat by lia.
    apply i32_validation_frame; [apply binop | | ]; assumption.
  - destruct Ht as [Hbound _]. exists (S height). split; [reflexivity | ].
    replace height with (height + 0)%nat at 1 by lia.
    replace (S height) with (height + 1)%nat by lia.
    apply i32_validation_frame; try assumption. apply local_get; try assumption.
    + apply N.ltb_lt. change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N.
      rewrite Hl, repeat_length. lia.
    + apply (introT eqP). apply spec_lookup_from_nth_error. rewrite Hl. now apply nth_error_repeat.
  - destruct Ht as [Hbound Ht]. destruct height as [ | n]; [contradiction | ].
    exists n. split; [reflexivity | ].
    replace (S n) with (n + 1)%nat by lia. replace n with (n + 0)%nat at 2 by lia.
    apply i32_validation_frame; try assumption. apply local_set; try assumption.
    + apply N.ltb_lt. change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N.
      rewrite Hl, repeat_length. lia.
    + apply (introT eqP). apply spec_lookup_from_nth_error. rewrite Hl. now apply nth_error_repeat.
Qed.

Lemma caller_facts_validate : forall code callees z C height,
  context_LOCALS C = repeat valtype_I32 (length (stateLocals z)) ->
  wf_context C -> calleeTypes C callees -> callerFacts callees z height code ->
  exists outputs, Instrs_ok C code (i32StackType height outputs).
Proof.
  induction code as [ | i rest IH]; intros callees z C height Hl HC Htypes Hfacts.
  - exists height. replace height with (height + 0)%nat by lia.
    unfold i32StackType. rewrite repeat_app. apply Instrs_ok__frame; [constructor | | ]; assumption || constructor.
  - assert (Hwf : Forall wf_instr (i :: rest)) by (eapply caller_program_wf; exact Hfacts).
    assert (Hstep : exists next, Instrs_ok C [i] (i32StackType height next) /\ callerFacts callees z next rest).
    { destruct i; try (destruct Hfacts as [Hs [Hw [Ht Hr]]];
        destruct (ordinary_instruction_validates _ _ _ _ Hl HC Hw Hs Ht) as [next [Hnext Hv]];
        rewrite Hnext in Hr; exists next; auto).
      cbn [callerFacts] in Hfacts.
      match type of Hfacts with context [callees ?x] => destruct (callees x) as [f | ] eqn:Hf end; [ | contradiction].
      destruct Hfacts as [Hcall [Hbound Hr]].
      exists (height - guestParams f + guestResults f)%nat. split; [ | exact Hr].
      replace height with (height - guestParams f + guestParams f)%nat at 1 by lia.
      apply i32_validation_frame; [ | exact HC | exact (Forall_inv Hwf)].
      apply call; try assumption; [ | | exact (Forall_inv Hwf)].
      + pose proof (Htypes _ _ Hf) as Hnth. apply N.ltb_lt.
        change ((v_funcidx :> N) < N.of_nat (length (context_FUNCS C)))%N.
        assert (N.to_nat (v_funcidx :> N) < length (context_FUNCS C))%nat by
          (apply nth_error_Some; rewrite Hnth; discriminate). lia.
      + apply (introT eqP). apply spec_lookup_from_nth_error. now apply Htypes.
    }
    destruct Hstep as [next [Hstep Hrest]].
    destruct (IH _ _ _ _ Hl HC Htypes Hrest) as [outputs Htail].
    exists outputs. change (Instrs_ok C ([i] ++ rest) (i32StackType height outputs)).
    eapply res_seq; [exact Hstep | exact Htail | exact HC | constructor; [exact (Forall_inv Hwf) | constructor] | exact (Forall_inv_tail Hwf)].
Qed.
