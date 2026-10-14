(* Numerical stack facts and context transport for the supported block body. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StackSimulation StraightLineValidation BlockSpec CallerValidation.
Import ListNotations.
Open Scope Z_scope.

Definition localIndexValid (locals : nat) (i : instr) : Prop :=
  match i with LOCAL_GET x | LOCAL_SET x => (N.to_nat (x :> N) < locals)%nat | _ => True end.
Lemma branch_heights_app : forall a b h,
  branchHeights (a ++ b) h =
    match branchHeights a h with Some next => branchHeights b next | None => None end.
Proof.
  induction a; intros; cbn; [reflexivity | ].
  destruct (branchHeight a h); [apply IHa | reflexivity].
Qed.

Lemma branch_instruction_height : forall C i inputs outputs,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  (i = BR_IF (mk_uN 0) \/ exists op, encodedOpcode i = Some op) ->
  (forall extra, branchHeight i (extra + length inputs) = Some (extra + length outputs)%nat) /\
  localIndexValid (length (context_LOCALS C)) i.
Proof.
  intros C i inputs outputs Hi [-> | [op Hop]].
  - inversion Hi; subst. split; [ | exact I]. intros extra.
    rewrite length_app. cbn [length]. replace (extra + (length outputs + 1))%nat with (S (extra + length outputs)) by lia.
    reflexivity.
  - destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
      inversion Hi; subst; cbn [length localIndexValid]; split.
    all: try (intros extra; replace (extra + 2)%nat with (S (S extra)) by lia;
      replace (extra + 1)%nat with (S extra) by lia; cbn [branchHeight instructionStackHeight]; f_equal; lia).
    all: try exact I.
    all: match goal with H : is_true (N.ltb _ _) |- _ =>
      apply N.ltb_lt in H; change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N in H; lia end.
Qed.

Lemma branch_validation_heights : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
    supportedBranchBody code ->
    (forall extra, branchHeights code (extra + length inputs) = Some (extra + length outputs)%nat) /\
    Forall (localIndexValid (length (context_LOCALS C))) code
  end.
Proof.
  apply Instrs_ok_ind.
  - intros; split; [reflexivity | constructor].
  - intros C i inputs outputs Hi HC Hw Hsupport.
    destruct (branch_instruction_height _ _ _ _ Hi (Forall_inv Hsupport)) as [Hheight Hlocal].
    split; [intro extra; cbn [branchHeights]; now rewrite Hheight | constructor; [exact Hlocal | constructor]].
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb Hsupport.
    apply Forall_app in Hsupport. destruct (IHa (proj1 Hsupport)) as [Hha Hla], (IHb (proj2 Hsupport)) as [Hhb Hlb].
    split; [intro extra; rewrite branch_heights_app, Hha; apply Hhb | now apply Forall_app].
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw Hsupport.
    rewrite (result_sub_length _ _ Hin), <- (result_sub_length _ _ Hout). exact (IH Hsupport).
  - intros C code prefix inputs outputs Hv IH HC Hw Hsupport.
    destruct (IH Hsupport) as [Hheight Hlocal]. split; [ | exact Hlocal].
    intros extra. rewrite !length_app, !Nat.add_assoc. apply Hheight.
Qed.

Lemma straight_branch_height_facts : forall code count before after,
  supportedStraightLineProgram code -> Forall (localIndexValid count) code ->
  branchHeights code before = Some after ->
  programTyped count before code /\ bodyStackDelta code = Some (Z.of_nat after - Z.of_nat before).
Proof.
  induction code as [ | i rest IH]; intros count before after Hsupport Hlocals Hheight.
  - cbn in Hheight. injection Hheight as <-. split; [exact I | cbn; f_equal; lia].
  - destruct (Forall_inv Hsupport) as [op Hop].
    assert (Hordinary : branchHeight i before = instructionStackHeight i before).
    { destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity. }
    cbn [branchHeights] in Hheight. rewrite Hordinary in Hheight.
    destruct (instructionStackHeight i before) as [next | ] eqn:Hnext; [ | discriminate].
    destruct (IH count next after (Forall_inv_tail Hsupport) (Forall_inv_tail Hlocals) Hheight) as [Htyped Hdelta].
    split.
    + pose proof (Forall_inv Hlocals) as Hlocal.
      destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
        cbn [programTyped]; rewrite Hnext; auto.
    + cbn [bodyStackDelta]. rewrite (instruction_stack_height_delta _ _ _ Hnext), Hdelta. f_equal; lia.
Qed.

Lemma typed_block_facts : forall count x tail,
  supportedStraightLineProgram tail ->
  typedBlock count (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail) ->
  (N.to_nat (x :> N) < count)%nat /\ programTyped count 0 tail /\ bodyStackDelta tail = Some 0.
Proof.
  intros count x tail Hsupport Htyped.
  assert (Hs : supportedBranchBody (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail)).
  { constructor; [right; eexists; reflexivity | ]. constructor; [left; reflexivity | ].
    eapply Forall_impl; [ | exact Hsupport]. intros i H; now right. }
  destruct (branch_validation_heights _ _ _ Htyped Hs) as [Hheight Hlocals].
  specialize (Hheight 0%nat). cbn [branchHeights branchHeight instructionStackHeight i32StackType length] in Hheight.
  cbn [blockTypingContext context_LOCALS] in Hlocals. rewrite repeat_length in Hlocals.
  pose proof (Forall_inv Hlocals) as Hindex.
  destruct (straight_branch_height_facts tail count 0 0 Hsupport (Forall_inv_tail (Forall_inv_tail Hlocals)) Hheight)
    as [Ht Hd]. auto.
Qed.

Lemma branch_validation_context : forall C code ft,
  Instrs_ok C code ft -> supportedBranchBody code ->
  nth_error (LABELS C) 0 = Some (mk_list _ []) ->
  forall D, wf_context D -> context_LOCALS D = context_LOCALS C ->
    nth_error (LABELS D) 0 = Some (mk_list _ []) -> Instrs_ok D code ft.
Proof.
  refine (Instrs_ok_ind (fun C code ft => supportedBranchBody code ->
    nth_error (LABELS C) 0 = Some (mk_list _ []) ->
    forall D, wf_context D -> context_LOCALS D = context_LOCALS C ->
      nth_error (LABELS D) 0 = Some (mk_list _ []) -> Instrs_ok D code ft) _ _ _ _ _).
  - intros; constructor; assumption.
  - intros C i inputs outputs Hi HC Hw Hsupport Hlabel D HD Hl HlabelD.
    apply Instrs_ok__instr; try assumption.
    destruct (Forall_inv Hsupport) as [-> | [op Hop]].
    + inversion Hi; subst.
      match goal with H : is_true (eq_op (proj_list_0 _ _) _) |- _ =>
        apply (elimT eqP) in H;
        rewrite (SpecTecCalls.spec_lookup_from_nth_error _ _ _ 0%N _ Hlabel) in H;
        cbn [proj_list_0] in H; subst end.
      apply br_if; try assumption.
      * apply N.ltb_lt. change (0 < N.of_nat (length (LABELS D)))%N.
        assert (0 < length (LABELS D))%nat by (apply nth_error_Some; rewrite HlabelD; discriminate). lia.
      * apply (introT eqP). rewrite (SpecTecCalls.spec_lookup_from_nth_error _ _ _ 0%N _ HlabelD). reflexivity.
    + destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
        inversion Hi; subst; econstructor; try eassumption; rewrite Hl; assumption.
  - intros C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb Hsupport Hlabel D HD Hl HlabelD.
    apply Forall_app in Hsupport. eapply res_seq; try eassumption.
    + apply IHa; tauto.
    + apply IHb; tauto.
  - intros C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw Hsupport Hlabel D HD Hl HlabelD.
    eapply sub; try eassumption. apply IH; assumption.
  - intros C code prefix inputs outputs Hv IH HC Hw Hsupport Hlabel D HD Hl HlabelD.
    eapply Instrs_ok__frame; try eassumption. apply IH; assumption.
Qed.

Lemma supported_block_body : forall bt body,
  supportedBlock bt body -> supportedBranchBody body.
Proof.
  intros bt body [_ [x [tail [-> Htail]]]].
  constructor; [right; eexists; reflexivity | ]. constructor; [left; reflexivity | ].
  eapply Forall_impl; [ | exact Htail]. intros i H; now right.
Qed.

Lemma block_validation_facts : forall C bt body inputs outputs count,
  Instr_ok C (BLOCK bt body) (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  context_LOCALS C = repeat valtype_I32 count -> supportedBlock bt body ->
  inputs = [] /\ outputs = [] /\ typedBlock count body.
Proof.
  intros C bt body inputs outputs count Hi Hl Hsupport.
  pose proof (proj1 Hsupport) as ->. inversion Hi; subst.
  match goal with H : Blocktype_ok _ (_RESULT None) _ |- _ => inversion H; subst end.
  split; [reflexivity | ]. split; [reflexivity | ].
  eapply branch_validation_context; [eassumption | eapply supported_block_body; exact Hsupport | reflexivity | repeat constructor | | reflexivity].
  symmetry. exact Hl.
Qed.

Lemma block_validates_context : forall count C body,
  context_LOCALS C = repeat valtype_I32 count -> wf_context C ->
  supportedBlock (_RESULT None) body -> wf_instr (BLOCK (_RESULT None) body) -> typedBlock count body ->
  Instr_ok C (BLOCK (_RESULT None) body) (i32StackType 0 0).
Proof.
  intros count C body Hl HC Hsupport Hw Htyped.
  apply block.
  - apply Blocktype_ok__valtype; [exact HC | repeat constructor].
  - eapply branch_validation_context; [exact Htyped | eapply supported_block_body; exact Hsupport | reflexivity | | exact Hl | reflexivity].
    inversion HC; constructor; assumption.
  - exact HC.
  - exact Hw.
  - repeat constructor.
Qed.

Lemma validated_code_wf : forall C code ft,
  Instrs_ok C code ft -> Forall wf_instr code.
Proof.
  intros C code ft H. destruct H; try assumption.
  - constructor.
  - constructor; [assumption | constructor].
  - apply Forall_app; auto.
Qed.

Lemma branch_height_ordinary : forall i h,
  (exists op, encodedOpcode i = Some op) -> branchHeight i h = instructionStackHeight i h.
Proof.
  intros i h [op H]. destruct (encoded_opcode_cases _ _ H) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity.
Qed.

Lemma branch_singleton_typed : forall i count before after,
  (exists op, encodedOpcode i = Some op) -> localIndexValid count i ->
  instructionStackHeight i before = Some after -> programTyped count before [i].
Proof.
  intros i count before after Hs Hl Hh.
  apply (proj1 (straight_branch_height_facts [i] count before after
    ltac:(constructor; [exact Hs | constructor]) ltac:(constructor; [exact Hl | constructor])
    ltac:(cbn [branchHeights]; rewrite branch_height_ordinary by exact Hs; now rewrite Hh))).
Qed.

Lemma balanced_tail : forall i code before after,
  branchSitesBalanced (i :: code) before -> branchHeight i before = Some after ->
  branchSitesBalanced code after.
Proof.
  intros i code before after H Hh. destruct i; cbn [branchSitesBalanced] in H; cbn [branchHeight] in Hh;
    try contradiction; try (rewrite Hh in H; exact H).
  destruct H as [-> H]. cbn [branchHeight] in Hh. injection Hh as <-. exact H.
Qed.


Lemma branch_facts_validate : forall code count before after,
  supportedBranchBody code -> Forall wf_instr code -> Forall (localIndexValid count) code ->
  branchHeights code before = Some after ->
  Instrs_ok (blockTypingContext count) code (i32StackType before after).
Proof.
  induction code as [ | i code IH]; intros count before after Hs Hw Hl Hheight.
  - cbn in Hheight. injection Hheight as <-.
    replace before with (before + 0)%nat by lia. unfold i32StackType. rewrite repeat_app.
    apply Instrs_ok__frame; [constructor | | ]; repeat constructor.
  - cbn [branchHeights] in Hheight.
    destruct (branchHeight i before) as [next | ] eqn:Hnext; [ | discriminate].
    assert (Hi : Instrs_ok (blockTypingContext count) [i] (i32StackType before next)).
    { destruct (Forall_inv Hs) as [-> | Hop].
      - destruct before; [discriminate | ]. injection Hnext as <-.
        replace (S before) with (before + 1)%nat by lia. replace before with (before + 0)%nat at 2 by lia.
        apply i32_validation_frame; [apply (br_if _ _ []) | | ]; try reflexivity; repeat constructor.
      - rewrite (branch_height_ordinary _ _ Hop) in Hnext.
        destruct (ordinary_instruction_validates (blockTypingContext count) count i before ltac:(reflexivity)
          ltac:(repeat constructor) (Forall_inv Hw) ltac:(constructor; [exact Hop | constructor])
          (branch_singleton_typed _ _ _ _ Hop (Forall_inv Hl) Hnext)) as [n [Hn Hvalid]].
        rewrite Hnext in Hn. injection Hn as <-. exact Hvalid. }
    change (Instrs_ok (blockTypingContext count) ([i] ++ code) (i32StackType before after)).
    eapply res_seq; [exact Hi | apply IH; exact (Forall_inv_tail Hs) || exact (Forall_inv_tail Hw) || exact (Forall_inv_tail Hl) || exact Hheight | repeat constructor |
      constructor; [exact (Forall_inv Hw) | constructor] | exact (Forall_inv_tail Hw)].
Qed.

Lemma branch_validation_tail : forall count before after i code next,
  Instrs_ok (blockTypingContext count) (i :: code) (i32StackType before after) ->
  supportedBranchBody (i :: code) -> branchHeight i before = Some next ->
  Instrs_ok (blockTypingContext count) code (i32StackType next after).
Proof.
  intros count before after i code next Hv Hs Hnext.
  destruct (branch_validation_heights _ _ _ Hv Hs) as [Hh Hl].
  specialize (Hh 0%nat). rewrite !repeat_length in Hh.
  cbn [Nat.add branchHeights] in Hh. rewrite Hnext in Hh.
  cbn [blockTypingContext context_LOCALS] in Hl. rewrite repeat_length in Hl.
  apply branch_facts_validate; [exact (Forall_inv_tail Hs) |
    exact (Forall_inv_tail (validated_code_wf _ _ _ Hv)) | exact (Forall_inv_tail Hl) | exact Hh].
Qed.
