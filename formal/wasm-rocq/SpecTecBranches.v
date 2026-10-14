(* Zero-result block contexts and br_if 0 reduction. Values below the block
   remain outside its label; taken branches discard its instruction continuation. *)
From Stdlib Require Import List ZArith Lia Bool.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers StraightLineSpec StackSimulation SpecTecCalls.
Import ListNotations.
Open Scope Z_scope.

Definition blockContext (c : config) : config :=
  match c with mk_config z code => mk_config z [LABEL_ 0%N [] code] end.

Lemma block_context_wf : forall z code, wf_config (mk_config z code) ->
  wf_config (mk_config z [LABEL_ 0%N [] code]).
Proof. intros z code H; inversion H; subst; repeat constructor; assumption. Qed.

Lemma steps_in_block : forall before after,
  Steps before after -> Steps (blockContext before) (blockContext after).
Proof.
  intros before after Hsteps. induction Hsteps.
  - apply Steps__refl. now apply block_context_wf.
  - eapply trans; [apply ctxt_label; eassumption | exact IHHsteps | | | ];
      apply block_context_wf; assumption.
Qed.

Lemma one_spec_step : forall before after,
  wf_config before -> wf_config after -> Step before after -> Steps before after.
Proof.
  intros [z code] [z' code'] Hb Ha Hstep.
  eapply trans; [exact Hstep | apply Steps__refl; exact Ha | exact Hb | exact Ha | exact Ha].
Qed.

Lemma enter_empty_block : forall z body,
  wf_state z -> Forall wf_instr body ->
  Steps (stackConfig z [] [BLOCK (_RESULT None) body]) (blockContext (stackConfig z [] body)).
Proof.
  intros z body Hz Hbody. apply one_spec_step.
  - apply stack_config_wf; [exact Hz | constructor | ]. repeat constructor; assumption.
  - apply block_context_wf. apply stack_config_wf; assumption || constructor.
  - apply read. apply (Step_read__block z 0%N [] (_RESULT None) body 0%N [] []);
      apply (introT eqP); reflexivity.
Qed.

Lemma exit_empty_block : forall z,
  wf_state z -> Steps (blockContext (stackConfig z [] [])) (stackConfig z [] []).
Proof.
  intros z Hz. apply one_spec_step.
  - apply block_context_wf. repeat constructor; exact Hz.
  - repeat constructor; exact Hz.
  - apply pure. apply (label_vals 0%N [] []).
Qed.

Definition branchBlockConfig z condition rest : config :=
  blockContext (stackConfig z [condition] (BR_IF (mk_uN 0) :: rest)).

Definition branchBlockResult z condition rest : config :=
  if Z.eqb condition 0 then blockContext (stackConfig z [] rest) else stackConfig z [] [].

Lemma br_if_block_reduces : forall z condition rest,
  wf_state z -> 0 <= condition < i32Modulus -> Forall wf_instr rest ->
  Steps (branchBlockConfig z condition rest) (branchBlockResult z condition rest).
Proof.
  intros z condition rest Hz Hbound Hrest.
  assert (Hb : wf_config (stackConfig z [condition] [BR_IF (mk_uN 0)])).
  { apply stack_config_wf; [exact Hz | constructor; [exact Hbound | constructor] | repeat constructor]. }
  assert (Hempty : wf_config (stackConfig z [] [])) by (repeat constructor; exact Hz).
  destruct (Z.eqb condition 0) eqn:Hcondition.
  - apply Z.eqb_eq in Hcondition. subst condition.
    assert (Hstep : Step (stackConfig z [0] [BR_IF (mk_uN 0)]) (stackConfig z [] [])).
    { apply pure. apply br_if_false; reflexivity. }
    pose proof (steps_in_block _ _ (steps_with_instruction_suffix rest _ _ Hrest
      (one_spec_step _ _ Hb Hempty Hstep))) as Hsteps.
    exact Hsteps.
  - apply Z.eqb_neq in Hcondition.
    assert (Hbr : wf_config (stackConfig z [] [BR (mk_uN 0)])) by (repeat constructor; exact Hz).
    assert (Hstep : Step (stackConfig z [condition] [BR_IF (mk_uN 0)]) (stackConfig z [] [BR (mk_uN 0)])).
    { apply pure. apply br_if_true; [reflexivity | ].
      change (is_true (negb (eq_op (Z.to_N condition) 0%N))).
      apply negb_true_iff. apply (introF eqP). intro Heq. apply (f_equal Z.of_N) in Heq.
      rewrite Z2N.id in Heq by lia. cbn in Heq. contradiction. }
    pose proof (steps_in_block _ _ (steps_with_instruction_suffix rest _ _ Hrest
      (one_spec_step _ _ Hb Hbr Hstep))) as Hsteps.
    assert (Hmiddle : wf_config (blockContext (stackConfig z [] (BR (mk_uN 0) :: rest)))).
    { apply block_context_wf. apply stack_config_wf; [exact Hz | constructor | constructor; [repeat constructor | exact Hrest]]. }
    assert (Hexit : Step (blockContext (stackConfig z [] (BR (mk_uN 0) :: rest))) (stackConfig z [] [])).
    { apply pure. apply (br_zero 0%N [] [] [] (map admininstr_instr rest)). reflexivity. }
    unfold branchBlockConfig, branchBlockResult. apply Z.eqb_neq in Hcondition. rewrite Hcondition.
    eapply spec_steps_transitive; [exact Hsteps | ]. apply one_spec_step; assumption.
Qed.

Definition blockBranchConfig z prefix condition rest continuation : config :=
  stackPrefix prefix (instructionSuffix continuation (branchBlockConfig z condition rest)).
Definition blockBranchResult z prefix condition rest continuation : config :=
  stackPrefix prefix (instructionSuffix continuation (branchBlockResult z condition rest)).

Lemma br_if_under_block_context : forall z prefix condition rest continuation,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) prefix ->
  0 <= condition < i32Modulus -> Forall wf_instr rest -> Forall wf_instr continuation ->
  Steps (blockBranchConfig z prefix condition rest continuation)
    (blockBranchResult z prefix condition rest continuation).
Proof.
  intros. apply steps_with_stack_prefix; [assumption | ].
  apply steps_with_instruction_suffix; [assumption | ]. apply br_if_block_reduces; assumption.
Qed.
