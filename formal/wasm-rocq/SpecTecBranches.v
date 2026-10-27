(* Zero-result block contexts and branch reduction: a br_if becomes a br, which
   leaves one label per step. Values below a block remain outside its label;
   taken branches discard its instruction continuation. *)
From Stdlib Require Import List ZArith Lia Bool.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers StraightLineSpec StackSimulation SpecTecCalls Locals Add.
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

Lemma wf_label_le : forall k k', (k <= k')%N -> wf_uN 32%N (mk_uN k') -> wf_uN 32%N (mk_uN k).
Proof.
  intros k k' Hk H. inversion H as [? ? Hb]; subst. constructor.
  apply andb_prop in Hb as [_ Hle]. apply andb_true_intro. split.
  - apply N.leb_le. lia.
  - apply N.leb_le. apply N.leb_le in Hle. lia.
Qed.

(* br_if l with an i32 condition: falls through on zero, otherwise becomes br l. *)
Lemma br_if_steps : forall z condition l rest,
  wf_state z -> 0 <= condition < i32Modulus -> wf_uN 32%N l -> Forall wf_instr rest ->
  Steps (stackConfig z [condition] (BR_IF l :: rest))
    (if Z.eqb condition 0 then stackConfig z [] rest else stackConfig z [] (BR l :: rest)).
Proof.
  intros z condition l rest Hz Hbound Hl Hrest.
  assert (Hb : wf_config (stackConfig z [condition] [BR_IF l])).
  { apply stack_config_wf; [exact Hz | constructor; [exact Hbound | constructor] | ].
    constructor; [constructor; exact Hl | constructor]. }
  destruct (Z.eqb condition 0) eqn:Hcondition.
  - apply Z.eqb_eq in Hcondition. subst condition.
    assert (Hstep : Step (stackConfig z [0] [BR_IF l]) (stackConfig z [] [])) by (apply pure; apply br_if_false; reflexivity).
    assert (Hempty : wf_config (stackConfig z [] [])) by (repeat constructor; exact Hz).
    exact (steps_with_instruction_suffix rest _ _ Hrest (one_spec_step _ _ Hb Hempty Hstep)).
  - apply Z.eqb_neq in Hcondition.
    assert (Hbr : wf_config (stackConfig z [] [BR l])).
    { apply stack_config_wf; [exact Hz | constructor | ]. constructor; [constructor; exact Hl | constructor]. }
    assert (Hstep : Step (stackConfig z [condition] [BR_IF l]) (stackConfig z [] [BR l])).
    { apply pure. apply br_if_true; [reflexivity | ].
      change (is_true (negb (eq_op (Z.to_N condition) 0%N))).
      apply negb_true_iff. apply (introF eqP). intro Heq. apply (f_equal Z.of_N) in Heq.
      rewrite Z2N.id in Heq by lia. cbn in Heq. contradiction. }
    exact (steps_with_instruction_suffix rest _ _ Hrest (one_spec_step _ _ Hb Hbr Hstep)).
Qed.

(* With operands below: the same steps, under them. *)
Lemma stack_prefix_config : forall z values more code,
  stackPrefix values (stackConfig z more code) = stackConfig z (values ++ more) code.
Proof. intros. unfold stackPrefix, stackConfig. rewrite map_app, app_assoc. reflexivity. Qed.

Lemma instruction_suffix_config : forall z values code more,
  instructionSuffix more (stackConfig z values code) = stackConfig z values (code ++ more).
Proof. intros. unfold instructionSuffix, stackConfig. rewrite map_app, app_assoc. reflexivity. Qed.

Lemma cat_app : forall (A : Type) (xs ys : list A), seq.cat xs ys = xs ++ ys.
Proof. intros A xs ys. induction xs as [ | x xs IH]; [reflexivity | cbn; f_equal; exact IH]. Qed.

Lemma i32_constants_values : forall values,
  map i32Constant values = seq.map (fun v => admininstr_val v) (map i32Value values).
Proof. induction values as [ | v values IH]; [reflexivity | cbn; f_equal; exact IH]. Qed.

Lemma br_if_steps_below : forall z below condition l rest,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) below -> 0 <= condition < i32Modulus ->
  wf_uN 32%N l -> Forall wf_instr rest ->
  Steps (stackConfig z (below ++ [condition]) (BR_IF l :: rest))
    (if Z.eqb condition 0 then stackConfig z below rest else stackConfig z below (BR l :: rest)).
Proof.
  intros z below condition l rest Hz Hbelow Hbound Hl Hrest.
  rewrite <- stack_prefix_config.
  pose proof (steps_with_stack_prefix below _ _ Hbelow (br_if_steps z condition l rest Hz Hbound Hl Hrest)) as H.
  destruct (Z.eqb condition 0); rewrite !stack_prefix_config in *; rewrite ?app_nil_r in H; exact H.
Qed.

(* A branch leaving a label carries every operand on its stack out of it. *)
Lemma br_exit_label_carrying : forall z values k rest,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values ->
  wf_uN 32%N (mk_uN (k + 1)%N) -> Forall wf_instr rest ->
  Steps (blockContext (stackConfig z values (BR (mk_uN (k + 1)%N) :: rest))) (stackConfig z values [BR (mk_uN k)]).
Proof.
  intros z values k rest Hz Hvalues Hl Hrest. apply one_spec_step.
  - apply block_context_wf. apply stack_config_wf; [exact Hz | exact Hvalues | ].
    constructor; [constructor; exact Hl | exact Hrest].
  - apply stack_config_wf; [exact Hz | exact Hvalues | ].
    constructor; [apply instr_case_7; exact (wf_label_le k (k + 1)%N ltac:(lia) Hl) | constructor].
  - apply pure. unfold blockContext, stackConfig. rewrite i32_constants_values.
    pose proof (br_succ 0%N [] (map i32Value values) (mk_uN k) (map admininstr_instr rest)) as Hstep.
    rewrite !cat_app, <- app_assoc in Hstep. exact Hstep.
Qed.

(* A branch reaching its zero-result target drops every operand on its stack. *)
Lemma br_exit_target_dropping : forall z values rest,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values -> Forall wf_instr rest ->
  Steps (blockContext (stackConfig z values (BR (mk_uN 0) :: rest))) (stackConfig z [] []).
Proof.
  intros z values rest Hz Hvalues Hrest. apply one_spec_step.
  - apply block_context_wf. apply stack_config_wf; [exact Hz | exact Hvalues | ].
    constructor; [repeat constructor | exact Hrest].
  - repeat constructor; exact Hz.
  - apply pure. unfold blockContext, stackConfig. rewrite i32_constants_values.
    pose proof (br_zero 0%N [] (map i32Value values) [] (map admininstr_instr rest) Logic.eq_refl) as Hstep.
    rewrite !cat_app, app_nil_r, <- app_assoc in Hstep. exact Hstep.
Qed.

(* A return leaving a label carries every operand on its stack out of it. *)
Lemma return_label_carrying : forall z values rest,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values -> Forall wf_instr rest ->
  Steps (blockContext (stackConfig z values (RETURN :: rest))) (stackConfig z values [RETURN]).
Proof.
  intros z values rest Hz Hvalues Hrest. apply one_spec_step.
  - apply block_context_wf. apply stack_config_wf; [exact Hz | exact Hvalues | ].
    constructor; [constructor | exact Hrest].
  - apply stack_config_wf; [exact Hz | exact Hvalues | ]. repeat constructor.
  - apply pure. unfold blockContext, stackConfig. rewrite i32_constants_values.
    pose proof (return_label 0%N [] (map i32Value values) (map admininstr_instr rest)) as Hstep.
    rewrite !cat_app, <- app_assoc in Hstep. exact Hstep.
Qed.

(* A return reaching its function's frame: the function's label passes it on, and
   the frame keeps the results, dropping every operand below them. *)
Lemma i32_function_returns_early : forall s caller callee dropped results rest,
  wf_state (mk_state s caller) -> wf_frame callee ->
  Forall (fun value => 0 <= value < i32Modulus) (dropped ++ results) -> Forall wf_instr rest ->
  Steps (callContext caller (N.of_nat (length results))
      (stackConfig (mk_state s callee) (dropped ++ results) (RETURN :: rest)))
    (mk_config (mk_state s caller) (map i32Constant results)).
Proof.
  intros s caller callee dropped results rest Hstate Hcallee Hvalues Hrest.
  set (n := N.of_nat (length results)).
  assert (Hconstants : forall values, Forall (fun value => 0 <= value < i32Modulus) values ->
      Forall wf_admininstr (map i32Constant values)).
  { intros values Hv. apply Forall_map. eapply Forall_impl; [ | exact Hv].
    intros value Hvalue. constructor. apply spectec_i32_well_formed. exact Hvalue. }
  pose proof (Hconstants _ Hvalues) as Hall.
  pose proof (Hconstants _ (proj2 (proj1 (Forall_app _ _ _) Hvalues))) as Hkept.
  assert (HinnerState : wf_state (mk_state s callee)) by (inversion Hstate; constructor; assumption).
  set (returning := map i32Constant (dropped ++ results) ++ [admininstr_RETURN]).
  assert (Hreturning : Forall wf_admininstr returning) by (apply Forall_app; split; [exact Hall | repeat constructor]).
  set (body := map i32Constant (dropped ++ results) ++ map admininstr_instr (RETURN :: rest)).
  assert (Hbody : Forall wf_admininstr body).
  { apply Forall_app; split; [exact Hall | ]. apply Forall_map. constructor; [constructor | ].
    eapply Forall_impl; [ | exact Hrest]. intros i Hi. apply instruction_admin_wf. exact Hi. }
  assert (Hlabel : wf_config (mk_config (mk_state s callee) [LABEL_ n [] body]))
    by (constructor; [exact HinnerState | repeat constructor; assumption]).
  assert (HinnerReturning : wf_config (mk_config (mk_state s callee) returning)) by (constructor; assumption).
  assert (Hbefore : wf_config (mk_config (mk_state s caller) [FRAME_ n callee [LABEL_ n [] body]]))
    by (constructor; [exact Hstate | repeat constructor; assumption]).
  assert (Hmiddle : wf_config (mk_config (mk_state s caller) [FRAME_ n callee returning]))
    by (constructor; [exact Hstate | repeat constructor; assumption]).
  assert (Hafter : wf_config (mk_config (mk_state s caller) (map i32Constant results))) by (constructor; assumption).
  change (Steps (mk_config (mk_state s caller) [FRAME_ n callee [LABEL_ n [] body]])
    (mk_config (mk_state s caller) (map i32Constant results))).
  eapply trans; [ | | exact Hbefore | exact Hafter | exact Hmiddle].
  - apply ctxt_frame; [ | exact Hlabel | exact HinnerReturning].
    apply pure. unfold returning, body. rewrite i32_constants_values.
    pose proof (return_label n [] (map i32Value (dropped ++ results)) (map admininstr_instr rest)) as Hstep.
    rewrite !cat_app, <- app_assoc in Hstep. exact Hstep.
  - eapply trans; [ | apply Steps__refl; exact Hafter | exact Hmiddle | exact Hafter | exact Hafter].
    apply pure. unfold returning. rewrite map_app, !i32_constants_values.
    pose proof (return_frame n callee (map i32Value dropped) (map i32Value results) []) as Hstep.
    rewrite !cat_app, !app_nil_r in Hstep. apply Hstep.
    apply (introT eqP). unfold n. rewrite length_map. reflexivity.
Qed.
