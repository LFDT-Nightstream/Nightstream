(* Trap propagation through operand, label and function contexts. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers Locals StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout Fragment ControlDecode ControlInvariant FrameFacts.
Import ListNotations.
Open Scope Z_scope.

Lemma trap_config_wf : forall z, wf_state z -> wf_config (mk_config z [admininstr_TRAP]).
Proof. intros; repeat constructor; assumption. Qed.

Lemma trap_discards_context : forall z values rest,
  wf_state z -> Forall (fun v => 0 <= v < i32Modulus) values -> Forall wf_instr rest ->
  Steps (instructionSuffix rest (stackPrefix values (mk_config z [admininstr_TRAP])))
    (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  assert (Hwf : wf_config (instructionSuffix rest (stackPrefix values (mk_config z [admininstr_TRAP])))).
  { apply instruction_suffix_wf; [exact Hr | ]. apply stack_prefix_wf; [exact Hv | ]. apply trap_config_wf; exact Hz. }
  pose proof (trap_vals (map i32Value values) (map admininstr_instr rest)) as Htrap.
  rewrite !cat_app, <- i32_constants_values in Htrap.
  destruct values as [ | v values], rest as [ | i rest].
  - apply steps_refl_wf. exact Hwf.
  all: apply one_spec_step; [exact Hwf | apply trap_config_wf; exact Hz | ].
  all: unfold instructionSuffix, stackPrefix; rewrite <- app_assoc; apply pure; apply Htrap; reflexivity.
Qed.

Lemma unreachable_steps : forall z values rest,
  wf_state z -> Forall (fun v => 0 <= v < i32Modulus) values -> Forall wf_instr rest ->
  Steps (stackConfig z values (UNREACHABLE :: rest)) (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  eapply spec_steps_transitive with (b := stackPrefix values (instructionSuffix rest (mk_config z [admininstr_TRAP]))).
  - change (Steps (stackPrefix values (instructionSuffix rest (mk_config z [admininstr_UNREACHABLE])))
      (stackPrefix values (instructionSuffix rest (mk_config z [admininstr_TRAP])))).
    apply steps_with_stack_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
    apply one_spec_step; [repeat constructor; assumption | apply trap_config_wf; exact Hz | ].
    apply pure. apply Step_pure__unreachable.
  - change (Steps (mk_config z (List.map i32Constant values ++ ([admininstr_TRAP] ++ List.map admininstr_instr rest))) (mk_config z [admininstr_TRAP])).
    rewrite app_assoc. apply trap_discards_context; assumption.
Qed.

Lemma label_traps : forall z arity continuation,
  wf_state z -> Forall wf_instr continuation ->
  Steps (blockContext arity continuation (mk_config z [admininstr_TRAP])) (mk_config z [admininstr_TRAP]).
Proof.
  intros. apply one_spec_step; [apply block_context_wf; [assumption | apply trap_config_wf; assumption] |
    apply trap_config_wf; assumption | ]. apply pure. apply trap_label.
Qed.

Lemma wrap_labels_trap : forall labels code segments z,
  LabelsReady code labels segments -> wf_state z ->
  Steps (wrapLabels code labels segments (mk_config z [admininstr_TRAP])) (mk_config z [admininstr_TRAP]).
Proof.
  induction labels as [ | site labels IH]; intros code segments z Hready Hz; [apply steps_refl_wf; now apply trap_config_wf | ].
  destruct segments as [ | values segments]; [apply steps_refl_wf; now apply trap_config_wf | ].
  destruct Hready as [Hv [Hr Hinner]]. cbn [wrapLabels].
  destruct (siteEntry code site) as [[i body] | ]; [ | apply steps_refl_wf; now apply trap_config_wf].
  destruct Hinner as [Hcont Hinner].
  eapply spec_steps_transitive.
  - apply steps_with_stack_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
    apply steps_in_block; [exact Hcont | ]. exact (IH _ _ _ Hinner Hz).
  - eapply spec_steps_transitive.
    + apply steps_with_stack_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
      apply label_traps; assumption.
    + cbn [stackPrefix instructionSuffix]. rewrite app_assoc. apply trap_discards_context; assumption.
Qed.

Lemma call_context_traps : forall s caller callee results,
  wf_state (mk_state s caller) -> wf_frame callee ->
  Steps (callContext caller results (mk_config (mk_state s callee) [admininstr_TRAP]))
    (mk_config (mk_state s caller) [admininstr_TRAP]).
Proof.
  intros s caller callee results Hcaller Hcallee.
  assert (Hinner : wf_config (mk_config (mk_state s callee) [admininstr_TRAP])).
  { apply trap_config_wf. inversion Hcaller; constructor; assumption. }
  eapply spec_steps_transitive with (b := mk_config (mk_state s caller) [FRAME_ results callee [admininstr_TRAP]]).
  - apply one_spec_step.
    + apply call_context_wf; [inversion Hcaller; assumption | exact Hinner].
    + constructor; [exact Hcaller | ]. constructor; [apply admininstr_case_72; [exact Hcallee | repeat constructor] | constructor].
    + apply ctxt_frame; [apply pure; apply trap_label | | exact Hinner].
      constructor; [inversion Hinner; assumption | repeat constructor].
  - apply one_spec_step; [ | apply trap_config_wf; exact Hcaller | ].
    + constructor; [exact Hcaller | ]. constructor; [apply admininstr_case_72; [exact Hcallee | repeat constructor] | constructor].
    + apply pure. apply trap_frame.
Qed.

Require Import CallRows.

(* A trap consumes every suspended caller, including its saved operands and
   continuation. The external caller of the activation remains. *)
Lemma plug_calls_trap : forall stack s caller callee arity,
  Forall SuspendedReady stack -> wf_state (mk_state s caller) -> wf_frame callee ->
  Steps (callContext caller arity (plugCalls stack (mk_config (mk_state s callee) [admininstr_TRAP])))
    (mk_config (mk_state s caller) [admininstr_TRAP]).
Proof.
  induction stack as [ | top rest IH]; intros s caller callee arity Hready Hz Hcallee.
  - apply call_context_traps; assumption.
  - inversion Hready as [ | ? ? [Hlabels [Hvalues [Hcode Hframe]]] Hrest]; subst.
    set (parent := activeFrame (callerFunction top) (viewLocals (callView top))).
    assert (Hparent : wf_state (mk_state s parent)) by (inversion Hz; constructor; assumption).
    assert (Hstep : Steps (callerConfig top (mk_config (mk_state s callee) [admininstr_TRAP]))
        (mk_config (mk_state s parent) [admininstr_TRAP])).
    { unfold callerConfig, frameConfig.
      eapply spec_steps_transitive.
      - apply steps_in_labels; [exact Hlabels | ].
        apply steps_with_stack_prefix; [exact Hvalues | ]. apply steps_with_instruction_suffix; [exact Hcode | ].
        apply call_context_traps; assumption.
      - eapply spec_steps_transitive.
        + apply steps_in_labels; [exact Hlabels | ].
          cbn [stackPrefix instructionSuffix]. rewrite app_assoc.
          apply trap_discards_context; assumption.
        + apply wrap_labels_trap; assumption. }
    cbn [plugCalls]. eapply spec_steps_transitive.
    + apply steps_in_call_frame; [inversion Hz; assumption | ]. exact (steps_in_plug_calls rest _ _ Hrest Hstep).
    + apply IH; assumption.
Qed.
