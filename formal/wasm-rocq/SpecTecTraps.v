(* Trap propagation through operand, label and function contexts. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers Locals StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout Fragment ControlDecode ControlInvariant FrameFacts.
Import ListNotations.
Open Scope Z_scope.

Lemma trap_config_wf : forall z, wf_state z -> wf_config (mk_config z [admininstr_TRAP]).
Proof. intros; repeat constructor; assumption. Qed.

Lemma value_trap_discards_context : forall z values rest,
  wf_state z -> Forall wf_val values -> Forall wf_instr rest ->
  Steps (instructionSuffix rest (valueStackPrefix values (mk_config z [admininstr_TRAP])))
    (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  assert (Hwf : wf_config (instructionSuffix rest (valueStackPrefix values (mk_config z [admininstr_TRAP])))).
  { apply instruction_suffix_wf; [exact Hr | ]. apply value_stack_prefix_wf; [exact Hv | ]. apply trap_config_wf; exact Hz. }
  pose proof (trap_vals values (map admininstr_instr rest)) as Htrap.
  rewrite !cat_app in Htrap.
  destruct values as [ | v values], rest as [ | i rest].
  - apply steps_refl_wf. exact Hwf.
  all: apply one_spec_step; [exact Hwf | apply trap_config_wf; exact Hz | ].
  all: unfold instructionSuffix, valueStackPrefix; rewrite <- app_assoc; apply pure; apply Htrap; reflexivity.
Qed.

Lemma trap_discards_context : forall z values rest,
  wf_state z -> Forall (fun v => 0 <= v < i32Modulus) values -> Forall wf_instr rest ->
  Steps (instructionSuffix rest (stackPrefix values (mk_config z [admininstr_TRAP])))
    (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  pose proof (value_trap_discards_context z (map i32Value values) rest Hz (i32_values_wf values Hv) Hr) as H.
  rewrite value_stack_prefix_i32 in H. exact H.
Qed.

Lemma value_unreachable_steps : forall z values rest,
  wf_state z -> Forall wf_val values -> Forall wf_instr rest ->
  Steps (valueStackConfig z values (UNREACHABLE :: rest)) (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  eapply spec_steps_transitive with (b := valueStackPrefix values (instructionSuffix rest (mk_config z [admininstr_TRAP]))).
  - change (Steps (valueStackPrefix values (instructionSuffix rest (mk_config z [admininstr_UNREACHABLE])))
      (valueStackPrefix values (instructionSuffix rest (mk_config z [admininstr_TRAP])))).
    apply steps_with_value_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
    apply one_spec_step; [repeat constructor; assumption | apply trap_config_wf; exact Hz | ].
    apply pure. apply Step_pure__unreachable.
  - change (Steps (mk_config z (List.map admininstr_val values ++ ([admininstr_TRAP] ++ List.map admininstr_instr rest))) (mk_config z [admininstr_TRAP])).
    rewrite app_assoc. apply value_trap_discards_context; assumption.
Qed.

Lemma unreachable_steps : forall z values rest,
  wf_state z -> Forall (fun v => 0 <= v < i32Modulus) values -> Forall wf_instr rest ->
  Steps (stackConfig z values (UNREACHABLE :: rest)) (mk_config z [admininstr_TRAP]).
Proof.
  intros z values rest Hz Hv Hr.
  pose proof (value_unreachable_steps z (map i32Value values) rest Hz (i32_values_wf values Hv) Hr) as H.
  rewrite value_stack_config_i32 in H. exact H.
Qed.

Lemma label_traps : forall z arity continuation,
  wf_state z -> Forall wf_instr continuation ->
  Steps (blockContext arity continuation (mk_config z [admininstr_TRAP])) (mk_config z [admininstr_TRAP]).
Proof.
  intros. apply one_spec_step; [apply block_context_wf; [assumption | apply trap_config_wf; assumption] |
    apply trap_config_wf; assumption | ]. apply pure. apply trap_label.
Qed.

Lemma wrap_value_labels_trap : forall labels code segments z,
  ValueLabelsReady code labels segments -> wf_state z ->
  Steps (wrapValueLabels code labels segments (mk_config z [admininstr_TRAP])) (mk_config z [admininstr_TRAP]).
Proof.
  induction labels as [ | site labels IH]; intros code segments z Hready Hz; [apply steps_refl_wf; now apply trap_config_wf | ].
  destruct segments as [ | values segments]; [apply steps_refl_wf; now apply trap_config_wf | ].
  destruct Hready as [Hv [Hr Hinner]]. cbn [wrapValueLabels].
  destruct (siteEntry code site) as [[i body] | ]; [ | apply steps_refl_wf; now apply trap_config_wf].
  destruct Hinner as [Hcont Hinner].
  eapply spec_steps_transitive.
  - apply steps_with_value_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
    apply steps_in_block; [exact Hcont | ]. exact (IH _ _ _ Hinner Hz).
  - eapply spec_steps_transitive.
    + apply steps_with_value_prefix; [exact Hv | ]. apply steps_with_instruction_suffix; [exact Hr | ].
      apply label_traps; assumption.
    + cbn [valueStackPrefix instructionSuffix]. rewrite app_assoc. apply value_trap_discards_context; assumption.
Qed.

Lemma wrap_labels_trap : forall labels code segments z,
  LabelsReady code labels segments -> wf_state z ->
  Steps (wrapLabels code labels segments (mk_config z [admininstr_TRAP])) (mk_config z [admininstr_TRAP]).
Proof.
  intros labels code segments z Hready Hz.
  pose proof (wrap_value_labels_trap labels code (map (map i32Value) segments) z Hready Hz) as H.
  rewrite wrap_value_labels_i32 in H. exact H.
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
