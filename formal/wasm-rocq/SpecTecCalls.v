(* SpecTec's direct-call expansion and return through its frame/label contexts. *)
From Stdlib Require Import List ZArith NArith Lia.
From mathcomp Require Import ssreflect ssrbool eqtype seq.
Require Import Wasm2 WasmIntegers Locals Add StraightLineSpec LocalSimulation.
Import ListNotations.
Open Scope Z_scope.

Definition i32Function (module : moduleinst) (typeIndex : typeidx)
    (params results locals : nat) (body : list instr) : funcinst :=
  MKfuncinst (mk_functype (mk_list _ (repeat valtype_I32 params))
    (mk_list _ (repeat valtype_I32 results))) module
    (func_FUNC typeIndex (List.map LOCAL (repeat valtype_I32 locals)) body).

Definition enteredFrame (module : moduleinst) (arguments : list Z) (locals : nat) : frame :=
  MKframe (List.map i32Value arguments ++ repeat (i32Value 0) locals) module.

Lemma spec_lookup_from_nth_error : forall (A : Type) (inhabited : Inhabited A) values index result,
  nth_error values (N.to_nat index) = Some result ->
  @lookup_total A inhabited values index = result.
Proof.
  intros A inhabited values index result Hlookup.
  unfold lookup_total. rewrite seq_nth_list_nth.
  apply nth_error_nth. exact Hlookup.
Qed.

Lemma direct_call_expands : forall z x address,
  nth_error (fun_funcaddr z) (N.to_nat (x :> N)) = Some address ->
  Step (mk_config z [admininstr_CALL x]) (mk_config z [CALL_ADDR address]).
Proof.
  intros z x address Hlookup.
  replace address with (lookup_total (fun_funcaddr z) (x :> N))
    by (apply spec_lookup_from_nth_error; exact Hlookup).
  apply read. apply Step_read__call.
  apply N.ltb_lt. change ((x :> N) < N.of_nat (length (fun_funcaddr z)))%N.
  assert (Hlength : (N.to_nat (x :> N) < length (fun_funcaddr z))%nat).
  { apply nth_error_Some. rewrite Hlookup. discriminate. }
  lia.
Qed.

Lemma entered_frame_wf : forall module arguments locals,
  wf_moduleinst module -> Forall (fun value => 0 <= value < i32Modulus) arguments ->
  wf_frame (enteredFrame module arguments locals).
Proof.
  intros module arguments locals Hmodule Hargs.
  constructor; [ | exact Hmodule].
  apply Forall_app. split.
  - apply Forall_map. eapply Forall_impl; [ | exact Hargs].
    intros value Hvalue. apply val_case_0. apply spectec_i32_well_formed. exact Hvalue.
  - apply Forall_forall; intros v Hin; apply repeat_spec in Hin; subst v. apply val_case_0. apply spectec_i32_well_formed.
    unfold i32Modulus; lia.
Qed.

Lemma call_address_enters_i32_frame : forall z address module typeIndex arguments results locals body,
  nth_error (fun_funcinst z) (N.to_nat address) =
    Some (i32Function module typeIndex (length arguments) results locals body) ->
  wf_funcinst (i32Function module typeIndex (length arguments) results locals body) ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  Step (mk_config z (List.map i32Constant arguments ++ [CALL_ADDR address]))
    (mk_config z [FRAME_ (N.of_nat results) (enteredFrame module arguments locals)
      [LABEL_ (N.of_nat results) [] (List.map admininstr_instr body)]]).
Proof.
  intros z address module typeIndex arguments results locals body Hlookup Hwf Hargs.
  assert (Hmodule : wf_moduleinst module) by (inversion Hwf; assumption).
  assert (Hcode : wf_func (func_FUNC typeIndex (List.map LOCAL (repeat valtype_I32 locals)) body))
    by (inversion Hwf; assumption).
  assert (Hframe := entered_frame_wf module arguments locals Hmodule Hargs).
  assert (Hdefaults : seq.map (fun t => the (default_ t)) (repeat valtype_I32 locals) =
    repeat (i32Value 0) locals).
  { clear Hlookup Hwf Hcode Hframe. induction locals; [reflexivity | ]. cbn. rewrite IHlocals. reflexivity. }
  assert (Hvalues : List.map admininstr_val (List.map i32Value arguments) = List.map i32Constant arguments).
  { rewrite map_map. reflexivity. }
  rewrite <- Hvalues. apply read.
  eapply call_addr with (k := N.of_nat (length arguments))
    (t_1_lst := repeat valtype_I32 (length arguments))
    (t_2_lst := repeat valtype_I32 results)
    (t_lst := repeat valtype_I32 locals) (mm := module)
    (v_func := func_FUNC typeIndex (List.map LOCAL (repeat valtype_I32 locals)) body) (x := typeIndex).
  - apply N.ltb_lt. change (address < N.of_nat (length (fun_funcinst z)))%N.
    assert (Hlength : (N.to_nat address < length (fun_funcinst z))%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. } lia.
  - apply /eqP. apply spec_lookup_from_nth_error. exact Hlookup.
  - apply /eqP. reflexivity.
  - apply Forall_forall; intros v Hin; apply repeat_spec in Hin; subst v. reflexivity.
  - apply /eqP. unfold enteredFrame. rewrite Hdefaults. reflexivity.
  - exact Hwf.
  - exact Hcode.
  - rewrite Hdefaults. exact Hframe.
  - apply /eqP. change (N.of_nat (length arguments) = N.of_nat (length (List.map i32Value arguments))).
    rewrite length_map. reflexivity.
  - apply /eqP. change (N.of_nat (length arguments) = N.of_nat (length (repeat valtype_I32 (length arguments)))).
    rewrite repeat_length. reflexivity.
  - apply /eqP. change (N.of_nat results = N.of_nat (length (repeat valtype_I32 results))).
    rewrite repeat_length. reflexivity.
Qed.

Lemma spec_steps_transitive : forall a b c,
  Steps a b -> Steps b c -> Steps a c.
Proof.
  intros a b c Hab Hbc. induction Hab; [exact Hbc | ].
  inversion Hbc; subst; eapply trans; eauto.
Qed.

Definition callContext (caller : frame) (results : N) (c : config) : config :=
  match c with mk_config (mk_state s callee) code =>
    mk_config (mk_state s caller) [FRAME_ results callee [LABEL_ results [] code]]
  end.

Lemma call_context_wf : forall caller results c,
  wf_frame caller -> wf_config c -> wf_config (callContext caller results c).
Proof.
  intros caller results [[s callee] code] Hcaller Hwf.
  inversion Hwf; subst. match goal with H : wf_state _ |- _ => inversion H; subst end.
  constructor; [constructor; assumption | ]. repeat constructor; assumption.
Qed.

Lemma step_in_call_frame : forall caller results before after,
  wf_frame caller -> wf_config before -> wf_config after -> Step before after ->
  Step (callContext caller results before) (callContext caller results after).
Proof.
  intros caller results [[s callee] code] [[s' callee'] code'] Hcaller Hb Ha Hstep.
  cbn. apply ctxt_frame.
  - apply ctxt_label; assumption.
  - inversion Hb; subst. constructor; [assumption | ]. repeat constructor; assumption.
  - inversion Ha; subst. constructor; [assumption | ]. repeat constructor; assumption.
Qed.

Lemma steps_in_call_frame : forall caller results before after,
  wf_frame caller -> Steps before after ->
  Steps (callContext caller results before) (callContext caller results after).
Proof.
  intros caller results before after Hcaller Hsteps.
  induction Hsteps as [z code Hwf | z code z'' code'' z' code' Hstep Hsteps IH Hb Ha Hm].
  - destruct z. apply Steps__refl. exact (call_context_wf caller results _ Hcaller Hwf).
  - pose proof (step_in_call_frame caller results _ _ Hcaller Hb Hm Hstep) as Hlift.
    pose proof (call_context_wf caller results _ Hcaller Hb) as Hlb.
    pose proof (call_context_wf caller results _ Hcaller Ha) as Hla.
    pose proof (call_context_wf caller results _ Hcaller Hm) as Hlm.
    destruct z, z', z''. eapply trans; eassumption.
Qed.

Lemma i32_function_returns : forall s caller callee values,
  wf_state (mk_state s caller) -> wf_frame callee ->
  Forall (fun value => 0 <= value < i32Modulus) values ->
  Steps (mk_config (mk_state s caller)
    [FRAME_ (N.of_nat (length values)) callee
      [LABEL_ (N.of_nat (length values)) [] (List.map i32Constant values)]])
    (mk_config (mk_state s caller) (List.map i32Constant values)).
Proof.
  intros s caller callee values Hstate Hcallee Hvalues.
  assert (Hconstants : Forall wf_admininstr (List.map i32Constant values)).
  { apply Forall_map. eapply Forall_impl; [ | exact Hvalues].
    intros value Hvalue. constructor. apply spectec_i32_well_formed. exact Hvalue. }
  assert (Hmap : List.map admininstr_val (List.map i32Value values) = List.map i32Constant values).
  { rewrite map_map. reflexivity. }
  assert (HinnerState : wf_state (mk_state s callee)) by (inversion Hstate; constructor; assumption).
  assert (Hlabel : wf_config (mk_config (mk_state s callee)
    [LABEL_ (N.of_nat (length values)) [] (List.map i32Constant values)])).
  { constructor; [exact HinnerState | ]. repeat constructor; assumption. }
  assert (HinnerValues : wf_config (mk_config (mk_state s callee) (List.map i32Constant values)))
    by (constructor; assumption).
  assert (Hbefore : wf_config (mk_config (mk_state s caller)
    [FRAME_ (N.of_nat (length values)) callee
      [LABEL_ (N.of_nat (length values)) [] (List.map i32Constant values)]])).
  { constructor; [exact Hstate | ]. repeat constructor; assumption. }
  assert (Hmiddle : wf_config (mk_config (mk_state s caller)
    [FRAME_ (N.of_nat (length values)) callee (List.map i32Constant values)])).
  { constructor; [exact Hstate | ]. repeat constructor; assumption. }
  assert (Hafter : wf_config (mk_config (mk_state s caller) (List.map i32Constant values)))
    by (constructor; assumption).
  eapply trans; [ | | exact Hbefore | exact Hafter | exact Hmiddle].
  - apply ctxt_frame; [ | exact Hlabel | exact HinnerValues].
    apply pure. rewrite <- Hmap. apply label_vals.
  - eapply trans; [ | apply Steps__refl; exact Hafter | exact Hmiddle | exact Hafter | exact Hafter].
    apply pure. rewrite <- Hmap. apply frame_vals. apply /eqP.
    change (N.of_nat (length values) = N.of_nat (length (List.map i32Value values))).
    rewrite length_map. reflexivity.
Qed.

Lemma instruction_admin_wf : forall instruction,
  wf_instr instruction -> wf_admininstr (admininstr_instr instruction).
Proof. intros instruction H; destruct H; cbn; solve [constructor | econstructor; eassumption]. Qed.

Lemma entered_body_wf : forall s caller module typeIndex arguments results locals body,
  wf_state (mk_state s caller) ->
  wf_funcinst (i32Function module typeIndex (length arguments) results locals body) ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  wf_config (stackConfig (mk_state s (enteredFrame module arguments locals)) [] body).
Proof.
  intros s caller module typeIndex arguments results locals body Hstate Hfunc Hargs.
  assert (Hmodule : wf_moduleinst module) by (inversion Hfunc; assumption).
  assert (Hcode : wf_func (func_FUNC typeIndex (List.map LOCAL (repeat valtype_I32 locals)) body))
    by (inversion Hfunc; assumption).
  inversion Hcode; subst. constructor.
  - inversion Hstate; subst. constructor; [assumption | ]. apply entered_frame_wf; assumption.
  - apply Forall_map. eapply Forall_impl; [ | eassumption]. intros instruction Hwf.
    apply instruction_admin_wf. exact Hwf.
Qed.

Lemma direct_i32_call_enters : forall s caller index address module typeIndex arguments results locals body,
  wf_state (mk_state s caller) -> wf_uN 32%N index ->
  nth_error (fun_funcaddr (mk_state s caller)) (N.to_nat (index :> N)) = Some address ->
  nth_error (fun_funcinst (mk_state s caller)) (N.to_nat address) =
    Some (i32Function module typeIndex (length arguments) results locals body) ->
  wf_funcinst (i32Function module typeIndex (length arguments) results locals body) ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  Steps (stackConfig (mk_state s caller) arguments [CALL index])
    (callContext caller (N.of_nat results)
      (stackConfig (mk_state s (enteredFrame module arguments locals)) [] body)).
Proof.
  intros s caller index address module typeIndex arguments results locals body
    Hstate Hindex Haddress Hfunction Hfunc Hargs.
  assert (Hconstants : Forall wf_admininstr (List.map i32Constant arguments)).
  { apply Forall_map. eapply Forall_impl; [ | exact Hargs].
    intros value Hvalue. apply StackSimulation.i32_constant_wf. exact Hvalue. }
  assert (HsmallBefore : wf_config (mk_config (mk_state s caller) [admininstr_CALL index]))
    by (constructor; [exact Hstate | repeat constructor; assumption]).
  assert (HsmallAfter : wf_config (mk_config (mk_state s caller) [CALL_ADDR address]))
    by (constructor; [exact Hstate | repeat constructor; assumption]).
  assert (Hbefore : wf_config (stackConfig (mk_state s caller) arguments [CALL index])).
  { constructor; [exact Hstate | ]. apply Forall_app. split; [exact Hconstants | repeat constructor; assumption]. }
  assert (Hmiddle : wf_config (mk_config (mk_state s caller)
      (List.map i32Constant arguments ++ [CALL_ADDR address]))).
  { constructor; [exact Hstate | ]. apply Forall_app. split; [exact Hconstants | repeat constructor; assumption]. }
  assert (Hafter : wf_config (callContext caller (N.of_nat results)
      (stackConfig (mk_state s (enteredFrame module arguments locals)) [] body))).
  { apply call_context_wf; [inversion Hstate; assumption | ].
    eapply entered_body_wf; eassumption. }
  assert (Hstep : Step (stackConfig (mk_state s caller) arguments [CALL index])
      (mk_config (mk_state s caller) (List.map i32Constant arguments ++ [CALL_ADDR address]))).
  { destruct arguments as [ | value arguments].
    - apply direct_call_expands. exact Haddress.
    - assert (Hmap : List.map admininstr_val (List.map i32Value (value :: arguments)) =
          List.map i32Constant (value :: arguments)) by (rewrite map_map; reflexivity).
      unfold stackConfig. rewrite <- Hmap.
      change (Step (mk_config (mk_state s caller)
        (List.map admininstr_val (List.map i32Value (value :: arguments)) ++ [admininstr_CALL index]))
        (mk_config (mk_state s caller)
        (List.map admininstr_val (List.map i32Value (value :: arguments)) ++ [CALL_ADDR address]))).
      replace [admininstr_CALL index] with ([admininstr_CALL index] ++ []) by reflexivity.
      replace [CALL_ADDR address] with ([CALL_ADDR address] ++ []) by reflexivity.
      apply ctxt_instrs; [apply direct_call_expands; exact Haddress | reflexivity | exact HsmallBefore | exact HsmallAfter]. }
  cbn [callContext stackConfig] in Hafter |- *.
  eapply trans; [exact Hstep | | exact Hbefore | exact Hafter | exact Hmiddle].
  eapply trans; [ | apply Steps__refl; exact Hafter | exact Hmiddle | exact Hafter | exact Hafter].
  eapply call_address_enters_i32_frame; eassumption.
Qed.

Definition stackPrefix (values : list Z) (c : config) : config :=
  match c with mk_config z code => mk_config z (List.map i32Constant values ++ code) end.

Lemma stack_prefix_wf : forall values c,
  Forall (fun value => 0 <= value < i32Modulus) values ->
  wf_config c -> wf_config (stackPrefix values c).
Proof.
  intros values [z code] Hvalues Hwf. inversion Hwf; subst.
  constructor; [assumption | ]. apply Forall_app. split; [ | assumption].
  apply Forall_map. eapply Forall_impl; [ | exact Hvalues].
  intros value Hv. apply StackSimulation.i32_constant_wf. exact Hv.
Qed.

Lemma steps_with_stack_prefix : forall values before after,
  Forall (fun value => 0 <= value < i32Modulus) values -> Steps before after ->
  Steps (stackPrefix values before) (stackPrefix values after).
Proof.
  intros values before after Hvalues Hsteps.
  induction Hsteps as [z code Hwf | z code z'' code'' z' code' Hstep Hsteps IH Hb Ha Hm].
  - apply Steps__refl. exact (stack_prefix_wf values _ Hvalues Hwf).
  - assert (Hlift : Step (stackPrefix values (mk_config z code)) (stackPrefix values (mk_config z' code'))).
    { destruct values as [ | value values]; [exact Hstep | ].
      assert (Hmap : List.map admininstr_val (List.map i32Value (value :: values)) =
          List.map i32Constant (value :: values)) by (rewrite map_map; reflexivity).
      cbn [stackPrefix]. rewrite <- Hmap.
      rewrite <- (app_nil_r code), <- (app_nil_r code').
      apply ctxt_instrs; [exact Hstep | reflexivity | exact Hb | exact Hm]. }
    eapply trans; [exact Hlift | exact IH | | | ].
    + exact (stack_prefix_wf values _ Hvalues Hb).
    + exact (stack_prefix_wf values _ Hvalues Ha).
    + exact (stack_prefix_wf values _ Hvalues Hm).
Qed.

(* A reduction may execute before a fixed, well-formed instruction continuation. *)
Definition instructionSuffix (code : list instr) (c : config) : config :=
  match c with mk_config z instructions =>
    mk_config z (instructions ++ List.map admininstr_instr code)%list
  end.

Lemma instruction_suffix_wf : forall code c,
  Forall wf_instr code -> wf_config c -> wf_config (instructionSuffix code c).
Proof.
  intros code [z instructions] Hcode Hwf. inversion Hwf; subst.
  constructor; [assumption | ]. apply Forall_app. split; [assumption | ].
  apply Forall_map. eapply Forall_impl; [ | exact Hcode].
  intros instruction H. apply instruction_admin_wf. exact H.
Qed.

Lemma steps_with_instruction_suffix : forall code before after,
  Forall wf_instr code -> Steps before after ->
  Steps (instructionSuffix code before) (instructionSuffix code after).
Proof.
  intros code before after Hcode Hsteps.
  induction Hsteps as [z instructions Hwf | z instructions z'' instructions'' z' instructions' Hstep Hsteps IH Hb Ha Hm].
  - apply Steps__refl. exact (instruction_suffix_wf code _ Hcode Hwf).
  - assert (Hlift : Step (instructionSuffix code (mk_config z instructions))
        (instructionSuffix code (mk_config z' instructions'))).
    { destruct code as [ | instruction code].
      - change (Step (mk_config z (instructions ++ [])%list) (mk_config z' (instructions' ++ [])%list)).
        rewrite !app_nil_r. exact Hstep.
      - change (Step (mk_config z (List.map admininstr_val [] ++
          (instructions ++ List.map admininstr_instr (instruction :: code))))
          (mk_config z' (List.map admininstr_val [] ++
          (instructions' ++ List.map admininstr_instr (instruction :: code))))).
        apply ctxt_instrs; [exact Hstep | reflexivity | exact Hb | exact Hm]. }
    eapply trans; [exact Hlift | exact IH | | | ].
    + exact (instruction_suffix_wf code _ Hcode Hb).
    + exact (instruction_suffix_wf code _ Hcode Ha).
    + exact (instruction_suffix_wf code _ Hcode Hm).
Qed.
