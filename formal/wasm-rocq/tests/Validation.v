From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 StraightLineSpec StraightLineValidation.
Import ListNotations.
Open Scope Z_scope.

Example validation_rejects_underflow : forall results,
  ~ Instrs_ok (i32TypingContext 0) [BINOP I32 (mk_binop__0 Inn_I32 SUB)]
    (i32StackType 1 results).
Proof.
  intros results Hvalid.
  destruct (i32_validation_facts _ _ _ _ Hvalid
    ltac:(repeat constructor; eexists; reflexivity)) as [Htyped _].
  exact Htyped.
Qed.

Example validation_rejects_missing_local : forall results,
  ~ Instrs_ok (i32TypingContext 2) [LOCAL_GET (mk_uN 2)] (i32StackType 0 results).
Proof.
  intros results Hvalid.
  destruct (i32_validation_facts _ _ _ _ Hvalid
    ltac:(repeat constructor; eexists; reflexivity)) as [[Hindex _] _].
  cbn in Hindex. lia.
Qed.

(* Empty instruction sequences preserve an arbitrary stack through the frame
   rule; they do not require an empty initial stack. *)
Example empty_sequence_validates_with_stack :
  Instrs_ok (i32TypingContext 0) [] (i32StackType 2 2).
Proof.
  apply (Instrs_ok__frame _ _ [valtype_I32; valtype_I32] [] []); repeat constructor.
Qed.

(* The height bridge handles subtyping too, not just syntax-directed rules.
   BOT is SpecTec's bottom type, used by polymorphic stack validation. *)
Example subtraction_validates_with_bottom_inputs :
  Instrs_ok (i32TypingContext 0) [BINOP I32 (mk_binop__0 Inn_I32 SUB)]
    (mk_functype (mk_list _ [BOT; BOT]) (mk_list _ [valtype_I32])).
Proof.
  eapply sub with (t_1_lst := [valtype_I32; valtype_I32]) (t_2_lst := [valtype_I32]).
  - apply Instrs_ok__instr; [apply binop | | ]; repeat constructor.
  - constructor; [reflexivity | repeat constructor].
  - constructor; [reflexivity | repeat constructor].
  - repeat constructor.
  - repeat constructor.
Qed.

Require Import FunctionValidation SpecTecCalls ProgramSpec ProgramValidation
  CallerValidation CallerSpec tests.Calls tests.Callers tests.Programs.

(* Validation must reject a type table that disagrees with the instantiated
   function, even when its body has the right stack effect in isolation. *)
Example wrong_function_type_index_rejected :
  ~ Funcinst_ok (fun_store sourceInitialState)
      (i32Function callModule (mk_uN 1) 0 1 1 readZeroBody) (i32StackType 0 1).
Proof.
  intros H. inversion H; subst.
  match goal with Hmodule : Moduleinst_ok _ _ _ |- _ => inversion Hmodule; subst end.
  match goal with Hfunc : Func_ok _ _ _ |- _ => inversion Hfunc; subst end.
  match goal with Hbound : is_true (N.ltb _ _) |- _ => discriminate Hbound end.
Qed.

Example wrong_module_call_type_rejected : forall C,
  Moduleinst_ok (fun_store sourceInitialState) callModule C ->
  nth_error (context_FUNCS C) 0 = Some (i32StackType 1 0) -> False.
Proof.
  intros C Hmodule Hwrong.
  pose proof (module_function_type _ _ _ 0%nat 0%N readZeroFunction Hmodule eq_refl eq_refl) as Htype.
  rewrite Hwrong in Htype. discriminate.
Qed.

