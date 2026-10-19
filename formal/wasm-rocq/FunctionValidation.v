(* Generated function validation and the context used for its active body. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec SpecTecCalls StraightLineValidation.
Import ListNotations.

Definition functionContext (C : context) (f : funcinst) : context :=
  match funcinst_TYPE f, CODE f with
  | mk_functype (mk_list params) (mk_list results), func_FUNC _ locals _ =>
      _append_context
        (MKcontext [] [] [] [] [] [] []
          (params ++ map (fun l => match l with LOCAL t => t end) locals)
          [mk_list _ results] (Some (mk_list _ results))) C
  end.

(* SpecTec's BLOCK rule types the body with the block's label prepended. *)
Definition labelContext (results : list valtype) (C : context) : context :=
  _append_context (MKcontext [] [] [] [] [] [] [] [] [mk_list _ results] None) C.

Lemma label_context_locals : forall results C,
  context_LOCALS (labelContext results C) = context_LOCALS C.
Proof. reflexivity. Qed.

Lemma label_context_label : forall results C,
  nth_error (LABELS (labelContext results C)) 0 = Some (mk_list _ results).
Proof. reflexivity. Qed.

Lemma label_context_wf : forall results C, wf_context C -> wf_context (labelContext results C).
Proof. intros results C H. inversion H; subst. constructor; assumption. Qed.

Lemma module_context_locals : forall s m C,
  Moduleinst_ok s m C -> context_LOCALS C = [].
Proof. intros s m C H; inversion H; reflexivity. Qed.

Lemma module_context_wf : forall s m C,
  Moduleinst_ok s m C -> wf_context C.
Proof. intros s m C H; inversion H; assumption. Qed.

Lemma function_context_locals : forall C m ti p r l body,
  context_LOCALS C = [] ->
  context_LOCALS (functionContext C (i32Function m ti p r l body)) = repeat valtype_I32 (p + l).
Proof.
  intros. cbn [functionContext i32Function funcinst_TYPE CODE _append_context context_LOCALS].
  rewrite map_map. change ((repeat valtype_I32 p ++ map (fun x : valtype => x) (repeat valtype_I32 l)) ++ context_LOCALS C = repeat valtype_I32 (p + l)).
  rewrite map_id, H, app_nil_r, repeat_app. reflexivity.
Qed.

Lemma function_context_wf : forall C f, wf_context C -> wf_context (functionContext C f).
Proof.
  intros C [ [ [params] [results]] m [ti locals body]] H.
  inversion H; constructor; assumption.
Qed.

(* Func_ok types the body in exactly functionContext, from an empty stack to the results. *)
Lemma funcinst_body_validated : forall s m ti p r l body,
  Funcinst_ok s (i32Function m ti p r l body) (i32StackType p r) ->
  exists C, Moduleinst_ok s m C /\
    Instrs_ok (functionContext C (i32Function m ti p r l body)) body (i32StackType 0 r).
Proof.
  intros s m ti p r l body Hinst.
  inversion Hinst; subst. match goal with H : Func_ok _ _ _ |- _ => inversion H; subst end.
  assert (Ht : t_lst = repeat valtype_I32 l).
  { change (map LOCAL t_lst = map LOCAL (repeat valtype_I32 l)) in H0.
    apply (f_equal (map (fun x => match x with LOCAL t => t end))) in H0.
    rewrite !map_map in H0. change (map (fun x : valtype => x) t_lst = map (fun x : valtype => x) (repeat valtype_I32 l)) in H0.
    now rewrite !map_id in H0. }
  subst t_lst.
  match goal with H : Expr_ok _ _ _ |- _ => inversion H; subst end.
  exists C. split; [assumption | ].
  unfold functionContext, i32Function. cbn [funcinst_TYPE CODE]. rewrite map_map.
  change (map (fun x : valtype => x) (repeat valtype_I32 l)) with (map id (repeat valtype_I32 l)).
  rewrite map_id. assumption.
Qed.

Lemma extern_function_lookup : forall s a t,
  Externaddr_ok s a t ->
  match a with externaddr_FUNC address =>
    exists f, nth_error (store_FUNCS s) (N.to_nat address) = Some f /\ t = FUNC (funcinst_TYPE f)
  | _ => True end.
Proof.
  intros s a t H. induction H; try exact I.
  - exists v_funcinst. split; [ | reflexivity].
    apply (elimT eqP) in H0. apply N.ltb_lt in H.
    unfold lookup_total in H0. rewrite LocalSimulation.seq_nth_list_nth in H0.
    rewrite <- H0. apply nth_error_nth'.
    change (a < N.of_nat (length (store_FUNCS s)))%N in H. lia.
  - destruct v_externaddr; try exact I.
    destruct IHExternaddr_ok as [f [Hnth ->]]. inversion H0; subst.
    match goal with Hsub : Functype_sub _ _ |- _ => inversion Hsub; subst end.
    exists f; auto.
Qed.

Lemma forall2_lookup : forall (A B : Type) (R : A -> B -> Prop) xs ys,
  Forall2 R xs ys -> forall n x, nth_error xs n = Some x ->
    exists y, nth_error ys n = Some y /\ R x y.
Proof.
  intros A B R xs ys H. induction H; intros [ | n] value Hnth; try discriminate.
  - injection Hnth as <-. exists y; auto.
  - eapply IHForall2; exact Hnth.
Qed.

Lemma module_function_type : forall s m C index address f,
  Moduleinst_ok s m C -> nth_error (FUNCS m) index = Some address ->
  nth_error (store_FUNCS s) (N.to_nat address) = Some f ->
  nth_error (context_FUNCS C) index = Some (funcinst_TYPE f).
Proof.
  intros s m C index address f Hmodule Haddress Hfunction.
  inversion Hmodule; subst.
  match goal with Hp : Forall2 (fun _ _ => Externaddr_ok _ (externaddr_FUNC _) _) _ _ |- _ =>
    destruct (forall2_lookup _ _ _ _ _ Hp _ _ Haddress) as [ft [Htype Hextern]] end.
  destruct (extern_function_lookup _ _ _ Hextern) as [actual [Hactual Heq]].
  rewrite Hfunction in Hactual. injection Hactual as <-. injection Heq as ->. exact Htype.
Qed.
