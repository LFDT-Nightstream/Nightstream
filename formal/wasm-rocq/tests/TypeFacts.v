(* Check the validation bridge on a mixed stack inside a block. The preserved
   outer operand and the inner operand are both i64; local 0 remains i32. *)
From Stdlib Require Import List NArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 CallerSpec CallerValidation FunctionValidation CodeLayout Fragment StackTypes TypeFacts CodeFacts.
Import ListNotations.

Definition mixedContext := MKcontext [] [] [] [] [] [] [] [valtype_I32; valtype_I64]
  [mk_list _ [valtype_I64]] (Some (mk_list _ [valtype_I64])).
Definition noCallees := calleeEnvironment (fun _ => None) [] [] True.
Definition transfer := [LOCAL_GET (mk_uN 1); LOCAL_SET (mk_uN 1)].
Definition nested := [LOCAL_GET (mk_uN 1); BLOCK (_RESULT None) transfer].

Lemma transfer_validates : forall C,
  context_LOCALS C = [valtype_I32; valtype_I64] -> wf_context C ->
  Instrs_ok C transfer (mk_functype (mk_list _ []) (mk_list _ [])).
Proof.
  intros C Hlocals HC.
  change (Instrs_ok C ([LOCAL_GET (mk_uN 1)] ++ [LOCAL_SET (mk_uN 1)])
    (mk_functype (mk_list _ []) (mk_list _ []))).
  eapply res_seq with (t_2_lst := [valtype_I64]).
  - apply Instrs_ok__instr; [apply local_get | exact HC | repeat constructor].
    + rewrite Hlocals. reflexivity.
    + rewrite Hlocals. reflexivity.
    + exact HC.
    + repeat constructor.
  - apply Instrs_ok__instr; [apply local_set | exact HC | repeat constructor].
    + rewrite Hlocals. reflexivity.
    + rewrite Hlocals. reflexivity.
    + exact HC.
    + repeat constructor.
  - exact HC.
  - repeat constructor.
  - repeat constructor.
Qed.

Lemma nested_validates :
  Instrs_ok mixedContext nested (mk_functype (mk_list _ []) (mk_list _ [valtype_I64])).
Proof.
  change (Instrs_ok mixedContext ([LOCAL_GET (mk_uN 1)] ++ [BLOCK (_RESULT None) transfer])
    (mk_functype (mk_list _ []) (mk_list _ [valtype_I64]))).
  eapply res_seq with (t_2_lst := [valtype_I64]).
  - apply Instrs_ok__instr; [apply local_get; try reflexivity | | ]; repeat constructor.
  - change (Instrs_ok mixedContext [BLOCK (_RESULT None) transfer]
      (mk_functype (mk_list _ ([valtype_I64] ++ [])) (mk_list _ ([valtype_I64] ++ [])))).
    apply Instrs_ok__frame.
    + apply Instrs_ok__instr.
      * apply block.
        -- apply Blocktype_ok__valtype; repeat constructor.
        -- apply transfer_validates; [reflexivity | repeat constructor].
        -- repeat constructor.
        -- unfold transfer; repeat constructor.
        -- repeat constructor.
      * repeat constructor.
      * unfold transfer; repeat constructor.
    + repeat constructor.
    + unfold transfer; repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - unfold transfer; repeat constructor.
Qed.

Lemma nested_supported : supportedCode noCallees nested.
Proof. unfold nested, transfer. repeat constructor. Qed.

(* The types are computed, but the continuation is validated by SpecTec. The
   outer i64 stays below the block; the inner i64 is consumed by local.set 1. *)
Example nested_i64_continuation_validates :
  Instrs_ok (labelContext [] mixedContext) [LOCAL_SET (mk_uN 1)]
    (mk_functype (mk_list _ [valtype_I64]) (mk_list _ [])).
Proof.
  assert (Hpath : pathTypes (context_LOCALS mixedContext) noCallees nested [labelSite 1 0] 1 =
    Some ([[valtype_I64]], [valtype_I64])) by (vm_compute; reflexivity).
  destruct (validated_path_types noCallees [labelSite 1 0] mixedContext [valtype_I64] nested 1
    [[valtype_I64]] [valtype_I64] nested_validates nested_supported
    ltac:(intros x f H; discriminate H) Logic.eq_refl ltac:(repeat constructor; discriminate)
    ltac:(intros x f H; discriminate H) Hpath)
    as [inner [current [outputs [Hinner [Hcontext [_ [_ Hvalid]]]]]]].
  cbn [pathCode siteEntry instrArm nested nth_error] in Hinner. injection Hinner as <-.
  cbn [pathValidationContext siteEntry instrArm nested nth_error structuredLabel structuredResults] in Hcontext.
  injection Hcontext as <- <-. exact Hvalid.
Qed.

(* A type-erasing height argument alone would not reject this stack: both
   candidate operands occupy one cell, but local.set 1 requires i64. *)
Example nested_i32_operand_rejected :
  ~ Instrs_ok (labelContext [] mixedContext) [LOCAL_SET (mk_uN 1)]
    (mk_functype (mk_list _ [valtype_I32]) (mk_list _ [])).
Proof.
  intros Hvalid.
  destruct (validation_concrete_head _ _ _ Hvalid _ [] Logic.eq_refl
    ltac:(repeat constructor; discriminate)) as [prefix [consumed [produced [Hinput [Hhead _]]]]].
  inversion Hhead; subst.
  match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H; cbn in H; subst end.
  pose proof (f_equal (@length valtype) Hinput) as Hlength.
  rewrite length_app in Hlength. cbn [length] in Hlength.
  assert (prefix = []) by (apply length_zero_iff_nil; lia).
  subst prefix. discriminate Hinput.
Qed.

(* The height proof no longer requires i32 locals, labels or return types. *)
Example mixed_context_height_facts :
  codeFacts 2 1 [1] noCallees 0 nested /\ endsAt noCallees nested 0 1.
Proof.
  assert (Hfits : contextFits mixedContext 2 1 [1] noCallees).
  { repeat split; try reflexivity. intros x f H; discriminate H. }
  destruct (code_validation_facts (S (codeSize nested)) _ _ _ nested_validates ltac:(lia)
    2 1 [1] noCallees Hfits nested_supported) as [Hfacts [Hend _]].
  split.
  - pose proof (Hfacts 0 [] I) as H. now rewrite app_nil_r in H.
  - exact (Hend 0).
Qed.
