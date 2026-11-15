(* Source programs for mixed i32/i64 witness tests. Validation uses SpecTec's rules. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers WasmValues SpecTecCalls CallerSpec FunctionValidation
  ProgramSpec ProgramEncoding Fragment CodeLayout.
Import ListNotations.
Open Scope Z_scope.

Definition signature inputs outputs := mk_functype (mk_list _ inputs) (mk_list _ outputs).

Lemma typed_cons : forall C i rest inputs middle outputs,
  Instrs_ok C [i] (signature inputs middle) -> Instrs_ok C rest (signature middle outputs) ->
  wf_context C -> wf_instr i -> Forall wf_instr rest ->
  Instrs_ok C (i :: rest) (signature inputs outputs).
Proof.
  intros. change (i :: rest) with ([i] ++ rest).
  eapply res_seq; eassumption || (constructor; assumption || constructor).
Qed.

Lemma typed_frame : forall C i prefix inputs outputs,
  Instr_ok C i (signature inputs outputs) -> wf_context C -> wf_instr i ->
  Instrs_ok C [i] (signature (prefix ++ inputs) (prefix ++ outputs)).
Proof.
  intros. apply Instrs_ok__frame; [apply Instrs_ok__instr; assumption | assumption | constructor; assumption || constructor].
Qed.

Ltac mixed_instr prefix inputs outputs :=
  apply (typed_frame _ _ (prefix : list valtype) (inputs : list valtype) (outputs : list valtype));
  [first [apply local_get | apply local_set | apply call]; try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor |
    repeat constructor | repeat constructor].
Ltac mixed_step prefix inputs outputs :=
  eapply typed_cons with (middle := (prefix : list valtype) ++ (outputs : list valtype));
  [mixed_instr prefix inputs outputs | | repeat constructor | repeat constructor | repeat constructor].

Module MixedCall.
Definition mainBody := [BLOCK (_RESULT (Some valtype_I64))
  [LOCAL_GET (mk_uN 0); BLOCK (_RESULT None)
    [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 1); LOCAL_SET (mk_uN 2)]];
  CALL (mk_uN 1)].
Definition middleBody := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 2);
  BLOCK (_RESULT None) [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 2); CALL (mk_uN 2); RETURN]];
  LOCAL_GET (mk_uN 2)].
Definition leafBody := [LOCAL_GET (mk_uN 0)].
Definition types := [signature [valtype_I64; valtype_I32] [valtype_I64]; signature [valtype_I64] [valtype_I64]].
Definition module := MKmoduleinst types [0%N; 1%N; 2%N] [] [] [] [] [] [].
Definition main := typedFunction module (mk_uN 0) [valtype_I64; valtype_I32] [valtype_I64] [valtype_I64] mainBody.
Definition middle := typedFunction module (mk_uN 1) [valtype_I64] [valtype_I64] [valtype_I32; valtype_I64] middleBody.
Definition leaf := typedFunction module (mk_uN 1) [valtype_I64] [valtype_I64] [] leafBody.
Definition program := [main; middle; leaf].
Definition store := MKstore program [] [] [] [] [].
Definition context := MKcontext types [signature [valtype_I64; valtype_I32] [valtype_I64];
  signature [valtype_I64] [valtype_I64]; signature [valtype_I64] [valtype_I64]] [] [] [] [] [] [] [] None.
Definition guest := mkGuestFunction 0%N module (mk_uN 0) [valtype_I64; valtype_I32] [valtype_I64] [valtype_I64] mainBody 0.

Lemma module_validates : Moduleinst_ok store module context.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  all: first [solve [eapply Externaddr_ok__func with (v_funcinst := main); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__func with (v_funcinst := middle); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__func with (v_funcinst := leaf); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]].
Qed.

Lemma main_validates : Instrs_ok (functionContext context main) mainBody (signature [] [valtype_I64]).
Proof.
  eapply typed_cons with (middle := [valtype_I64]); [ | mixed_instr (@nil valtype) [valtype_I64] [valtype_I64] | repeat constructor | repeat constructor | repeat constructor].
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] [valtype_I64]); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  mixed_step (@nil valtype) (@nil valtype) [valtype_I64].
  apply (typed_frame _ _ [valtype_I64] [] []); [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] []); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  mixed_step (@nil valtype) (@nil valtype) [valtype_I64]. mixed_step [valtype_I64] (@nil valtype) [valtype_I32].
  eapply typed_cons with (middle := [valtype_I64]); [ | mixed_instr (@nil valtype) [valtype_I64] (@nil valtype) | repeat constructor | repeat constructor | repeat constructor].
  apply Instrs_ok__instr; [apply (br_if _ _ [valtype_I64]) | | ]; try reflexivity; repeat constructor.
Qed.

Lemma middle_validates : Instrs_ok (functionContext context middle) middleBody (signature [] [valtype_I64]).
Proof.
  mixed_step (@nil valtype) (@nil valtype) [valtype_I64]. mixed_step (@nil valtype) [valtype_I64] (@nil valtype).
  eapply typed_cons with (middle := (@nil valtype)); [ | mixed_instr (@nil valtype) (@nil valtype) [valtype_I64] | repeat constructor | repeat constructor | repeat constructor].
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] []); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] []); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  mixed_step (@nil valtype) (@nil valtype) [valtype_I64]. mixed_step (@nil valtype) [valtype_I64] [valtype_I64].
  apply Instrs_ok__instr; [apply (res_return _ [] [valtype_I64] []) | | ];
    try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma leaf_validates : Instrs_ok (functionContext context leaf) leafBody (signature [] [valtype_I64]).
Proof. mixed_instr (@nil valtype) (@nil valtype) [valtype_I64]. Qed.

Ltac validate_function bodyProof :=
  apply mk_Funcinst_ok with (C := context);
  [repeat constructor | exact module_validates |
    apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity);
    [repeat constructor | apply mk_Expr_ok; [exact bodyProof | repeat constructor | repeat constructor] |
     repeat constructor | repeat constructor | repeat constructor] |
   repeat constructor | repeat constructor | repeat constructor].

Lemma validated : programValidated program store.
Proof.
  split.
  - constructor; [validate_function main_validates | ].
    constructor; [validate_function middle_validates | ]. constructor; [validate_function leaf_validates | constructor].
  - intros [ | [ | [ | n]]] f Hf; cbn in Hf; [injection Hf as <- .. | destruct n; discriminate].
    all: split; [ | vm_compute; intuition lia].
    all: repeat first [apply Forall_cons | apply Forall_nil | apply supported_local_get | apply supported_local_set |
      apply supported_return | apply supported_br_if | apply supported_block |
      solve [eapply supported_call; [reflexivity | cbn; lia | cbn; lia]] |
      solve [left; reflexivity] | solve [right; right; reflexivity]].
Qed.
End MixedCall.

Module MixedIndirect.
Definition mainBody := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); CALL_INDIRECT (mk_uN 0) (mk_uN 0)].
Definition leafBody := [LOCAL_GET (mk_uN 0)].
Definition types := [signature [valtype_I64] [valtype_I64]; signature [valtype_I64; valtype_I32] [valtype_I64]].
Definition module := MKmoduleinst types [0%N; 1%N] [] [0%N] [] [] [] [].
Definition main := typedFunction module (mk_uN 1) [valtype_I64; valtype_I32] [valtype_I64] [] mainBody.
Definition leaf := typedFunction module (mk_uN 0) [valtype_I64] [valtype_I64] [] leafBody.
Definition program := [main; leaf].
Definition tableType := mk_tabletype (mk_limits (mk_uN 1) None) FUNCREF.
Definition table := MKtableinst tableType [REF_FUNC_ADDR 1%N].
Definition store := MKstore program [] [table] [] [] [].
Definition context := MKcontext types [signature [valtype_I64; valtype_I32] [valtype_I64];
  signature [valtype_I64] [valtype_I64]] [] [tableType] [] [] [] [] [] None.
Definition guest := mkGuestFunction 0%N module (mk_uN 1) [valtype_I64; valtype_I32] [valtype_I64] [] mainBody 0.

Lemma module_validates : Moduleinst_ok store module context.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  all: first [solve [eapply Externaddr_ok__func with (v_funcinst := main); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__func with (v_funcinst := leaf); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__table with (v_tableinst := table); try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]].
Qed.
Lemma main_validates : Instrs_ok (functionContext context main) mainBody (signature [] [valtype_I64]).
Proof.
  mixed_step (@nil valtype) (@nil valtype) [valtype_I64]. mixed_step [valtype_I64] (@nil valtype) [valtype_I32].
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (call_indirect _ _ _ [valtype_I64] [valtype_I64] (mk_limits (mk_uN 1) None));
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.
Lemma leaf_validates : Instrs_ok (functionContext context leaf) leafBody (signature [] [valtype_I64]).
Proof. mixed_instr (@nil valtype) (@nil valtype) [valtype_I64]. Qed.
Ltac validate_function bodyProof :=
  apply mk_Funcinst_ok with (C := context);
  [ repeat constructor | exact module_validates |
    apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity);
    [repeat constructor | apply mk_Expr_ok; [exact bodyProof | repeat constructor | repeat constructor] |
     repeat constructor | repeat constructor | repeat constructor] |
   repeat constructor | repeat constructor | repeat constructor].
Lemma validated : programValidated program store.
Proof.
  split.
  - constructor; [validate_function main_validates | ].
    constructor; [validate_function leaf_validates | constructor].
  - intros [ | [ | n]] f Hf; cbn in Hf; [injection Hf as <- .. | destruct n; discriminate].
    + split; [ | vm_compute; intuition lia].
      repeat apply Forall_cons; try apply Forall_nil; try apply supported_local_get.
      apply (supported_call_indirect _ _ _ [valtype_I64] [valtype_I64]);
        [reflexivity | cbn; lia | cbn; lia | repeat apply Forall_cons; try apply Forall_nil; right; reflexivity | ].
      intros [ | [ | n]] fi Hfi; cbn in Hfi; [ | | destruct n; discriminate]; eexists; split; reflexivity.
    + split; [repeat constructor | vm_compute; intuition lia].
Qed.
End MixedIndirect.
