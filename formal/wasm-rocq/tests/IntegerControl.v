(* SpecTec control reductions with mixed operands. These test the control lemmas,
   not circuit support for i64 block results or the full run_refinement theorem. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues StraightLineSpec SpecTecCalls SpecTecBranches
  CodeLayout ControlDecode FrameFacts SpecTecTraps.
Import ListNotations.
Open Scope Z_scope.

Definition wide := i64Value 4294967305. (* low = 9, high = 1 *)
Definition testModule := MKmoduleinst [] [] [] [] [] [] [] [].
Definition testFrame := MKframe [] testModule.
Definition testStore := MKstore [] [] [] [] [] [].
Definition testState := mk_state testStore testFrame.

Lemma test_state_wf : wf_state testState.
Proof. repeat constructor. Qed.

Lemma wide_wf : wf_val wide.
Proof. apply i64_value_wf. unfold i64Modulus; lia. Qed.

Lemma narrow_wf : wf_val (i32Value 99).
Proof. apply i32_value_wf. unfold i32Modulus; lia. Qed.

(* A false condition leaves both the i64 and i32 values below it untouched. *)
Example false_branch_keeps_mixed_prefix :
  Steps (valueStackConfig testState [wide; i32Value 99; i32Value 0] [BR_IF (mk_uN 0)])
    (valueStackConfig testState [wide; i32Value 99] []).
Proof.
  exact (value_br_if_steps_below testState [wide; i32Value 99] 0 (mk_uN 0) []
    test_state_wf ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor])
    ltac:(unfold i32Modulus; lia) ltac:(repeat constructor) ltac:(constructor)).
Qed.

(* A true condition becomes BR and also preserves the mixed prefix. *)
Example true_branch_keeps_mixed_prefix :
  Steps (valueStackConfig testState [wide; i32Value 99; i32Value 1] [BR_IF (mk_uN 0)])
    (valueStackConfig testState [wide; i32Value 99] [BR (mk_uN 0)]).
Proof.
  exact (value_br_if_steps_below testState [wide; i32Value 99] 1 (mk_uN 0) []
    test_state_wf ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor])
    ltac:(unfold i32Modulus; lia) ltac:(repeat constructor) ltac:(constructor)).
Qed.

Definition innerBody := [BR (mk_uN 1); UNREACHABLE].
Definition outerBody := [BLOCK (_RESULT None) innerBody; UNREACHABLE].
Definition testLabels := [labelSite 0 0].
Definition segments := [[i32Value 99]].

Lemma outer_code_wf : Forall wf_instr outerBody.
Proof. repeat constructor. Qed.

Lemma test_labels_ready : ValueLabelsReady outerBody testLabels segments.
Proof.
  eapply value_labels_ready with (inner := innerBody).
  - exact outer_code_wf.
  - reflexivity.
  - reflexivity.
  - constructor; [exact narrow_wf | constructor].
Qed.

(* br 1 carries the wide value through the inner label, then the outer label
   drops the i32 temporary below it. Neither unreachable continuation executes. *)
Example nested_branch_keeps_wide_result :
  Steps (blockContext 1 []
    (wrapValueLabels outerBody testLabels segments (valueStackConfig testState [wide] innerBody)))
    (valueStackConfig testState [wide] []).
Proof.
  destruct (wrap_value_labels_exit testLabels outerBody innerBody segments testState [wide] 0 [UNREACHABLE]
    Logic.eq_refl Logic.eq_refl test_labels_ready test_state_wf
    ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor])
    ltac:(repeat constructor) ltac:(repeat constructor)) as [rest [Hrest Hsteps]].
  eapply spec_steps_transitive.
  - exact (steps_in_block 1 [] _ _ ltac:(constructor) Hsteps).
  - exact (value_br_exit_target [] testState [i32Value 99] [wide] rest
      ltac:(constructor) test_state_wf
      ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor]) Hrest).
Qed.

(* An early return carries mixed operands through two nested labels; the
   function frame then discards the temporaries and returns the i64 result. *)
Definition returnInner := [RETURN; UNREACHABLE].
Definition returnOuter := [BLOCK (_RESULT None) returnInner; UNREACHABLE].
Definition returnCode := [BLOCK (_RESULT None) returnOuter; UNREACHABLE].
Definition returnLabels := [labelSite 0 0; labelSite 0 0].
Definition returnSegments := [[wide]; [i32Value 99]].

Example nested_return_keeps_wide_result :
  Steps (callContext testFrame 1
    (wrapValueLabels returnCode returnLabels returnSegments
      (valueStackConfig testState [wide] returnInner)))
    (mk_config testState [admininstr_val wide]).
Proof.
  assert (Hready : ValueLabelsReady returnCode returnLabels returnSegments).
  { eapply value_labels_ready with (inner := returnInner); try reflexivity.
    - repeat constructor.
    - repeat first [exact wide_wf | exact narrow_wf | constructor]. }
  destruct (wrap_value_labels_return returnLabels returnCode returnInner returnSegments testState [wide]
    [UNREACHABLE] Logic.eq_refl Logic.eq_refl Hready test_state_wf
    ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor])
    ltac:(repeat constructor)) as [rest [Hrest Hsteps]].
  eapply spec_steps_transitive.
  - exact (steps_in_call_frame testFrame 1 _ _ ltac:(repeat constructor) Hsteps).
  - exact (value_function_returns_early testStore testFrame testFrame [wide; i32Value 99] [wide] rest
      test_state_wf ltac:(repeat constructor)
      ltac:(repeat first [exact wide_wf | exact narrow_wf | constructor]) Hrest).
Qed.

(* A loop's branch arity is zero here; normal exit can still leave an i64 result. *)
Example loop_end_keeps_wide_result :
  Steps (blockContext 0 [LOOP (_RESULT (Some valtype_I64)) []]
    (valueStackConfig testState [wide] [])) (valueStackConfig testState [wide] []).
Proof.
  apply value_exit_block; [repeat constructor | exact test_state_wf | ].
  constructor; [exact wide_wf | constructor].
Qed.

Example unreachable_discards_mixed_operands :
  Steps (valueStackConfig testState [wide; i32Value 99] [UNREACHABLE; NOP])
    (mk_config testState [admininstr_TRAP]).
Proof.
  apply value_unreachable_steps; [exact test_state_wf | | repeat constructor].
  repeat first [exact wide_wf | exact narrow_wf | constructor].
Qed.

Definition trapCode :=
  [BLOCK (_RESULT None) [BLOCK (_RESULT None) [UNREACHABLE; NOP]; NOP]; NOP].

(* Trapping discards the operands below both enclosing labels and skips their
   continuations, as well as discarding the innermost operand. *)
Example nested_trap_discards_mixed_operands :
  Steps (wrapValueLabels trapCode returnLabels returnSegments
    (valueStackConfig testState [wide] [UNREACHABLE; NOP]))
    (mk_config testState [admininstr_TRAP]).
Proof.
  assert (Hready : ValueLabelsReady trapCode returnLabels returnSegments).
  { eapply value_labels_ready with (inner := [UNREACHABLE; NOP]); try reflexivity.
    - repeat constructor.
    - repeat first [exact wide_wf | exact narrow_wf | constructor]. }
  eapply spec_steps_transitive.
  - apply steps_in_value_labels; [exact Hready | ].
    apply value_unreachable_steps; [exact test_state_wf | | repeat constructor].
    constructor; [exact wide_wf | constructor].
  - apply wrap_value_labels_trap; assumption || exact test_state_wf.
Qed.
