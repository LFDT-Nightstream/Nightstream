From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range TraceRefinement Wasm2 StraightLineSpec
  OpcodeCatalog MemoryTrace MemoryCatalog Locals StraightLineValidation.
Import ListNotations.
Open Scope Z_scope.

(* Start with an empty operand stack and two initialized i32 locals. *)
Definition localArithmetic :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1);
   BINOP I32 (mk_binop__0 Inn_I32 SUB);
   LOCAL_SET (mk_uN 0); LOCAL_GET (mk_uN 0)].

Example local_arithmetic_supported : supportedStraightLineProgram localArithmetic.
Proof.
  unfold supportedStraightLineProgram, localArithmetic. repeat constructor;
    eexists; reflexivity.
Qed.

Example local_arithmetic_no_initial_stack : programTyped 2 0 localArithmetic.
Proof. vm_compute. repeat constructor. Qed.

Local Ltac validation_wf :=
  match goal with
  | |- wf_context _ => repeat constructor
  | |- Forall _ _ => repeat constructor
  end.

Example local_arithmetic_validates :
  Instrs_ok (i32TypingContext 2) localArithmetic (i32StackType 0 1).
Proof.
  unfold localArithmetic, i32StackType.
  eapply res_seq with (instr_1_lst := [LOCAL_GET (mk_uN 0)])
    (t_2_lst := [valtype_I32]); try validation_wf.
  - apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - eapply res_seq with (instr_1_lst := [LOCAL_GET (mk_uN 1)])
      (t_2_lst := [valtype_I32; valtype_I32]); try validation_wf.
    + apply (Instrs_ok__frame _ _ [valtype_I32] [] [valtype_I32]); try validation_wf.
      apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + eapply res_seq with (instr_1_lst := [BINOP I32 (mk_binop__0 Inn_I32 SUB)])
        (t_2_lst := [valtype_I32]); try validation_wf.
      * apply Instrs_ok__instr; [apply binop | | ]; repeat constructor.
      * eapply res_seq with (instr_1_lst := [LOCAL_SET (mk_uN 0)])
          (t_2_lst := []); try validation_wf.
        -- apply Instrs_ok__instr; [apply local_set | | ]; try reflexivity; repeat constructor.
        -- apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
Qed.

(* Repeated writes to the same local are allowed; no readback-pair restriction. *)
Example overwrite_local_supported :
  programTyped 2 0
    [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1);
     LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1); LOCAL_GET (mk_uN 1)].
Proof. vm_compute. repeat constructor. Qed.

Example local_set_underflow_rejected : ~ programTyped 1 0 [LOCAL_SET (mk_uN 0)].
Proof. vm_compute. tauto. Qed.

Example nonexistent_local_rejected : ~ programTyped 1 0 [LOCAL_GET (mk_uN 1)].
Proof. vm_compute. lia. Qed.

Example unsupported_instruction_rejected : ~ supportedStraightLineProgram [NOP].
Proof.
  intros H. inversion H; subst. match goal with Hcode : exists _, _ |- _ =>
    destruct Hcode as [code Hcode]; discriminate Hcode end.
Qed.

Example binary_underflow_has_no_height :
  instructionStackHeight (BINOP I32 (mk_binop__0 Inn_I32 ADD)) 0 = None /\
  instructionStackHeight (BINOP I32 (mk_binop__0 Inn_I32 ADD)) 1 = None /\
  instructionStackHeight (BINOP I32 (mk_binop__0 Inn_I32 SUB)) 0 = None /\
  instructionStackHeight (BINOP I32 (mk_binop__0 Inn_I32 SUB)) 1 = None /\
  instructionStackHeight (LOCAL_SET (mk_uN 0)) 0 = None.
Proof. repeat split; reflexivity. Qed.

(* Constants and eqz are not in this fragment yet. They must not inherit the
   binary instructions' -1 delta or their two-operand typing rule. *)
Example unsupported_stack_effects_rejected : forall height,
  instructionStackHeight (CONST I32 (mk_num__0 Inn_I32 (mk_uN 0))) height = None /\
  instructionStackDelta (CONST I32 (mk_num__0 Inn_I32 (mk_uN 0))) = None /\
  instructionStackHeight (TESTOP I32 (mk_testop__0 Inn_I32 EQZ)) height = None /\
  instructionStackDelta (TESTOP I32 (mk_testop__0 Inn_I32 EQZ)) = None.
Proof. intros height. repeat split; reflexivity. Qed.

Example typing_rejects_unsupported_even_with_operands : ~ programTyped 0 2 [NOP].
Proof. exact (fun H => H). Qed.

Example body_delta_rejects_unsupported_instruction :
  bodyStackDelta [LOCAL_GET (mk_uN 0); NOP] = None.
Proof. reflexivity. Qed.
