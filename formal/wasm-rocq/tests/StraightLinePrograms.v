From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range TraceRefinement Wasm2 StraightLineSpec StraightLineRefinement
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

(* Concrete ROM contents, locals [20; 3] at frame base 7, and an empty stack.
   This checks that loading and representation can hold together; it does not
   construct a circuit witness or prove a validCircuitRun. *)
Definition localArithmeticMemory : MemoryState := fun memory address =>
  match address with
  | [pc] =>
      if pc <? 0 then None else
      if Nat.eqb memory programOpcodeId then
        nth_error [CODE_LOCAL_GET; CODE_LOCAL_GET; CODE_I32_SUB; CODE_LOCAL_SET; CODE_LOCAL_GET]
          (Z.to_nat pc)
      else if Nat.eqb memory pcEdgeKindId then
        nth_error [0; 0; 0; 0; 0] (Z.to_nat pc)
      else if Nat.eqb memory programLocalIndexId then
        List.nth (Z.to_nat pc) [Some 0; Some 1; None; Some 0; Some 0] None
      else None
  | [base; index] =>
      if Nat.eqb memory localLoId then
        match base, index with 7, 0 => Some 20 | 7, 1 => Some 3 | _, _ => None end
      else if Nat.eqb memory pcRomId then
        if (base <? 0) then None else
        if index =? 0 then nth_error [1; 2; 3; 4; 5] (Z.to_nat base) else None
      else None
  | _ => None
  end.

Definition localArithmeticState : state :=
  mk_state (MKstore [] [] [] [] [] [])
    (MKframe [i32Value 20; i32Value 3] (MKmoduleinst [] [] [] [] [] [] [] [])).

Example local_arithmetic_initial_state :
  loadedProgram localArithmetic localArithmeticMemory /\
  represents localArithmeticState localArithmeticMemory
    (mkBoundary 0 4 4 7 0 0 0 0 0) localArithmetic (stackConfig localArithmeticState [] localArithmetic).
Proof.
  split.
  - unfold loadedProgram. repeat split; intros pc;
      assert (Hnonnegative : (Z.of_nat pc <? 0) = false) by
        (apply Z.ltb_ge; lia).
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
    + intros choice. unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; cbn;
        destruct (choice =? 0); reflexivity.
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
  - exists []. split.
    + unfold stackMatches. cbn. repeat split; try lia; try constructor.
      intros [ | index] value H; discriminate.
    + split; [reflexivity | ]. split.
      * unfold stackConfig, localArithmeticState, localArithmetic, i32Value.
        cbn. repeat constructor.
      * intros [ | [ | index]] v H; cbn [stateLocals localArithmeticState nth_error] in H.
        -- inversion H; subst. exists 20. vm_compute. repeat split; congruence.
        -- inversion H; subst. exists 3. vm_compute. repeat split; congruence.
        -- destruct index; discriminate.
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

Example local_immediates_distinguished :
  programOpcodeAt localArithmetic 0 = programOpcodeAt localArithmetic 1 /\
  programLocalIndexAt localArithmetic 0 = Some 0 /\
  programLocalIndexAt localArithmetic 1 = Some 1 /\
  programLocalIndexAt localArithmetic 2 = None.
Proof. vm_compute. repeat constructor. Qed.

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

(* The straight-line corollary keeps the unexecuted source, just as a call prefix does. *)
Example local_arithmetic_prefix_refines : forall first second final,
  fieldPrime -> validCircuitRun localArithmeticMemory [first; second] final ->
  first COL_IS_PROGRAM_ROW = 1 -> second COL_IS_PROGRAM_ROW = 1 ->
  beforeBoundary first = mkBoundary 0 4 4 7 0 0 0 0 0 ->
  exists z c, Steps (stackConfig localArithmeticState [] localArithmetic) c /\
    programTyped (length (stateLocals z)) (stackHeight (afterBoundary second))
      [BINOP I32 (mk_binop__0 Inn_I32 SUB); LOCAL_SET (mk_uN 0); LOCAL_GET (mk_uN 0)] /\
    represents z final (afterBoundary second)
      [BINOP I32 (mk_binop__0 Inn_I32 SUB); LOCAL_SET (mk_uN 0); LOCAL_GET (mk_uN 0)] c /\
    second COL_PC_AFTER = 2.
Proof.
  intros first second final Hprime Hrun Hfirst Hsecond Hboundary.
  eapply (straight_line_run_refinement localArithmetic 1 localArithmeticState localArithmeticMemory final first [second]
    _ Hprime local_arithmetic_supported (proj1 local_arithmetic_initial_state) Hrun).
  - repeat constructor; assumption.
  - change (boundaryPC (beforeBoundary first) = 0). now rewrite Hboundary.
  - rewrite Hboundary. exact local_arithmetic_validates.
  - rewrite Hboundary. exact (proj2 local_arithmetic_initial_state).
Qed.
