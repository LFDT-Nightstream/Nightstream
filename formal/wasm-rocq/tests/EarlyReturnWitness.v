(* A real run with a return from inside a block: main(7) calls f(7), whose return
   drops the operand below its result. The exported rows meet every premise of
   run_refinement, and the run returns through a slide that moves the result down. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.EarlyReturnTrace.
Import ListNotations.
Open Scope Z_scope.

(* main: local.get 0; call 1. f: block (local.get 1; local.get 0; return); local.get 1 *)
Definition mainBody := [LOCAL_GET (mk_uN 0); CALL (mk_uN 1)].
Definition returnBody := [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 1); LOCAL_GET (mk_uN 0); RETURN]; LOCAL_GET (mk_uN 1)].
Definition returnModule := MKmoduleinst [i32StackType 1 1] [0%N; 1%N] [] [] [] [] [] [].
Definition mainFunction := i32Function returnModule (mk_uN 0) 1 1 0 mainBody.
Definition returningFunction := i32Function returnModule (mk_uN 0) 1 1 1 returnBody.
Definition returnProgram := [mainFunction; returningFunction].
Definition returnStore := MKstore returnProgram [] [] [] [] [].
Definition mainGuest := mkGuestFunction 0%N returnModule (mk_uN 0) 1 1 0 mainBody 0.

Definition returnContext := moduleContext [i32StackType 1 1] [i32StackType 1 1; i32StackType 1 1].

Example return_module_validates : Moduleinst_ok returnStore returnModule returnContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := mainFunction);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := returningFunction);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma return_function_validates : forall locals body, (locals <= 1)%nat ->
  Forall wf_instr body ->
  Instrs_ok (functionContext returnContext (i32Function returnModule (mk_uN 0) 1 1 locals body)) body (i32StackType 0 1) ->
  Funcinst_ok returnStore (i32Function returnModule (mk_uN 0) 1 1 locals body) (i32StackType 1 1).
Proof.
  intros locals body Hlocals Hw Hv. destruct locals as [ | [ | locals]]; [ | | lia].
  all: apply mk_Funcinst_ok with (C := returnContext).
  all: try exact return_module_validates.
  all: try (apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity);
    [repeat constructor | apply mk_Expr_ok; [exact Hv | repeat constructor | exact Hw] | repeat constructor |
     repeat constructor; exact Hw | repeat constructor]).
  all: repeat constructor; exact Hw.
Qed.

Section Typing.
Local Open Scope nat_scope.
Example main_body_validates : Instrs_ok (functionContext returnContext mainFunction) mainBody (i32StackType 0 1).
Proof.
  typed_step 0 1 0.
  apply Instrs_ok__instr; [apply call | | ]; try reflexivity; try (apply (introT eqP); reflexivity);
    repeat constructor.
Qed.

Example return_body_validates :
  Instrs_ok (functionContext returnContext returningFunction) returnBody (i32StackType 0 1).
Proof.
  apply (instrs_cons_ok _ _ _ 0 0 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
    typed_step 0 1 0. typed_step 0 1 1.
    apply Instrs_ok__instr; [apply (res_return _ [valtype_I32] [valtype_I32] []) | | ];
      try (apply (introT eqP); reflexivity); repeat constructor.
  - typed_instr 0 1 0.
Qed.
End Typing.

Example return_program_validated : programValidated returnProgram returnStore.
Proof.
  split.
  - constructor; [ | constructor; [ | constructor]].
    + apply return_function_validates; [lia | repeat constructor | exact main_body_validates].
    + apply return_function_validates; [lia | repeat constructor | exact return_body_validates].
  - intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- .. | destruct index; discriminate].
    + split; [ | vm_compute; tauto]. constructor; [constructor | constructor; [ | constructor]].
      apply supported_call with (f := mkGuestFunction 1%N returnModule (mk_uN 0) 1 1 1 returnBody 3);
        [reflexivity | cbn; lia | cbn; lia].
    + split; [ | vm_compute; tauto]. repeat constructor.
Qed.

(* The argument 7, as the entry rows before the run left it. *)
Definition returnMemory := loadProgram returnProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with [0] => Some 7 | [1] => Some 0 | _ => None end
  else None).
Definition returnWitness := map rowWitness earlyReturnRows.
Definition returnStart := mkBoundary 0 1 1 0 0 0 0 0 0 0.

Lemma return_run_checks : validRunb returnMemory returnWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma return_fragment : runFragment returnStart 0 returnWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma return_start_config :
  CircuitConfig returnProgram returnStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    returnMemory returnStart (stackConfig (mk_state returnStore (activeFrame mainGuest [7])) [] mainBody).
Proof.
  apply (function_start_config returnProgram returnStore 0 mainGuest [7] _ returnStart);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact return_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | i] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | destruct i; discriminate].
  - repeat split.
Qed.

Example early_return_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state returnStore (activeFrame mainGuest [7])) [] mainBody) c /\
    CircuitConfig returnProgram returnStore 0 shape (replayed returnMemory returnWitness)
      (rowsFinalBoundary returnStart returnWitness) c.
Proof.
  intros Hprime.
  destruct (run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ return_run_checks)
    return_fragment return_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The run returns through a slide: f's return at pc 6 pops back to main's pc 2,
   keeping its result and dropping f's two locals and the operand below the result
   (sp 5 to 2), and the next row moves the result, 7, from f's top cell 4 down to
   main's operand cell 1; the dropped 0 is not moved. *)
Example early_return_slides :
  let return_row := nth 6 returnWitness (rowWitness []) in
  let move_row := nth 7 returnWitness (rowWitness []) in
  return_row COL_SEL_RETURN = 1 /\ return_row COL_CALL_STACK_POP_PRESENT = 1 /\
  return_row COL_PC_BEFORE = 6 /\ return_row COL_PC_AFTER = 2 /\
  return_row COL_SP_BEFORE = 5 /\ return_row COL_SP_AFTER = 2 /\
  return_row COL_FRAME_MOVE_REMAINING_AFTER = 1 /\ return_row COL_FRAME_MOVE_DISTANCE_AFTER = 3 /\
  move_row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 /\ move_row COL_FRAME_MOVE_REMAINING_AFTER = 0 /\
  move_row (List.nth 0 COL_STACK_READ_ADDR_LO 0%nat) = 2 * 4 /\
  move_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 7 /\
  move_row COL_STACK_WRITE0_ADDR_LO = 2 * 1 /\ move_row COL_STACK_WRITE0_VALUE_LO = 7.
Proof. vm_compute. repeat split; reflexivity. Qed.
