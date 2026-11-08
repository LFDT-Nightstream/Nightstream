(* A real run with a br to the function's own label: main(7, 9) calls f(7, 9), whose
   br leaves its block and f, returning 9 and dropping the 7 below it, as a return
   would. The exported rows meet every premise of active_run_refinement. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.FunctionBrTrace.
Import ListNotations.
Open Scope Z_scope.

(* main: local.get 0; local.get 1; call 1. f: block (result i32) (local.get 0; local.get 1; br 1) *)
Definition mainBody := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); CALL (mk_uN 1)].
Definition branchBody := [BLOCK (_RESULT (Some valtype_I32)) [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BR (mk_uN 1)]].
Definition branchModule := MKmoduleinst [i32StackType 2 1] [0%N; 1%N] [] [] [] [] [] [].
Definition mainFunction := i32Function branchModule (mk_uN 0) 2 1 0 mainBody.
Definition branchFunction := i32Function branchModule (mk_uN 0) 2 1 0 branchBody.
Definition branchProgram := [mainFunction; branchFunction].
Definition branchStore := MKstore branchProgram [] [] [] [] [].
Definition mainGuest :=
  mkGuestFunction 0%N branchModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [] mainBody 0.

Definition branchContext := moduleContext [i32StackType 2 1] [i32StackType 2 1; i32StackType 2 1].

Example branch_module_validates : Moduleinst_ok branchStore branchModule branchContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := mainFunction);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := branchFunction);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma branch_function_validates : forall body, Forall wf_instr body ->
  Instrs_ok (functionContext branchContext (i32Function branchModule (mk_uN 0) 2 1 0 body)) body (i32StackType 0 1) ->
  Funcinst_ok branchStore (i32Function branchModule (mk_uN 0) 2 1 0 body) (i32StackType 2 1).
Proof.
  intros body Hw Hv. apply mk_Funcinst_ok with (C := branchContext); [repeat constructor | exact branch_module_validates | | ..].
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity);
      [repeat constructor | apply mk_Expr_ok; [exact Hv | repeat constructor | exact Hw] | repeat constructor |
       repeat constructor; exact Hw | repeat constructor].
  all: repeat constructor; exact Hw.
Qed.

Section Typing.
Local Open Scope nat_scope.
Example main_body_validates : Instrs_ok (functionContext branchContext mainFunction) mainBody (i32StackType 0 1).
Proof.
  typed_step 0 1 0. typed_step 0 1 1.
  apply Instrs_ok__instr; [apply call | | ]; try reflexivity; try (apply (introT eqP); reflexivity);
    repeat constructor.
Qed.

Example branch_body_validates :
  Instrs_ok (functionContext branchContext branchFunction) branchBody (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] [valtype_I32]); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).
  typed_step 0 1 0. typed_step 0 1 1.
  (* The br to the function's label keeps its result and drops the operand below. *)
  apply Instrs_ok__instr; [apply (br _ _ [valtype_I32] [valtype_I32] [valtype_I32]) | | ];
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.
End Typing.

Example branch_program_validated : programValidated branchProgram branchStore.
Proof.
  split.
  - constructor; [ | constructor; [ | constructor]].
    + apply branch_function_validates; [repeat constructor | exact main_body_validates].
    + apply branch_function_validates; [repeat constructor | exact branch_body_validates].
  - intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- .. | destruct index; discriminate].
    + split; [ | vm_compute; tauto]. constructor; [constructor | constructor; [constructor | constructor; [ | constructor]]].
      apply supported_call with (f :=
        mkGuestFunction 1%N branchModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [] branchBody 4);
        [reflexivity | cbn; lia | cbn; lia].
    + split.
      * constructor; [ | constructor]. apply supported_block; [repeat constructor | right; reflexivity].
      * vm_compute. repeat match goal with
          | |- _ /\ _ => split | |- Forall _ _ => constructor | |- True => exact I | |- _ => lia end.
Qed.

(* The arguments 7 and 9, as the entry rows before the run left them. *)
Definition branchMemory := loadProgram branchProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 9 | [3] => Some 0 | _ => None end
  else None).
Definition branchWitness := map rowWitness functionBrRows.
Definition branchStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma branch_run_checks : validRunb branchMemory branchWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma branch_fragment : runFragment branchStart 0 branchWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma branch_start_config :
  CircuitConfig branchProgram branchStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    branchMemory branchStart (stackConfig (mk_state branchStore (activeFrame mainGuest [7; 9])) [] mainBody).
Proof.
  apply (function_start_config branchProgram branchStore 0 mainGuest [7; 9] _ branchStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact branch_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

Example function_br_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state branchStore (activeFrame mainGuest [7; 9])) [] mainBody) c /\
    CircuitConfig branchProgram branchStore 0 shape (replayed branchMemory branchWitness)
      (rowsFinalBoundary branchStart branchWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ branch_run_checks)
    branch_fragment branch_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* f's br at pc 7 runs as a return: it pops back to main's pc 3, keeping its result
   and dropping the 7 below it and f's two parameters (sp 6 to 3), so the result
   moves down by 3. *)
Example function_br_returns :
  let br_row := nth 6 branchWitness (rowWitness []) in
  br_row COL_SEL_RETURN = 1 /\ br_row COL_CALL_STACK_POP_PRESENT = 1 /\
  br_row COL_PC_BEFORE = 7 /\ br_row COL_PC_AFTER = 3 /\
  br_row COL_SP_BEFORE = 6 /\ br_row COL_SP_AFTER = 3 /\ br_row COL_FRAME_MOVE_DISTANCE_AFTER = 3.
Proof. vm_compute. repeat split; reflexivity. Qed.
