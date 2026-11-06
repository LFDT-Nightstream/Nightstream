(* The unwinding br_if run meets every premise of run_refinement, so the theorem's
   conclusion holds of an actual run that discards operands on a taken branch. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.UnwindingTrace
  tests.UnwindingRun.
Import ListNotations.
Open Scope Z_scope.

Definition unwindingContext := moduleContext [i32StackType 1 1] [i32StackType 1 1].
Definition unwindingBodyContext := functionContext unwindingContext unwindingFunction.
Definition unwindingGuest := mkGuestFunction 0%N unwindingModule (mk_uN 0) 1 1 1 unwindingCode 0.

Section Typing.
Local Open Scope nat_scope.
Example unwinding_code_validates : Instrs_ok unwindingBodyContext unwindingCode (i32StackType 0 1).
Proof.
  apply (instrs_cons_ok _ _ _ 0 0 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
    typed_step 0 1 0.
    apply (instrs_cons_ok _ _ _ 1 1 0); [repeat constructor | repeat constructor | repeat constructor | | ].
    + apply (i32_validation_frame _ _ 0 0 1); [ | repeat constructor | repeat constructor].
      apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
        repeat constructor | repeat constructor].
      change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
      typed_step 0 1 0. typed_step 0 1 1. typed_step 1 0 1. typed_instr 1 0 0.
    + typed_instr 1 0 0.
  - typed_instr 0 1 0.
Qed.
End Typing.

Example unwinding_module_validates : Moduleinst_ok unwindingStore unwindingModule unwindingContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := unwindingFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example unwinding_program_validated : programValidated unwindingProgram unwindingStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := unwindingContext).
    + repeat constructor.
    + exact unwinding_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact unwinding_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split; [repeat constructor | vm_compute; repeat split; lia].
Qed.

Definition unwindingStart := mkBoundary 0 2 2 0 0 0 0 0 0 0.

Lemma unwinding_run_valid :
  validCircuitRun unwindingMemory unwindingWitness (replayed unwindingMemory unwindingWitness).
Proof. exact (validRunb_sound _ _ unwinding_run_checks). Qed.

Lemma unwinding_fragment : runFragment unwindingStart 0 unwindingWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma unwinding_start_config :
  CircuitConfig unwindingProgram unwindingStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    unwindingMemory unwindingStart
    (stackConfig (mk_state unwindingStore (activeFrame unwindingGuest [7; 0])) [] unwindingCode).
Proof.
  apply (function_start_config unwindingProgram unwindingStore 0 unwindingGuest [7; 0] _ unwindingStart);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact unwinding_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example unwinding_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state unwindingStore (activeFrame unwindingGuest [7; 0])) [] unwindingCode) c /\
    CircuitConfig unwindingProgram unwindingStore 0 shape (replayed unwindingMemory unwindingWitness)
      (rowsFinalBoundary unwindingStart unwindingWitness) c.
Proof.
  intros Hprime.
  destruct (run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) unwinding_run_valid
    unwinding_fragment unwinding_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The run takes the branch: the br_if at pc 5 slides by 2, from sp 5 down to the
   outer block's base at 2, and resumes after the outer block's END at pc 10. *)
Example unwinding_branch_taken :
  let row := nth 5 unwindingWitness (rowWitness []) in
  row COL_SEL_BR_IF_EQZ = 1 /\ row COL_PC_BEFORE = 5 /\ row COL_PC_AFTER = 10 /\
  row COL_SP_BEFORE = 5 /\ row COL_SP_AFTER = 2 /\ row COL_FRAME_MOVE_DISTANCE_AFTER = 2.
Proof. vm_compute. repeat split; reflexivity. Qed.
