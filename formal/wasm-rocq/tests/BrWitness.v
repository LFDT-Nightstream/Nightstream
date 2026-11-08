(* A real run with a br out of two blocks: main(7, 9) branches from inside a
   zero-result block to its enclosing result block, which keeps the result 9 and
   drops the 7 below it. The exported rows meet every premise of active_run_refinement,
   and the br's slide leaves the result to move down. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.BrTrace.
Import ListNotations.
Open Scope Z_scope.

(* block (result i32) (local.get 0; block (local.get 1; br 1)) *)
Definition brCode := [BLOCK (_RESULT (Some valtype_I32))
  [LOCAL_GET (mk_uN 0); BLOCK (_RESULT None) [LOCAL_GET (mk_uN 1); BR (mk_uN 1)]]].
Definition brModule := MKmoduleinst [i32StackType 2 1] [0%N] [] [] [] [] [] [].
Definition brFunction := i32Function brModule (mk_uN 0) 2 1 0 brCode.
Definition brProgram := [brFunction].
Definition brStore := MKstore brProgram [] [] [] [] [].
Definition brContext := moduleContext [i32StackType 2 1] [i32StackType 2 1].
Definition brGuest := mkGuestFunction 0%N brModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [] brCode 0.

Section Typing.
Local Open Scope nat_scope.
Example br_code_validates : Instrs_ok (functionContext brContext brFunction) brCode (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] [valtype_I32]); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).
  typed_step 0 1 0.
  apply (i32_validation_frame _ _ 0 0 1); [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
  typed_step 0 1 0.
  (* br 1 targets the result block: it takes the result and leaves nothing behind. *)
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (br _ _ [] [valtype_I32] []); try reflexivity; repeat constructor.
Qed.
End Typing.

Example br_module_validates : Moduleinst_ok brStore brModule brContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := brFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example br_program_validated : programValidated brProgram brStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := brContext).
    + repeat constructor.
    + exact br_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact br_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split; [ | vm_compute; repeat split; lia].
    constructor; [ | constructor]. apply supported_block; [ | right; reflexivity].
    constructor; [constructor | constructor; [ | constructor]].
    apply supported_block; [repeat constructor | left; reflexivity].
Qed.

(* The arguments 7 and 9, as the entry rows before the run left them. *)
Definition brMemory := loadProgram brProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 9 | [3] => Some 0 | _ => None end
  else None).
Definition brWitness := map rowWitness brRows.
Definition brStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma br_run_checks : validRunb brMemory brWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma br_fragment : runFragment brStart 0 brWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma br_start_config :
  CircuitConfig brProgram brStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    brMemory brStart (stackConfig (mk_state brStore (activeFrame brGuest [7; 9])) [] brCode).
Proof.
  apply (function_start_config brProgram brStore 0 brGuest [7; 9] _ brStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact br_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example br_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state brStore (activeFrame brGuest [7; 9])) [] brCode) c /\
    CircuitConfig brProgram brStore 0 shape (replayed brMemory brWitness)
      (rowsFinalBoundary brStart brWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ br_run_checks)
    br_fragment br_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The br at pc 4 keeps the result 9 and drops the 7 below it (sp 4 to 3), leaving
   the result one cell too high; the next row moves 9 from cell 3 down to cell 2,
   the result block's base. *)
Example br_slides :
  let br_row := nth 4 brWitness (rowWitness []) in
  let move_row := nth 5 brWitness (rowWitness []) in
  br_row COL_SEL_BR = 1 /\ br_row COL_PC_BEFORE = 4 /\ br_row COL_PC_AFTER = 7 /\
  br_row COL_SP_BEFORE = 4 /\ br_row COL_SP_AFTER = 3 /\
  br_row COL_FRAME_MOVE_REMAINING_AFTER = 1 /\ br_row COL_FRAME_MOVE_DISTANCE_AFTER = 1 /\
  move_row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 /\ move_row COL_FRAME_MOVE_REMAINING_AFTER = 0 /\
  move_row (List.nth 0 COL_STACK_READ_ADDR_LO 0%nat) = 2 * 3 /\
  move_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 9 /\
  move_row COL_STACK_WRITE0_ADDR_LO = 2 * 2 /\ move_row COL_STACK_WRITE0_VALUE_LO = 9.
Proof. vm_compute. repeat split; reflexivity. Qed.
