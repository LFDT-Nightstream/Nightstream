(* A real run with two br_tables: main(7, 1) takes arm 1 out of a zero-result
   block, then, with index 7 past the arms, takes the default out of the result
   block, which keeps the result 1 and drops the 7 below it. The exported rows meet
   every premise of active_run_refinement. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.BrTableTrace.
Require Import ControlInvariant.
Import ListNotations.
Open Scope Z_scope.

(* block (result i32) (local.get 0; block (local.get 1; br_table 0 0 0); local.get 1; local.get 0;
   br_table 0 0) *)
Definition tableCode := [BLOCK (_RESULT (Some valtype_I32))
  [LOCAL_GET (mk_uN 0);
   BLOCK (_RESULT None) [LOCAL_GET (mk_uN 1); BR_TABLE [mk_uN 0; mk_uN 0] (mk_uN 0)];
   LOCAL_GET (mk_uN 1); LOCAL_GET (mk_uN 0); BR_TABLE [mk_uN 0] (mk_uN 0)]].
Definition tableModule := MKmoduleinst [i32StackType 2 1] [0%N] [] [] [] [] [] [].
Definition tableFunction := i32Function tableModule (mk_uN 0) 2 1 0 tableCode.
Definition tableProgram := [tableFunction].
Definition tableStore := MKstore tableProgram [] [] [] [] [].
Definition tableContext := moduleContext [i32StackType 2 1] [i32StackType 2 1].
Definition tableGuest :=
  mkGuestFunction 0%N tableModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [] tableCode 0.

Section Typing.
Local Open Scope nat_scope.
Example table_code_validates :
  Instrs_ok (functionContext tableContext tableFunction) tableCode (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] [valtype_I32]); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).
  typed_step 0 1 0.
  apply (instrs_cons_ok _ _ _ 1 1 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply (i32_validation_frame _ _ 0 0 1); [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
    typed_step 0 1 0.
    (* Both arms and the default leave the zero-result block. *)
    apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (br_table _ _ _ [] [] []); try reflexivity; repeat constructor.
  - typed_step 0 1 1. typed_step 0 1 2.
    (* The table keeps the result block's i32 below its index. *)
    apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (br_table _ _ _ [valtype_I32] [valtype_I32] [valtype_I32]); try reflexivity; repeat constructor.
Qed.
End Typing.

Example table_module_validates : Moduleinst_ok tableStore tableModule tableContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := tableFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example table_program_validated : programValidated tableProgram tableStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := tableContext).
    + repeat constructor.
    + exact table_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact table_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split.
    + constructor; [ | constructor]. apply supported_block; [ | right; left; reflexivity].
      constructor; [constructor | constructor; [apply supported_block; [repeat constructor | left; reflexivity] |
        repeat constructor]].
    + vm_compute. repeat match goal with
        | |- _ /\ _ => split | |- Forall _ _ => constructor | |- True => exact I | |- _ => lia end.
Qed.

(* The arguments 7 and 1, as the entry rows before the run left them. *)
Definition tableMemory := loadProgram tableProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 1 | [3] => Some 0 | _ => None end
  else None).
Definition tableWitness := map rowWitness brTableRows.
Definition tableStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma table_run_checks : validRunb tableMemory tableWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma table_fragment : runFragment tableStart 0 tableWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma table_start_config :
  CircuitConfig tableProgram tableStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    tableMemory tableStart (stackConfig (mk_state tableStore (activeFrame tableGuest (map i32Value [7; 1]))) [] tableCode).
Proof.
  apply (function_start_config tableProgram tableStore 0 tableGuest [7; 1] _ tableStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact table_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example br_table_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state tableStore (activeFrame tableGuest (map i32Value [7; 1]))) [] tableCode) c /\
    CircuitConfig tableProgram tableStore 0 shape (replayed tableMemory tableWitness)
      (rowsFinalBoundary tableStart tableWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ table_run_checks)
    table_fragment table_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* Index 1 of two arms takes arm 1 (choice 2) at pc 4, out of the zero-result block
   with nothing to move. Index 7 of one arm takes the default (choice 0) at pc 10:
   it keeps the result 1 and drops the 7 below it (sp 5 to 3), and the next row
   moves 1 from cell 3 down to cell 2. *)
Example br_tables_choose :
  let arm_row := nth 4 tableWitness (rowWitness []) in
  let default_row := nth 7 tableWitness (rowWitness []) in
  let move_row := nth 8 tableWitness (rowWitness []) in
  arm_row COL_SEL_BR_TABLE = 1 /\ arm_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 1 /\
  arm_row COL_BR_TABLE_LEN = 2 /\ arm_row COL_CONTROL_CHOICE = 2 /\ arm_row COL_PC_AFTER = 8 /\
  arm_row COL_FRAME_MOVE_DISTANCE_AFTER = 0 /\
  default_row COL_SEL_BR_TABLE = 1 /\ default_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 7 /\
  default_row COL_BR_TABLE_LEN = 1 /\ default_row COL_CONTROL_CHOICE = 0 /\ default_row COL_PC_AFTER = 13 /\
  default_row COL_SP_BEFORE = 5 /\ default_row COL_SP_AFTER = 3 /\
  default_row COL_FRAME_MOVE_REMAINING_AFTER = 1 /\ default_row COL_FRAME_MOVE_DISTANCE_AFTER = 1 /\
  move_row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 /\
  move_row (List.nth 0 COL_STACK_READ_ADDR_LO 0%nat) = 2 * 3 /\
  move_row COL_STACK_WRITE0_ADDR_LO = 2 * 2 /\ move_row COL_STACK_WRITE0_VALUE_LO = 1.
Proof. vm_compute. repeat split; reflexivity. Qed.
