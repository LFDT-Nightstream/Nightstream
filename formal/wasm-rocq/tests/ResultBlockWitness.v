(* A real run through result blocks: main(7, 9) takes a br_if out of an inner
   result block, which keeps the result 9 and drops the 7 below it, and leaves the
   outer block through its END with its result. The exported rows meet every
   premise of run_refinement, and the branch's slide leaves the result to move down. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.ResultBlockTrace.
Import ListNotations.
Open Scope Z_scope.

(* block (result i32) (block (result i32) (local.get 0; local.get 1; local.get 0; br_if 0; i32.add);
   local.get 1; i32.sub) *)
Definition i32Block (body : list instr) := BLOCK (_RESULT (Some valtype_I32)) body.
Definition innerBody := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0);
  BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Definition outerBody := [i32Block innerBody; LOCAL_GET (mk_uN 1); BINOP I32 (mk_binop__0 Inn_I32 SUB)].
Definition resultCode := [i32Block outerBody].
Definition resultModule := MKmoduleinst [i32StackType 2 1] [0%N] [] [] [] [] [] [].
Definition resultFunction := i32Function resultModule (mk_uN 0) 2 1 0 resultCode.
Definition resultProgram := [resultFunction].
Definition resultStore := MKstore resultProgram [] [] [] [] [].
Definition resultContext := moduleContext [i32StackType 2 1] [i32StackType 2 1].
Definition resultGuest := mkGuestFunction 0%N resultModule (mk_uN 0) 2 1 0 resultCode 0.

Section Typing.
Local Open Scope nat_scope.

Ltac result_block :=
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor];
  apply (block _ _ _ [] [valtype_I32]); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor];
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).

Example result_code_validates :
  Instrs_ok (functionContext resultContext resultFunction) resultCode (i32StackType 0 1).
Proof.
  result_block.
  apply (instrs_cons_ok _ _ _ 0 1 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - result_block.
    typed_step 0 1 0. typed_step 0 1 1. typed_step 0 1 2.
    (* br_if 0 keeps the block's i32 result below its condition. *)
    apply instrs_cons_ok with (mid := 2); [repeat constructor | repeat constructor | repeat constructor | | ].
    + apply (i32_validation_frame _ _ 2 1 1); [ | repeat constructor | repeat constructor].
      apply (br_if _ _ [valtype_I32]); try reflexivity; repeat constructor.
    + typed_instr 2 1 0.
  - typed_step 0 1 1. typed_instr 2 1 0.
Qed.
End Typing.

Example result_module_validates : Moduleinst_ok resultStore resultModule resultContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := resultFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example result_program_validated : programValidated resultProgram resultStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := resultContext).
    + repeat constructor.
    + exact result_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact result_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split; [ | vm_compute; repeat split; lia].
    constructor; [ | constructor]. apply supported_block; [ | right; reflexivity].
    constructor; [apply supported_block; [repeat constructor | right; reflexivity] | repeat constructor].
Qed.

(* The arguments 7 and 9, as the entry rows before the run left them. *)
Definition resultMemory := loadProgram resultProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 9 | [3] => Some 0 | _ => None end
  else None).
Definition resultWitness := map rowWitness resultBlockRows.
Definition resultStart := mkBoundary 0 2 2 0 0 0 0 0 0 0.

Lemma result_run_checks : validRunb resultMemory resultWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma result_fragment : runFragment resultStart 0 resultWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma result_start_config :
  CircuitConfig resultProgram resultStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    resultMemory resultStart (stackConfig (mk_state resultStore (activeFrame resultGuest [7; 9])) [] resultCode).
Proof.
  apply (function_start_config resultProgram resultStore 0 resultGuest [7; 9] _ resultStart);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact result_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example result_block_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state resultStore (activeFrame resultGuest [7; 9])) [] resultCode) c /\
    CircuitConfig resultProgram resultStore 0 shape (replayed resultMemory resultWitness)
      (rowsFinalBoundary resultStart resultWitness) c.
Proof.
  intros Hprime.
  destruct (run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ result_run_checks)
    result_fragment result_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The branch at pc 5 is taken: it pops its condition, 7, keeps the result 9 and
   drops the 7 below it (sp 5 to 3), leaving the result one cell too high; the next
   row moves 9 from cell 3 down to cell 2, the inner block's base. *)
Example result_branch_slides :
  let branch_row := nth 5 resultWitness (rowWitness []) in
  let move_row := nth 6 resultWitness (rowWitness []) in
  branch_row COL_SEL_BR_IF_EQZ = 1 /\ branch_row COL_PC_BEFORE = 5 /\ branch_row COL_PC_AFTER = 8 /\
  branch_row COL_SP_BEFORE = 5 /\ branch_row COL_SP_AFTER = 3 /\
  branch_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 7 /\
  branch_row COL_FRAME_MOVE_REMAINING_AFTER = 1 /\ branch_row COL_FRAME_MOVE_DISTANCE_AFTER = 1 /\
  move_row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 /\ move_row COL_FRAME_MOVE_REMAINING_AFTER = 0 /\
  move_row (List.nth 0 COL_STACK_READ_ADDR_LO 0%nat) = 2 * 3 /\
  move_row (List.nth 0 COL_STACK_READ_VALUE_LO 0%nat) = 9 /\
  move_row COL_STACK_WRITE0_ADDR_LO = 2 * 2 /\ move_row COL_STACK_WRITE0_VALUE_LO = 9.
Proof. vm_compute. repeat split; reflexivity. Qed.
