(* A real run of a loop: main(3, 1) sums 3, 2 and 1. Each taken br_if drops the sum
   it pushed and runs the loop again; the last iteration falls out of the loop with
   the sum 6 as its result. The exported rows meet every premise of active_run_refinement. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.LoopTrace.
Require Import ControlInvariant.
Import ListNotations.
Open Scope Z_scope.

Definition i32Add := BINOP I32 (mk_binop__0 Inn_I32 ADD).
Definition i32Sub := BINOP I32 (mk_binop__0 Inn_I32 SUB).

(* loop (result i32) (local.get 2; local.get 0; i32.add; local.set 2;
   local.get 0; local.get 1; i32.sub; local.set 0; local.get 2; local.get 0; br_if 0) *)
Definition loopBody := [LOCAL_GET (mk_uN 2); LOCAL_GET (mk_uN 0); i32Add; LOCAL_SET (mk_uN 2);
  LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); i32Sub; LOCAL_SET (mk_uN 0);
  LOCAL_GET (mk_uN 2); LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0)].
Definition loopCode := [LOOP (_RESULT (Some valtype_I32)) loopBody].
Definition loopModule := MKmoduleinst [i32StackType 2 1] [0%N] [] [] [] [] [] [].
Definition loopFunction := i32Function loopModule (mk_uN 0) 2 1 1 loopCode.
Definition loopProgram := [loopFunction].
Definition loopStore := MKstore loopProgram [] [] [] [] [].
Definition loopContext := moduleContext [i32StackType 2 1] [i32StackType 2 1].
Definition loopGuest :=
  mkGuestFunction 0%N loopModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [valtype_I32] loopCode 0.

Section Typing.
Local Open Scope nat_scope.
Example loop_code_validates :
  Instrs_ok (functionContext loopContext loopFunction) loopCode (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (loop _ _ _ [] [valtype_I32]); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
    repeat constructor | repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).
  typed_step 0 1 0. typed_step 0 1 1. typed_step 2 1 0. typed_step 1 0 0.
  typed_step 0 1 0. typed_step 0 1 1. typed_step 2 1 0. typed_step 1 0 0.
  typed_step 0 1 0. typed_step 0 1 1.
  (* The branch to the loop's label keeps nothing of the sum below its condition. *)
  typed_instr 1 0 1.
Qed.
End Typing.

Example loop_module_validates : Moduleinst_ok loopStore loopModule loopContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := loopFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example loop_program_validated : programValidated loopProgram loopStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := loopContext).
    + repeat constructor.
    + exact loop_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact loop_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split.
    + constructor; [ | constructor]. apply supported_loop; [repeat constructor | right; reflexivity].
    + vm_compute. repeat match goal with
        | |- _ /\ _ => split | |- Forall _ _ => constructor | |- True => exact I | |- _ => lia end.
Qed.

(* The arguments 3 and 1 and the zeroed local, as the entry and zero rows before the
   run left them. *)
Definition loopMemory := loadProgram loopProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with
    | [0] => Some 3 | [1] => Some 0 | [2] => Some 1 | [3] => Some 0 | [4] => Some 0 | [5] => Some 0 | _ => None
    end
  else None).
Definition loopWitness := map rowWitness loopRows.
Definition loopStart := mkBoundary 0 3 3 0 0 0 0 0 0 0 0.

Lemma loop_run_checks : validRunb loopMemory loopWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma loop_fragment : runFragment loopStart 0 loopWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma loop_start_config :
  CircuitConfig loopProgram loopStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    loopMemory loopStart (stackConfig (mk_state loopStore (activeFrame loopGuest (map i32Value [3; 1; 0]))) [] loopCode).
Proof.
  apply (function_start_config loopProgram loopStore 0 loopGuest [3; 1; 0] _ loopStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact loop_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | [ | i]]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      injection Hv as <-; reflexivity | destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example loop_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state loopStore (activeFrame loopGuest (map i32Value [3; 1; 0]))) [] loopCode) c /\
    CircuitConfig loopProgram loopStore 0 shape (replayed loopMemory loopWitness)
      (rowsFinalBoundary loopStart loopWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ loop_run_checks)
    loop_fragment loop_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The LOOP at pc 0 enters its body at pc 1. The first two br_ifs at pc 11 are
   taken: each drops the sum below its condition (sp 5 to 3, the loop's base) and
   moves nothing, and runs the body again from pc 1. The third falls through, and
   the loop's END at pc 12 keeps the sum as its result. *)
Example loop_rows :
  let loop_row := nth 0 loopWitness (rowWitness []) in
  let back_row := nth 11 loopWitness (rowWitness []) in
  let exit_row := nth 33 loopWitness (rowWitness []) in
  let end_row := nth 34 loopWitness (rowWitness []) in
  loop_row COL_SEL_LOOP = 1 /\ loop_row COL_PC_AFTER = 1 /\
  back_row COL_SEL_BR_IF_EQZ = 1 /\ back_row COL_PC_BEFORE = 11 /\ back_row COL_PC_AFTER = 1 /\
  back_row COL_SP_BEFORE = 5 /\ back_row COL_SP_AFTER = 3 /\ back_row COL_FRAME_MOVE_REMAINING_AFTER = 0 /\
  exit_row COL_SEL_BR_IF_EQZ = 1 /\ exit_row COL_PC_AFTER = 12 /\ exit_row COL_SP_AFTER = 4 /\
  end_row COL_SEL_END = 1 /\ end_row COL_PC_BEFORE = 12 /\ end_row COL_PC_AFTER = 13 /\
  end_row COL_SP_AFTER = 4.
Proof. vm_compute. repeat split; reflexivity. Qed.
