(* A real run with three ifs: main(7, 0) takes the else-body on 0, the then-body on
   7, whose ELSE jumps past the else-body, and skips an if with no else-body on 0.
   The exported rows meet every premise of active_run_refinement. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.IfTrace.
Require Import ControlInvariant.
Import ListNotations.
Open Scope Z_scope.

Definition i32Add := BINOP I32 (mk_binop__0 Inn_I32 ADD).

(* local.get 0; local.get 1; if (result i32) (local.get 1) else (local.get 0); i32.add;
   local.get 0; if (result i32) (local.get 0) else (local.get 1); i32.add;
   local.get 1; if (local.get 0; local.set 1) *)
Definition ifCode := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1);
  IFELSE (_RESULT (Some valtype_I32)) [LOCAL_GET (mk_uN 1)] [LOCAL_GET (mk_uN 0)]; i32Add;
  LOCAL_GET (mk_uN 0); IFELSE (_RESULT (Some valtype_I32)) [LOCAL_GET (mk_uN 0)] [LOCAL_GET (mk_uN 1)]; i32Add;
  LOCAL_GET (mk_uN 1); IFELSE (_RESULT None) [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1)] []].
Definition ifModule := MKmoduleinst [i32StackType 2 1] [0%N] [] [] [] [] [] [].
Definition ifFunction := i32Function ifModule (mk_uN 0) 2 1 0 ifCode.
Definition ifProgram := [ifFunction].
Definition ifStore := MKstore ifProgram [] [] [] [] [].
Definition ifContext := moduleContext [i32StackType 2 1] [i32StackType 2 1].
Definition ifGuest := mkGuestFunction 0%N ifModule (mk_uN 0) [valtype_I32; valtype_I32] [valtype_I32] [] ifCode 0.

Section Typing.
Local Open Scope nat_scope.

(* An if above `extra` operands: it pops its condition and leaves its results. *)
Ltac typed_if extra results :=
  apply instrs_cons_ok with (mid := (extra + length results)%nat);
    [repeat constructor | repeat constructor | repeat constructor | | ];
    [apply (i32_validation_frame _ _ 1 (length results) extra); [ | repeat constructor | repeat constructor];
     apply (res_if _ _ _ _ [] results); [apply Blocktype_ok__valtype; repeat constructor | | |
       repeat constructor | repeat constructor | repeat constructor] | ].

Example if_code_validates :
  Instrs_ok (functionContext ifContext ifFunction) ifCode (i32StackType 0 1).
Proof.
  typed_step 0 1 0. typed_step 0 1 1.
  typed_if 1 [valtype_I32].
  1, 2: change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1);
    typed_instr 0 1 0.
  typed_step 2 1 0. typed_step 0 1 1.
  typed_if 1 [valtype_I32].
  1, 2: change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1);
    typed_instr 0 1 0.
  typed_step 2 1 0. typed_step 0 1 1.
  (* The last if leaves the function's result below it. *)
  apply (i32_validation_frame _ _ 1 0 1); [ | repeat constructor | repeat constructor].
  apply (res_if _ _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | | repeat constructor |
    repeat constructor | repeat constructor].
  - change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
    typed_step 0 1 0. typed_instr 1 0 0.
  - apply empty. repeat constructor.
Qed.
End Typing.

Example if_module_validates : Moduleinst_ok ifStore ifModule ifContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := ifFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example if_program_validated : programValidated ifProgram ifStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := ifContext).
    + repeat constructor.
    + exact if_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact if_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split.
    + cbn. repeat first [apply Forall_nil | apply Forall_cons | apply supported_if | apply supported_local_get |
        apply supported_local_set | apply supported_i32_add | (left; reflexivity) | (right; left; reflexivity)].
    + vm_compute. repeat match goal with
        | |- _ /\ _ => split | |- Forall _ _ => constructor | |- True => exact I | |- _ => lia end.
Qed.

(* The arguments 7 and 0, as the entry rows before the run left them. *)
Definition ifMemory := loadProgram ifProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 0 | [3] => Some 0 | _ => None end
  else None).
Definition ifWitness := map rowWitness ifRows.
Definition ifStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma if_run_checks : validRunb ifMemory ifWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma if_fragment : runFragment ifStart 0 ifWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma if_start_config :
  CircuitConfig ifProgram ifStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    ifMemory ifStart (stackConfig (mk_state ifStore (activeFrame ifGuest (map i32Value [7; 0]))) [] ifCode).
Proof.
  apply (function_start_config ifProgram ifStore 0 ifGuest [7; 0] _ ifStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact if_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example if_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state ifStore (activeFrame ifGuest (map i32Value [7; 0]))) [] ifCode) c /\
    CircuitConfig ifProgram ifStore 0 shape (replayed ifMemory ifWitness)
      (rowsFinalBoundary ifStart ifWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ if_run_checks)
    if_fragment if_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The condition 0 takes the else-body (choice 0) at pc 2, past the then-body and
   ELSE; its END at pc 7 falls through. The condition 7 takes the then-body
   (choice 1) at pc 10, whose ELSE at pc 13 jumps past the else-body. The last
   condition 0 skips the if with no else-body at pc 18. An if takes two pcs. *)
Example ifs_choose :
  let else_row := nth 2 ifWitness (rowWitness []) in
  let end_row := nth 4 ifWitness (rowWitness []) in
  let then_row := nth 7 ifWitness (rowWitness []) in
  let jump_row := nth 9 ifWitness (rowWitness []) in
  let skip_row := nth 12 ifWitness (rowWitness []) in
  else_row COL_SEL_IF = 1 /\ else_row COL_CONTROL_CHOICE = 0 /\ else_row COL_PC_AFTER = 6 /\
  end_row COL_SEL_END = 1 /\ end_row COL_PC_AFTER = 8 /\
  then_row COL_SEL_IF = 1 /\ then_row COL_CONTROL_CHOICE = 1 /\ then_row COL_PC_AFTER = 12 /\
  jump_row COL_SEL_ELSE = 1 /\ jump_row COL_PC_BEFORE = 13 /\ jump_row COL_PC_AFTER = 16 /\
  skip_row COL_SEL_IF = 1 /\ skip_row COL_CONTROL_CHOICE = 0 /\ skip_row COL_PC_AFTER = 23.
Proof. vm_compute. repeat split; reflexivity. Qed.
