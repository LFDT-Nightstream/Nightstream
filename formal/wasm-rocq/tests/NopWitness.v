(* A real run with nops at the top level and inside a block: main(7) leaves 7. The
   exported rows meet every premise of active_run_refinement. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms tests.WitnessCheck tests.NopTrace.
Require Import ControlInvariant.
Import ListNotations.
Open Scope Z_scope.

(* nop; local.get 0; block (nop); nop *)
Definition nopCode := [NOP; LOCAL_GET (mk_uN 0); BLOCK (_RESULT None) [NOP]; NOP].
Definition nopModule := MKmoduleinst [i32StackType 1 1] [0%N] [] [] [] [] [] [].
Definition nopFunction := i32Function nopModule (mk_uN 0) 1 1 0 nopCode.
Definition nopProgram := [nopFunction].
Definition nopStore := MKstore nopProgram [] [] [] [] [].
Definition nopContext := moduleContext [i32StackType 1 1] [i32StackType 1 1].
Definition nopGuest := mkGuestFunction 0%N nopModule (mk_uN 0) [valtype_I32] [valtype_I32] [] nopCode 0.

Section Typing.
Local Open Scope nat_scope.
Example nop_code_validates :
  Instrs_ok (functionContext nopContext nopFunction) nopCode (i32StackType 0 1).
Proof.
  apply (instrs_cons_ok _ _ _ 0 0 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply (i32_validation_frame _ _ 0 0 0); [apply nop; repeat constructor | repeat constructor | repeat constructor].
  - typed_step 0 1 0.
    apply (instrs_cons_ok _ _ _ 1 1 1); [repeat constructor | repeat constructor | repeat constructor | | ].
    + apply (i32_validation_frame _ _ 0 0 1); [ | repeat constructor | repeat constructor].
      apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
        repeat constructor | repeat constructor].
      change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
      apply (i32_validation_frame _ _ 0 0 0); [apply nop; repeat constructor | repeat constructor | repeat constructor].
    + apply (i32_validation_frame _ _ 0 0 1); [apply nop; repeat constructor | repeat constructor | repeat constructor].
Qed.
End Typing.

Example nop_module_validates : Moduleinst_ok nopStore nopModule nopContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := nopFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example nop_program_validated : programValidated nopProgram nopStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := nopContext).
    + repeat constructor.
    + exact nop_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact nop_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split.
    + cbn. repeat first [apply Forall_nil | apply Forall_cons | apply supported_nop | apply supported_local_get |
        apply supported_block | (left; reflexivity)].
    + vm_compute. repeat match goal with
        | |- _ /\ _ => split | |- Forall _ _ => constructor | |- True => exact I | |- _ => lia end.
Qed.

(* The argument 7, as the entry rows before the run left it. *)
Definition nopMemory := loadProgram nopProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with [0] => Some 7 | [1] => Some 0 | _ => None end
  else None).
Definition nopWitness := map rowWitness nopRows.
Definition nopStart := mkBoundary 0 1 1 0 0 0 0 0 0 0 0.

Lemma nop_run_checks : validRunb nopMemory nopWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma nop_fragment : runFragment nopStart 0 nopWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma nop_start_config :
  CircuitConfig nopProgram nopStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    nopMemory nopStart (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode).
Proof.
  apply (function_start_config nopProgram nopStore 0 nopGuest [7] _ nopStart); try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact nop_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | i] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example nop_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode) c /\
    CircuitConfig nopProgram nopStore 0 shape (replayed nopMemory nopWitness) (rowsFinalBoundary nopStart nopWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ nop_run_checks)
    nop_fragment nop_start_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The nops at pcs 0, 3 (inside the block) and 5 each step to the next pc and keep
   the stack. *)
Example nops_step :
  let rows := [nth 0 nopWitness (rowWitness []); nth 3 nopWitness (rowWitness []);
    nth 5 nopWitness (rowWitness [])] in
  map (fun w : Witness => (w COL_SEL_NOP, w COL_PC_AFTER - w COL_PC_BEFORE, w COL_SP_AFTER - w COL_SP_BEFORE)) rows =
    [(1, 1, 0); (1, 1, 0); (1, 1, 0)].
Proof. vm_compute. reflexivity. Qed.
