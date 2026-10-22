(* br_if 1 from inside two nested blocks: it jumps past the outer block's END. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment tests.BlockPrograms.
Import ListNotations.
Open Scope Z_scope.

(* block (block (local.get 0; br_if 1; local.get 1; local.set 0); local.get 1; local.set 0); local.get 0 *)
Definition innerBody := [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 1); LOCAL_GET (mk_uN 1); LOCAL_SET (mk_uN 0)].
Definition outerBody := [BLOCK (_RESULT None) innerBody; LOCAL_GET (mk_uN 1); LOCAL_SET (mk_uN 0)].
Definition deepCode := [BLOCK (_RESULT None) outerBody; LOCAL_GET (mk_uN 0)].
Definition deepModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition deepFunction := i32Function deepModule (mk_uN 0) 0 1 2 deepCode.
Definition deepProgram := [deepFunction].
Definition deepStore := MKstore deepProgram [] [] [] [] [].
Definition deepGuest := mkGuestFunction 0%N deepModule (mk_uN 0) 0 1 2 deepCode 0.
Definition deepState condition := mk_state deepStore (MKframe [i32Value condition; i32Value 9] deepModule).

(* PCs: 0 and 1 enter the blocks, 3 is the br_if, 6 and 9 are the ENDs, 10 follows. *)
Example deep_layout :
  encodeProgram deepProgram programOpcodeId [3] = Some CODE_BR_IF /\
  encodeProgram deepProgram pcRomId [3; 0] = Some 4 /\
  encodeProgram deepProgram pcRomId [3; 1] = Some 10 /\
  encodeProgram deepProgram programOpcodeId [9] = Some CODE_END /\
  encodeProgram deepProgram pcRomId [9; 0] = Some 10.
Proof. repeat split; reflexivity. Qed.

Definition deepContext := moduleContext [i32StackType 0 1] [i32StackType 0 1].
Definition deepBodyContext := functionContext deepContext deepFunction.

Section Typing.
Local Open Scope nat_scope.
Example deep_code_validates : Instrs_ok deepBodyContext deepCode (i32StackType 0 1).
Proof.
  apply (instrs_cons_ok _ _ _ 0 0 1); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
    apply (instrs_cons_ok _ _ _ 0 0 0); [repeat constructor | repeat constructor | repeat constructor | | ].
    + apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
      apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
        repeat constructor | repeat constructor].
      change (mk_functype (mk_list _ []) (mk_list _ [])) with (i32StackType 0 0).
      typed_step 0 1 0. typed_step 1 0 0. typed_step 0 1 0. typed_instr 1 0 0.
    + typed_step 0 1 0. typed_instr 1 0 0.
  - typed_instr 0 1 0.
Qed.
End Typing.

Example deep_module_validates : Moduleinst_ok deepStore deepModule deepContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := deepFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example deep_program_validated : programValidated deepProgram deepStore.
Proof.
  split.
  - constructor; [ | constructor]. apply mk_Funcinst_ok with (C := deepContext).
    + repeat constructor.
    + exact deep_module_validates.
    + apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
      * repeat constructor.
      * apply mk_Expr_ok; [exact deep_code_validates | repeat constructor | repeat constructor].
      * repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
    split; [repeat constructor | vm_compute; repeat split; lia].
Qed.

Definition deepRam condition : MemoryState := fun memory address =>
  if Nat.eqb memory localLoId then
    match address with [7; 0] => Some condition | [7; 1] => Some 9 | _ => None end
  else None.
Definition deepMemory condition := loadProgram deepProgram (deepRam condition).

Example deep_initial_config : forall condition,
  0 <= condition < i32Modulus ->
  CircuitConfig deepProgram deepStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    (deepMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) (stackConfig (deepState condition) [] deepCode).
Proof.
  intros condition Hcondition.
  apply (function_start_config deepProgram deepStore 0 deepGuest [condition; 9] _ (mkBoundary 0 4 4 7 0 0 0 0 0));
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact deep_program_validated.
  - repeat constructor; unfold i32Modulus in *; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* Any run from the start refines SpecTec, whichever way the branch goes. *)
Example deep_program_refines : forall condition rows final,
  fieldPrime -> 0 <= condition < i32Modulus -> validCircuitRun (deepMemory condition) rows final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 rows ->
  exists shape c, Steps (stackConfig (deepState condition) [] deepCode) c /\ programLoaded deepProgram final /\
    CircuitConfig deepProgram deepStore 0 shape final (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) rows) c.
Proof.
  intros condition rows final Hprime Hcondition Hrun Hfragment.
  exact (run_refinement deepProgram deepStore 0 _ rows (deepMemory condition) final _ _ Hprime
    (load_program_loaded _ _) Hrun Hfragment (deep_initial_config condition Hcondition)).
Qed.

(* At the br_if: the condition 7 is the only operand, inside both labels. *)
Definition branchMemory := loadProgram deepProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with [8] => Some 7 | _ => None end else deepRam 7 memory address).

Example deep_branch_site_config :
  CircuitConfig deepProgram deepStore 0 (controlShape [] (siteCode (codePosition 0 [0; 0]%nat 1)))
    branchMemory (mkBoundary 3 5 4 7 0 0 0 0 0)
    (mk_config (deepState 7)
      [LABEL_ 0%N [] [LABEL_ 0%N [] (i32Constant 7 :: List.map admininstr_instr (skipn 1 innerBody));
        admininstr_instr (LOCAL_GET (mk_uN 1)); admininstr_instr (LOCAL_SET (mk_uN 0))];
       admininstr_instr (LOCAL_GET (mk_uN 0))]).
Proof.
  split; [reflexivity | ]. split; [repeat constructor | ]. split; [exact deep_program_validated | ].
  exists True. split; [vm_compute; reflexivity | exact I].
Qed.

(* A pc that is not the branch's does not decode at this position. *)
Example deep_branch_wrong_pc_rejected :
  decodeControl deepProgram deepStore 0 (controlShape [] (siteCode (codePosition 0 [0; 0]%nat 1)))
    branchMemory (mkBoundary 4 5 4 7 0 0 0 0 0) = None.
Proof. vm_compute. reflexivity. Qed.
