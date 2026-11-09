(* Explicit base-frame return through a block, and an activation with no result. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout Fragment
  ProgramSpec ProgramEncoding ControlDecode ExecutionSpec ExecutionRefinement
  tests.BlockPrograms tests.WitnessCheck tests.TerminalWitness tests.NopWitness tests.BaseReturnTrace tests.VoidReturnTrace.
Import ListNotations.
Open Scope Z_scope.

Ltac single_module_validates function :=
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor];
  try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := function);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]];
  repeat constructor.

Ltac function_validates context moduleProof bodyProof :=
  apply mk_Funcinst_ok with (C := context);
  [ repeat constructor
  | exact moduleProof
  | apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity);
    [ repeat constructor
    | apply mk_Expr_ok; [exact bodyProof | repeat constructor | repeat constructor]
    | repeat constructor | repeat constructor | repeat constructor ]
  | repeat constructor | repeat constructor | repeat constructor ].

Definition baseReturnCode := [BLOCK (_RESULT (Some valtype_I32)) [LOCAL_GET (mk_uN 0); RETURN]].
Definition baseReturnFunction := i32Function nopModule (mk_uN 0) 1 1 0 baseReturnCode.
Definition baseReturnProgram := [baseReturnFunction].
Definition baseReturnStore := MKstore baseReturnProgram [] [] [] [] [].
Definition baseReturnGuest := mkGuestFunction 0%N nopModule (mk_uN 0) 1 1 0 baseReturnCode 0.
Definition baseReturnMemory := loadProgram baseReturnProgram (exportSchedules 5 1 (fun memory address =>
  if Nat.eqb memory stackId then match address with [0] => Some 7 | [1] => Some 0 | _ => None end else None)).
Definition baseReturnWitness := map rowWitness baseReturnRows.

Lemma base_return_module : Moduleinst_ok baseReturnStore nopModule nopContext.
Proof. single_module_validates baseReturnFunction. Qed.
Lemma base_return_body :
  Instrs_ok (functionContext nopContext baseReturnFunction) baseReturnCode (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  apply (block _ _ _ [] [valtype_I32]); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  change (mk_functype (mk_list _ []) (mk_list _ [valtype_I32])) with (i32StackType 0 1).
  typed_step 0%nat 1%nat 0%nat.
  apply Instrs_ok__instr; [apply (res_return _ [] [valtype_I32] [valtype_I32]) | | ];
    try (apply (introT eqP); reflexivity); repeat constructor.
Qed.
Lemma base_return_validated : programValidated baseReturnProgram baseReturnStore.
Proof.
  split.
  - constructor; [ | constructor]. function_validates nopContext base_return_module base_return_body.
  - intros [ | index] f H; cbn in H; [injection H as <- | destruct index; discriminate].
    split; [ | vm_compute; tauto].
    constructor; [apply supported_block; [repeat constructor | right; reflexivity] | constructor].
Qed.
Lemma base_return_valid_run : validRunb baseReturnMemory baseReturnWitness = true.
Proof. vm_compute; reflexivity. Qed.
Lemma base_return_fragment : executionFragment nopStart 0 baseReturnWitness.
Proof. vm_compute. repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I]. Qed.
Lemma base_return_initial : CircuitConfig baseReturnProgram baseReturnStore 0 entryShape baseReturnMemory nopStart
    (stackConfig (mk_state baseReturnStore (activeFrame baseReturnGuest [7])) [] baseReturnCode).
Proof.
  apply (function_start_config baseReturnProgram baseReturnStore 0 baseReturnGuest [7] _ nopStart);
    try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact base_return_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | index] value H; cbn in H; [injection H as <-; reflexivity | destruct index; discriminate].
  - repeat split.
Qed.
Example base_return_refines : fieldPrime ->
  Steps (callContext (outerFrame nopModule) 1
    (stackConfig (mk_state baseReturnStore (activeFrame baseReturnGuest [7])) [] baseReturnCode))
    (mk_config (mk_state baseReturnStore (outerFrame nopModule)) [i32Constant 7]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig baseReturnProgram baseReturnStore (outerFrame nopModule) 0 0 entryShape
      baseReturnMemory nopStart (callContext (outerFrame nopModule) 1
        (stackConfig (mk_state baseReturnStore (activeFrame baseReturnGuest [7])) [] baseReturnCode))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ]. exists baseReturnGuest, (stackConfig (mk_state baseReturnStore (activeFrame baseReturnGuest [7])) [] baseReturnCode).
    split; [reflexivity | ]. split; [exact base_return_initial | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ base_return_valid_run) base_return_fragment Hstart) as [outcome [c [Hsteps [_ Hconfig]]]].
  replace c with (mk_config (mk_state baseReturnStore (outerFrame nopModule)) [i32Constant 7]) in Hsteps; [exact Hsteps | ].
  symmetry. eapply terminal_return_decodes with (row := rowWitness (List.last baseReturnRows []));
    [exact Hconfig | reflexivity | | | | ]; vm_compute; reflexivity.
Qed.

Definition voidModule := MKmoduleinst [i32StackType 0 0] [0%N] [] [] [] [] [] [].
Definition voidFunction := i32Function voidModule (mk_uN 0) 0 0 0 [NOP].
Definition voidProgram := [voidFunction].
Definition voidStore := MKstore voidProgram [] [] [] [] [].
Definition voidContext := moduleContext [i32StackType 0 0] [i32StackType 0 0].
Definition voidGuest := mkGuestFunction 0%N voidModule (mk_uN 0) 0 0 0 [NOP] 0.
Definition voidMemory := loadProgram voidProgram (exportSchedules 1 0 (fun _ _ => Some 0)).
Definition voidWitness := map rowWitness voidReturnRows.
Lemma void_module : Moduleinst_ok voidStore voidModule voidContext.
Proof. single_module_validates voidFunction. Qed.
Lemma void_body : Instrs_ok (functionContext voidContext voidFunction) [NOP] (i32StackType 0 0).
Proof. apply Instrs_ok__instr; [apply nop | | ]; repeat constructor. Qed.
Lemma void_validated : programValidated voidProgram voidStore.
Proof.
  split; [constructor; [function_validates voidContext void_module void_body | constructor] | ].
  intros [ | index] f H; cbn in H; [injection H as <- | destruct index; discriminate].
  split; [repeat constructor | vm_compute; tauto].
Qed.
Lemma void_valid_run : validRunb voidMemory voidWitness = true.
Proof. vm_compute; reflexivity. Qed.
Lemma void_fragment : executionFragment trapStart 0 voidWitness.
Proof. vm_compute. repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I]. Qed.
Lemma void_initial : CircuitConfig voidProgram voidStore 0 entryShape voidMemory trapStart
    (stackConfig (mk_state voidStore (activeFrame voidGuest [])) [] [NOP]).
Proof.
  apply (function_start_config voidProgram voidStore 0 voidGuest [] _ trapStart); try reflexivity; try (apply TableState.no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact void_validated.
  - constructor.
  - intros index value H; destruct index; discriminate.
  - repeat split.
Qed.
Example void_return_refines : fieldPrime ->
  Steps (callContext (outerFrame voidModule) 0 (stackConfig (mk_state voidStore (activeFrame voidGuest [])) [] [NOP]))
    (mk_config (mk_state voidStore (outerFrame voidModule)) []).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig voidProgram voidStore (outerFrame voidModule) 0 0 entryShape voidMemory trapStart
      (callContext (outerFrame voidModule) 0 (stackConfig (mk_state voidStore (activeFrame voidGuest [])) [] [NOP]))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ]. exists voidGuest, (stackConfig (mk_state voidStore (activeFrame voidGuest [])) [] [NOP]).
    split; [reflexivity | ]. split; [exact void_initial | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ void_valid_run) void_fragment Hstart) as [outcome [c [Hsteps [_ Hconfig]]]].
  assert (Hlast : exists row, lastRow voidWitness = Some row /\ row COL_ACTIVATION_FINISHED_AFTER = 1 /\
    row COL_TRAPPED_AFTER = 0 /\ row COL_OUTPUT_CAPTURED = 0).
  { exists (rowWitness (List.last voidReturnRows [])). split; [reflexivity | ]. vm_compute; repeat split; reflexivity. }
  destruct Hlast as [row [Hlast [Hfinish [Htrap Hcapture]]]].
  destruct outcome as [shape | | [v | ]].
  - destruct Hconfig as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct Hconfig as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
  - destruct Hconfig as [last [Hlast' [_ [_ [[Hcap _] _]]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
  - destruct Hconfig as [last [Hlast' [_ [_ [_ Hc]]]]]. rewrite <- Hc; exact Hsteps.
Qed.
