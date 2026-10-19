From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range Add TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation
  BlockSpec BlockValidation SourceInstructions CodeLayout ProgramSpec ProgramEncoding ProgramRefinement ControlDecode ControlInvariant ControlBridge Fragment.
Import ListNotations.
Open Scope Z_scope.

Definition conditionalTail := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1)].
Definition conditionalCode := [conditionalBlock (mk_uN 0) conditionalTail; LOCAL_GET (mk_uN 1)].
Definition conditionalModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition conditionalFunction := i32Function conditionalModule (mk_uN 0) 0 1 2 conditionalCode.
Definition conditionalProgram := [conditionalFunction].
Definition conditionalStore := MKstore conditionalProgram [] [] [] [] [].
Definition conditionalState condition := mk_state conditionalStore
  (MKframe [i32Value condition; i32Value 9] conditionalModule).
Definition conditionalRam condition : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then match address with [8] => Some 42 | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some condition | [7; 1] => Some 9 | _ => None end
  else None.
Definition conditionalMemory condition := loadProgram conditionalProgram (conditionalRam condition).

Example conditional_layout :
  codeSize conditionalCode = 7%nat /\
  encodeProgram conditionalProgram programOpcodeId [0] = Some CODE_BLOCK /\
  encodeProgram conditionalProgram pcRomId [0; 0] = Some 1 /\
  encodeProgram conditionalProgram programOpcodeId [2] = Some CODE_BR_IF /\
  encodeProgram conditionalProgram pcRomId [2; 0] = Some 3 /\
  encodeProgram conditionalProgram pcRomId [2; 1] = Some 6 /\
  encodeProgram conditionalProgram programLocalIndexId [4] = Some 1 /\
  encodeProgram conditionalProgram programOpcodeId [5] = Some CODE_END /\
  encodeProgram conditionalProgram pcEdgeKindId [5] = Some 0 /\
  encodeProgram conditionalProgram pcRomId [5; 0] = Some 6 /\
  encodeProgram conditionalProgram programOpcodeId [7] = Some CODE_END /\
  encodeProgram conditionalProgram pcEdgeKindId [7] = Some 1.
Proof. repeat split; reflexivity. Qed.

Example conditional_block_validates :
  typedBlock 2 (LOCAL_GET (mk_uN 0) :: BR_IF (mk_uN 0) :: conditionalTail).
Proof.
  change (Instrs_ok (blockTypingContext 2)
    ([LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0)] ++ conditionalTail) (i32StackType 0 0)).
  eapply res_seq with (t_2_lst := []).
  - change (Instrs_ok (blockTypingContext 2)
      ([LOCAL_GET (mk_uN 0)] ++ [BR_IF (mk_uN 0)]) (i32StackType 0 0)).
    eapply res_seq with (t_2_lst := [valtype_I32]).
    + apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + apply Instrs_ok__instr; [apply (br_if _ _ []) | | ]; try reflexivity; repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - change (Instrs_ok (blockTypingContext 2)
      ([LOCAL_GET (mk_uN 0)] ++ [LOCAL_SET (mk_uN 1)]) (i32StackType 0 0)).
    eapply res_seq with (t_2_lst := [valtype_I32]).
    + apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + apply Instrs_ok__instr; [apply local_set | | ]; try reflexivity; repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Definition conditionalContext := MKcontext [i32StackType 0 1] [i32StackType 0 1] [] [] [] [] [] [] [] None.
Definition conditionalBodyContext := functionContext conditionalContext conditionalFunction.

Example conditional_supported : supportedBlock (_RESULT None)
    (LOCAL_GET (mk_uN 0) :: BR_IF (mk_uN 0) :: conditionalTail).
Proof.
  split; [reflexivity | ]. split.
  - unfold supportedBranchBody, conditionalTail.
    repeat (constructor; [first [left; reflexivity | right; eexists; reflexivity] | ]).
    constructor.
  - repeat split; reflexivity.
Qed.

Example conditional_code_validates : Instrs_ok conditionalBodyContext conditionalCode (i32StackType 0 1).
Proof.
  change (Instrs_ok conditionalBodyContext
    ([conditionalBlock (mk_uN 0) conditionalTail] ++ [LOCAL_GET (mk_uN 1)]) (i32StackType 0 1)).
  eapply res_seq with (t_2_lst := []).
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block_validates_context 2); try reflexivity; try solve [repeat constructor].
    + exact conditional_supported.
    + exact conditional_block_validates.
  - apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Example conditional_module_validates : forall condition,
  Moduleinst_ok (fun_store (conditionalState condition)) conditionalModule conditionalContext.
Proof.
  intros. apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := conditionalFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example conditional_function_validates : forall condition,
  Funcinst_ok (fun_store (conditionalState condition)) conditionalFunction (i32StackType 0 1).
Proof.
  intro condition. apply mk_Funcinst_ok with (C := conditionalContext).
  - repeat constructor.
  - apply conditional_module_validates.
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
    + repeat constructor.
    + apply mk_Expr_ok; [exact conditional_code_validates | repeat constructor | repeat constructor].
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

(* block (local.get 0; br_if 0; local.get 0; local.set 1); local.get 1 *)
Example conditional_body_supported :
  supportedCode (functionSite (programCallees conditionalProgram conditionalModule)) conditionalCode.
Proof.
  constructor; [ | constructor; [constructor | constructor]].
  apply supported_block; [repeat constructor | repeat split; reflexivity].
Qed.

Example conditional_program_supported : programSupported conditionalProgram.
Proof.
  intros [ | index] g Hg; cbn in Hg; [injection Hg as <-; exact conditional_body_supported | destruct index; discriminate].
Qed.

Example conditional_initial_state_exists : forall condition,
  0 <= condition < i32Modulus ->
  programLoaded conditionalProgram (conditionalMemory condition) /\
  SourceState conditionalProgram (sourcePosition 0 0) (conditionalState condition)
    (conditionalMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) conditionalCode
    (stackConfig (conditionalState condition) [] conditionalCode).
Proof.
  intros condition Hcondition. split; [apply load_program_loaded | ].
  split; [repeat split; reflexivity | ]. split; [split; [cbn; lia | reflexivity] | ]. split.
  - split; [split; [constructor; [apply conditional_function_validates | constructor] |
      exact conditional_program_supported] | ]. split.
    + exact conditional_body_supported.
    + exists conditionalContext. split; [apply conditional_module_validates | ].
      change (Instrs_ok conditionalBodyContext conditionalCode (i32StackType 0 1)).
      exact conditional_code_validates.
  - split; [ | reflexivity]. exists []. split.
    + unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split; [constructor | ].
      intros [ | index] value H; discriminate.
    + split; [reflexivity | ]. split.
      * apply stack_config_wf.
        -- constructor; [repeat constructor | ]. constructor; [ | repeat constructor].
           constructor; [apply val_case_0; apply spectec_i32_well_formed; exact Hcondition | repeat constructor].
        -- constructor.
        -- repeat constructor.
      * intros [ | [ | index]] value H; cbn [conditionalState stateLocals LOCALS nth_error] in H.
        -- injection H as <-. exists condition. repeat split; try reflexivity; lia.
        -- injection H as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
        -- destruct index; discriminate.
Qed.

(* Raw rows may stop anywhere inside the function, including inside the label. *)
Example conditional_program_refines : forall condition rows final,
  fieldPrime -> 0 <= condition < i32Modulus ->
  validCircuitRun (conditionalMemory condition) rows final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 rows ->
  exists shape c,
    Steps (stackConfig (conditionalState condition) [] conditionalCode) c /\
    programLoaded conditionalProgram final /\
    CircuitConfig conditionalProgram conditionalStore 0 shape final
      (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) rows) c.
Proof.
  intros condition rows final Hprime Hcondition Hrun Hfragment.
  destruct (conditional_initial_state_exists condition Hcondition) as [Hloaded Hsource].
  assert (Hidle : initializationIdle (mkBoundary 0 4 4 7 0 0 0 0 0)) by (repeat split).
  apply (run_refinement conditionalProgram conditionalStore 0 (controlShape [] (siteInstruction (sourcePosition 0 0)))
    rows (conditionalMemory condition) final _ _ Hprime Hloaded Hrun Hfragment).
  apply (control_state_circuit _ _ _ (controlCursor [] (atInstruction (sourcePosition 0 0)))
    (conditionalState condition)); [reflexivity | repeat constructor | reflexivity | ].
  exact (base_control_state conditionalProgram (atInstruction (sourcePosition 0 0)) (conditionalState condition)
    (conditionalMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) _
    (conj (ex_intro _ conditionalCode Hsource) Hidle) ltac:(cbn; lia)).
Qed.
