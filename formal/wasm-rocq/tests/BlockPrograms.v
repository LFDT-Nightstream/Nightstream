From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range Add TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation
  BlockSpec CodeLayout ProgramSpec ProgramEncoding ProgramRefinement ControlDecode ControlInvariant ControlBridge
  Fragment.
Import ListNotations.
Open Scope Z_scope.

(* Typing a sequence one instruction at a time. *)
Lemma instrs_cons_ok : forall C i rest h mid out,
  wf_context C -> wf_instr i -> Forall wf_instr rest ->
  Instrs_ok C [i] (i32StackType h mid) -> Instrs_ok C rest (i32StackType mid out) ->
  Instrs_ok C (i :: rest) (i32StackType h out).
Proof.
  intros C i rest h mid out HC Hi Hrest H1 H2. change (i :: rest) with ([i] ++ rest).
  unfold i32StackType in *. eapply res_seq; [exact H1 | exact H2 | exact HC | constructor; [exact Hi | constructor] |
    exact Hrest].
Qed.

(* One i32 instruction with `inputs` and `outputs`, above `extra` operands. *)
Ltac typed_instr inputs outputs extra :=
  apply (i32_validation_frame _ _ inputs outputs extra);
    [first [apply local_get | apply local_set | apply binop | apply (br_if _ _ [])]; try reflexivity; repeat constructor |
     repeat constructor | repeat constructor].

Ltac typed_step inputs outputs extra :=
  apply instrs_cons_ok with (mid := (extra + outputs)%nat);
    [repeat constructor | repeat constructor | repeat constructor | typed_instr inputs outputs extra | ].

(* A function's first instruction, with its locals as its first cells. *)
Lemma function_start_config : forall program s index f locals memory boundary,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  functionAt program index = Some f -> length locals = frameLocals f ->
  Forall (fun v => 0 <= v < i32Modulus) locals ->
  (forall i v, nth_error locals i = Some v ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat i)] = Some v) ->
  0 <= boundaryLocalsBase boundary ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (frameLocals f) ->
  boundarySP boundary = boundaryStackBase boundary ->
  boundaryPC boundary = guestEntry f -> initializationIdle boundary ->
  0 <= boundaryCallDepth boundary ->
  CircuitConfig program s (boundaryCallDepth boundary) (controlShape [] (siteCode (codePosition index [] 0)))
    memory boundary (stackConfig (mk_state s (activeFrame f locals)) [] (guestBody f)).
Proof.
  intros program s index f locals memory boundary Hstore Hwf Hvalid Hf Hlen Hbounds Hcells Hfbp Hbase Hsp Hpc Hidle
    Hdepth.
  apply (control_state_circuit _ _ _ (controlCursor [] (atCode (frameView (codePosition index [] 0) [] [] locals))));
    [exact Hstore | exact Hwf | exact Hvalid | ].
  exists f, (stackConfig (mk_state s (activeFrame f locals)) [] (guestBody f)).
  split; [exact Hf | ]. split.
  - pose proof (FrameRows.idle_active_state program s memory boundary f
      (frameView (codePosition index [] 0) [] [] locals)) as Hstate.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hstate.
    unfold innerCode in Hstate. cbn [pathCode] in Hstate.
    replace (stackConfig (mk_state s (activeFrame f locals)) [] (guestBody f)) with
      (frameConfig f (frameView (codePosition index [] 0) [] [] locals)
        (stackConfig (mk_state s (activeFrame f locals)) [] (skipn 0 (guestBody f)))) by reflexivity.
    apply Hstate; clear Hstate.
    + apply FrameRows.frame_at_unmoved_intro; unfold viewOperands;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset concat app];
        try assumption; try reflexivity.
      * constructor.
      * intros [ | i] v Hv; discriminate.
    + cbn. lia.
    + unfold viewOperands. cbn. lia.
    + rewrite Hpc. cbn. lia.
    + exact Hidle.
  - split; [exact I | ]. split; [exact Hdepth | ]. split; [cbn; lia | reflexivity].
Qed.

Definition conditionalTail := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1)].
Definition conditionalCode := [conditionalBlock (mk_uN 0) conditionalTail; LOCAL_GET (mk_uN 1)].
Definition conditionalModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition conditionalFunction := i32Function conditionalModule (mk_uN 0) 0 1 2 conditionalCode.
Definition conditionalProgram := [conditionalFunction].
Definition conditionalStore := MKstore conditionalProgram [] [] [] [] [].
Definition conditionalState condition := mk_state conditionalStore
  (MKframe [i32Value condition; i32Value 9] conditionalModule).
(* The frame's two locals at cells 7 and 8, and a stale value at cell 9, its
   empty operand stack's first cell. *)
Definition conditionalRam condition : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then
    match address with [14] => Some condition | [16] => Some 9 | [18] => Some 42 | _ => None end
  else None.
Definition conditionalMemory condition := loadProgram conditionalProgram (conditionalRam condition).

Definition conditionalContext := moduleContext [i32StackType 0 1] [i32StackType 0 1].
Definition conditionalBodyContext := functionContext conditionalContext conditionalFunction.

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
  Instrs_ok (labelContext [] conditionalBodyContext) (LOCAL_GET (mk_uN 0) :: BR_IF (mk_uN 0) :: conditionalTail)
    (i32StackType 0 0).
Proof.
  change (Instrs_ok (labelContext [] conditionalBodyContext)
    ([LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0)] ++ conditionalTail) (i32StackType 0 0)).
  eapply res_seq with (t_2_lst := []).
  - change (Instrs_ok (labelContext [] conditionalBodyContext)
      ([LOCAL_GET (mk_uN 0)] ++ [BR_IF (mk_uN 0)]) (i32StackType 0 0)).
    eapply res_seq with (t_2_lst := [valtype_I32]).
    + apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + apply Instrs_ok__instr; [apply (br_if _ _ []) | | ]; try reflexivity; repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - change (Instrs_ok (labelContext [] conditionalBodyContext)
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


Example conditional_code_validates : Instrs_ok conditionalBodyContext conditionalCode (i32StackType 0 1).
Proof.
  change (Instrs_ok conditionalBodyContext
    ([conditionalBlock (mk_uN 0) conditionalTail] ++ [LOCAL_GET (mk_uN 1)]) (i32StackType 0 1)).
  eapply res_seq with (t_2_lst := []).
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | exact conditional_block_validates |
      repeat constructor | repeat constructor | repeat constructor].
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
  supportedCode (programCallees conditionalProgram conditionalModule) conditionalCode.
Proof.
  constructor; [ | constructor; [constructor | constructor]].
  apply supported_block; [ | left; reflexivity]. repeat constructor.
Qed.

Example conditional_program_supported : programSupported conditionalProgram.
Proof.
  intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
  split; [exact conditional_body_supported | vm_compute; repeat split; lia].
Qed.

Example conditional_initial_config : forall condition,
  0 <= condition < i32Modulus ->
  CircuitConfig conditionalProgram conditionalStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    (conditionalMemory condition) (mkBoundary 0 9 9 7 0 0 0 0 0 0)
    (stackConfig (conditionalState condition) [] conditionalCode).
Proof.
  intros condition Hcondition.
  apply (function_start_config conditionalProgram conditionalStore 0
    (mkGuestFunction 0%N conditionalModule (mk_uN 0) 0 1 2 conditionalCode 0) [condition; 9] _
    (mkBoundary 0 9 9 7 0 0 0 0 0 0)); try reflexivity; try (cbn; lia).
  - repeat constructor.
  - split; [constructor; [apply (conditional_function_validates 0) | constructor] | exact conditional_program_supported].
  - repeat constructor; unfold i32Modulus in *; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* Raw rows may stop anywhere inside the function, including inside the label. *)
Example conditional_program_refines : forall condition rows final,
  fieldPrime -> 0 <= condition < i32Modulus ->
  validCircuitRun (conditionalMemory condition) rows final ->
  runFragment (mkBoundary 0 9 9 7 0 0 0 0 0 0) 0 rows ->
  exists shape c,
    Steps (stackConfig (conditionalState condition) [] conditionalCode) c /\
    programLoaded conditionalProgram final /\
    CircuitConfig conditionalProgram conditionalStore 0 shape final
      (rowsFinalBoundary (mkBoundary 0 9 9 7 0 0 0 0 0 0) rows) c.
Proof.
  intros condition rows final Hprime Hcondition Hrun Hfragment.
  exact (run_refinement conditionalProgram conditionalStore 0 _ rows (conditionalMemory condition) final _ _ Hprime
    (load_program_loaded _ _) Hrun Hfragment (conditional_initial_config condition Hcondition)).
Qed.
