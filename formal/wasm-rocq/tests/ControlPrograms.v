From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range Add TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec
  ProgramEncoding ProgramRefinement ControlDecode ControlInvariant ControlBridge Fragment tests.BlockPrograms.
Import ListNotations.
Open Scope Z_scope.

Definition computedBranches :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BINOP I32 (mk_binop__0 Inn_I32 SUB);
   BR_IF (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 0);
   LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1)].
Definition computedCode := [BLOCK (_RESULT None) computedBranches; LOCAL_GET (mk_uN 1)].
Definition computedModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition computedFunction := i32Function computedModule (mk_uN 0) 0 1 2 computedCode.
Definition computedProgram := [computedFunction].
Definition computedStore := MKstore computedProgram [] [] [] [] [].
Definition computedState condition := mk_state computedStore
  (MKframe [i32Value condition; i32Value 9] computedModule).
Definition computedRam condition : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then match address with [8] => Some 42 | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some condition | [7; 1] => Some 9 | _ => None end
  else None.
Definition computedMemory condition := loadProgram computedProgram (computedRam condition).

Definition computedContext := moduleContext [i32StackType 0 1] [i32StackType 0 1].
Definition computedBodyContext := functionContext computedContext computedFunction.
Definition computedGuest := mkGuestFunction 0%N computedModule (mk_uN 0) 0 1 2 computedCode 0.

Section Typing.
Local Open Scope nat_scope.
Example computed_branches_validate :
  Instrs_ok (labelContext [] computedBodyContext) computedBranches (i32StackType 0 0).
Proof.
  unfold computedBranches.
  typed_step 0 1 0. typed_step 0 1 1. typed_step 2 1 0. typed_step 1 0 0.
  typed_step 0 1 0. typed_step 1 0 0. typed_step 0 1 0. typed_instr 1 0 0.
Qed.
End Typing.

Example computed_code_validates : Instrs_ok computedBodyContext computedCode (i32StackType 0 1).
Proof.
  change (Instrs_ok computedBodyContext
    ([BLOCK (_RESULT None) computedBranches] ++ [LOCAL_GET (mk_uN 1)]) (i32StackType 0 1)).
  eapply res_seq with (t_2_lst := []).
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | exact computed_branches_validate |
      repeat constructor | repeat constructor | repeat constructor].
  - apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Example computed_module_validates : forall condition,
  Moduleinst_ok (fun_store (computedState condition)) computedModule computedContext.
Proof.
  intros. apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := computedFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example computed_function_validates : forall condition,
  Funcinst_ok (fun_store (computedState condition)) computedFunction (i32StackType 0 1).
Proof.
  intro condition. apply mk_Funcinst_ok with (C := computedContext).
  - repeat constructor.
  - apply computed_module_validates.
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
    + repeat constructor.
    + apply mk_Expr_ok; [exact computed_code_validates | repeat constructor | repeat constructor].
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

(* block (computed branches); local.get 1 *)
Example computed_body_supported :
  supportedCode (programCallees computedProgram computedModule) computedCode.
Proof.
  constructor; [ | constructor; [constructor | constructor]].
  apply supported_block. repeat constructor.
Qed.

Example computed_program_supported : programSupported computedProgram.
Proof.
  intros [ | index] g Hg; cbn in Hg; [injection Hg as <- | destruct index; discriminate].
  split; [exact computed_body_supported | vm_compute; repeat split; lia].
Qed.

Example computed_store_wf : wf_store computedStore.
Proof. repeat constructor. Qed.

Example computed_program_validated : programValidated computedProgram computedStore.
Proof.
  split; [constructor; [apply (computed_function_validates 0) | constructor] | exact computed_program_supported].
Qed.

Lemma computed_initial_config : forall condition,
  0 <= condition < i32Modulus ->
  CircuitConfig computedProgram computedStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    (computedMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) (stackConfig (computedState condition) [] computedCode).
Proof.
  intros condition Hcondition.
  apply (function_start_config computedProgram computedStore 0 computedGuest [condition; 9] _
    (mkBoundary 0 4 4 7 0 0 0 0 0)); try reflexivity; try (cbn; lia).
  - exact computed_store_wf.
  - exact computed_program_validated.
  - repeat constructor; unfold i32Modulus in *; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* Raw rows may stop anywhere inside the function, including inside the label. *)
Example computed_program_refines : forall condition rows final,
  fieldPrime -> 0 <= condition < i32Modulus ->
  validCircuitRun (computedMemory condition) rows final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 rows ->
  exists shape c,
    Steps (stackConfig (computedState condition) [] computedCode) c /\
    programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) rows) c.
Proof.
  intros condition rows final Hprime Hcondition Hrun Hfragment.
  exact (run_refinement computedProgram computedStore 0 _ rows _ final _ _ Hprime
    (load_program_loaded _ _) Hrun Hfragment
    (computed_initial_config condition Hcondition)).
Qed.

Example computed_branch_layout :
  encodeProgram computedProgram pcRomId [4; 0] = Some 5 /\
  encodeProgram computedProgram pcRomId [4; 1] = Some 10 /\
  encodeProgram computedProgram pcRomId [6; 0] = Some 7 /\
  encodeProgram computedProgram pcRomId [6; 1] = Some 10 /\
  encodeProgram computedProgram programOpcodeId [9] = Some CODE_END.
Proof. repeat split; reflexivity. Qed.

(* After local.get 0; local.get 1; i32.sub, the value is 7 - 9 modulo 2^32.
   It is the only value inside the block's label. *)
Definition computedInteriorMemory := loadProgram computedProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with
    | [8] => Some 4294967294 | _ => None end
  else computedRam 7 memory address).
Definition computedInteriorConfig := stackPrefix []
  (instructionSuffix [LOCAL_GET (mk_uN 1)]
    (SpecTecBranches.blockContext (stackConfig (computedState 7) [4294967294] (skipn 3 computedBranches)))).

(* Inside the label, the decoder splits the operand stack at the block's height. *)
Example computed_interior_config :
  CircuitConfig computedProgram computedStore 0 (controlShape [] (siteCode (codePosition 0 [0%nat] 3)))
    computedInteriorMemory (mkBoundary 4 5 4 7 0 0 0 0 0) computedInteriorConfig.
Proof.
  split; [reflexivity | ]. split; [exact computed_store_wf | ]. split; [exact computed_program_validated | ].
  exists True. split; [vm_compute; reflexivity | exact I].
Qed.

(* No branch outcome is needed, even when starting inside the block. The first
   run's output is precisely the second run's input. *)
Example computed_interior_two_runs : forall firstRows lastRows final,
  fieldPrime -> validCircuitRun computedInteriorMemory (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 4 5 4 7 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 4 5 4 7 0 0 0 0 0) firstRows) 0 lastRows ->
  exists shape c, Steps computedInteriorConfig c /\ programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final
      (rowsFinalBoundary (mkBoundary 4 5 4 7 0 0 0 0 0) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final Hprime Hrun Hfirst Hlast.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram computedStore 0 _ firstRows computedInteriorMemory between _ _ Hprime
    (load_program_loaded _ _) Hrun1 Hfirst computed_interior_config)
    as [shape1 [c1 [Hsteps1 [Hloaded1 Hconfig1]]]].
  destruct (run_refinement computedProgram computedStore 0 shape1 lastRows between final _ c1 Hprime Hloaded1 Hrun2
    Hlast Hconfig1) as [shape2 [c2 [Hsteps2 [Hloaded2 Hconfig2]]]].
  exists shape2, c2. split; [eapply spec_steps_transitive; eassumption | ].
  split; [exact Hloaded2 | ]. rewrite rows_final_boundary_app. exact Hconfig2.
Qed.

Lemma no_block_path : forall code b rest,
  Forall (fun i => forall bt body, i <> BLOCK bt body) code -> pathCode code (b :: rest) = None.
Proof.
  intros code b rest H. cbn [pathCode].
  destruct (nth_error code b) as [i | ] eqn:Hb; [ | reflexivity]. destruct i; try reflexivity.
  exfalso. eapply (proj1 (Forall_forall _ _) H); [eapply nth_error_In; exact Hb | reflexivity].
Qed.

(* The only code paths: the function body and the block body. *)
Lemma computed_paths : forall labels,
  pathCode computedCode labels = None \/ labels = [] \/ labels = [0%nat].
Proof.
  intros [ | [ | b] more]; [right; left; reflexivity | | left].
  - destruct more as [ | b' more]; [right; right; reflexivity | left].
    change (pathCode computedBranches (b' :: more) = None). apply no_block_path.
    repeat constructor; intros ? ? ?; discriminate.
  - destruct b as [ | b]; [reflexivity | destruct b; reflexivity].
Qed.

Lemma computed_no_call : forall labels offset x,
  nth_error (innerCode computedGuest (codePosition 0 labels offset)) offset <> Some (CALL x).
Proof.
  intros labels offset x H. unfold innerCode in H. cbn [positionLabels] in H.
  change (guestBody computedGuest) with computedCode in H.
  destruct (computed_paths labels) as [Hnone | [-> | ->]].
  - rewrite Hnone in H. destruct offset; discriminate.
  - apply nth_error_In in H. cbn in H. intuition discriminate.
  - apply nth_error_In in H. cbn in H. intuition discriminate.
Qed.

(* At pc 4 the frame is at the first br_if, inside the block. *)
Lemma computed_pc_four_shape : forall cursor memory boundary c,
  ControlState computedProgram computedStore 0 cursor memory boundary c -> boundaryPC boundary = 4 ->
  shapeOf cursor = controlShape [] (siteCode (codePosition 0 [0%nat] 3)).
Proof.
  intros [stack active] memory boundary c [f [inner [Hf [Hactive [Hsusp [_ [_ ->]]]]]]] Hpc.
  cbn [activePosition suspendedCalls] in *.
  destruct active as [[[index labels offset] segments values locals] | function arguments].
  - destruct Hactive as [f' [[Hf' [Hpath _]] [Hoffset [_ [Hpos _]]]]].
    cbn [activeFunction viewPosition positionFunction positionLabels positionOffset] in *.
    rewrite Hf in Hf'. injection Hf' as <-.
    destruct index as [ | index]; [ | destruct index; discriminate]. cbn in Hf. injection Hf as <-.
    cbn [guestBody] in Hpath.
    destruct (computed_paths labels) as [Hnone | [-> | ->]].
    + rewrite Hnone in Hpath. discriminate.
    + exfalso. unfold innerCode in Hoffset. cbn in Hoffset.
      destruct offset as [ | [ | [ | offset]]]; cbn in Hpos; lia.
    + unfold innerCode in Hoffset. cbn in Hoffset.
      destruct offset as [ | [ | [ | [ | [ | [ | [ | [ | [ | offset]]]]]]]]]; cbn in Hpos; try lia.
      destruct stack as [ | top more]; [reflexivity | exfalso].
      destruct Hsusp as [x [[Hg _] [Hcall _]]].
      destruct (callView top) as [[[ | caller] glabels goffset] gsegments gvalues glocals];
        cbn [viewPosition positionFunction positionLabels positionOffset] in Hg, Hcall; [ | destruct caller; discriminate].
      cbn in Hg. injection Hg as Hg. rewrite <- Hg in Hcall. exact (computed_no_call _ _ _ Hcall).
  - exfalso. destruct Hactive as [f' [Hf' [_ [_ [_ [_ [_ [Hpos _]]]]]]]].
    destruct function as [ | function]; [ | destruct function; discriminate].
    injection Hf' as <-. cbn in Hpos. lia.
Qed.

(* Splitting the raw trace before the first br_if yields the interior shape. *)
Example computed_split_before_branch : forall firstRows lastRows final,
  fieldPrime -> validCircuitRun (computedMemory 7) (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) 0 lastRows ->
  boundaryPC (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) = 4 ->
  exists between c1 shape c,
    CircuitConfig computedProgram computedStore 0 (controlShape [] (siteCode (codePosition 0 [0%nat] 3)))
      between (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) c1 /\
    Steps (stackConfig (computedState 7) [] computedCode) c1 /\ Steps c1 c /\
    programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final
      (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final Hprime Hrun Hfirst Hlast Hpc.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram computedStore 0 _ firstRows (computedMemory 7) between _ _ Hprime
    (load_program_loaded _ _) Hrun1 Hfirst (computed_initial_config 7 ltac:(unfold i32Modulus; lia)))
    as [shape1 [c1 [Hsteps1 [Hloaded1 Hconfig1]]]].
  destruct (run_refinement computedProgram computedStore 0 shape1 lastRows between final _ c1 Hprime Hloaded1 Hrun2
    Hlast Hconfig1) as [shape2 [c2 [Hsteps2 [Hloaded2 Hconfig2]]]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig1) as [cursor Hstate1].
  exists between, c1, shape2, c2.
  split; [rewrite <- (computed_pc_four_shape _ _ _ _ Hstate1 Hpc);
    exact (control_state_circuit _ _ _ _ _ _ _ Logic.eq_refl computed_store_wf computed_program_validated Hstate1) | ].
  split; [exact Hsteps1 | ]. split; [exact Hsteps2 | ].
  split; [exact Hloaded2 | ]. rewrite rows_final_boundary_app. exact Hconfig2.
Qed.
