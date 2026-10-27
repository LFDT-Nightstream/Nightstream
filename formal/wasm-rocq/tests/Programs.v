From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals SpecTecCalls StackSimulation StraightLineSpec CallSpec CallerSpec CallerRefinement
  ProgramSpec ControlDecode ControlInvariant ControlBridge ProgramEncoding ProgramRefinement FunctionValidation
  CallerValidation CodeLayout Fragment tests.Calls tests.Callers.
Import ListNotations.
Open Scope Z_scope.

(* Callee at PCs 0..1; caller at PCs 2..5. No hand-authored ROM tables. *)
Definition sourceProgram : Program :=
  [readZeroFunction; i32Function callModule (mk_uN 0) 0 1 1 getCallAdd].
Definition sourceStore := MKstore sourceProgram [] [] [] [] [].
Definition sourceInitialState := mk_state sourceStore (MKframe [i32Value 5] callModule).
(* The caller's frame starts at cell 7 (its one local, 5); a stale value sits at
   cell 9, where the callee's local will be. *)
Definition sourceRam (stale : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then
    match address with [14] => Some 5 | [18] => Some stale | _ => None end
  else None.
Definition sourceMemory (stale : Z) := loadProgram sourceProgram (sourceRam stale).

Example source_program_layout :
  functionEntry sourceProgram 0 = 0 /\ functionEntry sourceProgram 1 = 2 /\
  encodeProgram sourceProgram programOpcodeId [3] = Some CODE_CALL /\
  encodeProgram sourceProgram callTargetId [3] = Some 1 /\
  encodeProgram sourceProgram pcRomId [3; 0] = Some 0 /\
  encodeProgram sourceProgram pcRomId [3; 1] = Some 4 /\
  encodeProgram sourceProgram programLocalIndexId [4] = Some 0 /\
  encodeProgram sourceProgram functionLocalCountId [2] = Some 1 /\
  encodeProgram sourceProgram programOpcodeId [5] = Some CODE_END /\
  encodeProgram sourceProgram pcEdgeKindId [5] = Some 1 /\
  encodeProgram sourceProgram pcRomId [5; 0] = None.
Proof. repeat split; reflexivity. Qed.

Example source_program_loaded : forall stale, programLoaded sourceProgram (sourceMemory stale).
Proof. intros stale. apply load_program_loaded. Qed.

Example source_program_keeps_stale_locals : forall stale, sourceMemory stale stackId [18] = Some stale.
Proof. reflexivity. Qed.

Definition sourceModuleContext :=
  moduleContext [i32StackType 0 1] [i32StackType 0 1].

Example source_module_validates :
  Moduleinst_ok (fun_store sourceInitialState) callModule sourceModuleContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := readZeroFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Definition sourceBodyContext := functionContext sourceModuleContext readZeroFunction.

Example source_body_validates :
  Instrs_ok sourceBodyContext getCallAdd (i32StackType 0 1).
Proof.
  change (Instrs_ok sourceBodyContext
    ([LOCAL_GET (mk_uN 0)] ++ [CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)]) (i32StackType 0 1)).
  eapply res_seq with (t_2_lst := [valtype_I32]).
  - apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - change (Instrs_ok sourceBodyContext
      ([CALL (mk_uN 0)] ++ [BINOP I32 (mk_binop__0 Inn_I32 ADD)]) (i32StackType 1 1)).
    eapply res_seq with (t_2_lst := [valtype_I32; valtype_I32]).
    + apply (i32_validation_frame _ _ 0 1 1).
      * apply call; try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
      * repeat constructor.
      * repeat constructor.
    + apply Instrs_ok__instr; [apply binop | | ]; repeat constructor.
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Lemma source_function_validates : forall body,
  Forall wf_instr body -> Instrs_ok sourceBodyContext body (i32StackType 0 1) ->
  Funcinst_ok (fun_store sourceInitialState) (i32Function callModule (mk_uN 0) 0 1 1 body) (i32StackType 0 1).
Proof.
  intros body Hw Hv. apply mk_Funcinst_ok with (C := sourceModuleContext).
  - repeat constructor.
  - exact source_module_validates.
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
    + repeat constructor.
    + apply mk_Expr_ok; [exact Hv | repeat constructor | exact Hw].
    + repeat constructor.
    + repeat constructor; exact Hw.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor; exact Hw.
Qed.

(* local.get 0; call 0 (to readZero); i32.add *)
Example source_body_supported : supportedCode (programCallees sourceProgram callModule) getCallAdd.
Proof.
  constructor; [constructor | constructor; [ | constructor; [constructor | constructor]]].
  apply supported_call with (f := mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 0);
    [reflexivity | cbn; lia | cbn; lia].
Qed.

Example source_program_supported : programSupported sourceProgram.
Proof.
  intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- | injection Hg as <- | destruct index; discriminate].
  - split; [repeat constructor | vm_compute; tauto].
  - split; [exact source_body_supported | vm_compute; tauto].
Qed.

Example source_store_wf : wf_store sourceStore.
Proof. repeat constructor. Qed.

Example source_program_validated : programValidated sourceProgram sourceStore.
Proof.
  split; [ | exact source_program_supported]. constructor.
  - apply source_function_validates; [repeat constructor | ].
    apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - constructor; [apply source_function_validates; [repeat constructor | exact source_body_validates] | constructor].
Qed.

Example wrong_function_entry_rejected : forall stale,
  ~ programLoaded sourceProgram (update (sourceMemory stale) functionEntryId [1] 99).
Proof.
  intros stale H. specialize (H functionEntryId ltac:(unfold programRomIds; simpl; tauto) [1]).
  rewrite update_same in H. discriminate.
Qed.

Example non_i32_function_rejected :
  functionAt [MKfuncinst (mk_functype (mk_list _ [valtype_I64]) (mk_list _ [])) callModule
    (func_FUNC (mk_uN 0) [] [])] 0 = None.
Proof. reflexivity. Qed.

Example source_initial_config : forall stale,
  CircuitConfig sourceProgram sourceStore 0 (controlShape [] (siteCode (codePosition 1 [] 0)))
    (sourceMemory stale) (mkBoundary 2 8 8 7 0 0 0 0 0 0) (stackConfig sourceInitialState [] getCallAdd).
Proof.
  intros stale. split; [reflexivity | ]. split; [exact source_store_wf | ]. split; [exact source_program_validated | ].
  exists True. split; [vm_compute; reflexivity | exact I].
Qed.

(* The decoder reads the caller's one local from memory. *)
Example source_initial_decodes : forall stale,
  option_map fst (decodeControl sourceProgram sourceStore 0 (controlShape [] (siteCode (codePosition 1 [] 0)))
    (sourceMemory stale) (mkBoundary 2 8 8 7 0 0 0 0 0 0)) = Some (stackConfig sourceInitialState [] getCallAdd).
Proof. intros stale. vm_compute. reflexivity. Qed.

(* A pc off the position, a missing operand cell, or a depth without a saved
   caller does not decode. *)
Example source_wrong_pc_rejected : forall stale,
  decodeControl sourceProgram sourceStore 0 (controlShape [] (siteCode (codePosition 1 [] 0)))
    (sourceMemory stale) (mkBoundary 3 8 8 7 0 0 0 0 0 0) = None.
Proof. intros stale. vm_compute. reflexivity. Qed.

Example source_missing_operand_rejected : forall stale,
  decodeControl sourceProgram sourceStore 0 (controlShape [] (siteCode (codePosition 1 [] 0)))
    (sourceMemory stale) (mkBoundary 2 9 8 7 0 0 0 0 0 0) = None.
Proof. intros stale. vm_compute. reflexivity. Qed.

Example source_unsaved_caller_rejected :
  decodeControl sourceProgram sourceStore 0 (controlShape [codePosition 1 [] 1] (siteCode (codePosition 0 [] 0)))
    (sourceMemory 3) (mkBoundary 0 10 10 9 1 0 0 0 0 0) = None.
Proof. vm_compute. reflexivity. Qed.

(* The caller is suspended at its CALL: its local at cell 7, its operand [5] at
   cell 8, and the callee's frame from cell 9. Its return context is saved at
   depth 0: return to pc 4, frame base 7, operand base 8. *)
Definition suspendedMemory (calleeLocal : option Z) := loadProgram sourceProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [14] => Some 5 | [16] => Some 5 | [18] => calleeLocal | _ => None end
  else if Nat.eqb memory callStackReturnPcId then match address with [0] => Some 4 | _ => None end
  else if Nat.eqb memory callStackCallerFbpId then match address with [0] => Some 7 | _ => None end
  else if Nat.eqb memory callStackCallerSpBaseId then match address with [0] => Some 8 | _ => None end
  else None).

(* SpecTec's configuration inside the call: the caller's value, then the callee
   in the caller's FRAME_/LABEL_ context, then the caller's continuation. *)
Definition suspendedConfig (calleeFrame : frame) :=
  mk_config (mk_state sourceStore (MKframe [i32Value 5] callModule))
    [i32Constant 5;
     FRAME_ 1%N calleeFrame [LABEL_ 1%N [] [admininstr_instr (LOCAL_GET (mk_uN 0))]];
     admininstr_instr (BINOP I32 (mk_binop__0 Inn_I32 ADD))].

(* While the callee's local is still being zeroed, SpecTec has already entered
   the callee; its local is not yet in memory. *)
Example source_initializing_callee_decodes :
  option_map fst (decodeControl sourceProgram sourceStore 0
    (controlShape [codePosition 1 [] 1] (siteEntering 0))
    (suspendedMemory None) (mkBoundary 0 9 10 9 1 1 0 1 0 0)) =
  Some (suspendedConfig (MKframe [i32Value 0] callModule)).
Proof. vm_compute. reflexivity. Qed.

(* Once initialized, the callee's local is read from its own frame base. *)
Example source_suspended_call_decodes :
  option_map fst (decodeControl sourceProgram sourceStore 0
    (controlShape [codePosition 1 [] 1] (siteCode (codePosition 0 [] 0)))
    (suspendedMemory (Some 0)) (mkBoundary 0 10 10 9 1 0 0 0 0 0)) =
  Some (suspendedConfig (MKframe [i32Value 0] callModule)).
Proof. vm_compute. reflexivity. Qed.

(* At the caller's END no call is suspended: the program's only CALL targets
   readZero, so nothing calls the caller, and readZero's PCs are 0..1. *)
Lemma source_final_shape : forall cursor memory boundary c,
  ControlState sourceProgram sourceStore 0 cursor memory boundary c -> boundaryPC boundary = 5 ->
  shapeOf cursor = controlShape [] (siteCode (codePosition 1 [] 3)).
Proof.
  intros [stack active] memory boundary c [f [inner [Hf [Hactive [Hsusp [_ [_ ->]]]]]]] Hpc.
  cbn [activePosition suspendedCalls] in *.
  destruct active as [[[index labels offset] segments values locals] | function arguments].
  - destruct Hactive as [f' [[Hf' [Hpath _]] [Hoffset [_ [Hpos _]]]]].
    cbn [activeFunction viewPosition positionFunction positionLabels positionOffset] in *.
    rewrite Hf in Hf'. injection Hf' as <-.
    destruct index as [ | [ | index]]; [ | | destruct index; discriminate]; injection Hf as <-;
      (destruct labels as [ | b more]; [ | destruct b as [ | [ | [ | [ | b]]]]; discriminate]);
      unfold innerCode in Hoffset; cbn in Hoffset, Hpos.
    + destruct offset as [ | [ | offset]]; cbn in Hpos; lia.
    + destruct offset as [ | [ | [ | [ | offset]]]]; cbn in Hpos; try lia.
      destruct stack as [ | top more]; [reflexivity | exfalso].
      destruct Hsusp as [x [[Hg _] [Hcall [Hcallee _]]]].
      destruct (callView top) as [[[ | [ | caller]] glabels goffset] gsegments gvalues glocals];
        cbn [viewPosition positionFunction positionLabels positionOffset] in Hg, Hcall, Hcallee;
        [ | | destruct caller; discriminate]; cbn in Hg; injection Hg as Hg; rewrite <- Hg in Hcall, Hcallee;
        unfold innerCode in Hcall; (destruct glabels as [ | b more']; [ | destruct b as [ | [ | [ | [ | b]]]]; cbn in Hcall; destruct goffset; discriminate]);
        cbn in Hcall.
      * destruct goffset as [ | [ | goffset]]; try discriminate; destruct goffset; discriminate.
      * destruct goffset as [ | [ | [ | goffset]]]; try discriminate; [ | destruct goffset; discriminate].
        injection Hcall as <-. discriminate Hcallee.
  - exfalso. destruct Hactive as [f' [Hf' [_ [_ [_ [_ [_ [_ [Hpos _]]]]]]]]].
    destruct function as [ | [ | function]]; try (destruct function; discriminate);
      injection Hf' as <-; cbn in Hpos; lia.
Qed.

(* The public theorem composes directly through its returned control shape. The
   split may fall anywhere, including inside the call or its frame initialization. *)
Example source_program_two_runs : forall stale firstRows lastRows final,
  fieldPrime -> validCircuitRun (sourceMemory stale) (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 2 8 8 7 0 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 2 8 8 7 0 0 0 0 0 0) firstRows) 0 lastRows ->
  exists shape c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\ programLoaded sourceProgram final /\
    CircuitConfig sourceProgram sourceStore 0 shape final
      (rowsFinalBoundary (mkBoundary 2 8 8 7 0 0 0 0 0 0) (firstRows ++ lastRows)) c.
Proof.
  intros stale firstRows lastRows final Hprime Hrun Hfirst Hlast.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement sourceProgram sourceStore 0 _ firstRows (sourceMemory stale) between _ _ Hprime
    (source_program_loaded stale) Hrun1 Hfirst (source_initial_config stale))
    as [shape1 [c1 [Hsteps1 [Hloaded1 Hconfig1]]]].
  destruct (run_refinement sourceProgram sourceStore 0 shape1 lastRows between final _ c1 Hprime Hloaded1 Hrun2
    Hlast Hconfig1) as [shape2 [c2 [Hsteps2 [Hloaded2 Hconfig2]]]].
  exists shape2, c2. split; [eapply spec_steps_transitive; eassumption | ].
  split; [exact Hloaded2 | ]. rewrite rows_final_boundary_app. exact Hconfig2.
Qed.
