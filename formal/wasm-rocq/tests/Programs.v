From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals SpecTecCalls StackSimulation StraightLineSpec CallSpec CallerSpec CallerRefinement
  ProgramSpec ControlDecode ControlInvariant ControlBridge ProgramEncoding ProgramRefinement ProgramValidation FunctionValidation CallerValidation SourceInstructions SourceValidation CodeLayout Fragment tests.Calls tests.Callers.
Import ListNotations.
Open Scope Z_scope.

(* Callee at PCs 0..1; caller at PCs 2..5. No hand-authored ROM tables. *)
Definition sourceProgram : Program :=
  [readZeroFunction; i32Function callModule (mk_uN 0) 0 1 1 getCallAdd].
Definition sourceStore := MKstore sourceProgram [] [] [] [] [].
Definition sourceInitialState := mk_state sourceStore (MKframe [i32Value 5] callModule).
Definition sourceRam (stale : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory localLoId then
    match address with [7; 0] => Some 5 | [8; 0] => Some stale | _ => None end
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

Example source_program_keeps_stale_locals : forall stale, sourceMemory stale localLoId [8; 0] = Some stale.
Proof. reflexivity. Qed.

Example source_initial_typed :
  callerFacts (programCallees sourceProgram callModule) sourceInitialState 0 getCallAdd.
Proof.
  cbn [getCallAdd callerFacts instructionStackHeight].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; [vm_compute; repeat constructor | ].
  change (typedCall sourceInitialState (mk_uN 0) callModule (mk_uN 0) 0 1 1 readZeroBody /\
    (0 <= 1)%nat /\ callerFacts (programCallees sourceProgram callModule) sourceInitialState 2
      [BINOP I32 (mk_binop__0 Inn_I32 ADD)]).
  split.
  - unfold typedCall. split; [repeat constructor | ]. split; [repeat constructor | ].
    split; [repeat constructor | ]. split; lia.
  - split; [lia | ]. split; [repeat constructor; eexists; reflexivity | ].
    split; [repeat constructor | ]. split; exact I.
Qed.

Definition sourceModuleContext :=
  MKcontext [i32StackType 0 1] [i32StackType 0 1] [] [] [] [] [] [] [] None.

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
Example source_body_supported : supportedCode (functionSite (programCallees sourceProgram callModule)) getCallAdd.
Proof.
  constructor; [constructor | constructor; [ | constructor; [constructor | constructor]]].
  apply supported_call with (f := mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 0);
    [reflexivity | cbn; lia | cbn; lia].
Qed.

Example source_program_supported : programSupported sourceProgram.
Proof.
  intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- | injection Hg as <- | destruct index; discriminate].
  - repeat constructor.
  - exact source_body_supported.
Qed.

Example source_initial_validated :
  validatedCode sourceProgram
    (mkGuestFunction 1%N callModule (mk_uN 0) 0 1 1 getCallAdd 2)
    sourceInitialState 0 getCallAdd.
Proof.
  split; [split; [ | exact source_program_supported] | ].
  - constructor.
    + apply source_function_validates; [repeat constructor | ].
      apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + constructor; [apply source_function_validates; [repeat constructor | exact source_body_validates] | constructor].
  - split; [exact source_body_supported | ].
    exists sourceModuleContext. split; [exact source_module_validates | exact source_body_validates].
Qed.

Example source_initial_represents : forall stale,
  represents sourceInitialState (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0)
    getCallAdd (stackConfig sourceInitialState [] getCallAdd).
Proof.
  intros stale. exists []. split.
  - split; [cbn; lia | ]. split; [reflexivity | ]. split; [constructor | ]. intros [ | n] v H; discriminate.
  - split; [reflexivity | ]. split.
    + apply stack_config_wf; [repeat constructor | constructor | ].
      eapply caller_program_wf. exact source_initial_typed.
    + intros [ | n] value H; cbn [sourceInitialState stateLocals LOCALS nth_error] in H.
      * inversion H; subst. exists 5. split; [reflexivity | ]. split; [unfold i32Modulus; lia | reflexivity].
      * destruct n; discriminate.
Qed.

Example source_initial_matches : forall stale,
  SourceState sourceProgram (sourcePosition 1 0) sourceInitialState (sourceMemory stale)
    (mkBoundary 2 4 4 7 0 0 0 0 0) getCallAdd (stackConfig sourceInitialState [] getCallAdd).
Proof.
  intros stale. split; [repeat split; reflexivity | ]. split.
  - split; [cbn; lia | reflexivity].
  - split; [exact source_initial_validated | ]. split; [apply source_initial_represents | reflexivity].
Qed.

Example wrong_function_entry_rejected : forall stale,
  ~ programLoaded sourceProgram (update (sourceMemory stale) functionEntryId [1] 99).
Proof.
  intros stale H. specialize (H functionEntryId ltac:(unfold programRomIds; simpl; tauto) [1]).
  rewrite update_same in H. discriminate.
Qed.

Example caller_local_count_mismatch_rejected : forall stale,
  ~ SourceState sourceProgram (sourcePosition 1 0)
    (mk_state (MKstore sourceProgram [] [] [] [] []) (MKframe [i32Value 5; i32Value 9] callModule))
    (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0) getCallAdd
    (stackConfig sourceInitialState [] getCallAdd).
Proof. intros stale [[_ [_ Hcount]] _]. discriminate Hcount. Qed.

Example non_i32_function_rejected :
  functionAt [MKfuncinst (mk_functype (mk_list _ [valtype_I64]) (mk_list _ [])) callModule
    (func_FUNC (mk_uN 0) [] [])] 0 = None.
Proof. reflexivity. Qed.

Example wrong_module_function_addresses_rejected : forall stale,
  ~ SourceState sourceProgram (sourcePosition 1 0)
    (mk_state (MKstore sourceProgram [] [] [] [] [])
      (MKframe [i32Value 5] (MKmoduleinst [] [1%N] [] [] [] [] [] [])))
    (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0) getCallAdd
    (stackConfig sourceInitialState [] getCallAdd).
Proof. intros stale [[_ [Haddresses _]] _]. discriminate Haddresses. Qed.

(* PC layout makes the source offset observable, including after a prefix. *)
Lemma source_position_pc : forall offset z memory boundary code c,
  SourceState sourceProgram (sourcePosition 1 offset) z memory boundary code c ->
  (offset <= 3)%nat /\ boundaryPC boundary = 2 + Z.of_nat offset.
Proof.
  intros offset z memory boundary code c H.
  pose proof (proj1 (proj1 (proj2 H))) as Hbound.
  change (offset <= 3)%nat in Hbound. split; [exact Hbound | ].
  pose proof (proj2 (proj2 (proj2 (proj2 H)))) as Hpc.
  destruct offset as [ | [ | [ | [ | offset]]]]; try lia; exact Hpc.
Qed.

Example source_store_wf : wf_store sourceStore.
Proof. repeat constructor. Qed.

Example source_initial_config : forall stale,
  CircuitConfig sourceProgram sourceStore 0 (controlShape [] (siteInstruction (sourcePosition 1 0)))
    (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0) (stackConfig sourceInitialState [] getCallAdd).
Proof.
  intros stale.
  apply (control_state_circuit sourceProgram sourceStore 0 (controlCursor [] (atInstruction (sourcePosition 1 0)))
    sourceInitialState); [reflexivity | exact source_store_wf | reflexivity | ].
  apply (base_control_state sourceProgram (atInstruction (sourcePosition 1 0)) sourceInitialState
    (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0)); [ | cbn; lia].
  split; [exists getCallAdd; exact (source_initial_matches stale) | repeat split].
Qed.

(* The decoder reads the caller's one local from memory. *)
Example source_initial_decodes : forall stale,
  option_map fst (decodeControl sourceProgram sourceStore 0 (controlShape [] (siteInstruction (sourcePosition 1 0)))
    (sourceMemory stale) (mkBoundary 2 4 4 7 0 0 0 0 0)) = Some (stackConfig sourceInitialState [] getCallAdd).
Proof. intros stale. vm_compute. reflexivity. Qed.

(* A pc off the position, a missing operand cell, or a depth without a saved
   caller does not decode. *)
Example source_wrong_pc_rejected : forall stale,
  decodeControl sourceProgram sourceStore 0 (controlShape [] (siteInstruction (sourcePosition 1 0)))
    (sourceMemory stale) (mkBoundary 3 4 4 7 0 0 0 0 0) = None.
Proof. intros stale. vm_compute. reflexivity. Qed.

Example source_missing_operand_rejected : forall stale,
  decodeControl sourceProgram sourceStore 0 (controlShape [] (siteInstruction (sourcePosition 1 0)))
    (sourceMemory stale) (mkBoundary 2 5 4 7 0 0 0 0 0) = None.
Proof. intros stale. vm_compute. reflexivity. Qed.

Example source_unsaved_caller_rejected :
  decodeControl sourceProgram sourceStore 0 (controlShape [sourcePosition 1 1] (siteInstruction (sourcePosition 0 0)))
    (sourceMemory 3) (mkBoundary 0 4 4 8 1 0 0 0 0) = None.
Proof. vm_compute. reflexivity. Qed.

(* The caller is suspended at its CALL with [5] below the callee's stack base.
   Its return context is saved at depth 0: return to pc 4, frame base 7, stack base 4. *)
Definition suspendedMemory (calleeLocal : option Z) := loadProgram sourceProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with [8] => Some 5 | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some 5 | [8; 0] => calleeLocal | _ => None end
  else if Nat.eqb memory callStackReturnPcId then match address with [0] => Some 4 | _ => None end
  else if Nat.eqb memory callStackCallerFbpId then match address with [0] => Some 7 | _ => None end
  else if Nat.eqb memory callStackCallerSpBaseId then match address with [0] => Some 4 | _ => None end
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
    (controlShape [sourcePosition 1 1] (siteEntering 0))
    (suspendedMemory None) (mkBoundary 0 5 5 8 1 1 0 1 0)) =
  Some (suspendedConfig (MKframe [i32Value 0] callModule)).
Proof. vm_compute. reflexivity. Qed.

(* Once initialized, the callee's local is read from its own frame base. *)
Example source_suspended_call_decodes :
  option_map fst (decodeControl sourceProgram sourceStore 0
    (controlShape [sourcePosition 1 1] (siteInstruction (sourcePosition 0 0)))
    (suspendedMemory (Some 0)) (mkBoundary 0 5 5 8 1 0 0 0 0)) =
  Some (suspendedConfig (MKframe [i32Value 0] callModule)).
Proof. vm_compute. reflexivity. Qed.

(* At the caller's END no call is suspended: the program's only CALL targets
   readZero, so nothing calls the caller, and readZero's PCs are 0..1. *)
Lemma source_final_shape : forall cursor z memory boundary c,
  ControlState sourceProgram 0 cursor z memory boundary c -> boundaryPC boundary = 5 ->
  shapeOf cursor = controlShape [] (siteInstruction (sourcePosition 1 3)).
Proof.
  intros [stack active] z memory boundary c [f [inner [Hf [Hactive [Hsusp [_ [_ ->]]]]]]] Hpc.
  cbn [activePosition suspendedCalls] in *.
  destruct active as [[index offset] | [index outer] offset prefix | function arguments].
  - destruct Hactive as [[code Hstate] _]. destruct index as [ | [ | index]].
    + destruct Hstate as [_ [[Hbound _] [_ [_ Hpos]]]].
      change (offset <= 1)%nat in Hbound.
      destruct offset as [ | [ | offset]]; cbn in Hpos; lia.
    + pose proof (proj2 (source_position_pc _ _ _ _ _ _ Hstate)) as Hpos.
      assert (offset = 3)%nat by lia. subst offset.
      assert (code = []) by exact (Logic.eq_sym (proj2 (proj1 (proj2 Hstate)))).
      subst code. cbn [activeFunction sourceFunction] in Hf. injection Hf as <-.
      destruct stack as [ | top more].
      * reflexivity.
      * exfalso. destruct Hsusp as [g [x [Hg [Hcall [_ [Hcallee _]]]]]].
        destruct (callPosition top) as [[ | [ | caller]] site]; cbn [sourceFunction sourceOffset] in Hg, Hcall;
          [ | | destruct caller; discriminate]; injection Hg as <-; cbn [guestBody guestModule] in Hcall, Hcallee.
        -- destruct site as [ | site]; [discriminate | destruct site; discriminate].
        -- destruct site as [ | [ | [ | site]]]; try discriminate; [ | destruct site; discriminate].
           injection Hcall as <-. discriminate Hcallee.
    + unfold SourceState in Hstate. destruct index; contradiction.
  - destruct Hactive as [f' [body [continuation [values [Hf' [Hbody _]]]]]].
    destruct index as [ | [ | index]]; try (destruct index; discriminate); injection Hf' as <-.
    + destruct outer as [ | outer]; [discriminate | destruct outer; discriminate].
    + destruct outer as [ | [ | [ | outer]]]; try discriminate; destruct outer; discriminate.
  - exfalso. destruct Hactive as [f' [Hf' [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [Hpos _]]]]]]]]]]]]].
    destruct function as [ | [ | function]]; try (destruct function; discriminate);
      injection Hf' as <-; cbn in Hpos; lia.
Qed.

(* The public theorem composes directly through its returned control shape. The
   split may fall anywhere, including inside the call or its frame initialization. *)
Example source_program_two_runs : forall stale firstRows lastRows final,
  fieldPrime -> validCircuitRun (sourceMemory stale) (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 2 4 4 7 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 2 4 4 7 0 0 0 0 0) firstRows) 0 lastRows ->
  exists shape c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\ programLoaded sourceProgram final /\
    CircuitConfig sourceProgram sourceStore 0 shape final
      (rowsFinalBoundary (mkBoundary 2 4 4 7 0 0 0 0 0) (firstRows ++ lastRows)) c.
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
