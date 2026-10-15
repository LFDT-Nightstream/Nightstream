From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals SpecTecCalls StackSimulation StraightLineSpec CallSpec CallerSpec CallerRefinement
  ProgramSpec ControlSpec ProgramEncoding ProgramRefinement ProgramValidation FunctionValidation CallerValidation SourceInstructions SourceValidation CodeLayout tests.Calls tests.Callers.
Import ListNotations.
Open Scope Z_scope.

(* Callee at PCs 0..1; caller at PCs 2..5. No hand-authored ROM tables. *)
Definition sourceProgram : Program :=
  [readZeroFunction; i32Function callModule (mk_uN 0) 0 1 1 getCallAdd].
Definition sourceInitialState := mk_state (MKstore sourceProgram [] [] [] [] [])
  (MKframe [i32Value 5] callModule).
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
    split; [repeat constructor | ]. split; [repeat constructor; eexists; reflexivity | ].
    split; [exact read_zero_validates | split; lia].
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

Example source_initial_validated :
  validatedCode sourceProgram
    (mkGuestFunction 1%N callModule (mk_uN 0) 0 1 1 getCallAdd 2)
    sourceInitialState 0 getCallAdd.
Proof.
  split.
  - constructor.
    + apply source_function_validates; [repeat constructor | ].
      apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
    + constructor; [apply source_function_validates; [repeat constructor | exact source_body_validates] | constructor].
  - split.
    + constructor.
      * constructor; [eexists; reflexivity | constructor].
      * constructor.
        -- constructor; [ | constructor].
           exists (mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 0).
           split; [reflexivity | ]. split; [repeat constructor; eexists; reflexivity | split; cbn; lia].
        -- constructor; [constructor; [eexists; reflexivity | constructor] | constructor].
    + exists sourceModuleContext, 1%nat. split; [exact source_module_validates | exact source_body_validates].
Qed.

Example source_initial_represents : forall stale,
  represents sourceInitialState (sourceMemory stale) (mkBoundary 2 4 4 7)
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
    (mkBoundary 2 4 4 7) getCallAdd (stackConfig sourceInitialState [] getCallAdd).
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
    (sourceMemory stale) (mkBoundary 2 4 4 7) getCallAdd
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
    (sourceMemory stale) (mkBoundary 2 4 4 7) getCallAdd
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

Lemma source_final_control_state : forall cursor z memory boundary c,
  ControlState sourceProgram cursor z memory boundary c -> boundaryPC boundary = 5 ->
  SourceState sourceProgram (sourcePosition 1 3) z memory boundary [] c.
Proof.
  intros [[index offset] | [index outer] offset prefix] z memory boundary c Hstate Hpc.
  - destruct Hstate as [code Hstate]. destruct index as [ | [ | index]].
    + destruct Hstate as [_ [[Hbound _] [_ [_ Hpos]]]].
      change (offset <= 1)%nat in Hbound.
      destruct offset as [ | [ | offset]]; cbn in Hpos; lia.
    + pose proof (proj2 (source_position_pc _ _ _ _ _ _ Hstate)) as Hpos.
      assert (offset = 3)%nat by lia. subst offset.
      assert (code = []) by exact (Logic.eq_sym (proj2 (proj1 (proj2 Hstate)))).
      subst code. exact Hstate.
    + unfold SourceState in Hstate. destruct index; contradiction.
  - destruct Hstate as [f [body [continuation [values [Hf [Hbody _]]]]]].
    destruct index as [ | [ | index]]; try (destruct index; discriminate); injection Hf as <-.
    + destruct outer as [ | outer]; [discriminate | destruct outer; discriminate].
    + destruct outer as [ | [ | [ | outer]]]; try discriminate; destruct outer; discriminate.
Qed.

(* The public theorem composes directly through its returned control state. *)
Example source_program_two_runs : forall stale firstRows lastRows final depth,
  fieldPrime -> validCircuitRun (sourceMemory stale) (firstRows ++ lastRows) final ->
  callerFragment (mkBoundary 2 4 4 7) depth firstRows ->
  callerFragment (rowsFinalBoundary (mkBoundary 2 4 4 7) firstRows) depth lastRows ->
  exists cursor z c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\ programLoaded sourceProgram final /\
    ControlState sourceProgram cursor z final
      (rowsFinalBoundary (mkBoundary 2 4 4 7) (firstRows ++ lastRows)) c.
Proof.
  intros stale firstRows lastRows final depth Hprime Hrun Hfirst Hlast.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement sourceProgram (atInstruction (sourcePosition 1 0)) firstRows sourceInitialState
    (sourceMemory stale) between (mkBoundary 2 4 4 7) depth _ Hprime (source_program_loaded stale)
    Hrun1 Hfirst ltac:(exists getCallAdd; exact (source_initial_matches stale)))
    as [cursor1 [z1 [c1 [Hsteps1 [Hloaded1 Hsource1]]]]].
  destruct (run_refinement sourceProgram cursor1 lastRows z1 between final
    (rowsFinalBoundary (mkBoundary 2 4 4 7) firstRows) depth c1 Hprime Hloaded1 Hrun2 Hlast Hsource1)
    as [cursor2 [z2 [c2 [Hsteps2 [Hloaded2 Hsource2]]]]].
  exists cursor2, z2, c2. split; [eapply spec_steps_transitive; eassumption | ].
  split; [exact Hloaded2 | ]. rewrite TraceSegmentation.rows_final_boundary_app. exact Hsource2.
Qed.
