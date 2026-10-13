From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals SpecTecCalls StraightLineSpec CallSpec CallerSpec CallerRefinement
  ProgramSpec ProgramEncoding ProgramRefinement tests.Calls tests.Callers.
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
  callerTyped (programCallees sourceProgram callModule) sourceInitialState 0 getCallAdd.
Proof.
  cbn [getCallAdd callerTyped instructionStackHeight].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; [vm_compute; repeat constructor | ].
  change (typedCall sourceInitialState (mk_uN 0) callModule (mk_uN 0) 0 1 1 readZeroBody /\
    (0 <= 1)%nat /\ callerTyped (programCallees sourceProgram callModule) sourceInitialState 2
      [BINOP I32 (mk_binop__0 Inn_I32 ADD)]).
  split.
  - unfold typedCall. split; [repeat constructor | ]. split; [repeat constructor | ].
    split; [repeat constructor | ]. split; [repeat constructor; eexists; reflexivity | ].
    vm_compute. repeat constructor.
  - split; [lia | ]. split; [repeat constructor; eexists; reflexivity | ].
    split; [repeat constructor | ]. split; exact I.
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
  - split; [exact source_initial_typed | ]. split; [apply source_initial_represents | reflexivity].
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

(* Refine two runs at the public interface. The prefix supplies both the preserved
   program ROM binding and the dynamic state needed to execute the final ADD. *)
Example source_program_two_runs : forall stale initialGet call middle endRow finalAdd final,
  fieldPrime ->
  validCircuitRun (sourceMemory stale)
    (callerRows [instructionSegment initialGet; callSegment call middle endRow; instructionSegment finalAdd]) final ->
  initialGet COL_IS_PROGRAM_ROW = 1 -> call COL_IS_PROGRAM_ROW = 1 ->
  endRow COL_SEL_END = 1 -> Forall CallScope.bodyRow middle -> finalAdd COL_IS_PROGRAM_ROW = 1 ->
  beforeBoundary initialGet = mkBoundary 2 4 4 7 ->
  exists z c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\
    programLoaded sourceProgram final /\
    SourceState sourceProgram (sourcePosition 1 3) z final (afterBoundary finalAdd) [] c.
Proof.
  intros stale initialGet call middle endRow finalAdd final Hprime Hrun Hget Hcall Hend Hmiddle Hadd Hboundary.
  destruct (valid_run_split
    (callerRows [instructionSegment initialGet; callSegment call middle endRow]) [finalAdd]
    (sourceMemory stale) final ltac:(
      change (validCircuitRun (sourceMemory stale)
        (flat_map segmentRows [instructionSegment initialGet; callSegment call middle endRow] ++
         flat_map segmentRows [instructionSegment finalAdd]) final);
      rewrite <- flat_map_app; exact Hrun)) as [between [HprefixRun HlastRun]].
  destruct (run_refinement sourceProgram (sourcePosition 1 0)
    [LOCAL_GET (mk_uN 0); CALL (mk_uN 0)] [BINOP I32 (mk_binop__0 Inn_I32 ADD)]
    [instructionSegment initialGet; callSegment call middle endRow]
    sourceInitialState (sourceMemory stale) between (mkBoundary 2 4 4 7) _
    Hprime (source_program_loaded stale) HprefixRun
    ltac:(constructor; [exact Hget | ]; constructor; [repeat split; assumption | constructor])
    Hboundary (source_initial_matches stale)) as [z1 [c1 [Hprefix [Hloaded Hsource]]]].
  assert (Hstart : callerStartsAt (afterBoundary endRow) [instructionSegment finalAdd]).
  { symmetry. apply segment_chain_boundary with
      (segment := callSegment call middle endRow) (next := instructionSegment finalAdd) (rest := []).
    eapply chain_suffix with (prefix := [initialGet]). exact (proj1 (proj2 Hrun)). }
  destruct (run_refinement sourceProgram (sourcePosition 1 2)
    [BINOP I32 (mk_binop__0 Inn_I32 ADD)] [] [instructionSegment finalAdd]
    z1 between final (afterBoundary endRow) c1 Hprime Hloaded HlastRun
    ltac:(constructor; [exact Hadd | constructor]) Hstart Hsource)
    as [z2 [c2 [Hlast Hfinal]]].
  exists z2, c2. split; [eapply spec_steps_transitive; eassumption | exact Hfinal].
Qed.
