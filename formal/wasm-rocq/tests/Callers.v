From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals StraightLineSpec StackSimulation CallSpec CallScope CallerSpec CallerRefinement SpecTecCalls tests.Calls.
Import ListNotations.
Open Scope Z_scope.

Definition readZeroCallee := mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 10.
Definition callerCallees : CalleeEnvironment := fun index =>
  if N.eqb index 0 then Some readZeroCallee else None.
Definition getCallAdd : list instr :=
  [LOCAL_GET (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].

(* Caller PCs 0..2 and callee PCs 10..11 occupy the same ROMs. *)
Definition callerMemory (stale : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory programOpcodeId then
    match address with
    | [0] | [10] => Some CODE_LOCAL_GET | [1] => Some CODE_CALL
    | [2] => Some CODE_I32_ADD | [11] => Some CODE_END | _ => None end
  else if Nat.eqb memory pcRomId then
    match address with
    | [0; choice] => if choice =? 0 then Some 1 else None
    | [1; 0] => Some 10 | [1; 1] => Some 2
    | [2; choice] => if choice =? 0 then Some 3 else None
    | [10; choice] => if choice =? 0 then Some 11 else None | _ => None end
  else if Nat.eqb memory pcEdgeKindId then
    match address with [0] | [1] | [2] | [10] => Some 0 | [11] => Some 1 | _ => None end
  else if Nat.eqb memory callTargetId then
    match address with [1] => Some 1 | _ => None end
  else if Nat.eqb memory pcFunctionRefId then
    match address with [0] | [1] | [2] => Some 2 | [10] | [11] => Some 1 | _ => None end
  else if Nat.eqb memory programLocalIndexId then
    match address with [0] | [10] => Some 0 | _ => None end
  else reusedFrameMemory stale memory address.

Example get_call_add_typed : callerTyped callerCallees callState 0 getCallAdd.
Proof.
  pose proof (proj1 (proj2 (dirty_callee_is_a_valid_call_input 99))) as Hcall.
  cbn [getCallAdd callerTyped callerCallees N.eqb instructionStackHeight
    readZeroCallee guestParams guestResults guestModule guestTypeIndex guestLocals guestBody].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ].
  split; [vm_compute; repeat constructor | ].
  split; [exact Hcall | ]. split; [vm_compute; lia | ].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; exact I.
Qed.

Example get_call_add_loaded : forall stale, loadedCaller callerCallees callState (callerMemory stale) 0 getCallAdd.
Proof.
  intros stale. cbn [getCallAdd loadedCaller callerCallees N.eqb]. split.
  - unfold instructionImage. repeat split; try reflexivity.
  - split.
    + cbn [readZeroCallee guestAddress guestModule guestTypeIndex guestParams guestResults guestLocals guestBody guestEntry].
      unfold loadedCall, directCallImage, guestFunctionImage. split.
      * repeat split; reflexivity.
      * split; [exists 2; split; reflexivity | ]. split.
        -- cbn [readZeroBody loadedBody]. split; [ | exact I].
           unfold instructionImage. repeat split; try reflexivity.
        -- split; reflexivity.
    + split; [unfold instructionImage; repeat split; try reflexivity | exact I].
Qed.

Example get_call_add_initial : forall stale,
  CallerState callerCallees callState (callerMemory stale) (mkBoundary 0 4 4 7) 0
    getCallAdd (stackConfig callState [] getCallAdd).
Proof.
  intros stale. constructor; [apply get_call_add_loaded | exact get_call_add_typed | | reflexivity].
  exists []. split.
  - split; [cbn; lia | ]. split; [reflexivity | ]. split; [constructor | ]. intros n v H; destruct n; discriminate.
  - split; [reflexivity | ]. split.
    + apply stack_config_wf; [repeat constructor | constructor | ].
      eapply caller_program_wf. exact get_call_add_typed.
    + intros [ | n] value Hlookup; cbn [callState stateLocals LOCALS nth_error] in Hlookup.
      * inversion Hlookup; subst. exists 5. split; [reflexivity | ]. split; [unfold i32Modulus; lia | reflexivity].
      * destruct n; discriminate.
Qed.

Example add_after_call_needs_two_values :
  ~ callerTyped callerCallees callState 0 [CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Proof. intros [_ [_ [_ [_ [Hheight _]]]]]. exact Hheight. Qed.

Example unsupported_caller_instruction_rejected :
  ~ callerTyped callerCallees callState 0 [NOP].
Proof. intros [Hsupported _]. destruct (Forall_inv Hsupported) as [code Hcode]. discriminate. Qed.

Example missing_callee_rejected :
  ~ callerTyped callerCallees callState 0 [CALL (mk_uN 1)].
Proof. exact (fun H => H). Qed.

Example calls_share_one_environment_entry :
  callerTyped callerCallees callState 0
    [CALL (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Proof.
  pose proof (proj1 (proj2 (dirty_callee_is_a_valid_call_input 99))) as Hcall.
  split; [exact Hcall | ]. split; [vm_compute; lia | ].
  split; [exact Hcall | ]. split; [cbn; lia | ].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; exact I.
Qed.

Example wrong_callee_address_rejected : forall stale,
  ~ loadedCaller (fun _ => Some (mkGuestFunction 1%N callModule (mk_uN 0) 0 1 1 readZeroBody 10))
      callState (callerMemory stale) 1 [CALL (mk_uN 0)].
Proof. intros stale [[ [Haddress _] _] _]. discriminate Haddress. Qed.

(* This instantiates the composed theorem with live code on both sides of CALL.
   It assumes actual circuit validity, not the result of the isolated call proof. *)
Example get_call_add_refines : forall stale initialGet call middle endRow finalAdd final,
  fieldPrime ->
  validCircuitRun (callerMemory stale)
    (callerRows [instructionSegment initialGet; callSegment call middle endRow; instructionSegment finalAdd]) final ->
  initialGet COL_IS_PROGRAM_ROW = 1 -> call COL_IS_PROGRAM_ROW = 1 ->
  endRow COL_SEL_END = 1 -> Forall bodyRow middle -> finalAdd COL_IS_PROGRAM_ROW = 1 ->
  beforeBoundary initialGet = mkBoundary 0 4 4 7 ->
  exists z c,
    Steps (stackConfig callState [] [LOCAL_GET (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)]) c /\
    CallerState callerCallees z final (afterBoundary finalAdd) 3 [] c.
Proof.
  intros stale initialGet call middle endRow finalAdd final Hprime Hrun Hget Hcall Hend Hmiddle Hadd Hboundary.
  destruct (caller_run_refinement getCallAdd []
    [instructionSegment initialGet; callSegment call middle endRow; instructionSegment finalAdd]
    callerCallees callState (callerMemory stale) final (mkBoundary 0 4 4 7) 0 _ Hprime Hrun
    ltac:(constructor; [exact Hget | ]; constructor; [repeat split; assumption | ];
      constructor; [exact Hadd | constructor]) Hboundary (get_call_add_initial stale))
    as [z [c [Hsteps [Hfinal _]]]]. exists z, c. auto.
Qed.

(* A prefix containing a call retains the pending ADD in the SpecTec configuration. *)
Example get_call_prefix_refines : forall stale initialGet call middle endRow final,
  fieldPrime ->
  validCircuitRun (callerMemory stale)
    (callerRows [instructionSegment initialGet; callSegment call middle endRow]) final ->
  initialGet COL_IS_PROGRAM_ROW = 1 -> call COL_IS_PROGRAM_ROW = 1 ->
  endRow COL_SEL_END = 1 -> Forall bodyRow middle ->
  beforeBoundary initialGet = mkBoundary 0 4 4 7 ->
  exists z c, Steps (stackConfig callState [] getCallAdd) c /\
    CallerState callerCallees z final (afterBoundary endRow) 2 [BINOP I32 (mk_binop__0 Inn_I32 ADD)] c.
Proof.
  intros stale initialGet call middle endRow final Hprime Hrun Hget Hcall Hend Hmiddle Hboundary.
  destruct (caller_run_refinement [LOCAL_GET (mk_uN 0); CALL (mk_uN 0)] [BINOP I32 (mk_binop__0 Inn_I32 ADD)]
    [instructionSegment initialGet; callSegment call middle endRow]
    callerCallees callState (callerMemory stale) final (mkBoundary 0 4 4 7) 0 _ Hprime Hrun
    ltac:(constructor; [exact Hget | ]; constructor; [repeat split; assumption | constructor])
    Hboundary (get_call_add_initial stale)) as [z [c [Hsteps [Hfinal _]]]].
  exists z, c. auto.
Qed.

(* Feed the prefix's complete invariant into a second application. In particular,
   no fresh loading or typing premise is supplied for the ADD after the call. *)
Example get_call_add_in_two_runs : forall stale initialGet call middle endRow finalAdd final,
  fieldPrime ->
  validCircuitRun (callerMemory stale)
    (callerRows [instructionSegment initialGet; callSegment call middle endRow; instructionSegment finalAdd]) final ->
  initialGet COL_IS_PROGRAM_ROW = 1 -> call COL_IS_PROGRAM_ROW = 1 ->
  endRow COL_SEL_END = 1 -> Forall bodyRow middle -> finalAdd COL_IS_PROGRAM_ROW = 1 ->
  beforeBoundary initialGet = mkBoundary 0 4 4 7 ->
  exists z c, Steps (stackConfig callState [] getCallAdd) c /\
    CallerState callerCallees z final (afterBoundary finalAdd) 3 [] c.
Proof.
  intros stale initialGet call middle endRow finalAdd final Hprime Hrun Hget Hcall Hend Hmiddle Hadd Hboundary.
  destruct (valid_run_split
    (callerRows [instructionSegment initialGet; callSegment call middle endRow]) [finalAdd]
    (callerMemory stale) final ltac:(
      change (validCircuitRun (callerMemory stale)
        (flat_map segmentRows [instructionSegment initialGet; callSegment call middle endRow] ++
         flat_map segmentRows [instructionSegment finalAdd]) final);
      rewrite <- flat_map_app; exact Hrun))
    as [between [HprefixRun HlastRun]].
  destruct (get_call_prefix_refines stale initialGet call middle endRow between
    Hprime HprefixRun Hget Hcall Hend Hmiddle Hboundary) as [z1 [c1 [Hprefix Hsource]]].
  assert (Hstart : callerStartsAt (afterBoundary endRow) [instructionSegment finalAdd]).
  { symmetry. apply segment_chain_boundary with
      (segment := callSegment call middle endRow) (next := instructionSegment finalAdd) (rest := []).
    eapply chain_suffix with (prefix := [initialGet]). exact (proj1 (proj2 Hrun)). }
  destruct (caller_run_refinement [BINOP I32 (mk_binop__0 Inn_I32 ADD)] [] [instructionSegment finalAdd]
    callerCallees z1 between final (afterBoundary endRow) 2 c1 Hprime HlastRun
    ltac:(constructor; [exact Hadd | constructor]) Hstart Hsource) as [z2 [c2 [Hlast [Hfinal _]]]].
  exists z2, c2. split; [eapply spec_steps_transitive; eassumption | exact Hfinal].
Qed.
