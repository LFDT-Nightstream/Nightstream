(* Direct-call ROM bindings seed initialization; validity forces its complete prefix. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  TraceRefinement ProgramBinding StraightLineSpec StackSimulation CallInitialization
  CallBindings CallMemory FrameInitialization CallSpec SpecTecCalls Wasm2 WasmIntegers.
Import ListNotations.
Open Scope Z_scope.

Lemma direct_call_entry_bound : forall initial final w pc returnPC caller index address module typeIndex
    (arguments : list Z) results locals body entry,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  w COL_PC_BEFORE = pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  directCallImage initial pc returnPC caller index address module typeIndex
    (length arguments) results locals body entry ->
  (length arguments < 256)%nat -> (results < 256)%nat ->
  w COL_SEL_CALL = 1 /\ w COL_TARGET_FUNCTION_IS_GUEST = 1 /\
  w COL_CALL_PARAM_COUNT = Z.of_nat (length arguments) /\
  w COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat locals /\
  w COL_PARAM_INIT_REMAINING_AFTER = Z.of_nat (length arguments) /\
  w COL_PC_AFTER = entry.
Proof.
  intros initial final w pc returnPC caller index address module typeIndex arguments results locals body entry
    Hprime Hfull Hprogram Hpc Hreplay [_ [_ [Hop [Htarget [_ [Hmeta [Htotal [Hentry Href]]]]]]]] Hp Hr.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hcode.
  assert (Hcall : w COL_SEL_CALL = 1).
  { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_CALL); try eassumption; [reflexivity | congruence]. }
  destruct (direct_call_target_read _ _ _ Hreplay Hcall) as [HtargetRead HmetaRead].
  rewrite Hpc, Htarget in HtargetRead. injection HtargetRead as Hreference.
  rewrite <- Hreference, Hmeta in HmetaRead. injection HmetaRead as Hmetadata.
  destruct (guest_metadata_decodes w (Z.of_nat (length arguments)) (Z.of_nat results)
    Hprime Hfull ltac:(lia) ltac:(lia) (eq_sym Hmetadata)) as [Hparams [Hresults HguestTarget]].
  destruct (full_direct_guest_call w Hprime Hfull Hcall HguestTarget) as [Hguest _].
  destruct (guest_entry_reads _ _ _ Hreplay Hguest) as [HtotalRead HentryRead].
  rewrite <- Hreference, Htotal in HtotalRead.
  rewrite <- Hreference, Hentry in HentryRead.
  destruct (full_guest_initialization_seed w Hprime Hfull Hprogram Hguest) as [Hz HpCount].
  injection HtotalRead as HtotalValue. injection HentryRead as HentryValue.
  rewrite Nat2Z.inj_add in HtotalValue.
  repeat split; try assumption; congruence || lia.
Qed.

Lemma call_starts_frame_initialization : forall initial final w prefix arguments locals,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_CALL_PARAM_COUNT = Z.of_nat (length arguments) ->
  w COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat locals ->
  w COL_PARAM_INIT_REMAINING_AFTER = Z.of_nat (length arguments) ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ arguments) ->
  InitializingFrame final (afterBoundary w) arguments (Z.of_nat locals)
    (w COL_LOCAL_ZERO_REMAINING_AFTER) (w COL_PARAM_INIT_REMAINING_AFTER).
Proof.
  intros initial final w prefix arguments locals Hprime Hfull Hcall Htarget Hp Hz Hremaining Hreplay Hstack.
  destruct (full_direct_guest_call w Hprime Hfull Hcall Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_call_frame_entry w Hprime Hfull Hguest Hpush Hpop) as [_ [Hbase _]].
  destruct (full_direct_guest_call_stack w Hprime Hfull Hcall Htarget) as [Hlo [Hhi Hsp]].
  destruct Hstack as [HbaseBefore [Hheight [Hbounds Hcells]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hheight, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hheight.
  rewrite Hz, Hremaining. apply initializing_frame_start; [lia | | ].
  - cbn. lia.
  - intros index value Hlookup.
    rewrite (replay_preserves _ _ _ _ _ Hreplay (stack_writes_inactive w _ Hlo Hhi)).
    specialize (Hcells (length prefix + index)%nat value).
    rewrite nth_error_app2 in Hcells by lia.
    replace (length prefix + index - length prefix)%nat with index in Hcells by lia.
    specialize (Hcells Hlookup).
    change (initial stackId [2 * (w COL_STACK_FRAME_BASE_AFTER + Z.of_nat index)] = Some value).
    rewrite Nat2Z.inj_add in Hcells.
    replace (2 * (w COL_STACK_FRAME_BASE_AFTER + Z.of_nat index)) with
      (2 * (w COL_STACK_FRAME_BASE_BEFORE + (Z.of_nat (length prefix) + Z.of_nat index))) by lia.
    exact Hcells.
Qed.

Theorem direct_call_initialization_refinement : forall initial final callRow middle resume
    pc returnPC caller index address module typeIndex prefix arguments results locals body entry,
  fieldPrime ->
  validCircuitRun initial (callRow :: middle ++ [resume]) final ->
  callRow COL_IS_PROGRAM_ROW = 1 -> resume COL_IS_PROGRAM_ROW = 1 ->
  callRow COL_PC_BEFORE = pc ->
  directCallImage initial pc returnPC caller index address module typeIndex
    (length arguments) results locals body entry ->
  (length arguments < 256)%nat -> (results < 256)%nat ->
  stackMatches initial (beforeBoundary callRow) (prefix ++ arguments) ->
  exists initialization suffix ready,
    middle = initialization ++ suffix /\
    length initialization = (length arguments + locals)%nat /\
    Forall (fun w : Witness => w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/ w COL_PARAM_INIT_ACTIVE_BEFORE = 1) initialization /\
    validCircuitRun initial (callRow :: initialization) ready /\
    validCircuitRun ready (suffix ++ [resume]) final /\
    stackMatches ready (afterBoundary (finalRow callRow initialization)) [] /\
    localsMatch ready (afterBoundary (finalRow callRow initialization))
      (match caller with mk_state s _ => mk_state s (enteredFrame module arguments locals) end) /\
    (finalRow callRow initialization) COL_LOCAL_ZERO_REMAINING_AFTER = 0 /\
    (finalRow callRow initialization) COL_PARAM_INIT_REMAINING_AFTER = 0.
Proof.
  intros initial final callRow middle resume pc returnPC caller index address module typeIndex
    prefix arguments results locals body entry Hprime Hrun Hprogram Hresume Hpc Himage Hp Hr Hstack.
  destruct (valid_run_head _ _ _ _ Hrun) as [afterCall [Hcall [Hreplay Htail]]].
  destruct (direct_call_entry_bound _ _ _ _ _ _ _ _ _ _ arguments _ _ _ _
    Hprime Hcall Hprogram Hpc Hreplay Himage Hp Hr) as [Hselected [Htarget [Hparams [Hz [Hparam Hentry]]]]].
  pose proof (call_starts_frame_initialization _ _ _ prefix arguments locals Hprime Hcall
    Hselected Htarget Hparams Hz Hparam Hreplay Hstack) as Hstart.
  destruct Hrun as [Hsat [Hchain Htrace]].
  destruct (initialization_prefix_forced middle callRow resume Hprime Hsat Hchain Hresume)
    as [initialization [suffix [Hsplit [Hkinds [Hlength Hdone]]]]].
  assert (HchainInit : Chain continuityLinks (callRow :: initialization)).
  { rewrite Hsplit, <- app_assoc in Hchain.
    eapply (chain_prefix _ (callRow :: initialization) (suffix ++ [resume])). exact Hchain. }
  rewrite Hsplit, <- app_assoc in Htail.
  destruct (valid_run_split initialization (suffix ++ [resume]) afterCall final Htail)
    as [ready [Hinit Hsuffix]].
  destruct Himage as [_ [_ [_ [_ [_ [_ [Htotal [_ Hreference]]]]]]]].
  assert (HromRef : afterCall pcFunctionRefId [entry] = Some (Z.of_N address + 1)).
  { rewrite (replay_preserves _ _ _ _ _ Hreplay); [exact Hreference | ].
    apply read_only_row. vm_compute. reflexivity. }
  assert (HromCount : afterCall functionLocalCountId [Z.of_N address + 1] =
      Some (Z.of_nat (length arguments) + Z.of_nat locals)).
  { rewrite (replay_preserves _ _ _ _ _ Hreplay); [rewrite <- Nat2Z.inj_add; exact Htotal | ].
    apply read_only_row. vm_compute. reflexivity. }
  pose proof (initialization_rows_preserve_invariant initialization callRow afterCall ready arguments
    (Z.of_nat locals) entry (Z.of_N address + 1) Hprime Hcall Hinit HchainInit Hkinds
    Hentry HromRef HromCount Hstart) as Hready.
  assert (Hcounts : (finalRow callRow initialization) COL_LOCAL_ZERO_REMAINING_AFTER = 0 /\
      (finalRow callRow initialization) COL_PARAM_INIT_REMAINING_AFTER = 0).
  { unfold initializationRemaining in Hdone.
    pose proof (value_bound (finalRow callRow initialization) COL_LOCAL_ZERO_REMAINING_AFTER).
    pose proof (value_bound (finalRow callRow initialization) COL_PARAM_INIT_REMAINING_AFTER). lia. }
  destruct Hcounts as [HzDone HpDone]. rewrite HzDone, HpDone in Hready.
  destruct caller as [s frame].
  assert (Hbase : 0 <= boundaryStackBase (afterBoundary (finalRow callRow initialization))).
  { cbn. exact (proj1 (value_bound _ _)). }
  assert (Hargs : Forall (fun value => 0 <= value < i32Modulus) arguments).
  { destruct Hstack as [_ [_ [Hbounds _]]]. apply Forall_app in Hbounds. tauto. }
  destruct (initialized_frame_matches ready _ arguments locals s module Hbase Hargs Hready)
    as [Hempty Hlocals].
  exists initialization, suffix, ready. split; [exact Hsplit | ]. split.
  - unfold initializationRemaining in Hlength. rewrite Hz, Hparam in Hlength. lia.
  - split; [exact Hkinds | ]. split.
    + split; [constructor; [exact Hcall | exact (proj1 Hinit)] | ]. split; [exact HchainInit | ].
      cbn [eventsForTrace flat_map]. eapply replay_app; [exact Hreplay | exact (proj2 (proj2 Hinit))].
    + split; [exact Hsuffix | ]. split; [exact Hempty | ]. split; [exact Hlocals | ]. split; assumption.
Qed.
