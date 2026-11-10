(* Direct-call ROM bindings bound the callee entry and seed its frame initialization. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  TraceRefinement ProgramBinding StraightLineSpec StackSimulation CallInitialization
  CallBindings CallMemory IntegerMemory FrameInitialization CallSpec SpecTecCalls WasmValues Wasm2 WasmIntegers LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma direct_call_entry_bound : forall initial final w pc returnPC caller index address module typeIndex
    (params results locals : list valtype) body entry,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  w COL_PC_BEFORE = pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  directCallImage initial pc returnPC caller index address module typeIndex
    params results locals body entry ->
  (length params < 256)%nat -> (length results < 256)%nat ->
  w COL_SEL_CALL = 1 /\ w COL_TARGET_FUNCTION_IS_GUEST = 1 /\
  w COL_CALL_PARAM_COUNT = Z.of_nat (length params) /\
  w COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat (length locals) /\
  w COL_ENTERED_FUNCTION_NUM_LOCALS = Z.of_nat (length params) + Z.of_nat (length locals) /\
  w COL_PC_AFTER = entry.
Proof.
  intros initial final w pc returnPC caller index address module typeIndex params results locals body entry
    Hprime Hfull Hprogram Hpc Hreplay [_ [_ [Hop [Htarget [_ [Hmeta [Htotal [Hentry Href]]]]]]]] Hp Hr.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hcode.
  assert (Hcall : w COL_SEL_CALL = 1).
  { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_CALL); try eassumption; [reflexivity | congruence]. }
  destruct (direct_call_target_read _ _ _ Hreplay Hcall) as [HtargetRead HmetaRead].
  rewrite Hpc, Htarget in HtargetRead. injection HtargetRead as Hreference.
  rewrite <- Hreference, Hmeta in HmetaRead. injection HmetaRead as Hmetadata.
  destruct (guest_metadata_decodes w (Z.of_nat (length params)) (Z.of_nat (length results))
    Hprime Hfull ltac:(lia) ltac:(lia) (eq_sym Hmetadata)) as [Hparams [Hresults HguestTarget]].
  destruct (full_direct_guest_call w Hprime Hfull Hcall HguestTarget) as [Hguest _].
  destruct (guest_entry_reads _ _ _ Hreplay Hguest) as [HtotalRead HentryRead].
  rewrite <- Hreference, Htotal in HtotalRead.
  rewrite <- Hreference, Hentry in HentryRead.
  pose proof (full_guest_initialization_seed w Hprime Hfull Hprogram Hguest) as Hz.
  injection HtotalRead as HtotalValue. injection HentryRead as HentryValue.
  rewrite Nat2Z.inj_add in HtotalValue.
  repeat split; try assumption; congruence || lia.
Qed.

Lemma call_starts_integer_frame_initialization : forall initial final w prefix arguments locals,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_CALL_PARAM_COUNT = Z.of_nat (length arguments) ->
  w COL_ENTERED_FUNCTION_NUM_LOCALS = Z.of_nat (length arguments) + Z.of_nat locals ->
  w COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat locals ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  0 <= w COL_STACK_FRAME_BASE_BEFORE ->
  w COL_SP_BEFORE = w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (prefix ++ arguments)) ->
  (forall index value, nth_error (prefix ++ arguments) index = Some value ->
    CellMatches initial (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) value) ->
  InitializingFrame final (afterBoundary w) arguments (Z.of_nat locals) (w COL_LOCAL_ZERO_REMAINING_AFTER).
Proof.
  intros initial final w prefix arguments locals Hprime Hfull Hcall Htarget Hp Hentered Hz Hreplay HbaseBefore Hheight Hcells.
  assert (Hin : In COL_SEL_CALL selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_direct_guest_call w Hprime Hfull Hcall Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_call_frame_entry w Hprime Hfull Hguest Hpush Hpop
    (full_selected_is_program_row w COL_SEL_CALL Hprime Hfull Hin Hcall)) as [Hfbp [Hbase _]].
  destruct (full_direct_guest_call_stack w Hprime Hfull Hcall Htarget) as [Hlo [Hhi Hsp]].
  destruct (full_opcode_aux_gates_zero w COL_SEL_CALL Hprime Hfull Hin Hcall) as [_ [_ Hgather]].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { selected_row COL_SEL_CALL localWriteGateIndex Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  rewrite length_app, Nat2Z.inj_add in Hheight.
  rewrite Hz. apply initializing_frame_start; cbn [afterBoundary boundarySP boundaryStackBase boundaryLocalsBase].
  - lia.
  - lia.
  - lia.
  - lia.
  - intros index value Hlookup.
    specialize (Hcells (length prefix + index)%nat value).
    rewrite nth_error_app2 in Hcells by lia.
    replace (length prefix + index - length prefix)%nat with index in Hcells by lia.
    specialize (Hcells Hlookup).
    rewrite Nat2Z.inj_add in Hcells.
    replace (w COL_LOCALS_FBP_AFTER + Z.of_nat index) with
      (w COL_STACK_FRAME_BASE_BEFORE + (Z.of_nat (length prefix) + Z.of_nat index)) by lia.
    eapply cell_matches_memory; [ | exact Hcells]. intros limb _.
    exact (replay_preserves _ _ _ _ _ Hreplay
      (stack_writes_inactive w _ Hlo Hhi HlocalWrite HgatherLocalLo HgatherLocal)).
Qed.

Lemma call_starts_frame_initialization : forall initial final w prefix arguments locals,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_CALL_PARAM_COUNT = Z.of_nat (length arguments) ->
  w COL_ENTERED_FUNCTION_NUM_LOCALS = Z.of_nat (length arguments) + Z.of_nat locals ->
  w COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat locals ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ arguments) ->
  InitializingFrame final (afterBoundary w) (map i32Value arguments) (Z.of_nat locals) (w COL_LOCAL_ZERO_REMAINING_AFTER).
Proof.
  intros initial final w prefix arguments locals Hprime Hfull Hcall Htarget Hp Hentered Hz Hreplay
    [Hbase [Hsp [Hbounds Hcells]]].
  eapply call_starts_integer_frame_initialization with (prefix := map i32Value prefix);
    rewrite ?length_map; try eassumption.
  - rewrite <- map_app, length_map. exact Hsp.
  - intros index value Hlookup. rewrite <- map_app, nth_error_map in Hlookup.
    destruct (nth_error (prefix ++ arguments) index) as [v | ] eqn:Hv; [ | discriminate].
    injection Hlookup as <-. constructor.
    + exact (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hv)).
    + exact (Hcells _ _ Hv).
Qed.
