(* Bind initialization and return-context memory events to the exported catalog. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts
  MemoryTrace MemoryCatalog CallInitialization LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma full_initialization_local_writes : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  filter isLocalWrite (eventsForRow MemoryCatalog.memories w) =
    [WriteEvent MemoryCatalog.localLoId [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX]
      (w COL_LOCAL_VALUE) None].
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  destruct (full_initialization_auxiliary w chosen Hprime Hfull Hkind Hchosen)
    as [Hprogram [Hguest [Hpush [Hpop [Hturn Hquiet]]]]].
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hselectors.
  pose proof (quiet_gather_kinds w Hquiet) as Hkinds.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hwrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { auxiliary_row localWriteGateIndex Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  assert (Hgather : w COL_GATHER_LOCAL_WRITE_LO = 0).
  { full_row gatherLocalLowWriteRow Hrow. zero_gather_kinds Hkinds Hrow.
    pose proof (value_bound w COL_GATHER_LOCAL_WRITE_LO). field_linear Hrow. }
  assert (Hmodes : (w COL_PARAM_INIT_ACTIVE_BEFORE = 0 /\ w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1) \/
    (w COL_PARAM_INIT_ACTIVE_BEFORE = 1 /\ w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0)).
  { destruct Hkind as [-> | ->].
    - left. split; [ | exact Hchosen].
      exact (full_row_kind_exclusive w COL_LOCAL_ZERO_ACTIVE_BEFORE COL_PARAM_INIT_ACTIVE_BEFORE
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hchosen).
    - right. split; [exact Hchosen | ].
      exact (full_row_kind_exclusive w COL_PARAM_INIT_ACTIVE_BEFORE COL_LOCAL_ZERO_ACTIVE_BEFORE
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hchosen). }
  destruct local_catalog_split as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons.
  repeat rewrite filter_app.
  rewrite (filter_other_memories w _ Hbefore), (filter_other_memories w _ Hafter).
  cbn [app]. unfold localSpec, MemoryCatalog.localLoId.
  cbn [nth MemoryCatalog.memories ports memoryId flat_map filter].
  cbv [portEvent active].
  cbn [flat_map filter map Nat.eqb memoryId ports activation portKind addressColumns valueColumn].
  normalize_columns. rewrite Hwrite, Hgather.
  unfold isLocalWrite. repeat rewrite filter_app.
  destruct Hmodes as [[Hp Hz] | [Hp Hz]]; normalize_columns; rewrite Hp, Hz;
    let get := eval compute in COL_SEL_LOCAL_GET in
    let gather := eval compute in GMEM_LOCAL in
    destruct (w get =? 1); destruct (w gather =? 1);
      cbn [Z.eqb filter flat_map app Nat.eqb]; reflexivity.
Qed.

Theorem initialization_writes_local : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  forall address, final MemoryCatalog.localLoId address =
    update initial MemoryCatalog.localLoId [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX]
      (w COL_LOCAL_VALUE) MemoryCatalog.localLoId address.
Proof.
  intros initial final w chosen Hprime Hfull Hkind Hchosen Hreplay.
  eapply one_local_write_effect; [exact Hreplay | ].
  exact (full_initialization_local_writes w chosen Hprime Hfull Hkind Hchosen).
Qed.

Ltac exported_event memory portSpecification :=
  let rawSpec := eval vm_compute in
    (List.nth memory MemoryCatalog.memories (mkMemory 0%nat Ram [])) in
  eapply port_event_in_row with (specification := rawSpec) (port := portSpecification);
  [ apply nth_error_In with (n := memory); vm_compute; reflexivity
  | vm_compute; tauto
  | cbv [portEvent active]; cbn; normalize_columns;
    match goal with H : value _ ?c = 1 |- _ => rewrite H; reflexivity end ].

Ltac exported_read memory address value gate :=
  eapply replay_read_from_initial;
  [ eassumption
  | apply read_only_row; vm_compute; reflexivity
  | exported_event memory (mkPort address value Read (When gate)) ].

Lemma direct_call_target_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_SEL_CALL = 1 ->
  initial callTargetId [w COL_PC_BEFORE] = Some (w COL_FUNCTION_REF) /\
  initial functionCallMetadataId [w COL_FUNCTION_REF] = Some (w COL_CALL_TARGET_METADATA).
Proof.
  intros initial final w Hreplay Hcall. split.
  - exported_read callTargetId [COL_PC_BEFORE] COL_FUNCTION_REF COL_SEL_CALL.
  - exported_read functionCallMetadataId [COL_FUNCTION_REF] COL_CALL_TARGET_METADATA COL_SEL_CALL.
Qed.

Lemma guest_entry_reads : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_GUEST_ENTRY_ACTIVE = 1 ->
  initial functionLocalCountId [w COL_FUNCTION_REF] = Some (w COL_ENTERED_FUNCTION_NUM_LOCALS) /\
  initial functionEntryId [w COL_FUNCTION_REF] = Some (w COL_PC_AFTER).
Proof.
  intros initial final w Hreplay Hguest. split.
  - exported_read functionLocalCountId [COL_FUNCTION_REF] COL_ENTERED_FUNCTION_NUM_LOCALS COL_GUEST_ENTRY_ACTIVE.
  - exported_read functionEntryId [COL_FUNCTION_REF] COL_PC_AFTER COL_GUEST_ENTRY_ACTIVE.
Qed.

Lemma zero_row_local_count_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  initial functionLocalCountId [w COL_CURRENT_FUNCTION_REF] = Some (w COL_CURRENT_FUNCTION_NUM_LOCALS).
Proof.
  intros initial final w Hreplay Hzero.
  exported_read functionLocalCountId [COL_CURRENT_FUNCTION_REF] COL_CURRENT_FUNCTION_NUM_LOCALS COL_LOCAL_ZERO_ACTIVE_BEFORE.
Qed.

Lemma program_row_local_count_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_IS_PROGRAM_ROW = 1 ->
  initial functionLocalCountId [w COL_CURRENT_FUNCTION_REF] = Some (w COL_CURRENT_FUNCTION_NUM_LOCALS).
Proof.
  intros initial final w Hreplay Hprogram.
  exported_read functionLocalCountId [COL_CURRENT_FUNCTION_REF] COL_CURRENT_FUNCTION_NUM_LOCALS COL_IS_PROGRAM_ROW.
Qed.

Lemma current_function_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_PC_FREF_ACTIVE = 1 ->
  initial pcFunctionRefId [w COL_PC_BEFORE] = Some (w COL_CURRENT_FUNCTION_REF).
Proof.
  intros initial final w Hreplay Hactive.
  exported_read pcFunctionRefId [COL_PC_BEFORE] COL_CURRENT_FUNCTION_REF COL_PC_FREF_ACTIVE.
Qed.

Lemma call_return_pc_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_CALL_STACK_PUSH_PRESENT = 1 ->
  initial pcRomId [w COL_PC_BEFORE; w COL_PC_ROM_CALL_RETURN_CHOICE] = Some (w COL_CALL_STACK_RETURN_PC_VALUE).
Proof.
  intros initial final w Hreplay Hpush.
  exported_read pcRomId [COL_PC_BEFORE; COL_PC_ROM_CALL_RETURN_CHOICE]
    COL_CALL_STACK_RETURN_PC_VALUE COL_CALL_STACK_PUSH_PRESENT.
Qed.

Lemma full_initialization_stack : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  w COL_STACK_WRITE0_ACTIVE = 0 /\ w COL_STACK_WRITE0_HI_ACTIVE = 0 /\
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = w COL_PARAM_INIT_ACTIVE_BEFORE /\
  w COL_SP_AFTER = w COL_SP_BEFORE - w COL_PARAM_INIT_ACTIVE_BEFORE.
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  destruct (full_initialization_auxiliary w chosen Hprime Hfull Hkind Hchosen)
    as [Hprogram [Hguest [Hpush [Hpop [Hturn Hquiet]]]]].
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hselectors.
  destruct Hquiet as [Hkinds Hhi Hlocal Houtput Hdiscard].
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hgather : w COL_GATHER_ACTIVE = 0).
  { exact (full_row_kind_exclusive w chosen COL_GATHER_ACTIVE Hprime Hfull
      (initialization_kind_in chosen Hkind) ltac:(cbn; tauto)
      ltac:(destruct Hkind as [-> | ->]; discriminate) Hchosen). }
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { auxiliary_row outputCaptureSupportRow Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { auxiliary_row hostCallSupportRow Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. normalize_columns. rewrite Hhost in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = w COL_PARAM_INIT_ACTIVE_BEFORE).
  { auxiliary_row stackReadsArityRow Hrow.
    normalize_columns. rewrite Hone, Hgather in Hrow.
    pose proof (value_bound w COL_STACK_READS). pose proof (value_bound w COL_PARAM_INIT_ACTIVE_BEFORE).
    field_linear Hrow. }
  assert (Hread1 : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { auxiliary_row stackRead1SupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { auxiliary_row stackRead2SupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread0 : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = w COL_PARAM_INIT_ACTIVE_BEFORE).
  { full_row stackReadsActiveRow Hrow. normalize_columns. rewrite Hone, Hread1, Hread2, Hreads in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)).
    pose proof (value_bound w COL_PARAM_INIT_ACTIVE_BEFORE). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { auxiliary_row stackWritesArityRow Hrow.
    normalize_columns. rewrite Hone, Hgather in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. normalize_columns. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. normalize_columns. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  split; [exact Hwrite | ]. split; [exact HwriteHi | ]. split; [exact Hread0 | ].
  full_row stackPointerBalanceRow Hrow. zero_gather_kinds Hkinds Hrow.
  normalize_columns. rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
  circuit_range COL_SP_BEFORE Hspb. circuit_range COL_SP_AFTER Hspa.
  circuit_boolean COL_PARAM_INIT_ACTIVE_BEFORE Hparam.
  destruct Hparam; field_linear Hrow.
Qed.

Lemma initialization_parameter_address : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_PARAM_INIT_ACTIVE_BEFORE = 1 ->
  w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1).
Proof.
  intros w Hprime Hfull Hparam.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_SP_BEFORE Hsp.
  circuit_range (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) Haddr.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, 1); (COL_SP_BEFORE, -2); (COL_ONE, 2)],
    ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hparam, Hone in Hrow. field_linear Hrow.
Qed.

Lemma stack_writes_inactive : forall (w : Witness) address,
  w COL_STACK_WRITE0_ACTIVE = 0 -> w COL_STACK_WRITE0_HI_ACTIVE = 0 ->
  Forall (NoWriteTo stackId address) (eventsForRow MemoryCatalog.memories w).
Proof.
  intros w address Hlo Hhi.
  change (Forall (NoWriteTo stackId address)
    (flat_map (fun port => match portEvent stackId w port with
       | Some event => [event] | None => [] end)
       (ports (List.hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
     eventsForRow (tl MemoryCatalog.memories) w)).
  apply Forall_app. split.
  - cbn [eventsForRow MemoryCatalog.memories List.hd flat_map memoryId ports].
    cbv [portEvent active].
    cbn [activation portKind addressColumns valueColumn map].
    normalize_columns. rewrite Hlo, Hhi. cbn [Z.eqb flat_map app].
    repeat rewrite Forall_app. repeat split;
      repeat match goal with |- context [if ?b then _ else _] => destruct b end;
      cbn; repeat constructor.
  - apply read_only_row. vm_compute. reflexivity.
Qed.

Lemma initialization_preserves_stack_memory : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  forall address, final stackId address = initial stackId address.
Proof.
  intros initial final w chosen Hprime Hfull Hkind Hchosen Hreplay address.
  destruct (full_initialization_stack w chosen Hprime Hfull Hkind Hchosen) as [Hlo [Hhi _]].
  eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption].
Qed.

Lemma parameter_reads_initial_stack : forall initial final (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_PARAM_INIT_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial stackId [2 * (w COL_SP_BEFORE - 1)] = Some (w COL_LOCAL_VALUE).
Proof.
  intros initial final w Hprime Hfull Hparam Hreplay.
  destruct (full_initialization_stack w COL_PARAM_INIT_ACTIVE_BEFORE Hprime Hfull
    ltac:(right; reflexivity) Hparam) as [Hlo [Hhi [HreadActive _]]].
  rewrite Hparam in HreadActive.
  pose proof (full_parameter_initialization_row w Hprime Hfull Hparam) as [_ [_ [Hvalue _]]].
  rewrite Hvalue, <- (initialization_parameter_address w Hprime Hfull Hparam).
  eapply replay_read_from_initial; [exact Hreplay | apply stack_writes_inactive; assumption | ].
  exported_event stackId (mkPort [List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat]
    (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) Read (When (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat))).
Qed.

Lemma zero_row_uses_loaded_local_count : forall initial final (w : Witness) entry reference total,
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PC_BEFORE = entry ->
  initial pcFunctionRefId [entry] = Some reference ->
  initial functionLocalCountId [reference] = Some total ->
  w COL_CURRENT_FUNCTION_NUM_LOCALS = total.
Proof.
  intros initial final w entry reference total Hprime Hfull Hzero Hreplay Hpc Href Htotal.
  assert (Hactive : w COL_PC_FREF_ACTIVE = 1).
  { exact (full_frame_lookup_active w COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) Hzero). }
  pose proof (current_function_read initial final w Hreplay Hactive) as Hfunction.
  pose proof (zero_row_local_count_read initial final w Hreplay Hzero) as Hcount.
  rewrite Hpc, Href in Hfunction. injection Hfunction as Hreference.
  rewrite <- Hreference, Htotal in Hcount. congruence.
Qed.
