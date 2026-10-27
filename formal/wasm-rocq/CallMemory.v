(* Bind frame-initialization, frame-move and return-context memory events to the
   exported catalog. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts
  MemoryTrace MemoryCatalog CallInitialization LocalBindings.
Import ListNotations.
Open Scope Z_scope.

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

Lemma call_return_pc_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_CALL_STACK_PUSH_PRESENT = 1 ->
  initial pcRomId [w COL_PC_BEFORE; w COL_PC_ROM_CALL_RETURN_CHOICE] = Some (w COL_CALL_STACK_RETURN_PC_VALUE).
Proof.
  intros initial final w Hreplay Hpush.
  exported_read pcRomId [COL_PC_BEFORE; COL_PC_ROM_CALL_RETURN_CHOICE]
    COL_CALL_STACK_RETURN_PC_VALUE COL_CALL_STACK_PUSH_PRESENT.
Qed.

(* A zero row pushes one cell; a move row reads its source and writes its
   destination. Neither touches a local port or another stack port. *)
Record InitializationStackShape (w : Witness) (move : bool) : Prop := {
  init_read0_active : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = (if move then 1 else 0);
  init_read1_inactive : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  init_read2_inactive : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  init_write_active : w COL_STACK_WRITE0_ACTIVE = 1;
  init_write_hi_active : w COL_STACK_WRITE0_HI_ACTIVE = 1;
  init_capture_inactive : w COL_OUTPUT_CAPTURED = 0;
  init_local_get_inactive : w COL_SEL_LOCAL_GET = 0;
  init_local_write_inactive : w COL_LOCAL_WRITE_ENABLED = 0;
  init_gmem_local_inactive : w GMEM_LOCAL = 0;
  init_gather_local_lo_inactive : w COL_GATHER_LOCAL_WRITE_LO = 0;
  init_gather_local_inactive : w COL_GATHER_LOCAL_WRITE = 0;
  init_write_hi_address : w COL_STACK_WRITE0_ADDR_HI = w COL_STACK_WRITE0_ADDR_LO + 1;
  init_read_hi_address :
    w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat) = w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) + 1;
  init_stack_pointer : w COL_SP_AFTER = w COL_SP_BEFORE + (if move then 0 else 1)
}.

Lemma full_initialization_stack : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  InitializationStackShape w (Nat.eqb chosen COL_FRAME_MOVE_ACTIVE_BEFORE).
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  destruct (full_initialization_auxiliary w chosen Hprime Hfull Hkind Hchosen)
    as [Hprogram [Hguest [Hpush [Hpop [Hturn Hquiet]]]]].
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hselectors.
  destruct Hquiet as [Hkinds Hhi Hlocal Houtput].
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hmodes : (w COL_FRAME_MOVE_ACTIVE_BEFORE = 0 /\ w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 /\
      Nat.eqb chosen COL_FRAME_MOVE_ACTIVE_BEFORE = false) \/
    (w COL_FRAME_MOVE_ACTIVE_BEFORE = 1 /\ w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0 /\
      Nat.eqb chosen COL_FRAME_MOVE_ACTIVE_BEFORE = true)).
  { destruct Hkind as [-> | ->].
    - left. split; [ | split; [exact Hchosen | reflexivity]].
      exact (full_row_kind_exclusive w COL_LOCAL_ZERO_ACTIVE_BEFORE COL_FRAME_MOVE_ACTIVE_BEFORE
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hchosen).
    - right. split; [exact Hchosen | split; [ | reflexivity]].
      exact (full_row_kind_exclusive w COL_FRAME_MOVE_ACTIVE_BEFORE COL_LOCAL_ZERO_ACTIVE_BEFORE
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hchosen). }
  assert (Hgather : w COL_GATHER_ACTIVE = 0).
  { exact (full_row_kind_exclusive w chosen COL_GATHER_ACTIVE Hprime Hfull
      (initialization_kind_in chosen Hkind) ltac:(cbn; tauto)
      ltac:(destruct Hkind as [-> | ->]; discriminate) Hchosen). }
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { auxiliary_row outputCaptureSupportRow Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { auxiliary_row hostCallSupportRow Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. pose proof (full_no_host_split w Hfull Hhost) as Hsplit. normalize_columns.
    rewrite Hhost, Hsplit in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0) by (apply Hselectors; vm_compute; tauto).
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { auxiliary_row localWriteGateIndex Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  assert (Hlatch : w COL_HOST_EVENT_EXIT_LATCH = 0).
  { circuit_row ([(COL_HOST_EVENT_EXIT_LATCH, 1)], [(COL_ONE, 1); (COL_SEL_END, -1); (COL_SEL_RETURN, -1)],
      ([] : SparseRow)) Hrow.
    pose proof (Hselectors COL_SEL_END ltac:(vm_compute; tauto)) as Hend.
    pose proof (Hselectors COL_SEL_RETURN ltac:(vm_compute; tauto)) as Hret.
    expand_row Hrow. normalize_columns. rewrite Hone, Hend, Hret in Hrow.
    pose proof (value_bound w COL_HOST_EVENT_EXIT_LATCH). field_linear Hrow. }
  pose proof (full_frame_sp_drop_idle w Hprime Hfull (full_aux_no_slide w Hprime Hfull Hprogram Hpop)
    ltac:(congruence) Hturn Hlatch) as Hdiscard.
  assert (Hreads : w COL_STACK_READS = w COL_FRAME_MOVE_ACTIVE_BEFORE).
  { auxiliary_row stackReadsArityRow Hrow.
    normalize_columns. rewrite Hone, Hgather in Hrow.
    pose proof (value_bound w COL_STACK_READS). pose proof (value_bound w COL_FRAME_MOVE_ACTIVE_BEFORE).
    field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 1).
  { auxiliary_row stackWritesArityRow Hrow.
    normalize_columns. rewrite Hone, Hgather in Hrow.
    pose proof (value_bound w COL_STACK_WRITES).
    destruct Hmodes as [[Hm [Hz _]] | [Hm [Hz _]]]; normalize_columns; rewrite Hm, Hz in Hrow; field_linear Hrow. }
  assert (Hread1 : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { auxiliary_row stackRead1SupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { auxiliary_row stackRead2SupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread0 : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = w COL_FRAME_MOVE_ACTIVE_BEFORE).
  { full_row stackReadsActiveRow Hrow. normalize_columns. rewrite Hone, Hread1, Hread2, Hreads in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)).
    pose proof (value_bound w COL_FRAME_MOVE_ACTIVE_BEFORE). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 1).
  { full_row stackWritesActiveRow Hrow. normalize_columns. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 1).
  { full_row stackWriteHighActiveRow Hrow. normalize_columns. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  destruct (full_host_switch_absent w chosen Hprime Hfull (initialization_kind_in chosen Hkind)
    ltac:(destruct Hkind as [-> | ->]; discriminate) ltac:(destruct Hkind as [-> | ->]; discriminate)
    ltac:(destruct Hkind as [-> | ->]; discriminate) Hchosen) as [_ [Hresume _]].
  pose proof (full_no_resume_pop w Hfull Hresume) as HresumePop.
  constructor; try assumption.
  - destruct Hmodes as [[Hm [_ ->]] | [Hm [_ ->]]]; congruence.
  - circuit_row ([(COL_STACK_WRITE0_ADDR_HI, 1); (COL_STACK_WRITE0_ADDR_LO, -1); (COL_ONE, -1)], [(COL_ONE, 1)],
      ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_HI). circuit_range COL_STACK_WRITE0_ADDR_LO Hlo.
    field_linear Hrow.
  - circuit_row ([(List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat, 1); (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, -1);
      (COL_ONE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)).
    circuit_range (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) Hlo.
    field_linear Hrow.
  - full_row stackPointerBalanceRow Hrow. normalize_columns. rewrite HresumePop in Hrow. zero_gather_kinds Hkinds Hrow.
    normalize_columns. rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
    circuit_range COL_SP_BEFORE Hspb. circuit_range COL_SP_AFTER Hspa.
    destruct Hmodes as [[Hm [_ ->]] | [Hm [_ ->]]]; normalize_columns; rewrite Hm in Hrow; field_linear Hrow.
Qed.

Definition initializationStackEvents (w : Witness) (move : bool) : list Event :=
  (if move then
    [ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
       (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
     ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
       (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat))] else []) ++
  [WriteEvent stackId [w COL_STACK_WRITE0_ADDR_LO] (w COL_STACK_WRITE0_VALUE_LO) None;
   WriteEvent stackId [w COL_STACK_WRITE0_ADDR_HI] (w COL_STACK_WRITE0_VALUE_HI) None].

Lemma initialization_stack_event_prefix : forall w move,
  InitializationStackShape w move ->
  eventsForRow MemoryCatalog.memories w =
    initializationStackEvents w move ++ eventsForRow (tl MemoryCatalog.memories) w.
Proof.
  intros w move Hshape.
  destruct Hshape as [Hr0 Hr1 Hr2 Hw Hwh Hcap Hget Hset Hgmem Hglo Hg _ _ _].
  change (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with
      Some event => [event] | None => [] end)
    (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
    eventsForRow (tl MemoryCatalog.memories) w =
    initializationStackEvents w move ++ eventsForRow (tl MemoryCatalog.memories) w).
  cbn [hd MemoryCatalog.memories ports flat_map portEvent active activation
    addressColumns valueColumn portKind map option_map].
  unfold portEvent, active. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hr0, Hr1, Hr2, Hw, Hwh, Hcap, Hget, Hset, Hgmem, Hglo, Hg.
  destruct move; reflexivity.
Qed.

(* A zero row's effect: the cell at sp, both limbs, becomes zero. *)
Theorem zero_row_stack_replay : forall initial final w,
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  forall address, final stackId address =
    update (update initial stackId [2 * w COL_SP_BEFORE] 0) stackId [2 * w COL_SP_BEFORE + 1] 0 stackId address.
Proof.
  intros initial final w Hprime Hfull Hzero Hreplay address.
  pose proof (full_initialization_stack w COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull ltac:(left; reflexivity) Hzero)
    as Hshape.
  destruct (full_zero_initialization_row w Hprime Hfull Hzero) as [_ [Haddr [Hlo Hhi]]].
  pose proof (init_write_hi_address w _ Hshape) as HaddrHi.
  rewrite (initialization_stack_event_prefix w _ Hshape) in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [Hstack Hrest]].
  cbn [initializationStackEvents Nat.eqb app COL_LOCAL_ZERO_ACTIVE_BEFORE COL_FRAME_MOVE_ACTIVE_BEFORE] in Hstack.
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.
  erewrite replay_preserves; [ | exact Hrest | apply read_only_row; vm_compute; reflexivity].
  rewrite HaddrHi, Haddr, Hlo, Hhi. reflexivity.
Qed.

(* A move row's effect: it reads its source cell and writes the value to its
   destination. *)
Theorem move_row_stack_replay : forall initial final w,
  fieldPrime -> FullRelation.satFull w -> w COL_FRAME_MOVE_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  forall address, final stackId address =
    update (update initial stackId [w COL_STACK_WRITE0_ADDR_LO] (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)))
      stackId [w COL_STACK_WRITE0_ADDR_LO + 1] (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)) stackId address.
Proof.
  intros initial final w Hprime Hfull Hmove Hreplay.
  pose proof (full_initialization_stack w COL_FRAME_MOVE_ACTIVE_BEFORE Hprime Hfull ltac:(right; reflexivity) Hmove)
    as Hshape.
  destruct (full_frame_move_row w Hprime Hfull Hmove) as [_ [_ [_ [_ [Hlo Hhi]]]]].
  pose proof (init_write_hi_address w _ Hshape) as HaddrHi.
  rewrite (initialization_stack_event_prefix w _ Hshape) in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [Hstack Hrest]].
  cbn [initializationStackEvents Nat.eqb app COL_FRAME_MOVE_ACTIVE_BEFORE] in Hstack.
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.
  split; [assumption | ]. intro address.
  erewrite replay_preserves; [ | exact Hrest | apply read_only_row; vm_compute; reflexivity].
  rewrite HaddrHi, <- Hlo, <- Hhi. reflexivity.
Qed.

(* A padding row is a fixed point: with no countdown pending it keeps every
   carried register, writes no stack cell and touches no call context. *)
Lemma full_padding_row : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_PADDING_ACTIVE = 1 ->
  w COL_LOCAL_ZERO_REMAINING_BEFORE = 0 -> w COL_FRAME_MOVE_REMAINING_BEFORE = 0 ->
  w COL_PC_AFTER = w COL_PC_BEFORE /\ w COL_SP_AFTER = w COL_SP_BEFORE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE /\
  w COL_LOCAL_ZERO_REMAINING_AFTER = 0 /\ w COL_FRAME_MOVE_REMAINING_AFTER = 0 /\
  w COL_LOCAL_ZERO_ACTIVE_AFTER = 0 /\ w COL_FRAME_MOVE_ACTIVE_AFTER = 0 /\
  w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0 /\ w COL_FRAME_MOVE_ACTIVE_BEFORE = 0 /\
  w COL_FRAME_MOVE_DISTANCE_AFTER = w COL_FRAME_MOVE_DISTANCE_BEFORE /\
  (forall address, Forall (NoWriteTo stackId address) (eventsForRow MemoryCatalog.memories w)) /\
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_HOST_SUSPEND = 0 /\ w COL_HOST_CONTEXT_READ = 0.
Proof.
  intros w Hprime Hfull Hpadding Hzr Hmr.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hother : forall c, In c rowKindFlags -> c <> COL_PADDING_ACTIVE -> w c = 0).
  { intros c Hc Hneq. exact (full_row_kind_exclusive w COL_PADDING_ACTIVE c Hprime Hfull ltac:(cbn; tauto) Hc
      ltac:(congruence) Hpadding). }
  pose proof (Hother COL_IS_PROGRAM_ROW ltac:(cbn; tauto) ltac:(discriminate)) as Hprogram.
  pose proof (Hother COL_FRAME_MOVE_ACTIVE_BEFORE ltac:(cbn; tauto) ltac:(discriminate)) as Hma.
  pose proof (Hother COL_LOCAL_ZERO_ACTIVE_BEFORE ltac:(cbn; tauto) ltac:(discriminate)) as Hza.
  pose proof (Hother COL_GATHER_ACTIVE ltac:(cbn; tauto) ltac:(discriminate)) as Hgather.
  pose proof (Hother COL_TURN_BOUNDARY ltac:(cbn; tauto) ltac:(discriminate)) as Hturn.
  pose proof (Hother COL_HOST_RESUME ltac:(cbn; tauto) ltac:(discriminate)) as Hresume.
  pose proof (Hother COL_HOST_SUSPEND ltac:(cbn; tauto) ltac:(discriminate)) as Hsuspend.
  pose proof (Hother COL_PERM_PENDING_BEFORE ltac:(cbn; tauto) ltac:(discriminate)) as Hpending.
  assert (Hperm : w COL_PERM_ROUND_BEFORE_IS_ZERO = 1).
  { destruct (full_row_kind_sum w Hprime Hfull) as [Hsum Hbound].
    pose proof (selector_sum_one w rowKindFlags COL_PADDING_ACTIVE ltac:(cbn; tauto)). lia. }
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hselectors.
  pose proof (full_quiet_from_gates w Hprime Hfull Hgather) as [Hkinds Hhi Hlocal Houtput].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (Hnested : w COL_HOST_NESTED_ENTRY = 0).
  { circuit_row ([(COL_TURN_BOUNDARY, 1)], [(COL_HOST_AWAITING_BEFORE, 1)], [(COL_HOST_NESTED_ENTRY, 1)]) Hrow.
    expand_row Hrow. rewrite Hturn in Hrow.
    pose proof (value_bound w COL_HOST_NESTED_ENTRY). field_linear Hrow. }
  pose proof (full_no_host_context_read w Hfull Hresume Hnested) as Hread.
  pose proof (full_no_resume_pop w Hfull Hresume) as HresumePop.
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { full_row guestEntryFlagRow Hrow. zero_opcode_columns Hselectors Hrow.
    pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE). field_linear Hrow. }
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { full_row callStackPushRow Hrow. zero_opcode_columns Hselectors Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { full_row callStackPopSupportRow Hrow. zero_opcode_columns Hselectors Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  assert (Hlatch : w COL_HOST_EVENT_EXIT_LATCH = 0).
  { pose proof (Hselectors COL_SEL_END ltac:(vm_compute; tauto)) as Hend.
    pose proof (Hselectors COL_SEL_RETURN ltac:(vm_compute; tauto)) as Hret.
    circuit_row ([(COL_HOST_EVENT_EXIT_LATCH, 1)], [(COL_ONE, 1); (COL_SEL_END, -1); (COL_SEL_RETURN, -1)],
      ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hend, Hret in Hrow.
    pose proof (value_bound w COL_HOST_EVENT_EXIT_LATCH). field_linear Hrow. }
  pose proof (full_frame_sp_drop_idle w Hprime Hfull (full_aux_no_slide w Hprime Hfull Hprogram Hpop)
    ltac:(congruence) Hturn Hlatch) as Hdiscard.
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { auxiliary_row outputCaptureSupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { auxiliary_row hostCallSupportRow Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. pose proof (full_no_host_split w Hfull Hhost) as Hsplit. normalize_columns.
    rewrite Hhost, Hsplit in Hrow. pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = 0).
  { auxiliary_row stackReadsArityRow Hrow. normalize_columns. rewrite Hone, Hgather, Hma in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { auxiliary_row stackWritesArityRow Hrow. normalize_columns. rewrite Hone, Hgather, Hma, Hza in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. normalize_columns. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. normalize_columns. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { auxiliary_row localWriteGateIndex Hrow. normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  (* The registers the padding constraints pin directly. *)
  circuit_range COL_PC_BEFORE Hpcb. circuit_range COL_PC_AFTER Hpca.
  circuit_range COL_SP_BEFORE Hspb. circuit_range COL_SP_AFTER Hspa.
  circuit_range COL_STACK_FRAME_BASE_BEFORE Hbb. circuit_range COL_STACK_FRAME_BASE_AFTER Hba.
  circuit_range COL_LOCALS_FBP_BEFORE Hfb. circuit_range COL_LOCALS_FBP_AFTER Hfa.
  circuit_range COL_CALL_STACK_DEPTH_BEFORE Hdb. circuit_range COL_CALL_STACK_DEPTH_AFTER Hda.
  circuit_range COL_FRAME_MOVE_REMAINING_AFTER Hmra. circuit_range COL_LOCAL_ZERO_REMAINING_AFTER Hzra.
  circuit_range COL_FRAME_MOVE_DISTANCE_BEFORE Hdib. circuit_range COL_FRAME_MOVE_DISTANCE_AFTER Hdia.
  circuit_row ([(COL_HOST_SUSPEND, 1); (COL_FRAME_MOVE_ACTIVE_BEFORE, 1); (COL_LOCAL_ZERO_ACTIVE_BEFORE, 1);
    (COL_PADDING_ACTIVE, 1); (COL_GATHER_ACTIVE, 1); (COL_PERM_PENDING_BEFORE, 1); (COL_ONE, 1);
    (COL_PERM_ROUND_BEFORE_IS_ZERO, -1)], [(COL_PC_AFTER, 1); (COL_PC_BEFORE, -1)], ([] : SparseRow)) Hpc.
  circuit_row ([(COL_PADDING_ACTIVE, 1)], [(COL_LOCALS_FBP_AFTER, 1); (COL_LOCALS_FBP_BEFORE, -1)],
    ([] : SparseRow)) Hfbp.
  circuit_row ([(COL_PADDING_ACTIVE, 1)], [(COL_STACK_FRAME_BASE_AFTER, 1); (COL_STACK_FRAME_BASE_BEFORE, -1)],
    ([] : SparseRow)) Hbase.
  circuit_row ([(COL_PADDING_ACTIVE, 1)], [(COL_CALL_STACK_DEPTH_AFTER, 1); (COL_CALL_STACK_DEPTH_BEFORE, -1)],
    ([] : SparseRow)) Hdepth.
  circuit_row ([(COL_PADDING_ACTIVE, 1)], [(COL_FRAME_MOVE_REMAINING_AFTER, 1);
    (COL_FRAME_MOVE_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hmove.
  circuit_row ([(COL_FRAME_MOVE_ACTIVE_BEFORE, 1); (COL_PADDING_ACTIVE, 1)],
    [(COL_FRAME_MOVE_DISTANCE_AFTER, 1); (COL_FRAME_MOVE_DISTANCE_BEFORE, -1)], ([] : SparseRow)) Hdistance.
  circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_TURN_BOUNDARY, -1);
    (COL_LOCAL_ZERO_ACTIVE_BEFORE, -1)],
    [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hzero.
  expand_row Hpc. expand_row Hfbp. expand_row Hbase. expand_row Hdepth. expand_row Hmove. expand_row Hdistance.
  expand_row Hzero.
  rewrite Hsuspend, Hma, Hza, Hpadding, Hgather, Hpending, Hone, Hperm in Hpc.
  rewrite Hpadding in Hfbp, Hbase, Hdepth, Hmove. rewrite Hma, Hpadding in Hdistance.
  rewrite Hone, Hguest, Hturn, Hza, Hzr in Hzero.
  assert (HmoveAfter : w COL_FRAME_MOVE_REMAINING_AFTER = 0) by (rewrite Hmr in Hmove; field_linear Hmove).
  assert (HzeroAfter : w COL_LOCAL_ZERO_REMAINING_AFTER = 0) by field_linear Hzero.
  destruct (full_initialization_flags w Hprime Hfull) as [HmaFlag HzaFlag].
  rewrite HmoveAfter in HmaFlag, HzaFlag. rewrite HzeroAfter in HzaFlag. cbn in HmaFlag, HzaFlag.
  assert (Hsp : w COL_SP_AFTER = w COL_SP_BEFORE).
  { full_row stackPointerBalanceRow Hrow. normalize_columns. rewrite HresumePop in Hrow. zero_gather_kinds Hkinds Hrow.
    normalize_columns. rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
    field_linear Hrow. }
  repeat split; try assumption; try field_linear Hpc; try field_linear Hfbp; try field_linear Hbase;
    try field_linear Hdepth; try field_linear Hdistance.
  intros address. apply stack_writes_inactive; assumption.
Qed.
