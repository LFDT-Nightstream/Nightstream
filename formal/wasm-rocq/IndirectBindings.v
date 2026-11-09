(* The circuit's call_indirect rows. The resolve row replaces the table index on
   top with the checked function reference, or traps; the invocation row pops that
   reference and enters a guest callee. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings CallInitialization CallBindings CallMemory CallContext StackBindings TrapFacts.
Import ListNotations.
Open Scope Z_scope.

Lemma full_indirect_guest_call : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_GUEST_ENTRY_ACTIVE = 1 /\ w COL_CALL_STACK_PUSH_PRESENT = 1 /\
  w COL_CALL_STACK_POP_PRESENT = 0.
Proof.
  intros w Hprime Hfull Hchosen Htarget.
  assert (Hin : In COL_SEL_CALL_INDIRECT selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  selected_row COL_SEL_CALL_INDIRECT guestEntryFlagRow Hguest.
  selected_row COL_SEL_CALL_INDIRECT callStackPushRow Hpush.
  selected_row COL_SEL_CALL_INDIRECT callStackPopSupportRow Hpop.
  normalize_columns. rewrite Hone, Htarget in Hguest, Hpush. rewrite Hone in Hpop.
  pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE).
  pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT).
  pose proof (value_bound w COL_CALL_STACK_POP_PRESENT).
  repeat split; [field_linear Hguest | field_linear Hpush | field_linear Hpop].
Qed.

(* The invocation reads the reference from the top cell, writes nothing, and
   leaves the stack pointer below it. *)
Lemma full_indirect_guest_call_stack : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_FUNCTION_REF = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) /\
  w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1) /\
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 /\
  filter isStackWrite (eventsForRow memories w) = [] /\
  w COL_SP_AFTER = w COL_SP_BEFORE - 1.
Proof.
  intros w Hprime Hfull Hchosen Htarget.
  assert (Hin : In COL_SEL_CALL_INDIRECT selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero Hgather]].
  destruct (full_program_row_quiet w COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  destruct (full_indirect_guest_call w Hprime Hfull Hchosen Htarget) as [Hguest [Hpush Hpop]].
  assert (Hdiscard : w COL_FRAME_SP_DROP = 0).
  { pose proof (full_selected_is_program_row w COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hchosen) as Hprogram.
    apply full_frame_sp_drop_idle; try assumption; [ | congruence | | ].
    - apply (full_selected_no_slide w COL_SEL_CALL_INDIRECT); try assumption; vm_compute; intuition discriminate.
    - exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram).
    - selected_circuit_row COL_SEL_CALL_INDIRECT
        ([(COL_HOST_EVENT_EXIT_LATCH, 1)], [(COL_ONE, 1); (COL_SEL_END, -1); (COL_SEL_RETURN, -1)],
        ([] : SparseRow)) Hrow.
      rewrite Hone in Hrow. pose proof (value_bound w COL_HOST_EVENT_EXIT_LATCH). field_linear Hrow. }
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { selected_row COL_SEL_CALL_INDIRECT outputCaptureSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_HOST_CALL_ACTIVE, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1);
        (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1); (COL_GUEST_ENTRY_ACTIVE, 1)],
        [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    normalize_columns. rewrite Hone, Hguest in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. pose proof (full_no_host_split w Hfull Hhost) as Hsplit. normalize_columns.
    rewrite Hhost, Hsplit in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = 1).
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
        (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_STACK_READS, 1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1)],
        ([] : SparseRow)) Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
        (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_STACK_WRITES, 1)], ([] : SparseRow)) Hrow.
    rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { selected_row COL_SEL_CALL_INDIRECT localWriteGateIndex Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { selected_row COL_SEL_CALL_INDIRECT stackRead2SupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread1 : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { selected_row COL_SEL_CALL_INDIRECT stackRead1SupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hsp : w COL_SP_AFTER = w COL_SP_BEFORE - 1).
  { circuit_range COL_SP_BEFORE HspBefore. circuit_range COL_SP_AFTER HspAfter.
    full_row stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
    field_linear Hrow. }
  assert (Hpositive : 1 <= w COL_SP_BEFORE) by (circuit_range COL_SP_AFTER HspAfter; lia).
  split.
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_FUNCTION_REF, 1); (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hrow.
    rewrite Hone in Hrow. pose proof (value_bound w COL_FUNCTION_REF).
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)). field_linear Hrow. }
  split.
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, 1); (COL_SP_BEFORE, -2); (COL_ONE, 2)], ([] : SparseRow)) Hrow.
    normalize_columns. rewrite Hone in Hrow. circuit_range COL_SP_BEFORE HspRange.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow. }
  split.
  { full_row stackReadsActiveRow Hrow. rewrite Hone, Hreads, Hread1, Hread2 in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  split.
  { rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite Hwrite, HwriteHi, HlocalWrite, HgatherLocalLo, HgatherLocal. reflexivity. }
  exact Hsp.
Qed.

Lemma indirect_metadata_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_SEL_CALL_INDIRECT = 1 ->
  initial functionCallMetadataId [w COL_FUNCTION_REF] = Some (w COL_CALL_TARGET_METADATA).
Proof.
  intros initial final w Hreplay Hcall.
  exported_read functionCallMetadataId [COL_FUNCTION_REF] COL_CALL_TARGET_METADATA COL_SEL_CALL_INDIRECT.
Qed.

(* A row that reads the top operand and writes its cell in place. *)
Record ReplaceStackShape (w : Witness) : Prop := {
  replace_read0_active : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1;
  replace_read1_inactive : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  replace_read2_inactive : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  replace_write_active : w COL_STACK_WRITE0_ACTIVE = 1;
  replace_write_hi_active : w COL_STACK_WRITE0_HI_ACTIVE = 1;
  replace_capture_inactive : w COL_OUTPUT_CAPTURED = 0;
  replace_read_address : w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1);
  replace_write_address : w COL_STACK_WRITE0_ADDR_LO = 2 * (w COL_SP_BEFORE - 1);
  replace_write_hi_address : w COL_STACK_WRITE0_ADDR_HI = 2 * (w COL_SP_BEFORE - 1) + 1;
  replace_stack_pointer : w COL_SP_AFTER = w COL_SP_BEFORE;
  replace_stack_base : w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE;
  replace_local_get_inactive : w COL_SEL_LOCAL_GET = 0;
  replace_local_write_inactive : w COL_LOCAL_WRITE_ENABLED = 0;
  replace_gmem_inactive : w GMEM_LOCAL = 0;
  replace_gather_lo_inactive : w COL_GATHER_LOCAL_WRITE_LO = 0;
  replace_gather_inactive : w COL_GATHER_LOCAL_WRITE = 0
}.

Theorem full_resolve_stack_shape : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT_RESOLVE = 1 ->
  1 <= w COL_SP_BEFORE -> ReplaceStackShape w.
Proof.
  unary_stack_facts COL_SEL_CALL_INDIRECT_RESOLVE 1 1 0.
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w COL_SEL_CALL_INDIRECT_RESOLVE COL_SEL_LOCAL_GET Hprime Hfull
      Hchosen Hin _ _); [vm_compute; tauto | discriminate]. }
  assert (Haddr : w COL_STACK_WRITE0_ADDR_LO = 2 * (w COL_SP_BEFORE - 1)).
  { selected_row_where COL_SEL_CALL_INDIRECT_RESOLVE (@None SparseRow)
      (Some [(COL_STACK_WRITE0_ADDR_LO, 1); (COL_SP_BEFORE, -2); (COL_ONE, 2)]) (@None SparseRow) Hrow.
    rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_LO). field_linear Hrow. }
  constructor; try assumption.
  - selected_row COL_SEL_CALL_INDIRECT_RESOLVE unaryReadAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - selected_row COL_SEL_CALL_INDIRECT_RESOLVE stackWriteHighAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_HI). normalize_columns. rewrite Haddr in Hrow.
    field_linear Hrow.
  - selected_row COL_SEL_CALL_INDIRECT_RESOLVE stackPointerBalanceRow Hrow. zero_host_switch Hrow.
    zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    pose proof (value_bound w COL_SP_AFTER). field_linear Hrow.
  - selected_row COL_SEL_CALL_INDIRECT_RESOLVE stackFrameBasePreservedRow Hrow. zero_host_switch Hrow.
    rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

Definition replaceStackEvents (w : Witness) : list Event :=
  [ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
     (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
   ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
     (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat));
   WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO] (w COL_STACK_WRITE0_VALUE_LO) None;
   WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI] (w COL_STACK_WRITE0_VALUE_HI) None].

Lemma replace_stack_event_prefix : forall w,
  ReplaceStackShape w ->
  eventsForRow MemoryCatalog.memories w = replaceStackEvents w ++ eventsForRow (tl MemoryCatalog.memories) w.
Proof.
  intros w Hshape.
  destruct Hshape as [Hr0 Hr1 Hr2 Hw Hwh Hcap Ha0 Haw Hawh Hsp Hbase Hget Hset Hgmem Hglo Hg].
  change (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with
      Some event => [event] | None => [] end)
    (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
    eventsForRow (tl MemoryCatalog.memories) w =
    replaceStackEvents w ++ eventsForRow (tl MemoryCatalog.memories) w).
  cbn [hd MemoryCatalog.memories ports flat_map portEvent active activation
    addressColumns valueColumn portKind map option_map].
  unfold portEvent, active. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hr0, Hr1, Hr2, Hw, Hwh, Hcap, Hget, Hset, Hgmem, Hglo, Hg. reflexivity.
Qed.

Theorem replace_stack_replay : forall w initial final,
  ReplaceStackShape w -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  forall address, final MemoryCatalog.stackId address =
    update (update initial MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO]
      (w COL_STACK_WRITE0_VALUE_LO)) MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI]
      (w COL_STACK_WRITE0_VALUE_HI) MemoryCatalog.stackId address.
Proof.
  intros w initial final Hshape Hreplay.
  rewrite (replace_stack_event_prefix w Hshape) in Hreplay.
  destruct (replay_app_inv (replaceStackEvents w) _ _ _ Hreplay) as [middle [Hstack Hrest]].
  unfold replaceStackEvents in Hstack.
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.
  split; [assumption | ]. intro address.
  eapply replay_preserves; [exact Hrest | ].
  apply read_only_row. exact stack_only_first_memory.
Qed.

(* The resolve row's checks: the index against the table's size, the entry
   against null, and, when both pass, the entry's type id against the expected
   one. It traps unless all three pass, and otherwise writes the reference. *)
Lemma full_resolve_checks : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT_RESOLVE = 1 ->
  w COL_PROGRAM_TABLE_ID_ACTIVE = 1 /\ w COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE = 1 /\
  w COL_TABLE_SIZE_READ_ENABLED = 1 /\
  w COL_TABLE_INDEX = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) /\
  w COL_CI_OOB = (if w COL_TABLE_INDEX <? w COL_TABLE_SIZE then 0 else 1) /\
  w COL_TABLE_READ_ENABLED = 1 - w COL_CI_OOB /\
  w COL_CI_ENTRY_IS_NULL = (if w COL_TABLE_VALUE =? 0 then 1 else 0) /\
  w COL_CI_TYPE_EQ = (if w COL_FUNCTION_TYPE_ID =? w COL_EXPECTED_TYPE_ID then 1 else 0) /\
  w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = (1 - w COL_CI_OOB) * (1 - w COL_CI_ENTRY_IS_NULL) /\
  (w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 1 -> w COL_FUNCTION_REF = w COL_TABLE_VALUE) /\
  w COL_CALL_INDIRECT_IS_TRAP = 1 - w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE * w COL_CI_TYPE_EQ /\
  (w COL_CALL_INDIRECT_IS_TRAP = 0 -> w COL_STACK_WRITE0_VALUE_LO = w COL_FUNCTION_REF).
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hothers : w COL_SEL_TABLE_SIZE = 0 /\ w COL_SEL_TABLE_GET = 0 /\ w COL_SEL_TABLE_SET = 0).
  { repeat split; eapply full_opcode_selector_exclusive; try exact Hchosen; try exact Hin;
      try eassumption; vm_compute; first [tauto | discriminate]. }
  destruct Hothers as [Hsize [Hget Hset]].
  circuit_boolean COL_CMP_GE Hge. circuit_boolean COL_CI_ENTRY_IS_NULL Hnull.
  circuit_boolean COL_CI_TYPE_EQ Heq. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Htrap.
  circuit_range COL_TABLE_SIZE HsizeRange. circuit_range COL_CMP_LOW Hlow.
  pose proof (full_read0_u32 w Hprime Hfull) as Hread.
  pose proof (value_bound w COL_TABLE_INDEX). pose proof (value_bound w COL_TABLE_VALUE).
  pose proof (value_bound w COL_FUNCTION_TYPE_ID). pose proof (value_bound w COL_EXPECTED_TYPE_ID).
  pose proof (value_bound w COL_FUNCTION_REF). pose proof (value_bound w COL_STACK_WRITE0_VALUE_LO).
  pose proof (value_bound w COL_CI_OOB). pose proof (value_bound w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE).
  pose proof (value_bound w COL_TABLE_READ_ENABLED). pose proof (value_bound w COL_TABLE_SIZE_READ_ENABLED).
  pose proof (value_bound w COL_PROGRAM_TABLE_ID_ACTIVE).
  pose proof (value_bound w COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE).
  pose proof (value_bound w COL_CALL_INDIRECT_IS_NOT_TRAP).
  circuit_row ([(COL_PROGRAM_TABLE_ID_ACTIVE, 1); (COL_SEL_TABLE_SIZE, -1); (COL_SEL_TABLE_GET, -1); (COL_SEL_TABLE_SET, -1); (COL_SEL_CALL_INDIRECT_RESOLVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hid. expand_row Hid.
  circuit_row ([(COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE, 1); (COL_SEL_CALL_INDIRECT_RESOLVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Himm. expand_row Himm.
  circuit_row ([(COL_TABLE_SIZE_READ_ENABLED, 1); (COL_SEL_TABLE_SIZE, -1); (COL_SEL_TABLE_GET, -1); (COL_SEL_TABLE_SET, -1); (COL_SEL_CALL_INDIRECT_RESOLVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) HsizeRead. expand_row HsizeRead.
  circuit_row ([(COL_SEL_TABLE_GET, 1)], [(COL_TABLE_OOB, 1)], [(COL_SEL_TABLE_GET, 1); (COL_SEL_CALL_INDIRECT_RESOLVE, 1); (COL_CI_OOB, -1); (COL_TABLE_READ_ENABLED, -1)]) Hread'. expand_row Hread'.
  circuit_row ([(COL_SEL_CALL_INDIRECT_RESOLVE, 1)], [(COL_TABLE_INDEX, 1); (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hindex. expand_row Hindex.
  circuit_row ([(COL_SEL_TABLE_GET, 1); (COL_SEL_TABLE_SET, 1); (COL_SEL_CALL_INDIRECT_RESOLVE, 1)], [(COL_TABLE_INDEX, 1); (COL_TABLE_SIZE, -1); (COL_ONE, 4294967296); (COL_CMP_LOW, -1); (COL_CMP_GE, -4294967296)], ([] : SparseRow)) Hcmp. expand_row Hcmp.
  circuit_row ([(COL_SEL_CALL_INDIRECT_RESOLVE, 1)], [(COL_CMP_GE, 1)], [(COL_CI_OOB, 1)]) Hoob. expand_row Hoob.
  circuit_row ([(COL_TABLE_VALUE, 1)], [(COL_CI_ENTRY_NULL_INV, 1)], [(COL_ONE, 1); (COL_CI_ENTRY_IS_NULL, -1)]) HnullInv. expand_row HnullInv.
  circuit_row ([(COL_TABLE_VALUE, 1)], [(COL_CI_ENTRY_IS_NULL, 1)], ([] : SparseRow)) HnullZero. expand_row HnullZero.
  circuit_row ([(COL_FUNCTION_TYPE_ID, 1); (COL_EXPECTED_TYPE_ID, -1)], [(COL_CI_TYPE_EQ_INV, 1)], [(COL_ONE, 1); (COL_CI_TYPE_EQ, -1)]) HeqInv. expand_row HeqInv.
  circuit_row ([(COL_FUNCTION_TYPE_ID, 1); (COL_EXPECTED_TYPE_ID, -1)], [(COL_CI_TYPE_EQ, 1)], ([] : SparseRow)) HeqZero. expand_row HeqZero.
  circuit_row ([(COL_SEL_CALL_INDIRECT_RESOLVE, 1); (COL_CI_OOB, -1)], [(COL_ONE, 1); (COL_CI_ENTRY_IS_NULL, -1)], [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)]) Hgate. expand_row Hgate.
  circuit_row ([(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)], [(COL_CI_TYPE_EQ, 1)], [(COL_SEL_CALL_INDIRECT_RESOLVE, 1); (COL_CALL_INDIRECT_IS_TRAP, -1)]) Htrapped. expand_row Htrapped.
  circuit_row ([(COL_SEL_CALL_INDIRECT_RESOLVE, 1)], [(COL_ONE, 1); (COL_CALL_INDIRECT_IS_TRAP, -1)], [(COL_CALL_INDIRECT_IS_NOT_TRAP, 1)]) HnotTrap. expand_row HnotTrap.
  circuit_row ([(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)], [(COL_FUNCTION_REF, 1); (COL_TABLE_VALUE, -1)], ([] : SparseRow)) Hlookup. expand_row Hlookup.
  circuit_row ([(COL_CALL_INDIRECT_IS_NOT_TRAP, 1)], [(COL_STACK_WRITE0_VALUE_LO, 1); (COL_FUNCTION_REF, -1)], ([] : SparseRow)) Hwrite. expand_row Hwrite.
  normalize_columns.
  rewrite Hone, Hchosen, Hsize, Hget, Hset in *.
  assert (HindexEq : w COL_TABLE_INDEX = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)).
  { normalize_columns. field_linear Hindex. }
  assert (Hoobv : w COL_CI_OOB = (if w COL_TABLE_INDEX <? w COL_TABLE_SIZE then 0 else 1)).
  { normalize_columns. rewrite HindexEq in Hcmp |- *. destruct Hge as [Hg | Hg]; rewrite Hg in *.
    - assert (w COL_CI_OOB = 0) by field_linear Hoob.
      assert (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) < w COL_TABLE_SIZE) by field_linear Hcmp.
      normalize_columns. replace (_ <? _) with true by (symmetry; apply Z.ltb_lt; lia). assumption.
    - assert (w COL_CI_OOB = 1) by field_linear Hoob.
      assert (w COL_TABLE_SIZE <= w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) by field_linear Hcmp.
      normalize_columns. replace (_ <? _) with false by (symmetry; apply Z.ltb_ge; lia). assumption. }
  assert (Hnullv : w COL_CI_ENTRY_IS_NULL = (if w COL_TABLE_VALUE =? 0 then 1 else 0)).
  { destruct (Z.eqb_spec (w COL_TABLE_VALUE) 0) as [Hz | Hz]; normalize_columns.
    - rewrite Hz in HnullInv. field_linear HnullInv.
    - destruct Hnull as [Hn | Hn]; [exact Hn | exfalso; rewrite Hn in HnullZero; field_linear HnullZero]. }
  assert (Heqv : w COL_CI_TYPE_EQ = (if w COL_FUNCTION_TYPE_ID =? w COL_EXPECTED_TYPE_ID then 1 else 0)).
  { destruct (Z.eqb_spec (w COL_FUNCTION_TYPE_ID) (w COL_EXPECTED_TYPE_ID)) as [He | He]; normalize_columns.
    - rewrite He in HeqInv. field_linear HeqInv.
    - destruct Heq as [Hn | Hn]; [exact Hn | exfalso; rewrite Hn in HeqZero; field_linear HeqZero]. }
  assert (Hgatev : w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = (1 - w COL_CI_OOB) * (1 - w COL_CI_ENTRY_IS_NULL)).
  { normalize_columns. rewrite Hoobv, Hnullv in Hgate |- *.
    destruct (_ <? _), (_ =? 0); field_linear Hgate. }
  assert (Htrapv : w COL_CALL_INDIRECT_IS_TRAP =
      1 - w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE * w COL_CI_TYPE_EQ).
  { normalize_columns. rewrite Hgatev, Heqv in Htrapped |- *. rewrite Hoobv, Hnullv in Htrapped |- *.
    destruct (_ <? _), (_ =? 0), (_ =? _); field_linear Htrapped. }
  split; [field_linear Hid | ]. split; [field_linear Himm | ]. split; [field_linear HsizeRead | ].
  split; [exact HindexEq | ]. split; [exact Hoobv | ].
  split; [assert (0 <= w COL_CI_OOB <= 1) by (rewrite Hoobv; destruct (_ <? _); lia);
    normalize_columns; field_linear Hread' | ].
  split; [exact Hnullv | ]. split; [exact Heqv | ]. split; [exact Hgatev | ].
  split; [intros Hg; normalize_columns; rewrite Hg in Hlookup; field_linear Hlookup | ].
  split; [exact Htrapv | ].
  intros Ht. assert (Hnot : w COL_CALL_INDIRECT_IS_NOT_TRAP = 1) by (normalize_columns; rewrite Ht in HnotTrap; field_linear HnotTrap).
  normalize_columns. rewrite Hnot in Hwrite. field_linear Hwrite.
Qed.

Lemma resolve_rom_reads : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PROGRAM_TABLE_ID_ACTIVE = 1 -> w COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE = 1 ->
  w COL_TABLE_SIZE_READ_ENABLED = 1 ->
  initial programTableId [w COL_PC_BEFORE] = Some (w COL_TABLE_ID) /\
  initial expectedTypeId [w COL_PC_BEFORE] = Some (w COL_EXPECTED_TYPE_ID) /\
  initial tableSizeId [w COL_TABLE_ID] = Some (w COL_TABLE_SIZE).
Proof.
  intros initial final w Hreplay Hid Himm Hsize. split; [ | split].
  - exported_read programTableId [COL_PC_BEFORE] COL_TABLE_ID COL_PROGRAM_TABLE_ID_ACTIVE.
  - exported_read expectedTypeId [COL_PC_BEFORE] COL_EXPECTED_TYPE_ID COL_PROGRAM_CALL_INDIRECT_IMMEDIATES_ACTIVE.
  - exported_read tableSizeId [COL_TABLE_ID] COL_TABLE_SIZE COL_TABLE_SIZE_READ_ENABLED.
Qed.

Lemma resolve_type_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 1 ->
  initial functionTypeId [w COL_FUNCTION_REF] = Some (w COL_FUNCTION_TYPE_ID).
Proof.
  intros initial final w Hreplay Hgate.
  exported_read functionTypeId [COL_FUNCTION_REF] COL_FUNCTION_TYPE_ID COL_FUNCTION_CALL_TYPE_LOOKUP_GATE.
Qed.

(* Only table.set writes a table entry, so a row without it reads the entry as it was. *)
Lemma resolve_entry_read : forall initial final (w : Witness),
  FullRelation.satFull w ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final -> w COL_SEL_TABLE_SET <> 1 ->
  w COL_TABLE_READ_ENABLED = 1 ->
  initial tableElementId [w COL_TABLE_ID; w COL_TABLE_INDEX] = Some (w COL_TABLE_VALUE).
Proof.
  intros initial final w Hfull Hreplay Hset Hread.
  eapply replay_read_from_initial; [eassumption | | ].
  - apply gated_write_row with (column := COL_TABLE_SET_LIVE); [vm_compute; reflexivity | ].
    intros Hlive. apply Hset. pose proof (proj1 Hfull) as Hone.
    circuit_row ([(COL_TABLE_SET_LIVE, 1)], [(COL_ONE, 1); (COL_SEL_TABLE_SET, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hone, Hlive in Hrow.
    pose proof (value_bound w COL_SEL_TABLE_SET). field_linear Hrow.
  - exported_event tableElementId (mkPort [COL_TABLE_ID; COL_TABLE_INDEX] COL_TABLE_VALUE Read
      (When COL_TABLE_READ_ENABLED)).
Qed.

(* The resolve row stays in its frame: no call-stack change, the same registers. *)
Lemma full_resolve_frame : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT_RESOLVE = 1 ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_CALL_STACK_POP_PRESENT = 0 /\
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE.
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { selected_row COL_SEL_CALL_INDIRECT_RESOLVE callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { selected_row COL_SEL_CALL_INDIRECT_RESOLVE callStackPopSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate)
      (full_selected_is_program_row w _ Hprime Hfull Hin Hchosen)). }
  split; [exact Hpush | ]. split; [exact Hpop | ]. split.
  - full_row localsBasePreservedRow Hrow. zero_host_switch Hrow. rewrite Hone, Hpush, Hpop, Hturn in Hrow.
    pose proof (value_bound w COL_LOCALS_FBP_BEFORE).
    pose proof (value_bound w COL_LOCALS_FBP_AFTER). field_linear Hrow.
  - pose proof (full_call_depth_balance w _ Hprime Hfull Hin Hchosen). lia.
Qed.

(* A trapping resolve row finishes the activation. *)
Lemma full_resolve_trap_finishes : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT_RESOLVE = 1 ->
  w COL_CALL_INDIRECT_IS_TRAP = 1 -> w COL_TRAPPED_AFTER = 1 /\ w COL_ACTIVATION_FINISHED_AFTER = 1.
Proof.
  intros w Hprime Hfull Hchosen Htrap.
  assert (Hin : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w _ Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hothers : w COL_SEL_UNREACHABLE = 0) by
    (eapply full_opcode_selector_exclusive; try exact Hchosen; try exact Hin; try eassumption;
      vm_compute; first [tauto | discriminate]).
  assert (Htb : w COL_TRAPPED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow. }
  assert (Hfb : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  circuit_row ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1);
    (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Htrapped.
  circuit_boolean COL_TRAPPED_AFTER Hta. circuit_boolean COL_DIV_TRAP Hdiv. circuit_boolean COL_MEM_OOB Hmem.
  circuit_boolean COL_TABLE_OOB Htable.
  expand_row Htrapped. normalize_columns. rewrite Hone, Htb, Hothers, Htrap in Htrapped.
  assert (Htrapped' : w COL_TRAPPED_AFTER = 1).
  { normalize_columns. apply Z.mod_divide in Htrapped; [ | unfold modulus; lia].
    destruct Htrapped as [k Htrapped]. unfold modulus in *; lia. }
  split; [exact Htrapped' | ].
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hstate.
  expand_row Hstate. normalize_columns. rewrite Hone, Hfb, Htrapped', Htb, Hclear in Hstate.
  circuit_boolean COL_HOST_EVENT_EXIT_LATCH Hlatch. circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hfa.
  normalize_columns. apply Z.mod_divide in Hstate; [ | unfold modulus; lia].
  destruct Hstate as [k Hstate]. unfold modulus in *; lia.
Qed.

(* The invocation's reference is its first stack read: the top cell. *)
Lemma full_indirect_reference : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT = 1 -> 1 <= w COL_SP_BEFORE ->
  w COL_FUNCTION_REF = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) /\
  w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1) /\
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1.
Proof.
  intros w Hprime Hfull Hchosen Hpositive.
  assert (Hin : In COL_SEL_CALL_INDIRECT selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hchosen) as [Hparam [Hzero Hgather]].
  assert (Hreads : w COL_STACK_READS = 1).
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
        (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_STACK_READS, 1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1)],
        ([] : SparseRow)) Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { selected_row COL_SEL_CALL_INDIRECT stackRead2SupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread1 : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { selected_row COL_SEL_CALL_INDIRECT stackRead1SupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  split.
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_FUNCTION_REF, 1); (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hrow.
    rewrite Hone in Hrow. pose proof (value_bound w COL_FUNCTION_REF).
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)). field_linear Hrow. }
  split.
  { selected_circuit_row COL_SEL_CALL_INDIRECT
      ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, 1); (COL_SP_BEFORE, -2); (COL_ONE, 2)], ([] : SparseRow)) Hrow.
    normalize_columns. rewrite Hone in Hrow. circuit_range COL_SP_BEFORE HspRange.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow. }
  full_row stackReadsActiveRow Hrow. rewrite Hone, Hreads, Hread1, Hread2 in Hrow.
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow.
Qed.

(* The first read port's event comes first in the row, so it sees the memory
   the row starts from. *)
Lemma first_stack_read : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 ->
  initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)).
Proof.
  intros initial final w Hreplay Hactive.
  change (eventsForRow MemoryCatalog.memories w) with
    (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with Some event => [event] | None => [] end)
      (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
     eventsForRow (tl MemoryCatalog.memories) w) in Hreplay.
  cbn [hd MemoryCatalog.memories ports flat_map] in Hreplay.
  unfold portEvent at 1 in Hreplay. unfold active at 1 in Hreplay.
  cbn [activation addressColumns valueColumn portKind map] in Hreplay.
  normalize_columns. rewrite Hactive in Hreplay. cbn [Z.eqb Pos.eqb app] in Hreplay.
  inversion Hreplay as [ | ? ? ? ? ? Hstep _]; subst.
  inversion Hstep; subst. assumption.
Qed.

(* A resolve row finishes exactly when it traps; its edge is static. *)
Lemma full_resolve_finish : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT_RESOLVE = 1 -> w COL_PC_EDGE_KIND = 0 ->
  w COL_ACTIVATION_FINISHED_AFTER = w COL_CALL_INDIRECT_IS_TRAP.
Proof.
  intros w Hprime Hfull Hchosen Hedge.
  assert (Hin : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w _ Hprime Hfull Hin Hchosen) as Hprogram.
  destruct (full_program_row_quiet w _ Hprime Hfull Hin Hchosen) as [Hkinds _ _ _].
  assert (Hzero : w COL_SEL_UNREACHABLE = 0 /\ w COL_SEL_CALL_INDIRECT = 0 /\ w COL_SEL_RETURN_CALL_INDIRECT = 0).
  { repeat split; eapply full_opcode_selector_exclusive; try exact Hchosen; try exact Hin; try eassumption;
      vm_compute; first [tauto | discriminate]. }
  destruct Hzero as [Hunreachable [Hindirect Htail]].
  destruct (full_resolve_frame w Hprime Hfull Hchosen) as [_ [Hpop _]].
  assert (Hfb : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_selected_no_host_switch w _ Hprime Hfull Hin Hchosen) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  selected_row_where COL_SEL_CALL_INDIRECT_RESOLVE (@None SparseRow) (@None SparseRow)
    (Some [(COL_DIV_TRAP, 1); (COL_DIV_OVERFLOW, -1)]) Hdiv.
  selected_row_where COL_SEL_CALL_INDIRECT_RESOLVE (@None SparseRow) (Some [(COL_DIV_OVERFLOW_COND, 1)])
    (Some [(COL_DIV_OVERFLOW, 1)]) Hoverflow.
  selected_row_where COL_SEL_CALL_INDIRECT_RESOLVE (@None SparseRow) (Some [(COL_CMP_GE, 1)])
    (Some [(COL_MEM_OOB, 1)]) Hmem.
  selected_row_where COL_SEL_CALL_INDIRECT_RESOLVE (@None SparseRow) (Some [(COL_CMP_GE, 1)])
    (Some [(COL_TABLE_OOB, 1)]) Htable.
  zero_gather_kinds Hkinds Hmem. rewrite ?Hone in *.
  pose proof (value_bound w COL_DIV_OVERFLOW). pose proof (value_bound w COL_DIV_TRAP).
  pose proof (value_bound w COL_MEM_OOB). pose proof (value_bound w COL_TABLE_OOB).
  assert (Hov : w COL_DIV_OVERFLOW = 0) by field_linear Hoverflow.
  assert (Hdt : w COL_DIV_TRAP = 0) by field_linear Hdiv.
  assert (Hmo : w COL_MEM_OOB = 0) by field_linear Hmem.
  assert (Hto : w COL_TABLE_OOB = 0) by field_linear Htable.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
    (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
    (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
    (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci. circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hfa.
  normalize_columns. rewrite Hone, Hfb, Hclear, Hpop, Hedge, Hindirect, Htail, Hunreachable, Hdt, Hmo, Hto in Hrow.
  normalize_columns. apply Z.mod_divide in Hrow; [ | unfold modulus; lia]. destruct Hrow as [k Hrow].
  unfold modulus in *; lia.
Qed.

(* An invocation row, on its dynamic edge, never finishes the activation. *)
Lemma full_invocation_continues : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL_INDIRECT = 1 -> w COL_PC_EDGE_KIND = 2 ->
  w COL_ACTIVATION_FINISHED_AFTER = 0.
Proof.
  intros w Hprime Hfull Hchosen Hedge.
  assert (Hin : In COL_SEL_CALL_INDIRECT selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w _ Hprime Hfull Hin Hchosen) as Hprogram.
  destruct (full_program_row_quiet w _ Hprime Hfull Hin Hchosen) as [Hkinds _ _ _].
  assert (Hzero : w COL_SEL_UNREACHABLE = 0 /\ w COL_SEL_RETURN_CALL_INDIRECT = 0).
  { split; eapply full_opcode_selector_exclusive; try exact Hchosen; try exact Hin; try eassumption;
      vm_compute; first [tauto | discriminate]. }
  destruct Hzero as [Hunreachable Htail].
  assert (Hfb : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_selected_no_host_switch w _ Hprime Hfull Hin Hchosen) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (@None SparseRow)
    (Some [(COL_DIV_TRAP, 1); (COL_DIV_OVERFLOW, -1)]) Hdiv.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (Some [(COL_DIV_OVERFLOW_COND, 1)])
    (Some [(COL_DIV_OVERFLOW, 1)]) Hoverflow.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (Some [(COL_CMP_GE, 1)])
    (Some [(COL_MEM_OOB, 1)]) Hmem.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (Some [(COL_CMP_GE, 1)])
    (Some [(COL_TABLE_OOB, 1)]) Htable.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (Some [(COL_CMP_GE, 1)])
    (Some [(COL_CI_OOB, 1)]) Hoob.
  selected_row_where COL_SEL_CALL_INDIRECT (@None SparseRow) (Some [(COL_ONE, 1); (COL_CI_ENTRY_IS_NULL, -1)])
    (Some [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)]) Hgate.
  selected_row_where COL_SEL_CALL_INDIRECT (Some [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)]) (Some [(COL_CI_TYPE_EQ, 1)])
    (@None SparseRow) Htype.
  zero_gather_kinds Hkinds Hmem. rewrite ?Hone in *.
  pose proof (value_bound w COL_DIV_OVERFLOW). pose proof (value_bound w COL_DIV_TRAP).
  pose proof (value_bound w COL_MEM_OOB). pose proof (value_bound w COL_TABLE_OOB). pose proof (value_bound w COL_CI_OOB).
  pose proof (value_bound w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE). pose proof (value_bound w COL_CALL_INDIRECT_IS_TRAP).
  assert (Hov : w COL_DIV_OVERFLOW = 0) by field_linear Hoverflow.
  assert (Hdt : w COL_DIV_TRAP = 0) by field_linear Hdiv.
  assert (Hmo : w COL_MEM_OOB = 0) by field_linear Hmem.
  assert (Hto : w COL_TABLE_OOB = 0) by field_linear Htable.
  assert (Hco : w COL_CI_OOB = 0) by field_linear Hoob.
  normalize_columns. rewrite Hco in Hgate.
  assert (Hg : w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 0) by field_linear Hgate.
  normalize_columns. rewrite Hg in Htype.
  assert (Hct : w COL_CALL_INDIRECT_IS_TRAP = 0) by field_linear Htype.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
    (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
    (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
    (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop. circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hfa.
  normalize_columns. rewrite Hone, Hfb, Hclear, Hedge, Hchosen, Htail, Hunreachable, Hdt, Hmo, Hto, Hct in Hrow.
  normalize_columns. apply Z.mod_divide in Hrow; [ | unfold modulus; lia]. destruct Hrow as [k Hrow].
  unfold modulus in *; lia.
Qed.
