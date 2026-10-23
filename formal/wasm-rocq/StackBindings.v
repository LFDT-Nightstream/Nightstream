(* Stack port activation, addresses, and memory effects for arithmetic and local instructions. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  CallInitialization LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Definition binaryArithmeticSelector (column : Column) : Prop :=
  column = COL_SEL_I32_ADD \/ column = COL_SEL_I32_SUB.

(* Compute selector substitution, but prove its validity from one-hot constraints. *)
Local Ltac arithmetic_row index name :=
  match goal with
  | Hchoice : binaryArithmeticSelector ?chosen, Hprime : Range.fieldPrime,
    Hfull : FullRelation.satFull ?w, Hin : In ?chosen selectorColumns,
    Hchosen : value ?w ?chosen = 1 |- _ =>
  let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
  let reduced := eval vm_compute in (selectedRow COL_SEL_I32_ADD raw) in
  let Hsame := fresh "Hsame" in
  assert (Hsame : selectedRow chosen raw = reduced)
    by (destruct Hchoice; subst chosen; vm_compute; reflexivity);
  pose proof (full_selected_row w chosen index raw Hprime Hfull Hin Hchosen eq_refl) as name;
  rewrite Hsame in name; clear Hsame;
  unfold sat in name; cbn [eval fold_right] in name; normalize_columns
  end.

Lemma full_sp_before_u32 : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> 0 <= w COL_SP_BEFORE < 4294967296.
Proof.
  intros w Hprime Hfull.
  apply (Range.full_u32 w COL_SP_BEFORE Hprime Hfull). vm_compute. reflexivity.
Qed.

Record BinaryStackShape (w : Witness) : Prop := {
  stack_read0_active : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1;
  stack_read1_active : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 1;
  stack_read2_inactive : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  stack_write_active : w COL_STACK_WRITE0_ACTIVE = 1;
  stack_write_hi_active : w COL_STACK_WRITE0_HI_ACTIVE = 1;
  stack_capture_inactive : w COL_OUTPUT_CAPTURED = 0;
  stack_read0_address : w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 2);
  stack_read1_address : w (List.nth 1%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1);
  stack_write_address : w COL_STACK_WRITE0_ADDR_LO = 2 * (w COL_SP_BEFORE - 2);
  stack_write_hi_address : w COL_STACK_WRITE0_ADDR_HI = 2 * (w COL_SP_BEFORE - 2) + 1;
  stack_pointer_after : w COL_SP_AFTER = w COL_SP_BEFORE - 1;
  stack_base_after : w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE;
  stack_local_get_inactive : w COL_SEL_LOCAL_GET = 0;
  stack_local_write_inactive : w COL_LOCAL_WRITE_ENABLED = 0;
  stack_gmem_local_inactive : w GMEM_LOCAL = 0;
  stack_gather_local_lo_inactive : w COL_GATHER_LOCAL_WRITE_LO = 0;
  stack_gather_local_inactive : w COL_GATHER_LOCAL_WRITE = 0
}.

Theorem full_binary_stack_shape : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  binaryArithmeticSelector chosen -> w chosen = 1 -> 2 <= w COL_SP_BEFORE ->
  BinaryStackShape w.
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen Hdepth.
  assert (Hin : In chosen selectorColumns).
  { destruct Hchoice; subst chosen; vm_compute; tauto. }
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero Hgather]].
  destruct (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput].
  destruct (full_body_frame_sp_drop_idle w chosen Hprime Hfull Hin Hchosen
    ltac:(destruct Hchoice; subst chosen; vm_compute; intuition discriminate))
    as [HbodyPop HbodyPush Hguest Hturn Hdiscard].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  pose proof (full_sp_before_u32 w Hprime Hfull) as Hsp.
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { arithmetic_row outputCaptureSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { arithmetic_row hostCallSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { arithmetic_row hostCallParamsRow Hrow. pose proof (full_no_host_split w Hfull Hhost) as Hsplit. normalize_columns.
    rewrite Hhost, Hsplit in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = 2).
  { arithmetic_row stackReadsArityRow Hrow.
    rewrite Hone, Hgather, Hparam in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 1).
  { arithmetic_row stackWritesArityRow Hrow.
    rewrite Hone, Hgather, Hparam, Hzero in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0).
  { arithmetic_row stackRead2SupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)). field_linear Hrow. }
  assert (Hread01 : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 /\
    w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 1).
  { arithmetic_row stackReadsActiveRow Hrow.
    rewrite Hone, Hreads, Hread2 in Hrow.
    pose proof (full_boolean_at w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) stackRead0BooleanRow Hprime Hfull eq_refl).
    pose proof (full_boolean_at w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) stackRead1BooleanRow Hprime Hfull eq_refl).
    field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 1).
  { arithmetic_row stackWritesActiveRow Hrow. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 1).
  { arithmetic_row stackWriteHighActiveRow Hrow. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { arithmetic_row callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { arithmetic_row callStackPopSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w chosen COL_SEL_LOCAL_GET Hprime Hfull Hchosen Hin _ _);
      [vm_compute; tauto | destruct Hchoice; subst chosen; discriminate]. }
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { arithmetic_row localWriteGateIndex Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  destruct Hread01 as [Hread0 Hread1]. constructor; try assumption.
  - arithmetic_row binaryRead0AddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - arithmetic_row binaryRead1AddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - arithmetic_row binaryWriteAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_LO). field_linear Hrow.
  - arithmetic_row binaryWriteAddressRow Hlo. arithmetic_row stackWriteHighAddressRow Hrow.
    rewrite Hone in Hlo, Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_LO).
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_HI).
    assert (Haddr : w COL_STACK_WRITE0_ADDR_LO = 2 * (w COL_SP_BEFORE - 2)) by (field_linear Hlo).
    normalize_columns. rewrite Haddr in Hrow. field_linear Hrow.
  - arithmetic_row stackPointerBalanceRow Hrow. zero_host_switch Hrow.
    zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    pose proof (value_bound w COL_SP_AFTER). field_linear Hrow.
  - arithmetic_row stackFrameBasePreservedRow Hrow. zero_host_switch Hrow.
    rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

Definition binaryStackEvents (w : Witness) : list Event :=
  [ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
     (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
   ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
     (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat));
   ReadEvent MemoryCatalog.stackId [w (List.nth 1%nat COL_STACK_READ_ADDR_LO 0%nat)]
     (w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat));
   ReadEvent MemoryCatalog.stackId [w (List.nth 1%nat COL_STACK_READ_ADDR_HI 0%nat)]
     (w (List.nth 1%nat COL_STACK_READ_VALUE_HI 0%nat));
   WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO] (w COL_STACK_WRITE0_VALUE_LO) None;
   WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI] (w COL_STACK_WRITE0_VALUE_HI) None].

Lemma binary_stack_event_prefix : forall w,
  BinaryStackShape w ->
  eventsForRow MemoryCatalog.memories w =
    binaryStackEvents w ++ eventsForRow (tl MemoryCatalog.memories) w.
Proof.
  intros w Hshape.
  destruct Hshape as [Hr0 Hr1 Hr2 Hw Hwh Hcap Ha0 Ha1 Haw Hawh Hsp Hbase Hget Hset Hgmem Hglo Hg].
  change (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with
      Some event => [event] | None => [] end)
    (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
    eventsForRow (tl MemoryCatalog.memories) w =
    binaryStackEvents w ++ eventsForRow (tl MemoryCatalog.memories) w).
  cbn [hd MemoryCatalog.memories ports flat_map portEvent active activation
    addressColumns valueColumn portKind map option_map].
  unfold portEvent, active. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hr0, Hr1, Hr2, Hw, Hwh, Hcap, Hget, Hset, Hgmem, Hglo, Hg. reflexivity.
Qed.

Lemma stack_only_first_memory :
  readOnlyMemory MemoryCatalog.stackId (tl MemoryCatalog.memories) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem binary_stack_replay : forall w initial final,
  BinaryStackShape w -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  initial MemoryCatalog.stackId [w (List.nth 1%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  forall address, final MemoryCatalog.stackId address =
    update (update initial MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO]
      (w COL_STACK_WRITE0_VALUE_LO)) MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI]
      (w COL_STACK_WRITE0_VALUE_HI) MemoryCatalog.stackId address.
Proof.
  intros w initial final Hshape Hreplay.
  rewrite (binary_stack_event_prefix w Hshape) in Hreplay.
  destruct (replay_app_inv (binaryStackEvents w) _ _ _ Hreplay)
    as [middle [Hstack Hrest]].
  unfold binaryStackEvents in Hstack.
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.
  repeat split; try assumption. intro address.
  eapply replay_preserves; [exact Hrest | ].
  apply read_only_row. exact stack_only_first_memory.
Qed.

(* One operand popped (`isSet`) or pushed, and for local.get/local.set the
   local's cell read or written (`local`). *)
Record UnaryStackShape (w : Witness) (isSet local : bool) : Prop := {
  local_read_active : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = (if isSet then 1 else 0);
  local_read1_inactive : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  local_read2_inactive : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0;
  local_write_active : w COL_STACK_WRITE0_ACTIVE = (if isSet then 0 else 1);
  local_write_hi_active : w COL_STACK_WRITE0_HI_ACTIVE = (if isSet then 0 else 1);
  local_capture_inactive : w COL_OUTPUT_CAPTURED = 0;
  local_read_address : isSet = true ->
    w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * (w COL_SP_BEFORE - 1);
  local_write_address : isSet = false -> w COL_STACK_WRITE0_ADDR_LO = 2 * w COL_SP_BEFORE;
  local_write_hi_address : isSet = false -> w COL_STACK_WRITE0_ADDR_HI = 2 * w COL_SP_BEFORE + 1;
  local_stack_pointer_after : w COL_SP_AFTER = w COL_SP_BEFORE + (if isSet then -1 else 1);
  local_stack_base_after : w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE;
  local_get_gate : w COL_SEL_LOCAL_GET = (if local && negb isSet then 1 else 0);
  local_write_gate : w COL_LOCAL_WRITE_ENABLED = (if local && isSet then 1 else 0);
  local_gmem_inactive : w GMEM_LOCAL = 0;
  local_gather_lo_inactive : w COL_GATHER_LOCAL_WRITE_LO = 0;
  local_gather_inactive : w COL_GATHER_LOCAL_WRITE = 0
}.

Ltac unary_stack_facts chosen reads writes localWrite :=
  let Hrow := fresh "Hrow" in
  intros w Hprime Hfull Hchosen Hdepth;
  assert (Hin : In chosen selectorColumns) by
    (vm_compute; tauto);
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull);
  pose proof (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero Hgather]];
  destruct (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput];
  destruct (full_body_frame_sp_drop_idle w chosen Hprime Hfull Hin Hchosen
    ltac:(vm_compute; intuition discriminate)) as [HbodyPop HbodyPush Hguest Hturn Hdiscard];
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo];
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = localWrite) by
    (selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow);
  pose proof (full_sp_before_u32 w Hprime Hfull) as Hsp;
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0) by
    (selected_row chosen outputCaptureSupportRow Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w COL_OUTPUT_CAPTURED); field_linear Hrow);
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0) by
    (selected_row chosen hostCallSupportRow Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w COL_HOST_CALL_ACTIVE); field_linear Hrow);
  assert (Hparams : w GHC_PARAMS = 0) by
    (selected_row chosen hostCallParamsRow Hrow; pose proof (full_no_host_split w Hfull Hhost) as Hsplit; normalize_columns;
     rewrite Hhost, Hsplit in Hrow;
     pose proof (value_bound w GHC_PARAMS); field_linear Hrow);
  assert (Hreads : w COL_STACK_READS = reads) by
    (selected_row chosen stackReadsArityRow Hrow; rewrite Hone, Hgather, Hparam in Hrow;
     pose proof (value_bound w COL_STACK_READS); field_linear Hrow);
  assert (Hwrites : w COL_STACK_WRITES = writes) by
    (selected_row chosen stackWritesArityRow Hrow; rewrite Hone, Hgather, Hparam, Hzero in Hrow;
     pose proof (value_bound w COL_STACK_WRITES); field_linear Hrow);
  assert (Hread2 : w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) = 0) by
    (selected_row chosen stackRead2SupportRow Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)); field_linear Hrow);
  assert (Hread1 : w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) = 0) by
    (selected_row chosen stackRead1SupportRow Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)); field_linear Hrow);
  assert (Hread0 : w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = reads) by
    (selected_row chosen stackReadsActiveRow Hrow; rewrite Hone, Hreads, Hread1, Hread2 in Hrow;
     pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)); field_linear Hrow);
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = writes) by
    (selected_row chosen stackWritesActiveRow Hrow; rewrite Hone, Hwrites in Hrow;
     pose proof (value_bound w COL_STACK_WRITE0_ACTIVE); field_linear Hrow);
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = writes) by
    (selected_row chosen stackWriteHighActiveRow Hrow; rewrite Hone, Hwrite, Hhi in Hrow;
     pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE); field_linear Hrow);
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0) by
    (selected_row chosen callStackPushRow Hrow;
     pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT); field_linear Hrow);
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0) by
    (selected_row chosen callStackPopSupportRow Hrow; rewrite Hone in Hrow;
     pose proof (value_bound w COL_CALL_STACK_POP_PRESENT); field_linear Hrow).

Theorem full_local_set_stack_shape : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_SET = 1 ->
  1 <= w COL_SP_BEFORE -> UnaryStackShape w true true.
Proof.
  unary_stack_facts COL_SEL_LOCAL_SET 1 0 1.
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w COL_SEL_LOCAL_SET COL_SEL_LOCAL_GET Hprime Hfull Hchosen Hin _ _);
      [vm_compute; tauto | discriminate]. }
  constructor; try assumption;
  try (intro H; discriminate H);
  [ intro H; selected_row COL_SEL_LOCAL_SET unaryReadAddressRow Hrow; rewrite Hone in Hrow;
    pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)); field_linear Hrow
  | selected_row COL_SEL_LOCAL_SET stackPointerBalanceRow Hrow; zero_host_switch Hrow; zero_gather_kinds Hkinds Hrow;
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow;
    pose proof (value_bound w COL_SP_AFTER); field_linear Hrow
  | selected_row COL_SEL_LOCAL_SET stackFrameBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hguest, Hturn, Hpop in Hrow;
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE);
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER); field_linear Hrow ].
Qed.

Theorem full_local_get_stack_shape : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_GET = 1 ->
  0 <= w COL_SP_BEFORE -> UnaryStackShape w false true.
Proof.
  unary_stack_facts COL_SEL_LOCAL_GET 0 1 0.
  constructor; try assumption; try (intro H; discriminate H).
  - intro H. selected_row COL_SEL_LOCAL_GET nullaryWriteAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_LO). field_linear Hrow.
  - intro H. selected_row COL_SEL_LOCAL_GET nullaryWriteAddressRow Hlo.
    selected_row COL_SEL_LOCAL_GET stackWriteHighAddressRow Hrow.
    rewrite Hone in Hlo, Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_LO).
    pose proof (value_bound w COL_STACK_WRITE0_ADDR_HI).
    assert (Haddr : w COL_STACK_WRITE0_ADDR_LO = 2 * w COL_SP_BEFORE) by (field_linear Hlo).
    normalize_columns. rewrite Haddr in Hrow. field_linear Hrow.
  - selected_row COL_SEL_LOCAL_GET stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    pose proof (value_bound w COL_SP_AFTER). field_linear Hrow.
  - selected_row COL_SEL_LOCAL_GET stackFrameBasePreservedRow Hrow. zero_host_switch Hrow. rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

(* The operand port events, then the local's cell events, in port order. *)
Definition localStackEvents (w : Witness) (isSet local : bool) : list Event :=
  (if isSet then
    [ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
       (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
     ReadEvent MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
       (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat))]
  else
    [WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO] (w COL_STACK_WRITE0_VALUE_LO) None;
     WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI] (w COL_STACK_WRITE0_VALUE_HI) None]) ++
  (if negb local then [] else if isSet then
    [WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] (w COL_LOCAL_VALUE) None;
     WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_HI] (w COL_LOCAL_VALUE_HI) None]
  else
    [ReadEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] (w COL_LOCAL_VALUE);
     ReadEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_HI] (w COL_LOCAL_VALUE_HI)]).

Lemma local_stack_event_prefix : forall w isSet local,
  UnaryStackShape w isSet local ->
  eventsForRow MemoryCatalog.memories w =
    localStackEvents w isSet local ++ eventsForRow (tl MemoryCatalog.memories) w.
Proof.
  intros w isSet local Hshape.
  destruct Hshape as [Hr0 Hr1 Hr2 Hw Hwh Hcap Ha0 Haw Hawh Hsp Hbase Hget Hset Hgmem Hglo Hg].
  change (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with
      Some event => [event] | None => [] end)
    (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
    eventsForRow (tl MemoryCatalog.memories) w =
    localStackEvents w isSet local ++ eventsForRow (tl MemoryCatalog.memories) w).
  cbn [hd MemoryCatalog.memories ports flat_map portEvent active activation
    addressColumns valueColumn portKind map option_map].
  unfold portEvent, active. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hr0, Hr1, Hr2, Hw, Hwh, Hcap, Hget, Hset, Hgmem, Hglo, Hg.
  destruct isSet, local; reflexivity.
Qed.

Ltac unary_replay_steps :=
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.

Theorem pop_stack_replay : forall w initial final,
  UnaryStackShape w true false -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  forall address, final MemoryCatalog.stackId address = initial MemoryCatalog.stackId address.
Proof.
  intros w initial final Hshape Hreplay.
  rewrite (local_stack_event_prefix w true false Hshape) in Hreplay.
  destruct (replay_app_inv (localStackEvents w true false) _ _ _ Hreplay) as [middle [Hstack Hrest]].
  cbn [localStackEvents negb app] in Hstack. unary_replay_steps.
  split; [assumption | ]. intro address.
  eapply replay_preserves; [exact Hrest | ].
  apply read_only_row. exact stack_only_first_memory.
Qed.

Theorem local_set_stack_replay : forall w initial final,
  UnaryStackShape w true true -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  forall address, final MemoryCatalog.stackId address =
    update (update initial MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] (w COL_LOCAL_VALUE))
      MemoryCatalog.stackId [w COL_LOCAL_ADDR_HI] (w COL_LOCAL_VALUE_HI) MemoryCatalog.stackId address.
Proof.
  intros w initial final Hshape Hreplay.
  rewrite (local_stack_event_prefix w true true Hshape) in Hreplay.
  destruct (replay_app_inv (localStackEvents w true true) _ _ _ Hreplay) as [middle [Hstack Hrest]].
  cbn [localStackEvents negb app] in Hstack. unary_replay_steps.
  split; [assumption | ]. intro address.
  eapply replay_preserves; [exact Hrest | ].
  apply read_only_row. exact stack_only_first_memory.
Qed.

Theorem local_get_stack_replay : forall w initial final,
  UnaryStackShape w false true -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  let pushed := update (update initial MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO]
      (w COL_STACK_WRITE0_VALUE_LO)) MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI]
      (w COL_STACK_WRITE0_VALUE_HI) in
  pushed MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] = Some (w COL_LOCAL_VALUE) /\
  forall address, final MemoryCatalog.stackId address = pushed MemoryCatalog.stackId address.
Proof.
  intros w initial final Hshape Hreplay pushed.
  rewrite (local_stack_event_prefix w false true Hshape) in Hreplay.
  destruct (replay_app_inv (localStackEvents w false true) _ _ _ Hreplay)
    as [middle [Hstack Hrest]].
  cbn [localStackEvents negb app] in Hstack.
  repeat match goal with
  | H : Replay _ (_ :: _) _ |- _ => inversion H; subst; clear H
  | H : EventStep _ _ _ |- _ => inversion H; subst; clear H
  | H : Replay _ [] _ |- _ => inversion H; subst; clear H
  end.
  split; [assumption | ]. intro address.
  eapply replay_preserves; [exact Hrest | ].
  apply read_only_row. exact stack_only_first_memory.
Qed.
