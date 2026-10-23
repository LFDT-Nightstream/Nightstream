(* Direct guest-call entry and return-context register constraints. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts CallInitialization MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

Lemma guest_metadata_decodes : forall (w : Witness) params results,
  fieldPrime -> FullRelation.satFull w ->
  0 <= params < 256 -> 0 <= results < 256 ->
  w COL_CALL_TARGET_METADATA = params + 256 * results + 65536 ->
  w COL_CALL_PARAM_COUNT = params /\ w COL_CALL_RESULT_COUNT = results /\
  w COL_TARGET_FUNCTION_IS_GUEST = 1.
Proof.
  intros w Hparams Hresults Hprime Hfull Hp Hr Hmetadata.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_CALL_PARAM_COUNT Hpb.
  circuit_range COL_CALL_RESULT_COUNT Hrb.
  circuit_boolean COL_TARGET_FUNCTION_IS_GUEST Hguest.
  circuit_row ([(COL_CALL_TARGET_METADATA, 1); (COL_CALL_PARAM_COUNT, -1);
    (COL_CALL_RESULT_COUNT, -256); (COL_TARGET_FUNCTION_IS_GUEST, -65536)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone, Hmetadata in Hrow.
  destruct Hguest; normalize_columns; repeat split; try assumption; field_linear Hrow.
Qed.

Lemma full_direct_guest_call : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_GUEST_ENTRY_ACTIVE = 1 /\ w COL_CALL_STACK_PUSH_PRESENT = 1 /\
  w COL_CALL_STACK_POP_PRESENT = 0.
Proof.
  intros w Hprime Hfull Hchosen Htarget.
  assert (Hin : In COL_SEL_CALL selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  selected_row COL_SEL_CALL guestEntryFlagRow Hguest.
  selected_row COL_SEL_CALL callStackPushRow Hpush.
  selected_row COL_SEL_CALL callStackPopSupportRow Hpop.
  normalize_columns. rewrite Hone, Htarget in Hguest, Hpush. rewrite Hone in Hpop.
  pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE).
  pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT).
  pose proof (value_bound w COL_CALL_STACK_POP_PRESENT).
  repeat split; [field_linear Hguest | field_linear Hpush | field_linear Hpop].
Qed.

Lemma full_call_context_push : forall (w : Witness),
  FullRelation.satFull w -> w COL_CALL_STACK_PUSH_PRESENT = 1 ->
  w COL_CALL_STACK_ADDR = w COL_CALL_STACK_DEPTH_BEFORE /\
  w COL_CALL_STACK_CALLER_FBP_VALUE = w COL_LOCALS_FBP_BEFORE /\
  w COL_CALL_STACK_CALLER_SP_BASE_VALUE = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_PC_ROM_CALL_RETURN_CHOICE = 1.
Proof.
  intros w Hfull Hpush.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  repeat split;
    match goal with
    | |- value _ ?a = value _ ?b =>
      circuit_row ([(COL_CALL_STACK_PUSH_PRESENT, 1)], [(a, 1); (b, -1)], ([] : SparseRow)) Hrow;
      expand_row Hrow; normalize_columns; rewrite Hpush in Hrow;
      pose proof (value_bound w a); pose proof (value_bound w b); field_linear Hrow
    | |- value _ ?a = 1 =>
      circuit_row ([(COL_CALL_STACK_PUSH_PRESENT, 1)], [(a, 1); (COL_ONE, -1)], ([] : SparseRow)) Hrow;
      expand_row Hrow; normalize_columns; rewrite Hpush, Hone in Hrow;
      pose proof (value_bound w a); field_linear Hrow
    end.
Qed.

(* A called frame starts at its arguments; its operands start above its locals. *)
Lemma full_call_frame_entry : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_GUEST_ENTRY_ACTIVE = 1 ->
  w COL_CALL_STACK_PUSH_PRESENT = 1 -> w COL_CALL_STACK_POP_PRESENT = 0 -> w COL_IS_PROGRAM_ROW = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_SP_AFTER - w COL_CALL_PARAM_COUNT /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_LOCALS_FBP_AFTER + w COL_ENTERED_FUNCTION_NUM_LOCALS /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE + 1.
Proof.
  intros w Hprime Hfull Hguest Hpush Hpop Hprogram.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [Hsuspend [Hresume _]].
  pose proof (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY
    Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram) as Hturn.
  circuit_range COL_LOCALS_FBP_AFTER Hfa.
  circuit_range COL_ENTERED_FUNCTION_NUM_LOCALS Hlocals.
  circuit_range COL_STACK_FRAME_BASE_AFTER Hbase.
  circuit_range COL_SP_AFTER Hsp.
  circuit_range COL_CALL_PARAM_COUNT Hparams.
  circuit_range COL_CALL_STACK_DEPTH_BEFORE Hdb.
  circuit_range COL_CALL_STACK_DEPTH_AFTER Hda.
  circuit_row ([(COL_CALL_STACK_PUSH_PRESENT, 1)],
    [(COL_LOCALS_FBP_AFTER, 1); (COL_SP_AFTER, -1); (COL_CALL_PARAM_COUNT, 1)],
    ([] : SparseRow)) Hfbp.
  circuit_row ([(COL_GUEST_ENTRY_ACTIVE, 1); (COL_TURN_BOUNDARY, 1)],
    [(COL_STACK_FRAME_BASE_AFTER, 1); (COL_LOCALS_FBP_AFTER, -1); (COL_ENTERED_FUNCTION_NUM_LOCALS, -1)],
    ([] : SparseRow)) Hstack.
  circuit_row ([(COL_CALL_STACK_DEPTH_AFTER, 1); (COL_CALL_STACK_DEPTH_BEFORE, -1);
    (COL_CALL_STACK_PUSH_PRESENT, -1); (COL_CALL_STACK_POP_PRESENT, 1);
    (COL_HOST_SUSPEND, -1); (COL_HOST_RESUME, 1)], [(COL_ONE, 1)], ([] : SparseRow)) Hdepth.
  expand_row Hfbp. expand_row Hstack. expand_row Hdepth.
  normalize_columns. rewrite Hpush in Hfbp. rewrite Hguest, Hturn in Hstack.
  rewrite Hpush, Hpop, Hone, Hsuspend, Hresume in Hdepth.
  repeat split; [field_linear Hfbp | field_linear Hstack | field_linear Hdepth].
Qed.

Lemma full_return_context_restore : forall (w : Witness),
  FullRelation.satFull w -> w COL_CALL_STACK_POP_PRESENT = 1 ->
  w COL_PC_AFTER = w COL_CALL_STACK_RETURN_PC_VALUE /\
  w COL_LOCALS_FBP_AFTER = w COL_CALL_STACK_CALLER_FBP_VALUE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_CALL_STACK_CALLER_SP_BASE_VALUE /\
  w COL_CALL_STACK_ADDR = w COL_CALL_STACK_DEPTH_AFTER.
Proof.
  intros w Hfull Hpop. repeat split;
    match goal with |- value _ ?a = value _ ?b =>
      circuit_row ([(COL_CALL_STACK_POP_PRESENT, 1)], [(a, 1); (b, -1)], ([] : SparseRow)) Hrow;
      expand_row Hrow; normalize_columns; rewrite Hpop in Hrow;
      pose proof (value_bound w a); pose proof (value_bound w b); field_linear Hrow
    end.
Qed.

Lemma full_direct_guest_call_stack : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_CALL = 1 ->
  w COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  w COL_STACK_WRITE0_ACTIVE = 0 /\ w COL_STACK_WRITE0_HI_ACTIVE = 0 /\
  w COL_SP_AFTER = w COL_SP_BEFORE.
Proof.
  intros w Hprime Hfull Hchosen Htarget.
  assert (Hin : In COL_SEL_CALL selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w COL_SEL_CALL Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero Hgather]].
  destruct (full_program_row_quiet w COL_SEL_CALL Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput].
  assert (Hdiscard : w COL_FRAME_SP_DROP = 0).
  { destruct (full_direct_guest_call w Hprime Hfull Hchosen Htarget) as [Hguest [Hpush Hpop]].
    pose proof (full_selected_is_program_row w COL_SEL_CALL Hprime Hfull Hin Hchosen) as Hprogram.
    apply full_frame_sp_drop_idle; try assumption; [congruence | | ].
    - exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY
        Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram).
    - selected_circuit_row COL_SEL_CALL
        ([(COL_HOST_EVENT_EXIT_LATCH, 1)], [(COL_ONE, 1); (COL_SEL_END, -1); (COL_SEL_RETURN, -1)],
        ([] : SparseRow)) Hrow.
      rewrite Hone in Hrow. pose proof (value_bound w COL_HOST_EVENT_EXIT_LATCH). field_linear Hrow. }
  assert (Hcapture : w COL_OUTPUT_CAPTURED = 0).
  { selected_row COL_SEL_CALL outputCaptureSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow. }
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { selected_circuit_row COL_SEL_CALL
      ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_ONE, 1); (COL_CALL_INDIRECT_IS_TRAP, -1)],
        [(COL_CALL_INDIRECT_IS_NOT_TRAP, 1)]) Hci.
    assert (HciZero : w COL_CALL_INDIRECT_IS_NOT_TRAP = 0).
    { pose proof (value_bound w COL_CALL_INDIRECT_IS_NOT_TRAP). field_linear Hci. }
    selected_circuit_row COL_SEL_CALL
      ([(COL_HOST_CALL_ACTIVE, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1);
        (COL_CALL_INDIRECT_IS_NOT_TRAP, -1); (COL_GUEST_ENTRY_ACTIVE, 1)],
        [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    destruct (full_direct_guest_call w Hprime Hfull Hchosen Htarget) as [Hguest _].
    normalize_columns. rewrite Hone, Hguest, HciZero in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. pose proof (full_no_host_split w Hfull Hhost) as Hsplit. normalize_columns.
    rewrite Hhost, Hsplit in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = 0).
  { selected_circuit_row COL_SEL_CALL
      ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
        (COL_SEL_RETURN_CALL_INDIRECT, 1)],
        [(COL_STACK_READS, 1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1)],
        ([] : SparseRow)) Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { selected_circuit_row COL_SEL_CALL
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
  split; [exact Hwrite | ]. split; [exact HwriteHi | ].
  full_row stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
  rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
  pose proof (value_bound w COL_SP_BEFORE). pose proof (value_bound w COL_SP_AFTER).
  field_linear Hrow.
Qed.
