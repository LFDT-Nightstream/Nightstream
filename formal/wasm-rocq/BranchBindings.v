(* br_if's condition, stack pop, unchanged locals, and selected PC-ROM edge. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts ZeroTest
  StackBindings LocalBindings MemoryTrace MemoryCatalog ProgramBinding OpcodeCatalog.
Import ListNotations.
Open Scope Z_scope.

Theorem full_br_if_choice : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_IF_EQZ = 1 ->
  w COL_CONTROL_CHOICE =
    (if Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then 0 else 1).
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_BR_IF_EQZ selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  selected_circuit_row COL_SEL_BR_IF_EQZ
    ([(COL_SEL_BR_IF_EQZ, 1); (COL_SEL_IF, 1)],
     [(COL_ZERO_TEST_LO_INPUT, 1); (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hpin.
  pose proof (value_bound w COL_ZERO_TEST_LO_INPUT).
  pose proof (value_bound w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)).
  assert (Hinput : w COL_ZERO_TEST_LO_INPUT = w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) by (field_linear Hpin).
  circuit_zero_test COL_ZERO_TEST_LO_INPUT COL_ZERO_TEST_LO_INV COL_ZERO_TEST_LO_IS_ZERO Hzero.
  rewrite Hinput in Hzero.
  selected_circuit_row COL_SEL_BR_IF_EQZ
    ([(COL_SEL_BR_IF_EQZ, 1); (COL_SEL_IF, 1)],
     [(COL_CONTROL_CHOICE, 1); (COL_ZERO_TEST_LO_IS_ZERO, 1); (COL_ONE, -1)], ([] : SparseRow)) Hchoice.
  rewrite Hone, Hzero in Hchoice. pose proof (value_bound w COL_CONTROL_CHOICE).
  normalize_columns.
  let readColumn := eval compute in (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) in
    destruct (Z.eqb (w readColumn) 0); field_linear Hchoice.
Qed.

Theorem full_br_if_stack_shape : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_IF_EQZ = 1 ->
  1 <= w COL_SP_BEFORE -> UnaryStackShape w true.
Proof.
  unary_stack_facts COL_SEL_BR_IF_EQZ 1 0.
  constructor; try assumption; try (intro H; discriminate H).
  - intro H. selected_row COL_SEL_BR_IF_EQZ unaryReadAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - selected_row COL_SEL_BR_IF_EQZ stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    pose proof (value_bound w COL_SP_AFTER). field_linear Hrow.
  - selected_row COL_SEL_BR_IF_EQZ stackFrameBasePreservedRow Hrow. zero_host_switch Hrow. rewrite Hone, Hpush, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

Lemma full_br_if_locals_unchanged : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_IF_EQZ = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  filter isLocalWrite (eventsForRow memories w) = [].
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_BR_IF_EQZ selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  destruct (full_opcode_aux_gates_zero w _ Hprime Hfull Hin Hchosen) as [Hparam [Hzero _]].
  pose proof (full_no_gather_local_write w _ Hprime Hfull Hin Hchosen) as Hlo.
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { selected_row COL_SEL_BR_IF_EQZ guestEntryFlagRow Hrow.
    pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { selected_row COL_SEL_BR_IF_EQZ callStackPopSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  split.
  - selected_row COL_SEL_BR_IF_EQZ localsBasePreservedRow Hrow. zero_host_switch Hrow. rewrite Hone, Hguest, Hpop in Hrow.
    pose proof (value_bound w COL_LOCALS_FBP_BEFORE).
    pose proof (value_bound w COL_LOCALS_FBP_AFTER). field_linear Hrow.
  - rewrite (local_writes_in_row w Hparam Hzero Hlo).
    assert (Hwrite : w COL_LOCAL_WRITE_ENABLED = 0).
    { selected_row COL_SEL_BR_IF_EQZ localWriteGateIndex Hrow. rewrite Hone in Hrow.
      pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
    rewrite Hwrite. reflexivity.
Qed.

Definition branchImage (memory : MemoryState) (pc fallthrough target : Z) : Prop :=
  memory programOpcodeId [pc] = Some CODE_BR_IF /\
  memory pcEdgeKindId [pc] = Some 0 /\
  forall choice, memory pcRomId [pc; choice] =
    if choice =? 0 then Some fallthrough else if choice =? 1 then Some target else None.

Lemma circuit_br_if_control : forall initial final (w : Witness) pc fallthrough target,
  fieldPrime -> FullRelation.satFull w -> Replay initial (eventsForRow memories w) final ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = pc -> branchImage initial pc fallthrough target ->
  w COL_SEL_BR_IF_EQZ = 1 /\ w COL_PC_AFTER =
    if Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then fallthrough else target.
Proof.
  intros initial final w pc fallthrough target Hprime Hfull Hreplay Hprogram Hpc [Hop [Hkind Hedges]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hopcode.
  assert (Hchosen : w COL_SEL_BR_IF_EQZ = 1).
  { apply (full_opcode_selects w COL_SEL_BR_IF_EQZ CODE_BR_IF Hprime Hfull Hprogram); [vm_compute; reflexivity | congruence]. }
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindFetch.
  rewrite Hpc, Hkind in HkindFetch. injection HkindFetch as HkindValue.
  pose proof (full_static_pc_rom_active w Hprime Hfull Hprogram ltac:(congruence)) as Hgate.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hgate) as Hnext.
  rewrite Hpc, Hedges, (full_br_if_choice w Hprime Hfull Hchosen) in Hnext.
  split; [exact Hchosen | ]. destruct (Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0);
    cbn in Hnext; congruence.
Qed.
