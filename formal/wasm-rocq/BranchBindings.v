(* br_if's condition, stack pop, unchanged locals, and selected PC-ROM edge. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts ZeroTest CallInitialization
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

(* A br_if pops its condition and slides by its distance. *)
Theorem full_br_if_stack_shape : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_IF_EQZ = 1 ->
  1 <= w COL_SP_BEFORE -> UnaryStackShape w true false.
Proof.
  unary_stack_facts COL_SEL_BR_IF_EQZ 1 0 0.
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w COL_SEL_BR_IF_EQZ COL_SEL_LOCAL_GET Hprime Hfull Hchosen Hin _ _);
      [vm_compute; tauto | discriminate]. }
  constructor; try assumption; try (intro H; discriminate H).
  - intro H. selected_row COL_SEL_BR_IF_EQZ unaryReadAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - selected_row COL_SEL_BR_IF_EQZ stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    change (w COL_SP_AFTER = w COL_SP_BEFORE + -1 - w COL_FRAME_MOVE_DISTANCE_AFTER).
    circuit_range COL_SP_AFTER Hspa. circuit_range COL_SP_BEFORE Hspb. circuit_range COL_FRAME_MOVE_DISTANCE_AFTER Hd.
    field_linear Hrow.
  - selected_row COL_SEL_BR_IF_EQZ stackFrameBasePreservedRow Hrow. zero_host_switch Hrow. rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

Lemma full_branch_fbp_unchanged : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> In chosen [COL_SEL_BR; COL_SEL_BR_IF_EQZ; COL_SEL_BR_TABLE] -> w chosen = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE.
Proof.
  intros w chosen Hprime Hfull Hbranch Hchosen.
  assert (Hin : In chosen selectorColumns) by (destruct Hbranch as [<- | [<- | [<- | []]]]; vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  destruct (full_branch_frame w chosen Hprime Hfull Hin Hchosen Hbranch) as [Hpop Hpush _ Hturn _].
  pose proof (value_bound w COL_LOCALS_FBP_BEFORE). pose proof (value_bound w COL_LOCALS_FBP_AFTER).
  destruct Hbranch as [<- | [<- | [<- | []]]];
    [selected_row COL_SEL_BR localsBasePreservedRow Hrow | selected_row COL_SEL_BR_IF_EQZ localsBasePreservedRow Hrow
    | selected_row COL_SEL_BR_TABLE localsBasePreservedRow Hrow];
    zero_host_switch Hrow; rewrite Hone, Hpush, Hpop, Hturn in Hrow; field_linear Hrow.
Qed.

(* A br pops nothing and slides by its distance. *)
Theorem full_br_stack : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR = 1 -> 0 <= w COL_SP_BEFORE ->
  w COL_SP_AFTER = w COL_SP_BEFORE - w COL_FRAME_MOVE_DISTANCE_AFTER /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  filter isStackWrite (eventsForRow memories w) = [].
Proof.
  unary_stack_facts COL_SEL_BR 0 0 0.
  split; [ | split].
  - selected_row COL_SEL_BR stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    circuit_range COL_SP_AFTER Hspa. circuit_range COL_SP_BEFORE Hspb. circuit_range COL_FRAME_MOVE_DISTANCE_AFTER Hd.
    field_linear Hrow.
  - selected_row COL_SEL_BR stackFrameBasePreservedRow Hrow. zero_host_switch Hrow.
    rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
  - rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite Hwrite, HwriteHi, HlocalWrite, HgatherLocal, HgatherLocalLo. reflexivity.
Qed.

(* A br_table pops its index and slides by its distance. *)
Theorem full_br_table_stack_shape : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_TABLE = 1 ->
  1 <= w COL_SP_BEFORE -> UnaryStackShape w true false.
Proof.
  unary_stack_facts COL_SEL_BR_TABLE 1 0 0.
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w COL_SEL_BR_TABLE COL_SEL_LOCAL_GET Hprime Hfull Hchosen Hin _ _);
      [vm_compute; tauto | discriminate]. }
  constructor; try assumption; try (intro H; discriminate H).
  - intro H. selected_row COL_SEL_BR_TABLE unaryReadAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - selected_row COL_SEL_BR_TABLE stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    change (w COL_SP_AFTER = w COL_SP_BEFORE + -1 - w COL_FRAME_MOVE_DISTANCE_AFTER).
    circuit_range COL_SP_AFTER Hspa. circuit_range COL_SP_BEFORE Hspb. circuit_range COL_FRAME_MOVE_DISTANCE_AFTER Hd.
    field_linear Hrow.
  - selected_row COL_SEL_BR_TABLE stackFrameBasePreservedRow Hrow. zero_host_switch Hrow.
    rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

(* A br_table's edge: arm `index` (choice index + 1) for an index below the arm
   count, the default (choice 0) otherwise. *)
Theorem full_br_table_choice : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_BR_TABLE = 1 ->
  w COL_CONTROL_CHOICE =
    (if Z.ltb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) (w COL_BR_TABLE_LEN)
     then w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) + 1 else 0).
Proof.
  intros w Hprime Hfull Hchosen.
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_read0_u32 w Hprime Hfull) as Hindex.
  circuit_range COL_BR_TABLE_LEN Hlen. circuit_range COL_CMP_LOW Hlow. circuit_boolean COL_CMP_GE Hge.
  circuit_boolean COL_BR_TABLE_IN_BOUNDS Hbounds.
  pose proof (value_bound w COL_CONTROL_CHOICE).
  (* index - len + 2^32 = low + ge * 2^32, so ge says index >= len. *)
  circuit_row ([(COL_SEL_BR_TABLE, 1)], [(nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, 1); (COL_BR_TABLE_LEN, -1);
    (COL_ONE, 4294967296); (COL_CMP_LOW, -1); (COL_CMP_GE, -4294967296)], ([] : SparseRow)) Hcmp.
  circuit_row ([(COL_SEL_BR_TABLE, 1)], [(COL_ONE, 1); (COL_CMP_GE, -1)], [(COL_BR_TABLE_IN_BOUNDS, 1)]) Hin.
  circuit_row ([(COL_BR_TABLE_IN_BOUNDS, 1)],
    [(COL_CONTROL_CHOICE, 1); (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1); (COL_ONE, -1)], ([] : SparseRow)) Harm.
  circuit_row ([(COL_SEL_BR_TABLE, 1); (COL_BR_TABLE_IN_BOUNDS, -1)], [(COL_CONTROL_CHOICE, 1)], ([] : SparseRow))
    Hdefault.
  expand_row Hcmp. expand_row Hin. expand_row Harm. expand_row Hdefault.
  rewrite Hone, Hchosen in *.
  destruct Hge as [Hge | Hge]; rewrite Hge in *.
  - assert (Hinside : w COL_BR_TABLE_IN_BOUNDS = 1) by field_linear Hin.
    normalize_columns. rewrite Hinside in Harm.
    assert (Hlt : w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) < w COL_BR_TABLE_LEN) by field_linear Hcmp.
    replace (Z.ltb _ _) with true by (symmetry; apply Z.ltb_lt; exact Hlt). field_linear Harm.
  - assert (Houtside : w COL_BR_TABLE_IN_BOUNDS = 0) by field_linear Hin.
    normalize_columns. rewrite Houtside in Hdefault.
    assert (Hge' : w COL_BR_TABLE_LEN <= w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) by field_linear Hcmp.
    replace (Z.ltb _ _) with false by (symmetry; apply Z.ltb_ge; exact Hge'). field_linear Hdefault.
Qed.

(* An if's control choice: its true edge (choice 1) on a nonzero condition, its
   false edge (choice 0) on zero. *)
Theorem full_if_choice : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_IF = 1 ->
  w COL_CONTROL_CHOICE =
    (if Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then 0 else 1).
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_IF selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  selected_circuit_row COL_SEL_IF
    ([(COL_SEL_BR_IF_EQZ, 1); (COL_SEL_IF, 1)],
     [(COL_ZERO_TEST_LO_INPUT, 1); (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hpin.
  pose proof (value_bound w COL_ZERO_TEST_LO_INPUT).
  pose proof (value_bound w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)).
  assert (Hinput : w COL_ZERO_TEST_LO_INPUT = w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) by (field_linear Hpin).
  circuit_zero_test COL_ZERO_TEST_LO_INPUT COL_ZERO_TEST_LO_INV COL_ZERO_TEST_LO_IS_ZERO Hzero.
  rewrite Hinput in Hzero.
  selected_circuit_row COL_SEL_IF
    ([(COL_SEL_BR_IF_EQZ, 1); (COL_SEL_IF, 1)],
     [(COL_CONTROL_CHOICE, 1); (COL_ZERO_TEST_LO_IS_ZERO, 1); (COL_ONE, -1)], ([] : SparseRow)) Hchoice.
  rewrite Hone, Hzero in Hchoice. pose proof (value_bound w COL_CONTROL_CHOICE).
  normalize_columns.
  let readColumn := eval compute in (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) in
    destruct (Z.eqb (w readColumn) 0); field_linear Hchoice.
Qed.

(* An if pops its condition and slides nothing. *)
Theorem full_if_stack_shape : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_IF = 1 ->
  1 <= w COL_SP_BEFORE -> UnaryStackShape w true false /\ w COL_FRAME_MOVE_DISTANCE_AFTER = 0.
Proof.
  unary_stack_facts COL_SEL_IF 1 0 0.
  assert (HlocalGet : w COL_SEL_LOCAL_GET = 0).
  { refine (full_opcode_selector_exclusive w COL_SEL_IF COL_SEL_LOCAL_GET Hprime Hfull Hchosen Hin _ _);
      [vm_compute; tauto | discriminate]. }
  (* Outside slides and moves, the move distance is 0. *)
  assert (Hdistance : w COL_FRAME_MOVE_DISTANCE_AFTER = 0).
  { pose proof (full_selected_is_program_row w COL_SEL_IF Hprime Hfull Hin Hchosen) as Hprogram.
    pose proof (full_selected_no_slide w COL_SEL_IF Hprime Hfull Hin Hchosen
      ltac:(vm_compute; intuition discriminate) Hpop) as Hslide.
    assert (Hmove : w COL_FRAME_MOVE_ACTIVE_BEFORE = 0)
      by exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_FRAME_MOVE_ACTIVE_BEFORE Hprime Hfull
        ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram).
    assert (Hpadding : w COL_PADDING_ACTIVE = 0)
      by exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_PADDING_ACTIVE Hprime Hfull
        ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram).
    circuit_row ([(COL_ONE, 1); (COL_FRAME_SLIDE_ACTIVE, -1); (COL_GUEST_ENTRY_ACTIVE, -1);
      (COL_CALL_STACK_PUSH_PRESENT, 1); (COL_FRAME_MOVE_ACTIVE_BEFORE, -1); (COL_PADDING_ACTIVE, -1)],
      [(COL_FRAME_MOVE_DISTANCE_AFTER, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hone, Hslide, Hguest, Hpush, Hmove, Hpadding in Hrow.
    pose proof (value_bound w COL_FRAME_MOVE_DISTANCE_AFTER). field_linear Hrow. }
  split; [ | exact Hdistance].
  constructor; try assumption; try (intro H; discriminate H).
  - intro H. selected_row COL_SEL_IF unaryReadAddressRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w (nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
  - selected_row COL_SEL_IF stackPointerBalanceRow Hrow. zero_host_switch Hrow. zero_gather_kinds Hkinds Hrow.
    rewrite Hone, Hreads, Hwrites, Hdiscard, Hlocal, Houtput, Hcapture, Hparams in Hrow.
    change (w COL_SP_AFTER = w COL_SP_BEFORE + -1 - w COL_FRAME_MOVE_DISTANCE_AFTER). normalize_columns. rewrite Hdistance.
    circuit_range COL_SP_AFTER Hspa. circuit_range COL_SP_BEFORE Hspb. field_linear Hrow.
  - selected_row COL_SEL_IF stackFrameBasePreservedRow Hrow. zero_host_switch Hrow.
    rewrite Hone, Hguest, Hturn, Hpop in Hrow.
    pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE).
    pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER). field_linear Hrow.
Qed.

Lemma full_if_fbp_unchanged : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_IF = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE.
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_IF selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  destruct (full_body_frame_sp_drop_idle w COL_SEL_IF Hprime Hfull Hin Hchosen
    ltac:(vm_compute; intuition discriminate)) as [Hpop Hpush _ Hturn _].
  selected_row COL_SEL_IF localsBasePreservedRow Hrow. zero_host_switch Hrow.
  rewrite Hone, Hpush, Hpop, Hturn in Hrow.
  pose proof (value_bound w COL_LOCALS_FBP_BEFORE).
  pose proof (value_bound w COL_LOCALS_FBP_AFTER). field_linear Hrow.
Qed.

Definition branchImage (memory : MemoryState) (pc fallthrough target : Z) : Prop :=
  memory programOpcodeId [pc] = Some CODE_BR_IF /\
  memory pcEdgeKindId [pc] = Some 0 /\
  forall choice, memory pcRomId [pc; choice] =
    if choice =? 0 then Some fallthrough else if choice =? 1 then Some target else Some 0.

(* An if's row: the opcode selects it, and the choice it reads (1 for a nonzero
   condition) picks its next pc. *)
Lemma circuit_if_control : forall initial final (w : Witness) pc thenPC elsePC,
  fieldPrime -> FullRelation.satFull w -> Replay initial (eventsForRow memories w) final ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = pc ->
  initial programOpcodeId [pc] = Some CODE_IF -> initial pcEdgeKindId [pc] = Some 0 ->
  (forall choice, initial pcRomId [pc; choice] =
    if choice =? 1 then Some thenPC else if choice =? 0 then Some elsePC else Some 0) ->
  w COL_SEL_IF = 1 /\ w COL_PC_AFTER =
    if Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then elsePC else thenPC.
Proof.
  intros initial final w pc thenPC elsePC Hprime Hfull Hreplay Hprogram Hpc Hop Hkind Hedges.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hopcode.
  assert (Hchosen : w COL_SEL_IF = 1).
  { apply (full_opcode_selects w COL_SEL_IF CODE_IF Hprime Hfull Hprogram); [vm_compute; reflexivity | congruence]. }
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindFetch.
  rewrite Hpc, Hkind in HkindFetch. injection HkindFetch as HkindValue.
  pose proof (full_static_pc_rom_active w Hprime Hfull Hprogram ltac:(congruence)) as Hgate.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hgate) as Hnext.
  rewrite Hpc, Hedges, (full_if_choice w Hprime Hfull Hchosen) in Hnext.
  split; [exact Hchosen | ]. destruct (Z.eqb (w (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0);
    cbn in Hnext; congruence.
Qed.

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
