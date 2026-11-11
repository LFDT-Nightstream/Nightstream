From Stdlib Require Import List ZArith Lia.
Require Import StackPorts R1CS Columns FullRelation Range Selectors CircuitFacts MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

(* SPEC.md section 7.6: the planner's exported gate exclusions hold under the
   existing relation. Every row is checked by shape; no row position is pinned.
   To add a pair, follow an exclusion lemma below: assume both gates are 1,
   locate their defining equations and activation-support rows with circuit_row,
   and establish Boolean/range bounds with gate_bound or circuit_range. Use
   those bounds to lift the modular equations to integer equalities, then derive
   a contradiction. Add the lemma to gate_exclusions_sound so it covers the
   regenerated list. No additional circuit constraint should be needed. *)
Ltac gate_bound column name :=
  let Hbit := fresh "Hbit" in
  circuit_boolean column Hbit;
  match goal with Hfull : FullRelation.satFull ?w |- _ =>
    assert (name : 0 <= w column <= 1) by lia; clear Hbit
  end.

Lemma pc_exit_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_PC_ROM_ACTIVE = 1 -> w COL_HOST_EVENT_EXIT_LATCH <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_PC_EDGE_KIND B11.
  gate_bound COL_ACTIVATION_FINISHED_AFTER B18.
  gate_bound COL_IS_PROGRAM_ROW B19.
  gate_bound COL_PC_ROM_ACTIVE B20.
  gate_bound COL_PC_EDGE_KIND_IS_STATIC B23.
  gate_bound COL_CALL_STACK_POP_PRESENT B43.
  gate_bound COL_SEL_UNREACHABLE B91.
  gate_bound COL_SEL_CALL_INDIRECT B128.
  gate_bound COL_SEL_RETURN_CALL_INDIRECT B323.
  gate_bound COL_TRAPPED_BEFORE B325.
  gate_bound COL_TRAPPED_AFTER B326.
  gate_bound COL_DIV_TRAP B329.
  gate_bound COL_CALL_INDIRECT_IS_TRAP B340.
  gate_bound COL_TABLE_OOB B346.
  gate_bound COL_MEM_OOB B349.
  gate_bound COL_ACTIVATION_FINISHED_BEFORE B353.
  gate_bound COL_HOST_CLEAR_ACTIVATION_FINISHED B386.
  gate_bound COL_HOST_EVENT_EXIT_LATCH B485.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1); (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_13.
  expand_row modular_13.
  assert (equation_13 : ((1 * w COL_ACTIVATION_FINISHED_AFTER) + (-1 * w COL_ACTIVATION_FINISHED_BEFORE) + (-1 * w COL_TRAPPED_AFTER) + (1 * w COL_TRAPPED_BEFORE) + (1 * w COL_HOST_CLEAR_ACTIVATION_FINISHED) + (-1 * w COL_HOST_EVENT_EXIT_LATCH)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_13 Hone B18 B325 B326 B353 B386 B485. apply Z.mod_divide in modular_13; [destruct modular_13 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_13.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2); (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1); (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_19.
  expand_row modular_19.
  assert (equation_19 : ((1 * w COL_ACTIVATION_FINISHED_AFTER) + (-1 * w COL_ACTIVATION_FINISHED_BEFORE) + (1 * w COL_HOST_CLEAR_ACTIVATION_FINISHED) + (1 * w COL_CALL_STACK_POP_PRESENT) + (-1 * w COL_PC_EDGE_KIND) + (2 * w COL_SEL_CALL_INDIRECT) + (2 * w COL_SEL_RETURN_CALL_INDIRECT) + (2 * w COL_SEL_UNREACHABLE) + (-1 * w COL_DIV_TRAP) + (-1 * w COL_CALL_INDIRECT_IS_TRAP) + (-1 * w COL_MEM_OOB) + (-1 * w COL_TABLE_OOB)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_19 Hone B11 B18 B43 B91 B128 B323 B329 B340 B346 B349 B353 B386. apply Z.mod_divide in modular_19; [destruct modular_19 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_19.
  circuit_row ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1); (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_20.
  expand_row modular_20.
  assert (equation_20 : ((1 * w COL_TRAPPED_AFTER) + (-1 * w COL_TRAPPED_BEFORE) + (-1 * w COL_SEL_UNREACHABLE) + (-1 * w COL_DIV_TRAP) + (-1 * w COL_CALL_INDIRECT_IS_TRAP) + (-1 * w COL_MEM_OOB) + (-1 * w COL_TABLE_OOB)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_20 Hone B91 B325 B326 B329 B340 B346 B349. apply Z.mod_divide in modular_20; [destruct modular_20 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_20.
  circuit_row ([(COL_PC_EDGE_KIND, 1)], [(COL_PC_EDGE_KIND_IS_STATIC, 1)], ([] : SparseRow)) modular_21.
  expand_row modular_21.
  assert (equation_21 : ((1 * w COL_PC_EDGE_KIND)) * ((1 * w COL_PC_EDGE_KIND_IS_STATIC)) - (0) = 0).
  { normalize_columns. clear - modular_21 Hone B11 B23. apply Z.mod_divide in modular_21; [destruct modular_21 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_21.
  circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_PC_EDGE_KIND_IS_STATIC, 1)], [(COL_PC_ROM_ACTIVE, 1)]) modular_22.
  expand_row modular_22.
  assert (equation_22 : ((1 * w COL_IS_PROGRAM_ROW)) * ((1 * w COL_PC_EDGE_KIND_IS_STATIC)) - ((1 * w COL_PC_ROM_ACTIVE)) = 0).
  { normalize_columns. clear - modular_22 Hone B19 B20 B23. apply Z.mod_divide in modular_22; [destruct modular_22 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_22.
  normalize_columns. nia.
Qed.

Lemma push_host_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_CALL_STACK_PUSH_PRESENT = 1 -> w COL_HOST_CALL_ACTIVE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  gate_bound COL_GUEST_ENTRY_ACTIVE B41.
  gate_bound COL_CALL_STACK_PUSH_PRESENT B42.
  gate_bound COL_SEL_CALL B127.
  gate_bound COL_SEL_CALL_INDIRECT B128.
  gate_bound COL_TARGET_FUNCTION_IS_GUEST B169.
  gate_bound COL_SEL_RETURN_CALL B322.
  gate_bound COL_SEL_RETURN_CALL_INDIRECT B323.
  gate_bound COL_HOST_CALL_ACTIVE B465.
  circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)], [(COL_GUEST_ENTRY_ACTIVE, 1)]) modular_0.
  expand_row modular_0.
  assert (equation_0 : ((1 * w COL_SEL_CALL) + (1 * w COL_SEL_CALL_INDIRECT) + (1 * w COL_SEL_RETURN_CALL) + (1 * w COL_SEL_RETURN_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_GUEST_ENTRY_ACTIVE)) = 0).
  { normalize_columns. clear - modular_0 Hone B41 B127 B128 B169 B322 B323. apply Z.mod_divide in modular_0; [destruct modular_0 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_0.
  circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)], [(COL_CALL_STACK_PUSH_PRESENT, 1)]) modular_1.
  expand_row modular_1.
  assert (equation_1 : ((1 * w COL_SEL_CALL) + (1 * w COL_SEL_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_CALL_STACK_PUSH_PRESENT)) = 0).
  { normalize_columns. clear - modular_1 Hone B42 B127 B128 B169. apply Z.mod_divide in modular_1; [destruct modular_1 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_1.
  circuit_row ([(COL_HOST_CALL_ACTIVE, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1); (COL_GUEST_ENTRY_ACTIVE, 1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_5.
  expand_row modular_5.
  assert (equation_5 : ((1 * w COL_HOST_CALL_ACTIVE) + (-1 * w COL_SEL_CALL) + (-1 * w COL_SEL_RETURN_CALL) + (-1 * w COL_SEL_CALL_INDIRECT) + (-1 * w COL_SEL_RETURN_CALL_INDIRECT) + (1 * w COL_GUEST_ENTRY_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_5 Hone B41 B127 B128 B322 B323 B465. apply Z.mod_divide in modular_5; [destruct modular_5 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_5.
  normalize_columns. nia.
Qed.

Lemma push_indirect_host_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_CALL_STACK_PUSH_PRESENT = 1 -> w COL_CI_HOST_CALL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  gate_bound COL_CALL_STACK_PUSH_PRESENT B42.
  gate_bound COL_SEL_CALL B127.
  gate_bound COL_SEL_CALL_INDIRECT B128.
  gate_bound COL_TARGET_FUNCTION_IS_GUEST B169.
  gate_bound COL_SEL_RETURN_CALL_INDIRECT B323.
  gate_bound COL_CI_HOST_CALL B488.
  circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)], [(COL_CALL_STACK_PUSH_PRESENT, 1)]) modular_1.
  expand_row modular_1.
  assert (equation_1 : ((1 * w COL_SEL_CALL) + (1 * w COL_SEL_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_CALL_STACK_PUSH_PRESENT)) = 0).
  { normalize_columns. clear - modular_1 Hone B42 B127 B128 B169. apply Z.mod_divide in modular_1; [destruct modular_1 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_1.
  circuit_row ([(COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_ONE, 1); (COL_TARGET_FUNCTION_IS_GUEST, -1)], [(COL_CI_HOST_CALL, 1)]) modular_4.
  expand_row modular_4.
  assert (equation_4 : ((1 * w COL_SEL_CALL_INDIRECT) + (1 * w COL_SEL_RETURN_CALL_INDIRECT)) * ((1 * w COL_ONE) + (-1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_CI_HOST_CALL)) = 0).
  { normalize_columns. clear - modular_4 Hone B128 B169 B323 B488. apply Z.mod_divide in modular_4; [destruct modular_4 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_4.
  normalize_columns. nia.
Qed.

Lemma read0_gather_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 -> w COL_GATHER_LOCAL_WRITE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  gate_bound (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) B55.
  gate_bound (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) B56.
  gate_bound (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) B57.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE B483.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1); ((List.nth 3%nat GATHER_KIND 0%nat), -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_11.
  expand_row modular_11.
  assert (equation_11 : ((1 * w COL_GATHER_LOCAL_WRITE) + (-1 * w (List.nth 3%nat GATHER_KIND 0%nat))) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_11 Hone B483 B575. apply Z.mod_divide in modular_11; [destruct modular_11 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_11.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_14.
  expand_row modular_14.
  assert (equation_14 : ((1 * w COL_GATHER_LOCAL_WRITE)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_14 Hone B464 B483. apply Z.mod_divide in modular_14; [destruct modular_14 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_14.
  circuit_row ([((List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat), 1); (COL_STACK_READS, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_17.
  expand_row modular_17.
  assert (equation_17 : ((1 * w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)) + (-1 * w COL_STACK_READS)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_17 Hone B53 B55 B56 B57. apply Z.mod_divide in modular_17; [destruct modular_17 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_17.
  normalize_columns. nia.
Qed.

Lemma read0_gather_local_lo_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 -> w COL_GATHER_LOCAL_WRITE_LO <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  gate_bound (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) B55.
  gate_bound (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) B56.
  gate_bound (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) B57.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE_LO B484.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([((List.nth 3%nat GATHER_KIND 0%nat), 1)], [(COL_ONE, 1); (COL_HOST_EVENT_SLOT_VARIANT, -1)], [(COL_GATHER_LOCAL_WRITE_LO, 1)]) modular_12.
  expand_row modular_12.
  assert (equation_12 : ((1 * w (List.nth 3%nat GATHER_KIND 0%nat))) = 0 -> ((1 * w COL_GATHER_LOCAL_WRITE_LO)) = 0).
  { normalize_columns. intro Hzero. rewrite Hzero in modular_12. clear - modular_12 Hone B484 B575. apply Z.mod_divide in modular_12; [destruct modular_12 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_12.
  circuit_row ([(COL_GATHER_LOCAL_WRITE_LO, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_15.
  expand_row modular_15.
  assert (equation_15 : ((1 * w COL_GATHER_LOCAL_WRITE_LO)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_15 Hone B464 B484. apply Z.mod_divide in modular_15; [destruct modular_15 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_15.
  circuit_row ([((List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat), 1); (COL_STACK_READS, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_17.
  expand_row modular_17.
  assert (equation_17 : ((1 * w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)) + (-1 * w COL_STACK_READS)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_17 Hone B53 B55 B56 B57. apply Z.mod_divide in modular_17; [destruct modular_17 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_17.
  normalize_columns. nia.
Qed.

Lemma read0_gmem_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1 -> w GMEM_LOCAL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  gate_bound (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) B55.
  gate_bound (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat) B56.
  gate_bound (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat) B57.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(GMEM_LOCAL, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_16.
  expand_row modular_16.
  assert (equation_16 : ((1 * w GMEM_LOCAL)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_16 Hone B464 B588. apply Z.mod_divide in modular_16; [destruct modular_16 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_16.
  circuit_row ([((List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat), 1); ((List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat), 1); (COL_STACK_READS, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_17.
  expand_row modular_17.
  assert (equation_17 : ((1 * w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 1%nat COL_STACK_READ_ACTIVE 0%nat)) + (1 * w (List.nth 2%nat COL_STACK_READ_ACTIVE 0%nat)) + (-1 * w COL_STACK_READS)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_17 Hone B53 B55 B56 B57. apply Z.mod_divide in modular_17; [destruct modular_17 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_17.
  normalize_columns. nia.
Qed.

Lemma write0_gather_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_STACK_WRITE0_ACTIVE = 1 -> w COL_GATHER_LOCAL_WRITE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_STACK_WRITE0_ACTIVE B58.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE B483.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1); ((List.nth 3%nat GATHER_KIND 0%nat), -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_11.
  expand_row modular_11.
  assert (equation_11 : ((1 * w COL_GATHER_LOCAL_WRITE) + (-1 * w (List.nth 3%nat GATHER_KIND 0%nat))) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_11 Hone B483 B575. apply Z.mod_divide in modular_11; [destruct modular_11 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_11.
  circuit_row ([(COL_STACK_WRITE0_ACTIVE, 1); (COL_STACK_WRITES, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_18.
  expand_row modular_18.
  assert (equation_18 : ((1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w COL_STACK_WRITES)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_18 Hone B54 B58. apply Z.mod_divide in modular_18; [destruct modular_18 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_18.
  normalize_columns. nia.
Qed.

Lemma write0_gather_local_lo_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_STACK_WRITE0_ACTIVE = 1 -> w COL_GATHER_LOCAL_WRITE_LO <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_STACK_WRITE0_ACTIVE B58.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE_LO B484.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([((List.nth 3%nat GATHER_KIND 0%nat), 1)], [(COL_ONE, 1); (COL_HOST_EVENT_SLOT_VARIANT, -1)], [(COL_GATHER_LOCAL_WRITE_LO, 1)]) modular_12.
  expand_row modular_12.
  assert (equation_12 : ((1 * w (List.nth 3%nat GATHER_KIND 0%nat))) = 0 -> ((1 * w COL_GATHER_LOCAL_WRITE_LO)) = 0).
  { normalize_columns. intro Hzero. rewrite Hzero in modular_12. clear - modular_12 Hone B484 B575. apply Z.mod_divide in modular_12; [destruct modular_12 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_12.
  circuit_row ([(COL_GATHER_LOCAL_WRITE_LO, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_15.
  expand_row modular_15.
  assert (equation_15 : ((1 * w COL_GATHER_LOCAL_WRITE_LO)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_15 Hone B464 B484. apply Z.mod_divide in modular_15; [destruct modular_15 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_15.
  circuit_row ([(COL_STACK_WRITE0_ACTIVE, 1); (COL_STACK_WRITES, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_18.
  expand_row modular_18.
  assert (equation_18 : ((1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w COL_STACK_WRITES)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_18 Hone B54 B58. apply Z.mod_divide in modular_18; [destruct modular_18 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_18.
  normalize_columns. nia.
Qed.

Lemma write0_gmem_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_STACK_WRITE0_ACTIVE = 1 -> w GMEM_LOCAL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_STACK_WRITE0_ACTIVE B58.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(GMEM_LOCAL, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_16.
  expand_row modular_16.
  assert (equation_16 : ((1 * w GMEM_LOCAL)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_16 Hone B464 B588. apply Z.mod_divide in modular_16; [destruct modular_16 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_16.
  circuit_row ([(COL_STACK_WRITE0_ACTIVE, 1); (COL_STACK_WRITES, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_18.
  expand_row modular_18.
  assert (equation_18 : ((1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w COL_STACK_WRITES)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_18 Hone B54 B58. apply Z.mod_divide in modular_18; [destruct modular_18 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_18.
  normalize_columns. nia.
Qed.

Lemma write0_hi_gather_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_STACK_WRITE0_HI_ACTIVE = 1 -> w COL_GATHER_LOCAL_WRITE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_STACK_WRITE0_ACTIVE B58.
  gate_bound COL_STACK_WRITE0_HI_ACTIVE B59.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE B483.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(COL_STACK_WRITE0_HI_ACTIVE, 1); (COL_STACK_WRITE0_ACTIVE, -1); (GK2_HI, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_10.
  expand_row modular_10.
  assert (equation_10 : ((1 * w COL_STACK_WRITE0_HI_ACTIVE) + (-1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w GK2_HI)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_10 Hone B58 B59 B585. apply Z.mod_divide in modular_10; [destruct modular_10 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_10.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1); ((List.nth 3%nat GATHER_KIND 0%nat), -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_11.
  expand_row modular_11.
  assert (equation_11 : ((1 * w COL_GATHER_LOCAL_WRITE) + (-1 * w (List.nth 3%nat GATHER_KIND 0%nat))) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_11 Hone B483 B575. apply Z.mod_divide in modular_11; [destruct modular_11 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_11.
  circuit_row ([(COL_STACK_WRITE0_ACTIVE, 1); (COL_STACK_WRITES, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_18.
  expand_row modular_18.
  assert (equation_18 : ((1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w COL_STACK_WRITES)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_18 Hone B54 B58. apply Z.mod_divide in modular_18; [destruct modular_18 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_18.
  normalize_columns. nia.
Qed.

Lemma write0_hi_gmem_local_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_STACK_WRITE0_HI_ACTIVE = 1 -> w GMEM_LOCAL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_STACK_WRITE0_ACTIVE B58.
  gate_bound COL_STACK_WRITE0_HI_ACTIVE B59.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE B483.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(COL_STACK_WRITE0_HI_ACTIVE, 1); (COL_STACK_WRITE0_ACTIVE, -1); (GK2_HI, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_10.
  expand_row modular_10.
  assert (equation_10 : ((1 * w COL_STACK_WRITE0_HI_ACTIVE) + (-1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w GK2_HI)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_10 Hone B58 B59 B585. apply Z.mod_divide in modular_10; [destruct modular_10 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_10.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1); ((List.nth 3%nat GATHER_KIND 0%nat), -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_11.
  expand_row modular_11.
  assert (equation_11 : ((1 * w COL_GATHER_LOCAL_WRITE) + (-1 * w (List.nth 3%nat GATHER_KIND 0%nat))) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_11 Hone B483 B575. apply Z.mod_divide in modular_11; [destruct modular_11 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_11.
  circuit_row ([(GMEM_LOCAL, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_16.
  expand_row modular_16.
  assert (equation_16 : ((1 * w GMEM_LOCAL)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_16 Hone B464 B588. apply Z.mod_divide in modular_16; [destruct modular_16 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_16.
  circuit_row ([(COL_STACK_WRITE0_ACTIVE, 1); (COL_STACK_WRITES, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_18.
  expand_row modular_18.
  assert (equation_18 : ((1 * w COL_STACK_WRITE0_ACTIVE) + (-1 * w COL_STACK_WRITES)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_18 Hone B54 B58. apply Z.mod_divide in modular_18; [destruct modular_18 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_18.
  normalize_columns. nia.
Qed.

Lemma tail_host_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_RETURN_CALL = 1 -> w COL_HOST_CALL_ACTIVE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  gate_bound COL_GUEST_ENTRY_ACTIVE B41.
  gate_bound COL_SEL_CALL B127.
  gate_bound COL_SEL_CALL_INDIRECT B128.
  gate_bound COL_TARGET_FUNCTION_IS_GUEST B169.
  gate_bound COL_SEL_RETURN_CALL B322.
  gate_bound COL_SEL_RETURN_CALL_INDIRECT B323.
  gate_bound COL_HOST_CALL_ACTIVE B465.
  circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)], [(COL_GUEST_ENTRY_ACTIVE, 1)]) modular_0.
  expand_row modular_0.
  assert (equation_0 : ((1 * w COL_SEL_CALL) + (1 * w COL_SEL_CALL_INDIRECT) + (1 * w COL_SEL_RETURN_CALL) + (1 * w COL_SEL_RETURN_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_GUEST_ENTRY_ACTIVE)) = 0).
  { normalize_columns. clear - modular_0 Hone B41 B127 B128 B169 B322 B323. apply Z.mod_divide in modular_0; [destruct modular_0 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_0.
  circuit_row ([(COL_SEL_RETURN_CALL, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1); (COL_ONE, -1)], ([] : SparseRow)) modular_2.
  expand_row modular_2.
  assert (equation_2 : ((1 * w COL_SEL_RETURN_CALL)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST) + (-1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_2 Hone B169 B322. apply Z.mod_divide in modular_2; [destruct modular_2 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_2.
  circuit_row ([(COL_HOST_CALL_ACTIVE, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1); (COL_GUEST_ENTRY_ACTIVE, 1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_5.
  expand_row modular_5.
  assert (equation_5 : ((1 * w COL_HOST_CALL_ACTIVE) + (-1 * w COL_SEL_CALL) + (-1 * w COL_SEL_RETURN_CALL) + (-1 * w COL_SEL_CALL_INDIRECT) + (-1 * w COL_SEL_RETURN_CALL_INDIRECT) + (1 * w COL_GUEST_ENTRY_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_5 Hone B41 B127 B128 B322 B323 B465. apply Z.mod_divide in modular_5; [destruct modular_5 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_5.
  normalize_columns. nia.
Qed.

Lemma indirect_tail_host_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_RETURN_CALL_INDIRECT = 1 -> w COL_HOST_CALL_ACTIVE <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  gate_bound COL_GUEST_ENTRY_ACTIVE B41.
  gate_bound COL_SEL_CALL B127.
  gate_bound COL_SEL_CALL_INDIRECT B128.
  gate_bound COL_TARGET_FUNCTION_IS_GUEST B169.
  gate_bound COL_SEL_RETURN_CALL B322.
  gate_bound COL_SEL_RETURN_CALL_INDIRECT B323.
  gate_bound COL_HOST_CALL_ACTIVE B465.
  circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)], [(COL_GUEST_ENTRY_ACTIVE, 1)]) modular_0.
  expand_row modular_0.
  assert (equation_0 : ((1 * w COL_SEL_CALL) + (1 * w COL_SEL_CALL_INDIRECT) + (1 * w COL_SEL_RETURN_CALL) + (1 * w COL_SEL_RETURN_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST)) - ((1 * w COL_GUEST_ENTRY_ACTIVE)) = 0).
  { normalize_columns. clear - modular_0 Hone B41 B127 B128 B169 B322 B323. apply Z.mod_divide in modular_0; [destruct modular_0 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_0.
  circuit_row ([(COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1); (COL_ONE, -1)], ([] : SparseRow)) modular_3.
  expand_row modular_3.
  assert (equation_3 : ((1 * w COL_SEL_RETURN_CALL_INDIRECT)) * ((1 * w COL_TARGET_FUNCTION_IS_GUEST) + (-1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_3 Hone B169 B323. apply Z.mod_divide in modular_3; [destruct modular_3 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_3.
  circuit_row ([(COL_HOST_CALL_ACTIVE, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1); (COL_SEL_CALL_INDIRECT, -1); (COL_SEL_RETURN_CALL_INDIRECT, -1); (COL_GUEST_ENTRY_ACTIVE, 1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_5.
  expand_row modular_5.
  assert (equation_5 : ((1 * w COL_HOST_CALL_ACTIVE) + (-1 * w COL_SEL_CALL) + (-1 * w COL_SEL_RETURN_CALL) + (-1 * w COL_SEL_CALL_INDIRECT) + (-1 * w COL_SEL_RETURN_CALL_INDIRECT) + (1 * w COL_GUEST_ENTRY_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_5 Hone B41 B127 B128 B322 B323 B465. apply Z.mod_divide in modular_5; [destruct modular_5 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_5.
  normalize_columns. nia.
Qed.

Lemma gather_local_gmem_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_GATHER_LOCAL_WRITE = 1 -> w GMEM_LOCAL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE B483.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1); ((List.nth 3%nat GATHER_KIND 0%nat), -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_11.
  expand_row modular_11.
  assert (equation_11 : ((1 * w COL_GATHER_LOCAL_WRITE) + (-1 * w (List.nth 3%nat GATHER_KIND 0%nat))) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_11 Hone B483 B575. apply Z.mod_divide in modular_11; [destruct modular_11 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_11.
  circuit_row ([(GMEM_LOCAL, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_16.
  expand_row modular_16.
  assert (equation_16 : ((1 * w GMEM_LOCAL)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_16 Hone B464 B588. apply Z.mod_divide in modular_16; [destruct modular_16 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_16.
  normalize_columns. nia.
Qed.

Lemma gather_local_lo_gmem_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w COL_GATHER_LOCAL_WRITE_LO = 1 -> w GMEM_LOCAL <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_STACK_READS B53.
  circuit_range COL_STACK_WRITES B54.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound COL_GATHER_LOCAL_WRITE_LO B484.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  gate_bound GK2_HI B585.
  gate_bound GMEM_LOCAL B588.
  gate_bound GMEM_OUTPUT B589.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_READS, 1); ((List.nth 1%nat GATHER_KIND 0%nat), -1); ((List.nth 5%nat GATHER_KIND 0%nat), -1); ((List.nth 6%nat GATHER_KIND 0%nat), -1); (GMEM_LOCAL, 1); (GMEM_OUTPUT, 1)], ([] : SparseRow)) modular_7.
  expand_row modular_7.
  assert (equation_7 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_READS) + (-1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (-1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w GMEM_LOCAL) + (1 * w GMEM_OUTPUT)) - (0) = 0).
  { normalize_columns. clear - modular_7 Hone B53 B464 B573 B577 B578 B588 B589. apply Z.mod_divide in modular_7; [destruct modular_7 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_7.
  circuit_row ([(COL_GATHER_ACTIVE, 1)], [(COL_STACK_WRITES, 1); ((List.nth 2%nat GATHER_KIND 0%nat), -1); (GK2_HI, 1)], ([] : SparseRow)) modular_8.
  expand_row modular_8.
  assert (equation_8 : ((1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_STACK_WRITES) + (-1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w GK2_HI)) - (0) = 0).
  { normalize_columns. clear - modular_8 Hone B54 B464 B574 B585. apply Z.mod_divide in modular_8; [destruct modular_8 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_8.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  circuit_row ([((List.nth 3%nat GATHER_KIND 0%nat), 1)], [(COL_ONE, 1); (COL_HOST_EVENT_SLOT_VARIANT, -1)], [(COL_GATHER_LOCAL_WRITE_LO, 1)]) modular_12.
  expand_row modular_12.
  assert (equation_12 : ((1 * w (List.nth 3%nat GATHER_KIND 0%nat))) = 0 -> ((1 * w COL_GATHER_LOCAL_WRITE_LO)) = 0).
  { normalize_columns. intro Hzero. rewrite Hzero in modular_12. clear - modular_12 Hone B484 B575. apply Z.mod_divide in modular_12; [destruct modular_12 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_12.
  circuit_row ([(GMEM_LOCAL, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) modular_16.
  expand_row modular_16.
  assert (equation_16 : ((1 * w GMEM_LOCAL)) * ((1 * w COL_ONE) + (-1 * w COL_GATHER_ACTIVE)) - (0) = 0).
  { normalize_columns. clear - modular_16 Hone B464 B588. apply Z.mod_divide in modular_16; [destruct modular_16 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_16.
  normalize_columns. nia.
Qed.

Lemma gather_memory_read_write_exclusive : forall (w : Witness), fieldPrime -> FullRelation.satFull w ->
  w (List.nth 5%nat GATHER_KIND 0%nat) = 1 -> w (List.nth 6%nat GATHER_KIND 0%nat) <> 1.
Proof.
  intros w Hprime Hfull Ha Hb.
  pose proof (proj1 Hfull) as Hone.
  gate_bound COL_GATHER_ACTIVE B464.
  gate_bound (List.nth 0%nat GATHER_KIND 0%nat) B572.
  gate_bound (List.nth 1%nat GATHER_KIND 0%nat) B573.
  gate_bound (List.nth 2%nat GATHER_KIND 0%nat) B574.
  gate_bound (List.nth 3%nat GATHER_KIND 0%nat) B575.
  gate_bound (List.nth 4%nat GATHER_KIND 0%nat) B576.
  gate_bound (List.nth 5%nat GATHER_KIND 0%nat) B577.
  gate_bound (List.nth 6%nat GATHER_KIND 0%nat) B578.
  gate_bound (List.nth 7%nat GATHER_KIND 0%nat) B579.
  gate_bound (List.nth 8%nat GATHER_KIND 0%nat) B580.
  gate_bound (List.nth 9%nat GATHER_KIND 0%nat) B581.
  circuit_row ([((List.nth 0%nat GATHER_KIND 0%nat), 1); ((List.nth 1%nat GATHER_KIND 0%nat), 1); ((List.nth 2%nat GATHER_KIND 0%nat), 1); ((List.nth 3%nat GATHER_KIND 0%nat), 1); ((List.nth 4%nat GATHER_KIND 0%nat), 1); ((List.nth 5%nat GATHER_KIND 0%nat), 1); ((List.nth 6%nat GATHER_KIND 0%nat), 1); ((List.nth 7%nat GATHER_KIND 0%nat), 1); ((List.nth 8%nat GATHER_KIND 0%nat), 1); ((List.nth 9%nat GATHER_KIND 0%nat), 1); (COL_GATHER_ACTIVE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) modular_9.
  expand_row modular_9.
  assert (equation_9 : ((1 * w (List.nth 0%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 1%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 2%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 3%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 4%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 5%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 6%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 7%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 8%nat GATHER_KIND 0%nat)) + (1 * w (List.nth 9%nat GATHER_KIND 0%nat)) + (-1 * w COL_GATHER_ACTIVE)) * ((1 * w COL_ONE)) - (0) = 0).
  { normalize_columns. clear - modular_9 Hone B464 B572 B573 B574 B575 B576 B577 B578 B579 B580 B581. apply Z.mod_divide in modular_9; [destruct modular_9 as [k Hk]; unfold modulus in *; nia | unfold modulus; lia]. }
  clear modular_9.
  normalize_columns. nia.
Qed.

Theorem gate_exclusions_sound : forall (w : Witness) a b,
  fieldPrime -> FullRelation.satFull w -> In (a,b) gateExclusions ->
  w a = 1 -> w b <> 1.
Proof.
  intros w a b Hprime Hfull Hin.
  pose proof (full_stack_port_gates w Hprime Hfull) as Hports.
  pose proof (separate_read _ Hports) as Hread.
  pose proof (separate_write_lo _ Hports) as Hwrite.
  pose proof (separate_write_hi _ Hports) as Hwritehi.
  pose proof (rmw_nonnegative _ Hports) as Hrmw.
  gate_bound (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) Bread.
  gate_bound COL_STACK_WRITE0_ACTIVE Bwrite.
  gate_bound COL_STACK_WRITE0_HI_ACTIVE Bwritehi.
  assert (R : w (List.nth 0%nat COL_STACK_SEPARATE 0%nat) = 1 ->
    w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) = 1) by (normalize_columns; lia).
  assert (W : w (List.nth 1%nat COL_STACK_SEPARATE 0%nat) = 1 ->
    w COL_STACK_WRITE0_ACTIVE = 1) by (normalize_columns; lia).
  assert (WH : w (List.nth 2%nat COL_STACK_SEPARATE 0%nat) = 1 ->
    w COL_STACK_WRITE0_HI_ACTIVE = 1) by (normalize_columns; lia).
  cbn [gateExclusions] in Hin.
  repeat destruct Hin as [Hin | Hin]; try contradiction; inversion Hin; subst; intro Ha; normalize_columns;
    first [apply R in Ha | apply W in Ha | apply WH in Ha | idtac];
    first [eapply pc_exit_exclusive; eassumption
         | eapply push_host_exclusive; eassumption
         | eapply push_indirect_host_exclusive; eassumption
         | eapply read0_gather_local_exclusive; eassumption
         | eapply read0_gather_local_lo_exclusive; eassumption
         | eapply read0_gmem_local_exclusive; eassumption
         | eapply write0_gather_local_exclusive; eassumption
         | eapply write0_gather_local_lo_exclusive; eassumption
         | eapply write0_gmem_local_exclusive; eassumption
         | eapply write0_hi_gather_local_exclusive; eassumption
         | eapply write0_hi_gmem_local_exclusive; eassumption
         | eapply tail_host_exclusive; eassumption
         | eapply indirect_tail_host_exclusive; eassumption
         | eapply gather_local_gmem_exclusive; eassumption
         | eapply gather_local_lo_gmem_exclusive; eassumption
         | eapply gather_memory_read_write_exclusive; eassumption].
Qed.
