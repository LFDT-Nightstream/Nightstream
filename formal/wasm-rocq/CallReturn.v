(* Restore the suspended caller from the context actually written by the call. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings CallBindings CallMemory CallContext.
Import ListNotations.
Open Scope Z_scope.

(* Neither a call nor an END writes a stack cell: the callee's arguments become
   its locals in place, and an END's results move on later rows. *)
Lemma full_call_end_no_stack_writes : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w ->
  (chosen = COL_SEL_CALL \/ chosen = COL_SEL_END) -> w chosen = 1 ->
  filter isStackWrite (eventsForRow memories w) = [].
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [-> | ->]; vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen) as [Hp [Hz Hgather]].
  pose proof (quiet_gather_high w (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)) as Hhi.
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { destruct Hkind as [-> | ->].
    - selected_circuit_row COL_SEL_CALL
        ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
          (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_STACK_WRITES, 1)], ([] : SparseRow)) Hrow.
      rewrite Hone in Hrow. pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow.
    - selected_row COL_SEL_END stackWritesArityRow Hrow. rewrite Hone, Hgather, Hp, Hz in Hrow.
      pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { destruct Hkind as [-> | ->];
      match goal with Hchosen : value w ?chosen = 1 |- _ =>
        selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
        pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow end. }
  rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
  rewrite Hwrite, HwriteHi, HlocalWrite, HgatherLocal, HgatherLocalLo. reflexivity.
Qed.
