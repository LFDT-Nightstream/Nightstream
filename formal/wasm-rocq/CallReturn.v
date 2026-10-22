(* Restore the suspended caller from the context actually written by the call. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings CallBindings CallMemory CallContext.
Import ListNotations.
Open Scope Z_scope.

Lemma full_call_end_no_local_writes : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w ->
  (chosen = COL_SEL_CALL \/ chosen = COL_SEL_END) -> w chosen = 1 ->
  filter isLocalWrite (eventsForRow memories w) = [].
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [-> | ->]; vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen) as [Hp [Hz _]].
  pose proof (quiet_gather_kinds w (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)) as Hkinds.
  assert (Hgather : w COL_GATHER_LOCAL_WRITE_LO = 0).
  { full_row gatherLocalLowWriteRow Hrow. zero_gather_kinds Hkinds Hrow.
    pose proof (value_bound w COL_GATHER_LOCAL_WRITE_LO). field_linear Hrow. }
  assert (Hwrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { destruct Hkind as [-> | ->];
      match goal with Hchosen : value w ?chosen = 1 |- _ =>
        selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
        pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow end. }
  rewrite (local_writes_in_row w Hp Hz Hgather), Hwrite. reflexivity.
Qed.
