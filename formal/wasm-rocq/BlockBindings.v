(* BLOCK and an inner END preserve the operand stack and current frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings LocalSimulation CallMemory CallContext StackBindings StraightLineSpec StackSimulation.
Import ListNotations.
Open Scope Z_scope.

Lemma full_live_capture_zero : forall (w : Witness),
  FullRelation.satFull w -> w COL_ACTIVATION_FINISHED_AFTER = 0 -> w COL_OUTPUT_CAPTURED = 0.
Proof.
  intros w Hfull Hhalt. pose proof (proj1 Hfull) as Hone.
  circuit_row ([(COL_OUTPUT_CAPTURED, 1)], [(COL_ONE, 1); (COL_ACTIVATION_FINISHED_AFTER, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. rewrite Hone, Hhalt in Hrow.
  pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow.
Qed.

Lemma full_block_context : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w ->
  (chosen = COL_SEL_BLOCK \/ chosen = COL_SEL_END) -> w chosen = 1 ->
  w COL_CALL_STACK_POP_PRESENT = 0 -> w COL_ACTIVATION_FINISHED_AFTER = 0 ->
  w COL_SP_AFTER = w COL_SP_BEFORE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE /\
  w COL_STACK_WRITE0_ACTIVE = 0 /\ w COL_STACK_WRITE0_HI_ACTIVE = 0 /\
  filter isLocalWrite (eventsForRow memories w) = [].
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen Hpop Hhalt.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [-> | ->]; vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_live_capture_zero w Hfull Hhalt) as Hcapture.
  destruct (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen) as [Hparam [Hzero [Hgather _]]].
  destruct (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput Hdiscard].
  pose proof (full_no_gather_local_write w chosen Hprime Hfull Hin Hchosen) as HgatherLocal.
  destruct Hkind as [-> | ->];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0) by
        (selected_row chosen callStackPushRow Hrow;
         pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT); field_linear Hrow);
      assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0) by
        (selected_row chosen guestEntryFlagRow Hrow;
         pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE); field_linear Hrow);
      assert (Hhost : w COL_HOST_CALL_ACTIVE = 0) by
        (selected_row chosen hostCallSupportRow Hrow; rewrite Hone in Hrow;
         pose proof (value_bound w COL_HOST_CALL_ACTIVE); field_linear Hrow);
      assert (Hparams : w GHC_PARAMS = 0) by
        (full_row hostCallParamsRow Hrow; pose proof (full_no_host_split w Hfull Hhost) as Hsplit; normalize_columns;
     rewrite Hhost, Hsplit in Hrow;
         pose proof (value_bound w GHC_PARAMS); field_linear Hrow);
      assert (Hreads : w COL_STACK_READS = 0) by
        (selected_row chosen stackReadsArityRow Hrow; rewrite Hone, Hgather, Hparam in Hrow;
         pose proof (value_bound w COL_STACK_READS); field_linear Hrow);
      assert (Hwrites : w COL_STACK_WRITES = 0) by
        (selected_row chosen stackWritesArityRow Hrow; rewrite Hone, Hgather in Hrow;
         pose proof (value_bound w COL_STACK_WRITES); field_linear Hrow);
      assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0) by
        (full_row stackWritesActiveRow Hrow; rewrite Hone, Hwrites in Hrow;
         pose proof (value_bound w COL_STACK_WRITE0_ACTIVE); field_linear Hrow);
      assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0) by
        (full_row stackWriteHighActiveRow Hrow; rewrite Hone, Hwrite, Hhi in Hrow;
         pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE); field_linear Hrow);
      assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0) by
        (selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
         pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow);
      split;
      [ full_row stackPointerBalanceRow Hrow; zero_host_switch Hrow; zero_gather_kinds Hkinds Hrow;
        rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow;
        pose proof (value_bound w COL_SP_BEFORE); pose proof (value_bound w COL_SP_AFTER); field_linear Hrow
      | split;
        [ full_row stackFrameBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hpush, Hpop in Hrow;
          pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE);
          pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER); field_linear Hrow
        | split;
          [ full_row localsBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hguest, Hpop in Hrow;
            pose proof (value_bound w COL_LOCALS_FBP_BEFORE);
            pose proof (value_bound w COL_LOCALS_FBP_AFTER); field_linear Hrow
          | split;
            [ pose proof (full_call_depth_balance w chosen Hprime Hfull Hin Hchosen); lia
            | split; [exact Hwrite | split; [exact HwriteHi | ]];
              rewrite (local_writes_in_row w Hparam Hzero HgatherLocal), HlocalWrite; reflexivity ] ] ] ]
    end.
Qed.
