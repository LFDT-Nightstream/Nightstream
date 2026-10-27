(* BLOCK and an inner END preserve the operand stack and current frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings LocalSimulation CallMemory CallContext StackBindings StraightLineSpec StackSimulation
  CallInitialization.
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
  filter isStackWrite (eventsForRow memories w) = [].
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen Hpop Hhalt.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [-> | ->]; vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_live_capture_zero w Hfull Hhalt) as Hcapture.
  destruct (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen) as [Hparam [Hzero Hgather]].
  destruct (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate)
      (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen)). }
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
      assert (Hlatch : w COL_HOST_EVENT_EXIT_LATCH = 0) by
        (circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1);
           (COL_TRAPPED_AFTER, -1); (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
           (COL_HOST_EVENT_EXIT_LATCH, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow;
         destruct (full_selected_no_host_switch w chosen Hprime Hfull Hin Hchosen) as [_ [Hresume _]];
         pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear;
         pose proof (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen) as Hprogram;
         circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hfb;
         circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Htb;
         circuit_boolean COL_TRAPPED_AFTER Hta; circuit_boolean COL_HOST_EVENT_EXIT_LATCH Hlb;
         expand_row Hrow; expand_row Hfb; expand_row Htb; normalize_columns;
         rewrite Hone, Hhalt, Hclear in Hrow; rewrite Hprogram in Hfb, Htb;
         pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE);
         pose proof (value_bound w COL_TRAPPED_BEFORE);
         assert (w COL_ACTIVATION_FINISHED_BEFORE = 0) by field_linear Hfb;
         assert (w COL_TRAPPED_BEFORE = 0) by field_linear Htb;
         normalize_columns; destruct Hta, Hlb; field_linear Hrow);
      assert (Hdiscard : w COL_FRAME_SP_DROP = 0) by
        (apply full_frame_sp_drop_idle; try assumption;
         [apply (full_selected_no_slide w _ Hprime Hfull Hin Hchosen); [vm_compute; intuition discriminate | exact Hpop]
         | congruence]);
      assert (Hwrites : w COL_STACK_WRITES = 0) by
        (selected_row chosen stackWritesArityRow Hrow; rewrite Hone, Hgather, Hparam, Hzero in Hrow;
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
        [ full_row stackFrameBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hguest, Hturn, Hpop in Hrow;
          pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE);
          pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER); field_linear Hrow
        | split;
          [ full_row localsBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hpush, Hpop, Hturn in Hrow;
            pose proof (value_bound w COL_LOCALS_FBP_BEFORE);
            pose proof (value_bound w COL_LOCALS_FBP_AFTER); field_linear Hrow
          | split;
            [ pose proof (full_call_depth_balance w chosen Hprime Hfull Hin Hchosen); lia
            | rewrite stack_writes_in_row; unfold stackPortWrites, writeIf;
              rewrite Hwrite, HwriteHi, HlocalWrite, HgatherLocal, HgatherLocalLo; reflexivity ] ] ] ]
    end.
Qed.
