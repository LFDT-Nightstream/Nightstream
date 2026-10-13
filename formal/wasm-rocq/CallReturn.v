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

Theorem call_return_restores_context : forall initial afterCall beforeEnd final callRow endRow returnPC,
  fieldPrime -> FullRelation.satFull callRow -> FullRelation.satFull endRow ->
  callRow COL_SEL_CALL = 1 -> callRow COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  endRow COL_SEL_END = 1 -> endRow COL_PC_EDGE_KIND = 1 ->
  Replay initial (eventsForRow memories callRow) afterCall ->
  Replay beforeEnd (eventsForRow memories endRow) final ->
  initial pcRomId [callRow COL_PC_BEFORE; 1] = Some returnPC ->
  endRow COL_CALL_STACK_DEPTH_BEFORE = callRow COL_CALL_STACK_DEPTH_AFTER ->
  (forall memory column address, In (memory, column) contextMemories ->
    beforeEnd memory address = afterCall memory address) ->
  endRow COL_PC_AFTER = returnPC /\
  endRow COL_LOCALS_FBP_AFTER = callRow COL_LOCALS_FBP_BEFORE /\
  endRow COL_STACK_FRAME_BASE_AFTER = callRow COL_STACK_FRAME_BASE_BEFORE /\
  endRow COL_CALL_STACK_DEPTH_AFTER = callRow COL_CALL_STACK_DEPTH_BEFORE /\
  endRow COL_SP_AFTER = endRow COL_SP_BEFORE /\
  (forall address, final stackId address = beforeEnd stackId address) /\
  (forall address, final localLoId address = beforeEnd localLoId address).
Proof.
  intros initial afterCall beforeEnd final callRow endRow returnPC Hprime Hcall Hend Hselected Hguest
    HendSelected Hedge HcallReplay HendReplay HreturnImage Hdepth Hpreserve.
  destruct (full_direct_guest_call callRow Hprime Hcall Hselected Hguest) as [Hentry [Hpush Hpop]].
  destruct (full_call_context_push callRow Hcall Hpush) as [HcallAddr [HsavedFbp [HsavedBase Hchoice]]].
  pose proof (full_call_depth_balance callRow Hprime Hcall) as HcallDepth.
  pose proof (value_bound callRow COL_CALL_STACK_DEPTH_BEFORE) as Hbound.
  destruct (full_end_pops_nonempty_context endRow Hprime Hend HendSelected Hedge ltac:(lia))
    as [HendPop [HendPush Hcapture]].
  pose proof (full_call_depth_balance endRow Hprime Hend) as HendDepth.
  destruct (full_return_context_restore endRow Hend HendPop) as [Hpc [Hfbp [Hbase HendAddr]]].
  assert (Haddress : endRow COL_CALL_STACK_ADDR = callRow COL_CALL_STACK_ADDR) by lia.
  assert (Hvalues : forall memory column, In (memory, column) contextMemories -> endRow column = callRow column).
  { intros memory column Hin.
    pose proof (context_push_records initial afterCall callRow memory column Hin Hpush HcallReplay) as Hsaved.
    pose proof (context_pop_reads beforeEnd final endRow memory column Hin HendPush HendPop HendReplay) as Hread.
    rewrite Haddress, (Hpreserve memory column _ Hin), Hsaved in Hread. congruence. }
  pose proof (Hvalues callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE ltac:(cbn; auto)) as Hreturn.
  pose proof (Hvalues callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE ltac:(cbn; auto)) as HcallerFbp.
  pose proof (Hvalues callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE ltac:(cbn; auto)) as HcallerBase.
  pose proof (call_return_pc_read initial afterCall callRow HcallReplay Hpush) as HreturnRead.
  rewrite Hchoice, HreturnImage in HreturnRead.
  destruct (full_end_stack endRow Hprime Hend HendSelected Hcapture) as [Hlo [Hhi Hsp]].
  split; [congruence | ]. split; [congruence | ]. split; [congruence | ].
  split; [lia | ]. split; [exact Hsp | ]. split.
  - intros address. eapply replay_preserves; [exact HendReplay | apply stack_writes_inactive; assumption].
  - eapply no_local_writes_preserve; [exact HendReplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact Hend | right; reflexivity | exact HendSelected].
Qed.

Lemma stack_rejoins_caller : forall initial beforeEnd final (callRow endRow : Witness) prefix arguments values calleeBase,
  StraightLineSpec.stackMatches initial (StraightLineSpec.beforeBoundary callRow) (prefix ++ arguments) ->
  calleeBase = callRow COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix) ->
  endRow COL_STACK_FRAME_BASE_AFTER = callRow COL_STACK_FRAME_BASE_BEFORE ->
  endRow COL_SP_AFTER = calleeBase + Z.of_nat (length values) ->
  Forall (fun value => 0 <= value < WasmIntegers.i32Modulus) values ->
  (forall index value, nth_error values index = Some value ->
    beforeEnd stackId [2 * (calleeBase + Z.of_nat index)] = Some value) ->
  (forall index, index < calleeBase -> beforeEnd stackId [2 * index] = initial stackId [2 * index]) ->
  (forall address, final stackId address = beforeEnd stackId address) ->
  StraightLineSpec.stackMatches final (StraightLineSpec.afterBoundary endRow) (prefix ++ values).
Proof.
  intros initial beforeEnd final callRow endRow prefix arguments values calleeBase
    [Hbase [Hsp [Hbounds Hcells]]] Hcallee Hrestore HfinalSP Hvalues Hresults Hprefix Hfinal.
  unfold StraightLineSpec.stackMatches. cbn [StraightLineSpec.afterBoundary StraightLineSpec.boundarySP StraightLineSpec.boundaryStackBase].
  cbn [StraightLineSpec.beforeBoundary StraightLineSpec.boundarySP StraightLineSpec.boundaryStackBase] in Hbase, Hcells.
  split; [lia | ]. split; [rewrite length_app, Nat2Z.inj_add; lia | ].
  split; [apply Forall_app; apply Forall_app in Hbounds; tauto | ].
  intros index value Hlookup. rewrite Hfinal, Hrestore.
  destruct (lt_dec index (length prefix)) as [Hin | Hout].
  - rewrite Hprefix by lia. apply Hcells.
    rewrite nth_error_app1 in Hlookup |- * by exact Hin. exact Hlookup.
  - rewrite nth_error_app2 in Hlookup by lia.
    replace (2 * (callRow COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) with
      (2 * (calleeBase + Z.of_nat (index - length prefix))) by (rewrite Nat2Z.inj_sub by lia; lia).
    apply Hresults. exact Hlookup.
Qed.
