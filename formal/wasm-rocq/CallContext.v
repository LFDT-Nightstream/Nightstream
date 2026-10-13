(* The three return-context memories, including their concrete push/pop ports. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  TraceRefinement LocalBindings CallBindings CallMemory CallInitialization.
Import ListNotations.
Open Scope Z_scope.

Definition contextMemories : list (nat * Column) :=
  [(callStackReturnPcId, COL_CALL_STACK_RETURN_PC_VALUE);
   (callStackCallerFbpId, COL_CALL_STACK_CALLER_FBP_VALUE);
   (callStackCallerSpBaseId, COL_CALL_STACK_CALLER_SP_BASE_VALUE)].

Definition contextSpec (memory : nat) (column : Column) : MemorySpec :=
  mkMemory memory Ram
    [mkPort [COL_CALL_STACK_ADDR] column (Write None) (When COL_CALL_STACK_PUSH_PRESENT);
     mkPort [COL_CALL_STACK_ADDR] column Read (When COL_CALL_STACK_POP_PRESENT)].

Lemma context_catalog_split : forall memory column,
  In (memory, column) contextMemories ->
  memories = firstn memory memories ++ contextSpec memory column :: skipn (S memory) memories /\
  readOnlyMemory memory (firstn memory memories) = true /\
  readOnlyMemory memory (skipn (S memory) memories) = true.
Proof.
  intros memory column [H | [H | [H | []]]]; inversion H; subst;
    vm_compute; repeat split; reflexivity.
Qed.

Lemma context_preserved_without_push : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories -> w COL_CALL_STACK_PUSH_PRESENT = 0 ->
  Replay initial (eventsForRow memories w) final ->
  forall address, final memory address = initial memory address.
Proof.
  intros initial final w memory column Hin Hpush Hreplay address.
  eapply replay_preserves; [exact Hreplay | ].
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons.
  apply Forall_app. split; [apply read_only_row; exact Hbefore | ].
  apply Forall_app. split; [ | apply read_only_row; exact Hafter].
  unfold contextSpec. cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map].
  cbv [portEvent active]. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hpush. cbn. let pop := eval compute in COL_CALL_STACK_POP_PRESENT in destruct (w pop =? 1); repeat constructor.
Qed.

Lemma context_push_records : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories -> w COL_CALL_STACK_PUSH_PRESENT = 1 ->
  Replay initial (eventsForRow memories w) final ->
  final memory [w COL_CALL_STACK_ADDR] = Some (w column).
Proof.
  intros initial final w memory column Hin Hpush Hreplay.
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [before [_ Hrest]].
  destruct (replay_app_inv _ _ _ _ Hrest) as [after [Hport Htail]].
  rewrite (replay_preserves _ _ _ _ _ Htail (read_only_row _ _ _ _ Hafter)).
  unfold contextSpec in Hport.
  cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map] in Hport.
  cbv [portEvent active] in Hport. cbn [activation addressColumns valueColumn portKind map option_map] in Hport.
  normalize_columns. rewrite Hpush in Hport. cbn in Hport.
  inversion Hport; subst. match goal with H : EventStep _ (WriteEvent _ _ _ _) _ |- _ => inversion H; subst end.
  match goal with H : Replay _ _ after |- _ =>
    rewrite (replay_preserves _ _ _ _ _ H); [apply update_same | ] end.
  let pop := eval compute in COL_CALL_STACK_POP_PRESENT in destruct (w pop =? 1); repeat constructor.
Qed.

Lemma context_pop_reads : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 -> w COL_CALL_STACK_POP_PRESENT = 1 ->
  Replay initial (eventsForRow memories w) final ->
  initial memory [w COL_CALL_STACK_ADDR] = Some (w column).
Proof.
  intros initial final w memory column Hin Hpush Hpop Hreplay.
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [before [Hprefix Hrest]].
  destruct (replay_app_inv _ _ _ _ Hrest) as [after [Hport Htail]].
  unfold contextSpec in Hport.
  cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map] in Hport.
  cbv [portEvent active] in Hport. cbn [activation addressColumns valueColumn portKind map option_map] in Hport.
  normalize_columns. rewrite Hpush, Hpop in Hport. cbn in Hport.
  inversion Hport; subst. match goal with H : EventStep _ (ReadEvent _ _ _) _ |- _ => inversion H; subst end.
  rewrite <- (replay_preserves _ _ _ _ _ Hprefix (read_only_row _ _ _ _ Hbefore)). assumption.
Qed.

Lemma full_call_depth_balance : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE +
    w COL_CALL_STACK_PUSH_PRESENT - w COL_CALL_STACK_POP_PRESENT.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_CALL_STACK_DEPTH_BEFORE Hb. circuit_range COL_CALL_STACK_DEPTH_AFTER Ha.
  circuit_boolean COL_CALL_STACK_PUSH_PRESENT Hp. circuit_boolean COL_CALL_STACK_POP_PRESENT Hq.
  circuit_row ([(COL_CALL_STACK_DEPTH_AFTER, 1); (COL_CALL_STACK_DEPTH_BEFORE, -1);
    (COL_CALL_STACK_PUSH_PRESENT, -1); (COL_CALL_STACK_POP_PRESENT, 1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone in Hrow.
  destruct Hp, Hq; field_linear Hrow.
Qed.

Lemma full_end_pops_nonempty_context : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 ->
  w COL_PC_EDGE_KIND = 1 -> 0 < w COL_CALL_STACK_DEPTH_BEFORE ->
  w COL_CALL_STACK_POP_PRESENT = 1 /\ w COL_CALL_STACK_PUSH_PRESENT = 0 /\
  w COL_OUTPUT_CAPTURED = 0.
Proof.
  intros w Hprime Hfull Hchosen Hedge Hdepth.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_selected_is_program_row w COL_SEL_END Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  assert (HhaltBefore : w COL_HALTED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_HALTED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_HALTED_BEFORE). field_linear Hrow. }
  assert (HtrapBefore : w COL_TRAPPED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow. }
  circuit_boolean COL_HALTED Hhalt. circuit_boolean COL_TRAPPED_AFTER Htrap.
  circuit_boolean COL_DIV_TRAP Hdiv. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci.
  circuit_boolean COL_MEM_OOB Hmem. circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop.
  circuit_range COL_CALL_STACK_DEPTH_BEFORE HdepthBound.
  assert (Hsame : w COL_HALTED = w COL_TRAPPED_AFTER).
  { circuit_row ([(COL_HALTED, 1); (COL_TRAPPED_AFTER, -1)],
      [(COL_CALL_STACK_DEPTH_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. destruct Hhalt, Htrap; normalize_columns; try congruence; field_linear Hrow. }
  selected_circuit_row COL_SEL_END
    ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1);
      (COL_DIV_TRAP, -1); (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) HtrapRow.
  selected_circuit_row COL_SEL_END
    ([(COL_HALTED, 1); (COL_HALTED_BEFORE, -1); (COL_TURN_BOUNDARY, 1);
      (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
      (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) HedgeRow.
  normalize_columns. rewrite Hone, HtrapBefore in HtrapRow.
  rewrite Hone, HhaltBefore, Hturn, Hedge, Hsame in HedgeRow.
  assert (HtrapSum : w COL_TRAPPED_AFTER = w COL_DIV_TRAP + w COL_CALL_INDIRECT_IS_TRAP + w COL_MEM_OOB).
  { destruct Htrap, Hdiv, Hci, Hmem; field_linear HtrapRow. }
  assert (HpopOne : w COL_CALL_STACK_POP_PRESENT = 1).
  { normalize_columns. rewrite HtrapSum in HedgeRow. pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear HedgeRow. }
  split; [exact HpopOne | ]. split.
  - selected_row COL_SEL_END callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow.
  - circuit_row ([(COL_OUTPUT_CAPTURED, 1)], [(COL_ONE, 1); (COL_HALTED, -1)], ([] : SparseRow)) Hlive.
    circuit_row ([(COL_OUTPUT_CAPTURED, 1)], [(COL_TRAPPED_AFTER, 1)], ([] : SparseRow)) Hclean.
    expand_row Hlive. expand_row Hclean. normalize_columns. rewrite Hone, Hsame in Hlive.
    pose proof (value_bound w COL_OUTPUT_CAPTURED).
    destruct Htrap as [Htrap | Htrap]; normalize_columns; rewrite Htrap in Hlive, Hclean;
      field_linear Hlive || field_linear Hclean.
Qed.

Lemma full_end_stack : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 -> w COL_OUTPUT_CAPTURED = 0 ->
  w COL_STACK_WRITE0_ACTIVE = 0 /\ w COL_STACK_WRITE0_HI_ACTIVE = 0 /\
  w COL_SP_AFTER = w COL_SP_BEFORE.
Proof.
  intros w Hprime Hfull Hchosen Hcapture.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w COL_SEL_END Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero [Hgather _]]].
  destruct (full_program_row_quiet w COL_SEL_END Hprime Hfull Hin Hchosen)
    as [Hkinds Hhi Hlocal Houtput Hdiscard].
  assert (Hhost : w COL_HOST_CALL_ACTIVE = 0).
  { selected_row COL_SEL_END hostCallSupportRow Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_HOST_CALL_ACTIVE). field_linear Hrow. }
  assert (Hparams : w GHC_PARAMS = 0).
  { full_row hostCallParamsRow Hrow. rewrite Hhost in Hrow.
    pose proof (value_bound w GHC_PARAMS). field_linear Hrow. }
  assert (Hreads : w COL_STACK_READS = 0).
  { selected_row COL_SEL_END stackReadsArityRow Hrow. rewrite Hone, Hgather, Hparam in Hrow.
    pose proof (value_bound w COL_STACK_READS). field_linear Hrow. }
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { selected_row COL_SEL_END stackWritesArityRow Hrow. rewrite Hone, Hgather in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  split; [exact Hwrite | ]. split; [exact HwriteHi | ].
  full_row stackPointerBalanceRow Hrow. zero_gather_kinds Hkinds Hrow.
  rewrite Hone, Hreads, Hwrites, Hlocal, Houtput, Hdiscard, Hcapture, Hparams in Hrow.
  pose proof (value_bound w COL_SP_BEFORE). pose proof (value_bound w COL_SP_AFTER).
  field_linear Hrow.
Qed.

Lemma full_body_context_quiet : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> straightLineSelector chosen -> w chosen = 1 ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_CALL_STACK_POP_PRESENT = 0 /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE.
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen.
  pose proof (straight_line_selector_in chosen Hchoice) as Hin.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (proj1 (full_straight_line_local_gates w chosen Hprime Hfull Hchoice Hchosen)) as Hfbp.
  destruct Hchoice as [<- | [<- | [<- | [<- | []]]]];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0) by
        (selected_row chosen callStackPushRow Hrow;
         pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT); field_linear Hrow);
      assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0) by
        (selected_row chosen callStackPopSupportRow Hrow; rewrite Hone in Hrow;
         pose proof (value_bound w COL_CALL_STACK_POP_PRESENT); field_linear Hrow);
      split; [exact Hpush | ]; split; [exact Hpop | ]; split;
      [ pose proof (full_call_depth_balance w Hprime Hfull); lia
      | split; [ | exact Hfbp];
        selected_row chosen stackFrameBasePreservedRow Hrow; rewrite Hone, Hpush, Hpop in Hrow;
        pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE);
        pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER); field_linear Hrow ]
    end.
Qed.

Lemma full_end_opcode : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 ->
  w COL_OPCODE_CODE = OpcodeCatalog.CODE_END.
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  selected_circuit_row COL_SEL_END opcodeDecodeRow Hrow. rewrite Hone in Hrow.
  pose proof (value_bound w COL_OPCODE_CODE). unfold OpcodeCatalog.CODE_END. field_linear Hrow.
Qed.
