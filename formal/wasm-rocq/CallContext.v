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

(* A host suspend saves the same context in its own column, which a resume reads. *)
Definition contextHostColumn (memory : nat) : Column := List.nth (memory - 1) COL_HOST_CONTEXT 0%nat.

Definition contextSpec (memory : nat) (column : Column) : MemorySpec :=
  mkMemory memory Ram
    [mkPort [COL_CALL_STACK_ADDR] column (Write None) (When COL_CALL_STACK_PUSH_PRESENT);
     mkPort [COL_CALL_STACK_ADDR] column Read (When COL_CALL_STACK_POP_PRESENT);
     mkPort [COL_HOST_CONTEXT_ADDR] (contextHostColumn memory) (Write None) (When COL_HOST_SUSPEND);
     mkPort [COL_HOST_CONTEXT_ADDR] (contextHostColumn memory) Read (When COL_HOST_CONTEXT_READ)].

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
  w COL_HOST_SUSPEND = 0 -> w COL_HOST_CONTEXT_READ = 0 -> Replay initial (eventsForRow memories w) final ->
  forall address, final memory address = initial memory address.
Proof.
  intros initial final w memory column Hin Hpush Hsuspend Hread Hreplay address.
  eapply replay_preserves; [exact Hreplay | ].
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons.
  apply Forall_app. split; [apply read_only_row; exact Hbefore | ].
  apply Forall_app. split; [ | apply read_only_row; exact Hafter].
  unfold contextSpec. cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map].
  cbv [portEvent active]. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hpush, Hsuspend, Hread. cbn.
  let pop := eval compute in COL_CALL_STACK_POP_PRESENT in destruct (w pop =? 1); repeat constructor.
Qed.

(* A push writes each context memory only at the push address. *)
Lemma context_push_elsewhere : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories -> w COL_HOST_SUSPEND = 0 -> w COL_HOST_CONTEXT_READ = 0 ->
  Replay initial (eventsForRow memories w) final ->
  forall address, address <> [w COL_CALL_STACK_ADDR] -> final memory address = initial memory address.
Proof.
  intros initial final w memory column Hin Hsuspend Hread Hreplay address Haddress.
  eapply replay_preserves; [exact Hreplay | ].
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons.
  apply Forall_app. split; [apply read_only_row; exact Hbefore | ].
  apply Forall_app. split; [ | apply read_only_row; exact Hafter].
  unfold contextSpec. cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map].
  cbv [portEvent active]. cbn [activation addressColumns valueColumn portKind map option_map].
  normalize_columns. rewrite Hsuspend, Hread.
  let push := eval compute in COL_CALL_STACK_PUSH_PRESENT in destruct (w push =? 1);
  let pop := eval compute in COL_CALL_STACK_POP_PRESENT in destruct (w pop =? 1);
  repeat first [apply Forall_nil | apply Forall_cons | exact I | (cbn; exact I) | (cbn; right; congruence)].
Qed.

Lemma context_push_records : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories -> w COL_CALL_STACK_PUSH_PRESENT = 1 ->
  w COL_HOST_SUSPEND = 0 -> w COL_HOST_CONTEXT_READ = 0 -> Replay initial (eventsForRow memories w) final ->
  final memory [w COL_CALL_STACK_ADDR] = Some (w column).
Proof.
  intros initial final w memory column Hin Hpush Hsuspend Hread Hreplay.
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [before [_ Hrest]].
  destruct (replay_app_inv _ _ _ _ Hrest) as [after [Hport Htail]].
  rewrite (replay_preserves _ _ _ _ _ Htail (read_only_row _ _ _ _ Hafter)).
  unfold contextSpec in Hport.
  cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map] in Hport.
  cbv [portEvent active] in Hport. cbn [activation addressColumns valueColumn portKind map option_map] in Hport.
  normalize_columns. rewrite Hpush, Hsuspend, Hread in Hport. cbn in Hport.
  inversion Hport; subst. match goal with H : EventStep _ (WriteEvent _ _ _ _) _ |- _ => inversion H; subst end.
  match goal with H : Replay _ _ after |- _ =>
    rewrite (replay_preserves _ _ _ _ _ H); [apply update_same | ] end.
  let pop := eval compute in COL_CALL_STACK_POP_PRESENT in destruct (w pop =? 1); repeat constructor.
Qed.

Lemma context_pop_reads : forall initial final (w : Witness) memory column,
  In (memory, column) contextMemories ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 -> w COL_CALL_STACK_POP_PRESENT = 1 ->
  w COL_HOST_SUSPEND = 0 -> w COL_HOST_CONTEXT_READ = 0 -> Replay initial (eventsForRow memories w) final ->
  initial memory [w COL_CALL_STACK_ADDR] = Some (w column).
Proof.
  intros initial final w memory column Hin Hpush Hpop Hsuspend Hread Hreplay.
  destruct (context_catalog_split memory column Hin) as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit, events_for_row_app, events_for_row_cons in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [before [Hprefix Hrest]].
  destruct (replay_app_inv _ _ _ _ Hrest) as [after [Hport Htail]].
  unfold contextSpec in Hport.
  cbn [memoryId ports flat_map portEvent active activation addressColumns valueColumn portKind map] in Hport.
  cbv [portEvent active] in Hport. cbn [activation addressColumns valueColumn portKind map option_map] in Hport.
  normalize_columns. rewrite Hpush, Hpop, Hsuspend, Hread in Hport. cbn in Hport.
  inversion Hport; subst. match goal with H : EventStep _ (ReadEvent _ _ _) _ |- _ => inversion H; subst end.
  rewrite <- (replay_preserves _ _ _ _ _ Hprefix (read_only_row _ _ _ _ Hbefore)). assumption.
Qed.

(* On a program row, the depth moves only by a push or a pop. *)
Lemma full_call_depth_balance : forall (w : Witness) selector,
  fieldPrime -> FullRelation.satFull w -> In selector selectorColumns -> w selector = 1 ->
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE +
    w COL_CALL_STACK_PUSH_PRESENT - w COL_CALL_STACK_POP_PRESENT.
Proof.
  intros w selector Hprime Hfull Hin Hselected.
  pose proof (full_selected_is_program_row w selector Hprime Hfull Hin Hselected) as Hprogram.
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [Hsuspend [Hresume _]].
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_CALL_STACK_DEPTH_BEFORE Hb. circuit_range COL_CALL_STACK_DEPTH_AFTER Ha.
  circuit_boolean COL_CALL_STACK_PUSH_PRESENT Hp. circuit_boolean COL_CALL_STACK_POP_PRESENT Hq.
  circuit_row ([(COL_CALL_STACK_DEPTH_AFTER, 1); (COL_CALL_STACK_DEPTH_BEFORE, -1);
    (COL_CALL_STACK_PUSH_PRESENT, -1); (COL_CALL_STACK_POP_PRESENT, 1);
    (COL_HOST_SUSPEND, -1); (COL_HOST_RESUME, 1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone, Hsuspend, Hresume in Hrow.
  destruct Hp, Hq; field_linear Hrow.
Qed.

(* A function-ending END exits even at depth zero, where it halts instead
   of popping. A block END has a different ROM edge kind. *)
Lemma full_function_end_exits : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 ->
  w COL_PC_EDGE_KIND = 1 ->
  w COL_CALL_STACK_POP_PRESENT = 1 \/ w COL_ACTIVATION_FINISHED_AFTER = 1.
Proof.
  intros w Hprime Hfull Hchosen Hedge.
  circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop. circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hhalt.
  destruct Hpop as [Hpop | Hpop]; [ | left; exact Hpop].
  destruct Hhalt as [Hhalt | Hhalt]; [ | right; exact Hhalt].
  exfalso.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_selected_is_program_row w COL_SEL_END Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  assert (Hbefore : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  selected_circuit_row COL_SEL_END
    ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
      (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
      (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) HedgeRow.
  normalize_columns. rewrite Hone, Hhalt, Hbefore, Hclear, Hpop, Hedge in HedgeRow.
  circuit_boolean COL_DIV_TRAP Hdiv. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci.
  circuit_boolean COL_MEM_OOB Hmem.
  destruct Hdiv, Hci, Hmem; field_linear HedgeRow.
Qed.

(* BLOCK and br_if rows never touch the call stack. *)
Lemma full_block_selector_quiet : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> (chosen = COL_SEL_BLOCK \/ chosen = COL_SEL_BR_IF_EQZ) -> w chosen = 1 ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_CALL_STACK_POP_PRESENT = 0.
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [-> | ->]; vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  destruct Hkind as [-> | ->];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      split;
      [ selected_row chosen callStackPushRow Hrow; pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT); field_linear Hrow
      | selected_row chosen callStackPopSupportRow Hrow; rewrite Hone in Hrow;
        pose proof (value_bound w COL_CALL_STACK_POP_PRESENT); field_linear Hrow ]
    end.
Qed.

(* A block's END (a static edge) in any frame: without a halt, the trap accounting
   leaves no trap flag set, so the row neither pops nor pushes a call context. *)
Lemma full_block_end_quiet : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 ->
  w COL_PC_EDGE_KIND = 0 -> w COL_ACTIVATION_FINISHED_AFTER = 0 ->
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_CALL_STACK_POP_PRESENT = 0.
Proof.
  intros w Hprime Hfull Hchosen Hedge Hhalt.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_selected_is_program_row w COL_SEL_END Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  assert (Hbefore : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { selected_row COL_SEL_END callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  split; [exact Hpush | ].
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1); (COL_TRAPPED_BEFORE, 1);
    (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hstate.
  expand_row Hstate. normalize_columns.
  selected_circuit_row COL_SEL_END
    ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Htrap.
  selected_circuit_row COL_SEL_END
    ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
      (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
      (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) Hedgerow.
  normalize_columns. rewrite Hone in Hstate, Htrap, Hedgerow.
  circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop. circuit_boolean COL_DIV_TRAP Hdiv.
  circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci. circuit_boolean COL_MEM_OOB Hmem.
  circuit_boolean COL_HOST_EVENT_EXIT_LATCH Hlatch. circuit_boolean COL_TRAPPED_AFTER Hta.
  circuit_boolean COL_TRAPPED_BEFORE Htb.
  normalize_columns.
  apply Z.mod_divide in Hstate; [ | unfold modulus; lia ]. destruct Hstate as [k1 Hstate].
  apply Z.mod_divide in Htrap; [ | unfold modulus; lia ]. destruct Htrap as [k2 Htrap].
  apply Z.mod_divide in Hedgerow; [ | unfold modulus; lia ]. destruct Hedgerow as [k3 Hedgerow].
  unfold modulus in *. lia.
Qed.

(* A function-ending END of a callee, in a run whose activation does not finish:
   it pops its caller's context. *)
Lemma full_end_pops_nonempty_context : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 ->
  w COL_PC_EDGE_KIND = 1 -> w COL_ACTIVATION_FINISHED_AFTER = 0 ->
  w COL_CALL_STACK_POP_PRESENT = 1 /\ w COL_CALL_STACK_PUSH_PRESENT = 0 /\
  w COL_OUTPUT_CAPTURED = 0.
Proof.
  intros w Hprime Hfull Hchosen Hedge Hlive.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_selected_is_program_row w COL_SEL_END Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  assert (HhaltBefore : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  assert (HtrapBefore : w COL_TRAPPED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow. }
  circuit_boolean COL_TRAPPED_AFTER Htrap. circuit_boolean COL_HOST_EVENT_EXIT_LATCH Hlatch.
  circuit_boolean COL_DIV_TRAP Hdiv. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci.
  circuit_boolean COL_MEM_OOB Hmem.
  (* An unfinished activation has not trapped. *)
  assert (HtrapAfter : w COL_TRAPPED_AFTER = 0).
  { circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
      (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hlive, HhaltBefore, HtrapBefore, Hclear in Hrow.
    destruct Htrap as [Ht | Ht]; [exact Ht | exfalso].
    destruct Hlatch as [Hl | Hl]; normalize_columns; rewrite Ht, Hl in Hrow; field_linear Hrow. }
  selected_circuit_row COL_SEL_END
    ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1);
      (COL_DIV_TRAP, -1); (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) HtrapRow.
  selected_circuit_row COL_SEL_END
    ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
      (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
      (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1)],
      [(COL_ONE, 1)], ([] : SparseRow)) HedgeRow.
  normalize_columns. rewrite Hone, HtrapBefore, HtrapAfter in HtrapRow.
  rewrite Hone, Hlive, HhaltBefore, Hclear, Hedge in HedgeRow.
  assert (Htraps : w COL_DIV_TRAP = 0 /\ w COL_CALL_INDIRECT_IS_TRAP = 0 /\ w COL_MEM_OOB = 0).
  { destruct Hdiv, Hci, Hmem; normalize_columns; repeat split; try assumption; exfalso; field_linear HtrapRow. }
  destruct Htraps as [Hd [Hc Hm]].
  assert (HpopOne : w COL_CALL_STACK_POP_PRESENT = 1).
  { normalize_columns. rewrite Hd, Hc, Hm in HedgeRow. pose proof (value_bound w COL_CALL_STACK_POP_PRESENT).
    field_linear HedgeRow. }
  split; [exact HpopOne | ]. split.
  - selected_row COL_SEL_END callStackPushRow Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow.
  - circuit_row ([(COL_OUTPUT_CAPTURED, 1)], [(COL_ONE, 1); (COL_ACTIVATION_FINISHED_AFTER, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hlive in Hrow.
    pose proof (value_bound w COL_OUTPUT_CAPTURED). field_linear Hrow.
Qed.

(* A callee's END writes nothing; it lowers sp to the callee's frame base plus
   its results and seeds the move of those results down to it. *)
Lemma full_end_return_stack : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_END = 1 -> w COL_CALL_STACK_POP_PRESENT = 1 ->
  w COL_STACK_WRITE0_ACTIVE = 0 /\ w COL_STACK_WRITE0_HI_ACTIVE = 0 /\ w COL_LOCAL_WRITE_ENABLED = 0 /\
  w COL_GATHER_LOCAL_WRITE_LO = 0 /\ w COL_GATHER_LOCAL_WRITE = 0 /\
  w COL_FRAME_MOVE_REMAINING_AFTER = w COL_SP_BEFORE - w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_FRAME_MOVE_DISTANCE_AFTER = w COL_STACK_FRAME_BASE_BEFORE - w COL_LOCALS_FBP_BEFORE /\
  w COL_SP_AFTER = w COL_LOCALS_FBP_BEFORE + w COL_FRAME_MOVE_REMAINING_AFTER.
Proof.
  intros w Hprime Hfull Hchosen Hpop.
  assert (Hin : In COL_SEL_END selectorColumns) by (vm_compute; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w COL_SEL_END Hprime Hfull Hin Hchosen) as [Hparam [Hzero Hgather]].
  destruct (full_program_row_quiet w COL_SEL_END Hprime Hfull Hin Hchosen) as [Hkinds Hhi Hlocal Houtput].
  destruct (full_no_gather_local_ports w Hfull Hgather) as [HgatherLocal HgatherLocalLo].
  assert (Hwrites : w COL_STACK_WRITES = 0).
  { selected_row COL_SEL_END stackWritesArityRow Hrow. rewrite Hone, Hgather, Hparam, Hzero in Hrow.
    pose proof (value_bound w COL_STACK_WRITES). field_linear Hrow. }
  assert (Hwrite : w COL_STACK_WRITE0_ACTIVE = 0).
  { full_row stackWritesActiveRow Hrow. rewrite Hone, Hwrites in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_ACTIVE). field_linear Hrow. }
  assert (HwriteHi : w COL_STACK_WRITE0_HI_ACTIVE = 0).
  { full_row stackWriteHighActiveRow Hrow. rewrite Hone, Hwrite, Hhi in Hrow.
    pose proof (value_bound w COL_STACK_WRITE0_HI_ACTIVE). field_linear Hrow. }
  assert (HlocalWrite : w COL_LOCAL_WRITE_ENABLED = 0).
  { selected_row COL_SEL_END localWriteGateIndex Hrow. rewrite Hone in Hrow.
    pose proof (value_bound w COL_LOCAL_WRITE_ENABLED). field_linear Hrow. }
  circuit_range COL_SP_BEFORE Hspb. circuit_range COL_SP_AFTER Hspa.
  circuit_range COL_STACK_FRAME_BASE_BEFORE Hbase. circuit_range COL_LOCALS_FBP_BEFORE Hfbp.
  circuit_range COL_FRAME_MOVE_REMAINING_AFTER Hrem. circuit_range COL_FRAME_MOVE_DISTANCE_AFTER Hdist.
  circuit_row ([(COL_CALL_STACK_POP_PRESENT, 1)], [(COL_FRAME_MOVE_REMAINING_AFTER, 1); (COL_SP_BEFORE, -1);
    (COL_STACK_FRAME_BASE_BEFORE, 1)], ([] : SparseRow)) Hcount.
  circuit_row ([(COL_CALL_STACK_POP_PRESENT, 1)], [(COL_FRAME_MOVE_DISTANCE_AFTER, 1);
    (COL_STACK_FRAME_BASE_BEFORE, -1); (COL_LOCALS_FBP_BEFORE, 1)], ([] : SparseRow)) Hdistance.
  circuit_row ([(COL_CALL_STACK_POP_PRESENT, 1)], [(COL_SP_AFTER, 1); (COL_LOCALS_FBP_BEFORE, -1);
    (COL_FRAME_MOVE_REMAINING_AFTER, -1)], ([] : SparseRow)) Hsp.
  expand_row Hcount. expand_row Hdistance. expand_row Hsp. rewrite Hpop in Hcount, Hdistance, Hsp.
  repeat split; try assumption; [field_linear Hcount | field_linear Hdistance | field_linear Hsp].
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
      destruct (full_body_frame_sp_drop_idle w chosen Hprime Hfull Hin Hchosen
        ltac:(vm_compute; intuition discriminate)) as [_ _ Hguest Hturn _];
      split; [exact Hpush | ]; split; [exact Hpop | ]; split;
      [ pose proof (full_call_depth_balance w chosen Hprime Hfull Hin Hchosen); lia
      | split; [ | exact Hfbp];
        selected_row chosen stackFrameBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hguest, Hturn, Hpop in Hrow;
        pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE);
        pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER); field_linear Hrow ]
    end.
Qed.
