From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog CircuitFacts
  InactiveStackBindings RangeObligations TraceRefinement.
Import ListNotations.
Open Scope Z_scope.

Definition memoryValueColumn (column : nat) : bool :=
 existsb (fun memory => existsb (fun port =>
   Nat.eqb column (valueColumn port) ||
   match portKind port with
   | Write (Some prior) => Nat.eqb column prior
   | _ => false
   end) (ports memory)) memories.

Definition memoryRangeObligation (w : Witness) (column : nat) : Prop :=
 memoryValueColumn column = true /\ rangeObligation w (column, Repairable 32%nat).

Definition returnPcMemory : MemorySpec :=
 match find (fun memory => Nat.eqb (memoryId memory) callStackReturnPcId) memories with
 | Some memory => memory
 | None => mkMemory callStackReturnPcId Ram []
 end.

Lemma return_pc_memory_bound : forall w kind gate,
 In (kind, gate) [(Write None, COL_CALL_STACK_PUSH_PRESENT); (Read, COL_CALL_STACK_POP_PRESENT)] ->
 boundedMemoryEvents (eventsForRow memories w) -> w gate = 1 ->
 u32Value (w COL_CALL_STACK_RETURN_PC_VALUE).
Proof.
 intros w kind gate Hin Hbounds Hactive.
 eapply bounded_port_value with
   (specification := returnPcMemory)
   (port := mkPort [COL_CALL_STACK_ADDR] COL_CALL_STACK_RETURN_PC_VALUE kind (When gate)).
 - exact Hbounds.
 - vm_compute; tauto.
 - destruct Hin as [H|[H|[]]]; inversion H; subst; vm_compute; tauto.
 - cbn [active activation]. rewrite Hactive. reflexivity.
Qed.

Lemma full_return_pc_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_CALL_STACK_RETURN_PC_VALUE, Repairable 32%nat).
Proof.
 intros w Hprime Hfull Hbounds. unfold rangeObligation.
 destruct (Z_lt_ge_dec (w COL_CALL_STACK_RETURN_PC_VALUE) 4294967296) as [Hsmall|Hlarge].
 - exists w. split; [exact Hfull|]. split; [pose proof (value_bound w COL_CALL_STACK_RETURN_PC_VALUE);cbn;lia|].
   repeat split; reflexivity.
 - circuit_boolean COL_CALL_STACK_PUSH_PRESENT Hpush.
   circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop.
   assert (Hzpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
   { destruct Hpush as [H|H]; [exact H|].
     pose proof (return_pc_memory_bound w (Write None) COL_CALL_STACK_PUSH_PRESENT
       ltac:(left;reflexivity) Hbounds H). unfold u32Value in *; lia. }
   assert (Hzpop : w COL_CALL_STACK_POP_PRESENT = 0).
   { destruct Hpop as [H|H]; [exact H|].
     pose proof (return_pc_memory_bound w Read COL_CALL_STACK_POP_PRESENT
       ltac:(right;left;reflexivity) Hbounds H). unfold u32Value in *; lia. }
   change (exists w', FullRelation.satFull w' /\
     0 <= w' COL_CALL_STACK_RETURN_PC_VALUE < 4294967296 /\
     eventsForRow memories w' = eventsForRow memories w /\
     map (fun '(a,b) => (w' a,w' b)) continuityLinks =
     map (fun '(a,b) => (w a,w b)) continuityLinks /\
     (forall c, c <> COL_CALL_STACK_RETURN_PC_VALUE -> w' c = w c)).
   finish_call_stack_repair COL_CALL_STACK_RETURN_PC_VALUE.
Qed.

Ltac memory_value_inactive column gate :=
 lazymatch goal with
 | Hfull : FullRelation.satFull ?w,
   Hbounds : boundedMemoryEvents (eventsForRow memories ?w) |- _ =>
 let selected := eval vm_compute in
   (find (fun pair => Nat.eqb (valueColumn (snd pair)) column &&
      match activation (snd pair) with When g => Nat.eqb g gate | _ => false end)
     (flat_map (fun m => map (fun p => (m,p)) (ports m)) memories)) in
 lazymatch selected with
 | Some (?memory, ?chosenPort) =>
   let Hb := fresh "Hb" in let Hz := fresh "Hz" in
   circuit_boolean gate Hb;
   assert (Hz : w gate = 0) by
     (destruct Hb as [Hb|Hb]; [exact Hb|];
      let Hr := fresh "Hr" in
      assert (Hr : u32Value (w column)) by
        (eapply bounded_port_value with (specification := memory) (port := chosenPort);
         [exact Hbounds | vm_compute; tauto | vm_compute; tauto |
          change (Z.eqb (w gate) 1 = true); rewrite Hb; reflexivity]);
      unfold u32Value in Hr; lia);
   clear Hb
 end
 end.

Ltac start_memory_repair column :=
 intros w Hprime Hfull Hbounds; unfold rangeObligation;
 change (exists w', FullRelation.satFull w' /\
   0 <= w' column < 4294967296 /\
   eventsForRow memories w' = eventsForRow memories w /\
   map (fun '(a,b) => (w' a,w' b)) continuityLinks =
   map (fun '(a,b) => (w a,w b)) continuityLinks /\
   (forall c, c <> column -> w' c = w c));
 destruct (Z_lt_ge_dec (w column) 4294967296) as [Hsmall|Hlarge];
 [exists w; split; [exact Hfull|]; split;
  [pose proof (value_bound w column); lia|]; repeat split; reflexivity | ].

Ltac finish_memory_repair column repairGates :=
 lazymatch goal with Hfull : FullRelation.satFull ?w |- _ =>
 exists (zeroColumn column w); split;
 [ eapply private_full_repair with (gates := repairGates);
   [vm_compute; discriminate | vm_compute; reflexivity |
    let rec prove_gates := lazymatch goal with
    | |- Forall _ [] => constructor
    | |- Forall _ (_ :: _) => apply Forall_cons;
      [try assumption; cbn [eval fold_right]; normalize_columns;
       repeat match goal with H : _ = 0 |- _ => progress rewrite H end;
       reflexivity | prove_gates]
    end in prove_gates
    | exact Hfull]
 | split; [rewrite zero_column_value; lia|]; split;
   [repair_memory_events column
   | split; [reflexivity|apply zero_column_same]]]
 end.

Lemma full_host_frame_0_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w ((List.nth 0%nat COL_HOST_CONTEXT_FRAME 0%nat), Repairable 32%nat).
Proof.
 start_memory_repair (List.nth 0%nat COL_HOST_CONTEXT_FRAME 0%nat).
 memory_value_inactive (List.nth 0%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_SUSPEND.
 memory_value_inactive (List.nth 0%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_RESUME.
 finish_memory_repair (List.nth 0%nat COL_HOST_CONTEXT_FRAME 0%nat)
   [[(COL_HOST_SUSPEND,1)]; [(COL_HOST_RESUME,1)]].
Qed.

Lemma full_host_frame_1_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w ((List.nth 1%nat COL_HOST_CONTEXT_FRAME 0%nat), Repairable 32%nat).
Proof.
 start_memory_repair (List.nth 1%nat COL_HOST_CONTEXT_FRAME 0%nat).
 memory_value_inactive (List.nth 1%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_SUSPEND.
 memory_value_inactive (List.nth 1%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_RESUME.
 finish_memory_repair (List.nth 1%nat COL_HOST_CONTEXT_FRAME 0%nat)
   [[(COL_HOST_SUSPEND,1)]; [(COL_HOST_RESUME,1)]].
Qed.

Lemma full_host_frame_2_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w ((List.nth 2%nat COL_HOST_CONTEXT_FRAME 0%nat), Repairable 32%nat).
Proof.
 start_memory_repair (List.nth 2%nat COL_HOST_CONTEXT_FRAME 0%nat).
 memory_value_inactive (List.nth 2%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_SUSPEND.
 memory_value_inactive (List.nth 2%nat COL_HOST_CONTEXT_FRAME 0%nat) COL_HOST_RESUME.
 finish_memory_repair (List.nth 2%nat COL_HOST_CONTEXT_FRAME 0%nat)
   [[(COL_HOST_SUSPEND,1)]; [(COL_HOST_RESUME,1)]].
Qed.

Lemma full_host_sp_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_HOST_CONTEXT_SP, Repairable 32%nat).
Proof.
 start_memory_repair COL_HOST_CONTEXT_SP.
 memory_value_inactive COL_HOST_CONTEXT_SP COL_HOST_SUSPEND.
 memory_value_inactive COL_HOST_CONTEXT_SP COL_HOST_CONTEXT_READ.
 circuit_boolean COL_HOST_NESTED_ENTRY Bentry.
 circuit_boolean COL_HOST_RESUME Bresume.
 circuit_row ([(COL_HOST_CONTEXT_READ,1); (COL_HOST_NESTED_ENTRY,-1); (COL_HOST_RESUME,-1)],
   [(COL_ONE,1)], ([] : SparseRow)) Hr.
 assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
 expand_row Hr. rewrite Hone in Hr.
 assert (Hzentry : w COL_HOST_NESTED_ENTRY = 0 /\ w COL_HOST_RESUME = 0)
   by (repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr end; field_linear Hr).
 destruct Hzentry as [Hzentry Hzresume].
 finish_memory_repair COL_HOST_CONTEXT_SP
   [[(COL_HOST_SUSPEND,1)]; [(COL_HOST_NESTED_ENTRY,1)]; [(COL_HOST_RESUME,1)]].
Qed.

Lemma full_host_event_base_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_HOST_EVENT_BASE, Repairable 32%nat).
Proof.
 start_memory_repair COL_HOST_EVENT_BASE.
 memory_value_inactive COL_HOST_EVENT_BASE COL_HOST_CALL_ACTIVE.
 memory_value_inactive COL_HOST_EVENT_BASE COL_HOST_RESUME.
 memory_value_inactive COL_HOST_EVENT_BASE COL_HOST_EVENT_EXIT_LATCH.
 memory_value_inactive COL_HOST_EVENT_BASE COL_TURN_BOUNDARY.
 finish_memory_repair COL_HOST_EVENT_BASE
   [[(COL_HOST_CALL_ACTIVE,1)]; [(COL_HOST_RESUME,1)];
    [(COL_HOST_EVENT_EXIT_LATCH,1)]; [(COL_TURN_BOUNDARY,1)]].
Qed.

Definition memoryOffsetGates : list SparseRow := [[(COL_SEL_I32_LOAD,1); (COL_SEL_I32_LOAD8_S,1); (COL_SEL_I32_LOAD8_U,1); (COL_SEL_I32_LOAD16_S,1); (COL_SEL_I32_LOAD16_U,1); (COL_SEL_I64_LOAD,1); (COL_SEL_I32_STORE,1); (COL_SEL_I32_STORE8,1); (COL_SEL_I32_STORE16,1); (COL_SEL_I64_STORE,1); (COL_SEL_I64_STORE8,1); (COL_SEL_I64_STORE16,1); (COL_SEL_I64_STORE32,1); (COL_SEL_I64_LOAD8_U,1); (COL_SEL_I64_LOAD16_U,1); (COL_SEL_I64_LOAD32_U,1); (COL_SEL_I64_LOAD8_S,1); (COL_SEL_I64_LOAD16_S,1); (COL_SEL_I64_LOAD32_S,1)]].

Lemma full_memory_offset_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_LINEAR_MEM_IMM_OFFSET, Repairable 32%nat).
Proof.
 start_memory_repair COL_LINEAR_MEM_IMM_OFFSET.
 memory_value_inactive COL_LINEAR_MEM_IMM_OFFSET COL_LINEAR_MEM_USE_LANE0.
 circuit_boolean COL_SEL_I32_LOAD B0.
 assert (R0 : 0 <= w COL_SEL_I32_LOAD <= 1) by lia; clear B0.
 circuit_boolean COL_SEL_I32_LOAD8_S B1.
 assert (R1 : 0 <= w COL_SEL_I32_LOAD8_S <= 1) by lia; clear B1.
 circuit_boolean COL_SEL_I32_LOAD8_U B2.
 assert (R2 : 0 <= w COL_SEL_I32_LOAD8_U <= 1) by lia; clear B2.
 circuit_boolean COL_SEL_I32_LOAD16_S B3.
 assert (R3 : 0 <= w COL_SEL_I32_LOAD16_S <= 1) by lia; clear B3.
 circuit_boolean COL_SEL_I32_LOAD16_U B4.
 assert (R4 : 0 <= w COL_SEL_I32_LOAD16_U <= 1) by lia; clear B4.
 circuit_boolean COL_SEL_I64_LOAD B5.
 assert (R5 : 0 <= w COL_SEL_I64_LOAD <= 1) by lia; clear B5.
 circuit_boolean COL_SEL_I32_STORE B6.
 assert (R6 : 0 <= w COL_SEL_I32_STORE <= 1) by lia; clear B6.
 circuit_boolean COL_SEL_I32_STORE8 B7.
 assert (R7 : 0 <= w COL_SEL_I32_STORE8 <= 1) by lia; clear B7.
 circuit_boolean COL_SEL_I32_STORE16 B8.
 assert (R8 : 0 <= w COL_SEL_I32_STORE16 <= 1) by lia; clear B8.
 circuit_boolean COL_SEL_I64_STORE B9.
 assert (R9 : 0 <= w COL_SEL_I64_STORE <= 1) by lia; clear B9.
 circuit_boolean COL_SEL_I64_STORE8 B10.
 assert (R10 : 0 <= w COL_SEL_I64_STORE8 <= 1) by lia; clear B10.
 circuit_boolean COL_SEL_I64_STORE16 B11.
 assert (R11 : 0 <= w COL_SEL_I64_STORE16 <= 1) by lia; clear B11.
 circuit_boolean COL_SEL_I64_STORE32 B12.
 assert (R12 : 0 <= w COL_SEL_I64_STORE32 <= 1) by lia; clear B12.
 circuit_boolean COL_SEL_I64_LOAD8_U B13.
 assert (R13 : 0 <= w COL_SEL_I64_LOAD8_U <= 1) by lia; clear B13.
 circuit_boolean COL_SEL_I64_LOAD16_U B14.
 assert (R14 : 0 <= w COL_SEL_I64_LOAD16_U <= 1) by lia; clear B14.
 circuit_boolean COL_SEL_I64_LOAD32_U B15.
 assert (R15 : 0 <= w COL_SEL_I64_LOAD32_U <= 1) by lia; clear B15.
 circuit_boolean COL_SEL_I64_LOAD8_S B16.
 assert (R16 : 0 <= w COL_SEL_I64_LOAD8_S <= 1) by lia; clear B16.
 circuit_boolean COL_SEL_I64_LOAD16_S B17.
 assert (R17 : 0 <= w COL_SEL_I64_LOAD16_S <= 1) by lia; clear B17.
 circuit_boolean COL_SEL_I64_LOAD32_S B18.
 assert (R18 : 0 <= w COL_SEL_I64_LOAD32_S <= 1) by lia; clear B18.
 circuit_row ([(COL_LINEAR_MEM_USE_LANE0,1); (COL_SEL_I32_LOAD,-1); (COL_SEL_I32_LOAD8_S,-1); (COL_SEL_I32_LOAD8_U,-1); (COL_SEL_I32_LOAD16_S,-1); (COL_SEL_I32_LOAD16_U,-1); (COL_SEL_I64_LOAD,-1); (COL_SEL_I32_STORE,-1); (COL_SEL_I32_STORE8,-1); (COL_SEL_I32_STORE16,-1); (COL_SEL_I64_STORE,-1); (COL_SEL_I64_STORE8,-1); (COL_SEL_I64_STORE16,-1); (COL_SEL_I64_STORE32,-1); (COL_SEL_I64_LOAD8_U,-1); (COL_SEL_I64_LOAD16_U,-1); (COL_SEL_I64_LOAD32_U,-1); (COL_SEL_I64_LOAD8_S,-1); (COL_SEL_I64_LOAD16_S,-1); (COL_SEL_I64_LOAD32_S,-1)], [(COL_ONE,1)], ([] : SparseRow)) Hr.
 assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
 expand_row Hr. rewrite Hone in Hr.
 assert (Hzsum : eval w [(COL_SEL_I32_LOAD,1); (COL_SEL_I32_LOAD8_S,1); (COL_SEL_I32_LOAD8_U,1); (COL_SEL_I32_LOAD16_S,1); (COL_SEL_I32_LOAD16_U,1); (COL_SEL_I64_LOAD,1); (COL_SEL_I32_STORE,1); (COL_SEL_I32_STORE8,1); (COL_SEL_I32_STORE16,1); (COL_SEL_I64_STORE,1); (COL_SEL_I64_STORE8,1); (COL_SEL_I64_STORE16,1); (COL_SEL_I64_STORE32,1); (COL_SEL_I64_LOAD8_U,1); (COL_SEL_I64_LOAD16_U,1); (COL_SEL_I64_LOAD32_U,1); (COL_SEL_I64_LOAD8_S,1); (COL_SEL_I64_LOAD16_S,1); (COL_SEL_I64_LOAD32_S,1)] = 0).
 { cbn [eval fold_right]; normalize_columns.
   repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr end.
   field_linear Hr. }
 let gates := eval cbv delta [memoryOffsetGates] in memoryOffsetGates in
 finish_memory_repair COL_LINEAR_MEM_IMM_OFFSET gates.
Qed.

Lemma full_function_locals_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_ENTERED_FUNCTION_NUM_LOCALS, Repairable 32%nat).
Proof.
 start_memory_repair COL_ENTERED_FUNCTION_NUM_LOCALS.
 memory_value_inactive COL_ENTERED_FUNCTION_NUM_LOCALS COL_GUEST_ENTRY_ACTIVE.
 memory_value_inactive COL_ENTERED_FUNCTION_NUM_LOCALS COL_TURN_BOUNDARY.
 finish_memory_repair COL_ENTERED_FUNCTION_NUM_LOCALS
   [[(COL_GUEST_ENTRY_ACTIVE,1); (COL_TURN_BOUNDARY,1)]].
Qed.

Lemma full_global_index_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_GLOBAL_INDEX, Repairable 32%nat).
Proof.
 start_memory_repair COL_GLOBAL_INDEX.
 memory_value_inactive COL_GLOBAL_INDEX COL_PROGRAM_GLOBAL_INDEX_ACTIVE.
 circuit_boolean COL_SEL_GLOBAL_GET Bget.
 circuit_boolean COL_SEL_GLOBAL_SET Bset.
 circuit_row ([(COL_PROGRAM_GLOBAL_INDEX_ACTIVE,1); (COL_SEL_GLOBAL_GET,-1); (COL_SEL_GLOBAL_SET,-1)],
   [(COL_ONE,1)], ([] : SparseRow)) Hr.
 assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
 expand_row Hr. rewrite Hone in Hr.
 assert (HglobalOff : w COL_SEL_GLOBAL_GET = 0 /\ w COL_SEL_GLOBAL_SET = 0)
   by (repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr end; field_linear Hr).
 destruct HglobalOff as [Hget Hset].
 finish_memory_repair COL_GLOBAL_INDEX ([] : list SparseRow).
Qed.

Lemma full_br_table_len_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_BR_TABLE_LEN, Repairable 32%nat).
Proof.
 start_memory_repair COL_BR_TABLE_LEN.
 memory_value_inactive COL_BR_TABLE_LEN COL_SEL_BR_TABLE.
 finish_memory_repair COL_BR_TABLE_LEN [[(COL_SEL_BR_TABLE,1)]].
Qed.

Lemma full_table_size_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_TABLE_SIZE, Repairable 32%nat).
Proof.
 start_memory_repair COL_TABLE_SIZE.
 memory_value_inactive COL_TABLE_SIZE COL_TABLE_SIZE_READ_ENABLED.
 circuit_boolean COL_SEL_TABLE_SIZE Bsize.
 circuit_boolean COL_SEL_TABLE_GET Bget.
 circuit_boolean COL_SEL_TABLE_SET Bset.
 circuit_boolean COL_SEL_CALL_INDIRECT_RESOLVE Bresolve.
 circuit_row ([(COL_TABLE_SIZE_READ_ENABLED,1); (COL_SEL_TABLE_SIZE,-1); (COL_SEL_TABLE_GET,-1);
   (COL_SEL_TABLE_SET,-1); (COL_SEL_CALL_INDIRECT_RESOLVE,-1)], [(COL_ONE,1)], ([] : SparseRow)) Hr.
 assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
 expand_row Hr. rewrite Hone in Hr.
 assert (Hoff : w COL_SEL_TABLE_SIZE = 0 /\ w COL_SEL_TABLE_GET = 0 /\
   w COL_SEL_TABLE_SET = 0 /\ w COL_SEL_CALL_INDIRECT_RESOLVE = 0)
   by (repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr end; field_linear Hr).
 destruct Hoff as [Hsize [Hget [Hset Hresolve]]].
 finish_memory_repair COL_TABLE_SIZE
   [[(COL_SEL_TABLE_SIZE,1)]; [(COL_SEL_TABLE_GET,1); (COL_SEL_TABLE_SET,1); (COL_SEL_CALL_INDIRECT_RESOLVE,1)]].
Qed.

Ltac local_writes_inactive :=
 let Bset := fresh "Bset" in let Btee := fresh "Btee" in let Hr := fresh "Hr" in
 circuit_boolean COL_SEL_LOCAL_SET Bset;
 circuit_boolean COL_SEL_LOCAL_TEE Btee;
 circuit_row ([(COL_LOCAL_WRITE_ENABLED,1); (COL_SEL_LOCAL_SET,-1); (COL_SEL_LOCAL_TEE,-1)],
   [(COL_ONE,1)], ([] : SparseRow)) Hr;
 lazymatch goal with Hfull : FullRelation.satFull ?w |- _ =>
 assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull);
 expand_row Hr; rewrite Hone in Hr;
 assert (Hoff : w COL_SEL_LOCAL_SET = 0 /\ w COL_SEL_LOCAL_TEE = 0)
   by (repeat match goal with H : w _ = 0 |- _ => rewrite H in Hr end; field_linear Hr);
 destruct Hoff as [Hset Htee]
 end.

Lemma full_local_value_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_LOCAL_VALUE, Repairable 32%nat).
Proof.
 start_memory_repair COL_LOCAL_VALUE.
 memory_value_inactive COL_LOCAL_VALUE COL_SEL_LOCAL_GET.
 memory_value_inactive COL_LOCAL_VALUE COL_LOCAL_WRITE_ENABLED.
 memory_value_inactive COL_LOCAL_VALUE COL_GATHER_LOCAL_WRITE_LO.
 memory_value_inactive COL_LOCAL_VALUE GMEM_LOCAL.
 local_writes_inactive.
 finish_memory_repair COL_LOCAL_VALUE
   [[(COL_SEL_LOCAL_GET,1)]; [(COL_SEL_LOCAL_SET,1)]; [(COL_SEL_LOCAL_TEE,1)];
    [(COL_GATHER_LOCAL_WRITE_LO,1)]; [(GMEM_LOCAL,1)]].
Qed.

Lemma full_local_value_hi_memory_repair : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 rangeObligation w (COL_LOCAL_VALUE_HI, Repairable 32%nat).
Proof.
 start_memory_repair COL_LOCAL_VALUE_HI.
 memory_value_inactive COL_LOCAL_VALUE_HI COL_SEL_LOCAL_GET.
 memory_value_inactive COL_LOCAL_VALUE_HI COL_LOCAL_WRITE_ENABLED.
 memory_value_inactive COL_LOCAL_VALUE_HI COL_GATHER_LOCAL_WRITE.
 memory_value_inactive COL_LOCAL_VALUE_HI GMEM_LOCAL.
 local_writes_inactive.
 circuit_boolean COL_GATHER_LOCAL_WRITE_LO Blo.
 circuit_row ([(COL_GATHER_LOCAL_WRITE_LO,1)], [(COL_LOCAL_VALUE_HI,1)], ([] : SparseRow)) Hlo.
 expand_row Hlo.
 assert (Hzlo : w COL_GATHER_LOCAL_WRITE_LO = 0).
 { destruct Blo as [H|H]; [exact H|]. rewrite H in Hlo.
   pose proof (value_bound w COL_LOCAL_VALUE_HI). field_linear Hlo. }
 finish_memory_repair COL_LOCAL_VALUE_HI
   [[(COL_SEL_LOCAL_GET,1)]; [(COL_SEL_LOCAL_SET,1)]; [(COL_SEL_LOCAL_TEE,1)];
    [(COL_GATHER_LOCAL_WRITE_LO,1)]; [(COL_GATHER_LOCAL_WRITE,1); (COL_GATHER_LOCAL_WRITE_LO,-1)];
    [(GMEM_LOCAL,1)]].
Qed.

Theorem memory_range_obligations_sound : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 Forall (memoryRangeObligation w) memoryRangeObligations.
Proof.
 intros w Hprime Hfull Hbounds. unfold memoryRangeObligations.
 constructor; [split; [vm_compute; reflexivity | exact (full_return_pc_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_function_locals_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_local_value_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_local_value_hi_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_global_index_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_table_size_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_memory_offset_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_br_table_len_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_host_frame_0_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_host_frame_1_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_host_frame_2_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_host_sp_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor; [split; [vm_compute; reflexivity | exact (full_host_event_base_memory_repair w Hprime Hfull Hbounds)] | ].
 constructor.
Qed.

Theorem memory_range_obligations_run : forall initial rows final w,
 fieldPrime -> validCircuitRun initial rows final -> In w rows ->
 Forall (memoryRangeObligation w) memoryRangeObligations.
Proof.
 intros initial rows final w Hprime Hrun Hin.
 apply memory_range_obligations_sound; [exact Hprime| |].
 - exact (proj1 (Forall_forall _ _) (proj1 Hrun) w Hin).
 - eapply valid_run_memory_bounds; eassumption.
Qed.
