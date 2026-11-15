From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog CircuitFacts
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

Theorem memory_range_obligations_sound : forall w,
 fieldPrime -> FullRelation.satFull w -> boundedMemoryEvents (eventsForRow memories w) ->
 Forall (memoryRangeObligation w) memoryRangeObligations.
Proof.
 intros w Hprime Hfull Hbounds. unfold memoryRangeObligations.
 constructor; [|constructor]. split; [vm_compute;reflexivity|].
 exact (full_return_pc_memory_repair w Hprime Hfull Hbounds).
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
