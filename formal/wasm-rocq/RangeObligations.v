From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog CircuitFacts
  InactiveOffset InactiveStackBindings.
Import ListNotations.
Open Scope Z_scope.

Definition rangeObligation (w : Witness) (entry : nat * Width) : Prop :=
 let '(column, width) := entry in
 match width with
 | Implied bits => 0 <= w column < 2 ^ Z.of_nat bits
 | Repairable bits => exists w',
     FullRelation.satFull w' /\
     0 <= w' column < 2 ^ Z.of_nat bits /\
     eventsForRow memories w' = eventsForRow memories w /\
     map (fun '(a,b) => (w' a,w' b)) continuityLinks =
     map (fun '(a,b) => (w a,w b)) continuityLinks /\
     (forall c, c <> column -> w' c = w c)
 | _ => False
 end.

Theorem range_obligations_sound : forall w,
 fieldPrime -> FullRelation.satFull w ->
 Forall (rangeObligation w) rangeObligations.
Proof.
 intros w Hprime Hfull.
 unfold rangeObligations.
 apply Forall_cons.
 { change (0 <= w COL_OPCODE_CODE < 4294967296).
   pose proof (full_opcode_code_bound w Hprime Hfull); lia. }
 apply Forall_cons.
 { change (0 <= w COL_STACK_READS < 4294967296).
   pose proof (full_stack_reads_bound w Hprime Hfull); lia. }
 apply Forall_cons.
 { change (0 <= w COL_STACK_WRITES < 4294967296).
   pose proof (full_stack_writes_bound w Hprime Hfull); lia. }
 apply Forall_cons. { exact (full_global_lo_repair w Hprime Hfull). }
 apply Forall_cons. { exact (full_global_hi_repair w Hprime Hfull). }
 apply Forall_cons. { exact (full_table_index_repair w Hprime Hfull). }
 apply Forall_cons.
 { change (0 <= w COL_MAX_MEMORY_PAGES_AFTER < 4294967296).
   pose proof (full_max_memory_pages_after_bound w Hprime Hfull); lia. }
 apply Forall_cons. { exact (full_offset_repair w Hprime Hfull). }
 apply Forall_cons.
 { change (0 <= w COL_HOST_IMPORT_SCHEDULE_COUNTS < 4294967296).
   pose proof (full_import_schedule_counts_bound w Hprime Hfull); lia. }
 constructor.
Qed.
