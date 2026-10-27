(* The exported run's rows satisfy the circuit, chain, and replay from the loaded
   program; checked by computation. *)
From Stdlib Require Import List ZArith.
Require Import R1CS MemoryTrace MemoryCatalog TraceRefinement ProgramSpec tests.WitnessCheck tests.BlockCallTrace tests.BlockCalls.
Import ListNotations.
Open Scope Z_scope.

Definition witnessRows := map rowWitness blockCallRows.
Definition witnessMemory := loadProgram blockProgram (fun memory _ =>
  if Nat.eqb memory stackId then Some 0 else None).

Lemma witness_run_checks : validRunb witnessMemory witnessRows = true.
Proof. vm_compute. reflexivity. Qed.

Lemma witness_run_valid : validCircuitRun witnessMemory witnessRows (replayed witnessMemory witnessRows).
Proof. exact (validRunb_sound _ _ witness_run_checks). Qed.

