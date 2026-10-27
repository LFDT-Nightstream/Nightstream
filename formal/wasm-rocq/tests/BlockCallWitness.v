(* A real run of tests/BlockCalls.v's program, exported from the Rust trace: the
   caller's frame from its local zeroing to just before its final END, through the
   call, the callee's zeroing, its END and result move. Its rows meet every premise
   of run_refinement, so the theorem's conclusion holds of an actual run. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 StraightLineSpec SpecTecCalls
  ProgramSpec ProgramEncoding ProgramRefinement ControlDecode FrameInitialization tests.WitnessCheck tests.BlockCallTrace
  tests.BlockCalls tests.BlockCallRun.
Import ListNotations.
Open Scope Z_scope.

Definition witnessStart := beforeBoundary (hd (rowWitness []) witnessRows).
Definition witnessShape := controlShape [] (siteEntering 1).
Definition witnessEntry :=
  stackConfig (mk_state blockStore (enteredFrame blockModule [] 1)) [] callerBody.

Lemma witness_start : witnessStart = mkBoundary 2 0 1 0 0 1 0 1 0 0.
Proof. vm_compute. reflexivity. Qed.

Lemma witness_fragment : runFragment witnessStart 0 witnessRows.
Proof. split; [reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma witness_entry_config : CircuitConfig blockProgram blockStore 0 witnessShape witnessMemory witnessStart witnessEntry.
Proof.
  split; [reflexivity | ]. split; [repeat constructor | ]. split; [exact block_program_validated | ].
  exists (InitializingFrame witnessMemory witnessStart [] 1 (boundaryZeroRemaining witnessStart)).
  split; [vm_compute; reflexivity | ].
  rewrite witness_start. constructor; cbn [boundaryLocalsBase boundaryStackBase boundarySP boundaryZeroRemaining length];
    try lia.
  - intros index value Hv. destruct index; discriminate.
Qed.

(* The refinement's conclusion for this run. *)
Example block_call_witness_refines : fieldPrime ->
  exists shape c, Steps witnessEntry c /\
    CircuitConfig blockProgram blockStore 0 shape (replayed witnessMemory witnessRows)
      (rowsFinalBoundary witnessStart witnessRows) c.
Proof.
  intros Hprime.
  destruct (run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) witness_run_valid witness_fragment
    witness_entry_config) as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.
