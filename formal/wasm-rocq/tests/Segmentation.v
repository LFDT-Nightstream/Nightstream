From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec ProgramSpec ControlDecode ControlInvariant ControlBridge ProgramRefinement tests.Programs tests.Callers.
Import ListNotations.
Open Scope Z_scope.

(* No supplied chunks or instruction/row correspondence. Ending at the caller's
   END pins the final shape. *)
Example plain_rows_get_call_add : forall stale rows final,
  fieldPrime -> validCircuitRun (sourceMemory stale) rows final ->
  runFragment (mkBoundary 2 8 8 7 0 0 0 0 0 0) 0 rows ->
  boundaryPC (rowsFinalBoundary (mkBoundary 2 8 8 7 0 0 0 0 0 0) rows) = 5 ->
  exists c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\
    programLoaded sourceProgram final /\
    CircuitConfig sourceProgram sourceStore 0 (controlShape [] (siteCode (codePosition 1 [] 3))) final
      (rowsFinalBoundary (mkBoundary 2 8 8 7 0 0 0 0 0 0) rows) c.
Proof.
  intros stale rows final Hprime Hrun Hfragment Hpc.
  destruct (run_refinement sourceProgram sourceStore 0 _ rows (sourceMemory stale) final _ _ Hprime
    (source_program_loaded stale) Hrun Hfragment (source_initial_config stale))
    as [shape [c [Hsteps [Hloaded Hconfig]]]].
  exists c. split; [exact Hsteps | ]. split; [exact Hloaded | ].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig) as [cursor Hstate].
  rewrite <- (source_final_shape _ _ _ _ Hstate Hpc).
  exact (control_state_circuit _ _ _ _ _ _ _ Logic.eq_refl source_store_wf source_program_validated Hstate).
Qed.

(* These rows test the scope predicate alone, not full circuit satisfaction or
   support for block/return instructions in the source fragment. *)
Definition scopeFlags (ones : list Column) : Witness.
Proof.
  refine {| entries := fun column => exist _
    (if existsb (Nat.eqb column) ones then 1 else 0) _ |}.
  destruct (existsb (Nat.eqb column) ones); unfold modulus; lia.
Defined.

Example block_end_allowed_by_caller_scope :
  callerRow 0 (scopeFlags [COL_IS_PROGRAM_ROW; COL_SEL_END]).
Proof. vm_compute. auto. Qed.

Example explicit_return_rejected_by_caller_scope :
  ~ callerRow 1 (scopeFlags [COL_IS_PROGRAM_ROW; COL_SEL_RETURN;
    COL_CALL_STACK_DEPTH_BEFORE; COL_CALL_STACK_POP_PRESENT]).
Proof. vm_compute. intuition discriminate. Qed.

Example depth_zero_halt_rejected_by_caller_scope :
  ~ callerRow 0 (scopeFlags [COL_IS_PROGRAM_ROW; COL_SEL_END; COL_ACTIVATION_FINISHED_AFTER]).
Proof. vm_compute. intuition discriminate. Qed.
