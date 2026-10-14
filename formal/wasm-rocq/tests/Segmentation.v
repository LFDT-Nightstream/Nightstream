From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec ProgramSpec ProgramRefinement TraceSegmentation tests.Programs tests.Callers.
Import ListNotations.
Open Scope Z_scope.

(* No supplied chunks or instruction/row correspondence: reaching the caller's
   final PC establishes that the entire get/call/add program was executed. *)
Example plain_rows_get_call_add : forall stale rows final depth,
  fieldPrime -> validCircuitRun (sourceMemory stale) rows final ->
  callerFragment (mkBoundary 2 4 4 7) depth rows ->
  boundaryPC (rowsFinalBoundary (mkBoundary 2 4 4 7) rows) = 5 ->
  exists z c, Steps (stackConfig sourceInitialState [] getCallAdd) c /\
    programLoaded sourceProgram final /\
    SourceState sourceProgram (sourcePosition 1 3) z final
      (rowsFinalBoundary (mkBoundary 2 4 4 7) rows) [] c.
Proof.
  intros stale rows final depth Hprime Hrun Hfragment Hfinish.
  destruct (run_refinement sourceProgram (sourcePosition 1 0) getCallAdd rows sourceInitialState
    (sourceMemory stale) final (mkBoundary 2 4 4 7) depth _ Hprime (source_program_loaded stale)
    Hrun Hfragment ltac:(exists 3%nat; split; [cbn; lia | exact Hfinish]) (source_initial_matches stale))
    as [executed [remaining [z [c [Hcode [Hsteps [Hloaded Hsource]]]]]]].
  assert (Hpc : boundaryPC (rowsFinalBoundary (mkBoundary 2 4 4 7) rows) = 2 + Z.of_nat (length executed)).
  { exact (proj2 (source_position_pc _ _ _ _ _ _ Hsource)). }
  assert (Hlength : length executed = 3%nat) by lia.
  assert (Hempty : remaining = []).
  { pose proof (f_equal (@length instr) Hcode) as Hsize.
    rewrite length_app in Hsize. change (3 = length executed + length remaining)%nat in Hsize.
    apply length_zero_iff_nil. lia. }
  subst remaining. exists z, c. split; [exact Hsteps | ]. split; [exact Hloaded | ].
  change (SourceState sourceProgram (sourcePosition 1 (length executed)) z final
    (rowsFinalBoundary (mkBoundary 2 4 4 7) rows) [] c) in Hsource.
  now rewrite Hlength in Hsource.
Qed.

(* A call cannot be accepted as a complete caller prefix without an END row,
   even when all the rows it does contain satisfy the circuit and memory rules. *)
Example unfinished_call_cannot_close : forall index rest rows row callees z initial final pc depth c0,
  fieldPrime -> validCircuitRun initial (row :: rows) final ->
  Forall (callerRow depth) (row :: rows) -> row COL_CALL_STACK_DEPTH_BEFORE = depth ->
  CallerState callees z initial (beforeBoundary row) pc (CALL index :: rest) c0 ->
  Forall (fun w : Witness => w COL_SEL_END = 0) (row :: rows) ->
  rowsFinalDepth depth (row :: rows) <> depth.
Proof.
  intros index rest rows row callees z initial final pc depth c0
    Hprime Hrun Hkinds Hdepth Hstate HnoEnd Hfinish.
  destruct (next_segment (CALL index) rest rows row callees z initial final pc depth c0
    Hprime Hrun Hkinds Hdepth Hfinish Hstate) as [segment [suffix [Hsplit [_ Hmatch]]]].
  destruct segment as [first | first middle last]; [contradiction | ].
  destruct Hmatch as [_ [Hend _]]. rewrite Hsplit in HnoEnd.
  apply Forall_forall with (x := last) in HnoEnd; [congruence | ].
  apply in_or_app. left. cbn [segmentRows]. right. apply in_or_app. right. cbn; auto.
Qed.

Example empty_caller_fragment : forall boundary depth, callerFragment boundary depth [].
Proof. intros; split; [exact I | ]. split; [constructor | reflexivity]. Qed.

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
  ~ callerRow 0 (scopeFlags [COL_IS_PROGRAM_ROW; COL_SEL_END; COL_HALTED]).
Proof. vm_compute. intuition discriminate. Qed.
