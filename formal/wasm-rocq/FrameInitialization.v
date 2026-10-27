(* A callee's frame during entry: its arguments are already its first local cells,
   and zero rows push its declared locals upward from them. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  StraightLineSpec CallInitialization CallMemory TraceRefinement StackSimulation.
Import ListNotations.
Open Scope Z_scope.

Record InitializingFrame (memory : MemoryState) (boundary : Boundary)
    (arguments : list Z) (declaredLocals zeroRemaining : Z) : Prop := {
  zero_remaining_bounds : 0 <= zeroRemaining <= declaredLocals;
  initialization_fbp : 0 <= boundaryLocalsBase boundary;
  initialization_base : boundaryStackBase boundary =
    boundaryLocalsBase boundary + Z.of_nat (length arguments) + declaredLocals;
  initialization_sp : boundarySP boundary = boundaryStackBase boundary - zeroRemaining;
  argument_cells : forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index)] = Some value;
  zeroed_locals : forall index,
    Z.of_nat (length arguments) <= index < Z.of_nat (length arguments) + declaredLocals - zeroRemaining ->
    memory stackId [2 * (boundaryLocalsBase boundary + index)] = Some 0
}.

Lemma initializing_frame_start : forall memory boundary arguments declaredLocals,
  0 <= declaredLocals -> 0 <= boundaryLocalsBase boundary ->
  boundarySP boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) + declaredLocals ->
  (forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index)] = Some value) ->
  InitializingFrame memory boundary arguments declaredLocals declaredLocals.
Proof.
  intros memory boundary arguments declaredLocals Hlocals Hfbp Hsp Hbase Hargs.
  constructor; try assumption; try lia.
Qed.

Lemma zero_row_initializes_frame : forall initial final w arguments declaredLocals,
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  InitializingFrame initial (beforeBoundary w) arguments declaredLocals (w COL_LOCAL_ZERO_REMAINING_BEFORE) ->
  InitializingFrame final (afterBoundary w) arguments declaredLocals (w COL_LOCAL_ZERO_REMAINING_AFTER).
Proof.
  intros initial final w arguments declaredLocals Hprime Hfull Hzero Hreplay Hinv.
  destruct Hinv as [Hzb Hfbp Hbase Hsp Hargs Hzeros].
  destruct (full_zero_initialization_row w Hprime Hfull Hzero) as [Hdec _].
  pose proof (full_initialization_stack w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero) as Hshape.
  pose proof (init_stack_pointer w _ Hshape) as HspAfter. cbn in HspAfter.
  destruct (full_initialization_preserves_frame w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero) as [HfbpAfter [HbaseAfter _]].
  pose proof (zero_row_stack_replay initial final w Hprime Hfull Hzero Hreplay) as Hwrite.
  pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_AFTER) as Ha.
  cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase] in *.
  constructor; cbn [afterBoundary boundarySP boundaryStackBase boundaryLocalsBase].
  - lia.
  - lia.
  - lia.
  - lia.
  - intros index value Hlookup. rewrite Hwrite, HfbpAfter.
    assert (Hlength : (index < length arguments)%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. }
    rewrite !update_other by cell_differs.
    apply Hargs. exact Hlookup.
  - intros index Hrange. rewrite Hwrite, HfbpAfter.
    destruct (Z.eq_dec (w COL_LOCALS_FBP_BEFORE + index) (w COL_SP_BEFORE)) as [Heq | Hneq].
    + rewrite update_other by cell_differs.
      rewrite Heq, update_same. reflexivity.
    + rewrite !update_other by cell_differs.
      apply Hzeros. lia.
Qed.
