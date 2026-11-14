(* A callee's frame during entry: its arguments are already its first local cells,
   and zero rows push its declared locals upward from them. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  Wasm2 WasmIntegers WasmValues IntegerMemory IntegerDecode StraightLineSpec CallInitialization CallMemory TraceRefinement StackSimulation.
Import ListNotations.
Open Scope Z_scope.

Record InitializingFrame (memory : MemoryState) (boundary : Boundary)
    (arguments : list val) (declaredLocals zeroRemaining : Z) : Prop := {
  zero_remaining_bounds : 0 <= zeroRemaining <= declaredLocals;
  initialization_fbp : 0 <= boundaryLocalsBase boundary;
  initialization_base : boundaryStackBase boundary =
    boundaryLocalsBase boundary + Z.of_nat (length arguments) + declaredLocals;
  initialization_sp : boundarySP boundary = boundaryStackBase boundary - zeroRemaining;
  argument_cells : forall index value, nth_error arguments index = Some value ->
    CellMatches memory (boundaryLocalsBase boundary + Z.of_nat index) value;
  zeroed_locals : forall index,
    Z.of_nat (length arguments) <= index < Z.of_nat (length arguments) + declaredLocals - zeroRemaining ->
    memory stackId [2 * (boundaryLocalsBase boundary + index)] = Some 0 /\
    memory stackId [2 * (boundaryLocalsBase boundary + index) + 1] = Some 0
}.

Lemma initializing_frame_start : forall memory boundary arguments declaredLocals,
  0 <= declaredLocals -> 0 <= boundaryLocalsBase boundary ->
  boundarySP boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) + declaredLocals ->
  (forall index value, nth_error arguments index = Some value ->
    CellMatches memory (boundaryLocalsBase boundary + Z.of_nat index) value) ->
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
  - intros index value Hlookup. rewrite HfbpAfter.
    assert (Hlength : (index < length arguments)%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. }
    eapply cell_matches_memory; [ | exact (Hargs _ _ Hlookup)].
    intros limb Hlimb. assert (0 <= limb <= 1) by (destruct Hlimb; lia).
    rewrite Hwrite, !update_other by cell_differs. reflexivity.
  - intros index Hrange. rewrite !Hwrite, HfbpAfter.
    destruct (Z.eq_dec (w COL_LOCALS_FBP_BEFORE + index) (w COL_SP_BEFORE)) as [Heq | Hneq].
    + rewrite Heq. split.
      * rewrite update_other by cell_differs. apply update_same.
      * apply update_same.
    + rewrite !update_other by cell_differs.
      apply Hzeros. lia.
Qed.

Lemma initializing_i32_frame_start : forall memory boundary arguments declaredLocals,
  0 <= declaredLocals -> 0 <= boundaryLocalsBase boundary ->
  boundarySP boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (length arguments) + declaredLocals ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  (forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index)] = Some value) ->
  InitializingFrame memory boundary (map i32Value arguments) declaredLocals declaredLocals.
Proof.
  intros memory boundary arguments declaredLocals Hl Hfbp Hsp Hbase Hbounds Hcells.
  apply initializing_frame_start; rewrite ?length_map; try assumption.
  intros index value Hlookup. rewrite nth_error_map in Hlookup.
  destruct (nth_error arguments index) as [raw | ] eqn:Hraw; [ | discriminate].
  injection Hlookup as <-. constructor.
  - exact (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hraw)).
  - exact (Hcells _ _ Hraw).
Qed.

Lemma initializing_i32_arguments : forall memory boundary arguments declaredLocals remaining,
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  InitializingFrame memory boundary (map i32Value arguments) declaredLocals remaining ->
  forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index)] = Some value.
Proof.
  intros memory boundary arguments declaredLocals remaining Hbounds Hinit index value Hindex.
  eapply cell_matches_i32_read.
  - exact (proj1 (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hindex))).
  - apply (argument_cells _ _ _ _ _ Hinit). rewrite nth_error_map, Hindex. reflexivity.
Qed.

Lemma initialized_frame_reads : forall memory boundary arguments params locals defaults,
  Forall2 (fun t value => integerValueType value = Some t) params arguments ->
  integerZeros locals = Some defaults ->
  InitializingFrame memory boundary arguments (Z.of_nat (length locals)) 0 ->
  readIntegerCells memory (fun index => boundaryLocalsBase boundary + Z.of_nat index)
    (params ++ locals) = Some (arguments ++ defaults).
Proof.
  intros memory boundary arguments params locals defaults Htypes Hdefaults Hinit.
  assert (Hlength : length params = length arguments) by (exact (Forall2_length Htypes)).
  rewrite read_integer_cells_app.
  assert (Hargs : readIntegerCells memory
    (fun index => boundaryLocalsBase boundary + Z.of_nat index) params = Some arguments).
  { apply read_integer_cells_some. split; [exact Htypes | exact (argument_cells _ _ _ _ _ Hinit)]. }
  rewrite Hargs.
  erewrite integer_zeros_cells; [reflexivity | exact Hdefaults | ].
  intros index Hindex. cbn. apply (zeroed_locals _ _ _ _ _ Hinit).
  rewrite Nat2Z.inj_add, Hlength. lia.
Qed.

(* Both source limbs come from replay; this applies to i32 and i64 results alike. *)
Lemma move_row_preserves_integer_operands : forall initial final w base types values,
  fieldPrime -> FullRelation.satFull w -> w COL_FRAME_MOVE_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_SP_BEFORE = base + Z.of_nat (length types) ->
  (Z.to_nat (w COL_FRAME_MOVE_REMAINING_BEFORE) <= length types)%nat ->
  readIntegerOperands initial base types (Z.to_nat (w COL_FRAME_MOVE_REMAINING_BEFORE))
    (w COL_FRAME_MOVE_DISTANCE_BEFORE) = Some values ->
  readIntegerOperands final base types (Z.to_nat (w COL_FRAME_MOVE_REMAINING_AFTER))
    (w COL_FRAME_MOVE_DISTANCE_AFTER) = Some values.
Proof.
  intros initial final w base types values Hprime Hfull Hmove Hreplay Hsp Hbound Hread.
  destruct (full_frame_move_row w Hprime Hfull Hmove) as [Hdec [Hdistance [Hdest [Hsource _]]]].
  destruct (move_row_stack_replay _ _ _ Hprime Hfull Hmove Hreplay) as [Hlo [Hhi Hmemory]].
  pose proof (value_bound w COL_FRAME_MOVE_REMAINING_BEFORE) as Hbefore.
  pose proof (value_bound w COL_FRAME_MOVE_REMAINING_AFTER) as Hafter.
  pose proof (value_bound w COL_FRAME_MOVE_DISTANCE_BEFORE) as Hdist.
  set (remaining := Z.to_nat (w COL_FRAME_MOVE_REMAINING_BEFORE)) in *.
  set (target := base + Z.of_nat (length types - remaining)).
  assert (Htarget : w COL_STACK_WRITE0_ADDR_LO = 2 * target) by (unfold target, remaining; lia).
  assert (Haddress : w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) =
      2 * (target + w COL_FRAME_MOVE_DISTANCE_BEFORE)) by lia.
  rewrite Haddress in Hlo, Hhi. rewrite Hdistance.
  replace (Z.to_nat (w COL_FRAME_MOVE_REMAINING_AFTER)) with (Nat.pred remaining) by (unfold remaining; lia).
  pose proof (read_integer_operands_copy initial base types remaining (w COL_FRAME_MOVE_DISTANCE_BEFORE)
    values (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))
    (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)) ltac:(unfold remaining in *; lia)
    ltac:(lia) Hread Hlo Hhi) as Hcopy.
  unfold readIntegerOperands in Hcopy |- *.
  eapply read_integer_cells_preserved; [exact Hcopy | ].
  intros index limb _ _. unfold writeCell. rewrite Hmemory, Htarget. reflexivity.
Qed.
