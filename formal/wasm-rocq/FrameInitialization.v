(* Partial local contents during entry: completed zero writes grow upward,
   copied parameters grow downward, and argument cells remain readable. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  StraightLineSpec CallInitialization CallMemory TraceRefinement StackSimulation.
Import ListNotations.
Open Scope Z_scope.

Record InitializingFrame (memory : MemoryState) (boundary : Boundary)
    (arguments : list Z) (declaredLocals zeroRemaining paramRemaining : Z) : Prop := {
  zero_remaining_bounds : 0 <= zeroRemaining <= declaredLocals;
  param_remaining_bounds : 0 <= paramRemaining <= Z.of_nat (length arguments);
  initialization_phase : zeroRemaining > 0 -> paramRemaining = Z.of_nat (length arguments);
  initialization_sp : boundarySP boundary = boundaryStackBase boundary + paramRemaining;
  argument_cells : forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryStackBase boundary + Z.of_nat index)] = Some value;
  copied_parameters : forall index value, nth_error arguments index = Some value ->
    paramRemaining <= Z.of_nat index ->
    memory localLoId [boundaryLocalsBase boundary; Z.of_nat index] = Some value;
  zeroed_locals : forall index,
    Z.of_nat (length arguments) <= index < Z.of_nat (length arguments) + declaredLocals - zeroRemaining ->
    memory localLoId [boundaryLocalsBase boundary; index] = Some 0
}.

Lemma initializing_frame_start : forall memory boundary arguments declaredLocals,
  0 <= declaredLocals ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length arguments) ->
  (forall index value, nth_error arguments index = Some value ->
    memory stackId [2 * (boundaryStackBase boundary + Z.of_nat index)] = Some value) ->
  InitializingFrame memory boundary arguments declaredLocals declaredLocals (Z.of_nat (length arguments)).
Proof.
  intros memory boundary arguments declaredLocals Hlocals Hsp Hargs.
  constructor; try assumption; try lia.
  - intros index value Hlookup Hrange.
    assert (Hlength : (index < length arguments)%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. } lia.
Qed.

Lemma zero_row_initializes_frame : forall initial final w arguments declaredLocals,
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_CURRENT_FUNCTION_NUM_LOCALS = Z.of_nat (length arguments) + declaredLocals ->
  InitializingFrame initial (beforeBoundary w) arguments declaredLocals
    (w COL_LOCAL_ZERO_REMAINING_BEFORE) (w COL_PARAM_INIT_REMAINING_BEFORE) ->
  InitializingFrame final (afterBoundary w) arguments declaredLocals
    (w COL_LOCAL_ZERO_REMAINING_AFTER) (w COL_PARAM_INIT_REMAINING_AFTER).
Proof.
  intros initial final w arguments declaredLocals Hprime Hfull Hzero Hreplay Htotal Hinv.
  destruct Hinv as [Hzb Hpb Hphase Hsp Hargs Hparams Hzeros].
  destruct (full_zero_initialization_row w Hprime Hfull Hzero) as [Hdec [Hindex [Hvalue _]]].
  assert (Hparam : w COL_PARAM_INIT_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive w COL_LOCAL_ZERO_ACTIVE_BEFORE COL_PARAM_INIT_ACTIVE_BEFORE
      Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hzero). }
  pose proof (full_initialization_counter_carry w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero) as [_ Hcarry].
  specialize (Hcarry Hparam).
  destruct (full_initialization_preserves_frame w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero) as [Hfbp [Hbase _]].
  pose proof (full_initialization_stack w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero) as [_ [_ [_ HspAfter]]].
  pose proof (initialization_writes_local initial final w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero Hreplay) as Hwrite.
  pose proof (initialization_preserves_stack_memory initial final w COL_LOCAL_ZERO_ACTIVE_BEFORE
    Hprime Hfull ltac:(left; reflexivity) Hzero Hreplay) as Hstack.
  pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_AFTER) as Ha.
  cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase] in *.
  constructor; cbn [afterBoundary boundarySP boundaryStackBase boundaryLocalsBase].
  - lia.
  - lia.
  - intros Hpositive. specialize (Hphase ltac:(lia)). lia.
  - rewrite Hparam in HspAfter. lia.
  - intros index value Hlookup. rewrite Hstack, Hbase. apply Hargs. exact Hlookup.
  - intros index value Hlookup Hrange. rewrite Hwrite, Hfbp.
    rewrite update_other.
    + apply Hparams; [exact Hlookup | lia].
    + right. intro Heq. inversion Heq.
      assert (Hlength : (index < length arguments)%nat).
      { apply nth_error_Some. rewrite Hlookup. discriminate. }
      lia.
  - intros index Hrange. rewrite Hwrite, Hfbp.
    destruct (Z.eq_dec index (w COL_LOCAL_INDEX)) as [-> | Hneq].
    + rewrite update_same, Hvalue. reflexivity.
    + rewrite update_other by (right; congruence).
      apply Hzeros. lia.
Qed.

(* A parameter row copies one argument once the zero countdown has finished. *)
Lemma parameter_row_after_zeroing : forall initial final w arguments declaredLocals,
  fieldPrime -> FullRelation.satFull w -> w COL_PARAM_INIT_ACTIVE_BEFORE = 1 ->
  w COL_LOCAL_ZERO_REMAINING_BEFORE = 0 ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  InitializingFrame initial (beforeBoundary w) arguments declaredLocals
    (w COL_LOCAL_ZERO_REMAINING_BEFORE) (w COL_PARAM_INIT_REMAINING_BEFORE) ->
  InitializingFrame final (afterBoundary w) arguments declaredLocals
    (w COL_LOCAL_ZERO_REMAINING_AFTER) (w COL_PARAM_INIT_REMAINING_AFTER).
Proof.
  intros initial final w arguments declaredLocals Hprime Hfull Hparam HzeroBefore Hreplay Hinv.
  destruct Hinv as [Hzb Hpb Hphase Hsp Hargs Hparams Hzeros].
  destruct (full_parameter_initialization_row w Hprime Hfull Hparam) as [Hdec [Hindex _]].
  assert (Hzero : w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive w COL_PARAM_INIT_ACTIVE_BEFORE COL_LOCAL_ZERO_ACTIVE_BEFORE
      Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hparam). }
  pose proof (full_initialization_counter_carry w COL_PARAM_INIT_ACTIVE_BEFORE
    Hprime Hfull ltac:(right; reflexivity) Hparam) as [Hcarry _].
  specialize (Hcarry Hzero).
  destruct (full_initialization_preserves_frame w COL_PARAM_INIT_ACTIVE_BEFORE
    Hprime Hfull ltac:(right; reflexivity) Hparam) as [Hfbp [Hbase _]].
  pose proof (full_initialization_stack w COL_PARAM_INIT_ACTIVE_BEFORE
    Hprime Hfull ltac:(right; reflexivity) Hparam) as [_ [_ [_ HspAfter]]].
  pose proof (initialization_writes_local initial final w COL_PARAM_INIT_ACTIVE_BEFORE
    Hprime Hfull ltac:(right; reflexivity) Hparam Hreplay) as Hwrite.
  pose proof (initialization_preserves_stack_memory initial final w COL_PARAM_INIT_ACTIVE_BEFORE
    Hprime Hfull ltac:(right; reflexivity) Hparam Hreplay) as Hstack.
  pose proof (parameter_reads_initial_stack initial final w Hprime Hfull Hparam Hreplay) as Hread.
  pose proof (value_bound w COL_PARAM_INIT_REMAINING_AFTER) as Ha.
  cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase] in *.
  constructor; cbn [afterBoundary boundarySP boundaryStackBase boundaryLocalsBase].
  - lia.
  - lia.
  - lia.
  - rewrite Hparam in HspAfter. lia.
  - intros index value Hlookup. rewrite Hstack, Hbase. apply Hargs. exact Hlookup.
  - intros index value Hlookup Hrange. rewrite Hwrite, Hfbp.
    destruct (Z.eq_dec (Z.of_nat index) (w COL_LOCAL_INDEX)) as [Heq | Hneq].
    + rewrite Heq, update_same.
      specialize (Hargs index value Hlookup).
      replace (2 * (w COL_SP_BEFORE - 1)) with
        (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) in Hread by lia.
      congruence.
    + rewrite update_other by (right; congruence).
      apply Hparams; [exact Hlookup | lia].
  - intros index Hrange. rewrite Hwrite, Hfbp.
    rewrite update_other by (right; intro Heq; inversion Heq; lia).
    apply Hzeros. lia.
Qed.

Lemma initialized_frame_matches : forall memory boundary arguments locals s module,
  0 <= boundaryStackBase boundary ->
  Forall (fun value => 0 <= value < WasmIntegers.i32Modulus) arguments ->
  InitializingFrame memory boundary arguments (Z.of_nat locals) 0 0 ->
  stackMatches memory boundary [] /\
  localsMatch memory boundary (Wasm2.mk_state s
    (Wasm2.MKframe (map Locals.i32Value arguments ++ repeat (Locals.i32Value 0) locals) module)).
Proof.
  intros memory boundary arguments locals s module Hbase Hbounds Hinit.
  destruct Hinit as [Hz Hp Hphase Hsp Hargs Hparams Hzeros].
  split.
  - unfold stackMatches. cbn. repeat split; try assumption; try lia; try constructor.
    intros [ | index] value H; discriminate.
  - intros index v Hlookup. cbn [stateLocals Wasm2.LOCALS] in Hlookup.
    assert (Hlength : (index < length arguments + locals)%nat).
    { assert (Hlt : (index < length (map Locals.i32Value arguments ++ repeat (Locals.i32Value 0) locals))%nat).
      { apply nth_error_Some. rewrite Hlookup. discriminate. }
      rewrite length_app, length_map, repeat_length in Hlt. exact Hlt. }
    destruct (lt_dec index (length arguments)) as [Hin | Hout].
    + rewrite nth_error_app1 in Hlookup by (rewrite length_map; exact Hin).
      rewrite nth_error_map in Hlookup.
      destruct (nth_error arguments index) as [value | ] eqn:Hvalue; [ | discriminate].
      injection Hlookup as <-. exists value. split; [reflexivity | ]. split.
      * apply (proj1 (Forall_forall _ _) Hbounds). apply nth_error_In with (n := index). exact Hvalue.
      * apply Hparams; [exact Hvalue | lia].
    + rewrite nth_error_app2 in Hlookup by (rewrite length_map; lia).
      apply nth_error_In in Hlookup. apply repeat_spec in Hlookup. subst v.
      exists 0. split; [reflexivity | ]. split; [unfold WasmIntegers.i32Modulus; lia | ].
      apply Hzeros. lia.
Qed.
