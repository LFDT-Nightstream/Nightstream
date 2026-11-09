(* Memory outside the active frame remains available to its suspended caller. *)
From Stdlib Require Import List ZArith Lia.
Require Import CircuitFacts TraceRefinement R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog
  StraightLineSpec StackBindings StackSimulation LocalBindings LocalSimulation
  CallContext CallMemory CallInitialization StraightLineInstructions Wasm2.
Import ListNotations.
Open Scope Z_scope.

(* A frame and everything above it may change; the stack cells below its fbp,
   which hold its suspended callers, and the saved call contexts may not. *)
Record FrameEffect (fbp : Z) (initial final : MemoryState) : Prop := {
  preserves_below_frame : forall index, index < fbp ->
    final stackId [2 * index] = initial stackId [2 * index];
  preserves_call_context : forall memory column address, In (memory, column) contextMemories ->
    final memory address = initial memory address
}.

Lemma body_row_call_context : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> straightLineSelector chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  forall memory column address, In (memory, column) contextMemories ->
    final memory address = initial memory address.
Proof.
  intros initial final w chosen Hprime Hfull Hchoice Hchosen Hreplay memory column address Hin.
  destruct (full_selected_no_host_switch w chosen Hprime Hfull (straight_line_selector_in chosen Hchoice) Hchosen)
    as [Hsuspend [_ [_ [_ Hread]]]].
  eapply context_preserved_without_push; [exact Hin | | exact Hsuspend | exact Hread | exact Hreplay].
  exact (proj1 (full_body_context_quiet w chosen Hprime Hfull Hchoice Hchosen)).
Qed.

Lemma binary_row_frame_effect : forall initial final w chosen prefix a b,
  fieldPrime -> FullRelation.satFull w -> binaryArithmeticSelector chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_LOCALS_FBP_BEFORE <= w COL_STACK_FRAME_BASE_BEFORE ->
  stackMatches initial (beforeBoundary w) (prefix ++ [a; b]) ->
  FrameEffect (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w chosen prefix a b Hprime Hfull Hchoice Hchosen Hreplay Hlayout
    [Hbase [Hsp [Hbounds Hcells]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_binary_stack_shape w chosen Hprime Hfull Hchoice Hchosen ltac:(lia)) as Hshape.
  destruct (binary_stack_replay w initial final Hshape Hreplay) as [_ [_ Hwrites]].
  assert (HlocalChoice : straightLineSelector chosen) by (destruct Hchoice as [-> | ->]; cbn; auto).
  constructor; [ | exact (body_row_call_context initial final w chosen Hprime Hfull HlocalChoice Hchosen Hreplay)].
  intros index Hindex. rewrite Hwrites.
  rewrite (stack_write_address w Hshape), (stack_write_hi_address w Hshape).
  rewrite !update_other by cell_differs. reflexivity.
Qed.

Lemma local_get_row_frame_effect : forall initial final w values,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_GET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_LOCALS_FBP_BEFORE <= w COL_STACK_FRAME_BASE_BEFORE ->
  stackMatches initial (beforeBoundary w) values ->
  FrameEffect (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w values Hprime Hfull Hchosen Hreplay Hlayout [Hbase [Hsp Hrest]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  pose proof (full_local_get_stack_shape w Hprime Hfull Hchosen ltac:(lia)) as Hshape.
  destruct (local_get_stack_replay w initial final Hshape Hreplay) as [_ Hwrites].
  constructor; [ | exact (body_row_call_context initial final w COL_SEL_LOCAL_GET Hprime Hfull
    ltac:(cbn; auto) Hchosen Hreplay)].
  intros index Hindex. rewrite Hwrites.
  rewrite (local_write_address w false true Hshape eq_refl), (local_write_hi_address w false true Hshape eq_refl).
  rewrite !update_other by cell_differs. reflexivity.
Qed.

Lemma local_set_row_frame_effect : forall initial final w prefix value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_SET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [value]) ->
  FrameEffect (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w prefix value Hprime Hfull Hchosen Hreplay [Hbase [Hsp Hrest]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_local_set_stack_shape w Hprime Hfull Hchosen ltac:(lia)) as Hshape.
  destruct (local_set_stack_replay w initial final Hshape Hreplay) as [_ Hwrites].
  destruct (full_local_addresses w Hprime Hfull) as [Hlo Hhi].
  pose proof (value_bound w COL_LOCAL_INDEX).
  constructor; [ | exact (body_row_call_context initial final w COL_SEL_LOCAL_SET Hprime Hfull
    ltac:(cbn; auto) Hchosen Hreplay)].
  intros index Hindex. rewrite Hwrites, Hhi, Hlo.
  rewrite !update_other by cell_differs. reflexivity.
Qed.

(* A zero or move row writes one cell at or above the frame's base pointer. *)
Lemma initialization_frame_effect : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  2 * w COL_LOCALS_FBP_BEFORE <= w COL_STACK_WRITE0_ADDR_LO ->
  FrameEffect (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w chosen Hprime Hfull Hkind Hchosen Hreplay Habove.
  constructor.
  - intros index Hindex.
    destruct Hkind as [-> | ->].
    + destruct (full_zero_initialization_row w Hprime Hfull Hchosen) as [_ [Haddr _]].
      rewrite (zero_row_stack_replay initial final w Hprime Hfull Hchosen Hreplay).
      rewrite !update_other by cell_differs. reflexivity.
    + destruct (move_row_stack_replay initial final w Hprime Hfull Hchosen Hreplay) as [_ Hwrites].
      rewrite Hwrites. rewrite !update_other by cell_differs. reflexivity.
  - intros memory column address Hin.
    destruct (full_host_switch_absent w chosen Hprime Hfull (initialization_kind_in chosen Hkind)
      ltac:(destruct Hkind as [-> | ->]; discriminate) ltac:(destruct Hkind as [-> | ->]; discriminate)
      ltac:(destruct Hkind as [-> | ->]; discriminate) Hchosen) as [Hsuspend [Hresume Hnested]].
    eapply context_preserved_without_push; [exact Hin | | exact Hsuspend |
      exact (full_no_host_context_read w Hfull Hresume Hnested) | exact Hreplay].
    destruct (full_initialization_auxiliary w chosen Hprime Hfull Hkind Hchosen) as [_ [_ [Hpush _]]].
    exact Hpush.
Qed.

Definition frameRegistersPreserved (w : Witness) : Prop :=
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE.

Lemma frame_row_scope : forall instruction rest z initial final row c0,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  encodedOpcode instruction = Some (row COL_OPCODE_CODE) ->
  row COL_LOCALS_FBP_BEFORE <= row COL_STACK_FRAME_BASE_BEFORE ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (instruction :: rest) /\
    represents z initial (beforeBoundary row) (instruction :: rest) c0) ->
  FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final /\
  frameRegistersPreserved row.
Proof.
  intros instruction rest z initial final row c0 Hprime Hfull Hprogram Hreplay Hcode Hlayout Hframe.
  destruct (encoded_opcode_cases instruction _ Hcode)
    as [[-> HcodeValue] | [[-> HcodeValue] | [[x [-> HcodeValue]] | [x [-> HcodeValue]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_ADD); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_I32_ADD) by (cbn; auto).
    split; [ | exact (proj2 (proj2 (full_body_context_quiet row _ Hprime Hfull Hchoice Hchosen)))].
    apply represented_values in Hframe. destruct Hframe as [values [Hstack [Htyped Hrest]]].
    assert (Hheight : (2 <= length values)%nat).
    { destruct (length values) as [ | [ | n]]; cbn [programTyped instructionStackHeight] in Htyped; tauto || lia. }
    destruct (stack_has_two_values values Hheight) as [prefix [a [b ->]]].
    eapply binary_row_frame_effect with (chosen := COL_SEL_I32_ADD); try eassumption.
    left; reflexivity.
  - assert (Hchosen : row COL_SEL_I32_SUB = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_SUB); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_I32_SUB) by (cbn; auto).
    split; [ | exact (proj2 (proj2 (full_body_context_quiet row _ Hprime Hfull Hchoice Hchosen)))].
    apply represented_values in Hframe. destruct Hframe as [values [Hstack [Htyped Hrest]]].
    assert (Hheight : (2 <= length values)%nat).
    { destruct (length values) as [ | [ | n]]; cbn [programTyped instructionStackHeight] in Htyped; tauto || lia. }
    destruct (stack_has_two_values values Hheight) as [prefix [a [b ->]]].
    eapply binary_row_frame_effect with (chosen := COL_SEL_I32_SUB); try eassumption.
    right; reflexivity.
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_GET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_GET) by (cbn; auto).
    split; [ | exact (proj2 (proj2 (full_body_context_quiet row _ Hprime Hfull Hchoice Hchosen)))].
    apply represented_values in Hframe. destruct Hframe as [values [Hstack [Htyped Hrest]]].
    eapply local_get_row_frame_effect; eassumption.
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_SET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_SET) by (cbn; auto).
    split; [ | exact (proj2 (proj2 (full_body_context_quiet row _ Hprime Hfull Hchoice Hchosen)))].
    apply represented_values in Hframe. destruct Hframe as [values [Hstack [Htyped Hrest]]].
    cbn [programTyped instructionStackHeight] in Htyped. destruct Htyped as [_ Hheight].
    assert (Hpositive : (0 < length values)%nat) by (destruct (length values); [contradiction | lia]).
    destruct (stack_has_last values Hpositive) as [prefix [value ->]].
    eapply local_set_row_frame_effect; eassumption.
Qed.
