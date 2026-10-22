(* Memory outside the active frame remains available to its suspended caller. *)
From Stdlib Require Import List ZArith Lia.
Require Import CircuitFacts TraceRefinement R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog
  StraightLineSpec StackBindings StackSimulation LocalBindings LocalSimulation
  CallContext CallMemory CallInitialization StraightLineInstructions Wasm2.
Import ListNotations.
Open Scope Z_scope.

Record FrameEffect (base fbp : Z) (initial final : MemoryState) : Prop := {
  preserves_stack_prefix : forall index, index < base ->
    final stackId [2 * index] = initial stackId [2 * index];
  preserves_other_locals : forall other index, other <> fbp ->
    final localLoId [other; index] = initial localLoId [other; index];
  preserves_call_context : forall memory column address, In (memory, column) contextMemories ->
    final memory address = initial memory address
}.

Lemma body_row_other_memories : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> straightLineSelector chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  (forall other index, other <> w COL_LOCALS_FBP_BEFORE ->
    final localLoId [other; index] = initial localLoId [other; index]) /\
  (forall memory column address, In (memory, column) contextMemories ->
    final memory address = initial memory address).
Proof.
  intros initial final w chosen Hprime Hfull Hchoice Hchosen Hreplay.
  split.
  - pose proof (full_straight_line_local_writes w chosen Hprime Hfull Hchoice Hchosen) as Hwrites.
    destruct (Nat.eqb chosen COL_SEL_LOCAL_SET) eqn:Hset.
    + intros other index Hneq. rewrite (one_local_write_effect _ _ _ _ _ _ Hreplay Hwrites).
      apply update_other. right. congruence.
    + intros other index Hneq. eapply no_local_writes_preserve; eassumption.
  - intros memory column address Hin.
    destruct (full_selected_no_host_switch w chosen Hprime Hfull (straight_line_selector_in chosen Hchoice) Hchosen)
      as [Hsuspend [_ [_ [_ Hread]]]].
    eapply context_preserved_without_push; [exact Hin | | exact Hsuspend | exact Hread | exact Hreplay].
    exact (proj1 (full_body_context_quiet w chosen Hprime Hfull Hchoice Hchosen)).
Qed.

Lemma binary_row_frame_effect : forall initial final w chosen prefix a b,
  fieldPrime -> FullRelation.satFull w -> binaryArithmeticSelector chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [a; b]) ->
  FrameEffect (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w chosen prefix a b Hprime Hfull Hchoice Hchosen Hreplay
    [Hbase [Hsp [Hbounds Hcells]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_binary_stack_shape w chosen Hprime Hfull Hchoice Hchosen ltac:(lia)) as Hshape.
  destruct (binary_stack_replay w initial final Hshape Hreplay) as [_ [_ Hwrites]].
  assert (HlocalChoice : straightLineSelector chosen) by (destruct Hchoice as [-> | ->]; cbn; auto).
  destruct (body_row_other_memories initial final w chosen Hprime Hfull HlocalChoice Hchosen Hreplay)
    as [Hlocals Hcontext].
  constructor; try assumption.
  intros index Hindex. rewrite Hwrites.
  rewrite !update_other; [reflexivity | | ]; right; intro Heq; injection Heq as Heq.
  - change (w COL_STACK_WRITE0_ADDR_LO = 2 * index) in Heq.
    rewrite (stack_write_address w Hshape) in Heq. lia.
  - change (w COL_STACK_WRITE0_ADDR_HI = 2 * index) in Heq.
    rewrite (stack_write_hi_address w Hshape) in Heq. lia.
Qed.

Lemma local_get_row_frame_effect : forall initial final w values,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_GET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  stackMatches initial (beforeBoundary w) values ->
  FrameEffect (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w values Hprime Hfull Hchosen Hreplay [Hbase [Hsp Hrest]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  pose proof (full_local_get_stack_shape w Hprime Hfull Hchosen ltac:(lia)) as Hshape.
  pose proof (local_get_stack_replay w initial final Hshape Hreplay) as Hwrites.
  destruct (body_row_other_memories initial final w COL_SEL_LOCAL_GET Hprime Hfull
    ltac:(cbn; auto) Hchosen Hreplay) as [Hlocals Hcontext].
  constructor; try assumption.
  intros index Hindex. rewrite Hwrites.
  rewrite !update_other; [reflexivity | | ]; right; intro Heq; injection Heq as Heq.
  - change (w COL_STACK_WRITE0_ADDR_LO = 2 * index) in Heq.
    rewrite (local_write_address w false Hshape eq_refl) in Heq. lia.
  - change (w COL_STACK_WRITE0_ADDR_HI = 2 * index) in Heq.
    rewrite (local_write_hi_address w false Hshape eq_refl) in Heq. lia.
Qed.

Lemma local_set_row_frame_effect : forall initial final w prefix value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_SET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [value]) ->
  FrameEffect (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w prefix value Hprime Hfull Hchosen Hreplay [Hbase [Hsp Hrest]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_local_set_stack_shape w Hprime Hfull Hchosen ltac:(lia)) as Hshape.
  destruct (pop_stack_replay w initial final Hshape Hreplay) as [_ Hwrites].
  destruct (body_row_other_memories initial final w COL_SEL_LOCAL_SET Hprime Hfull
    ltac:(cbn; auto) Hchosen Hreplay) as [Hlocals Hcontext].
  constructor; try assumption. intros; apply Hwrites.
Qed.

Lemma initialization_frame_effect : forall initial final w chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  Replay initial (eventsForRow memories w) final ->
  FrameEffect (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final w chosen Hprime Hfull Hkind Hchosen Hreplay.
  constructor.
  - intros index Hindex. eapply initialization_preserves_stack_memory; eassumption.
  - intros other index Hneq.
    rewrite (initialization_writes_local initial final w chosen Hprime Hfull Hkind Hchosen Hreplay).
    apply update_other. right. congruence.
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
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (instruction :: rest) /\
    represents z initial (beforeBoundary row) (instruction :: rest) c0) ->
  FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
  frameRegistersPreserved row.
Proof.
  intros instruction rest z initial final row c0 Hprime Hfull Hprogram Hreplay Hcode Hframe.
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
