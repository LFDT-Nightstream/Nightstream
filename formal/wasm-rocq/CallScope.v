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

Lemma frame_effect_refl : forall base fbp memory, FrameEffect base fbp memory memory.
Proof. intros; constructor; reflexivity. Qed.

Lemma frame_effect_trans : forall base fbp initial middle final,
  FrameEffect base fbp initial middle -> FrameEffect base fbp middle final ->
  FrameEffect base fbp initial final.
Proof.
  intros base fbp initial middle final [Hs Hl Hc] [Hs' Hl' Hc']; constructor; intros.
  - rewrite Hs', Hs by assumption. reflexivity.
  - rewrite Hl', Hl by assumption. reflexivity.
  - rewrite (Hc' _ column), (Hc _ column) by assumption. reflexivity.
Qed.

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
    eapply context_preserved_without_push; [exact Hin | | exact Hreplay].
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
  destruct (local_set_stack_replay w initial final Hshape Hreplay) as [_ Hwrites].
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
  - intros memory column address Hin. eapply context_preserved_without_push; [exact Hin | | exact Hreplay].
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

Lemma final_row_follows : forall first rows last,
  Chain continuityLinks (first :: rows ++ [last]) ->
  Follows continuityLinks (finalRow first rows) last.
Proof.
  intros first rows. revert first. induction rows; intros first last Hchain; cbn in *.
  - inversion Hchain; subst; assumption.
  - apply IHrows. inversion Hchain; subst; assumption.
Qed.

Lemma initialization_rows_scope : forall rows previous initial final,
  fieldPrime -> validCircuitRun initial rows final -> Chain continuityLinks (previous :: rows) ->
  Forall (fun w : Witness => w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/ w COL_PARAM_INIT_ACTIVE_BEFORE = 1) rows ->
  FrameEffect (previous COL_STACK_FRAME_BASE_AFTER) (previous COL_LOCALS_FBP_AFTER) initial final /\
  (finalRow previous rows) COL_CALL_STACK_DEPTH_AFTER = previous COL_CALL_STACK_DEPTH_AFTER /\
  (finalRow previous rows) COL_STACK_FRAME_BASE_AFTER = previous COL_STACK_FRAME_BASE_AFTER /\
  (finalRow previous rows) COL_LOCALS_FBP_AFTER = previous COL_LOCALS_FBP_AFTER /\
  (finalRow previous rows) COL_PC_AFTER = previous COL_PC_AFTER.
Proof.
  induction rows as [ | row rows IH]; intros previous initial final Hprime Hrun Hchain Hkinds.
  - destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst.
    split; [apply frame_effect_refl | repeat split; reflexivity].
  - destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (HtailChain : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
    assert (Hrow : FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial middle /\
      row COL_CALL_STACK_DEPTH_AFTER = row COL_CALL_STACK_DEPTH_BEFORE /\
      row COL_STACK_FRAME_BASE_AFTER = row COL_STACK_FRAME_BASE_BEFORE /\
      row COL_LOCALS_FBP_AFTER = row COL_LOCALS_FBP_BEFORE /\ row COL_PC_AFTER = row COL_PC_BEFORE).
    { destruct (Forall_inv Hkinds) as [Hzero | Hparam].
      - destruct (full_initialization_preserves_frame row COL_LOCAL_ZERO_ACTIVE_BEFORE
          Hprime Hfull ltac:(left; reflexivity) Hzero) as [Hf [Hb Hd]].
        split; [eapply initialization_frame_effect; [exact Hprime | exact Hfull | left; reflexivity | exact Hzero | exact Hreplay] | ].
        repeat split; try assumption. apply full_zero_pc_preserved; assumption.
      - destruct (full_initialization_preserves_frame row COL_PARAM_INIT_ACTIVE_BEFORE
          Hprime Hfull ltac:(right; reflexivity) Hparam) as [Hf [Hb Hd]].
        split; [eapply initialization_frame_effect; [exact Hprime | exact Hfull | right; reflexivity | exact Hparam | exact Hreplay] | ].
        repeat split; try assumption. apply full_parameter_pc_preserved; assumption. }
    destruct Hrow as [Heffect [Hd [Hb [Hf Hp]]]].
    destruct (IH row middle final Hprime Htail HtailChain (Forall_inv_tail Hkinds))
      as [Hrest [Hd' [Hb' [Hf' Hp']]]].
    pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)) as Hdl.
    pose proof (Hlink COL_STACK_FRAME_BASE_AFTER COL_STACK_FRAME_BASE_BEFORE ltac:(vm_compute; tauto)) as Hbl.
    pose proof (Hlink COL_LOCALS_FBP_AFTER COL_LOCALS_FBP_BEFORE ltac:(vm_compute; tauto)) as Hfl.
    pose proof (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)) as Hpl.
    split.
    + rewrite Hbl, Hfl. rewrite Hb, Hf in Hrest. eapply frame_effect_trans; eassumption.
    + cbn [finalRow]. repeat split; lia.
Qed.

Definition bodyRow (w : Witness) : Prop :=
  (exists chosen, straightLineSelector chosen /\ w chosen = 1) \/
  w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/ w COL_PARAM_INIT_ACTIVE_BEFORE = 1.

Lemma completed_initialization_forces_body_opcode : forall previous row,
  fieldPrime -> FullRelation.satFull previous ->
  Follows continuityLinks previous row ->
  previous COL_LOCAL_ZERO_REMAINING_AFTER = 0 ->
  previous COL_PARAM_INIT_REMAINING_AFTER = 0 -> bodyRow row ->
  exists chosen, straightLineSelector chosen /\ row chosen = 1.
Proof.
  intros previous row Hprime Hprevious Hlink Hz Hp Hkind.
  destruct (full_initialization_flags previous Hprime Hprevious) as [HzFlag HpFlag].
  rewrite Hz in HzFlag. rewrite Hz, Hp in HpFlag.
  destruct (initialization_follows_counts previous row Hlink) as [_ [_ [HzLink HpLink]]].
  cbn in HzFlag, HpFlag. destruct Hkind as [Hbody | [Hzero | Hparam]]; [exact Hbody | congruence | congruence].
Qed.

Lemma full_body_initialization_done : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> straightLineSelector chosen -> w chosen = 1 ->
  w COL_LOCAL_ZERO_REMAINING_BEFORE = 0 -> w COL_PARAM_INIT_REMAINING_BEFORE = 0 ->
  w COL_LOCAL_ZERO_REMAINING_AFTER = 0 /\ w COL_PARAM_INIT_REMAINING_AFTER = 0.
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen Hz Hp.
  pose proof (straight_line_selector_in chosen Hchoice) as Hin.
  pose proof (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen) as [Hparam [Hzero _]].
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { destruct Hchoice as [<- | [<- | [<- | [<- | []]]]];
      match goal with Hchosen : value w ?chosen = 1 |- _ =>
        selected_row chosen guestEntryFlagRow Hrow;
        pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE); field_linear Hrow end. }
  split.
  - circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_TURN_BOUNDARY, -1);
      (COL_LOCAL_ZERO_ACTIVE_BEFORE, -1)],
      [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hguest, Hturn, Hzero, Hz in Hrow.
    pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_AFTER). field_linear Hrow.
  - circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_PARAM_INIT_ACTIVE_BEFORE, -1);
      (COL_TURN_BOUNDARY, -1)],
      [(COL_PARAM_INIT_REMAINING_AFTER, 1); (COL_PARAM_INIT_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hguest, Hturn, Hparam, Hp in Hrow.
    pose proof (value_bound w COL_PARAM_INIT_REMAINING_AFTER). field_linear Hrow.
Qed.

Lemma body_rows_initialized : forall rows previous initial final,
  fieldPrime -> FullRelation.satFull previous -> validCircuitRun initial rows final ->
  Chain continuityLinks (previous :: rows) ->
  previous COL_LOCAL_ZERO_REMAINING_AFTER = 0 -> previous COL_PARAM_INIT_REMAINING_AFTER = 0 ->
  Forall bodyRow rows ->
  Forall (fun w : Witness => exists chosen, straightLineSelector chosen /\ w chosen = 1) rows.
Proof.
  induction rows as [ | row rows IH]; intros previous initial final Hprime Hprevious Hrun Hchain Hz Hp Hkinds;
    [constructor | ].
  destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
  assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
  assert (HtailChain : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
  destruct (completed_initialization_forces_body_opcode previous row Hprime Hprevious Hlink Hz Hp
    (Forall_inv Hkinds)) as [chosen [Hchoice Hselected]].
  constructor; [exists chosen; auto | ].
  destruct (initialization_follows_counts previous row Hlink) as [HzLink [HpLink _]].
  destruct (full_body_initialization_done row chosen Hprime Hfull Hchoice Hselected
    ltac:(lia) ltac:(lia)) as [HzAfter HpAfter].
  eapply IH; eassumption || exact (Forall_inv_tail Hkinds).
Qed.

Lemma frame_row_stack_delta : forall instruction rest z initial row c0,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  encodedOpcode instruction = Some (row COL_OPCODE_CODE) ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (instruction :: rest) /\
    represents z initial (beforeBoundary row) (instruction :: rest) c0) ->
  instructionStackDelta instruction = Some (row COL_SP_AFTER - row COL_SP_BEFORE).
Proof.
  intros instruction rest z initial row c0 Hprime Hfull Hprogram Hcode
    Hframe. apply represented_values in Hframe.
  destruct Hframe as [values [[Hbase [Hsp Hstack]] [Htyped Hrest]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
  destruct (encoded_opcode_cases instruction _ Hcode)
    as [[-> HcodeValue] | [[-> HcodeValue] | [[x [-> HcodeValue]] | [x [-> HcodeValue]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_ADD); try eassumption; reflexivity. }
    assert (Hheight : (2 <= length values)%nat).
    { destruct (length values) as [ | [ | n]]; cbn [programTyped instructionStackHeight] in Htyped; tauto || lia. }
    pose proof (full_binary_stack_shape row COL_SEL_I32_ADD Hprime Hfull ltac:(left; reflexivity) Hchosen ltac:(lia)) as Hshape.
    pose proof (stack_pointer_after row Hshape). cbn [instructionStackDelta]; f_equal; lia.
  - assert (Hchosen : row COL_SEL_I32_SUB = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_SUB); try eassumption; reflexivity. }
    assert (Hheight : (2 <= length values)%nat).
    { destruct (length values) as [ | [ | n]]; cbn [programTyped instructionStackHeight] in Htyped; tauto || lia. }
    pose proof (full_binary_stack_shape row COL_SEL_I32_SUB Hprime Hfull ltac:(right; reflexivity) Hchosen ltac:(lia)) as Hshape.
    pose proof (stack_pointer_after row Hshape). cbn [instructionStackDelta]; f_equal; lia.
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_GET); try eassumption; reflexivity. }
    pose proof (full_local_get_stack_shape row Hprime Hfull Hchosen ltac:(lia)) as Hshape.
    pose proof (local_stack_pointer_after row false Hshape) as Hnext.
    change (row COL_SP_AFTER = row COL_SP_BEFORE + 1) in Hnext.
    cbn [instructionStackDelta]; f_equal; lia.
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1).
    { eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_SET); try eassumption; reflexivity. }
    cbn [programTyped instructionStackHeight] in Htyped. destruct Htyped as [_ Htyped].
    assert (Hheight : (1 <= length values)%nat) by (destruct (length values); [contradiction | lia]).
    pose proof (full_local_set_stack_shape row Hprime Hfull Hchosen ltac:(lia)) as Hshape.
    pose proof (local_stack_pointer_after row true Hshape). cbn [instructionStackDelta] in *; f_equal; lia.
Qed.

Lemma chain_after_prefix : forall prefix first suffix,
  Chain continuityLinks (first :: prefix ++ suffix) ->
  Chain continuityLinks (finalRow first prefix :: suffix).
Proof.
  induction prefix as [ | row rows IH]; intros first suffix Hchain; [exact Hchain | ].
  apply IH. inversion Hchain; subst; assumption.
Qed.

Lemma final_row_satisfies : forall rows first (P : Witness -> Prop),
  Forall P (first :: rows) -> P (finalRow first rows).
Proof.
  induction rows; intros first P H; [exact (Forall_inv H) | ].
  cbn [finalRow]. eapply IHrows. exact (Forall_inv_tail H).
Qed.
