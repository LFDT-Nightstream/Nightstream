(* Reuse the instruction proofs inside a function whose code has an arbitrary entry PC. *)
From Stdlib Require Import List ZArith Lia.
Require Import CallSpec R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub Locals
  StraightLineSpec StackSimulation LocalSimulation LocalBindings StraightLineInstructions SpecTecCalls CallScope.
Import ListNotations.
Open Scope Z_scope.

Lemma frame_row_refinement : forall instruction rest z initial final row c0,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  instructionImage initial (row COL_PC_BEFORE) instruction ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (instruction :: rest) /\
    represents z initial (beforeBoundary row) (instruction :: rest) c0) ->
  exists z' c1, Step c0 c1 /\ (programTyped (length (stateLocals z')) (stackHeight (afterBoundary row)) rest /\
    represents z' final (afterBoundary row) rest c1) /\
    row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
    frameRegistersPreserved row /\
    instructionStackDelta instruction = Some (row COL_SP_AFTER - row COL_SP_BEFORE) /\
    fun_store z' = fun_store z /\ fun_moduleinst z' = fun_moduleinst z /\
    length (stateLocals z') = length (stateLocals z).
Proof.
  assert (Hstore : forall z x value, fun_store (with_local z x value) = fun_store z).
  { intros [s f] x value. reflexivity. }
  intros instruction rest z initial final row c0 Hprime Hfull Hprogram Hreplay
    [Hop [Hkind [Hnext Hlocal]]] Hframe.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch.
  pose proof (frame_row_stack_delta instruction rest z initial row c0 Hprime Hfull Hprogram Hfetch Hframe) as Hdelta.
  pose proof (frame_row_scope instruction rest z initial final row c0
    Hprime Hfull Hprogram Hreplay Hfetch Hframe) as Hscope.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindRead.
  rewrite Hkind in HkindRead.
  assert (Hstatic : row COL_PC_EDGE_KIND = 0) by congruence.
  pose proof (full_static_pc_rom_active row Hprime Hfull Hprogram Hstatic) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hactive) as HpcRead.
  rewrite Hnext in HpcRead. destruct (row COL_CONTROL_CHOICE =? 0); [ | discriminate].
  assert (Hpc : row COL_PC_AFTER = row COL_PC_BEFORE + 1) by congruence.
  destruct (encoded_opcode_cases instruction _ Hfetch)
    as [[-> Hcode] | [[-> Hcode] | [[x [-> Hcode]] | [x [-> Hcode]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1).
    { eapply full_opcode_selects with (code := CODE_I32_ADD); try eassumption; reflexivity. }
    destruct (binary_frame_refinement z initial final row c0 _ rest COL_SEL_I32_ADD
      Hprime Hfull Hreplay ltac:(left; reflexivity) Hchosen
      (Add.full_relation_implies_spectec_add row Hprime Hchosen Hfull) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_I32_SUB = 1).
    { eapply full_opcode_selects with (code := CODE_I32_SUB); try eassumption; reflexivity. }
    destruct (binary_frame_refinement z initial final row c0 _ rest COL_SEL_I32_SUB
      Hprime Hfull Hreplay ltac:(right; reflexivity) Hchosen
      (Sub.full_relation_implies_spectec_sub row Hprime Hchosen Hfull) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1).
    { eapply full_opcode_selects with (code := CODE_LOCAL_GET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_GET) by (cbn; auto).
    destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [_ [_ Hgate]].
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as HindexRead.
    rewrite Hlocal in HindexRead.
    destruct (local_get_frame_refinement z initial final row c0 x rest Hprime Hfull Hreplay
      Hchosen ltac:(congruence) Hframe) as [c1 [Hstep Hafter]].
    exists z, c1. split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | repeat split; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1).
    { eapply full_opcode_selects with (code := CODE_LOCAL_SET); try eassumption; reflexivity. }
    assert (Hchoice : straightLineSelector COL_SEL_LOCAL_SET) by (cbn; auto).
    destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [_ [_ Hgate]].
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as HindexRead.
    rewrite Hlocal in HindexRead.
    destruct (local_set_frame_refinement z initial final row c0 x rest Hprime Hfull Hreplay
      Hchosen ltac:(congruence) Hframe) as [c1 [Hstep Hafter]].
    exists (with_local z x (i32Value (row COL_LOCAL_VALUE))), c1.
    split; [exact Hstep | ]. split; [exact Hafter | ].
    split; [exact Hpc | ]. split; [exact (proj1 Hscope) | ].
    split; [exact (proj2 Hscope) | ]. split; [exact Hdelta | ]. split; [apply Hstore | ]. split.
    + destruct z as [s f]. reflexivity.
    + apply locals_length_after_set.
Qed.

Lemma instruction_row_next_pc : forall initial final w instruction,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories w) final ->
  instructionImage initial (w COL_PC_BEFORE) instruction ->
  w COL_PC_AFTER = w COL_PC_BEFORE + 1.
Proof.
  intros initial final w instruction Hprime Hfull Hprogram Hreplay [_ [Hkind [Hnext _]]].
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hkind in Hfetch.
  pose proof (full_static_pc_rom_active w Hprime Hfull Hprogram ltac:(congruence)) as Hactive.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hactive) as Hread.
  rewrite Hnext in Hread. destruct (w COL_CONTROL_CHOICE =? 0); congruence.
Qed.
