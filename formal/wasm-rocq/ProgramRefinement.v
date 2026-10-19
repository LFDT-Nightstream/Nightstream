(* Public refinement: one program ROM binding, one row at a time. Each row advances
   the active function, enters or initializes a callee, or returns to its caller. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerRefinement ProgramSpec ProgramEncoding
  SourceInstructions CodeLayout BlockSpec ControlDecode ControlInvariant ControlState ControlRows CallStack ControlBridge CallContext Fragment CallScope
  ProgramBinding Selectors.
Import ListNotations.
Open Scope Z_scope.

Lemma source_instruction_cases : forall callees i, Supported (functionSite callees) i ->
  (exists bt body, i = BLOCK bt body) \/ (exists x, i = CALL x) \/ (exists op, encodedOpcode i = Some op).
Proof.
  intros callees i H. inversion H; subst;
    first [right; right; eexists; reflexivity | solve [right; left; eauto] | solve [left; eauto]].
Qed.

Lemma active_state_function : forall program cursor z memory boundary c,
  ActiveState program cursor z memory boundary c -> exists f, activeFunction program cursor = Some f.
Proof.
  intros program [position | position offset prefix | function arguments] z memory boundary c H; cbn.
  - destruct H as [[code H] _]. unfold SourceState in H.
    destruct (functionAt program (sourceFunction position)) as [f | ]; [exists f; reflexivity | contradiction].
  - destruct H as [f [_ [_ [_ [Hf _]]]]]. exists f. exact Hf.
  - destruct H as [f [Hf _]]. exists f. exact Hf.
Qed.

(* A run starting in a frame with no suspended callers: that frame is the base. *)
Lemma base_control_state : forall program cursor z memory boundary c,
  ActiveState program cursor z memory boundary c -> 0 <= boundaryCallDepth boundary ->
  ControlState program (boundaryCallDepth boundary) (controlCursor [] cursor) z memory boundary c.
Proof.
  intros program cursor z memory boundary c Hstate Hdepth.
  destruct (active_state_function _ _ _ _ _ _ Hstate) as [f Hf].
  exists f, c. split; [exact Hf | ]. split; [exact Hstate | ]. split; [exact I | ].
  split; [exact Hdepth | ]. split; [cbn; lia | reflexivity].
Qed.

(* A function-ending END exits its frame: it pops a context or halts. *)
Lemma function_end_exits_row : forall program position z initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  SourceState program position z initial (beforeBoundary row) [] inner ->
  row COL_CALL_STACK_POP_PRESENT = 1 \/ row COL_HALTED = 1.
Proof.
  intros program [index offset] z initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hsource.
  unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource.
  destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
  destruct Hsource as [_ [[Hbound Hcode] [_ [_ Hpc]]]].
  assert (Hoffset : offset = length (guestBody f)).
  { pose proof (f_equal (@length instr) Hcode) as Hlength. rewrite length_skipn in Hlength. cbn in Hlength; lia. }
  subst offset. rewrite code_offset_end in Hpc.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))) in Hpc.
  destruct (encoded_function_end _ _ _ _ Hloaded Hf) as [Hop Hedge].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hcode'.
  assert (Hend : row COL_SEL_END = 1).
  { apply (full_opcode_selects row COL_SEL_END CODE_END Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact Hcode']. }
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
  rewrite Hpc, Hedge in Hkind. injection Hkind as Hkind.
  apply full_function_end_exits; [exact Hprime | exact Hfull | exact Hend | lia].
Qed.

(* One row of a run that stays inside the base frame. *)
Lemma control_row : forall program baseDepth cursor z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final -> callerRow baseDepth row ->
  ControlState program baseDepth cursor z initial (beforeBoundary row) c0 ->
  exists cursor' z' c1, fun_store z' = fun_store z /\ Steps c0 c1 /\ ControlState program baseDepth cursor' z' final (afterBoundary row) c1.
Proof.
  intros program baseDepth [stack active] z initial final row c0 Hprime Hloaded Hfull Hreplay
    [Hkind [Hhalt Hbase]] Hstate.
  pose proof Hstate as Hwhole.
  destruct Hstate as [f [inner [Hactive [Hactivestate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in Hactive, Hactivestate, Hsusp, Hdepth.
  cbn [beforeBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth] in Hsusp, Hdepth.
  assert (HbaseFrame : stack = [] -> row COL_IS_PROGRAM_ROW = 1 /\ row COL_CALL_STACK_POP_PRESENT = 0)
    by (intros ->; apply Hbase; cbn in Hdepth; lia).
  assert (Hidle_program : initializationIdle (beforeBoundary row) -> row COL_IS_PROGRAM_ROW = 1).
  { intros [_ [_ [Hza Hpa]]]. cbn in Hza, Hpa. destruct Hkind as [Hp | [Hz | Hq]]; [exact Hp | lia | lia]. }
  (* A row that stays in the active function, keeps the frame registers and leaves
     the memory below the frame alone keeps every suspended caller. *)
  assert (Hkeep : forall position' z' inner', activeFunction program position' = Some f ->
      fun_store z' = fun_store z -> Steps inner inner' ->
      ActiveState program position' z' final (afterBoundary row) inner' -> frameRegistersPreserved row ->
      FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final ->
      exists cursor' z'' c1, fun_store z'' = fun_store z /\ Steps (plugCalls stack inner) c1 /\
        ControlState program baseDepth cursor' z'' final (afterBoundary row) c1).
  { intros position' z' inner' Hf' Hstore Hsteps Hstate' [Hd [Hb Hfbp]] Heffect.
    exists (controlCursor stack position'), z', (plugCalls stack inner').
    split; [exact Hstore | ].
    split; [apply steps_in_plug_calls; [eapply suspended_stack_ready; exact Hsusp | exact Hsteps] | ].
    exists f, inner'. split; [exact Hf' | ]. split; [exact Hstate' | ].
    split; [ | split; [exact Hbase0 | split; [cbn [afterBoundary boundaryCallDepth suspendedCalls] in *; lia |
      reflexivity]]].
    cbn [suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth].
    rewrite Hd, Hb, Hfbp, Hstore. eapply frame_effect_suspended; [exact Heffect | exact Hsusp]. }
  destruct active as [[index offset] | position offset prefix | function arguments].
  - (* An outer instruction, or the function's own END. *)
    destruct Hactivestate as [[code Hsource] Hidle].
    pose proof (Hidle_program Hidle) as Hprogram.
    cbn [activeFunction sourceFunction] in Hactive.
    destruct code as [ | i rest].
    + destruct stack as [ | top more].
      * exfalso. destruct (HbaseFrame eq_refl) as [_ Hpop].
        destruct (function_end_exits_row _ _ _ _ _ _ _ Hprime Hloaded Hfull Hprogram Hreplay Hsource); lia.
      * exact (return_row_step program baseDepth top more (sourcePosition index offset) z initial final row inner f
          Hprime Hloaded Hfull Hprogram Hreplay Hactive Hsource Hidle Hsusp Hbase0
          ltac:(cbn [length] in Hdepth; exact Hdepth)).
    + pose proof Hsource as Hparts. unfold SourceState in Hparts. cbn [sourceFunction sourceOffset] in Hparts.
      rewrite Hactive in Hparts. destruct Hparts as [_ [Hcode [Hvalid _]]].
      destruct (source_instruction_cases _ _ (Forall_inv (proj1 (proj2 Hvalid))))
        as [[bt [body ->]] | [[x ->] | Hordinary]].
      * destruct (block_enter_row program (sourcePosition index offset) bt body rest z initial final row inner
          Hprime Hloaded Hfull Hreplay Hprogram Hhalt Hsource Hidle) as [prefix [c1 [Hsteps [Hstate' [Hregs Heffect]]]]].
        exact (Hkeep (inBlock (sourcePosition index offset) 0 prefix) z c1 Hactive eq_refl Hsteps Hstate'
          Hregs Heffect).
      * exact (call_row_enters program baseDepth stack index offset x rest f z initial final row inner Hprime
          Hloaded Hfull Hprogram Hreplay Hactive Hsource Hsusp Hbase0 Hdepth).
      * destruct (source_ordinary_row program index offset i rest z initial final row inner Hprime Hloaded Hfull
          Hprogram Hreplay Hordinary Hsource Hidle) as [z' [c1 [Hsteps [Hsource' [Hidle' [Heffect [Hregs Hstore]]]]]]].
        exact (Hkeep (atInstruction (sourcePosition index (S offset))) z' c1 Hactive Hstore Hsteps
          (conj (ex_intro _ rest Hsource') Hidle') Hregs Heffect).
  - (* Inside a block. *)
    pose proof Hactivestate as Hcopy.
    destruct Hcopy as [? [? [? [? [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [Hidle _]]]]]]]]]]]]]]]]]].
    pose proof (Hidle_program Hidle) as Hprogram.
    destruct (block_control_row program position offset prefix z initial final row inner Hprime Hloaded Hfull
      Hreplay Hprogram Hhalt Hactivestate) as [cursor' [z' [c1 [Hsame [Hstore [Hsteps [Hstate' [Hregs Heffect]]]]]]]].
    cbn [activeFunction] in Hactive.
    exact (Hkeep cursor' z' c1 (eq_trans Hsame Hactive) Hstore Hsteps Hstate' Hregs Heffect).
  - (* A callee whose frame is being initialized. *)
    destruct (init_row_step program baseDepth stack function arguments z initial final row _ Hprime Hloaded Hfull
      Hreplay Hwhole) as [cursor' [c1 [Hsteps Hstate']]].
    exists cursor', z, c1. split; [reflexivity | split; assumption].
Qed.

Lemma rows_final_boundary_app : forall prefix suffix boundary,
  rowsFinalBoundary boundary (prefix ++ suffix) =
    rowsFinalBoundary (rowsFinalBoundary boundary prefix) suffix.
Proof. induction prefix; intros; cbn; auto. Qed.

(* Every row is consumed, over the relational invariant. *)
Lemma control_run : forall program baseDepth cursor rows z initial final boundary c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  runFragment boundary baseDepth rows ->
  ControlState program baseDepth cursor z initial boundary c0 ->
  exists cursor' z' c1, fun_store z' = fun_store z /\ Steps c0 c1 /\ programLoaded program final /\
    ControlState program baseDepth cursor' z' final (rowsFinalBoundary boundary rows) c1.
Proof.
  intros program baseDepth cursor rows. revert cursor.
  induction rows as [ | row rows IH]; intros cursor z initial final boundary c0 Hprime Hloaded Hrun
    [Hstart Hkinds] Hstate.
  - destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst.
    exists cursor, z, c0. split; [reflexivity | ]. split; [apply steps_refl_wf; eapply control_state_wf; exact Hstate | ].
    split; [exact Hloaded | exact Hstate].
  - cbn [runStartsAt] in Hstart. subst boundary.
    destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    destruct (control_row program baseDepth cursor z initial middle row c0 Hprime Hloaded Hfull Hreplay
      (Forall_inv Hkinds) Hstate) as [cursor1 [z1 [c1 [Hstore1 [Hsteps1 Hstate1]]]]].
    assert (Hsingle : validCircuitRun initial [row] middle).
    { split; [constructor; [exact Hfull | constructor] | ]. split; [constructor | ].
      cbn [eventsForTrace flat_map]. rewrite app_nil_r. exact Hreplay. }
    assert (Hloaded1 : programLoaded program middle)
      by (eapply program_loaded_preserved; [eapply run_preserves_roms; exact Hsingle | exact Hloaded]).
    assert (Hfragment : runFragment (afterBoundary row) baseDepth rows).
    { split; [ | exact (Forall_inv_tail Hkinds)].
      destruct rows as [ | next rows]; [exact I | ].
      cbn [runStartsAt]. symmetry. apply follows_boundary.
      destruct Hrun as [_ [Hchain _]]. inversion Hchain; subst; assumption. }
    destruct (IH cursor1 z1 middle final (afterBoundary row) c1 Hprime Hloaded1 Htail Hfragment Hstate1)
      as [cursor2 [z2 [c2 [Hstore2 [Hsteps2 [Hloaded2 Hstate2]]]]]].
    exists cursor2, z2, c2. split; [congruence | ]. split; [eapply spec_steps_transitive; eassumption | ].
    split; [exact Hloaded2 | exact Hstate2].
Qed.

(* Every row is consumed. The run may stop anywhere inside the base frame,
   including inside a block or a call; the final state resumes the next run. *)
Theorem run_refinement : forall program s baseDepth (shape: ControlShape) rows initial final boundary c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  runFragment boundary baseDepth rows ->
  CircuitConfig program s baseDepth shape initial boundary c0 ->
  exists shape' c1, Steps c0 c1 /\ programLoaded program final /\
    CircuitConfig program s baseDepth shape' final (rowsFinalBoundary boundary rows) c1.
Proof.
  intros program s baseDepth shape rows initial final boundary c0 Hprime Hloaded Hrun Hfragment Hconfig.
  pose proof Hconfig as [Hfuncs [Hwf _]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig) as [cursor [z [Hstore Hstate]]].
  destruct (control_run _ _ _ _ _ _ _ _ _ Hprime Hloaded Hrun Hfragment Hstate)
    as [cursor' [z' [c1 [Hstore' [Hsteps [Hloaded' Hstate']]]]]].
  exists (shapeOf cursor'), c1. split; [exact Hsteps | ]. split; [exact Hloaded' | ].
  apply control_state_circuit with (z := z'); [exact Hfuncs | exact Hwf | congruence | exact Hstate'].
Qed.
