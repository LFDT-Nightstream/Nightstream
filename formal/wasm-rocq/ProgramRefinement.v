(* Public refinement: one program ROM binding, one row at a time. Each row advances
   the active frame's code, enters or initializes a callee, or returns to its caller. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerRefinement ProgramSpec ProgramEncoding CallContext CallScope Fragment
  ControlDecode ControlInvariant FrameFacts ControlBridge FrameRows CallRows.
Import ListNotations.
Open Scope Z_scope.

Lemma supported_cases : forall callees i, Supported callees i ->
  (exists op, encodedOpcode i = Some op) \/ (exists x, i = CALL x) \/
  (exists body, i = BLOCK (_RESULT None) body) \/ (exists l, i = BR_IF l) \/ i = RETURN.
Proof.
  intros callees i H. inversion H; subst;
    first [solve [left; eexists; reflexivity] | solve [right; left; eauto] | solve [right; right; left; eauto] |
      solve [right; right; right; left; eauto] | solve [right; right; right; right; reflexivity]].
Qed.

Section Run.
Variables (program : Program) (s : store).
Hypothesis Hstore : store_FUNCS s = program.
Hypothesis Hwf : wf_store s.
Hypothesis Hvalid : programValidated program s.

(* One row of a run that stays inside the base frame. *)
Lemma control_row : forall baseDepth cursor initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final -> callerRow baseDepth row ->
  ControlState program s baseDepth cursor initial (beforeBoundary row) c0 ->
  exists cursor' c1, Steps c0 c1 /\ ControlState program s baseDepth cursor' final (afterBoundary row) c1.
Proof.
  intros baseDepth [stack active] initial final row c0 Hprime Hloaded Hfull Hreplay [Hkind [Hhalt Hbase]] Hstate.
  pose proof Hstate as Hwhole.
  destruct Hstate as [f [inner [Hactive [Hactivestate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in Hactive, Hactivestate, Hsusp, Hdepth.
  cbn [beforeBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth] in Hsusp, Hdepth.
  (* Padding stutters in any state. *)
  destruct (Z.eq_dec (row COL_PADDING_ACTIVE) 1) as [Hpadding | Hnotpadding].
  { eapply padding_row; eassumption. }
  destruct active as [view | function arguments].
  2: eapply init_row; eassumption.
  cbn [activeFunction] in Hactive.
  pose proof Hactivestate as [f' [Hframe [Hoffset [_ [_ [[Hma Hza] [Hzr _]]]]]]].
  pose proof Hframe as [Hf' _]. rewrite Hactive in Hf'. injection Hf' as <-.
  cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining boundaryZeroActive boundaryZeroRemaining]
    in Hma, Hza, Hzr.
  (* A pending move runs its rows before the frame's code resumes. *)
  destruct (Z.eq_dec (row COL_FRAME_MOVE_ACTIVE_BEFORE) 1) as [Hmove | Hidle].
  { eapply move_row; eassumption. }
  assert (Hprogram : row COL_IS_PROGRAM_ROW = 1).
  { rewrite Hzr in Hza. destruct (row COL_FRAME_MOVE_REMAINING_BEFORE =? 0); cbn in Hza;
      destruct Hkind as [Hp | [Hz | [Hq | Hpad]]]; first [exact Hp | lia]. }
  (* A row that stays in the active frame keeps every suspended caller. *)
  assert (Hkeep : RowResult program s initial final row f inner view ->
      exists cursor' c1, Steps (plugCalls stack inner) c1 /\
        ControlState program s baseDepth cursor' final (afterBoundary row) c1).
  { intros [view' [inner' [Hf' [Hsteps [Hstate' [[Hd [Hb Hfbp]] Heffect]]]]]].
    exists (controlCursor stack (atCode view')), (plugCalls stack inner').
    split; [eapply steps_in_plug_calls; try exact Hsteps; try eassumption; eapply suspended_stack_ready; eassumption | ].
    exists f, inner'. split; [exact Hf' | ]. split; [exact Hstate' | ].
    split; [ | split; [exact Hbase0 | split; [cbn [afterBoundary boundaryCallDepth suspendedCalls] in *; lia |
      reflexivity]]].
    cbn [suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth].
    rewrite Hd, Hfbp. eapply frame_effect_suspended; [exact Heffect | exact Hsusp]. }
  destruct (nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view))) as [i | ] eqn:Hi.
  - destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [[Hsup _] _]].
    rewrite (skipn_at_instruction _ _ _ Hi) in Hsup.
    destruct (supported_cases _ _ (Forall_inv Hsup)) as [Hop | [[x ->] | [[body ->] | [[l ->] | ->]]]].
    + apply Hkeep. eapply code_ordinary_row; eassumption.
    + eapply call_row; eassumption.
    + apply Hkeep. eapply code_block_enter_row; eassumption.
    + apply Hkeep. eapply code_branch_row; eassumption.
    + (* A return pops back to its caller; in the base frame it would leave the run. *)
      destruct stack as [ | top rest].
      * exfalso. pose proof (Hbase ltac:(cbn in Hdepth; lia)) as Hpop.
        destruct (return_selected program s view f initial final row _ Hprime Hloaded Hfull
          Hprogram Hreplay Hactivestate Hactive Hi) as [Hsel Hedge].
        destruct (full_return_exits row COL_SEL_RETURN Hprime Hfull ltac:(cbn; tauto) Hsel Hedge); lia.
      * eapply early_return_row; eassumption.
  - apply nth_error_None in Hi.
    assert (Hend : positionOffset (viewPosition view) = length (innerCode f (viewPosition view))) by lia.
    destruct (positionLabels (viewPosition view)) as [ | b more] eqn:Hlabels.
    + (* The function's own END. *)
      assert (Hbody : innerCode f (viewPosition view) = guestBody f) by (unfold innerCode; rewrite Hlabels; reflexivity).
      rewrite Hbody in Hend.
      destruct stack as [ | top rest].
      * exfalso. pose proof (Hbase ltac:(cbn in Hdepth; lia)) as Hpop.
        destruct (function_end_selected program s view f initial final row _ Hprime Hloaded Hfull
          Hprogram Hreplay Hactivestate Hactive Hlabels Hend) as [HselEnd Hedge].
        destruct (full_return_exits row COL_SEL_END Hprime Hfull ltac:(cbn; tauto) HselEnd Hedge); lia.
      * eapply return_row; eassumption.
    + apply Hkeep. eapply code_block_end_row; try eassumption. rewrite Hlabels. discriminate.
Qed.

Lemma rows_final_boundary_app : forall prefix suffix boundary,
  rowsFinalBoundary boundary (prefix ++ suffix) =
    rowsFinalBoundary (rowsFinalBoundary boundary prefix) suffix.
Proof. induction prefix; intros; cbn; auto. Qed.

(* Every row is consumed, over the relational invariant. *)
Lemma control_run : forall baseDepth cursor rows initial final boundary c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  runFragment boundary baseDepth rows ->
  ControlState program s baseDepth cursor initial boundary c0 ->
  exists cursor' c1, Steps c0 c1 /\ programLoaded program final /\
    ControlState program s baseDepth cursor' final (rowsFinalBoundary boundary rows) c1.
Proof.
  intros baseDepth cursor rows. revert cursor.
  induction rows as [ | row rows IH]; intros cursor initial final boundary c0 Hprime Hloaded Hrun
    [Hstart Hkinds] Hstate.
  - destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst final.
    exists cursor, c0. split; [apply steps_refl_wf; eapply control_state_wf; eassumption | ].
    split; [exact Hloaded | exact Hstate].
  - cbn [runStartsAt] in Hstart. subst boundary.
    destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    destruct (control_row baseDepth cursor initial middle row c0 Hprime Hloaded Hfull Hreplay
      (Forall_inv Hkinds) Hstate) as [cursor1 [c1 [Hsteps1 Hstate1]]].
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
    destruct (IH cursor1 middle final (afterBoundary row) c1 Hprime Hloaded1 Htail Hfragment Hstate1)
      as [cursor2 [c2 [Hsteps2 [Hloaded2 Hstate2]]]].
    exists cursor2, c2. split; [eapply spec_steps_transitive; eassumption | ].
    split; [exact Hloaded2 | exact Hstate2].
Qed.

End Run.

(* Every row is consumed. The run may stop anywhere inside the base frame,
   including inside blocks or calls; the final state resumes the next run. *)
Theorem run_refinement : forall program s baseDepth (shape : ControlShape) rows initial final boundary c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  runFragment boundary baseDepth rows ->
  CircuitConfig program s baseDepth shape initial boundary c0 ->
  exists shape' c1, Steps c0 c1 /\ programLoaded program final /\
    CircuitConfig program s baseDepth shape' final (rowsFinalBoundary boundary rows) c1.
Proof.
  intros program s baseDepth shape rows initial final boundary c0 Hprime Hloaded Hrun Hfragment Hconfig.
  pose proof Hconfig as [Hfuncs [Hwf [Hvalid _]]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig) as [cursor Hstate].
  destruct (control_run program s Hfuncs Hwf Hvalid _ _ _ _ _ _ _ Hprime Hloaded Hrun Hfragment Hstate)
    as [cursor' [c1 [Hsteps [Hloaded' Hstate']]]].
  exists (shapeOf cursor'), c1. split; [exact Hsteps | ]. split; [exact Hloaded' | ].
  apply control_state_circuit; assumption.
Qed.
