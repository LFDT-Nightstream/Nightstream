(* One public refinement theorem for running, trapped and returned activations.
   Only the last row may terminate; the proof derives its source instruction and
   composes that terminal step with the running prefix. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerRefinement ProgramSpec ProgramEncoding ControlDecode
  ControlInvariant ControlBridge ProgramRefinement ExecutionSpec TerminalRows.
Import ListNotations.
Open Scope Z_scope.

Lemma shape_root_cursor : forall cursor, shapeRoot (shapeOf cursor) = cursorRoot cursor.
Proof.
  assert (Hroot : forall stack active,
      rootSite (map (fun top => viewPosition (callView top)) stack) active = rootFunction stack active).
  { induction stack as [ | top rest IH]; intros active; cbn [map rootSite rootFunction]; auto. }
  intros [stack position]. unfold shapeRoot, shapeOf, cursorRoot. cbn [callSites activeSite suspendedCalls activePosition].
  destruct position; cbn [siteOf siteFunction positionFunctionIndex]; apply Hroot.
Qed.

Lemma execution_rows_split : forall baseDepth rows,
  executionRows baseDepth rows ->
  Forall (callerRow baseDepth) rows \/ exists prefix last,
    rows = prefix ++ [last] /\ Forall (callerRow baseDepth) prefix /\ terminalRow baseDepth last.
Proof.
  intros baseDepth rows. induction rows as [ | row rows IH]; intros Hrows; [left; constructor | ].
  destruct rows as [ | next rest].
  - destruct Hrows as [Hrow | Hrow].
    + left; constructor; [exact Hrow | constructor].
    + right; exists [], row; split; [reflexivity | ]. split; [constructor | exact Hrow].
  - destruct Hrows as [Hrow Hrest]. destruct (IH Hrest) as [Hactive | [prefix [last [Heq [Hprefix Hlast]]]]].
    + left; constructor; assumption.
    + right; exists (row :: prefix), last. split; [cbn; rewrite Heq; reflexivity | ].
      split; [constructor; assumption | exact Hlast].
Qed.

Lemma prefix_starts_at : forall prefix suffix boundary,
  runStartsAt boundary (prefix ++ suffix) -> runStartsAt boundary prefix.
Proof. intros [ | row rest] suffix boundary H; [exact I | exact H]. Qed.

Lemma boundary_before_last : forall prefix row boundary,
  Chain continuityLinks (prefix ++ [row]) -> runStartsAt boundary (prefix ++ [row]) ->
  rowsFinalBoundary boundary prefix = beforeBoundary row.
Proof.
  induction prefix as [ | first prefix IH]; intros row boundary Hchain Hstart; [symmetry; exact Hstart | ].
  cbn [rowsFinalBoundary]. apply IH.
  - eapply chain_suffix with (prefix := [first]); exact Hchain.
  - destruct prefix as [ | next rest]; cbn [runStartsAt]; symmetry; apply follows_boundary;
      inversion Hchain; subst; assumption.
Qed.

Theorem run_refinement : forall program s caller baseDepth root shape rows initial final boundary c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  executionFragment boundary baseDepth rows ->
  ActivationConfig program s caller baseDepth root shape initial boundary c0 ->
  exists outcome c1, Steps c0 c1 /\ programLoaded program final /\
    ExecutionConfig program s caller baseDepth root final (rowsFinalBoundary boundary rows) rows outcome c1.
Proof.
  intros program s caller baseDepth root shape rows initial final boundary c0 Hprime Hloaded Hrun [Hstart Hrows]
    [Hcaller [HrootShape [f [inner [Hroot [Hconfig ->]]]]]].
  pose proof Hconfig as [Hstore [Hwf [Hvalid _]]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig) as [cursor [Hshape Hstate]].
  assert (HrootCursor : cursorRoot cursor = root) by (rewrite <- shape_root_cursor, Hshape; exact HrootShape).
  assert (HloadedFinal : programLoaded program final)
    by (eapply program_loaded_preserved; [eapply run_preserves_roms; exact Hrun | exact Hloaded]).
  destruct (execution_rows_split _ _ Hrows) as [Hactive | [prefix [row [-> [Hprefix Hterminal]]]]].
  - destruct (control_run program s Hstore Hwf Hvalid _ _ _ _ _ _ _ Hprime Hloaded Hrun
      (conj Hstart Hactive) Hstate) as [cursor' [inner' [Hroot' [Hsteps [_ Hstate']]]]].
    exists (running (shapeOf cursor')), (callContext caller (N.of_nat (guestResults f)) inner').
    split; [apply steps_in_call_frame; assumption | ]. split; [exact HloadedFinal | ].
    split.
    { unfold lastRow. pose proof (Forall_rev Hactive) as Hrev.
      destruct (rev rows) as [ | last rest]; [exact I | exact (proj1 (proj2 (Forall_inv Hrev)))]. }
    split; [exact Hcaller | ]. split; [rewrite shape_root_cursor, Hroot'; exact HrootCursor | ].
    exists f, inner'. split; [exact Hroot | ]. split; [ | reflexivity].
    apply control_state_circuit; assumption.
  - destruct (valid_run_split prefix [row] initial final Hrun) as [middle [HrunPrefix HrunRow]].
    destruct (control_run program s Hstore Hwf Hvalid _ _ _ _ _ _ _ Hprime Hloaded HrunPrefix
      (conj (prefix_starts_at _ _ _ Hstart) Hprefix) Hstate)
      as [cursor' [inner' [Hroot' [Hsteps [HloadedMiddle Hstate']]]]].
    assert (Hboundary : rowsFinalBoundary boundary prefix = beforeBoundary row)
      by (eapply boundary_before_last; [exact (proj1 (proj2 Hrun)) | exact Hstart]).
    rewrite Hboundary in Hstate'.
    destruct HrunRow as [Hsat [_ Hreplay]].
    cbn [eventsForTrace flat_map] in Hreplay. rewrite app_nil_r in Hreplay.
    destruct (terminal_control_row program s baseDepth cursor' root f middle final row inner' caller Hstore Hwf Hvalid
      Hprime HloadedMiddle (Forall_inv Hsat) Hreplay Hterminal Hcaller Hroot ltac:(congruence) Hstate')
      as [outcome [c1 [HterminalSteps HterminalConfig]]].
    exists outcome, c1. split.
    + eapply spec_steps_transitive; [apply steps_in_call_frame; eassumption | exact HterminalSteps].
    + split; [exact HloadedFinal | ].
      destruct outcome; [destruct HterminalConfig as [_ Hfalse]; contradiction | | ].
      all: exists row; split; [unfold lastRow; rewrite rev_app_distr; reflexivity | exact HterminalConfig].
Qed.
