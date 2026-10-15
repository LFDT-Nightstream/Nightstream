(* Public refinement: one program ROM binding and a changing source position. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns StackSimulation SpecTecCalls Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec CallerRefinement ProgramSpec ProgramEncoding TraceSegmentation
  MemoryCatalog OpcodeCatalog ProgramValidation SourceInstructions SourceValidation SourceRefinement
  CodeLayout BlockSpec ControlSpec ControlState ControlRows.
Import ListNotations.
Open Scope Z_scope.

Lemma rows_start_after_prefix : forall prefix suffix boundary depth,
  Chain continuityLinks (prefix ++ suffix) -> rowsStartAt boundary depth (prefix ++ suffix) ->
  rowsStartAt (rowsFinalBoundary boundary prefix) (rowsFinalDepth depth prefix) suffix.
Proof.
  induction prefix as [ | row prefix IH]; intros suffix boundary depth Hchain Hstart; [exact Hstart | ].
  cbn [rowsFinalBoundary rowsFinalDepth]. apply IH.
  - eapply chain_suffix with (prefix := [row]). exact Hchain.
  - cbn [app] in Hchain. destruct (prefix ++ suffix) as [ | next rest] eqn:Hrest; [exact I | ].
    assert (Hlink : Follows continuityLinks row next) by (inversion Hchain; subst; assumption).
    split; [symmetry; exact (follows_boundary _ _ Hlink) | ].
    symmetry. exact (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)).
Qed.

Lemma finished_source_has_no_row : forall program index offset first rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial (first :: rows) final ->
  callerFragment boundary depth (first :: rows) ->
  SourceState program (sourcePosition index offset) z initial boundary [] c0 -> False.
Proof.
  intros program index offset first rows z initial final boundary depth c0 Hprime Hloaded Hrun
    [[Hboundary Hdepth] [Hkinds Hfinish]] Hsource.
    unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource.
    destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
    destruct Hsource as [Hframe [[Hbound Hcode] [Hvalidation [Hrep Hpc]]]].
    assert (Hoffset : offset = length (guestBody f)).
    { pose proof (f_equal (@length instr) Hcode) as Hlength. rewrite length_skipn in Hlength. cbn in Hlength; lia. }
    subst offset. rewrite code_offset_end in Hpc.
    destruct (proj2 (Forall_inv Hkinds) Hdepth) as [Hprogram [Hpop Hhalt]].
    destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
    destruct (encoded_function_end _ _ _ _ Hloaded Hf) as [Hop Hedge].
    assert (HfirstPC : first COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))).
    { change (boundaryPC (beforeBoundary first) = guestEntry f + Z.of_nat (codeSize (guestBody f))). now rewrite Hboundary. }
    pose proof (fetched_end_selector _ _ _ _ Hprime Hfull Hprogram Hreplay HfirstPC Hop) as Hend.
    pose proof (ProgramBinding.circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
    rewrite HfirstPC, Hedge in Hkind.
    destruct (CallContext.full_function_end_exits first Hprime Hfull Hend ltac:(congruence)); congruence.
Qed.

(* Consume one ordinary row, one block row, or a whole direct call. The trace
   determines which prefix is consumed; no grouping is supplied by the caller. *)
Lemma control_run_prefix : forall program cursor first rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial (first :: rows) final ->
  callerFragment boundary depth (first :: rows) ->
  ControlState program cursor z initial boundary c0 ->
  exists consumed suffix between cursor' z' c1,
    first :: rows = consumed ++ suffix /\ consumed <> [] /\
    validCircuitRun initial consumed between /\ validCircuitRun between suffix final /\
    Steps c0 c1 /\ ControlState program cursor' z' between (rowsFinalBoundary boundary consumed) c1 /\
    rowsFinalDepth depth consumed = depth.
Proof.
  intros program cursor first rows z initial final boundary depth c0 Hprime Hloaded Hrun Hfragment Hstate.
  pose proof Hfragment as HfragmentWhole.
  destruct Hfragment as [[Hboundary Hdepth] [Hkinds Hfinish]].
  destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
  destruct (proj2 (Forall_inv Hkinds) Hdepth) as [Hprogram [Hpop Hhalt]].
  assert (Hsingle : validCircuitRun initial [first] between).
  { split; [constructor; [exact Hfull | constructor] | ]. split; [constructor | ].
    cbn [eventsForTrace flat_map]. rewrite app_nil_r. exact Hreplay. }
  destruct cursor as [[index offset] | position offset prefix].
  - destruct Hstate as [code Hsource]. destruct code as [ | i rest].
    + exfalso. exact (finished_source_has_no_row program index offset first rows z initial final boundary depth c0
        Hprime Hloaded Hrun HfragmentWhole Hsource).
    + destruct i eqn:Hi;
        try solve [match type of Hsource with
        | SourceState _ _ _ _ _ (?instruction :: _) _ =>
          destruct (outer_instruction_prefix program index offset instruction rest first rows z initial final boundary depth c0
          Hprime Hloaded Hrun HfragmentWhole ltac:(intros; discriminate) Hsource)
          as [consumed [suffix [middle [z' [c1 [Hs [Hne [Hp [Ht [Hsteps [Hstate Hd]]]]]]]]]]];
          exists consumed, suffix, middle, (atInstruction (sourcePosition index (S offset))), z', c1;
          split; [exact Hs | ]; split; [exact Hne | ]; split; [exact Hp | ];
          split; [exact Ht | ]; split; [exact Hsteps | ]; split; assumption end].
      rewrite <- Hboundary in Hsource.
      destruct (block_enter_row _ _ _ _ _ _ _ _ _ _ Hprime Hloaded Hfull Hreplay Hprogram Hpop Hhalt Hsource)
        as [prefix [c1 [Hsteps [Hstate Hd]]]].
      exists [first], rows, between, (inBlock (sourcePosition index offset) 0 prefix), z, c1.
      split; [reflexivity | ]. split; [discriminate | ]. split; [exact Hsingle | ].
      split; [exact Htail | ]. split; [exact Hsteps | ]. split; [exact Hstate | cbn; lia].
  - rewrite <- Hboundary in Hstate.
    destruct (block_control_row _ _ _ _ _ _ _ _ _ Hprime Hloaded Hfull Hreplay Hprogram Hpop Hhalt Hstate)
      as [cursor' [z' [c1 [Hsteps [Hstate' Hd]]]]].
    exists [first], rows, between, cursor', z', c1.
    split; [reflexivity | ]. split; [discriminate | ]. split; [exact Hsingle | ].
    split; [exact Htail | ]. split; [exact Hsteps | ]. split; [exact Hstate' | cbn; lia].
Qed.

(* Every row is consumed. The resulting control state can be supplied to a
   subsequent run, including when the boundary lies inside a block. Direct calls
   still finish within the run, as required by callerFragment's depth boundary. *)
Theorem run_refinement : forall program cursor rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  callerFragment boundary depth rows ->
  ControlState program cursor z initial boundary c0 ->
  exists cursor' z' c1, Steps c0 c1 /\ programLoaded program final /\
    ControlState program cursor' z' final (rowsFinalBoundary boundary rows) c1.
Proof.
  intros program cursor rows. remember (length rows) as n eqn:Hn.
  revert cursor rows Hn. induction n using lt_wf_ind.
  intros cursor rows Hn z initial final boundary depth c0 Hprime Hloaded Hrun Hfragment Hstate.
  destruct rows as [ | first rows].
  - destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst.
    exists cursor, z, c0. split; [destruct c0; apply Steps__refl; eapply control_state_wf; exact Hstate | auto].
  - destruct (control_run_prefix _ _ _ _ _ _ _ _ _ _ Hprime Hloaded Hrun Hfragment Hstate)
      as [consumed [suffix [between [cursor1 [z1 [c1 [Hsplit [Hnonempty [Hprefix [Hsuffix [Hsteps [Hstate1 Hdepth1]]]]]]]]]]]].
    destruct Hfragment as [Hstart [Hkinds Hfinish]].
    assert (Hloaded1 : programLoaded program between)
      by (eapply program_loaded_preserved; [eapply run_preserves_roms; exact Hprefix | exact Hloaded]).
    assert (Hfragment1 : callerFragment (rowsFinalBoundary boundary consumed) depth suffix).
    { split.
      - rewrite <- Hdepth1. apply rows_start_after_prefix.
        + rewrite <- Hsplit. exact (proj1 (proj2 Hrun)).
        + rewrite <- Hsplit. exact Hstart.
      - split; [rewrite Hsplit, Forall_app in Hkinds; exact (proj2 Hkinds) | ].
        rewrite Hsplit, rows_final_depth_app, Hdepth1 in Hfinish. exact Hfinish. }
    assert (Hshorter : (length suffix < n)%nat).
    { rewrite Hsplit, length_app in Hn. destruct consumed; [contradiction | cbn in Hn; lia]. }
    destruct (H (length suffix) Hshorter cursor1 suffix eq_refl z1 between final
      (rowsFinalBoundary boundary consumed) depth c1 Hprime Hloaded1 Hsuffix Hfragment1 Hstate1)
      as [cursor2 [z2 [c2 [Hsteps2 [Hloaded2 Hstate2]]]]].
    exists cursor2, z2, c2. split; [eapply spec_steps_transitive; eassumption | ].
    split; [exact Hloaded2 | ]. rewrite Hsplit, rows_final_boundary_app. exact Hstate2.
Qed.
