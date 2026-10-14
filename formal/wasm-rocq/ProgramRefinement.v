(* Public refinement: one program ROM binding and a changing source position. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns StackSimulation SpecTecCalls Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec CallerRefinement ProgramSpec ProgramEncoding TraceSegmentation
  MemoryCatalog OpcodeCatalog ProgramValidation SourceInstructions SourceValidation SourceRefinement
  CodeLayout BlockSpec.
Import ListNotations.
Open Scope Z_scope.

Lemma code_at_advance : forall f offset executed remaining,
  codeAt f offset (executed ++ remaining) ->
  codeAt f (offset + length executed) remaining.
Proof.
  intros f offset executed remaining [Hbound Hcode]. split.
  - pose proof (f_equal (@length instr) Hcode) as Hlength.
    rewrite length_skipn, length_app in Hlength. lia.
  - replace (offset + length executed)%nat with (length executed + offset)%nat by lia.
    rewrite <- skipn_skipn, Hcode, skipn_app, skipn_all, Nat.sub_diag. reflexivity.
Qed.

Lemma frame_matches_preserved : forall program f before after,
  sameCaller before after -> frameMatches program f before -> frameMatches program f after.
Proof.
  intros program f before after [Hs [Ha Hl]] [Hfunctions [Haddresses Hlocals]].
  unfold frameMatches. rewrite Ha, Hl. split; [ | auto].
  destruct before, after. cbn [fun_store fun_funcinst] in *. congruence.
Qed.

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

(* Every row is consumed. Blocks and calls derive their row extents internally.
   The endpoint premise excludes stopping inside a block; it supplies no split,
   branch decisions, or instruction/row correspondence. *)
Theorem run_refinement : forall program position code rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  callerFragment boundary depth rows ->
  sourceEndpoint program position (rowsFinalBoundary boundary rows) ->
  SourceState program position z initial boundary code c0 ->
  exists executed remaining z' c1,
    code = executed ++ remaining /\ Steps c0 c1 /\ programLoaded program final /\
    SourceState program (advanceSource position (length executed)) z' final
      (rowsFinalBoundary boundary rows) remaining c1.
Proof.
  intros program position code. revert position.
  induction code as [ | i rest IH]; intros [index offset] rows z initial final boundary depth c0
    Hprime Hloaded Hrun Hfragment Hendpoint Hsource;
    destruct rows as [ | first rows].
  - destruct Hrun as [_ [_ Hreplay]]; inversion Hreplay; subst;
    exists [], [], z, c0; split; [reflexivity | ]; split;
    [ destruct c0; apply Steps__refl; eapply represents_wf;
      unfold SourceState in Hsource; cbn [sourceFunction sourceOffset] in Hsource;
      destruct (functionAt program index); [exact (proj1 (proj2 (proj2 (proj2 Hsource)))) | contradiction]
    | split; [exact Hloaded | ]; unfold advanceSource; cbn [length sourceFunction sourceOffset];
      rewrite Nat.add_0_r; exact Hsource ].
  - destruct Hfragment as [[Hboundary Hdepth] [Hkinds Hfinish]].
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
  - destruct Hrun as [_ [_ Hreplay]]; inversion Hreplay; subst;
    exists [], (i :: rest), z, c0; split; [reflexivity | ]; split;
    [ destruct c0; apply Steps__refl; eapply represents_wf;
      unfold SourceState in Hsource; cbn [sourceFunction sourceOffset] in Hsource;
      destruct (functionAt program index); [exact (proj1 (proj2 (proj2 (proj2 Hsource)))) | contradiction]
    | split; [exact Hloaded | ]; unfold advanceSource; cbn [length sourceFunction sourceOffset];
      rewrite Nat.add_0_r; exact Hsource ].
  - destruct Hfragment as [Hstart [Hkinds Hfinish]]. destruct Hstart as [Hboundary Hdepth].
    pose proof Hsource as Hparts. unfold SourceState in Hparts. cbn [sourceFunction sourceOffset] in Hparts.
    destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
    destruct Hparts as [Hframe [Hcode [Hvalidation [Hrep Hpc]]]].
    pose proof (validated_code_facts _ _ _ _ _ Hframe (represented_state_wf _ _ _ _ _ Hrep) Hvalidation) as Hfacts.
    pose proof (source_facts_wf _ _ _ _ Hfacts) as Hwf.
    destruct Hfacts as [Hif Hrest].
    assert (Hlookup : nth_error (guestBody f) offset = Some i).
    { pose proof (f_equal (fun xs => nth_error xs 0) (proj2 Hcode)) as H.
      cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
    destruct Hrep as [values [Hstack [-> [HconfigWf Hlocals]]]].
    assert (Hz : wf_state z) by (inversion HconfigWf; assumption).
    assert (Hsmall : represents z initial (beforeBoundary first) [i] (stackConfig z values [i])).
    { rewrite Hboundary. exists values. split; [exact Hstack | ]. split; [reflexivity | ].
      split; [apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) | constructor; [exact (Forall_inv Hwf) | constructor]] | exact Hlocals]. }
    destruct (source_instruction_refinement program index f offset i first rows z initial final depth _
      Hprime Hloaded Hf Hlookup Hframe Hrun Hkinds Hdepth Hfinish
      ltac:(change (boundaryPC (beforeBoundary first) = guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)); now rewrite Hboundary)
      ltac:(exact Hendpoint) ltac:(now rewrite Hboundary) Hsmall)
      as [consumed [suffix [between [z1 [small1 [Hsplit [Hnonempty [Hprefix [Hsuffix [Hsteps [Hrep1 [Hsame [Hpc1 [Hdepth1 Hheight1]]]]]]]]]]]]]].
    assert (HboundaryPrefix : rowsFinalBoundary (beforeBoundary first) consumed = rowsFinalBoundary boundary consumed)
      by now rewrite Hboundary.
    rewrite HboundaryPrefix in Hrep1, Hpc1, Hheight1.
    rewrite Hboundary in Hheight1. rewrite Hheight1 in Hrest.
    pose proof (represented_state_wf _ _ _ _ _ Hrep1) as Hz1.
    pose proof (source_facts_preserved _ _ _ _ _ Hsame Hz1 Hrest) as Hrest1.
    destruct Hrep1 as [outputs [Hstack1 [-> [HsmallWf1 Hlocals1]]]].
    assert (Hsteps1 : Steps (stackConfig z values (i :: rest)) (stackConfig z1 outputs rest)).
    { exact (stack_suffix_steps z z1 values outputs [i] rest Hsteps (Forall_inv_tail Hwf)). }
    assert (Hloaded1 : programLoaded program between)
      by (eapply program_loaded_preserved; [eapply run_preserves_roms; exact Hprefix | exact Hloaded]).
    assert (Hsource1 : SourceState program (sourcePosition index (S offset)) z1 between
      (rowsFinalBoundary boundary consumed) rest (stackConfig z1 outputs rest)).
    { unfold SourceState. cbn [sourceFunction sourceOffset]. rewrite Hf.
      split; [eapply frame_matches_preserved; eassumption | ]. split.
      - replace (S offset) with (offset + length [i])%nat by (cbn; lia). apply code_at_advance. exact Hcode.
      - split.
        + eapply validated_code_remaining with (executed := [i]); try eassumption.
          eapply frame_matches_preserved; eassumption.
        + split.
          * exists outputs. split; [exact Hstack1 | ]. split; [reflexivity | ].
            split; [apply stack_config_wf; [exact Hz1 | exact (proj1 (proj2 (proj2 Hstack1))) | exact (Forall_inv_tail Hwf)] | exact Hlocals1].
          * rewrite (code_offset_next _ _ _ Hlookup), Nat2Z.inj_add. lia. }
    assert (Hfragment1 : callerFragment (rowsFinalBoundary boundary consumed) depth suffix).
    { split.
      - rewrite <- Hdepth1. apply rows_start_after_prefix.
        + rewrite <- Hsplit. exact (proj1 (proj2 Hrun)).
        + rewrite <- Hsplit. split; assumption.
      - split; [rewrite Hsplit, Forall_app in Hkinds; exact (proj2 Hkinds) | ].
        rewrite Hsplit, rows_final_depth_app, Hdepth1 in Hfinish. exact Hfinish. }
    assert (Hendpoint1 : sourceEndpoint program (sourcePosition index (S offset))
      (rowsFinalBoundary (rowsFinalBoundary boundary consumed) suffix)).
    { rewrite <- rows_final_boundary_app, <- Hsplit. exact Hendpoint. }
    destruct (IH (sourcePosition index (S offset)) suffix z1 between final (rowsFinalBoundary boundary consumed) depth
      (stackConfig z1 outputs rest) Hprime Hloaded1 Hsuffix Hfragment1 Hendpoint1 Hsource1)
      as [executed [remaining [z2 [c2 [Hcode2 [Hsteps2 [Hloaded2 Hsource2]]]]]]].
    exists (i :: executed), remaining, z2, c2. split; [cbn; now rewrite Hcode2 | ].
    split; [eapply spec_steps_transitive; eassumption | ]. split; [exact Hloaded2 | ].
    rewrite Hsplit, rows_final_boundary_app.
    unfold advanceSource in *. cbn [sourceFunction sourceOffset length] in *.
    replace (offset + S (length executed))%nat with (S offset + length executed)%nat by lia. exact Hsource2.
Qed.
