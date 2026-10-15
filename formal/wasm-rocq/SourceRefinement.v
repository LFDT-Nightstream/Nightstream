(* Ordinary instructions and complete direct calls consume a derived row prefix. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation StraightLineInstructions CallSpec CallerSpec CallerRefinement
  SourceInstructions SourceValidation CodeLayout ProgramSpec ProgramEncoding TraceSegmentation
  BlockSpec ProgramValidation ControlSpec ControlState.
Import ListNotations.
Open Scope Z_scope.

Lemma program_loaded_singleton : forall program memory index f offset i z height,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some i -> frameMatches program f z ->
  callerFacts (programCallees program (guestModule f)) z height [i] ->
  loadedCaller (programCallees program (guestModule f)) z memory
    (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) [i].
Proof.
  intros program memory index f offset i z height Hloaded Hf Hi Hframe Hfacts.
  assert (Hordinary : forall j,
    supportedStraightLineProgram [j] -> nth_error (guestBody f) offset = Some j ->
    instructionImage memory (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) j /\ True).
  { intros j Hs Hj. split; [eapply encoded_instruction_image; try eassumption; exact (Forall_inv Hs) | exact I]. }
  destruct i; try (apply Hordinary; [exact (proj1 Hfacts) | exact Hi]).
  cbn [callerFacts loadedCaller] in *.
  destruct (programCallees program (guestModule f) (v_funcidx :> N)) as [callee | ] eqn:Hcallee; [ | contradiction].
  split; [ | exact I]. eapply encoded_call_loaded; try eassumption.
  exact (proj1 (proj2 (proj2 (proj2 (proj1 Hfacts))))).
Qed.

Lemma caller_fact_instruction_size : forall callees z height i,
  callerFacts callees z height [i] -> instructionSize i = 1%nat.
Proof.
  intros callees z height i H. destruct i; try reflexivity.
  apply ordinary_instruction_size. exact (Forall_inv (proj1 H)).
Qed.

Theorem source_instruction_refinement : forall program index f offset i first rows z initial final depth c0,
  fieldPrime -> programLoaded program initial -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some i -> frameMatches program f z ->
  validCircuitRun initial (first :: rows) final -> Forall (callerRow depth) (first :: rows) ->
  first COL_CALL_STACK_DEPTH_BEFORE = depth -> rowsFinalDepth depth (first :: rows) = depth ->
  first COL_PC_BEFORE = guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) ->
  callerFacts (programCallees program (guestModule f)) z (stackHeight (beforeBoundary first)) [i] ->
  represents z initial (beforeBoundary first) [i] c0 ->
  exists consumed suffix between z' c1,
    first :: rows = consumed ++ suffix /\ consumed <> [] /\
    validCircuitRun initial consumed between /\ validCircuitRun between suffix final /\
    Steps c0 c1 /\ represents z' between (rowsFinalBoundary (beforeBoundary first) consumed) [] c1 /\
    sameCaller z z' /\
    boundaryPC (rowsFinalBoundary (beforeBoundary first) consumed) =
      guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) + Z.of_nat (instructionSize i) /\
    rowsFinalDepth depth consumed = depth /\
    sourceHeight (programCallees program (guestModule f)) i (stackHeight (beforeBoundary first)) =
      Some (stackHeight (rowsFinalBoundary (beforeBoundary first) consumed)).
Proof.
  intros program index f offset i first rows z initial final depth c0 Hprime Hloaded Hf Hi Hframe
    Hrun Hkinds Hdepth Hfinish Hpc Htyped Hrep.
    pose proof (program_loaded_singleton _ _ _ _ _ _ _ _ Hloaded Hf Hi Hframe Htyped) as Himage.
    pose proof (caller_fact_instruction_size _ _ _ _ Htyped) as Hsize.
    assert (Hstate : CallerState (programCallees program (guestModule f)) z initial (beforeBoundary first)
      (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) [i] c0) by (constructor; assumption).
    destruct (next_segment i [] rows first _ z initial final _ depth c0 Hprime Hrun Hkinds Hdepth Hfinish Hstate)
      as [segment [suffix [Hsplit [Hfirst Hmatch]]]].
    rewrite Hsplit in Hrun. destruct (valid_run_split _ _ _ _ Hrun) as [between [Hprefix Hsuffix]].
    destruct (caller_segment_refinement i [] segment _ z initial between _ c0 Hprime Hprefix Hmatch Himage
      ltac:(rewrite Hfirst; exact Hpc) ltac:(rewrite Hfirst; auto))
      as [z' [c1 [Hsteps [[_ Hrep'] [Hpc' [Hsame [Hdepth' Hheight]]]]]]].
    exists (segmentRows segment), suffix, between, z', c1.
    split; [exact Hsplit | ]. split; [destruct segment; discriminate | ]. split; [exact Hprefix | ].
    split; [exact Hsuffix | ]. split; [exact Hsteps | ]. rewrite segment_final_boundary.
    split; [exact Hrep' | ]. split; [exact Hsame | ]. split; [rewrite Hsize; exact Hpc' | ].
    split; [rewrite segment_final_depth, Hdepth', Hfirst; exact Hdepth | ]. now rewrite Hfirst in Hheight.
Qed.

Lemma outer_instruction_prefix : forall program index offset i rest first rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial (first :: rows) final ->
  callerFragment boundary depth (first :: rows) ->
  (forall bt body, i <> BLOCK bt body) ->
  SourceState program (sourcePosition index offset) z initial boundary (i :: rest) c0 ->
  exists consumed suffix between z' c1,
    first :: rows = consumed ++ suffix /\ consumed <> [] /\
    validCircuitRun initial consumed between /\ validCircuitRun between suffix final /\
    Steps c0 c1 /\
    ControlState program (atInstruction (sourcePosition index (S offset))) z' between
      (rowsFinalBoundary boundary consumed) c1 /\ rowsFinalDepth depth consumed = depth.
Proof.
  intros program index offset i rest first rows z initial final boundary depth c0
    Hprime Hloaded Hrun [Hstart [Hkinds Hfinish]] Hnotblock Hsource.
  destruct Hstart as [Hboundary Hdepth].
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
      ltac:(rewrite Hboundary; destruct i; try exact Hif; exfalso; eapply Hnotblock; reflexivity) Hsmall)
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

    exists consumed, suffix, between, z1, (stackConfig z1 outputs rest).
    split; [exact Hsplit | ]. split; [exact Hnonempty | ]. split; [exact Hprefix | ].
    split; [exact Hsuffix | ]. split; [exact Hsteps1 | ]. split; [exists rest; exact Hsource1 | exact Hdepth1].
Qed.
