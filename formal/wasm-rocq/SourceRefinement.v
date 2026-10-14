(* Consume one complete source instruction from a raw circuit trace. Calls and
   blocks derive their multi-row extent from validity, rather than a supplied split. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation StraightLineInstructions CallSpec CallerSpec CallerRefinement
  SourceInstructions SourceValidation CodeLayout ProgramSpec ProgramEncoding TraceSegmentation
  BlockSpec BlockEncoding BlockRefinement.
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
  sourceEndpoint program (sourcePosition index offset) (rowsFinalBoundary (beforeBoundary first) (first :: rows)) ->
  sourceInstructionFacts (programCallees program (guestModule f)) z (stackHeight (beforeBoundary first)) i ->
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
    Hrun Hkinds Hdepth Hfinish Hpc Hendpoint Hfacts Hrep.
  assert (Hcases : (exists bt body, i = BLOCK bt body) \/ (forall bt body, i <> BLOCK bt body)).
  { destruct i; try (right; intros; discriminate). left; eauto. }
  destruct Hcases as [[bt [body ->]] | Hordinary].
  - destruct Hfacts as [[-> [x [tail [-> Hsupport]]]] [Hwf Htyped]].
    pose proof (encoded_block_image _ _ _ _ _ _ _ Hloaded Hf Hi Hsupport) as Himage.
    pose proof (block_endpoint_outside _ (sourcePosition index offset) _ _ _ _ _ Hf Hi Hendpoint) as Houtside.
    destruct (conditional_block_refinement x tail first rows initial final _ depth z c0 Hprime Hrun Hkinds Hdepth Hpc
      Houtside Himage Hsupport Htyped Hrep)
      as [consumed [suffix [between [z' [c1 [Hsplit [Hnonempty [Hprefix [Hsuffix [Hsteps [Hrep' [Hsame [Hpc' [Hdepth' Hheight]]]]]]]]]]]]]].
    exists consumed, suffix, between, z', c1.
    split; [exact Hsplit | ]. split; [exact Hnonempty | ]. split; [exact Hprefix | ]. split; [exact Hsuffix | ].
    split; [exact Hsteps | ]. split; [exact Hrep' | ]. split; [exact Hsame | ]. split.
    + unfold instructionSize, flattenInstruction. cbn [length]. rewrite length_app, block_code_length.
      cbn [length]. unfold blockExitPC in Hpc'. lia.
    + split; [exact Hdepth' | ]. cbn [sourceHeight]. now rewrite Hheight.
  - assert (Htyped : callerFacts (programCallees program (guestModule f)) z (stackHeight (beforeBoundary first)) [i]).
    { destruct i; try exact Hfacts. exfalso; eapply Hordinary; reflexivity. }
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
