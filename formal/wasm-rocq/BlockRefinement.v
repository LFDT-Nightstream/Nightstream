(* Complete the fallthrough side of a conditional block, including its END. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers StraightLineSpec StackSimulation LocalSimulation CallSpec CallBody CallScope CallContext
  SpecTecCalls SpecTecBranches CallerRefinement ProgramSpec TraceSegmentation BlockSpec BlockBindings
  BlockSimulation BlockInstructions BlockTrace LocalBindings ProgramBinding Selectors BranchBindings BranchSpec BranchRefinement BlockValidation.
Import ListNotations.
Open Scope Z_scope.

Lemma row_preserves_roms : forall initial final row,
  Replay initial (eventsForRow memories row) final -> sameRoms initial final.
Proof.
  intros initial final row H memory Hro address. eapply replay_preserves; [exact H | apply read_only_row; exact Hro].
Qed.

Lemma block_fallthrough_refinement : forall tail rows previous prefix initial final pc depth z,
  fieldPrime -> validCircuitRun initial rows final -> Chain continuityLinks (previous :: rows) ->
  Forall (callerRow depth) rows -> previous COL_PC_AFTER = pc + 3 ->
  previous COL_CALL_STACK_DEPTH_AFTER = depth ->
  ~ (pc < boundaryPC (rowsFinalBoundary (afterBoundary previous) rows) < blockExitPC pc tail) ->
  loadedBody initial (pc + 3) tail ->
  controlImage initial (pc + 3 + Z.of_nat (length tail)) CODE_END (blockExitPC pc tail) ->
  wf_state z -> Forall wf_instr tail -> programTyped (length (stateLocals z)) 0 tail -> bodyStackDelta tail = Some 0 ->
  stackMatches initial (afterBoundary previous) prefix -> localsMatch initial (afterBoundary previous) z ->
  exists middle endRow suffix between z',
    rows = middle ++ endRow :: suffix /\ validCircuitRun initial (middle ++ [endRow]) between /\
    validCircuitRun between suffix final /\
    Steps (stackPrefix prefix (blockContext (stackConfig z [] tail))) (stackConfig z' prefix []) /\
    represents z' between (afterBoundary endRow) [] (stackConfig z' prefix []) /\ sameCaller z z' /\
    endRow COL_PC_AFTER = blockExitPC pc tail /\ endRow COL_CALL_STACK_DEPTH_AFTER = depth /\
    stackHeight (afterBoundary endRow) = length prefix.
Proof.
  intros tail rows previous prefix initial final pc depth z Hprime Hrun Hchain Hkinds Hpc Hdepth Houtside
    Hbody Hend Hz Hwf Htyped Hdelta Hstack Hlocals.
  destruct (scan_block_tail tail rows previous initial final (pc + 3) depth pc (blockExitPC pc tail)
    Hprime Hrun Hchain Hkinds Hpc Hdepth ltac:(lia) ltac:(unfold blockExitPC; lia) Houtside Hbody Hend)
    as [middle [endRow [suffix [Hsplit [Hlength [Hprogram HendSelected]]]]]].
  rewrite Hsplit in Hrun, Hchain, Hkinds.
  assert (HsplitRun : validCircuitRun initial ((middle ++ [endRow]) ++ suffix) final).
  { rewrite <- app_assoc. exact Hrun. }
  destruct (valid_run_split _ _ _ _ HsplitRun) as [between [Hprefix Hsuffix]].
  destruct (valid_run_split middle [endRow] initial between Hprefix) as [beforeEnd [Hmiddle HendRun]].
  assert (HprefixChain : Chain continuityLinks (previous :: middle ++ [endRow])).
  { eapply chain_prefix with (suffix := suffix). cbn [app]. rewrite <- app_assoc. exact Hchain. }
  assert (HmiddleChain : Chain continuityLinks (previous :: middle)).
  { eapply chain_prefix with (suffix := [endRow]). exact HprefixChain. }
  destruct (body_above_prefix tail middle previous prefix [] initial beforeEnd (pc + 3) z Hprime Hmiddle
    HmiddleChain Hprogram Hlength Hbody Hpc Hz Hwf Htyped ltac:(rewrite app_nil_r; exact Hstack) Hlocals)
    as [z' [values [Hsteps [Hz' [Hstack' [Hlocals' [Hsame [Hpc' [Hdepth' Hdelta']]]]]]]]].
  rewrite Hdelta in Hdelta'. assert (Hempty : values = []).
  { injection Hdelta' as Hlen. apply length_zero_iff_nil. cbn [length] in Hlen. lia. }
  subst values. rewrite app_nil_r in Hstack'.
  pose proof (final_row_follows _ _ _ HprefixChain) as Hlink.
  rewrite (follows_boundary _ _ Hlink) in Hstack', Hlocals'.
  pose proof (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)) as HpcLink.
  pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)) as HdepthLink.
  assert (HendDepth : endRow COL_CALL_STACK_DEPTH_BEFORE = depth) by lia.
  assert (HendKind : callerRow depth endRow).
  { rewrite Forall_app in Hkinds. exact (Forall_inv (proj2 Hkinds)). }
  destruct (proj2 HendKind HendDepth) as [HendProgram [Hpop Hhalt]].
  destruct (valid_run_head _ _ _ _ HendRun) as [afterEnd [Hfull [Hreplay HemptyRun]]].
  assert (HsameEnd : afterEnd = between).
  { destruct HemptyRun as [_ [_ HreplayEmpty]]. inversion HreplayEmpty; reflexivity. }
  subst afterEnd.
  assert (HendImage : controlImage beforeEnd (pc + 3 + Z.of_nat (length tail)) CODE_END (blockExitPC pc tail)).
  { pose proof (run_preserves_roms _ _ _ Hmiddle) as Hrom. unfold controlImage in *. preserve_roms Hrom. exact Hend. }
  destruct (control_row beforeEnd between endRow (pc + 3 + Z.of_nat (length tail)) CODE_END COL_SEL_END (blockExitPC pc tail) Hprime Hfull HendProgram Hreplay ltac:(lia)
    ltac:(vm_compute; reflexivity) HendImage) as [_ Htarget].
  destruct (block_memory_preserved _ _ _ COL_SEL_END Hprime Hfull ltac:(right; reflexivity) HendSelected Hpop Hhalt Hreplay)
    as [HstackPreserved HlocalsPreserved].
  pose proof (HstackPreserved prefix Hstack') as HstackFinal.
  pose proof (HlocalsPreserved z' Hlocals') as HlocalsFinal.
  assert (HfinishDepth : endRow COL_CALL_STACK_DEPTH_AFTER = depth).
  { pose proof (full_block_context endRow COL_SEL_END Hprime Hfull ltac:(right; reflexivity) HendSelected Hpop Hhalt)
      as [_ [_ [_ [Hd _]]]]. lia. }
  assert (HprefixBounds : Forall (fun value => 0 <= value < i32Modulus) prefix)
    by exact (proj1 (proj2 (proj2 Hstack))).
  exists middle, endRow, suffix, between, z'. split; [exact Hsplit | ].
  split; [exact Hprefix | ]. split; [exact Hsuffix | ]. split.
  - pose proof (steps_in_block _ _ Hsteps) as Hinside.
    pose proof (spec_steps_transitive _ _ _ Hinside (exit_empty_block z' Hz')) as Hexit.
    pose proof (steps_with_stack_prefix prefix _ _ HprefixBounds Hexit) as Hlift. exact Hlift.
  - split.
    + exists prefix. split; [exact HstackFinal | ]. split; [reflexivity | ].
      split; [apply stack_config_wf; assumption || constructor | exact HlocalsFinal].
    + split; [exact Hsame | ]. split; [exact Htarget | ]. split; [exact HfinishDepth | ].
      exact (stack_matches_height _ _ _ HstackFinal).
Qed.

Lemma prepend_valid_row : forall initial row middle rows final,
  FullRelation.satFull row -> Replay initial (eventsForRow memories row) middle ->
  Chain continuityLinks (row :: rows) -> validCircuitRun middle rows final ->
  validCircuitRun initial (row :: rows) final.
Proof.
  intros initial row middle rows final Hfull Hreplay Hchain [Hrows [_ Hrest]].
  split; [constructor; assumption | ]. split; [exact Hchain | ].
  cbn [eventsForTrace flat_map]. eapply replay_app; eassumption.
Qed.

Theorem conditional_block_refinement : forall x tail first rows initial final pc depth z c0,
  fieldPrime -> validCircuitRun initial (first :: rows) final ->
  Forall (callerRow depth) (first :: rows) -> first COL_CALL_STACK_DEPTH_BEFORE = depth ->
  first COL_PC_BEFORE = pc ->
  ~ (pc < boundaryPC (rowsFinalBoundary (beforeBoundary first) (first :: rows)) < blockExitPC pc tail) ->
  loadedBlock initial pc x tail -> supportedStraightLineProgram tail ->
  typedBlock (length (stateLocals z)) (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail) ->
  represents z initial (beforeBoundary first) [conditionalBlock x tail] c0 ->
  exists consumed suffix between z' c1,
    first :: rows = consumed ++ suffix /\ consumed <> [] /\
    validCircuitRun initial consumed between /\ validCircuitRun between suffix final /\
    Steps c0 c1 /\ represents z' between (rowsFinalBoundary (beforeBoundary first) consumed) [] c1 /\
    sameCaller z z' /\ boundaryPC (rowsFinalBoundary (beforeBoundary first) consumed) = blockExitPC pc tail /\
    rowsFinalDepth depth consumed = depth /\
    stackHeight (rowsFinalBoundary (beforeBoundary first) consumed) = stackHeight (beforeBoundary first).
Proof.
  intros x tail first rows initial final pc depth z c0 Hprime Hrun Hkinds Hdepth Hpc Houtside
    [Hentry [Hget [Hbranch [Hbody Hend]]]] Hsupport Htyped Hrep.
  destruct Hrep as [prefix [Hstack [-> [Hwf Hlocals]]]].
  assert (Hz : wf_state z) by (inversion Hwf; assumption).
  assert (HbodyWf : Forall wf_instr (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail)).
  { pose proof Hwf as Hw. unfold stackConfig, conditionalBlock in Hw.
    inversion Hw as [state instructions Hstate Hall]; subst.
    apply Forall_app in Hall. destruct Hall as [_ Hcode].
    pose proof (Forall_inv Hcode) as HblockWf. inversion HblockWf; assumption. }
  pose proof (Forall_inv_tail (Forall_inv_tail HbodyWf)) as HtailWf.
  destruct (typed_block_facts _ _ _ Hsupport Htyped) as [Hx [HtailTyped HtailDelta]].
  destruct (proj2 (Forall_inv Hkinds) Hdepth) as [Hprogram [Hpop Hhalt]].
  destruct (valid_run_head _ _ _ _ Hrun) as [afterEntry [Hfull [Hreplay HrestRun]]].
  destruct (control_row _ _ _ _ CODE_BLOCK COL_SEL_BLOCK _ Hprime Hfull Hprogram Hreplay Hpc
    ltac:(vm_compute; reflexivity) Hentry) as [Hblock Hnext].
  pose proof (full_block_context first COL_SEL_BLOCK Hprime Hfull ltac:(left; reflexivity) Hblock Hpop Hhalt)
    as [_ [_ [_ [HdepthAfter _]]]].
  destruct (block_memory_preserved _ _ _ COL_SEL_BLOCK Hprime Hfull ltac:(left; reflexivity) Hblock Hpop Hhalt Hreplay)
    as [HstackEntry HlocalsEntry].
  specialize (HstackEntry prefix Hstack). specialize (HlocalsEntry z Hlocals).
  pose proof (row_preserves_roms _ _ _ Hreplay) as Hrom1.
  assert (Hget1 : instructionImage afterEntry (pc + 1) (LOCAL_GET x))
    by (eapply instruction_image_same_roms; eassumption).
  destruct rows as [ | getRow rows].
  { exfalso. apply Houtside. cbn [rowsFinalBoundary afterBoundary boundaryPC]. unfold blockExitPC. lia. }
  assert (Hlink1 : Follows continuityLinks first getRow) by (pose proof (proj1 (proj2 Hrun)) as Hchain; inversion Hchain; subst; assumption).
  assert (HgetDepth : getRow COL_CALL_STACK_DEPTH_BEFORE = depth).
  { pose proof (Hlink1 COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)); lia. }
  assert (HgetPC : getRow COL_PC_BEFORE = pc + 1).
  { pose proof (Hlink1 COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); lia. }
  pose proof (proj1 (proj2 (Forall_inv (Forall_inv_tail Hkinds)) HgetDepth)) as HgetProgram.
  destruct (valid_run_head _ _ _ _ HrestRun) as [afterGet [HgetFull [HgetReplay HafterGetRun]]].
  rewrite (follows_boundary _ _ Hlink1) in HstackEntry, HlocalsEntry.
  rewrite <- HgetPC in Hget1.
  assert (HgetSelector : getRow COL_SEL_LOCAL_GET = 1).
  { pose proof (circuit_row_opcode_fetch _ _ _ HgetReplay HgetProgram) as Hfetch.
    rewrite (proj1 Hget1) in Hfetch.
    exact (full_opcode_selects getRow COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime HgetFull HgetProgram
      eq_refl ltac:(cbn [encodedOpcode] in Hfetch; congruence)). }
  assert (HgetIndex : getRow COL_LOCAL_INDEX = Z.of_N (x :> N)).
  { pose proof (proj2 (proj2 (full_straight_line_local_gates getRow COL_SEL_LOCAL_GET Hprime HgetFull
      ltac:(cbn; auto) HgetSelector))) as Hgate.
    pose proof (circuit_row_local_index_fetch _ _ _ HgetReplay Hgate) as Hread.
    rewrite (proj2 (proj2 (proj2 Hget1))) in Hread. congruence. }
  pose proof (instruction_row_next_pc _ _ _ _ Hprime HgetFull HgetProgram HgetReplay Hget1) as HgetNext.
  pose proof (proj2 (proj2 (full_body_context_quiet getRow COL_SEL_LOCAL_GET Hprime HgetFull ltac:(cbn; auto) HgetSelector))) as HgetRegisters.
  destruct (framed_local_get prefix [] z afterEntry afterGet getRow x Hprime HgetFull HgetReplay HgetSelector HgetIndex
    Hz (Forall_inv HbodyWf) Hx ltac:(rewrite app_nil_r; exact HstackEntry) HlocalsEntry)
    as [values [HgetStep [_ [HstackGet [HlocalsGet [_ Hheight]]]]]].
  destruct values as [ | condition [ | extra values]]; cbn [instructionStackHeight length] in Hheight; try discriminate.
  assert (HgetAfterDepth : getRow COL_CALL_STACK_DEPTH_AFTER = depth) by (destruct HgetRegisters as [Hd _]; lia).
  destruct rows as [ | branchRow rows].
  { exfalso. apply Houtside. cbn [rowsFinalBoundary afterBoundary boundaryPC]. unfold blockExitPC. lia. }
  assert (Hlink2 : Follows continuityLinks getRow branchRow) by (pose proof (proj1 (proj2 HrestRun)) as Hchain; inversion Hchain; subst; assumption).
  assert (HbranchDepth : branchRow COL_CALL_STACK_DEPTH_BEFORE = depth).
  { pose proof (Hlink2 COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)); lia. }
  assert (HbranchPC : branchRow COL_PC_BEFORE = pc + 2).
  { pose proof (Hlink2 COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); lia. }
  pose proof (proj1 (proj2 (Forall_inv (Forall_inv_tail (Forall_inv_tail Hkinds))) HbranchDepth)) as HbranchProgram.
  destruct (valid_run_head _ _ _ _ HafterGetRun) as [afterBranch [HbranchFull [HbranchReplay HtailRun]]].
  pose proof (row_preserves_roms _ _ _ HgetReplay) as Hrom2.
  pose proof (row_preserves_roms _ _ _ HbranchReplay) as Hrom3.
  assert (Hbranch2 : branchImage afterGet (pc + 2) (pc + 3) (blockExitPC pc tail)).
  { unfold branchImage in *. preserve_roms Hrom2. preserve_roms Hrom1. exact Hbranch. }
  destruct (circuit_br_if_control _ _ _ _ _ _ Hprime HbranchFull HbranchReplay HbranchProgram HbranchPC Hbranch2)
    as [HbranchSelector _].
  assert (HbranchAfterDepth : branchRow COL_CALL_STACK_DEPTH_AFTER = depth)
    by (pose proof (br_if_depth_preserved branchRow Hprime HbranchFull HbranchSelector); lia).
  rewrite (follows_boundary _ _ Hlink2) in HstackGet, HlocalsGet.
  assert (Hbounds : Forall (fun value => 0 <= value < i32Modulus) prefix)
    by exact (proj1 (proj2 (proj2 Hstack))).
  assert (Hcondition : 0 <= condition < i32Modulus).
  { pose proof (proj1 (proj2 (proj2 HstackGet))) as H. apply Forall_app in H. exact (Forall_inv (proj2 H)). }
  assert (HentrySteps : Steps (stackConfig z prefix [conditionalBlock x tail])
    (stackPrefix prefix (blockContext (stackConfig z [] (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail))))).
  { exact (steps_with_stack_prefix prefix _ _ Hbounds (enter_empty_block z _ Hz HbodyWf)). }
  assert (HgetSteps : Steps (stackPrefix prefix (blockContext (stackConfig z [] (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail))))
    (blockBranchConfig z prefix condition tail [])).
  { apply steps_with_stack_prefix; [exact Hbounds | ].
    change (Steps (blockContext (stackConfig z [] (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail)))
      (blockContext (stackConfig z [condition] (BR_IF (mk_uN 0) :: tail)))).
    apply steps_in_block.
    pose proof (one_spec_step _ _ ltac:(apply stack_config_wf; [exact Hz | constructor | constructor; [exact (Forall_inv HbodyWf) | constructor]])
      ltac:(apply stack_config_wf; [exact Hz | constructor; [exact Hcondition | constructor] | constructor]) HgetStep) as Hsmall.
    exact (stack_suffix_steps z z [] [condition] [LOCAL_GET x] (BR_IF (mk_uN 0) :: tail) Hsmall (Forall_inv_tail HbodyWf)). }
  assert (HbranchRep : blockBranchRepresents z afterGet (beforeBoundary branchRow) (pc + 2) prefix tail [] BeforeBranch
      (blockBranchConfig z prefix condition tail [])).
  { split; [exact Hz | ]. split; [exact HtailWf | ]. split; [constructor | ]. split; [exact HlocalsGet | ].
    split; [exact HbranchPC | ]. exists condition; auto. }
  assert (HbranchImage : blockBranchImage afterGet (pc + 2) tail).
  { split; [exact Hsupport | ]. unfold blockBranchPC, blockExitPC in *.
    replace (pc + 2 + 1) with (pc + 3) by lia.
    replace (pc + 2 + 2 + Z.of_nat (length tail)) with (pc + 4 + Z.of_nat (length tail)) by lia. exact Hbranch2. }
  destruct (block_branch_row_refinement _ _ _ _ _ _ _ _ _ Hprime HbranchFull HbranchProgram HbranchReplay HbranchImage HbranchRep)
    as [position [c1 [Hposition [HbranchSteps Hafter]]]].
  assert (HbeforeSteps : Steps (stackConfig z prefix [conditionalBlock x tail]) c1).
  { eapply spec_steps_transitive; [exact HentrySteps | ]. eapply spec_steps_transitive; eassumption. }
  destruct position; [contradiction | | ].
  - destruct Hafter as [_ [_ [_ [HlocalsAfter [HpcAfter [HstackAfter ->]]]]]].
    assert (Hbody3 : loadedBody afterBranch (pc + 3) tail).
    { eapply loaded_body_same_roms; [exact Hrom3 | ]. eapply loaded_body_same_roms; [exact Hrom2 | ].
      eapply loaded_body_same_roms; eassumption. }
    assert (Hend3 : controlImage afterBranch (pc + 3 + Z.of_nat (length tail)) CODE_END (blockExitPC pc tail)).
    { unfold controlImage in *. preserve_roms Hrom3. preserve_roms Hrom2. preserve_roms Hrom1. exact Hend. }
    destruct (block_fallthrough_refinement tail rows branchRow prefix afterBranch final pc depth z Hprime HtailRun
      (proj1 (proj2 HafterGetRun)) (Forall_inv_tail (Forall_inv_tail (Forall_inv_tail Hkinds)))
      ltac:(cbn [blockBranchPC beforeBoundary afterBoundary boundaryPC] in HpcAfter; lia) HbranchAfterDepth
      Houtside Hbody3 Hend3 Hz HtailWf HtailTyped HtailDelta HstackAfter HlocalsAfter)
      as [middle [endRow [suffix [between [z' [Hsplit [HprefixRun [HsuffixRun [Hsteps [Hrep [Hsame [HfinishPC [HfinishDepth HfinishHeight]]]]]]]]]]]]].
    exists (first :: getRow :: branchRow :: middle ++ [endRow]), suffix, between, z', (stackConfig z' prefix []).
    split; [cbn; rewrite Hsplit, <- app_assoc; reflexivity | ]. split; [discriminate | ]. split.
    + eapply prepend_valid_row; [exact Hfull | exact Hreplay | | ].
      * eapply chain_prefix with (suffix := suffix). cbn [app]. rewrite <- app_assoc. cbn [app]. rewrite <- Hsplit. exact (proj1 (proj2 Hrun)).
      * eapply prepend_valid_row; [exact HgetFull | exact HgetReplay | | ].
        -- eapply chain_prefix with (suffix := suffix). cbn [app]. rewrite <- app_assoc. cbn [app]. rewrite <- Hsplit. exact (proj1 (proj2 HrestRun)).
        -- eapply prepend_valid_row; [exact HbranchFull | exact HbranchReplay | | exact HprefixRun].
           eapply chain_prefix with (suffix := suffix). cbn [app]. rewrite <- app_assoc. cbn [app]. rewrite <- Hsplit. exact (proj1 (proj2 HafterGetRun)).
    + split; [exact HsuffixRun | ]. split; [eapply spec_steps_transitive; eassumption | ].
      cbn [rowsFinalBoundary rowsFinalDepth]. rewrite rows_final_boundary_app, rows_final_depth_app. cbn [rowsFinalBoundary rowsFinalDepth]. split; [exact Hrep | ]. split; [exact Hsame | ].
      split; [exact HfinishPC | ]. split; [exact HfinishDepth | ].
      rewrite (stack_matches_height _ _ _ Hstack). exact HfinishHeight.
  - destruct Hafter as [_ [_ [_ [HlocalsAfter [HpcAfter [HstackAfter ->]]]]]].
    exists [first; getRow; branchRow], rows, afterBranch, z, (stackConfig z prefix []).
    split; [reflexivity | ]. split; [discriminate | ]. split.
    + eapply prepend_valid_row; [exact Hfull | exact Hreplay | | ].
      * eapply chain_prefix; exact (proj1 (proj2 Hrun)).
      * eapply prepend_valid_row; [exact HgetFull | exact HgetReplay | | ].
        -- eapply chain_prefix; exact (proj1 (proj2 HrestRun)).
        -- split; [constructor; [exact HbranchFull | constructor] | ]. split; [constructor | ].
           cbn [eventsForTrace flat_map]. now rewrite app_nil_r.
    + split; [exact HtailRun | ]. split; [exact HbeforeSteps | ]. split.
      * exists prefix. split; [exact HstackAfter | ]. split; [reflexivity | ].
        split; [apply stack_config_wf; assumption || constructor | exact HlocalsAfter].
      * split; [repeat split; reflexivity | ]. split.
        -- cbn [rowsFinalBoundary afterBoundary boundaryPC blockBranchPC] in *. unfold blockExitPC; lia.
        -- split; [exact HbranchAfterDepth | ].
           rewrite (stack_matches_height _ _ _ Hstack). exact (stack_matches_height _ _ _ HstackAfter).
Qed.
