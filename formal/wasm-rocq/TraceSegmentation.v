(* Derive instruction/call chunks from plain rows. The ROM determines which
   opcodes can occur inside a callee; endpoint depth forces its END to occur. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement Wasm2 StraightLineSpec StackBindings StackSimulation StraightLineInstructions
  LocalBindings ProgramBinding CallSpec CallInitialization CallBindings CallContext CallEntry CallScope CallBody
  CallerSpec CallerRefinement ProgramSpec.
Import ListNotations.
Open Scope Z_scope.

Lemma fetched_straight_line_selector : forall initial final row instruction,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  instructionImage initial (row COL_PC_BEFORE) instruction ->
  exists chosen, straightLineSelector chosen /\ row chosen = 1.
Proof.
  intros initial final row instruction Hprime Hfull Hprogram Hreplay [Hop _].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch.
  destruct (encoded_opcode_cases _ _ Hfetch) as [[-> Hcode] | [[-> Hcode] |
    [[x [-> Hcode]] | [x [-> Hcode]]]]];
    [exists COL_SEL_I32_ADD | exists COL_SEL_I32_SUB | exists COL_SEL_LOCAL_GET | exists COL_SEL_LOCAL_SET];
    split; try (cbn; auto); eapply full_opcode_selects; try eassumption; reflexivity || congruence.
Qed.

Lemma fetched_end_selector : forall initial final row pc,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc ->
  initial programOpcodeId [pc] = Some CODE_END -> row COL_SEL_END = 1.
Proof.
  intros initial final row pc Hprime Hfull Hprogram Hreplay Hpc Hop.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hcode.
  eapply full_opcode_selects with (code := CODE_END); try eassumption; reflexivity || congruence.
Qed.

Lemma scan_call_body : forall rows previous body initial final entry depth,
  fieldPrime -> validCircuitRun initial rows final ->
  Chain continuityLinks (previous :: rows) -> Forall (callerRow depth) rows ->
  previous COL_PC_AFTER = entry -> previous COL_CALL_STACK_DEPTH_AFTER = depth + 1 ->
  rowsFinalDepth (previous COL_CALL_STACK_DEPTH_AFTER) rows = depth ->
  loadedBody initial entry body ->
  initial programOpcodeId [entry + Z.of_nat (length body)] = Some CODE_END ->
  exists middle endRow suffix,
    rows = middle ++ endRow :: suffix /\ Forall bodyRow middle /\ endRow COL_SEL_END = 1.
Proof.
  induction rows as [ | row rows IH]; intros previous body initial final entry depth
    Hprime Hrun Hchain Hkinds Hpc Hdepth Hfinish Hbody Hend.
  - cbn [rowsFinalDepth] in Hfinish. lia.
  - destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (Hchain' : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
    assert (HpcBefore : row COL_PC_BEFORE = entry).
    { pose proof (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)); lia. }
    assert (HdepthBefore : row COL_CALL_STACK_DEPTH_BEFORE = depth + 1).
    { pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)); lia. }
    assert (Hrom : sameRoms initial between).
    { apply run_preserves_roms with (rows := [row]). split; [constructor; auto | ].
      split; [constructor | cbn [eventsForTrace flat_map]; rewrite app_nil_r; exact Hreplay]. }
    destruct (proj1 (Forall_inv Hkinds)) as [Hprogram | Hinit].
    + destruct body as [ | instruction body].
      * assert (HendRow : row COL_SEL_END = 1).
        { eapply fetched_end_selector; try eassumption. cbn [length] in Hend. now rewrite Z.add_0_r in Hend. }
        exists [], row, rows. cbn. repeat split; auto.
      * destruct Hbody as [Himage Hbody].
        assert (HimageRow : instructionImage initial (row COL_PC_BEFORE) instruction) by now rewrite HpcBefore.
        destruct (fetched_straight_line_selector _ _ _ _ Hprime Hfull Hprogram Hreplay HimageRow)
          as [chosen [Hchoice Hchosen]].
        pose proof (instruction_row_next_pc _ _ _ _ Hprime Hfull Hprogram Hreplay HimageRow) as Hnext.
        pose proof (full_body_context_quiet row chosen Hprime Hfull Hchoice Hchosen) as [_ [_ [Hsame _]]].
        destruct (IH row body between final (entry + 1) depth Hprime Htail Hchain'
          (Forall_inv_tail Hkinds) ltac:(lia) ltac:(lia) Hfinish
          (loaded_body_same_roms _ _ _ _ Hrom Hbody) ltac:(
            rewrite (Hrom programOpcodeId ltac:(vm_compute; reflexivity));
            replace (entry + 1 + Z.of_nat (length body)) with (entry + Z.of_nat (length (instruction :: body)))
              by (cbn [length]; rewrite Nat2Z.inj_succ; lia); exact Hend))
          as [middle [endRow [suffix [Hsplit [Hmiddle Hselected]]]]].
        exists (row :: middle), endRow, suffix. split; [cbn; now rewrite Hsplit | ].
        split; [constructor; [left; eauto | exact Hmiddle] | exact Hselected].
    + assert (HinitKind : exists chosen, initializationKind chosen /\ row chosen = 1).
      { destruct Hinit; [exists COL_LOCAL_ZERO_ACTIVE_BEFORE | exists COL_PARAM_INIT_ACTIVE_BEFORE];
          split; try assumption; unfold initializationKind; auto. }
      destruct HinitKind as [chosen [Hchoice Hchosen]].
      pose proof (full_initialization_preserves_frame row chosen Hprime Hfull Hchoice Hchosen)
        as [_ [_ Hsame]].
      assert (Hnext : row COL_PC_AFTER = entry).
      { destruct Hinit as [Hz | Hp].
        - pose proof (full_zero_pc_preserved row Hprime Hfull Hz); lia.
        - pose proof (full_parameter_pc_preserved row Hfull Hp); lia. }
      destruct (IH row body between final entry depth Hprime Htail Hchain'
        (Forall_inv_tail Hkinds) Hnext ltac:(lia) Hfinish
        (loaded_body_same_roms _ _ _ _ Hrom Hbody) ltac:(
          rewrite (Hrom programOpcodeId ltac:(vm_compute; reflexivity)); exact Hend))
        as [middle [endRow [suffix [Hsplit [Hmiddle Hselected]]]]].
      exists (row :: middle), endRow, suffix. split; [cbn; now rewrite Hsplit | ].
      split; [constructor; [right; exact Hinit | exact Hmiddle] | exact Hselected].
Qed.

Lemma next_segment : forall instruction rest rows row callees z initial final pc depth c0,
  fieldPrime -> validCircuitRun initial (row :: rows) final ->
  Forall (callerRow depth) (row :: rows) -> row COL_CALL_STACK_DEPTH_BEFORE = depth ->
  rowsFinalDepth depth (row :: rows) = depth ->
  CallerState callees z initial (beforeBoundary row) pc (instruction :: rest) c0 ->
  exists segment suffix,
    row :: rows = segmentRows segment ++ suffix /\ segmentFirst segment = row /\
    segmentMatches instruction segment.
Proof.
  intros instruction rest rows row callees z initial final pc depth c0 Hprime Hrun Hkinds Hdepth Hfinish
    [Hloaded Htyped Hrep Hpc].
  destruct (proj2 (Forall_inv Hkinds) Hdepth) as [Hprogram _].
  destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
  assert (Hrom : sameRoms initial between).
  { apply run_preserves_roms with (rows := [row]). split; [constructor; auto | ].
    split; [constructor | cbn [eventsForTrace flat_map]; rewrite app_nil_r; exact Hreplay]. }
  assert (Hcase : (exists index f, instruction = CALL index /\ callees (index :> N) = Some f /\
      typedCall z index (guestModule f) (guestTypeIndex f) (guestParams f) (guestResults f)
        (guestLocals f) (guestBody f) /\
      loadedCall initial pc (pc + 1) z index (guestAddress f) (guestModule f) (guestTypeIndex f)
        (guestParams f) (guestResults f) (guestLocals f) (guestBody f) (guestEntry f)) \/
    (forall index, instruction <> CALL index)).
  { destruct instruction; try (right; intros; discriminate).
    left. cbn [callerFacts loadedCaller] in Htyped, Hloaded.
    match type of Htyped with context [callees (?x :> N)] =>
      destruct (callees (x :> N)) as [f | ] eqn:Hlookup; [exists x, f | contradiction] end.
    split; [reflexivity | ]. split; [exact Hlookup | ]. split; [exact (proj1 Htyped) | exact (proj1 Hloaded)]. }
  destruct Hcase as [[index [f [-> [Hlookup [HcallTyped Himage]]]]] | Hordinary].
  - destruct HcallTyped as [Hwf [Hindex [Hfunction [Hsupported [Hvalidated [Hp Hr]]]]]].
    destruct Himage as [Hdirect [Hframe [Hbody [Hend Hedge]]]].
    destruct (direct_call_entry_bound initial between row pc (pc + 1) z index
      (guestAddress f) (guestModule f) (guestTypeIndex f) (repeat 0 (guestParams f))
      (guestResults f) (guestLocals f) (guestBody f) (guestEntry f)
      Hprime Hfull Hprogram Hpc Hreplay ltac:(now rewrite repeat_length) ltac:(now rewrite repeat_length) Hr)
      as [Hcall [Htarget [_ [_ [_ Hentry]]]]].
    destruct (full_direct_guest_call row Hprime Hfull Hcall Htarget) as [Hguest [Hpush Hpop]].
    pose proof (full_call_frame_entry row Hprime Hfull Hguest Hpush Hpop) as [_ [_ HafterDepth]].
    destruct (scan_call_body rows row (guestBody f) between final (guestEntry f) depth Hprime Htail
      (proj1 (proj2 Hrun)) (Forall_inv_tail Hkinds) Hentry ltac:(lia) Hfinish
      (loaded_body_same_roms _ _ _ _ Hrom Hbody) ltac:(
        rewrite (Hrom programOpcodeId ltac:(vm_compute; reflexivity)); exact Hend))
      as [middle [endRow [suffix [Hsplit [Hmiddle Hselected]]]]].
    exists (callSegment row middle endRow), suffix. split.
    + cbn [segmentRows app]. rewrite Hsplit, <- app_assoc. reflexivity.
    + cbn [segmentFirst segmentMatches]. auto.
  - exists (instructionSegment row), rows. split; [reflexivity | ]. split; [reflexivity | ].
    destruct instruction; cbn [segmentMatches]; auto. exfalso. eapply Hordinary. reflexivity.
Qed.

Lemma rows_final_boundary_app : forall prefix suffix boundary,
  rowsFinalBoundary boundary (prefix ++ suffix) =
    rowsFinalBoundary (rowsFinalBoundary boundary prefix) suffix.
Proof. induction prefix; intros; cbn; auto. Qed.

Lemma rows_final_depth_app : forall prefix suffix depth,
  rowsFinalDepth depth (prefix ++ suffix) = rowsFinalDepth (rowsFinalDepth depth prefix) suffix.
Proof. induction prefix; intros; cbn; auto. Qed.

Lemma segment_final_boundary : forall segment boundary,
  rowsFinalBoundary boundary (segmentRows segment) = afterBoundary (segmentLast segment).
Proof.
  intros [row | first middle last] boundary; cbn [segmentRows segmentLast rowsFinalBoundary]; [reflexivity | ].
  rewrite rows_final_boundary_app. reflexivity.
Qed.

Lemma segment_final_depth : forall segment depth,
  rowsFinalDepth depth (segmentRows segment) = segmentLast segment COL_CALL_STACK_DEPTH_AFTER.
Proof.
  intros [row | first middle last] depth; cbn [segmentRows segmentLast rowsFinalDepth]; [reflexivity | ].
  rewrite rows_final_depth_app. reflexivity.
Qed.

Lemma segment_suffix_start : forall segment suffix depth,
  Chain continuityLinks (segmentRows segment ++ suffix) ->
  rowsFinalDepth depth (segmentRows segment) = depth ->
  rowsStartAt (afterBoundary (segmentLast segment)) depth suffix.
Proof.
  intros segment [ | row rest] depth Hchain Hdepth; [exact I | ].
  assert (Hlink : Follows continuityLinks (segmentLast segment) row).
  { destruct segment as [first | first middle last]; cbn [segmentRows segmentLast app] in *.
    - inversion Hchain; subst; assumption.
    - rewrite <- app_assoc in Hchain. cbn [app] in Hchain.
      clear Hdepth. revert first Hchain. induction middle as [ | next middle IH]; intros first Hchain.
      + inversion Hchain; subst. match goal with H : Chain _ (last :: _) |- _ => inversion H; subst; assumption end.
      + apply (IH next). inversion Hchain; subst; assumption. }
  split; [symmetry; now apply follows_boundary | ].
  rewrite segment_final_depth in Hdepth.
  pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)). lia.
Qed.

Theorem valid_caller_run_has_segments : forall code rows callees z initial final boundary pc depth c0,
  fieldPrime -> validCircuitRun initial rows final -> callerFragment boundary depth rows ->
  CallerState callees z initial boundary pc code c0 ->
  initial programOpcodeId [pc + Z.of_nat (length code)] = Some CODE_END ->
  initial pcEdgeKindId [pc + Z.of_nat (length code)] = Some 1 ->
  exists executed remaining segments,
    code = executed ++ remaining /\ rows = callerRows segments /\
    Forall2 segmentMatches executed segments /\ callerStartsAt boundary segments.
Proof.
  induction code as [ | instruction rest IH];
    intros rows callees z initial final boundary pc depth c0 Hprime Hrun [Hstart [Hkinds Hfinish]]
      Hstate Hend Hedge; destruct rows as [ | row rows].
  - exists [], [], []. repeat split; constructor.
  - destruct Hstart as [Hboundary Hdepth].
    destruct (proj2 (Forall_inv Hkinds) Hdepth) as [Hprogram [HnoPop HnoHalt]].
    destruct Hstate as [Hloaded Htyped Hrep Hpc].
    destruct (valid_run_head _ _ _ _ Hrun) as [between [Hfull [Hreplay Htail]]].
    assert (HrowPC : row COL_PC_BEFORE = pc).
    { change (boundaryPC (beforeBoundary row) = pc). now rewrite Hboundary. }
    cbn [length] in Hend, Hedge. rewrite Z.add_0_r in Hend, Hedge.
    pose proof (fetched_end_selector _ _ _ _ Hprime Hfull Hprogram Hreplay HrowPC Hend) as Hselected.
    pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
    rewrite HrowPC, Hedge in Hkind. injection Hkind as Hkind.
    destruct (full_function_end_exits row Hprime Hfull Hselected (eq_sym Hkind)); congruence.
  - exists [], (instruction :: rest), []. repeat split; constructor.
  - destruct Hstart as [Hboundary Hdepth].
    assert (HrowState : CallerState callees z initial (beforeBoundary row) pc (instruction :: rest) c0)
      by now rewrite Hboundary.
    destruct (next_segment instruction rest rows row callees z initial final pc depth c0 Hprime
      Hrun Hkinds Hdepth Hfinish HrowState) as [segment [suffix [Hsplit [Hfirst Hmatch]]]].
    rewrite Hsplit in Hrun, Hkinds, Hfinish.
    destruct (valid_run_split (segmentRows segment) suffix initial final Hrun)
      as [between [HsegmentRun HtailRun]].
    destruct HrowState as [Hloaded Htyped Hrep Hpc].
    assert (HsegmentPC : boundaryPC (beforeBoundary (segmentFirst segment)) = pc) by now rewrite Hfirst.
    assert (Hready : callerFacts callees z (stackHeight (beforeBoundary (segmentFirst segment))) (instruction :: rest) /\
      represents z initial (beforeBoundary (segmentFirst segment)) (instruction :: rest) c0)
      by (rewrite Hfirst; auto).
    destruct (caller_segment_refinement instruction rest segment callees z initial between pc c0
      Hprime HsegmentRun Hmatch Hloaded HsegmentPC Hready)
      as [z' [c1 [Hsteps [[Htyped' Hrep'] [Hpc' [Hcaller Hdepth']]]]]].
    assert (HsegmentDepth : rowsFinalDepth depth (segmentRows segment) = depth).
    { rewrite segment_final_depth, Hdepth', Hfirst. exact Hdepth. }
    assert (Hfragment : callerFragment (afterBoundary (segmentLast segment)) depth suffix).
    { split; [eapply segment_suffix_start; [exact (proj1 (proj2 Hrun)) | exact HsegmentDepth] | ].
      split; [apply Forall_app in Hkinds; tauto | ].
      rewrite rows_final_depth_app, HsegmentDepth in Hfinish. exact Hfinish. }
    pose proof (run_preserves_roms _ _ _ HsegmentRun) as Hrom.
    assert (Hstate' : CallerState callees z' between (afterBoundary (segmentLast segment)) (pc + 1) rest c1).
    { constructor; try assumption. eapply loaded_caller_preserved; [exact Hrom | exact Hcaller | exact (proj2 Hloaded)]. }
    assert (Hend' : between programOpcodeId [pc + 1 + Z.of_nat (length rest)] = Some CODE_END).
    { rewrite (Hrom programOpcodeId ltac:(vm_compute; reflexivity)).
      cbn [length] in Hend. rewrite Nat2Z.inj_succ in Hend.
      replace (pc + 1 + Z.of_nat (length rest)) with (pc + Z.succ (Z.of_nat (length rest))) by lia. exact Hend. }
    assert (Hedge' : between pcEdgeKindId [pc + 1 + Z.of_nat (length rest)] = Some 1).
    { rewrite (Hrom pcEdgeKindId ltac:(vm_compute; reflexivity)).
      cbn [length] in Hedge. rewrite Nat2Z.inj_succ in Hedge.
      replace (pc + 1 + Z.of_nat (length rest)) with (pc + Z.succ (Z.of_nat (length rest))) by lia. exact Hedge. }
    destruct (IH suffix callees z' between final (afterBoundary (segmentLast segment)) (pc + 1) depth c1
      Hprime HtailRun Hfragment Hstate' Hend' Hedge')
      as [executed [remaining [segments [Hcode [Hrows [Hmatches Hstart']]]]]].
    exists (instruction :: executed), remaining, (segment :: segments).
    split; [cbn; now rewrite Hcode | ]. split.
    + change (row :: rows = segmentRows segment ++ callerRows segments).
      rewrite <- Hrows. exact Hsplit.
    + split; [constructor; assumption | ]. cbn [callerStartsAt]. now rewrite Hfirst.
Qed.

Lemma caller_rows_final_boundary : forall segments boundary,
  rowsFinalBoundary boundary (callerRows segments) = callerFinalBoundary boundary segments.
Proof.
  induction segments as [ | segment segments IH]; intros boundary; [reflexivity | ].
  cbn [callerRows flat_map callerFinalBoundary].
  rewrite rows_final_boundary_app, segment_final_boundary. apply IH.
Qed.
