(* A complete direct i32 guest call: derive initialization and body segmentation,
   execute the body in SpecTec's frame, then restore the suspended caller. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement ProgramBinding Wasm2 WasmIntegers Locals StraightLineSpec StackSimulation LocalSimulation
  LocalBindings CallSpec SpecTecCalls CallInitialization CallBindings CallMemory FrameInitialization
  CallEntry CallContext CallScope CallBody CallReturn.
Import ListNotations.
Open Scope Z_scope.

Theorem direct_call_run_refinement : forall initial final callRow middle endRow pc returnPC
    caller index address module typeIndex prefix arguments results locals body entry,
  fieldPrime ->
  loadedCall initial pc returnPC caller index address module typeIndex
    (length arguments) results locals body entry ->
  typedCall caller index module typeIndex (length arguments) results locals body ->
  validCircuitRun initial (callRow :: middle ++ [endRow]) final ->
  callRow COL_IS_PROGRAM_ROW = 1 -> endRow COL_SEL_END = 1 -> Forall bodyRow middle ->
  callRow COL_PC_BEFORE = pc ->
  stackMatches initial (beforeBoundary callRow) (prefix ++ arguments) ->
  localsMatch initial (beforeBoundary callRow) caller ->
  exists values,
    length values = results /\
    Steps (stackConfig caller (prefix ++ arguments) [CALL index])
      (stackConfig caller (prefix ++ values) []) /\
    stackMatches final (afterBoundary endRow) (prefix ++ values) /\
    localsMatch final (afterBoundary endRow) caller /\
    endRow COL_PC_AFTER = returnPC /\
    endRow COL_LOCALS_FBP_AFTER = callRow COL_LOCALS_FBP_BEFORE /\
    endRow COL_STACK_FRAME_BASE_AFTER = callRow COL_STACK_FRAME_BASE_BEFORE /\
    endRow COL_CALL_STACK_DEPTH_AFTER = callRow COL_CALL_STACK_DEPTH_BEFORE.
Proof.
  intros initial final callRow middle endRow pc returnPC caller index address module typeIndex
    prefix arguments results locals body entry Hprime
    [Himage [HcallerImage [Hloaded [HendImage HkindImage]]]]
    [Hstate [Hindex [HfunctionWf [Hsupported [Htyped [Harity [Hp Hr]]]]]]]
    Hrun HcallProgram HendSelected Hkinds Hpc Hstack HcallerLocals.
  assert (HendFull : FullRelation.satFull endRow).
  { apply (proj1 (Forall_forall _ _) (proj1 Hrun)). right. apply in_or_app. right. left. reflexivity. }
  pose proof (full_selected_is_program_row endRow COL_SEL_END Hprime HendFull
    ltac:(vm_compute; tauto) HendSelected) as HendProgram.
  destruct (direct_call_initialization_refinement initial final callRow middle endRow pc returnPC
    caller index address module typeIndex prefix arguments results locals body entry Hprime Hrun
    HcallProgram HendProgram Hpc Himage Hp Hr Hstack)
    as [initialization [rows [ready [Hsplit [HinitLength [HinitKinds [HentryRun [HrestRun
      [Hempty [HcalleeLocals [HzDone HpDone]]]]]]]]]]].
  destruct (valid_run_head _ _ _ _ HentryRun) as [afterCall [HcallFull [HcallReplay HinitRun]]].
  pose proof (proj1 (proj2 HentryRun)) as HinitChain.
  assert (HbodyChain : Chain continuityLinks (finalRow callRow initialization :: rows ++ [endRow])).
  { apply chain_after_prefix. pose proof (proj1 (proj2 Hrun)) as Hchain.
    rewrite Hsplit, <- app_assoc in Hchain. exact Hchain. }
  destruct (valid_run_split rows [endRow] ready final HrestRun) as [beforeEnd [HbodyRun HendRun]].
  assert (HbodyOnlyChain : Chain continuityLinks (finalRow callRow initialization :: rows)).
  { eapply (chain_prefix _ (finalRow callRow initialization :: rows) [endRow]). exact HbodyChain. }
  assert (HlastInit : FullRelation.satFull (finalRow callRow initialization)).
  { eapply final_row_satisfies. exact (proj1 HentryRun). }
  assert (HrowKinds : Forall bodyRow rows).
  { rewrite Hsplit in Hkinds. apply Forall_app in Hkinds. exact (proj2 Hkinds). }
  pose proof (body_rows_initialized rows (finalRow callRow initialization) ready beforeEnd Hprime HlastInit
    HbodyRun HbodyOnlyChain HzDone HpDone HrowKinds) as Hchoices.
  assert (HprogramRows : Forall (fun w : Witness => w COL_IS_PROGRAM_ROW = 1) rows).
  { apply Forall_forall. intros w Hin.
    destruct (proj1 (Forall_forall _ _) Hchoices w Hin) as [chosen [Hchoice Hselected]].
    eapply full_selected_is_program_row; [exact Hprime | | apply straight_line_selector_in; exact Hchoice | exact Hselected].
    exact (proj1 (Forall_forall _ _) (proj1 HbodyRun) w Hin). }
  destruct (direct_call_entry_bound initial afterCall callRow pc returnPC caller index address module typeIndex
    arguments results locals body entry Hprime HcallFull HcallProgram Hpc HcallReplay Himage Hp Hr)
    as [HcallSelected [Htarget [Hparams [HzSeed [HpSeed HentryPC]]]]].
  destruct (initialization_rows_scope initialization callRow afterCall ready Hprime HinitRun HinitChain HinitKinds)
    as [HinitEffect [HinitDepth [HinitBase [HinitFbp HinitPC]]]].
  assert (HloadedReady : loadedBody ready entry body).
  { eapply loaded_body_preserved; [exact Hloaded | exact (proj2 (proj2 HentryRun)) | ].
    intros memory addr Hreadonly. apply read_only_trace. exact Hreadonly. }
  assert (HendImageReady : ready programOpcodeId [entry + Z.of_nat (length body)] = Some CODE_END).
  { rewrite (replay_preserves _ _ _ _ _ (proj2 (proj2 HentryRun))
      (read_only_trace _ _ _ _ opcode_memory_read_only)). exact HendImage. }
  pose proof (body_rows_length body rows (finalRow callRow initialization) endRow ready final entry
    Hprime HrestRun HbodyChain Hchoices HendSelected ltac:(lia) HloadedReady HendImageReady) as Hlength.
  assert (Hbounds : Forall (fun value => 0 <= value < i32Modulus) prefix /\
    Forall (fun value => 0 <= value < i32Modulus) arguments).
  { apply Forall_app. exact (proj1 (proj2 (proj2 Hstack))). }
  destruct Hbounds as [HprefixBounds HargBounds].
  destruct caller as [s callerFrame].
  assert (HframeReady :
    programTyped (length (stateLocals (mk_state s (enteredFrame module arguments locals))))
      (stackHeight (afterBoundary (finalRow callRow initialization))) body /\
    represents (mk_state s (enteredFrame module arguments locals)) ready
      (afterBoundary (finalRow callRow initialization)) body
      (stackConfig (mk_state s (enteredFrame module arguments locals)) [] body)).
  { apply represented_values. exists []. split; [exact Hempty | ]. split.
    - cbn [stateLocals enteredFrame LOCALS]. rewrite length_app, length_map, repeat_length. exact Htyped.
    - split; [reflexivity | ]. split; [eapply entered_body_wf; eassumption | exact HcalleeLocals]. }
  destruct (body_run_refinement body rows (finalRow callRow initialization) ready beforeEnd entry
    (mk_state s (enteredFrame module arguments locals)) _ Hprime HbodyRun HbodyOnlyChain HprogramRows
    Hlength HloadedReady ltac:(lia) HframeReady)
    as [z' [c1 [HbodySteps [HbodyRep [HbodyPC [HbodyEffect [HbodyDepth [HbodyBase
      [HbodyFbp [HbodySP Hstore]]]]]]]]]].
  destruct z' as [s' calleeFinal]. cbn [fun_store] in Hstore. subst s'.
  apply represented_values in HbodyRep. destruct HbodyRep as [values [HresultStack [_ [Hc1 [HresultWf HresultLocals]]]]]. subst c1.
  pose proof (final_row_follows (finalRow callRow initialization) rows endRow HbodyChain) as HendLink.
  assert (HendPC : endRow COL_PC_BEFORE = entry + Z.of_nat (length body)).
  { rewrite <- (HendLink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)). exact HbodyPC. }
  destruct (valid_run_head _ _ _ _ HendRun) as [afterEnd [HendFull' [HendReplay HemptyRun]]].
  assert (HafterEnd : afterEnd = final) by (destruct HemptyRun as [_ [_ Hnil]]; inversion Hnil; reflexivity).
  subst afterEnd.
  assert (Hedge : endRow COL_PC_EDGE_KIND = 1).
  { pose proof (circuit_row_edge_kind_fetch _ _ _ HendReplay HendProgram) as Hread.
    rewrite HendPC in Hread.
    rewrite (replay_preserves _ _ _ _ _ (proj2 (proj2 HbodyRun))
      (read_only_trace _ _ _ _ edge_kind_memory_read_only)) in Hread.
    rewrite (replay_preserves _ _ _ _ _ (proj2 (proj2 HentryRun))
      (read_only_trace _ _ _ _ edge_kind_memory_read_only)), HkindImage in Hread. congruence. }
  assert (Heffect : FrameEffect (callRow COL_STACK_FRAME_BASE_AFTER) (callRow COL_LOCALS_FBP_AFTER) afterCall beforeEnd).
  { rewrite HinitBase, HinitFbp in HbodyEffect. eapply frame_effect_trans; eassumption. }
  pose proof Himage as [Haddress [Hfunction [_ [_ [HreturnImage _]]]]].
  assert (HendDepth : endRow COL_CALL_STACK_DEPTH_BEFORE = callRow COL_CALL_STACK_DEPTH_AFTER).
  { rewrite <- (HendLink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)). lia. }
  rewrite <- Hpc in HreturnImage.
  destruct (call_return_restores_context initial afterCall beforeEnd final callRow endRow returnPC
    Hprime HcallFull HendFull HcallSelected Htarget HendSelected Hedge HcallReplay HendReplay HreturnImage
    HendDepth (preserves_call_context _ _ _ _ Heffect))
    as [HreturnPC [HreturnFbp [HreturnBase [HreturnDepth [HreturnSP [HreturnStack HreturnLocals]]]]]].
  destruct (full_direct_guest_call callRow Hprime HcallFull HcallSelected Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_call_frame_entry callRow Hprime HcallFull Hguest Hpush Hpop) as [HcallFbp [HcallBase HcallDepth]].
  destruct (full_direct_guest_call_stack callRow Hprime HcallFull HcallSelected Htarget) as [HcallLo [HcallHi HcallSP]].
  assert (HcallStack : forall addr, afterCall stackId addr = initial stackId addr).
  { intros addr. eapply replay_preserves; [exact HcallReplay | apply stack_writes_inactive; assumption]. }
  assert (HcallLocals : forall addr, afterCall localLoId addr = initial localLoId addr).
  { eapply no_local_writes_preserve; [exact HcallReplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact HcallFull | left; reflexivity | exact HcallSelected]. }
  assert (HcalleeBase : callRow COL_STACK_FRAME_BASE_AFTER =
      callRow COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix)).
  { pose proof (proj1 (proj2 Hstack)) as Hsp.
    cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. lia. }
  assert (HcallerCount : callRow COL_CURRENT_FUNCTION_NUM_LOCALS = Z.of_nat (length (stateLocals (mk_state s callerFrame)))).
  { destruct HcallerImage as [reference [Hreference Hcount]].
    pose proof (current_function_read initial afterCall callRow HcallReplay
      (full_frame_lookup_active callRow COL_IS_PROGRAM_ROW Hprime HcallFull ltac:(cbn; tauto) HcallProgram)) as HrefRead.
    rewrite Hpc, Hreference in HrefRead. injection HrefRead as HrefValue.
    pose proof (program_row_local_count_read initial afterCall callRow HcallReplay HcallProgram) as HcountRead.
    rewrite <- HrefValue, Hcount in HcountRead. congruence. }
  assert (HresultLength : length values = results).
  { pose proof (proj1 (proj2 Hempty)) as HzeroSP.
    pose proof (proj1 (proj2 HresultStack)) as HfinalSP.
    cbn [afterBoundary boundarySP boundaryStackBase length] in HzeroSP, HfinalSP.
    rewrite Harity in HbodySP. injection HbodySP as HbodySP. lia. }
  assert (HstackFinal : stackMatches final (afterBoundary endRow) (prefix ++ values)).
  { eapply stack_rejoins_caller with (arguments := arguments) (calleeBase := callRow COL_STACK_FRAME_BASE_AFTER)
      (beforeEnd := beforeEnd); [exact Hstack | exact HcalleeBase | exact HreturnBase | | | | | exact HreturnStack].
    - pose proof (proj1 (proj2 HresultStack)) as Hsp.
      cbn [afterBoundary boundarySP boundaryStackBase] in Hsp.
      pose proof (HendLink COL_SP_AFTER COL_SP_BEFORE ltac:(vm_compute; tauto)). lia.
    - exact (proj1 (proj2 (proj2 HresultStack))).
    - intros n value Hlookup. pose proof (proj2 (proj2 (proj2 HresultStack)) n value Hlookup) as Hcell.
      cbn [afterBoundary boundaryStackBase] in Hcell. rewrite HbodyBase, HinitBase in Hcell. exact Hcell.
    - intros n Hn. rewrite (preserves_stack_prefix _ _ _ _ Heffect n Hn). apply HcallStack. }
  assert (HlocalsFinal : localsMatch final (afterBoundary endRow) (mk_state s callerFrame)).
  { intros n v Hlookup. destruct (HcallerLocals n v Hlookup) as [value [Hv [Hb Hcell]]].
    exists value. split; [exact Hv | ]. split; [exact Hb | ].
    cbn [afterBoundary boundaryLocalsBase]. rewrite HreturnLocals, HreturnFbp.
    rewrite (preserves_other_locals _ _ _ _ Heffect), HcallLocals; [exact Hcell | ].
    assert (Hnonempty : (0 < length (stateLocals (mk_state s callerFrame)))%nat).
    { assert (n < length (stateLocals (mk_state s callerFrame)))%nat.
      { apply nth_error_Some. rewrite Hlookup. discriminate. } lia. }
    lia. }
  exists values. split; [exact HresultLength | ]. split.
  - pose proof (direct_i32_call_enters s callerFrame index address module typeIndex arguments results locals body
      Hstate Hindex Haddress Hfunction HfunctionWf HargBounds) as Henter.
    assert (HcallerWf : wf_frame callerFrame) by (inversion Hstate; assumption).
    pose proof (steps_in_call_frame callerFrame (N.of_nat results) _ _ HcallerWf HbodySteps) as HbodyContext.
    assert (HcalleeWf : wf_frame calleeFinal).
    { inversion HresultWf; subst. match goal with H : wf_state _ |- _ => inversion H; assumption end. }
    pose proof (i32_function_returns s callerFrame calleeFinal values Hstate HcalleeWf
      (proj1 (proj2 (proj2 HresultStack)))) as Hleave.
    rewrite HresultLength in Hleave.
    assert (Hall : Steps (stackConfig (mk_state s callerFrame) arguments [CALL index])
        (stackConfig (mk_state s callerFrame) values [])).
    { eapply spec_steps_transitive; [exact Henter | ].
      eapply spec_steps_transitive; [exact HbodyContext | ].
      cbn [callContext stackConfig] in Hleave.
      unfold stackConfig. cbn [map]. rewrite app_nil_r. exact Hleave. }
    pose proof (steps_with_stack_prefix prefix _ _ HprefixBounds Hall) as Hfull.
    unfold stackPrefix, stackConfig in Hfull |- *. rewrite !map_app, <- !app_assoc. exact Hfull.
  - split; [exact HstackFinal | ]. split; [exact HlocalsFinal | ]. repeat split; assumption.
Qed.
