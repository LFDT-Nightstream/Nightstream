(* Rows that change the active frame: a direct call enters its callee, frame
   initialization rows set the callee's locals, and a function's END returns to
   its suspended caller. Suspended callers survive every row: their contexts lift
   SpecTec steps, and their memory lies below the active frame. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerRefinement CallSpec CallEntry
  CallBindings CallContext CallMemory CallReturn CallInitialization CallScope LocalBindings FrameInitialization
  Locals ProgramSpec ProgramEncoding CodeLayout BlockEncoding Fragment CodeFacts ControlDecode ControlInvariant
  FrameFacts ControlBridge FrameRows ProgramBinding Selectors CircuitFacts SlideLayout SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Lemma split_top_values : forall (values : list Z) n,
  (n <= length values)%nat -> exists prefix top, values = prefix ++ top /\ length top = n.
Proof.
  intros values n H. exists (firstn (length values - n) values), (skipn (length values - n) values).
  split; [symmetry; apply firstn_skipn | rewrite length_skipn; lia].
Qed.

Lemma stack_config_rejoin : forall z values results code,
  stackPrefix values (instructionSuffix code (mk_config z (List.map i32Constant results))) =
    stackConfig z (values ++ results) code.
Proof. intros. unfold stackPrefix, instructionSuffix, stackConfig. rewrite map_app, <- app_assoc. reflexivity. Qed.

Lemma steps_refl_wf : forall c, wf_config c -> Steps c c.
Proof. intros [z code] H. apply Steps__refl. exact H. Qed.

Lemma entered_frame_locals : forall f arguments,
  enteredFrame (guestModule f) arguments (guestLocals f) = activeFrame f (arguments ++ repeat 0 (guestLocals f)).
Proof. intros. unfold enteredFrame, activeFrame. rewrite map_app, map_repeat. reflexivity. Qed.

(* A frame keeps its view when its operand cells and locals are unchanged. *)
Lemma frame_at_preserved : forall program memory memory' base fbp pending moved distance f view,
  FrameAt program memory base fbp pending moved distance f view ->
  (forall index, (index < length (viewOperands view))%nat ->
    memory' stackId [2 * operandCell base (length (viewOperands view)) moved distance index] =
    memory stackId [2 * operandCell base (length (viewOperands view)) moved distance index]) ->
  (forall index, (index < length (viewLocals view))%nat ->
    memory' stackId [2 * (fbp + Z.of_nat index)] = memory stackId [2 * (fbp + Z.of_nat index)]) ->
  FrameAt program memory' base fbp pending moved distance f view.
Proof.
  intros program memory memory' base fbp pending moved distance f view
    [Hf [Hpath [Hheights [Hfbp [Hbase [Hbounds [Hcells [Hlen [Hlb Hlc]]]]]]]]] Hstack Hlocals.
  unfold FrameAt. split; [exact Hf | ]. split; [exact Hpath | ]. split; [exact Hheights | ].
  split; [exact Hfbp | ]. split; [exact Hbase | ]. split; [exact Hbounds | ].
  split; [intros index value Hi; rewrite Hstack; [exact (Hcells _ _ Hi) | apply nth_error_Some; rewrite Hi; discriminate] | ].
  split; [exact Hlen | ]. split; [exact Hlb | ].
  intros index value Hi. rewrite Hlocals; [exact (Hlc _ _ Hi) | apply nth_error_Some; rewrite Hi; discriminate].
Qed.

(* Callers lie below the active frame's base pointer: earlier stack cells, and
   return contexts saved at lower depths. *)
Lemma suspended_stack_preserved : forall program stack memory memory' callee fbp depth,
  SuspendedStack program memory callee fbp depth stack ->
  (forall index, index < fbp -> memory' stackId [2 * index] = memory stackId [2 * index]) ->
  (forall id column address, In (id, column) contextMemories -> address < depth ->
    memory' id [address] = memory id [address]) ->
  SuspendedStack program memory' callee fbp depth stack.
Proof.
  intros program stack. induction stack as [ | suspended rest IH];
    intros memory memory' callee fbp depth H Hstack Hcontext; [exact I | ].
  destruct H as [x [Hframe [Hnth [Hcallee [Hres [Hfbp [Hpc [HsavedFbp [HsavedBase Hrest]]]]]]]]].
  pose proof Hframe as [_ [_ [_ [HcallerFbp [HcallerBase [_ [_ [Hlen _]]]]]]]].
  cbn [SuspendedStack]. exists x.
  split.
  { eapply frame_at_preserved; [exact Hframe | | ].
    - intros index Hindex. rewrite operand_cell_unmoved by exact Hindex. apply Hstack. lia.
    - intros index Hindex. apply Hstack. rewrite Hlen in Hindex. lia. }
  split; [exact Hnth | ]. split; [exact Hcallee | ]. split; [exact Hres | ]. split; [exact Hfbp | ].
  split; [rewrite (Hcontext callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE) by (cbn; auto || lia); exact Hpc | ].
  split; [rewrite (Hcontext callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE) by (cbn; auto || lia);
    exact HsavedFbp | ].
  split; [rewrite (Hcontext callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE) by (cbn; auto || lia);
    exact HsavedBase | ].
  eapply IH; [exact Hrest | | ].
  - intros index Hindex. apply Hstack. pose proof (Zle_0_nat (length (viewOperands (callView suspended)))).
    pose proof (Zle_0_nat (frameLocals (callerFunction suspended))). lia.
  - intros id column address Hin Haddress. apply (Hcontext id column); [exact Hin | lia].
Qed.

Lemma frame_effect_suspended : forall program stack callee fbp depth memory memory',
  FrameEffect fbp memory memory' ->
  SuspendedStack program memory callee fbp depth stack ->
  SuspendedStack program memory' callee fbp depth stack.
Proof.
  intros program stack callee fbp depth memory memory' [Hstack Hcontext] H.
  eapply suspended_stack_preserved; [exact H | exact Hstack | ].
  intros id column address Hin _. eapply Hcontext. exact Hin.
Qed.

Section Calls.
Variables (program : Program) (s : store).
Hypothesis Hstore : store_FUNCS s = program.
Hypothesis Hwf : wf_store s.
Hypothesis Hvalid : programValidated program s.

(* What a suspended caller's context needs to lift its callee's steps. *)
Definition SuspendedReady (suspended : SuspendedCall) : Prop :=
  let g := callerFunction suspended in
  let view := callView suspended in
  let position := viewPosition view in
  LabelsReady (guestBody g) (positionLabels position) (viewSegments view) /\
  Forall (fun value => 0 <= value < i32Modulus) (viewValues view) /\
  Forall wf_instr (skipn (S (positionOffset position)) (innerCode g position)) /\
  wf_frame (activeFrame g (viewLocals view)).

Lemma frame_ready : forall memory base fbp pending moved distance g view,
  FrameAt program memory base fbp pending moved distance g view ->
  LabelsReady (guestBody g) (positionLabels (viewPosition view)) (viewSegments view) /\
  Forall (fun value => 0 <= value < i32Modulus) (viewValues view) /\
  Forall wf_instr (innerCode g (viewPosition view)) /\
  wf_state (mk_state s (activeFrame g (viewLocals view))).
Proof.
  intros memory base fbp pending moved distance g view Hframe.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [_ [Hready Hcode]]]]].
  pose proof Hframe as [Hg [_ [_ [_ [_ [Hoperands [_ [_ [Hlocals _]]]]]]]]].
  unfold viewOperands in Hoperands. apply Forall_app in Hoperands.
  split; [exact Hready | ]. split; [exact (proj2 Hoperands) | ]. split; [exact Hcode | ].
  exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hg Hlocals).
Qed.

Lemma suspended_stack_ready : forall stack memory callee fbp depth,
  SuspendedStack program memory callee fbp depth stack -> Forall SuspendedReady stack.
Proof.
  induction stack as [ | suspended rest IH]; intros memory callee fbp depth H; [constructor | ].
  destruct H as [x [Hframe [_ [_ [_ [_ [_ [_ [_ Hrest]]]]]]]]].
  constructor; [ | eapply IH; exact Hrest].
  destruct (frame_ready _ _ _ _ _ _ _ _ Hframe) as [Hready [Hvalues [Hcode Hz]]].
  split; [exact Hready | ]. split; [exact Hvalues | ]. split; [apply forall_skipn; exact Hcode | ].
  inversion Hz; assumption.
Qed.

Lemma plug_calls_wf : forall stack c,
  Forall SuspendedReady stack -> wf_config c -> wf_config (plugCalls stack c).
Proof.
  induction stack as [ | suspended rest IH]; intros c Hready Hc; [exact Hc | ].
  inversion Hready as [ | ? ? [Hlabels [Hvalues [Hcode Hframe]]] Hrest]; subst.
  cbn [plugCalls]. apply IH; [exact Hrest | ]. unfold callerConfig, frameConfig.
  apply wrap_labels_wf; [exact Hlabels | ]. apply stack_prefix_wf; [exact Hvalues | ].
  apply instruction_suffix_wf; [exact Hcode | ]. apply call_context_wf; assumption.
Qed.

Lemma steps_in_plug_calls : forall stack before after,
  Forall SuspendedReady stack -> Steps before after -> Steps (plugCalls stack before) (plugCalls stack after).
Proof.
  induction stack as [ | suspended rest IH]; intros before after Hready Hsteps; [exact Hsteps | ].
  inversion Hready as [ | ? ? [Hlabels [Hvalues [Hcode Hframe]]] Hrest]; subst.
  cbn [plugCalls]. apply IH; [exact Hrest | ]. unfold callerConfig, frameConfig.
  apply steps_in_labels; [exact Hlabels | ]. apply steps_with_stack_prefix; [exact Hvalues | ].
  apply steps_with_instruction_suffix; [exact Hcode | ]. apply steps_in_call_frame; assumption.
Qed.

Lemma function_inst_wf : forall index f, functionAt program index = Some f ->
  wf_funcinst (i32Function (guestModule f) (guestTypeIndex f) (guestParams f) (guestResults f)
    (guestLocals f) (guestBody f)).
Proof.
  intros index f Hf. destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) (proj1 Hvalid) _ (nth_error_In _ _ Hnth)) as Hinst.
  inversion Hinst; assumption.
Qed.

Lemma active_state_wf : forall memory boundary position c,
  ActiveState program s memory boundary position c -> wf_config c.
Proof.
  intros memory boundary [view | function arguments] c H.
  - destruct H as [f [Hframe [_ [_ [_ [_ [_ [_ ->]]]]]]]].
    destruct (frame_ready _ _ _ _ _ _ _ _ Hframe) as [Hready [Hvalues [Hcode Hz]]].
    apply wrap_labels_wf; [exact Hready | ].
    apply stack_config_wf; [exact Hz | exact Hvalues | apply forall_skipn; exact Hcode].
  - destruct H as [f [Hf [Hlen [Hargs [_ [_ [_ [_ [_ ->]]]]]]]]].
    apply (entered_body_wf s (activeFrame f []) _ (guestTypeIndex f) _ (guestResults f)).
    + exact (frame_state_wf _ _ _ _ [] Hstore Hwf Hf (Forall_nil _)).
    + rewrite Hlen. exact (function_inst_wf _ _ Hf).
    + exact Hargs.
Qed.

Lemma control_state_wf : forall baseDepth cursor memory boundary c,
  ControlState program s baseDepth cursor memory boundary c -> wf_config c.
Proof.
  intros baseDepth cursor memory boundary c [f [inner [_ [Hactive [Hstack [_ [_ ->]]]]]]].
  apply plug_calls_wf; [eapply suspended_stack_ready; exact Hstack | eapply active_state_wf; exact Hactive].
Qed.

(* A callee being entered: still zeroing its declared locals, or, once that is
   done, at the start of its body. *)
Lemma callee_active_state : forall function callee arguments memory boundary,
  functionAt program function = Some callee -> length arguments = guestParams callee ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  InitializingFrame memory boundary arguments (Z.of_nat (guestLocals callee)) (boundaryZeroRemaining boundary) ->
  initializationFlags boundary -> boundaryMoveRemaining boundary = 0 ->
  boundaryPC boundary = guestEntry callee ->
  exists position, activeFunction program position = Some callee /\
    ActiveState program s memory boundary position
      (stackConfig (mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))) [] (guestBody callee)).
Proof.
  intros function callee arguments memory boundary Hf Hlength Hbounds Hinit Hflags Hmove Hpc.
  pose proof (zero_remaining_bounds _ _ _ _ _ Hinit) as Hzb.
  destruct (Z.eq_dec (boundaryZeroRemaining boundary) 0) as [Hz0 | Hpending].
  2: { exists (entering function arguments). split; [exact Hf | ].
       exists callee. repeat (split; [eassumption | ]). split; [lia | ]. split; [exact Hpc | reflexivity]. }
  pose proof Hinit as [_ Hfbp Hbase Hsp Hargs Hzeros].
  exists (atCode (frameView (codePosition function [] 0) [] [] (arguments ++ repeat 0 (guestLocals callee)))).
  split; [exact Hf | ].
  pose proof (FrameRows.idle_active_state program s memory boundary callee
    (frameView (codePosition function [] 0) [] [] (arguments ++ repeat 0 (guestLocals callee)))) as Hstate.
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hstate.
  unfold innerCode in Hstate. cbn [pathCode] in Hstate.
  replace (stackConfig (mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))) [] (guestBody callee))
    with (frameConfig callee (frameView (codePosition function [] 0) [] [] (arguments ++ repeat 0 (guestLocals callee)))
      (stackConfig (mk_state s (activeFrame callee (arguments ++ repeat 0 (guestLocals callee)))) [] (skipn 0 (guestBody callee))))
    by (unfold frameConfig; cbn [wrapLabels skipn viewPosition positionLabels viewSegments];
        rewrite entered_frame_locals; reflexivity).
  apply Hstate; clear Hstate.
  - apply FrameRows.frame_at_unmoved_intro;
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset];
      unfold viewOperands; cbn [viewSegments viewValues concat app].
    + exact Hf.
    + reflexivity.
    + reflexivity.
    + exact Hfbp.
    + rewrite Hbase, Hlength. unfold frameLocals. lia.
    + constructor.
    + intros [ | index] value Hv; discriminate.
    + unfold frameLocals; rewrite length_app, repeat_length; lia.
    + apply Forall_app. split; [exact Hbounds | ]. apply Forall_forall. intros v Hv. apply repeat_spec in Hv.
      subst v. unfold i32Modulus. lia.
    + intros index value Hv.
      destruct (Nat.lt_ge_cases index (length arguments)) as [Hin | Hout].
      * rewrite nth_error_app1 in Hv by exact Hin. exact (Hargs _ _ Hv).
      * rewrite nth_error_app2 in Hv by exact Hout.
        assert (Hlt : (index - length arguments < guestLocals callee)%nat).
        { rewrite <- (repeat_length 0 (guestLocals callee)). apply nth_error_Some. rewrite Hv. discriminate. }
        apply nth_error_In, repeat_spec in Hv. subst value.
        apply Hzeros. lia.
  - cbn. lia.
  - rewrite Hsp, Hz0. cbn. lia.
  - rewrite Hpc. cbn. lia.
  - destruct Hflags as [Hma Hza]. rewrite Hmove in Hma. rewrite Hmove, Hz0 in Hza. cbn in Hma, Hza.
    repeat split; assumption.
Qed.

(* A direct guest call from any code path: SpecTec enters the callee's frame; the
   circuit saves the caller's return context one level above the callers already
   suspended, and starts the callee's frame at its arguments. *)
Lemma call_row : forall baseDepth stack view f x initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (CALL x) ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) stack ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (length stack) ->
  exists cursor c1, Steps (plugCalls stack inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth stack view f x initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi
    Hsusp Hbase0 Hdepth.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [HbaseLayout [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hi, Hpath, Hheights, Hbounds, Hcells, Hllen, Hlb, Hlc, Hsp, Hpc |- *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup.
  destruct Hfacts as [[HwfCall [callee [Hcallee Hle]]] Hrest]. rewrite Nat.add_0_r in Hrest, Hle.
  cbn [sourceHeight] in Hrest. rewrite Hcallee in Hrest.
  replace (Nat.leb (guestParams callee) (length values)) with true in Hrest by (symmetry; apply Nat.leb_le; exact Hle).
  assert (Hcounts : (guestParams callee < 256)%nat /\ (guestResults callee < 256)%nat).
  { inversion Hsup as [ | ? ? Hhead _]; subst. inversion Hhead; subst.
    match goal with Hg : _ (x :> N) = Some ?g |- _ => assert (g = callee) as -> by congruence end.
    split; assumption. }
  destruct Hcounts as [Hp Hr].
  assert (Hidx : wf_uN 32%N x) by (inversion HwfCall; assumption).
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  destruct (encoded_path_call program initial index f Hloaded Hf labels code offset s (List.map i32Value locals) x
    callee Hpath Hi Hstore ltac:(rewrite length_map; exact Hllen) Hcallee) as [Himage _].
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (split_top_values values (guestParams callee) Hle) as [prefix [arguments [-> Hlen]]].
  rewrite <- Hlen in Himage.
  pose proof (FrameRows.frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack. rewrite app_assoc in Hstack.
  destruct (direct_call_entry_bound initial final row _ (row COL_PC_BEFORE + 1) z x (guestAddress callee)
    (guestModule callee) (guestTypeIndex callee) arguments (guestResults callee) (guestLocals callee)
    (guestBody callee) (guestEntry callee) Hprime Hfull Hprogram Logic.eq_refl Hreplay Himage ltac:(lia) Hr)
    as [HcallSel [Htarget [HparamCount [HzSeed [Hentered Hentry]]]]].
  pose proof (call_starts_frame_initialization initial final row (concat segments ++ prefix) arguments
    (guestLocals callee) Hprime Hfull HcallSel Htarget HparamCount Hentered HzSeed Hreplay Hstack) as Hinit.
  destruct (full_direct_guest_call row Hprime Hfull HcallSel Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_selected_no_host_switch row COL_SEL_CALL Hprime Hfull ltac:(vm_compute; tauto) HcallSel)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  destruct (full_call_frame_entry row Hprime Hfull Hguest Hpush Hpop Hprogram) as [HfbpAfter [_ HdepthAfter]].
  destruct (full_direct_guest_call_stack row Hprime Hfull HcallSel Htarget) as [_ [_ HspAfter]].
  destruct (full_call_context_push row Hfull Hpush) as [Haddr [HsavedFbp [HsavedBase Hchoice]]].
  pose proof (no_stack_writes_preserve _ _ _ Hreplay
    (full_call_end_no_stack_writes row COL_SEL_CALL Hprime Hfull (or_introl Logic.eq_refl) HcallSel)) as HcallStack.
  (* A call seeds no frame move. *)
  assert (HmoveAfter : row COL_FRAME_MOVE_REMAINING_AFTER = 0).
  { destruct Hidle as [_ [Hmr [_ Hma]]]. cbn [beforeBoundary boundaryMoveRemaining boundaryMoveActive] in Hmr, Hma.
    assert (Hone : row COL_ONE = 1) by exact (proj1 Hfull).
    circuit_row ([(COL_ONE, 1); (COL_FRAME_SLIDE_ACTIVE, -1); (COL_GUEST_ENTRY_ACTIVE, -1);
      (COL_CALL_STACK_PUSH_PRESENT, 1); (COL_FRAME_MOVE_ACTIVE_BEFORE, -1)],
      [(COL_FRAME_MOVE_REMAINING_AFTER, 1); (COL_FRAME_MOVE_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    pose proof (full_selected_no_slide row COL_SEL_CALL Hprime Hfull ltac:(vm_compute; tauto) HcallSel
      ltac:(vm_compute; intuition discriminate) Hpop) as Hslide.
    expand_row Hrow. normalize_columns. rewrite Hone, Hslide, Hguest, Hpush, Hma, Hmr in Hrow.
    pose proof (value_bound row COL_FRAME_MOVE_REMAINING_AFTER). field_linear Hrow. }
  pose proof (context_push_records initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedPcMem.
  pose proof (context_push_records initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedFbpMem.
  pose proof (context_push_records initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedBaseMem.
  pose proof (call_return_pc_read initial final row Hreplay Hpush) as HreturnRead.
  destruct Himage as [Haddress [Hfunction [_ [_ [HreturnImage _]]]]].
  rewrite Hchoice, HreturnImage in HreturnRead. injection HreturnRead as HreturnValue.
  apply Forall_app in Hbounds. destruct Hbounds as [HsegBounds HvalBounds].
  apply Forall_app in HvalBounds. destruct HvalBounds as [HprefixBounds HargBounds].
  (* SpecTec enters the callee. *)
  pose proof (program_callee_validates program s (guestModule f) (x :> N) callee (proj1 Hvalid) Hcallee) as Hinst.
  assert (Hfuncwf : wf_funcinst (i32Function (guestModule callee) (guestTypeIndex callee) (length arguments)
      (guestResults callee) (guestLocals callee) (guestBody callee))) by (rewrite Hlen; inversion Hinst; assumption).
  pose proof (direct_i32_call_enters s (activeFrame f locals) x (guestAddress callee) (guestModule callee)
    (guestTypeIndex callee) arguments (guestResults callee) (guestLocals callee) (guestBody callee) Hz Hidx
    Haddress Hfunction Hfuncwf HargBounds) as Henter.
  pose proof Hcallee as HcalleeAt. unfold programCallees in HcalleeAt.
  destruct (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) as [address | ]; [ | discriminate].
  destruct (callee_active_state (N.to_nat address) callee arguments final (afterBoundary row) HcalleeAt Hlen
    HargBounds Hinit (full_initialization_flags row Hprime Hfull) HmoveAfter Hentry)
    as [position [Hactive Hentered']].
  exists (controlCursor (suspendedCall (frameView (codePosition index labels offset) segments prefix locals) f
      (guestResults callee) (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) :: stack) position),
    (plugCalls (suspendedCall (frameView (codePosition index labels offset) segments prefix locals) f
      (guestResults callee) (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) :: stack)
      (stackConfig (mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))) []
        (guestBody callee))).
  split.
  { cbn [plugCalls]. apply steps_in_plug_calls; [eapply suspended_stack_ready; exact Hsusp | ].
    unfold callerConfig, frameConfig.
    cbn [callView callerFunction callResults viewPosition viewSegments viewValues viewLocals positionLabels
      positionOffset].
    apply steps_in_labels; [exact Hready | ].
    rewrite (skipn_at_instruction _ _ _ Hi), stack_config_split.
    apply steps_with_stack_prefix; [exact HprefixBounds | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | exact Henter]. }
  exists callee, (stackConfig (mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))) []
    (guestBody callee)).
  split; [exact Hactive | ]. split; [exact Hentered' | ].
  split; [ | split; [exact Hbase0 | split; [cbn [suspendedCalls length afterBoundary boundaryCallDepth]; lia |
    reflexivity]]].
  cbn [SuspendedStack suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth
    callView callerFunction callResults callerBase callerFbp viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset].
  exists x. split.
  { apply FrameRows.frame_at_unmoved_intro;
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset];
      unfold viewOperands; cbn [viewSegments viewValues].
    - exact Hf.
    - exact Hpath.
    - rewrite Hheights, length_app, Hlen, Nat.add_0_r. reflexivity.
    - exact Hfbp0.
    - exact HbaseLayout.
    - apply Forall_app; split; assumption.
    - intros i value Hv. rewrite HcallStack. apply Hcells. unfold viewOperands. cbn [viewSegments viewValues].
      rewrite app_assoc, nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate].
    - exact Hllen.
    - exact Hlb.
    - intros i value Hv. rewrite HcallStack. exact (Hlc _ _ Hv). }
  split; [exact Hi | ]. split; [exact Hcallee | ]. split; [reflexivity | ].
  split.
  { unfold viewOperands in Hsp |- *. cbn [beforeBoundary boundarySP boundaryStackBase viewSegments viewValues] in Hsp |- *.
    rewrite !length_app in Hsp; rewrite ?length_app.
    rewrite HfbpAfter, HspAfter, HparamCount. lia. }
  replace (row COL_CALL_STACK_DEPTH_AFTER - 1) with (row COL_CALL_STACK_ADDR) by lia.
  split.
  { rewrite HsavedPcMem, <- HreturnValue, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi).
    change (instructionSize (CALL x)) with 1%nat. f_equal. lia. }
  split; [rewrite HsavedFbpMem, HsavedFbp; reflexivity | ].
  split; [rewrite HsavedBaseMem, HsavedBase; reflexivity | ].
  (* The callers already suspended keep their memory: the push writes only its own level. *)
  replace (row COL_CALL_STACK_ADDR) with (row COL_CALL_STACK_DEPTH_BEFORE) by lia.
  eapply suspended_stack_preserved; [exact Hsusp | intros i _; apply HcallStack | ].
  intros id column level Hin Hbelow. eapply context_push_elsewhere; [exact Hin | exact Hsuspend | exact HcontextRead | exact Hreplay | ].
  intros Heq. injection Heq as Heq. lia.
Qed.

(* One zero row: no SpecTec step; one more declared local of the callee is zeroed. *)
Lemma init_row : forall baseDepth stack function arguments initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final ->
  ControlState program s baseDepth (controlCursor stack (entering function arguments)) initial
    (beforeBoundary row) c0 ->
  exists cursor c1, Steps c0 c1 /\ ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth stack function arguments initial final row c0 Hprime Hloaded Hfull Hreplay
    [f [inner [Hactive [Hstate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition activeFunction suspendedCalls] in Hactive, Hstate, Hsusp, Hdepth.
  destruct Hstate as [callee [Hfn [Hlen [Hargs [Hinit [Hflags [Hmove [Hpending [Hpc ->]]]]]]]]].
  rewrite Hfn in Hactive. injection Hactive as Hsame. subst callee.
  destruct Hflags as [HmFlag HzFlag].
  cbn [beforeBoundary boundaryZeroRemaining boundaryMoveRemaining boundaryZeroActive boundaryMoveActive
    boundaryPC] in HmFlag, HzFlag, Hmove, Hpending, Hpc.
  (* With no move pending, a pending zero countdown selects a zero row. *)
  assert (Hzero : row COL_LOCAL_ZERO_ACTIVE_BEFORE = 1).
  { rewrite Hmove in HzFlag. cbn in HzFlag.
    destruct (row COL_LOCAL_ZERO_REMAINING_BEFORE =? 0) eqn:Hz; [apply Z.eqb_eq in Hz; lia | exact HzFlag]. }
  assert (Hkind : initializationKind COL_LOCAL_ZERO_ACTIVE_BEFORE) by (left; reflexivity).
  pose proof (zero_row_initializes_frame initial final row arguments _ Hprime Hfull Hzero Hreplay Hinit) as Hinit'.
  pose proof (full_initialization_pc_preserved row _ Hprime Hfull Hkind Hzero) as HpcAfter.
  destruct (full_initialization_preserves_frame row _ Hprime Hfull Hkind Hzero) as [Hfbp [Hbase Hd]].
  destruct (full_initialization_counter_carry row _ Hprime Hfull Hkind Hzero) as [_ Hcarry].
  assert (HmoveBefore : row COL_FRAME_MOVE_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive row COL_LOCAL_ZERO_ACTIVE_BEFORE COL_FRAME_MOVE_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hzero). }
  destruct (Hcarry HmoveBefore) as [HmoveAfter HdistAfter].
  assert (Habove : 2 * row COL_LOCALS_FBP_BEFORE <= row COL_STACK_WRITE0_ADDR_LO).
  { destruct (full_zero_initialization_row row Hprime Hfull Hzero) as [_ [Haddr _]].
    destruct Hinit as [Hzb _ Hlayout Hsp _ _]. cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase] in *.
    rewrite Haddr. lia. }
  pose proof (initialization_frame_effect initial final row _ Hprime Hfull Hkind Hzero Hreplay Habove) as Heffect.
  destruct (callee_active_state function f arguments final (afterBoundary row) Hfn Hlen Hargs Hinit'
    (full_initialization_flags row Hprime Hfull) ltac:(cbn [afterBoundary boundaryMoveRemaining]; lia)
    ltac:(cbn [afterBoundary boundaryPC]; rewrite HpcAfter; exact Hpc))
    as [position [Hactive' Hstate']].
  exists (controlCursor stack position),
    (plugCalls stack (stackConfig (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f))) []
      (guestBody f))).
  split.
  - apply steps_refl_wf. apply plug_calls_wf; [eapply suspended_stack_ready; exact Hsusp | ].
    eapply active_state_wf. exact Hstate'.
  - exists f, (stackConfig (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f))) [] (guestBody f)).
    split; [exact Hactive' | ]. split; [exact Hstate' | ].
    split; [ | split; [exact Hbase0 | split; [cbn [afterBoundary beforeBoundary boundaryCallDepth suspendedCalls] in *;
      lia | reflexivity]]].
    cbn [suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth].
    cbn [beforeBoundary boundaryLocalsBase boundaryCallDepth] in Hsusp.
    rewrite Hfbp, Hd. eapply frame_effect_suspended; [exact Heffect | exact Hsusp].
Qed.

(* At the end of a function's own code the row is its END, which leaves the frame. *)
Lemma function_end_selected : forall view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) = [] -> positionOffset (viewPosition view) = length (guestBody f) ->
  row COL_SEL_END = 1 /\ row COL_PC_EDGE_KIND = 1.
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hlabels Hend.
  destruct Hstate as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' _]. rewrite Hf in Hf'. injection Hf' as <-.
  rewrite Hlabels, Hend in Hpc. cbn [pathOffset] in Hpc. rewrite code_offset_end in Hpc.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))) in Hpc.
  destruct (encoded_function_end _ _ _ _ Hloaded Hf) as [Hop Hedge].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as Hcode.
  split.
  - apply (full_opcode_selects row COL_SEL_END CODE_END Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact Hcode].
  - pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
    rewrite Hpc, Hedge in Hkind. injection Hkind as Hkind. lia.
Qed.

(* A popping function END slides its results down over the frame's locals. *)
Lemma function_end_slide : forall view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) = [] -> positionOffset (viewPosition view) = length (guestBody f) ->
  row COL_SEL_END = 1 -> row COL_CALL_STACK_POP_PRESENT = 1 ->
  row COL_FRAME_MOVE_DISTANCE_AFTER = Z.of_nat (frameLocals f) /\
  row COL_FRAME_MOVE_REMAINING_AFTER =
    (if Z.of_nat (frameLocals f) =? 0 then 0 else Z.of_nat (guestResults f)).
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hreplay Hstate Hf Hlabels Hend HselEnd Hpop.
  destruct Hstate as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' _]. rewrite Hf in Hf'. injection Hf' as <-.
  rewrite Hlabels, Hend in Hpc. cbn [pathOffset] in Hpc. rewrite code_offset_end in Hpc.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))) in Hpc.
  assert (Hslide : row COL_FRAME_SLIDE_ACTIVE = 1).
  { rewrite (full_frame_slide_gate row Hprime Hfull), Hpop.
    repeat rewrite (full_opcode_selector_exclusive row COL_SEL_END) by (assumption || vm_compute; tauto || discriminate).
    reflexivity. }
  destruct (circuit_row_slide_fetch _ _ _ Hreplay Hslide) as [Hremaining Hdistance].
  destruct (encoded_end_slide _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf) as [Hrom Hdrom].
  rewrite <- Hpc in Hrom, Hdrom. rewrite Hremaining in Hrom. rewrite Hdistance in Hdrom.
  rewrite (full_selected_choice_pinned row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) HselEnd
    ltac:(vm_compute; intuition discriminate)) in Hrom, Hdrom.
  unfold frameLocals. injection Hrom as ->. injection Hdrom as ->. split; reflexivity.
Qed.

(* A popping END or RETURN: SpecTec leaves the callee's FRAME_ with its results,
   and the circuit restores the caller's registers from its saved context. The
   callee's operands are `dropped ++ returned`; its slide drops its locals and
   `dropped`. The caller resumes after its CALL, inside its labels, while its new
   top operands (the results) still sit in the callee's frame, until the move rows
   copy them down. *)
Lemma return_to_caller : forall baseDepth top rest f index locals chosen dropped returned initial final row inner,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  In chosen [COL_SEL_END; COL_SEL_RETURN] -> row chosen = 1 ->
  row COL_CALL_STACK_POP_PRESENT = 1 -> row COL_CALL_STACK_PUSH_PRESENT = 0 -> row COL_OUTPUT_CAPTURED = 0 ->
  initializationIdle (beforeBoundary row) ->
  functionAt program index = Some f -> Forall (fun value => 0 <= value < i32Modulus) locals ->
  row COL_STACK_FRAME_BASE_BEFORE = row COL_LOCALS_FBP_BEFORE + Z.of_nat (frameLocals f) ->
  Forall (fun value => 0 <= value < i32Modulus) (dropped ++ returned) ->
  (forall i v, nth_error (dropped ++ returned) i = Some v ->
    initial stackId [2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat i)] = Some v) ->
  row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (dropped ++ returned)) ->
  length returned = guestResults f ->
  row COL_FRAME_MOVE_DISTANCE_AFTER = Z.of_nat (frameLocals f) + Z.of_nat (length dropped) ->
  row COL_FRAME_MOVE_REMAINING_AFTER =
    (if Z.of_nat (frameLocals f) + Z.of_nat (length dropped) =? 0 then 0 else Z.of_nat (guestResults f)) ->
  (forall caller, wf_state (mk_state s caller) ->
    Steps (callContext caller (N.of_nat (guestResults f)) inner)
      (mk_config (mk_state s caller) (List.map i32Constant returned))) ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) (top :: rest) ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (S (length rest)) ->
  exists cursor c1, Steps (plugCalls (top :: rest) inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth top rest f index locals chosen dropped returned initial final row inner Hprime Hfull Hprogram
    Hreplay Hkind Hchosen Hpop Hpush Hcapture Hidle Hf Hlb HbaseLayout Hbounds Hcells Hsp Hlength HdistRom HremRom
    Hleave Hsusp Hbase0 Hdepth.
  assert (Hin : In chosen selectorColumns) by (destruct Hkind as [<- | [<- | []]]; vm_compute; tauto).
  destruct (full_selected_no_host_switch row chosen Hprime Hfull Hin Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  destruct (full_return_context_restore row Hfull Hpop) as [HpcAfter [HfbpAfter [HbaseAfter HaddrAfter]]].
  pose proof (full_call_depth_balance row chosen Hprime Hfull Hin Hchosen) as HdepthBalance.
  rewrite Hpush, Hpop in HdepthBalance.
  destruct (full_return_stack row chosen Hprime Hfull Hkind Hchosen Hpop Hcapture)
    as [Hlo [Hhi [HlocalWrite [HgatherLo [Hgather HspEnd]]]]].
  assert (HstackMem : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves;
        [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HcontextMem : forall id column address, In (id, column) contextMemories ->
      final id address = initial id address)
    by (intros id column address Hin'; eapply context_preserved_without_push; eassumption).
  (* No zeroing is pending after the return. *)
  assert (HzeroAfter : row COL_LOCAL_ZERO_REMAINING_AFTER = 0).
  { destruct Hidle as [Hzr [_ [Hza _]]]. cbn [beforeBoundary boundaryZeroRemaining boundaryZeroActive] in Hzr, Hza.
    assert (Hone : row COL_ONE = 1) by exact (proj1 Hfull).
    pose proof (full_row_kind_exclusive row COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY
      Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram) as Hturn.
    assert (Hguest : row COL_GUEST_ENTRY_ACTIVE = 0).
    { destruct Hkind as [<- | [<- | []]]; let c := chosen_of Hchosen in selected_row c guestEntryFlagRow Hrow;
        pose proof (value_bound row COL_GUEST_ENTRY_ACTIVE); field_linear Hrow. }
    circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_TURN_BOUNDARY, -1);
      (COL_LOCAL_ZERO_ACTIVE_BEFORE, -1)],
      [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hguest, Hturn, Hza, Hzr in Hrow.
    pose proof (value_bound row COL_LOCAL_ZERO_REMAINING_AFTER). field_linear Hrow. }
  destruct Hsusp as [x [HframeC [Hcall [Hcallee [Hres [HfbpC [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]].
  destruct top as [[[gindex glabels goffset] gsegments gvalues glocals] g results callerB callerF].
  cbn [callView callerFunction callResults callerBase callerFbp viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset viewOperands] in HframeC, Hcall, Hcallee, Hres, HfbpC,
    Hret, HsavedFbp, HsavedBase, Hrest |- *.
  pose proof HframeC as [Hg [HpathC [HheightsC [HfbpC0 [HbaseC [HboundsC [_ [HllenC [HlbC HlcC]]]]]]]]].
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ HframeC) as HcellsC.
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hg, HpathC, HheightsC, HboundsC, HcellsC, HllenC, HlbC, HlcC.
  destruct (frame_ready _ _ _ _ _ _ _ _ HframeC) as [HreadyC [HvaluesC [HcodeC HstateC]]].
  unfold viewOperands in HfbpC, HboundsC, HcellsC. cbn [viewSegments viewValues] in HfbpC, HboundsC, HcellsC.
  cbn [viewPosition viewSegments viewValues viewLocals positionLabels] in HreadyC, HvaluesC, HcodeC, HstateC.
  assert (Haddress : row COL_CALL_STACK_ADDR = row COL_CALL_STACK_DEPTH_BEFORE - 1) by lia.
  pose proof (context_pop_reads initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hpop Hsuspend HcontextRead Hreplay) as HretRead.
  pose proof (context_pop_reads initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hpop Hsuspend HcontextRead Hreplay) as HfbpRead.
  pose proof (context_pop_reads initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hpop Hsuspend HcontextRead Hreplay) as HbaseRead.
  rewrite Haddress, Hret in HretRead. rewrite Haddress, HsavedFbp in HfbpRead.
  rewrite Haddress, HsavedBase in HbaseRead.
  injection HretRead as HretValue. injection HfbpRead as HfbpValue. injection HbaseRead as HbaseValue.
  apply Forall_app in Hbounds as [Hdropped Hreturned].
  set (resumed := frameView (codePosition gindex glabels (S goffset)) gsegments (gvalues ++ returned) glocals).
  exists (controlCursor rest (atCode resumed)),
    (plugCalls rest (frameConfig g resumed (stackConfig (mk_state s (activeFrame g glocals)) (gvalues ++ returned)
      (skipn (S goffset) (innerCode g (codePosition gindex glabels goffset)))))).
  split.
  { cbn [plugCalls]. unfold callerConfig.
    cbn [callView callerFunction callResults viewPosition viewValues viewLocals positionOffset].
    apply steps_in_plug_calls; [eapply suspended_stack_ready; exact Hrest | ].
    unfold frameConfig, resumed. cbn [viewPosition positionLabels viewSegments].
    apply steps_in_labels; [exact HreadyC | ].
    rewrite <- (stack_config_rejoin _ gvalues returned). apply steps_with_stack_prefix; [exact HvaluesC | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact HcodeC | ].
    rewrite Hres. exact (Hleave _ HstateC). }
  assert (Hsource : sourceHeight (programCallees program (guestModule g)) (CALL x) (length gvalues + guestParams f) =
      Some (length (gvalues ++ returned) + 0)%nat).
  { cbn [sourceHeight]. rewrite Hcallee.
    replace (Nat.leb (guestParams f) (length gvalues + guestParams f)) with true by (symmetry; apply Nat.leb_le; lia).
    rewrite length_app. f_equal. lia. }
  (* The results slide down over the callee's locals and the dropped operands;
     with neither, they are already in place and nothing moves. *)
  assert (HdistAfter : row COL_FRAME_MOVE_DISTANCE_AFTER =
      row COL_STACK_FRAME_BASE_BEFORE - row COL_LOCALS_FBP_BEFORE + Z.of_nat (length dropped)) by lia.
  assert (Hmoved : (Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER) <= length returned)%nat).
  { rewrite HremRom. destruct (_ =? 0); lia. }
  assert (Hcell : forall b n i, operandCell b n (Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER))
      (row COL_FRAME_MOVE_DISTANCE_AFTER) i =
    operandCell b n (length returned)
      (row COL_STACK_FRAME_BASE_BEFORE - row COL_LOCALS_FBP_BEFORE + Z.of_nat (length dropped)) i).
  { intros b n i. unfold operandCell. rewrite HremRom.
    destruct (Z.of_nat (frameLocals f) + Z.of_nat (length dropped) =? 0) eqn:Hzero.
    - apply Z.eqb_eq in Hzero. rewrite <- HdistAfter, HdistRom, Hzero.
      destruct (_ <? _)%nat, (_ <? _)%nat; lia.
    - replace (Z.to_nat (Z.of_nat (guestResults f))) with (length returned) by lia.
      rewrite HdistAfter. reflexivity. }
  exists g, (frameConfig g resumed (stackConfig (mk_state s (activeFrame g glocals)) (gvalues ++ returned)
    (skipn (S goffset) (innerCode g (codePosition gindex glabels goffset))))).
  split; [exact Hg | ]. split.
  - exists g. cbn [afterBoundary boundaryMoveRemaining boundaryStackBase boundaryLocalsBase boundaryMoveDistance
      boundarySP boundaryPC boundaryZeroRemaining].
    unfold resumed.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
    split.
    + split; [exact Hg | ]. split; [exact HpathC | ].
      split; [exact (path_heights_step _ _ _ _ _ _ _ _ _ HpathC Hcall HheightsC Hsource) | ].
      split; [rewrite HfbpAfter, <- HfbpValue; exact HfbpC0 | ].
      split; [rewrite HbaseAfter, HfbpAfter, <- HbaseValue, <- HfbpValue; exact HbaseC | ].
      unfold viewOperands. cbn [viewSegments viewValues].
      split; [rewrite app_assoc; apply Forall_app; split; assumption | ].
      split.
      { intros i value Hv. rewrite Hcell, HbaseAfter, <- HbaseValue, HstackMem.
        rewrite app_assoc in Hv. unfold operandCell.
        replace (length (concat gsegments ++ gvalues ++ returned) - length returned)%nat
          with (length (concat gsegments) + length gvalues)%nat by (rewrite !length_app; lia).
        destruct (Nat.lt_ge_cases i (length (concat gsegments ++ gvalues))) as [Hlt | Hge].
        - rewrite length_app in Hlt.
          replace (i <? length (concat gsegments) + length gvalues)%nat with true by (symmetry; apply Nat.ltb_lt; lia).
          rewrite Z.add_0_r. apply HcellsC. rewrite nth_error_app1 in Hv by (rewrite length_app; lia). exact Hv.
        - rewrite length_app in Hge.
          replace (i <? length (concat gsegments) + length gvalues)%nat with false by (symmetry; apply Nat.ltb_ge; lia).
          rewrite nth_error_app2 in Hv by (rewrite length_app; lia). rewrite length_app in Hv.
          assert (Hat : nth_error (dropped ++ returned)
              (length dropped + (i - (length (concat gsegments) + length gvalues))) = Some value)
            by (rewrite nth_error_app2, Nat.add_comm, Nat.add_sub by lia; exact Hv).
          replace (2 * (callerB + Z.of_nat i +
              (row COL_STACK_FRAME_BASE_BEFORE - row COL_LOCALS_FBP_BEFORE + Z.of_nat (length dropped)))) with
            (2 * (row COL_STACK_FRAME_BASE_BEFORE +
              Z.of_nat (length dropped + (i - (length (concat gsegments) + length gvalues)))))
            by (rewrite HfbpC, length_app, Nat2Z.inj_add, Nat2Z.inj_sub by lia; lia).
          exact (Hcells _ _ Hat). }
      split; [exact HllenC | ]. split; [exact HlbC | ].
      intros i v Hv. rewrite HfbpAfter, <- HfbpValue, HstackMem. exact (HlcC _ _ Hv).
    + split; [assert (goffset < length (innerCode g (codePosition gindex glabels goffset)))%nat
        by (apply nth_error_Some; rewrite Hcall; discriminate); unfold innerCode in *; cbn in *; lia | ].
      split.
      { unfold viewOperands. cbn [viewSegments viewValues].
        rewrite HspEnd, HbaseAfter, <- HbaseValue, HdistAfter, Hsp, HfbpC.
        rewrite !length_app. lia. }
      split; [rewrite HpcAfter, <- HretValue; reflexivity | ].
      split; [exact (full_initialization_flags row Hprime Hfull) | ].
      split; [exact HzeroAfter | ].
      split; [rewrite length_app; lia | reflexivity].
  - split.
    + cbn [suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth].
      rewrite HfbpAfter, <- HfbpValue, HdepthBalance.
      replace (row COL_CALL_STACK_DEPTH_BEFORE + 0 - 1) with (row COL_CALL_STACK_DEPTH_BEFORE - 1) by lia.
      eapply suspended_stack_preserved; [exact Hrest | | ].
      * intros i _. apply HstackMem.
      * intros id column address Hin' _. eapply HcontextMem. exact Hin'.
    + split; [exact Hbase0 | ]. split; [ | reflexivity].
      cbn [afterBoundary boundaryCallDepth suspendedCalls]. rewrite HdepthBalance, Hdepth, Nat2Z.inj_succ. lia.
Qed.

(* A callee's function-ending END: its operands are exactly its results. *)
Lemma return_row : forall baseDepth top rest view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) = [] -> positionOffset (viewPosition view) = length (guestBody f) ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) (top :: rest) ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (S (length rest)) ->
  exists cursor c1, Steps (plugCalls (top :: rest) inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth top rest view f initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf
    Hlabels Hend Hsusp Hbase0 Hdepth.
  destruct (function_end_selected view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf
    Hlabels Hend) as [HselEnd Hkind].
  destruct (full_return_pops_nonempty_context row COL_SEL_END Hprime Hfull ltac:(cbn; tauto) HselEnd Hkind Hhalt)
    as [Hpop [Hpush Hcapture]].
  destruct (function_end_slide view f initial final row inner Hprime Hloaded Hfull Hreplay Hstate Hf Hlabels Hend
    HselEnd Hpop) as [HdistRom HremRom].
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [_ [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [_ [Hheights [Hfbp0 [HbaseLayout [Hbounds [_ [_ [Hlb _]]]]]]]]].
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [Hresults _]]]].
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hlabels, Hend, Hheights, Hbounds, Hcells, Hlb, Hsp, Hresults |- *.
  subst labels offset. specialize (Hresults Logic.eq_refl). unfold innerCode in Hresults.
  cbn [pathCode positionLabels] in Hresults.
  cbn [pathHeights] in Hheights. rewrite firstn_all in Hheights.
  destruct (sourceHeights _ (guestBody f) 0) as [height | ] eqn:Hheight; [ | discriminate].
  rewrite (Hresults _ Hheight) in Hheights. cbn [option_map] in Hheights.
  injection Hheights as Hsegments Hlength. destruct segments; [ | discriminate].
  rewrite Nat.add_0_r in Hlength. unfold viewOperands in Hbounds, Hcells, Hsp.
  cbn [viewSegments viewValues concat app] in Hbounds, Hcells, Hsp.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
  (* SpecTec leaves the callee's frame with its operands. *)
  assert (HcalleeWf : wf_frame (activeFrame f locals))
    by (pose proof (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb) as H; inversion H; assumption).
  apply (return_to_caller baseDepth _ rest f index locals COL_SEL_END [] values initial final row);
    try assumption; try (cbn; tauto).
  all: try (cbn [length]; rewrite Z.add_0_r; assumption).
  all: try (rewrite HdistRom; cbn; lia).
  { lia. }
  intros caller Hcaller.
  pose proof (i32_function_returns s caller (activeFrame f locals) values Hcaller HcalleeWf Hbounds) as Hleave.
  unfold frameConfig, innerCode. cbn [wrapLabels viewPosition positionLabels viewSegments pathCode].
  rewrite skipn_all. unfold callContext, stackConfig in Hleave |- *. cbn [List.map] in *.
  rewrite app_nil_r. replace (guestResults f) with (length values) by lia. exact Hleave.
Qed.

(* One move row: no SpecTec step; the lowest displaced operand of the active frame
   reaches its cell. *)
Lemma move_row : forall baseDepth stack view initial final row c0,
  fieldPrime -> FullRelation.satFull row -> Replay initial (eventsForRow memories row) final ->
  row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 ->
  ControlState program s baseDepth (controlCursor stack (atCode view)) initial (beforeBoundary row) c0 ->
  exists cursor c1, Steps c0 c1 /\ ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth stack view initial final row c0 Hprime Hfull Hreplay Hmove
    [f [inner [Hactive [Hstate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition activeFunction suspendedCalls] in Hactive, Hstate, Hsusp, Hdepth.
  destruct Hstate as [f' [Hframe [Hoffset [Hsp [Hpc [[Hma Hza] [Hzr [Hmoved Hinner]]]]]]]].
  cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining boundaryZeroRemaining boundaryMoveDistance
    boundaryStackBase boundaryLocalsBase boundarySP boundaryPC boundaryCallDepth] in *.
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hlayout [Hbounds [Hcells [Hllen [Hlb Hlc]]]]]]]]].
  rewrite Hactive in Hf'. injection Hf' as Hsame. subst f'.
  assert (Hkind : initializationKind COL_FRAME_MOVE_ACTIVE_BEFORE) by (right; reflexivity).
  destruct (full_frame_move_row row Hprime Hfull Hmove) as [Hdec [HdistSame [HwriteAddr [HreadAddr [HvalueLo _]]]]].
  destruct (move_row_stack_replay initial final row Hprime Hfull Hmove Hreplay) as [Hread Hfinal].
  pose proof (full_initialization_stack row _ Hprime Hfull Hkind Hmove) as Hshape.
  pose proof (init_stack_pointer row _ Hshape) as HspAfter. cbn in HspAfter. rewrite Z.add_0_r in HspAfter.
  pose proof (full_initialization_pc_preserved row _ Hprime Hfull Hkind Hmove) as HpcAfter.
  destruct (full_initialization_preserves_frame row _ Hprime Hfull Hkind Hmove) as [Hfbp [Hbase Hd]].
  destruct (full_initialization_counter_carry row _ Hprime Hfull Hkind Hmove) as [Hcarry _].
  assert (HzeroBefore : row COL_LOCAL_ZERO_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive row COL_FRAME_MOVE_ACTIVE_BEFORE COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hmove). }
  pose proof (Hcarry HzeroBefore) as HzeroAfter.
  pose proof (proj1 (value_bound row COL_FRAME_MOVE_REMAINING_BEFORE)) as Hmr.
  pose proof (proj1 (value_bound row COL_FRAME_MOVE_DISTANCE_BEFORE)) as Hdist.
  pose proof (value_bound row COL_FRAME_MOVE_REMAINING_AFTER) as HremBound.
  set (n := length (viewOperands view)) in *.
  set (r := row COL_FRAME_MOVE_REMAINING_BEFORE) in *.
  set (d := row COL_FRAME_MOVE_DISTANCE_BEFORE) in *.
  (* The moved operand: the lowest of the displaced ones. *)
  assert (Hmoved' : (Z.to_nat r <= n)%nat) by (unfold n, viewOperands; rewrite length_app; lia).
  set (i0 := (n - Z.to_nat r)%nat).
  assert (Hi0 : (i0 < n)%nat) by (unfold i0; lia).
  destruct (nth_error (viewOperands view) i0) as [v0 | ] eqn:Hv0; [ | apply nth_error_None in Hv0; lia].
  assert (Hsource : row (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * operandCell
      (row COL_STACK_FRAME_BASE_BEFORE) n (Z.to_nat r) d i0).
  { unfold operandCell. replace (i0 <? n - Z.to_nat r)%nat with false by (symmetry; apply Nat.ltb_ge; lia).
    rewrite HreadAddr, HwriteAddr. unfold i0. lia. }
  rewrite Hsource, (Hcells _ _ Hv0) in Hread. injection Hread as HreadValue.
  assert (Hdest : row COL_STACK_WRITE0_ADDR_LO = 2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat i0))
    by (rewrite HwriteAddr; unfold i0; lia).
  (* The frame after the row: one fewer operand displaced. *)
  assert (Hframe' : FrameAt program final (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) 0
      (Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER)) d f view).
  { split; [exact Hactive | ]. split; [exact Hpath | ]. split; [exact Hheights | ]. split; [exact Hfbp0 | ].
    split; [exact Hlayout | ]. split; [exact Hbounds | ].
    split; [ | split; [exact Hllen | split; [exact Hlb | ]]].
    - intros i value Hv. fold n. rewrite Hfinal, Hdest.
      assert (Hi : (i < n)%nat) by (apply nth_error_Some; rewrite Hv; discriminate).
      destruct (Nat.eq_dec i i0) as [-> | Hne].
      + unfold operandCell. replace (i0 <? n - Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER))%nat with true
          by (symmetry; apply Nat.ltb_lt; unfold i0; lia).
        rewrite Z.add_0_r, update_other by cell_differs. rewrite update_same.
        rewrite Hv0 in Hv. injection Hv as <-. normalize_columns. congruence.
      + rewrite !update_other by (unfold operandCell; destruct (_ <? _)%nat eqn:Hb;
          [apply Nat.ltb_lt in Hb | apply Nat.ltb_ge in Hb]; unfold i0 in *; cell_differs).
        rewrite <- (Hcells _ _ Hv). f_equal. f_equal. unfold operandCell.
        destruct (Nat.lt_ge_cases i i0) as [Hlt | Hge].
        * replace (i <? n - Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER))%nat with true
            by (symmetry; apply Nat.ltb_lt; unfold i0 in *; lia).
          replace (i <? n - Z.to_nat r)%nat with true by (symmetry; apply Nat.ltb_lt; unfold i0 in *; lia). reflexivity.
        * replace (i <? n - Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER))%nat with false
            by (symmetry; apply Nat.ltb_ge; unfold i0 in *; lia).
          replace (i <? n - Z.to_nat r)%nat with false by (symmetry; apply Nat.ltb_ge; unfold i0 in *; lia).
          reflexivity.
    - intros index value Hv. rewrite Hfinal, Hdest.
      assert (Hindex : (index < length (viewLocals view))%nat) by (apply nth_error_Some; rewrite Hv; discriminate).
      rewrite !update_other by cell_differs. exact (Hlc _ _ Hv). }
  assert (Habove : 2 * row COL_LOCALS_FBP_BEFORE <= row COL_STACK_WRITE0_ADDR_LO) by (rewrite Hdest; lia).
  pose proof (initialization_frame_effect initial final row _ Hprime Hfull Hkind Hmove Hreplay Habove) as Heffect.
  exists (controlCursor stack (atCode view)), (plugCalls stack inner).
  assert (Hstate' : ActiveState program s final (afterBoundary row) (atCode view) inner).
  { exists f. cbn [afterBoundary boundaryMoveRemaining boundaryStackBase boundaryLocalsBase boundaryMoveDistance
      boundarySP boundaryPC boundaryZeroRemaining].
    rewrite Hbase, Hfbp, HdistSame. split; [exact Hframe' | ].
    split; [exact Hoffset | ]. split; [rewrite HspAfter; exact Hsp | ]. split; [rewrite HpcAfter; exact Hpc | ].
    split; [exact (full_initialization_flags row Hprime Hfull) | ].
    split; [rewrite HzeroAfter; exact Hzr | ]. split; [lia | exact Hinner]. }
  split.
  - apply steps_refl_wf. apply plug_calls_wf; [eapply suspended_stack_ready; exact Hsusp | ].
    eapply active_state_wf. exact Hstate'.
  - exists f, inner. split; [exact Hactive | ]. split; [exact Hstate' | ].
    split; [ | split; [exact Hbase0 | split; [cbn [afterBoundary boundaryCallDepth suspendedCalls]; lia | reflexivity]]].
    cbn [suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth].
    rewrite Hfbp, Hd. eapply frame_effect_suspended; [exact Heffect | exact Hsusp].
Qed.

(* A control state reads memory only through stack cells and saved call contexts. *)
Lemma control_state_memory_congr : forall baseDepth cursor initial final boundary c,
  (forall address, final stackId address = initial stackId address) ->
  (forall id column address, In (id, column) contextMemories -> final id address = initial id address) ->
  ControlState program s baseDepth cursor initial boundary c ->
  ControlState program s baseDepth cursor final boundary c.
Proof.
  intros baseDepth [stack position] initial final boundary c Hstack Hcontext
    [f [inner [Hactive [Hstate [Hsusp Hrest]]]]].
  exists f, inner. split; [exact Hactive | ]. split; [ | split; [ | exact Hrest]].
  - cbn [activePosition] in *. destruct position as [view | function arguments].
    + destruct Hstate as [g [Hframe Hfacts]]. exists g. split; [ | exact Hfacts].
      eapply frame_at_preserved; [exact Hframe | intros; apply Hstack | intros; apply Hstack].
    + destruct Hstate as [g [Hg [Hlen [Hbounds [Hinit Hfacts]]]]].
      exists g. split; [exact Hg | ]. split; [exact Hlen | ]. split; [exact Hbounds | ]. split; [ | exact Hfacts].
      destruct Hinit as [Hzb Hfbp Hbase Hsp Hargs Hzeros].
      constructor; try assumption.
      * intros index value Hv. rewrite Hstack. exact (Hargs _ _ Hv).
      * intros index Hindex. rewrite Hstack. exact (Hzeros _ Hindex).
  - eapply suspended_stack_preserved; [exact Hsusp | intros; apply Hstack | ].
    intros id column address Hin _. exact (Hcontext id column [address] Hin).
Qed.

(* A padding row is a stutter step: no SpecTec step, and the same control state. *)
Lemma padding_row : forall baseDepth cursor initial final row c0,
  fieldPrime -> FullRelation.satFull row -> Replay initial (eventsForRow memories row) final ->
  row COL_PADDING_ACTIVE = 1 ->
  ControlState program s baseDepth cursor initial (beforeBoundary row) c0 ->
  exists cursor' c1, Steps c0 c1 /\ ControlState program s baseDepth cursor' final (afterBoundary row) c1.
Proof.
  intros baseDepth cursor initial final row c0 Hprime Hfull Hreplay Hpadding Hstate.
  (* No countdown is pending: padding is a row kind of its own. *)
  assert (Hidle : row COL_LOCAL_ZERO_REMAINING_BEFORE = 0 /\ row COL_FRAME_MOVE_REMAINING_BEFORE = 0).
  { assert (Hma : row COL_FRAME_MOVE_ACTIVE_BEFORE = 0) by exact (full_row_kind_exclusive row COL_PADDING_ACTIVE
      COL_FRAME_MOVE_ACTIVE_BEFORE Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hpadding).
    assert (Hza : row COL_LOCAL_ZERO_ACTIVE_BEFORE = 0) by exact (full_row_kind_exclusive row COL_PADDING_ACTIVE
      COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hpadding).
    destruct Hstate as [f [inner [_ [Hactive _]]]].
    destruct (activePosition cursor) as [view | function arguments]; cbn in Hactive.
    - destruct Hactive as [g [_ [_ [_ [_ [[HmaFlag HzaFlag] [Hzr _]]]]]]].
      cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining boundaryZeroActive boundaryZeroRemaining]
        in HmaFlag, HzaFlag, Hzr.
      rewrite Hma in HmaFlag. split; [exact Hzr | ].
      destruct (row COL_FRAME_MOVE_REMAINING_BEFORE =? 0) eqn:H; [apply Z.eqb_eq; exact H | discriminate].
    - destruct Hactive as [g [_ [_ [_ [_ [[HmaFlag HzaFlag] [Hmr [Hzr _]]]]]]]].
      cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining boundaryZeroActive boundaryZeroRemaining]
        in HmaFlag, HzaFlag, Hmr, Hzr.
      exfalso. rewrite Hmr, Hza in HzaFlag. cbn in HzaFlag.
      destruct (row COL_LOCAL_ZERO_REMAINING_BEFORE =? 0) eqn:H; [apply Z.eqb_eq in H; lia | discriminate]. }
  destruct Hidle as [Hzr Hmr].
  destruct (full_padding_row row Hprime Hfull Hpadding Hzr Hmr)
    as [Hpc [Hsp [Hbase [Hfbp [Hdepth [Hzra [Hmra [Hzaa [Hmaa [Hzab [Hmab [Hdist [Hwrites [Hpush [Hsuspend Hread]]]]]]]]]]]]]]].
  assert (Hboundary : afterBoundary row = beforeBoundary row).
  { unfold afterBoundary, beforeBoundary. rewrite Hpc, Hsp, Hbase, Hfbp, Hdepth, Hzra, Hmra, Hzaa, Hmaa, Hdist,
      Hzr, Hmr, Hzab, Hmab. reflexivity. }
  exists cursor, c0. split; [apply steps_refl_wf; eapply control_state_wf; exact Hstate | ].
  rewrite Hboundary. eapply control_state_memory_congr; [ | | exact Hstate].
  - intros address. eapply replay_preserves; [exact Hreplay | exact (Hwrites address)].
  - intros id column address Hin. eapply context_preserved_without_push; eassumption.
Qed.

(* A return's row runs the RETURN opcode, with a return-like edge. *)
Lemma return_selected : forall view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some RETURN ->
  row COL_SEL_RETURN = 1 /\ row COL_PC_EDGE_KIND = 1.
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi.
  destruct Hstate as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' [Hpath _]]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct (encoded_path_return _ _ _ _ Hloaded Hf _ _ _ Hpath Hi) as [Hop Hedge].
  cbn [beforeBoundary boundaryPC] in Hpc. unfold pathPC in Hop, Hedge. rewrite <- Hpc in Hop, Hedge.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch. injection Hfetch as Hcode.
  split.
  - apply (full_opcode_selects row COL_SEL_RETURN CODE_RETURN Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact Hcode].
  - pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
    rewrite Hedge in Hkind. injection Hkind as Hkind. lia.
Qed.

(* A return inside any number of blocks: SpecTec carries every operand out of the
   blocks and the function's label, and its frame keeps the results; the circuit's
   slide drops the frame's locals and the operands below the results. *)
Lemma early_return_row : forall baseDepth top rest view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some RETURN ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) (top :: rest) ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (S (length rest)) ->
  exists cursor c1, Steps (plugCalls (top :: rest) inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth top rest view f initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hi
    Hsusp Hbase0 Hdepth.
  destruct (return_selected view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi)
    as [Hsel Hkind].
  destruct (full_return_pops_nonempty_context row COL_SEL_RETURN Hprime Hfull ltac:(cbn; tauto) Hsel Hkind Hhalt)
    as [Hpop [Hpush Hcapture]].
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [HbaseLayout [Hbounds [_ [Hllen [Hlb _]]]]]]]]].
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [_ Htotal].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [_ [_ [_ [Hready Hwfcode]]]]].
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset]
    in Hf, Hi, Hpath, Hheights, Hbounds, Hcells, Hlb, Hsp, Hpc, Htotal, Hfacts, Hready, Hwfcode |- *.
  (* Typing: the return's own block holds the results. *)
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts. cbn [codeFacts instrFacts] in Hfacts.
  destruct Hfacts as [[_ Hresults] _].
  cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase boundaryPC] in Hsp, Hpc, HbaseLayout.
  rewrite Nat.add_0_r in Htotal.
  (* The ROM's slide: the frame holds the results, which the slide keeps. *)
  assert (Hslide : row COL_FRAME_SLIDE_ACTIVE = 1).
  { rewrite (full_frame_slide_gate row Hprime Hfull), Hpop.
    repeat rewrite (full_opcode_selector_exclusive row COL_SEL_RETURN) by
      (assumption || vm_compute; tauto || discriminate). reflexivity. }
  destruct (circuit_row_slide_fetch _ _ _ Hreplay Hslide) as [Hremaining Hdistance].
  destruct (encoded_return_slide _ _ _ _ _ _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf Hpath Hi Hheights)
    as [Hkept Hrom].
  rewrite <- Hpc, Hremaining in Hkept. rewrite <- Hpc, Hdistance in Hrom.
  rewrite Nat.add_0_r, Htotal in Hkept, Hrom.
  rewrite (full_selected_choice_pinned row COL_SEL_RETURN Hprime Hfull ltac:(vm_compute; tauto) Hsel
    ltac:(vm_compute; intuition discriminate)) in Hkept, Hrom.
  match type of Hkept with context [Nat.leb ?r ?t] => assert (Hfits : (r <= t)%nat) end.
  { unfold viewOperands. cbn [viewSegments viewValues]. rewrite length_app. lia. }
  rewrite (proj2 (Nat.leb_le _ _) Hfits) in Hkept, Hrom. change ((0 =? 0)%Z && true) with true in Hkept, Hrom.
  cbn iota in Hkept, Hrom.
  injection Hkept as HremRom. injection Hrom as HdistRom.
  unfold viewOperands in Hfits, HremRom, HdistRom, Hbounds, Hcells, Hsp. cbn [viewSegments viewValues] in *.
  rewrite !Nat.add_0_r in Hfits, HremRom, HdistRom.
  (* The operands below the results, and the results. *)
  destruct (split_top_values (concat segments ++ values) (guestResults f) Hfits) as [dropped [returned [Hsplit Hlength]]].
  assert (Hdropped : length dropped = (length (concat segments ++ values) - guestResults f)%nat)
    by (rewrite Hsplit, length_app; lia).
  assert (HcalleeWf : wf_frame (activeFrame f locals))
    by (pose proof (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb) as H; inversion H; assumption).
  pose proof (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb) as Hz.
  apply (return_to_caller baseDepth top rest f index locals COL_SEL_RETURN dropped returned initial final row);
    try assumption; try (cbn; tauto).
  - rewrite <- Hsplit. exact Hbounds.
  - rewrite <- Hsplit. exact Hcells.
  - rewrite <- Hsplit. exact Hsp.
  - rewrite HdistRom, Hdropped. unfold frameLocals. lia.
  - rewrite HremRom, Hdropped. unfold frameLocals. reflexivity.
  - intros caller Hcaller.
    unfold frameConfig. cbn [viewPosition positionLabels viewSegments].
    rewrite (skipn_at_instruction _ _ _ Hi).
    destruct (wrap_labels_return labels (guestBody f) _ segments (mk_state s (activeFrame f locals)) values
      (skipn (S offset) (innerCode f (codePosition index labels offset))) Hpath
      ltac:(pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H)
      Hready Hz Hbounds (forall_skipn _ _ _ _ Hwfcode)) as [rest' [Hrest' Hreturn]].
    eapply spec_steps_transitive; [exact (steps_in_call_frame caller _ _ _ ltac:(inversion Hcaller; assumption) Hreturn) | ].
    rewrite Hsplit, <- Hlength. apply i32_function_returns_early; try assumption.
    rewrite <- Hsplit. exact Hbounds.
Qed.

End Calls.
