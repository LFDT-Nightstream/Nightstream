(* Rows that change the active frame: a direct call enters its callee, frame
   initialization rows set the callee's locals, and a function's END returns to
   its suspended caller. Suspended callers survive every row: their contexts lift
   SpecTec steps, and their memory lies below the active frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerRefinement CallSpec CallEntry
  CallBindings CallContext CallMemory CallReturn CallInitialization CallScope LocalBindings FrameInitialization
  Locals ProgramSpec ProgramEncoding CodeLayout BlockEncoding Fragment CodeFacts ControlDecode ControlInvariant
  FrameFacts ControlBridge FrameRows ProgramBinding Selectors.
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
Lemma frame_at_preserved : forall program memory memory' base fbp pending f view,
  FrameAt program memory base fbp pending f view ->
  (forall index, (index < length (viewOperands view))%nat ->
    memory' stackId [2 * (base + Z.of_nat index)] = memory stackId [2 * (base + Z.of_nat index)]) ->
  (forall index, (index < length (viewLocals view))%nat ->
    memory' localLoId [fbp; Z.of_nat index] = memory localLoId [fbp; Z.of_nat index]) ->
  FrameAt program memory' base fbp pending f view.
Proof.
  intros program memory memory' base fbp pending f view
    [Hf [Hpath [Hheights [Hbase [Hbounds [Hcells [Hlen [Hlb Hlc]]]]]]]] Hstack Hlocals.
  unfold FrameAt. split; [exact Hf | ]. split; [exact Hpath | ]. split; [exact Hheights | ].
  split; [exact Hbase | ]. split; [exact Hbounds | ].
  split; [intros index value Hi; rewrite Hstack; [exact (Hcells _ _ Hi) | apply nth_error_Some; rewrite Hi; discriminate] | ].
  split; [exact Hlen | ]. split; [exact Hlb | ].
  intros index value Hi. rewrite Hlocals; [exact (Hlc _ _ Hi) | apply nth_error_Some; rewrite Hi; discriminate].
Qed.

(* Callers lie below the active frame: earlier stack cells, earlier local frames,
   and return contexts saved at lower depths. *)
Lemma suspended_stack_preserved : forall program stack memory memory' callee base fbp depth,
  SuspendedStack program memory callee base fbp depth stack ->
  (forall index, index < base -> memory' stackId [2 * index] = memory stackId [2 * index]) ->
  (forall other index, other < fbp -> memory' localLoId [other; index] = memory localLoId [other; index]) ->
  (forall id column address, In (id, column) contextMemories -> address < depth ->
    memory' id [address] = memory id [address]) ->
  SuspendedStack program memory' callee base fbp depth stack.
Proof.
  intros program stack. induction stack as [ | suspended rest IH];
    intros memory memory' callee base fbp depth H Hstack Hlocals Hcontext; [exact I | ].
  destruct H as [x [Hframe [Hnth [Hcallee [Hres [Hbase [Hfbp [Hpc [HsavedFbp [HsavedBase Hrest]]]]]]]]]].
  pose proof Hframe as [_ [_ [_ [_ [_ [_ [Hlen _]]]]]]].
  cbn [SuspendedStack]. exists x.
  split.
  { eapply frame_at_preserved; [exact Hframe | | ].
    - intros index Hindex. apply Hstack. lia.
    - intros index Hindex. apply Hlocals. lia. }
  split; [exact Hnth | ]. split; [exact Hcallee | ]. split; [exact Hres | ]. split; [exact Hbase | ].
  split; [exact Hfbp | ].
  split; [rewrite (Hcontext callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE) by (cbn; auto || lia); exact Hpc | ].
  split; [rewrite (Hcontext callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE) by (cbn; auto || lia);
    exact HsavedFbp | ].
  split; [rewrite (Hcontext callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE) by (cbn; auto || lia);
    exact HsavedBase | ].
  eapply IH; [exact Hrest | | | ].
  - intros index Hindex. apply Hstack. pose proof (Zle_0_nat (length (viewOperands (callView suspended)))). lia.
  - intros other index Hother. apply Hlocals. pose proof (Zle_0_nat (frameLocals (callerFunction suspended))). lia.
  - intros id column address Hin Haddress. apply (Hcontext id column); [exact Hin | lia].
Qed.

Lemma frame_effect_suspended : forall program stack callee base fbp depth memory memory',
  FrameEffect base fbp memory memory' ->
  SuspendedStack program memory callee base fbp depth stack ->
  SuspendedStack program memory' callee base fbp depth stack.
Proof.
  intros program stack callee base fbp depth memory memory' [Hstack Hlocals Hcontext] H.
  eapply suspended_stack_preserved; [exact H | exact Hstack | | ].
  - intros other index Hother. apply Hlocals. lia.
  - intros id column address Hin _. eapply Hcontext. exact Hin.
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

Lemma frame_ready : forall memory base fbp pending g view,
  FrameAt program memory base fbp pending g view ->
  LabelsReady (guestBody g) (positionLabels (viewPosition view)) (viewSegments view) /\
  Forall (fun value => 0 <= value < i32Modulus) (viewValues view) /\
  Forall wf_instr (innerCode g (viewPosition view)) /\
  wf_state (mk_state s (activeFrame g (viewLocals view))).
Proof.
  intros memory base fbp pending g view Hframe.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [_ [Hready Hcode]]]]].
  pose proof Hframe as [Hg [_ [_ [_ [Hoperands [_ [_ [Hlocals _]]]]]]]].
  unfold viewOperands in Hoperands. apply Forall_app in Hoperands.
  split; [exact Hready | ]. split; [exact (proj2 Hoperands) | ]. split; [exact Hcode | ].
  exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hg Hlocals).
Qed.

Lemma suspended_stack_ready : forall stack memory callee base fbp depth,
  SuspendedStack program memory callee base fbp depth stack -> Forall SuspendedReady stack.
Proof.
  induction stack as [ | suspended rest IH]; intros memory callee base fbp depth H; [constructor | ].
  destruct H as [x [Hframe [_ [_ [_ [_ [_ [_ [_ [_ Hrest]]]]]]]]]].
  constructor; [ | eapply IH; exact Hrest].
  destruct (frame_ready _ _ _ _ _ _ Hframe) as [Hready [Hvalues [Hcode Hz]]].
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
  - destruct H as [f [Hframe [_ [_ [_ [_ ->]]]]]].
    destruct (frame_ready _ _ _ _ _ _ Hframe) as [Hready [Hvalues [Hcode Hz]]].
    apply wrap_labels_wf; [exact Hready | ].
    apply stack_config_wf; [exact Hz | exact Hvalues | apply forall_skipn; exact Hcode].
  - destruct H as [f [Hf [Hlen [Hargs [_ [_ [_ [_ ->]]]]]]]].
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

(* A callee being entered: still initializing, or, once both countdowns end, at
   the start of its body. *)
Lemma callee_active_state : forall function callee arguments memory boundary,
  functionAt program function = Some callee -> length arguments = guestParams callee ->
  Forall (fun value => 0 <= value < i32Modulus) arguments ->
  InitializingFrame memory boundary arguments (Z.of_nat (guestLocals callee))
    (boundaryZeroRemaining boundary) (boundaryParamRemaining boundary) ->
  initializationFlags boundary -> boundaryPC boundary = guestEntry callee -> 0 <= boundaryStackBase boundary ->
  exists position, activeFunction program position = Some callee /\
    ActiveState program s memory boundary position
      (stackConfig (mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))) [] (guestBody callee)).
Proof.
  intros function callee arguments memory boundary Hf Hlength Hbounds Hinit Hflags Hpc Hbase.
  pose proof (zero_remaining_bounds _ _ _ _ _ _ Hinit) as Hzb.
  pose proof (param_remaining_bounds _ _ _ _ _ _ Hinit) as Hpb.
  destruct (Z.eq_dec (boundaryZeroRemaining boundary + boundaryParamRemaining boundary) 0) as [Hdone | Hpending].
  2: { exists (entering function arguments). split; [exact Hf | ].
       exists callee. repeat (split; [eassumption | ]). split; [lia | ]. split; [exact Hpc | reflexivity]. }
  assert (Hz0 : boundaryZeroRemaining boundary = 0) by lia.
  assert (Hp0 : boundaryParamRemaining boundary = 0) by lia.
  rewrite Hz0, Hp0 in Hinit.
  exists (atCode (frameView (codePosition function [] 0) [] [] (arguments ++ repeat 0 (guestLocals callee)))).
  split; [exact Hf | ]. exists callee. unfold FrameAt, viewOperands.
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset concat app].
  split.
  - split; [exact Hf | ]. split; [reflexivity | ]. split; [reflexivity | ]. split; [exact Hbase | ].
    split; [constructor | ]. split; [intros [ | index] value Hv; discriminate | ].
    split; [unfold frameLocals; rewrite length_app, repeat_length; lia | ].
    split.
    { apply Forall_app. split; [exact Hbounds | ]. apply Forall_forall. intros v Hv. apply repeat_spec in Hv.
      subst v. unfold i32Modulus. lia. }
    intros index value Hv.
    destruct (Nat.lt_ge_cases index (length arguments)) as [Hin | Hout].
    + rewrite nth_error_app1 in Hv by exact Hin. apply (copied_parameters _ _ _ _ _ _ Hinit _ _ Hv). lia.
    + rewrite nth_error_app2 in Hv by exact Hout.
      assert (Hlt : (index - length arguments < guestLocals callee)%nat).
      { rewrite <- (repeat_length 0 (guestLocals callee)). apply nth_error_Some. rewrite Hv. discriminate. }
      apply nth_error_In, repeat_spec in Hv. subst value.
      apply (zeroed_locals _ _ _ _ _ _ Hinit). lia.
  - split; [lia | ]. split; [rewrite (initialization_sp _ _ _ _ _ _ Hinit); cbn; lia | ].
    split; [rewrite Hpc; cbn; lia | ].
    split; [destruct Hflags as [Hza Hpa]; rewrite Hz0 in Hza, Hpa; rewrite Hp0 in Hpa; repeat split; assumption | ].
    unfold frameConfig. cbn [wrapLabels skipn viewPosition positionLabels viewSegments innerCode pathCode].
    rewrite entered_frame_locals. reflexivity.
Qed.

(* A direct guest call from any code path: SpecTec enters the callee's frame; the
   circuit saves the caller's return context one level above the callers already
   suspended, and begins initializing the callee's frame. *)
Lemma call_row : forall baseDepth stack view f x initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (CALL x) ->
  SuspendedStack program initial f (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE)
    (row COL_CALL_STACK_DEPTH_BEFORE) stack ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (length stack) ->
  exists cursor c1, Steps (plugCalls stack inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth stack view f x initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi
    Hsusp Hbase0 Hdepth.
  destruct Hstate as [f' [Hframe [_ [Hsp [Hpc [_ ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [HbaseNonneg [Hbounds [Hcells [Hllen [Hlb Hlc]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hi, Hpath, Hheights, Hbounds, Hcells, Hllen, Hlb, Hlc, Hsp, Hpc |- *.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup.
  destruct Hfacts as [HwfCall Hrest]. rewrite Nat.add_0_r in Hrest. cbn [sourceHeight] in Hrest.
  destruct (programCallees program (guestModule f) (x :> N)) as [callee | ] eqn:Hcallee; [ | contradiction].
  destruct (Nat.leb (guestParams callee) (length values)) eqn:Hle; [ | contradiction]. apply Nat.leb_le in Hle.
  assert (Hcounts : (guestParams callee < 256)%nat /\ (guestResults callee < 256)%nat).
  { inversion Hsup as [ | ? ? Hhead _]; subst. inversion Hhead; subst.
    match goal with Hg : _ (x :> N) = Some ?g |- _ => assert (g = callee) as -> by congruence end.
    split; assumption. }
  destruct Hcounts as [Hp Hr].
  assert (Hidx : wf_uN 32%N x) by (inversion HwfCall; assumption).
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  destruct (encoded_path_call program initial index f Hloaded Hf labels code offset s (List.map i32Value locals) x
    callee Hpath Hi Hstore ltac:(rewrite length_map; exact Hllen) Hcallee) as [Himage HcallerImage].
  unfold pathPC in Himage, HcallerImage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage, HcallerImage.
  destruct (split_top_values values (guestParams callee) Hle) as [prefix [arguments [-> Hlen]]].
  rewrite <- Hlen in Himage.
  pose proof (frame_stack_matches _ _ _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack. rewrite app_assoc in Hstack.
  destruct (direct_call_entry_bound initial final row _ (row COL_PC_BEFORE + 1) z x (guestAddress callee)
    (guestModule callee) (guestTypeIndex callee) arguments (guestResults callee) (guestLocals callee)
    (guestBody callee) (guestEntry callee) Hprime Hfull Hprogram Logic.eq_refl Hreplay Himage ltac:(lia) Hr)
    as [HcallSel [Htarget [HparamCount [HzSeed [HpSeed Hentry]]]]].
  pose proof (call_starts_frame_initialization initial final row (concat segments ++ prefix) arguments
    (guestLocals callee) Hprime Hfull HcallSel Htarget HparamCount HzSeed HpSeed Hreplay Hstack) as Hinit.
  destruct (full_direct_guest_call row Hprime Hfull HcallSel Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_call_frame_entry row Hprime Hfull Hguest Hpush Hpop) as [HfbpAfter [HbaseAfter HdepthAfter]].
  destruct (full_direct_guest_call_stack row Hprime Hfull HcallSel Htarget) as [Hlo [Hhi HspAfter]].
  destruct (full_call_context_push row Hfull Hpush) as [Haddr [HsavedFbp [HsavedBase Hchoice]]].
  assert (HcallStack : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HcallLocals : forall address, final localLoId address = initial localLoId address).
  { eapply no_local_writes_preserve; [exact Hreplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact Hfull | left; reflexivity | exact HcallSel]. }
  assert (HcallerCount : row COL_CURRENT_FUNCTION_NUM_LOCALS = Z.of_nat (frameLocals f)).
  { destruct HcallerImage as [reference [Hreference Hcount]].
    pose proof (current_function_read initial final row Hreplay
      (full_frame_lookup_active row COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; tauto) Hprogram)) as HrefRead.
    rewrite Hreference in HrefRead. injection HrefRead as HrefValue.
    pose proof (program_row_local_count_read initial final row Hreplay Hprogram) as HcountRead.
    rewrite <- HrefValue, Hcount in HcountRead. cbn [stateLocals LOCALS] in HcountRead.
    rewrite length_map, Hllen in HcountRead. congruence. }
  pose proof (context_push_records initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedPcMem.
  pose proof (context_push_records initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedFbpMem.
  pose proof (context_push_records initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedBaseMem.
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
    HargBounds Hinit (full_initialization_flags row Hprime Hfull) Hentry (proj1 (value_bound row _)))
    as [position [Hactive Hstate]].
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
  split; [exact Hactive | ]. split; [exact Hstate | ].
  split; [ | split; [exact Hbase0 | split; [cbn [suspendedCalls length afterBoundary boundaryCallDepth]; lia |
    reflexivity]]].
  cbn [SuspendedStack suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth
    callView callerFunction callResults callerBase callerFbp viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset].
  exists x. split.
  { unfold FrameAt, viewOperands.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
    split; [exact Hf | ]. split; [exact Hpath | ].
    split; [rewrite Hheights, length_app, Hlen, Nat.add_0_r; reflexivity | ].
    split; [exact HbaseNonneg | ].
    split; [apply Forall_app; split; assumption | ].
    split.
    { intros i value Hv. rewrite HcallStack. apply Hcells.
      unfold viewOperands; cbn [viewSegments viewValues].
      rewrite app_assoc, nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate]. }
    split; [exact Hllen | ]. split; [exact Hlb | ].
    intros i value Hv. rewrite HcallLocals. exact (Hlc _ _ Hv). }
  split; [exact Hi | ]. split; [exact Hcallee | ]. split; [reflexivity | ].
  split.
  { unfold viewOperands in Hsp |- *. cbn [beforeBoundary boundarySP boundaryStackBase viewSegments viewValues] in Hsp |- *.
    rewrite !length_app in Hsp; rewrite ?length_app.
    rewrite HbaseAfter, HspAfter, HparamCount. lia. }
  split; [rewrite HfbpAfter, HcallerCount; reflexivity | ].
  replace (row COL_CALL_STACK_DEPTH_AFTER - 1) with (row COL_CALL_STACK_ADDR) by lia.
  split.
  { rewrite HsavedPcMem, <- HreturnValue, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi).
    change (instructionSize (CALL x)) with 1%nat. f_equal. lia. }
  split; [rewrite HsavedFbpMem, HsavedFbp; reflexivity | ].
  split; [rewrite HsavedBaseMem, HsavedBase; reflexivity | ].
  (* The callers already suspended keep their memory: the push writes only its own level. *)
  replace (row COL_CALL_STACK_ADDR) with (row COL_CALL_STACK_DEPTH_BEFORE) by lia.
  eapply suspended_stack_preserved; [exact Hsusp | intros i _; apply HcallStack | intros other i _; apply HcallLocals | ].
  intros id column level Hin Hbelow. eapply context_push_elsewhere; [exact Hin | exact Hreplay | ].
  intros Heq. injection Heq as Heq. lia.
Qed.

(* One frame-initialization row: no SpecTec step; one more callee local is set. *)
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
  destruct Hstate as [callee [Hfn [Hlen [Hargs [Hinit [Hflags [Hpending [Hpc ->]]]]]]]].
  rewrite Hfn in Hactive. injection Hactive as Hsame. subst callee.
  destruct Hflags as [HzFlag HpFlag].
  cbn [beforeBoundary boundaryZeroRemaining boundaryParamRemaining boundaryZeroActive boundaryParamActive
    boundaryPC] in HzFlag, HpFlag, Hpending, Hpc.
  assert (Hkind : row COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/
      (row COL_PARAM_INIT_ACTIVE_BEFORE = 1 /\ row COL_LOCAL_ZERO_REMAINING_BEFORE = 0)).
  { case_eq (row COL_LOCAL_ZERO_REMAINING_BEFORE =? 0); intro Hz; rewrite Hz in HzFlag, HpFlag.
    - apply Z.eqb_eq in Hz. right. split; [ | exact Hz].
      case_eq (row COL_PARAM_INIT_REMAINING_BEFORE =? 0); intro Hp; rewrite Hp in HpFlag; [ | exact HpFlag].
      apply Z.eqb_eq in Hp. lia.
    - left. exact HzFlag. }
  destruct (encoded_guest_image _ _ _ _ Hloaded Hfn) as [_ [Hcount [_ Href]]].
  assert (Hstep : exists chosen, initializationKind chosen /\ row chosen = 1 /\
      InitializingFrame final (afterBoundary row) arguments (Z.of_nat (guestLocals f))
        (row COL_LOCAL_ZERO_REMAINING_AFTER) (row COL_PARAM_INIT_REMAINING_AFTER) /\
      row COL_PC_AFTER = row COL_PC_BEFORE).
  { destruct Hkind as [Hzero | [Hparam HzeroBefore]].
    - exists COL_LOCAL_ZERO_ACTIVE_BEFORE. split; [left; reflexivity | ]. split; [exact Hzero | ].
      split; [ | exact (full_zero_pc_preserved row Hprime Hfull Hzero)].
      eapply zero_row_initializes_frame; [exact Hprime | exact Hfull | exact Hzero | exact Hreplay | | exact Hinit].
      rewrite (zero_row_uses_loaded_local_count initial final row (guestEntry f) _ _ Hprime Hfull Hzero Hreplay
        Hpc Href Hcount), Nat2Z.inj_add, Hlen. reflexivity.
    - exists COL_PARAM_INIT_ACTIVE_BEFORE. split; [right; reflexivity | ]. split; [exact Hparam | ].
      split; [ | exact (full_parameter_pc_preserved row Hfull Hparam)].
      exact (parameter_row_after_zeroing initial final row arguments _ Hprime Hfull Hparam HzeroBefore Hreplay Hinit). }
  destruct Hstep as [chosen [Hchoice [Hchosen [Hinit' HpcAfter]]]].
  destruct (full_initialization_preserves_frame row chosen Hprime Hfull Hchoice Hchosen) as [Hfbp [Hbase Hd]].
  pose proof (initialization_frame_effect initial final row chosen Hprime Hfull Hchoice Hchosen Hreplay) as Heffect.
  destruct (callee_active_state function f arguments final (afterBoundary row) Hfn Hlen Hargs Hinit'
    (full_initialization_flags row Hprime Hfull)
    ltac:(cbn [afterBoundary boundaryPC]; rewrite HpcAfter; exact Hpc) (proj1 (value_bound row _)))
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
    cbn [suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth].
    cbn [beforeBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth] in Hsusp.
    rewrite Hfbp, Hbase, Hd. eapply frame_effect_suspended; [exact Heffect | exact Hsusp].
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

(* A callee's function-ending END: SpecTec leaves the LABEL_ and FRAME_ with the
   results, and the circuit restores the caller's registers from its saved context.
   The caller resumes after its CALL, inside its labels. *)
Lemma return_row : forall baseDepth top rest view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) = [] -> positionOffset (viewPosition view) = length (guestBody f) ->
  SuspendedStack program initial f (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE)
    (row COL_CALL_STACK_DEPTH_BEFORE) (top :: rest) ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (S (length rest)) ->
  exists cursor c1, Steps (plugCalls (top :: rest) inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth top rest view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf
    Hlabels Hend Hsusp Hbase0 Hdepth.
  destruct (function_end_selected view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf
    Hlabels Hend) as [HselEnd Hkind].
  destruct Hstate as [f' [Hframe [_ [Hsp [_ [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [_ [Hheights [_ [Hbounds [Hcells [_ [Hlb _]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [Hresults _]]]].
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hlabels, Hend, Hheights, Hbounds, Hcells, Hlb, Hsp, Hresults |- *.
  subst labels offset. specialize (Hresults Logic.eq_refl). unfold innerCode in Hresults.
  cbn [pathCode positionLabels] in Hresults.
  cbn [pathHeights] in Hheights. rewrite firstn_all, Hresults in Hheights. cbn [option_map] in Hheights.
  injection Hheights as Hsegments Hlength. destruct segments; [ | discriminate].
  rewrite Nat.add_0_r in Hlength. unfold viewOperands in Hbounds, Hcells, Hsp.
  cbn [viewSegments viewValues concat app] in Hbounds, Hcells, Hsp.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
  (* The END row pops the context saved one level down. *)
  destruct (full_end_pops_nonempty_context row Hprime Hfull HselEnd Hkind ltac:(lia)) as [Hpop [Hpush Hcapture]].
  destruct (full_return_context_restore row Hfull Hpop) as [HpcAfter [HfbpAfter [HbaseAfter HaddrAfter]]].
  pose proof (full_call_depth_balance row Hprime Hfull) as HdepthBalance.
  rewrite Hpush, Hpop in HdepthBalance.
  destruct (full_end_stack row Hprime Hfull HselEnd Hcapture) as [Hlo [Hhi HspEnd]].
  assert (HstackMem : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HlocalMem : forall address, final localLoId address = initial localLoId address).
  { eapply no_local_writes_preserve; [exact Hreplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact Hfull | right; reflexivity | exact HselEnd]. }
  assert (HcontextMem : forall id column address, In (id, column) contextMemories ->
      final id address = initial id address)
    by (intros id column address Hin; eapply context_preserved_without_push; eassumption).
  destruct Hsusp as [x [HframeC [Hcall [Hcallee [Hres [HbaseC [HfbpC [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]]].
  destruct top as [[[gindex glabels goffset] gsegments gvalues glocals] g results callerB callerF].
  cbn [callView callerFunction callResults callerBase callerFbp viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset viewOperands] in HframeC, Hcall, Hcallee, Hres, HbaseC, HfbpC,
    Hret, HsavedFbp, HsavedBase, Hrest |- *.
  pose proof HframeC as [Hg [HpathC [HheightsC [HbaseC0 [HboundsC [HcellsC [HllenC [HlbC HlcC]]]]]]]].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hg, HpathC, HheightsC, HboundsC, HcellsC, HllenC, HlbC, HlcC.
  destruct (frame_ready _ _ _ _ _ _ HframeC) as [HreadyC [HvaluesC [HcodeC HstateC]]].
  unfold viewOperands in HbaseC, HboundsC, HcellsC. cbn [viewSegments viewValues] in HbaseC, HboundsC, HcellsC.
  cbn [viewPosition viewSegments viewValues viewLocals positionLabels] in HreadyC, HvaluesC, HcodeC, HstateC.
  assert (Haddress : row COL_CALL_STACK_ADDR = row COL_CALL_STACK_DEPTH_BEFORE - 1) by lia.
  pose proof (context_pop_reads initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hpop Hreplay) as HretRead.
  pose proof (context_pop_reads initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hpop Hreplay) as HfbpRead.
  pose proof (context_pop_reads initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hpop Hreplay) as HbaseRead.
  rewrite Haddress, Hret in HretRead. rewrite Haddress, HsavedFbp in HfbpRead.
  rewrite Haddress, HsavedBase in HbaseRead.
  injection HretRead as HretValue. injection HfbpRead as HfbpValue. injection HbaseRead as HbaseValue.
  (* SpecTec leaves the callee's frame. *)
  assert (HcalleeWf : wf_frame (activeFrame f locals))
    by (pose proof (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb) as H; inversion H; assumption).
  pose proof (i32_function_returns s (activeFrame g glocals) (activeFrame f locals) values HstateC HcalleeWf Hbounds)
    as Hleave.
  assert (Hinner : callContext (activeFrame g glocals) (N.of_nat results)
      (stackConfig (mk_state s (activeFrame f locals)) values []) =
    mk_config (mk_state s (activeFrame g glocals)) [FRAME_ (N.of_nat (length values)) (activeFrame f locals)
      [LABEL_ (N.of_nat (length values)) [] (List.map i32Constant values)]]).
  { unfold callContext, stackConfig. cbn [List.map]. rewrite app_nil_r, Hres, Hlength. reflexivity. }
  assert (Hdone : frameConfig f (frameView (codePosition index [] (length (guestBody f))) [] values locals)
      (stackConfig (mk_state s (activeFrame f locals)) values
        (skipn (length (guestBody f)) (innerCode f (codePosition index [] (length (guestBody f)))))) =
    stackConfig (mk_state s (activeFrame f locals)) values []).
  { unfold frameConfig, innerCode. cbn [viewPosition positionLabels viewSegments wrapLabels pathCode].
    rewrite skipn_all. reflexivity. }
  set (resumed := frameView (codePosition gindex glabels (S goffset)) gsegments (gvalues ++ values) glocals).
  exists (controlCursor rest (atCode resumed)),
    (plugCalls rest (frameConfig g resumed (stackConfig (mk_state s (activeFrame g glocals)) (gvalues ++ values)
      (skipn (S goffset) (innerCode g (codePosition gindex glabels goffset)))))).
  split.
  { rewrite Hdone. cbn [plugCalls]. unfold callerConfig.
    cbn [callView callerFunction callResults viewPosition viewValues viewLocals positionOffset].
    rewrite Hinner. apply steps_in_plug_calls; [eapply suspended_stack_ready; exact Hrest | ].
    unfold frameConfig, resumed. cbn [viewPosition positionLabels viewSegments].
    apply steps_in_labels; [exact HreadyC | ].
    rewrite <- (stack_config_rejoin _ gvalues values). apply steps_with_stack_prefix; [exact HvaluesC | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact HcodeC | exact Hleave]. }
  assert (Hsource : sourceHeight (programCallees program (guestModule g)) (CALL x) (length gvalues + guestParams f) =
      Some (length (gvalues ++ values) + 0)%nat).
  { cbn [sourceHeight]. rewrite Hcallee.
    replace (Nat.leb (guestParams f) (length gvalues + guestParams f)) with true by (symmetry; apply Nat.leb_le; lia).
    rewrite length_app. f_equal. lia. }
  exists g, (frameConfig g resumed (stackConfig (mk_state s (activeFrame g glocals)) (gvalues ++ values)
    (skipn (S goffset) (innerCode g (codePosition gindex glabels goffset))))).
  split; [exact Hg | ]. split.
  - exists g. unfold FrameAt, resumed, viewOperands.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
    split.
    + split; [exact Hg | ]. split; [exact HpathC | ].
      split; [exact (path_heights_step _ _ _ _ _ _ _ _ _ HpathC Hcall HheightsC Hsource) | ].
      split; [cbn [afterBoundary boundaryStackBase]; rewrite HbaseAfter, <- HbaseValue; exact HbaseC0 | ].
      split; [rewrite app_assoc; apply Forall_app; split; assumption | ].
      split.
      { intros i value Hv. cbn [afterBoundary boundaryStackBase]. rewrite HbaseAfter, <- HbaseValue, HstackMem.
        rewrite app_assoc in Hv.
        destruct (Nat.lt_ge_cases i (length (concat gsegments ++ gvalues))) as [Hin | Hout].
        - apply HcellsC. rewrite nth_error_app1 in Hv by exact Hin. exact Hv.
        - rewrite nth_error_app2 in Hv by exact Hout.
          replace (2 * (callerB + Z.of_nat i)) with
            (2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (i - length (concat gsegments ++ gvalues))))
            by (rewrite HbaseC, Nat2Z.inj_sub by exact Hout; lia).
          exact (Hcells _ _ Hv). }
      split; [exact HllenC | ]. split; [exact HlbC | ].
      intros i v Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite HfbpAfter, <- HfbpValue, HlocalMem.
      exact (HlcC _ _ Hv).
    + split; [assert (goffset < length (innerCode g (codePosition gindex glabels goffset)))%nat
        by (apply nth_error_Some; rewrite Hcall; discriminate); unfold innerCode in *; cbn in *; lia | ].
      split.
      { cbn [afterBoundary boundarySP boundaryStackBase]. rewrite HspEnd, HbaseAfter, <- HbaseValue, Hsp, HbaseC.
        rewrite !length_app. lia. }
      split; [cbn [afterBoundary boundaryPC]; rewrite HpcAfter, <- HretValue; reflexivity | ].
      split; [apply (program_row_keeps_idle row COL_SEL_END); [exact Hprime | exact Hfull | vm_compute; tauto |
        exact HselEnd | not_call_selector | exact Hidle] | reflexivity].
  - split.
    + cbn [suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth].
      rewrite HbaseAfter, <- HbaseValue, HfbpAfter, <- HfbpValue, HdepthBalance.
      replace (row COL_CALL_STACK_DEPTH_BEFORE + 0 - 1) with (row COL_CALL_STACK_DEPTH_BEFORE - 1) by lia.
      eapply suspended_stack_preserved; [exact Hrest | | | ].
      * intros i _. apply HstackMem.
      * intros other i _. apply HlocalMem.
      * intros id column address Hin _. eapply HcontextMem. exact Hin.
    + split; [exact Hbase0 | ]. split; [ | reflexivity].
      cbn [afterBoundary boundaryCallDepth suspendedCalls]. rewrite HdepthBalance, Hdepth, Nat2Z.inj_succ. lia.
Qed.

End Calls.
