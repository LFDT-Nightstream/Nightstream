(* Suspended callers survive the active frame's rows: their contexts lift SpecTec
   steps, and their memory lies below the active frame. Ordinary instructions advance
   by one row in any frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers R1CS Columns Range FullRelation MemoryTrace MemoryCatalog TraceRefinement
  StraightLineSpec StackSimulation StraightLineInstructions SpecTecCalls CallerSpec CallerRefinement
  CallContext CallScope CallBody ProgramSpec ProgramEncoding ProgramValidation SourceInstructions
  SourceValidation CodeLayout ControlDecode ControlInvariant ControlState SpecTecBranches FrameInitialization Locals
  CallEntry CallBindings CallMemory CallReturn LocalBindings CallInitialization FunctionValidation CallSpec Fragment FragmentViews
  ProgramBinding Selectors CallerValidation OpcodeCatalog.
Import ListNotations.
Open Scope Z_scope.

Definition suspendedReady (suspended : SuspendedCall) : Prop :=
  wf_frame (callerFrame suspended) /\
  Forall (fun value => 0 <= value < i32Modulus) (callerValues suspended) /\
  Forall wf_instr (callContinuation suspended).

Lemma suspended_stack_ready : forall program stack callee base fbp depth s memory,
  SuspendedStack program callee base fbp depth s memory stack -> Forall suspendedReady stack.
Proof.
  intros program stack. induction stack as [ | suspended rest IH]; intros callee base fbp depth s memory H;
    [constructor | ].
  destruct H as [g [x [_ [_ [_ [_ [_ [_ [HwfFrame [Hvalid [Hbounds [_ [_ [_ [_ [_ [_ [_ [_ Hrest]]]]]]]]]]]]]]]]]]].
  constructor; [ | eapply IH; exact Hrest].
  split; [exact HwfFrame | ]. split; [exact Hbounds | eapply validated_instructions_wf; exact Hvalid].
Qed.

Lemma plug_calls_wf : forall stack c,
  Forall suspendedReady stack -> wf_config c -> wf_config (plugCalls stack c).
Proof.
  induction stack as [ | suspended rest IH]; intros c Hready Hwf; [exact Hwf | ].
  inversion Hready as [ | ? ? [Hframe [Hvalues Hcode]] Hrest]; subst.
  cbn [plugCalls]. apply IH; [exact Hrest | ].
  apply stack_prefix_wf; [exact Hvalues | ]. apply instruction_suffix_wf; [exact Hcode | ].
  apply call_context_wf; assumption.
Qed.

Lemma steps_in_plug_calls : forall stack before after,
  Forall suspendedReady stack -> Steps before after ->
  Steps (plugCalls stack before) (plugCalls stack after).
Proof.
  induction stack as [ | suspended rest IH]; intros before after Hready Hsteps; [exact Hsteps | ].
  inversion Hready as [ | ? ? [Hframe [Hvalues Hcode]] Hrest]; subst.
  cbn [plugCalls]. apply IH; [exact Hrest | ].
  apply steps_with_stack_prefix; [exact Hvalues | ]. apply steps_with_instruction_suffix; [exact Hcode | ].
  apply steps_in_call_frame; assumption.
Qed.

Lemma control_state_wf : forall program baseDepth cursor z memory boundary c,
  ControlState program baseDepth cursor z memory boundary c -> wf_config c.
Proof.
  intros program baseDepth cursor z memory boundary c [f [inner [_ [Hactive [Hstack [_ [_ ->]]]]]]].
  apply plug_calls_wf; [eapply suspended_stack_ready; exact Hstack | eapply active_state_wf; exact Hactive].
Qed.

(* Callers lie below the active frame: earlier stack cells, earlier local frames,
   and return contexts saved at lower depths. *)
Lemma suspended_stack_preserved : forall program stack callee base fbp depth s memory memory',
  SuspendedStack program callee base fbp depth s memory stack ->
  (forall index, index < base -> memory' stackId [2 * index] = memory stackId [2 * index]) ->
  (forall other index, other < fbp -> memory' localLoId [other; index] = memory localLoId [other; index]) ->
  (forall id column address, In (id, column) contextMemories -> address < depth ->
    memory' id [address] = memory id [address]) ->
  SuspendedStack program callee base fbp depth s memory' stack.
Proof.
  intros program stack. induction stack as [ | suspended rest IH];
    intros callee base fbp depth s memory memory' H Hstack Hlocals Hcontext; [exact I | ].
  destruct H as [g [x [Hg [Hcall [Hcont [Hcallee [Hresults [Hframe [HwfFrame [Hvalid [Hbounds
    [Hbase0 [Hbase [Hcells [Hfbp [Hlocal [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]]]]]]]]]]]].
  exists g, x. repeat (split; [eassumption | ]).
  split.
  - intros index value Hlookup. rewrite Hstack; [apply Hcells; exact Hlookup | ].
    assert (index < length (callerValues suspended))%nat by (apply nth_error_Some; rewrite Hlookup; discriminate). lia.
  - split; [exact Hfbp | ]. split.
    + intros index v Hlookup. destruct (Hlocal index v Hlookup) as [value [Hv [Hb Hcell]]].
      exists value. split; [exact Hv | ]. split; [exact Hb | ]. rewrite Hlocals; [exact Hcell | ].
      assert (index < length (LOCALS (callerFrame suspended)))%nat by
        (change (index < length (stateLocals (mk_state s (callerFrame suspended))))%nat;
         apply nth_error_Some; rewrite Hlookup; discriminate). lia.
    + split; [rewrite (Hcontext callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE) by (cbn; auto || lia); exact Hret | ].
      split; [rewrite (Hcontext callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE) by (cbn; auto || lia); exact HsavedFbp | ].
      split; [rewrite (Hcontext callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE) by (cbn; auto || lia); exact HsavedBase | ].
      eapply IH; [exact Hrest | | | ].
      * intros index Hindex. apply Hstack. pose proof (Zle_0_nat (length (callerValues suspended))). lia.
      * intros other index Hother. apply Hlocals. pose proof (Zle_0_nat (length (LOCALS (callerFrame suspended)))). lia.
      * intros id column address Hin Haddress. apply (Hcontext id column); [exact Hin | lia].
Qed.

Lemma frame_effect_suspended : forall program stack callee base fbp depth s memory memory',
  FrameEffect base fbp memory memory' ->
  SuspendedStack program callee base fbp depth s memory stack ->
  SuspendedStack program callee base fbp depth s memory' stack.
Proof.
  intros program stack callee base fbp depth s memory memory' [Hstack Hlocals Hcontext] H.
  eapply suspended_stack_preserved; [exact H | exact Hstack | | ].
  - intros other index Hother. apply Hlocals. lia.
  - intros id column address Hin _. eapply Hcontext. exact Hin.
Qed.

Lemma ordinary_source_facts : forall callees z height i,
  (exists op, encodedOpcode i = Some op) -> sourceInstructionFacts callees z height i ->
  wf_instr i /\ programTyped (length (stateLocals z)) height [i] /\
  exists next, instructionStackHeight i height = Some next /\ sourceHeight callees i height = Some next.
Proof.
  intros callees z height i [op Hop] Hfacts.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]].
  all: destruct Hfacts as [_ [Hwf [Htyped Hnext]]].
  all: split; [exact Hwf | split; [exact Htyped | ]].
  all: match type of Hnext with match ?height with _ => _ end =>
         destruct height as [next | ] eqn:Hheight; [ | contradiction] end.
  all: exists next; split; [reflexivity | exact Hheight].
Qed.

(* One row of an ordinary instruction in the active function, whichever frame it is. *)
Lemma source_ordinary_row : forall program index offset i rest z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> (exists op, encodedOpcode i = Some op) ->
  SourceState program (sourcePosition index offset) z initial (beforeBoundary row) (i :: rest) c0 ->
  initializationIdle (beforeBoundary row) ->
  exists z' c1, Steps c0 c1 /\
    SourceState program (sourcePosition index (S offset)) z' final (afterBoundary row) rest c1 /\
    initializationIdle (afterBoundary row) /\
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
    frameRegistersPreserved row /\ fun_store z' = fun_store z.
Proof.
  intros program index offset i rest z initial final row c0 Hprime Hloaded Hfull Hprogram Hreplay Hordinary Hsource
    Hidle.
  unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource.
  destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
  destruct Hsource as [Hframe [Hcode [Hvalidation [Hrep Hpc]]]].
  pose proof (validated_code_facts _ _ _ _ _ Hframe (represented_state_wf _ _ _ _ _ Hrep) Hvalidation) as Hfacts.
  pose proof (source_facts_wf _ _ _ _ Hfacts) as Hwf.
  destruct Hfacts as [Hif Hrest].
  destruct (ordinary_source_facts _ _ _ _ Hordinary Hif) as [Hwfi [Htyped [next [Hnext Hsource]]]].
  rewrite Hsource in Hrest.
  assert (Hlookup : nth_error (guestBody f) offset = Some i).
  { pose proof (f_equal (fun xs => nth_error xs 0) (proj2 Hcode)) as H.
    cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
  pose proof (encoded_instruction_image _ _ _ _ _ _ Hloaded Hf Hlookup Hordinary) as Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  pose proof (fetched_ordinary_keeps_idle _ _ _ _ Hprime Hfull Hprogram Hreplay (proj1 Himage) Hordinary Hidle)
    as HidleAfter.
  destruct Hrep as [values [Hstack [-> [HconfigWf Hlocals]]]].
  assert (Hz : wf_state z) by (inversion HconfigWf; assumption).
  pose proof (proj1 (proj2 (proj2 Hstack))) as Hbounds.
  assert (Hsmall : programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) [i] /\
      represents z initial (beforeBoundary row) [i] (stackConfig z values [i])).
  { split; [exact Htyped | ]. exists values. split; [exact Hstack | ]. split; [reflexivity | ].
    split; [apply stack_config_wf; [exact Hz | exact Hbounds | constructor; [exact Hwfi | constructor]] | exact Hlocals]. }
  destruct (frame_row_refinement i [] z initial final row _ Hprime Hfull Hprogram Hreplay Himage Hsmall)
    as [z' [c1 [Hstep [[_ Hrep'] [HpcNext [Heffect [Hregs [Hdelta [Hstore [Haddresses Hlength]]]]]]]]]].
  pose proof (represented_state_wf _ _ _ _ _ Hrep') as Hz'.
  destruct Hrep' as [outputs [Hstack' [-> [Hwf' Hlocals']]]].
  assert (Hsame : sameCaller z z') by (split; [exact Hstore | split; assumption]).
  assert (Hafter : next = stackHeight (afterBoundary row)).
  { pose proof (instruction_stack_height_delta _ _ _ Hnext) as Hd. rewrite Hd in Hdelta. injection Hdelta as Hdelta.
    destruct Hregs as [_ [Hbase _]].
    pose proof (stack_matches_height _ _ _ Hstack) as Hh. pose proof (stack_matches_height _ _ _ Hstack') as Hh'.
    pose proof (proj1 (proj2 Hstack)) as Hsp. pose proof (proj1 (proj2 Hstack')) as Hsp'.
    cbn [beforeBoundary afterBoundary boundarySP boundaryStackBase] in Hsp, Hsp'.
    rewrite Hh in Hdelta. rewrite Hh'. lia. }
  subst next.
  exists z', (stackConfig z' outputs rest). split.
  - assert (HwfSmall : wf_config (stackConfig z values [i]))
      by (apply stack_config_wf; [exact Hz | exact Hbounds | constructor; [exact Hwfi | constructor]]).
    pose proof (one_spec_step _ _ HwfSmall Hwf' Hstep) as Hone.
    exact (stack_suffix_steps z z' values outputs [i] rest Hone (Forall_inv_tail Hwf)).
  - split; [ | split; [exact HidleAfter | split; [exact Heffect | split; [exact Hregs | exact Hstore]]]].
    unfold SourceState. cbn [sourceFunction sourceOffset]. rewrite Hf.
    split; [eapply frame_matches_preserved; eassumption | ]. split.
    + replace (S offset) with (offset + length [i])%nat by (cbn; lia). apply code_at_advance. exact Hcode.
    + split.
      * eapply validated_code_remaining with (executed := [i]); [exact Hsame | eapply frame_matches_preserved; eassumption |
          exact Hvalidation | cbn [sourceHeights]; rewrite Hsource; reflexivity | ].
        eapply source_facts_preserved; [exact Hsame | exact Hz' | exact Hrest].
      * split.
        -- exists outputs. split; [exact Hstack' | ]. split; [reflexivity | ].
           split; [apply stack_config_wf; [exact Hz' | exact (proj1 (proj2 (proj2 Hstack'))) | exact (Forall_inv_tail Hwf)] | exact Hlocals'].
        -- change (row COL_PC_AFTER = guestEntry f + Z.of_nat (codeOffset (guestBody f) (S offset))).
           rewrite (code_offset_next _ _ _ Hlookup), (ordinary_instruction_size _ Hordinary), Nat2Z.inj_add. lia.
Qed.

(* A callee being entered: still initializing, or, once both countdowns end, at
   the first instruction of its body. *)
Lemma callee_active_state : forall program function callee z arguments memory boundary,
  functionAt program function = Some callee -> length arguments = guestParams callee ->
  z = mk_state (fun_store z) (enteredFrame (guestModule callee) arguments (guestLocals callee)) ->
  frameMatches program callee z -> wf_state z -> validatedCode program callee z 0 (guestBody callee) ->
  wf_config (stackConfig z [] (guestBody callee)) ->
  InitializingFrame memory boundary arguments (Z.of_nat (guestLocals callee))
    (boundaryZeroRemaining boundary) (boundaryParamRemaining boundary) ->
  initializationFlags boundary -> boundaryPC boundary = guestEntry callee ->
  0 <= boundaryStackBase boundary -> Forall (fun value => 0 <= value < i32Modulus) arguments ->
  exists position, activeFunction program position = Some callee /\
    ActiveState program position z memory boundary (stackConfig z [] (guestBody callee)).
Proof.
  intros program function callee z arguments memory boundary Hf Hlength Hz Hframe Hwf Hvalid HconfigWf
    Hinit Hflags Hpc Hbase Hbounds.
  pose proof (zero_remaining_bounds _ _ _ _ _ _ Hinit) as Hzb.
  pose proof (param_remaining_bounds _ _ _ _ _ _ Hinit) as Hpb.
  destruct (Z.eq_dec (boundaryZeroRemaining boundary + boundaryParamRemaining boundary) 0) as [Hdone | Hpending].
  - assert (Hz0 : boundaryZeroRemaining boundary = 0) by lia.
    assert (Hp0 : boundaryParamRemaining boundary = 0) by lia.
    rewrite Hz0, Hp0 in Hinit.
    destruct (initialized_frame_matches memory boundary arguments (guestLocals callee) (fun_store z)
      (guestModule callee) Hbase Hbounds Hinit) as [Hempty Hlocals].
    exists (atInstruction (sourcePosition function 0)). split; [exact Hf | ]. split.
    + exists (guestBody callee). unfold SourceState. cbn [sourceFunction sourceOffset]. rewrite Hf.
      split; [exact Hframe | ]. split; [split; [lia | reflexivity] | ].
      split; [rewrite (stack_matches_height _ _ _ Hempty); exact Hvalid | ].
      split; [ | rewrite Hpc; unfold codeOffset; cbn; lia].
      exists []. split; [exact Hempty | ]. split; [reflexivity | ]. split; [exact HconfigWf | ].
      rewrite Hz. exact Hlocals.
    + destruct Hflags as [Hza Hpa]. rewrite Hz0 in Hza, Hpa. rewrite Hp0 in Hpa.
      repeat split; assumption.
  - exists (entering function arguments). split; [exact Hf | ].
    exists callee. repeat (split; [eassumption | ]). split; [lia | ]. split; [exact Hpc | reflexivity].
Qed.

Lemma split_top_values : forall (values : list Z) n,
  (n <= length values)%nat -> exists prefix top, values = prefix ++ top /\ length top = n.
Proof.
  intros values n H. exists (firstn (length values - n) values), (skipn (length values - n) values).
  split; [symmetry; apply firstn_skipn | rewrite length_skipn; lia].
Qed.

Lemma stack_config_split : forall z prefix top i rest,
  stackConfig z (prefix ++ top) (i :: rest) =
    stackPrefix prefix (instructionSuffix rest (stackConfig z top [i])).
Proof. intros. unfold stackConfig, stackPrefix, instructionSuffix. rewrite map_app, <- !app_assoc. reflexivity. Qed.

(* A direct guest call: SpecTec enters the callee's frame; the circuit saves the
   caller's return context one level above the callers already suspended, and
   begins initializing the callee's frame. *)
Lemma call_row_enters : forall program baseDepth stack index offset x rest f z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  functionAt program index = Some f ->
  SourceState program (sourcePosition index offset) z initial (beforeBoundary row) (CALL x :: rest) c0 ->
  SuspendedStack program f (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE)
    (row COL_CALL_STACK_DEPTH_BEFORE) (fun_store z) initial stack ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (length stack) ->
  exists cursor z' c1, fun_store z' = fun_store z /\ Steps (plugCalls stack c0) c1 /\
    ControlState program baseDepth cursor z' final (afterBoundary row) c1.
Proof.
  intros program baseDepth stack index offset x rest f z initial final row c0 Hprime Hloaded Hfull Hprogram
    Hreplay Hf Hsource Hsusp Hbase0 Hdepth.
  unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource. rewrite Hf in Hsource.
  destruct Hsource as [Hframe [Hcode [Hvalidation [Hrep Hpc]]]].
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) in Hpc.
  pose proof (represented_state_wf _ _ _ _ _ Hrep) as Hz.
  pose proof (validated_code_facts _ _ _ _ _ Hframe Hz Hvalidation) as Hfacts.
  pose proof (source_facts_wf _ _ _ _ Hfacts) as Hwf.
  destruct Hfacts as [Hif Hrest].
  cbn [sourceInstructionFacts callerFacts sourceHeight] in Hif, Hrest.
  destruct (programCallees program (guestModule f) (x :> N)) as [callee | ] eqn:Hcallee; [ | contradiction].
  destruct Hif as [Htyped [Hparams _]].
  destruct Htyped as [Hzwf [Hidx [Hfuncwf [Hp Hr]]]].
  assert (Hlookup : nth_error (guestBody f) offset = Some (CALL x)).
  { pose proof (f_equal (fun xs => nth_error xs 0) (proj2 Hcode)) as H.
    cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
  pose proof (encoded_call_image _ _ _ _ _ _ _ _ Hloaded Hf Hlookup Hframe Hcallee) as Hcall.
  rewrite <- Hpc in Hcall. destruct Hcall as [Himage HcallerImage].
  destruct Hrep as [values [Hstack [-> [HconfigWf Hlocals]]]].
  pose proof (stack_matches_height _ _ _ Hstack) as Hheight.
  destruct (split_top_values values (guestParams callee) ltac:(lia)) as [prefix [arguments [-> Hlen]]].
  rewrite <- Hlen in Himage, Hfuncwf.
  destruct (direct_call_entry_bound initial final row _ (row COL_PC_BEFORE + 1) z x (guestAddress callee)
    (guestModule callee) (guestTypeIndex callee) arguments (guestResults callee) (guestLocals callee)
    (guestBody callee) (guestEntry callee) Hprime Hfull Hprogram eq_refl Hreplay Himage ltac:(lia) Hr)
    as [HcallSel [Htarget [HparamCount [HzSeed [HpSeed Hentry]]]]].
  pose proof (call_starts_frame_initialization initial final row prefix arguments (guestLocals callee)
    Hprime Hfull HcallSel Htarget HparamCount HzSeed HpSeed Hreplay Hstack) as Hinit.
  destruct (full_direct_guest_call row Hprime Hfull HcallSel Htarget) as [Hguest [Hpush Hpop]].
  destruct (full_call_frame_entry row Hprime Hfull Hguest Hpush Hpop) as [HfbpAfter [HbaseAfter HdepthAfter]].
  destruct (full_direct_guest_call_stack row Hprime Hfull HcallSel Htarget) as [Hlo [Hhi Hsp]].
  destruct (full_call_context_push row Hfull Hpush) as [Haddr [HsavedFbp [HsavedBase Hchoice]]].
  assert (HcallStack : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HcallLocals : forall address, final localLoId address = initial localLoId address).
  { eapply no_local_writes_preserve; [exact Hreplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact Hfull | left; reflexivity | exact HcallSel]. }
  assert (HcallerCount : row COL_CURRENT_FUNCTION_NUM_LOCALS = Z.of_nat (length (stateLocals z))).
  { destruct HcallerImage as [reference [Hreference Hcount]].
    pose proof (current_function_read initial final row Hreplay
      (full_frame_lookup_active row COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; tauto) Hprogram)) as HrefRead.
    rewrite Hreference in HrefRead. injection HrefRead as HrefValue.
    pose proof (program_row_local_count_read initial final row Hreplay Hprogram) as HcountRead.
    rewrite <- HrefValue, Hcount in HcountRead. congruence. }
  pose proof (context_push_records initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedPcMem.
  pose proof (context_push_records initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedFbpMem.
  pose proof (context_push_records initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hreplay) as HsavedBaseMem.
  pose proof (call_return_pc_read initial final row Hreplay Hpush) as HreturnRead.
  destruct Himage as [Haddress [Hfunction [_ [_ [HreturnImage _]]]]].
  rewrite Hchoice, HreturnImage in HreturnRead. injection HreturnRead as HreturnValue.
  pose proof (proj1 (proj2 (proj2 Hstack))) as Hbounds. apply Forall_app in Hbounds.
  destruct Hbounds as [HprefixBounds HargBounds].
  destruct (proj1 Hvalidation) as [Hfunctions Hsupported].
  destruct z as [s frame0]. cbn [fun_store] in Hfunctions, Hsusp.
  set (zc := mk_state s (enteredFrame (guestModule callee) arguments (guestLocals callee))).
  pose proof (entered_body_wf s frame0 (guestModule callee) (guestTypeIndex callee) arguments
    (guestResults callee) (guestLocals callee) (guestBody callee) Hzwf Hfuncwf HargBounds) as HbodyWf.
  assert (Hzc : wf_state zc) by (inversion HbodyWf; assumption).
  pose proof (direct_i32_call_enters s frame0 x (guestAddress callee) (guestModule callee) (guestTypeIndex callee)
    arguments (guestResults callee) (guestLocals callee) (guestBody callee) Hzwf Hidx Haddress Hfunction
    Hfuncwf HargBounds) as Henter.
  pose proof (program_callee_validates program s (guestModule f) (x :> N) callee Hfunctions Hcallee) as Hinst.
  destruct (funcinst_body_validated _ _ _ _ _ _ _ Hinst) as [Cc [HCc HbodyTyped]].
  pose proof Hcallee as HcalleeAt. unfold programCallees in HcalleeAt.
  destruct (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) as [address | ]; [ | discriminate].
  assert (HcalleeValid : validatedCode program callee zc 0 (guestBody callee)).
  { split; [exact (proj1 Hvalidation) | ]. split; [exact (Hsupported _ _ HcalleeAt) | ].
    exists Cc. split; [exact HCc | exact HbodyTyped]. }
  assert (HcalleeFrame : frameMatches program callee zc).
  { split; [exact (proj1 Hframe) | ]. split; [reflexivity | ].
    cbn [stateLocals zc enteredFrame LOCALS]. rewrite length_app, length_map, repeat_length. lia. }
  destruct (callee_active_state program (N.to_nat address) callee zc arguments final (afterBoundary row)
    HcalleeAt Hlen eq_refl HcalleeFrame Hzc HcalleeValid HbodyWf Hinit
    (full_initialization_flags row Hprime Hfull) Hentry (proj1 (value_bound row _)) HargBounds)
    as [position [Hactive Hstate]].
  exists (controlCursor (suspendedCall (sourcePosition index offset) frame0 prefix (row COL_STACK_FRAME_BASE_BEFORE)
    (row COL_LOCALS_FBP_BEFORE) rest (guestResults callee) :: stack) position), zc,
    (plugCalls (suspendedCall (sourcePosition index offset) frame0 prefix (row COL_STACK_FRAME_BASE_BEFORE)
      (row COL_LOCALS_FBP_BEFORE) rest (guestResults callee) :: stack) (stackConfig zc [] (guestBody callee))).
  split; [reflexivity | ]. split.
  { cbn [plugCalls callerValues callContinuation callerFrame callResults].
    apply steps_in_plug_calls; [eapply suspended_stack_ready; exact Hsusp | ].
    rewrite stack_config_split. apply steps_with_stack_prefix; [exact HprefixBounds | ].
    apply steps_with_instruction_suffix; [exact (Forall_inv_tail Hwf) | exact Henter]. }
  exists callee, (stackConfig zc [] (guestBody callee)).
  split; [exact Hactive | ]. split; [exact Hstate | ].
  split; [ | split; [exact Hbase0 | split; [cbn [suspendedCalls length afterBoundary boundaryCallDepth]; lia | reflexivity]]].
  cbn [SuspendedStack callPosition callerFrame callerValues callerBase callerFbp callContinuation callResults
    sourceFunction sourceOffset afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth].
  assert (HcodeNext : codeOffset (guestBody f) (S offset) = (codeOffset (guestBody f) offset + 1)%nat)
    by (rewrite (code_offset_next _ _ _ Hlookup); reflexivity).
  exists f, x.
  cbn [callPosition callerFrame callerValues callerBase callerFbp callContinuation callResults sourceFunction sourceOffset].
  split; [exact Hf | ]. split; [exact Hlookup | ].
  split; [replace (S offset) with (offset + length [CALL x])%nat by (cbn; lia); apply code_at_advance; exact Hcode | ].
  split; [exact Hcallee | ]. split; [reflexivity | ]. split; [exact Hframe | ].
  split; [inversion Hzwf; assumption | ].
  split.
  { assert (Hafter : (stackHeight (beforeBoundary row) - guestParams callee + guestResults callee =
        length prefix + guestResults callee)%nat) by (rewrite Hheight, length_app, Hlen; lia).
    rewrite Hafter in Hrest.
    eapply validated_code_remaining with (executed := [CALL x]); [repeat split | exact Hframe |
      exact Hvalidation | cbn [sourceHeights sourceHeight]; rewrite Hcallee, Hafter; reflexivity | exact Hrest]. }
  split; [exact HprefixBounds | ]. split; [exact (proj1 Hstack) | ].
  split.
  { pose proof (proj1 (proj2 Hstack)) as HspBefore. cbn [beforeBoundary boundarySP boundaryStackBase] in HspBefore.
    rewrite length_app, Nat2Z.inj_add in HspBefore. rewrite HbaseAfter, Hsp, HparamCount. lia. }
  split.
  { intros i value Hi. rewrite HcallStack. apply (proj2 (proj2 (proj2 Hstack))).
    rewrite nth_error_app1; [exact Hi | apply nth_error_Some; rewrite Hi; discriminate]. }
  split; [rewrite HfbpAfter, HcallerCount; reflexivity | ].
  split.
  { intros i v Hi. destruct (Hlocals i v Hi) as [value [Hv [Hb Hcell]]].
    exists value. split; [exact Hv | ]. split; [exact Hb | ]. rewrite HcallLocals. exact Hcell. }
  replace (row COL_CALL_STACK_DEPTH_AFTER - 1) with (row COL_CALL_STACK_ADDR) by lia.
  split; [rewrite HsavedPcMem, <- HreturnValue, Hpc, HcodeNext, Nat2Z.inj_add; f_equal; lia | ].
  split; [rewrite HsavedFbpMem, HsavedFbp; reflexivity | ].
  split; [rewrite HsavedBaseMem, HsavedBase; reflexivity | ].
  (* The callers already suspended keep their memory: the push writes only its own level. *)
  replace (row COL_CALL_STACK_ADDR) with (row COL_CALL_STACK_DEPTH_BEFORE) by lia.
  eapply suspended_stack_preserved; [exact Hsusp | intros i _; apply HcallStack | intros other i _; apply HcallLocals | ].
  intros id column level Hin Hbelow. eapply context_push_elsewhere; [exact Hin | exact Hreplay | ].
  intros Heq. injection Heq as Heq. lia.
Qed.

Lemma steps_refl_wf : forall c, wf_config c -> Steps c c.
Proof. intros [z code] H. apply Steps__refl. exact H. Qed.

(* One frame-initialization row: no SpecTec step; one more callee local is set. *)
Lemma init_row_step : forall program baseDepth stack function arguments z initial final row c0,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final ->
  ControlState program baseDepth (controlCursor stack (entering function arguments)) z initial
    (beforeBoundary row) c0 ->
  exists cursor c1, Steps c0 c1 /\ ControlState program baseDepth cursor z final (afterBoundary row) c1.
Proof.
  intros program baseDepth stack function arguments z initial final row c0 Hprime Hloaded Hfull Hreplay
    [f [inner [Hactive [Hstate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition activeFunction suspendedCalls] in Hactive, Hstate, Hsusp, Hdepth.
  destruct Hstate as [callee [Hfn [Hlen [Hargs [Hzshape [Hframe [Hzwf [Hvalid [HbodyWf [Hinit [Hflags
    [Hpending [Hpc ->]]]]]]]]]]]]].
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
  destruct (callee_active_state program function f z arguments final (afterBoundary row) Hfn Hlen Hzshape
    Hframe Hzwf Hvalid HbodyWf Hinit' (full_initialization_flags row Hprime Hfull)
    ltac:(cbn [afterBoundary boundaryPC]; rewrite HpcAfter; exact Hpc) (proj1 (value_bound row _)) Hargs)
    as [position [Hactive' Hstate']].
  exists (controlCursor stack position), (plugCalls stack (stackConfig z [] (guestBody f))). split.
  - apply steps_refl_wf. apply plug_calls_wf; [eapply suspended_stack_ready; exact Hsusp | exact HbodyWf].
  - exists f, (stackConfig z [] (guestBody f)). split; [exact Hactive' | ]. split; [exact Hstate' | ].
    split; [ | split; [exact Hbase0 | split; [cbn [afterBoundary beforeBoundary boundaryCallDepth suspendedCalls] in *; lia | reflexivity]]].
    cbn [suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth].
    cbn [beforeBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth] in Hsusp.
    rewrite Hfbp, Hbase, Hd. eapply frame_effect_suspended; [exact Heffect | exact Hsusp].
Qed.

Lemma stack_config_rejoin : forall z values results code,
  stackPrefix values (instructionSuffix code (mk_config z (List.map i32Constant results))) =
    stackConfig z (values ++ results) code.
Proof. intros. unfold stackPrefix, instructionSuffix, stackConfig. rewrite map_app, <- app_assoc. reflexivity. Qed.

(* A callee's function-ending END: SpecTec leaves the LABEL_ and FRAME_ with the
   results, and the circuit restores the caller's registers from its saved context. *)
Lemma return_row_step : forall program baseDepth top rest position z initial final row inner f,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  functionAt program (sourceFunction position) = Some f ->
  SourceState program position z initial (beforeBoundary row) [] inner ->
  initializationIdle (beforeBoundary row) ->
  SuspendedStack program f (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE)
    (row COL_CALL_STACK_DEPTH_BEFORE) (fun_store z) initial (top :: rest) ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (S (length rest)) ->
  exists cursor z' c1, fun_store z' = fun_store z /\ Steps (plugCalls (top :: rest) inner) c1 /\
    ControlState program baseDepth cursor z' final (afterBoundary row) c1.
Proof.
  intros program baseDepth top rest [index offset] z initial final row inner f Hprime Hloaded Hfull Hprogram
    Hreplay Hf Hsource Hidle Hsusp Hbase0 Hdepth.
  unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource, Hf. rewrite Hf in Hsource.
  destruct Hsource as [Hframe [[Hbound Hcode] [Hvalid [Hrep Hpc]]]].
  assert (Hoffset : offset = length (guestBody f)).
  { pose proof (f_equal (@length instr) Hcode) as Hlength. rewrite length_skipn in Hlength. cbn in Hlength; lia. }
  subst offset. rewrite code_offset_end in Hpc.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))) in Hpc.
  destruct Hrep as [values [Hstack [-> [HinnerWf Hlocals]]]].
  (* The remaining code is empty and typed to the declared results. *)
  destruct Hvalid as [_ [_ [C [HC Htyp]]]].
  assert (Hctx : context_LOCALS (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) = repeat valtype_I32 (length (stateLocals z))).
  { rewrite function_context_locals by (eapply module_context_locals; eassumption).
    destruct Hframe as [_ [_ Hlen]]. now rewrite Hlen. }
  pose proof (source_validation_height _ _ _ Htyp (fun _ => None) z Hctx ltac:(intros ? ? Hnone; discriminate)
    (Forall_nil _) 0%nat) as Hheights.
  rewrite !repeat_length in Hheights. cbn [sourceHeights Nat.add] in Hheights. injection Hheights as Hresults.
  rewrite (stack_matches_height _ _ _ Hstack) in Hresults.
  (* The END row pops the context saved one level down. *)
  destruct (encoded_function_end _ _ _ _ Hloaded Hf) as [Hop Hedge].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hop in Hfetch. injection Hfetch as HcodeValue.
  assert (Hend : row COL_SEL_END = 1).
  { apply (full_opcode_selects row COL_SEL_END CODE_END Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact HcodeValue]. }
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindFetch.
  rewrite Hpc, Hedge in HkindFetch. injection HkindFetch as Hkind.
  destruct (full_end_pops_nonempty_context row Hprime Hfull Hend ltac:(lia) ltac:(lia)) as [Hpop [Hpush Hcapture]].
  destruct (full_return_context_restore row Hfull Hpop) as [HpcAfter [HfbpAfter [HbaseAfter HaddrAfter]]].
  pose proof (full_call_depth_balance row Hprime Hfull) as HdepthBalance.
  rewrite Hpush, Hpop in HdepthBalance.
  destruct (full_end_stack row Hprime Hfull Hend Hcapture) as [Hlo [Hhi Hsp]].
  assert (HstackMem : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HlocalMem : forall address, final localLoId address = initial localLoId address).
  { eapply no_local_writes_preserve; [exact Hreplay | ].
    eapply full_call_end_no_local_writes; [exact Hprime | exact Hfull | right; reflexivity | exact Hend]. }
  assert (HcontextMem : forall id column address, In (id, column) contextMemories ->
      final id address = initial id address)
    by (intros id column address Hin; eapply context_preserved_without_push; eassumption).
  destruct Hsusp as [g [x [Hg [Hcall [Hcont [Hcallee [Hres [HframeC [HwfC [HvalidC [Hbounds
    [HbaseC0 [HbaseC [Hcells [HfbpC [HlocalsC [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]]]]]]]]]]]].
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
  destruct z as [s calleeFrame].
  assert (Hz : wf_state (mk_state s calleeFrame)) by (inversion HinnerWf; assumption).
  assert (HcalleeWf : wf_frame calleeFrame) by (inversion Hz; assumption).
  assert (HcallerState : wf_state (mk_state s (callerFrame top))) by (inversion Hz; constructor; assumption).
  pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues.
  pose proof (i32_function_returns s (callerFrame top) calleeFrame values HcallerState HcalleeWf Hvalues) as Hleave.
  assert (Hinner : callContext (callerFrame top) (N.of_nat (callResults top))
      (stackConfig (mk_state s calleeFrame) values []) =
    mk_config (mk_state s (callerFrame top)) [FRAME_ (N.of_nat (length values)) calleeFrame
      [LABEL_ (N.of_nat (length values)) [] (List.map i32Constant values)]]).
  { unfold callContext, stackConfig. cbn [List.map]. rewrite app_nil_r, Hres, <- Hresults. reflexivity. }
  pose proof (suspended_stack_ready _ _ _ _ _ _ _ _ Hrest) as Hready.
  exists (controlCursor rest (atInstruction (advanceSource (callPosition top) 1))), (mk_state s (callerFrame top)),
    (plugCalls rest (stackConfig (mk_state s (callerFrame top)) (callerValues top ++ values) (callContinuation top))).
  split; [reflexivity | ]. split.
  { cbn [plugCalls]. rewrite Hinner, <- stack_config_rejoin. apply steps_in_plug_calls; [exact Hready | ].
    apply steps_with_stack_prefix; [exact Hbounds | ].
    apply steps_with_instruction_suffix; [eapply validated_instructions_wf; exact HvalidC | exact Hleave]. }
  pose proof (proj1 (proj2 Hstack)) as HspBefore.
  cbn [beforeBoundary boundarySP boundaryStackBase] in HspBefore.
  assert (HheightAfter : stackHeight (afterBoundary row) = (length (callerValues top) + callResults top)%nat).
  { unfold stackHeight. cbn [afterBoundary boundarySP boundaryStackBase].
    rewrite Hsp, HbaseAfter, <- HbaseValue, HspBefore, HbaseC, Hres, <- Hresults. lia. }
  exists g, (stackConfig (mk_state s (callerFrame top)) (callerValues top ++ values) (callContinuation top)).
  split; [exact Hg | ]. split.
  - split; [ | apply (program_row_keeps_idle row COL_SEL_END); [exact Hprime | exact Hfull | vm_compute; tauto |
      exact Hend | not_call_selector | exact Hidle]].
    exists (callContinuation top). unfold SourceState, advanceSource. cbn [sourceFunction sourceOffset].
    rewrite Hg, Nat.add_1_r.
    split; [exact HframeC | ]. split; [exact Hcont | ]. split; [rewrite HheightAfter; exact HvalidC | ]. split.
    + exists (callerValues top ++ values). split; [ | split; [reflexivity | split]].
      * unfold stackMatches. cbn [afterBoundary boundaryStackBase boundarySP].
        split; [rewrite HbaseAfter, <- HbaseValue; exact HbaseC0 | ].
        split; [rewrite length_app, Nat2Z.inj_add, Hsp, HbaseAfter, <- HbaseValue, HspBefore, HbaseC; lia | ].
        split; [apply Forall_app; split; assumption | ].
        intros i value Hi. rewrite HbaseAfter, <- HbaseValue, HstackMem.
        destruct (Nat.lt_ge_cases i (length (callerValues top))) as [Hin | Hout].
        -- apply Hcells. rewrite nth_error_app1 in Hi by exact Hin. exact Hi.
        -- rewrite nth_error_app2 in Hi by exact Hout.
           replace (2 * (callerBase top + Z.of_nat i)) with
             (2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (i - length (callerValues top))))
             by (rewrite HbaseC, Nat2Z.inj_sub by exact Hout; lia).
           exact (proj2 (proj2 (proj2 Hstack)) _ _ Hi).
      * apply stack_config_wf; [exact HcallerState | apply Forall_app; split; assumption |
          eapply validated_instructions_wf; exact HvalidC].
      * intros i v Hi. destruct (HlocalsC i v Hi) as [value [Hv [Hb Hcell]]].
        exists value. split; [exact Hv | ]. split; [exact Hb | ].
        cbn [afterBoundary boundaryLocalsBase]. rewrite HfbpAfter, <- HfbpValue, HlocalMem. exact Hcell.
    + cbn [afterBoundary boundaryPC]. rewrite HpcAfter, <- HretValue. reflexivity.
  - split.
    + cbn [suspendedCalls afterBoundary boundaryStackBase boundaryLocalsBase boundaryCallDepth fun_store].
      rewrite HbaseAfter, <- HbaseValue, HfbpAfter, <- HfbpValue, HdepthBalance.
      replace (row COL_CALL_STACK_DEPTH_BEFORE + 0 - 1) with (row COL_CALL_STACK_DEPTH_BEFORE - 1) by lia.
      eapply suspended_stack_preserved; [exact Hrest | | | ].
      * intros i _. apply HstackMem.
      * intros other i _. apply HlocalMem.
      * intros id column address Hin _. eapply HcontextMem. exact Hin.
    + split; [exact Hbase0 | ]. split; [ | reflexivity].
      cbn [afterBoundary boundaryCallDepth suspendedCalls]. rewrite HdepthBalance, Hdepth, Nat2Z.inj_succ. lia.
Qed.
