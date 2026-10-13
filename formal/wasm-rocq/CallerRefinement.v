(* Compose circuit instruction rows and complete calls in their caller continuation. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers StraightLineSpec StackSimulation LocalSimulation SpecTecCalls
  CallSpec CallBody CallScope CallRefinement CallerSpec.
Import ListNotations.
Open Scope Z_scope.

Definition sameCaller (before after : state) : Prop :=
  fun_store after = fun_store before /\ fun_funcaddr after = fun_funcaddr before /\
  length (stateLocals after) = length (stateLocals before).

Definition sameRoms (before after : MemoryState) : Prop :=
  forall memory, readOnlyMemory memory memories = true ->
    forall address, after memory address = before memory address.

Lemma run_preserves_roms : forall initial rows final,
  validCircuitRun initial rows final -> sameRoms initial final.
Proof.
  intros initial rows final [_ [_ Hreplay]] memory Hreadonly address.
  eapply replay_preserves; [exact Hreplay | apply read_only_trace; exact Hreadonly].
Qed.

Ltac preserve_roms H :=
  repeat match goal with
  | |- context [ ?after ?memory ] =>
      progress setoid_rewrite (H memory ltac:(vm_compute; reflexivity))
  end.

Lemma instruction_image_same_roms : forall initial final pc instruction,
  sameRoms initial final -> instructionImage initial pc instruction -> instructionImage final pc instruction.
Proof.
  intros initial final pc instruction Hrom Himage.
  unfold instructionImage in *. preserve_roms Hrom.
  destruct instruction; cbn in *; preserve_roms Hrom; exact Himage.
Qed.

Lemma loaded_body_same_roms : forall initial final entry body,
  sameRoms initial final -> loadedBody initial entry body -> loadedBody final entry body.
Proof.
  intros initial final entry body. revert entry.
  induction body as [ | instruction rest IH]; intros entry Hrom Hloaded; [exact I | ].
  destruct Hloaded as [Himage Hrest]. split.
  - eapply instruction_image_same_roms; eassumption.
  - eapply IH; eassumption.
Qed.

Lemma loaded_call_preserved : forall initial final pc returnPC z z' index address module typeIndex
    params results locals body entry,
  sameRoms initial final -> sameCaller z z' ->
  loadedCall initial pc returnPC z index address module typeIndex params results locals body entry ->
  loadedCall final pc returnPC z' index address module typeIndex params results locals body entry.
Proof.
  intros initial final pc returnPC z z' index address module typeIndex params results locals body entry
    Hrom [Hstore [Haddresses Hcount]] [Hcall [Hcaller [Hbody Hend]]].
  assert (Hfunctions : fun_funcinst z' = fun_funcinst z).
  { destruct z, z'. cbn in Hstore |- *. now rewrite Hstore. }
  split.
  - unfold directCallImage, guestFunctionImage in *. rewrite Haddresses, Hfunctions.
    preserve_roms Hrom. exact Hcall.
  - split.
    + unfold callerFrameImage in *. rewrite Hcount. preserve_roms Hrom. exact Hcaller.
    + split; [eapply loaded_body_same_roms; eassumption | ]. preserve_roms Hrom. exact Hend.
Qed.

Lemma loaded_caller_preserved : forall program callees initial final pc z z',
  sameRoms initial final -> sameCaller z z' ->
  loadedCaller callees z initial pc program -> loadedCaller callees z' final pc program.
Proof.
  induction program as [ | instruction rest IH]; intros callees initial final pc z z' Hrom Hcaller Hloaded;
    [exact I | ]. destruct Hloaded as [Hfirst Hrest]. split.
  - destruct instruction; cbn in *; try (eapply instruction_image_same_roms; eassumption).
    match goal with |- context [callees ?x] => destruct (callees x) as [f | ] end; [ | contradiction].
    eapply loaded_call_preserved; eassumption.
  - eapply IH; eassumption.
Qed.

Lemma caller_typed_preserved : forall program callees z z' height,
  length (stateLocals z') = length (stateLocals z) -> wf_state z' ->
  callerTyped callees z height program -> callerTyped callees z' height program.
Proof.
  induction program as [ | instruction rest IH]; intros callees z z' height Hcount Hwf Htyped; [exact I | ].
  assert (Hcontinue : forall next,
    (match next with Some h => callerTyped callees z h rest | None => False end) ->
    (match next with Some h => callerTyped callees z' h rest | None => False end)).
  { intros [h | ] H; [eapply IH; eassumption | contradiction]. }
  pose proof (Hcontinue (instructionStackHeight instruction height)) as Hnext.
  destruct instruction; cbn [callerTyped] in *;
    try (destruct Htyped as [Hs [Hi [Ht Hr]]];
      split; [exact Hs | ]; split; [exact Hi | ]; split; [now rewrite Hcount | exact (Hnext Hr)]).
  match goal with |- context [callees ?x] => destruct (callees x) as [f | ] end; [ | contradiction].
  destruct Htyped as [[_ Hcall] [Hheight Hrest]].
  split; [split; assumption | ]. split; [exact Hheight | eapply IH; eassumption].
Qed.

Lemma caller_program_wf : forall program callees z height,
  callerTyped callees z height program -> Forall wf_instr program.
Proof.
  induction program as [ | instruction rest IH]; intros callees z height Htyped; [constructor | ].
  assert (Hcontinue : forall next,
    (match next with Some h => callerTyped callees z h rest | None => False end) ->
    Forall wf_instr rest).
  { intros [h | ] H; [eapply IH; exact H | contradiction]. }
  pose proof (Hcontinue (instructionStackHeight instruction height)) as Hnext.
  destruct instruction; cbn [callerTyped] in Htyped;
    try (destruct Htyped as [_ [Hi [_ Hr]]]; constructor; [exact Hi | exact (Hnext Hr)]).
  match type of Htyped with context [callees ?x] => destruct (callees x) as [f | ] end; [ | contradiction].
  destruct Htyped as [[_ [Hindex _]] [_ Hrest]].
  constructor; [constructor; exact Hindex | eapply IH; exact Hrest].
Qed.

Lemma stack_config_wf : forall z values code,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values ->
  Forall wf_instr code -> wf_config (stackConfig z values code).
Proof.
  intros z values code Hz Hvalues Hcode. constructor; [exact Hz | ].
  apply Forall_app. split; apply Forall_map.
  - eapply Forall_impl; [ | exact Hvalues]. intros value H. apply i32_constant_wf. exact H.
  - eapply Forall_impl; [ | exact Hcode]. intros i H. apply instruction_admin_wf. exact H.
Qed.

Lemma stack_suffix_steps : forall z z' values values' code rest,
  Steps (stackConfig z values code) (stackConfig z' values' []) ->
  Forall wf_instr rest ->
  Steps (stackConfig z values (code ++ rest)) (stackConfig z' values' rest).
Proof.
  intros z z' values values' code rest Hsteps Hrest.
  pose proof (steps_with_instruction_suffix rest _ _ Hrest Hsteps) as Hlift.
  unfold instructionSuffix, stackConfig in Hlift |- *.
  rewrite map_app. cbn [map] in Hlift. rewrite app_nil_r in Hlift.
  rewrite <- app_assoc in Hlift. exact Hlift.
Qed.

Lemma ordinary_result_height : forall i before after next initial final row,
  instructionStackHeight i (length before) = Some next ->
  stackMatches initial (beforeBoundary row) before ->
  stackMatches final (afterBoundary row) after ->
  frameRegistersPreserved row ->
  instructionStackDelta i = Some (row COL_SP_AFTER - row COL_SP_BEFORE) ->
  length after = next.
Proof.
  intros i before after next initial final row Hheight [_ [Hb _]] [_ [Ha _]] [_ [Hbase _]] Hdelta.
  pose proof (instruction_stack_height_delta i _ _ Hheight) as HheightDelta.
  rewrite Hdelta in HheightDelta. injection HheightDelta as HheightDelta.
  cbn [beforeBoundary afterBoundary boundarySP boundaryStackBase] in Hb, Ha. lia.
Qed.

Lemma caller_segment_refinement : forall instruction rest segment callees z initial final pc c0,
  fieldPrime -> validCircuitRun initial (segmentRows segment) final ->
  segmentMatches instruction segment ->
  loadedCaller callees z initial pc (instruction :: rest) ->
  boundaryPC (beforeBoundary (segmentFirst segment)) = pc ->
  (callerTyped callees z (stackHeight (beforeBoundary (segmentFirst segment))) (instruction :: rest) /\
    represents z initial (beforeBoundary (segmentFirst segment)) (instruction :: rest) c0) ->
  exists z' c1, Steps c0 c1 /\
    (callerTyped callees z' (stackHeight (afterBoundary (segmentLast segment))) rest /\
    represents z' final (afterBoundary (segmentLast segment)) rest c1) /\
    boundaryPC (afterBoundary (segmentLast segment)) = pc + 1 /\ sameCaller z z'.
Proof.
  intros instruction rest segment callees z initial final pc c0 Hprime Hrun Hmatches [Himage Hloaded] Hpc
    Hready. apply represented_values in Hready.
  destruct Hready as [values [Hstack [HtypedAll [-> [Hwf Hlocals]]]]].
  assert (Hz : wf_state z) by (inversion Hwf; assumption).
  destruct segment as [row | first middle last].
  - assert (Hplain : supportedStraightLineProgram [instruction] /\ wf_instr instruction /\
        programTyped (length (stateLocals z)) (length values) [instruction] /\
        match instructionStackHeight instruction (length values) with
        | Some remaining => callerTyped callees z remaining rest | None => False end).
    { destruct instruction; cbn [segmentMatches] in Hmatches; try contradiction; exact HtypedAll. }
    assert (HimagePlain : instructionImage initial pc instruction).
    { destruct instruction; cbn [segmentMatches] in Hmatches; try contradiction; exact Himage. }
    assert (Hprogram : row COL_IS_PROGRAM_ROW = 1).
    { destruct instruction; cbn [segmentMatches] in Hmatches; tauto. }
    destruct Hplain as [Hsupported [Hi [Htyped Hrest]]].
    destruct (instructionStackHeight instruction (length values)) as [nextHeight | ] eqn:Hheight; [ | contradiction].
    pose proof (caller_program_wf rest callees z _ Hrest) as HrestWf.
    destruct (valid_run_head _ _ _ _ Hrun) as [after [Hfull [Hreplay Hempty]]].
    destruct Hempty as [_ [_ Hempty]].
    assert (Heq : after = final) by (inversion Hempty; reflexivity). subst after.
    assert (Hsmall : programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) [instruction] /\
      represents z initial (beforeBoundary row) [instruction] (stackConfig z values [instruction])).
    { apply represented_values. exists values. split; [exact Hstack | ]. split; [exact Htyped | ]. split; [reflexivity | ].
      split; [apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) | constructor; auto] | exact Hlocals]. }
    change (row COL_PC_BEFORE = pc) in Hpc. rewrite <- Hpc in HimagePlain.
    destruct (frame_row_refinement instruction [] z initial final row _ Hprime Hfull Hprogram Hreplay HimagePlain Hsmall)
      as [z' [c1 [Hstep [Hafter [Hnext [_ [Hregs [Hdelta Hidentity]]]]]]]].
    apply represented_values in Hafter.
    destruct Hafter as [outputs [Hout [_ [-> [HoutWf HoutLocals]]]]].
    assert (Hlength : length outputs = nextHeight).
    { eapply ordinary_result_height; eassumption. }
    assert (Hz' : wf_state z') by (inversion HoutWf; assumption).
    assert (Hrest' : callerTyped callees z' (length outputs) rest).
    { rewrite Hlength. eapply caller_typed_preserved; [exact (proj2 (proj2 Hidentity)) | exact Hz' | exact Hrest]. }
    assert (Hsteps : Steps (stackConfig z values [instruction]) (stackConfig z' outputs [])).
    { eapply trans; [exact Hstep | apply Steps__refl; exact HoutWf | | exact HoutWf | exact HoutWf].
      pose proof (represents_wf _ _ _ _ _ (proj2 Hsmall)) as HsmallWf. exact HsmallWf. }
    exists z', (stackConfig z' outputs rest). split.
    + exact (stack_suffix_steps z z' values outputs [instruction] rest Hsteps HrestWf).
    + split.
      * apply represented_values. exists outputs. split; [exact Hout | ]. split; [exact Hrest' | ]. split; [reflexivity | ].
        split; [apply stack_config_wf; [exact Hz' | exact (proj1 (proj2 (proj2 Hout))) | exact HrestWf] | exact HoutLocals].
      * split; [change (row COL_PC_AFTER = pc + 1); lia | exact Hidentity].
  - destruct instruction; cbn [segmentMatches] in Hmatches; try contradiction.
    match type of Hwf with wf_config (stackConfig _ _ (CALL ?x :: _)) => rename x into index end.
    cbn [callerTyped] in HtypedAll. cbv beta iota in Himage.
    destruct (callees (index :> N)) as [f | ] eqn:Hlookup; [ | contradiction].
    destruct Hmatches as [Hfirst [Hlast Hmiddle]]. destruct HtypedAll as [HcallTyped [Hheight Hrest]].
    pose proof (caller_program_wf rest callees z _ Hrest) as HrestWf.
    set (prefix := firstn (length values - guestParams f) values).
    set (arguments := skipn (length values - guestParams f) values).
    assert (Hvalues : values = prefix ++ arguments) by (symmetry; apply firstn_skipn).
    assert (Harguments : length arguments = guestParams f) by (unfold arguments; rewrite length_skipn; lia).
    rewrite <- Harguments in Himage, HcallTyped. rewrite Hvalues in Hstack.
    destruct (direct_call_run_refinement initial final first middle last pc (pc + 1) z index
      (guestAddress f) (guestModule f) (guestTypeIndex f) prefix arguments (guestResults f)
      (guestLocals f) (guestBody f) (guestEntry f) Hprime Himage HcallTyped Hrun Hfirst Hlast Hmiddle Hpc Hstack Hlocals)
      as [outputs [Hlength [Hsteps [Hout [HoutLocals [Hnext Hregisters]]]]]].
    assert (HheightOut : length (prefix ++ outputs) = (length values - guestParams f + guestResults f)%nat).
    { unfold prefix. rewrite length_app, length_firstn, Hlength. lia. }
    exists z, (stackConfig z (prefix ++ outputs) rest). split.
    + rewrite Hvalues.
      exact (stack_suffix_steps z z (prefix ++ arguments) (prefix ++ outputs) [CALL index] rest Hsteps HrestWf).
    + split.
      * apply represented_values. exists (prefix ++ outputs). split; [exact Hout | ]. split; [now rewrite HheightOut | ].
        split; [reflexivity | ]. split.
        -- apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hout))) | exact HrestWf].
        -- exact HoutLocals.
      * split; [exact Hnext | repeat split; reflexivity].
Qed.

Lemma segment_chain_boundary : forall segment next rest,
  Chain continuityLinks (segmentRows segment ++ callerRows (next :: rest)) ->
  afterBoundary (segmentLast segment) = beforeBoundary (segmentFirst next).
Proof.
  intros segment next rest Hchain. apply follows_boundary.
  assert (Hhead : exists tail, callerRows (next :: rest) = segmentFirst next :: tail).
  { destruct next; eexists; reflexivity. }
  destruct Hhead as [tail Hhead]. rewrite Hhead in Hchain.
  destruct segment as [row | first middle last]; cbn [segmentRows segmentLast app] in *.
  - inversion Hchain; subst; assumption.
  - assert (Hprefix : Chain continuityLinks (first :: middle ++ last :: segmentFirst next :: tail)).
    { now rewrite <- app_assoc in Hchain. }
    clear Hchain. revert first Hprefix. induction middle as [ | row middle IH]; intros first Hchain.
    + inversion Hchain; subst. match goal with H : Chain _ (last :: _) |- _ => inversion H; subst; assumption end.
    + apply (IH row). inversion Hchain; subst; assumption.
Qed.

(* Execute a source prefix, retaining its continuation in the final configuration.
   Segmentation is explicit; each segment is checked against its source instruction. *)
Theorem run_refinement : forall executed remaining segments callees z initial final boundary pc c0,
  fieldPrime ->
  validCircuitRun initial (callerRows segments) final ->
  Forall2 segmentMatches executed segments ->
  loadedCaller callees z initial pc (executed ++ remaining) ->
  boundaryPC boundary = pc -> callerStartsAt boundary segments ->
  callerTyped callees z (stackHeight boundary) (executed ++ remaining) ->
  represents z initial boundary (executed ++ remaining) c0 ->
  exists z' c1,
    Steps c0 c1 /\
    callerTyped callees z' (stackHeight (callerFinalBoundary boundary segments)) remaining /\
    represents z' final (callerFinalBoundary boundary segments) remaining c1 /\
    boundaryPC (callerFinalBoundary boundary segments) = pc + Z.of_nat (length executed).
Proof.
  induction executed as [ | instruction rest IH];
    intros remaining segments callees z initial final boundary pc c0 Hprime Hrun Hmatches Hloaded Hpc Hstart Htyped Hrep.
  - inversion Hmatches; subst segments. destruct Hrun as [_ [_ Hreplay]].
    assert (Heq : initial = final) by (inversion Hreplay; reflexivity). subst final.
    exists z, c0. split.
    + pose proof (represents_wf _ _ _ _ _ Hrep) as Hwf. destruct c0. apply Steps__refl. exact Hwf.
    + split; [exact Htyped | ]. split; [exact Hrep | cbn [callerFinalBoundary length]; lia].
  - inversion Hmatches as [ | instruction' segment rest' segments' Hmatch Htail]; subst instruction' rest' segments.
    destruct (valid_run_split (segmentRows segment) (callerRows segments') initial final Hrun)
      as [middle [HfirstRun HrestRun]].
    change (beforeBoundary (segmentFirst segment) = boundary) in Hstart.
    assert (HsegmentPC : boundaryPC (beforeBoundary (segmentFirst segment)) = pc) by now rewrite Hstart.
    assert (Hbefore : callerTyped callees z (stackHeight (beforeBoundary (segmentFirst segment)))
        (instruction :: rest ++ remaining) /\
      represents z initial (beforeBoundary (segmentFirst segment)) (instruction :: rest ++ remaining) c0).
    { rewrite Hstart. split; assumption. }
    destruct (caller_segment_refinement instruction (rest ++ remaining) segment callees z initial middle pc c0 Hprime HfirstRun
      Hmatch Hloaded HsegmentPC Hbefore) as [z1 [c1 [Hsteps [[HtypedAfter Hafter] [Hnext Hcaller]]]]].
    assert (HloadedRest : loadedCaller callees z1 middle (pc + 1) (rest ++ remaining)).
    { eapply loaded_caller_preserved; [eapply run_preserves_roms; exact HfirstRun | exact Hcaller | exact (proj2 Hloaded)]. }
    assert (HstartRest : callerStartsAt (afterBoundary (segmentLast segment)) segments').
    { destruct segments' as [ | next segments']; [exact I | ].
      symmetry. apply segment_chain_boundary with (rest := segments'). exact (proj1 (proj2 Hrun)). }
    destruct (IH remaining segments' callees z1 middle final (afterBoundary (segmentLast segment)) (pc + 1) c1
      Hprime HrestRun Htail HloadedRest Hnext HstartRest HtypedAfter Hafter)
      as [z2 [c2 [HrestSteps [HtypedFinal [Hfinal HfinalPC]]]]].
    exists z2, c2. split; [eapply spec_steps_transitive; eassumption | ].
    split; [exact HtypedFinal | ]. split; [exact Hfinal | ].
    cbn [callerFinalBoundary length]; rewrite Nat2Z.inj_succ; lia.
Qed.
