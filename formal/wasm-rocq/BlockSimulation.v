(* Run straight-line code above a protected stack prefix, for block bodies. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers StraightLineSpec StackSimulation LocalSimulation SpecTecCalls SpecTecBranches
  CallSpec CallScope CallBody CallerRefinement BlockInstructions.
Import ListNotations.
Open Scope Z_scope.

Lemma program_typed_tail : forall count before i rest after,
  programTyped count before (i :: rest) -> instructionStackHeight i before = Some after ->
  programTyped count after rest.
Proof.
  intros count before i rest after Ht Hh. destruct i;
    cbn [programTyped] in Ht; try (rewrite Hh in Ht; exact Ht);
    destruct Ht as [_ Ht]; rewrite Hh in Ht; exact Ht.
Qed.

Lemma body_above_prefix : forall body rows previous prefix values initial final entry z,
  fieldPrime -> validCircuitRun initial rows final -> Chain continuityLinks (previous :: rows) ->
  Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) rows -> length rows = length body ->
  loadedBody initial entry body -> previous COL_PC_AFTER = entry ->
  wf_state z -> Forall wf_instr body ->
  programTyped (length (stateLocals z)) (length values) body ->
  stackMatches initial (afterBoundary previous) (prefix ++ values) ->
  localsMatch initial (afterBoundary previous) z ->
  exists z' values',
    Steps (stackConfig z values body) (stackConfig z' values' []) /\ wf_state z' /\
    stackMatches final (afterBoundary (finalRow previous rows)) (prefix ++ values') /\
    localsMatch final (afterBoundary (finalRow previous rows)) z' /\ sameCaller z z' /\
    finalRow previous rows COL_PC_AFTER = entry + Z.of_nat (length body) /\
    finalRow previous rows COL_CALL_STACK_DEPTH_AFTER = previous COL_CALL_STACK_DEPTH_AFTER /\
    bodyStackDelta body = Some (Z.of_nat (length values') - Z.of_nat (length values)).
Proof.
  induction body as [ | i rest IH]; intros rows previous prefix values initial final entry z
    Hprime Hrun Hchain Hprogram Hlength Hloaded Hpc Hz Hwf Htyped Hstack Hlocals.
  - destruct rows; [ | discriminate]. destruct Hrun as [_ [_ Hreplay]]. inversion Hreplay; subst.
    exists z, values. split.
    + apply Steps__refl. apply stack_config_wf; [exact Hz | | constructor].
      pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues. apply Forall_app in Hvalues; tauto.
    + split; [exact Hz | ]. split; [exact Hstack | ]. split; [exact Hlocals | ].
      split; [repeat split; reflexivity | ]. split; [cbn; lia | ]. split; [reflexivity | cbn; f_equal; lia].
  - destruct rows as [ | row rows]; [discriminate | ].
    destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
    assert (Hlink : Follows continuityLinks previous row) by (inversion Hchain; subst; assumption).
    assert (Hchain' : Chain continuityLinks (row :: rows)) by (inversion Hchain; subst; assumption).
    assert (HpcBefore : row COL_PC_BEFORE = entry).
    { rewrite <- (Hlink COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)). exact Hpc. }
    destruct Hloaded as [Himage Hloaded]. rewrite <- HpcBefore in Himage.
    rewrite (follows_boundary _ _ Hlink) in Hstack, Hlocals.
    assert (Hsingleton : programTyped (length (stateLocals z)) (length values) [i]).
    { destruct i; cbn [programTyped] in *;
        repeat match goal with H : _ /\ _ |- _ => destruct H end;
        repeat match goal with |- context [match ?x with Some _ => _ | None => _ end] =>
          destruct x; try contradiction end; auto. }
    destruct (framed_instruction prefix values i z initial middle row Hprime Hfull (Forall_inv Hprogram)
      Hreplay Himage Hz (Forall_inv Hwf) Hsingleton Hstack Hlocals)
      as [z1 [values1 [[Hstep [Hz1 [Hstack1 [Hlocals1 [Hsame Hheight]]]]] [Hpc1 Hregisters]]]].
    assert (Htyped1 : programTyped (length (stateLocals z1)) (length values1) rest).
    { rewrite (proj2 (proj2 Hsame)). eapply program_typed_tail; eassumption. }
    assert (Hloaded1 : loadedBody middle (entry + 1) rest).
    { eapply loaded_body_same_roms; [ | exact Hloaded].
      intros memory Hro address. eapply replay_preserves; [exact Hreplay | apply read_only_row; exact Hro]. }
    destruct (IH rows row prefix values1 middle final (entry + 1) z1 Hprime Htail Hchain'
      (Forall_inv_tail Hprogram) ltac:(cbn in Hlength; lia) Hloaded1 ltac:(lia)
      Hz1 (Forall_inv_tail Hwf) Htyped1 Hstack1 Hlocals1)
      as [z2 [values2 [Hsteps [Hz2 [Hstack2 [Hlocals2 [Hsame2 [Hpc2 [Hdepth2 Hdelta2]]]]]]]]].
    assert (Hv : Forall (fun v => 0 <= v < i32Modulus) values).
    { pose proof (proj1 (proj2 (proj2 Hstack))) as H. apply Forall_app in H; tauto. }
    assert (Hv1 : Forall (fun v => 0 <= v < i32Modulus) values1).
    { pose proof (proj1 (proj2 (proj2 Hstack1))) as H. apply Forall_app in H; tauto. }
    pose proof (one_spec_step _ _ ltac:(apply stack_config_wf; [exact Hz | exact Hv | constructor; [exact (Forall_inv Hwf) | constructor]])
      ltac:(apply stack_config_wf; [exact Hz1 | exact Hv1 | constructor]) Hstep) as Hone.
    pose proof (stack_suffix_steps z z1 values values1 [i] rest Hone (Forall_inv_tail Hwf)) as Hfirst.
    change (Steps (stackConfig z values (i :: rest)) (stackConfig z1 values1 rest)) in Hfirst.
    exists z2, values2. split; [eapply spec_steps_transitive; eassumption | ].
    split; [exact Hz2 | ]. split; [exact Hstack2 | ]. split; [exact Hlocals2 | ]. split.
    + destruct Hsame as [Hs [Ha Hl]], Hsame2 as [Hs2 [Ha2 Hl2]]. repeat split; congruence.
    + split; [cbn [finalRow length]; rewrite Nat2Z.inj_succ; lia | ]. split.
      * pose proof (Hlink COL_CALL_STACK_DEPTH_AFTER COL_CALL_STACK_DEPTH_BEFORE ltac:(vm_compute; tauto)).
        destruct Hregisters as [Hd _]. cbn [finalRow]. lia.
      * cbn [bodyStackDelta]. rewrite (instruction_stack_height_delta _ _ _ Hheight), Hdelta2. f_equal; lia.
Qed.
