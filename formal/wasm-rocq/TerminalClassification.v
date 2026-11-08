(* The ROM prevents a terminal row from masquerading as an ordinary instruction. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog Wasm2 StraightLineSpec StackSimulation CallerSpec
  ProgramSpec ProgramEncoding CodeLayout BlockEncoding BlockSpec Fragment CodeFacts ControlDecode ControlInvariant
  FrameFacts FrameRows OpcodeCatalog ProgramBinding Selectors TrapFacts TerminalFacts FunctionResults.
Import ListNotations.
Open Scope Z_scope.

Lemma supported_opcode_exists : forall callees i program f pc targets,
  Supported callees i ->
  instructionAtPC program pc = Some (f, Some (flatInstruction (opInstr i) targets)) ->
  exists opcode, instructionRom program programOpcodeId pc 0 = Some opcode.
Proof.
  intros callees i program f pc targets Hsup Hslot. unfold instructionRom. rewrite Hslot.
  inversion Hsup; subst; destruct targets as [ | [target | ] [ | next rest]]; eexists; reflexivity.
Qed.

Lemma source_terminal_instruction : forall callees i program f pc targets offset height,
  Supported callees i -> instrBranches callees i height (length targets) ->
  instructionAtPC program pc = Some (f, Some (flatInstruction (opInstr i) (slotTargets targets offset i))) ->
  romValue (instructionRom program pcEdgeKindId pc 0) <> 0 ->
  i = UNREACHABLE \/ returnsFrom (length targets) i.
Proof.
  intros callees i program f pc targets offset height Hsup Hbranches Hslot Hkind.
  unfold instructionRom in Hkind. rewrite Hslot in Hkind.
  inversion Hsup; subst; cbn [slotTargets branchTargets map] in Hkind;
    try solve [exfalso; apply Hkind; reflexivity];
    try solve [left; reflexivity]; try solve [right; left; reflexivity].
  destruct (nth_error targets (N.to_nat (l :> N))) eqn:Htarget;
    [exfalso; apply Hkind; reflexivity | ].
  right; right. exists l. split; [reflexivity | ]. apply nth_error_None in Htarget.
  cbn [instrBranches] in Hbranches. lia.
Qed.

Lemma terminal_instruction : forall program s view f i initial final row inner,
  store_FUNCS s = program -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 1 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  i = UNREACHABLE \/ (exists x y, i = CALL_INDIRECT x y) \/
    returnsFrom (length (positionLabels (viewPosition view))) i.
Proof.
  intros program s view f i initial final row inner Hstore Hvalid Hprime Hloaded Hfull Hprogram Hfinish Hreplay
    Hstate Hf Hi.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights _]]]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [[Hsup Hbranches] _]].
  rewrite (skipn_at_instruction _ _ _ Hi) in Hsup, Hbranches.
  pose proof (Forall_inv Hsup) as Hsupported. destruct Hbranches as [Hbranches _].
  pose proof (encoded_path_head _ _ _ Hf _ _ _ _ Hpath Hi) as Hslot.
  unfold pathPC in Hslot. cbn [beforeBoundary boundaryPC] in Hpc. rewrite <- Hpc in Hslot.
  destruct (supported_opcode_exists _ _ _ _ _ _ Hsupported Hslot) as [opcode Hop].
  assert (Hrom : initial programOpcodeId [row COL_PC_BEFORE] = Some opcode).
  { program_cell Hloaded. unfold encodeProgram. cbn [Nat.eqb programOpcodeId pcRomId pcFunctionRefId
      functionEntryId functionLocalCountId functionCallMetadataId orb]. rewrite Hop. reflexivity. }
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hrom in Hfetch. injection Hfetch as Hcode.
  destruct (supported_opcode_trap_classification _ _ _ _ _ _ _ Hsupported Hslot Hop)
    as [[-> _] | [[x [y [-> _]]] | [chosen [Hin Hcodes]]]]; [left; reflexivity | right; left; eauto | ].
  enough (Hcases : i = UNREACHABLE \/ returnsFrom (length (positionLabels (viewPosition view))) i)
    by (destruct Hcases; [left | right; right]; assumption).
  pose proof (full_opcode_selects_member row chosen opcode Hprime Hfull Hprogram Hcodes (eq_sym Hcode)) as Hsel.
  pose proof (full_nontrapping_finish_edge row chosen Hprime Hfull Hin Hsel Hfinish) as Hedge.
  assert (HromKind : initial pcEdgeKindId [row COL_PC_BEFORE] =
      Some (romValue (instructionRom program pcEdgeKindId (row COL_PC_BEFORE) 0))).
  { program_cell Hloaded. reflexivity. }
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hkind.
  rewrite HromKind in Hkind. injection Hkind as Hkind.
  assert (Hlength : length (pathTargets (guestBody f) (positionLabels (viewPosition view))) =
      length (positionLabels (viewPosition view))) by (eapply path_targets_length; exact Hpath).
  rewrite <- Hlength. eapply source_terminal_instruction; [exact Hsupported | | exact Hslot | ].
  - rewrite Hlength. pose proof (path_heights_length _ _ _ _ _ _ Hheights) as Hlen.
    rewrite length_map in Hlen. rewrite <- Hlen. exact Hbranches.
  - rewrite Hkind, Hedge. pose proof (proj1 (value_bound row COL_CALL_STACK_POP_PRESENT)); lia.
Qed.

Lemma terminal_end_is_function_end : forall program s view f initial final row inner,
  store_FUNCS s = program -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 1 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionOffset (viewPosition view) = length (innerCode f (viewPosition view)) ->
  positionLabels (viewPosition view) = [].
Proof.
  intros program s view f initial final row inner Hstore Hvalid Hprime Hloaded Hfull Hprogram Hfinish Hreplay
    Hstate Hf Hend.
  destruct (positionLabels (viewPosition view)) as [ | site rest] eqn:Hlabels; [reflexivity | ].
  exfalso.
  assert (Hne : positionLabels (viewPosition view) <> []) by (rewrite Hlabels; discriminate).
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' _]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in *.
  destruct (innermost_label program _ _ _ _ _ _ _ _ _ _ _ _ _ Hframe Hne)
    as [labels' [site' [segments' [segment [outer [i [height [Hlabels' [Hsegments' [Hlen' [Hpath' [Hs Hrest]]]]]]]]]]]].
  rewrite Hlabels' in Hend, Hpc, Hs.
  pose proof (encoded_path_arm_end _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath' Hs) as Himage.
  unfold pathPC in Himage. rewrite <- Hend in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (labels' ++ [site']) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct Himage as [Hop [Hkind _]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindFetch.
  rewrite Hop in Hfetch. injection Hfetch as Hcode.
  rewrite Hkind in HkindFetch. injection HkindFetch as Hzero.
  assert (Hmarker : armEndOpcode i (siteArm site') = CODE_END \/ armEndOpcode i (siteArm site') = CODE_ELSE).
  { unfold armEndOpcode. destruct i; try (left; reflexivity).
    lazymatch goal with |- context [match ?e with [] => _ | _ :: _ => _ end] => destruct e end;
      [left; reflexivity | ]. destruct (siteArm site'); [right | left]; reflexivity. }
  assert (Hchoose : exists chosen, In chosen nonTrappingSelectors /\ row chosen = 1).
  { destruct Hmarker as [HendCode | HelseCode].
    - exists COL_SEL_END. split; [cbn; tauto | ]. apply (full_opcode_selects row COL_SEL_END CODE_END Hprime Hfull Hprogram);
        [vm_compute; reflexivity | congruence].
    - exists COL_SEL_ELSE. split; [cbn; tauto | ]. apply (full_opcode_selects row COL_SEL_ELSE CODE_ELSE Hprime Hfull Hprogram);
        [vm_compute; reflexivity | congruence]. }
  destruct Hchoose as [chosen [Hin Hsel]].
  pose proof (full_nontrapping_finish_edge row chosen Hprime Hfull Hin Hsel Hfinish) as Hedge.
  pose proof (proj1 (value_bound row COL_CALL_STACK_POP_PRESENT)); lia.
Qed.
