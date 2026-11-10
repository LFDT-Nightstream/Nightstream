(* An UNREACHABLE row traps through every active label and suspended call. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog Wasm2 WasmIntegers
  StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches CallerSpec ProgramSpec ProgramEncoding
  CodeLayout BlockEncoding Fragment ControlDecode ControlInvariant FrameFacts FrameRows CallRows
  ProgramBinding OpcodeCatalog Selectors TrapFacts SpecTecTraps TableState BlockSpec BlockTrace IndirectBindings
  SpecTecIndirect FunctionValidation IndirectRows.
Require Import WasmValues IntegerMemory IntegerDecode.
Import ListNotations.
Open Scope Z_scope.

(* A trap consumes every suspended caller, including its saved operands and
   continuation. The external caller of the activation remains. *)
Lemma plug_calls_trap : forall stack s caller callee arity,
  Forall SuspendedReady stack -> wf_state (mk_state s caller) -> wf_frame callee ->
  Steps (callContext caller arity (plugCalls stack (mk_config (mk_state s callee) [admininstr_TRAP])))
    (mk_config (mk_state s caller) [admininstr_TRAP]).
Proof.
  induction stack as [ | top rest IH]; intros s caller callee arity Hready Hz Hcallee.
  - apply call_context_traps; assumption.
  - inversion Hready as [ | ? ? [Hlabels [Hvalues [Hcode Hframe]]] Hrest]; subst.
    set (parent := activeFrame (callerFunction top) (viewLocals (callView top))).
    assert (Hparent : wf_state (mk_state s parent)) by (inversion Hz; constructor; assumption).
    assert (Hstep : Steps (callerConfig top (mk_config (mk_state s callee) [admininstr_TRAP]))
        (mk_config (mk_state s parent) [admininstr_TRAP])).
    { unfold callerConfig, frameConfig.
      eapply spec_steps_transitive.
      - apply steps_in_value_labels; [exact Hlabels | ].
        apply steps_with_value_prefix; [exact Hvalues | ]. apply steps_with_instruction_suffix; [exact Hcode | ].
        apply call_context_traps; assumption.
      - eapply spec_steps_transitive.
        + apply steps_in_value_labels; [exact Hlabels | ].
          cbn [valueStackPrefix instructionSuffix]. rewrite app_assoc.
          apply value_trap_discards_context; assumption.
        + apply wrap_value_labels_trap; assumption. }
    cbn [plugCalls]. eapply spec_steps_transitive.
    + apply steps_in_call_frame; [inversion Hz; assumption | ]. exact (steps_in_plug_calls rest _ _ Hrest Hstep).
    + apply IH; assumption.
Qed.

Lemma unreachable_row : forall program s stack view f initial final row inner caller arity,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> wf_frame caller ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some UNREACHABLE ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) stack ->
  row COL_TRAPPED_AFTER = 1 /\ row COL_ACTIVATION_FINISHED_AFTER = 1 /\
  Steps (callContext caller arity (plugCalls stack inner)) (mk_config (mk_state s caller) [admininstr_TRAP]).
Proof.
  intros program s stack view f initial final row inner caller arity Hstore Hwf Hvalid Hprime Hloaded
    Hfull Hprogram Hreplay Hcaller Hstate Hf Hi Hsusp.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [_ [_ [_ [Hbounds _]]]]]]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct (encoded_path_unreachable _ _ _ _ Hloaded Hf _ _ _ Hpath Hi) as [Hop _].
  cbn [beforeBoundary boundaryPC] in Hpc. unfold pathPC in Hop. rewrite <- Hpc in Hop.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch. injection Hfetch as Hcode.
  assert (Hsel : row COL_SEL_UNREACHABLE = 1).
  { apply (full_opcode_selects row COL_SEL_UNREACHABLE CODE_UNREACHABLE Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact Hcode]. }
  destruct (full_unreachable_traps row Hprime Hfull Hsel) as [Htrap Hfinish].
  split; [exact Htrap | ]. split; [exact Hfinish | ].
  destruct (frame_ready _ _ Hstore Hwf Hvalid _ _ _ _ _ _ _ _ Hframe) as [Hready [Hvalues [HcodeWf Hz]]].
  assert (Hinner : Steps
      (frameConfig f view (valueStackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
        (skipn (positionOffset (viewPosition view)) (innerCode f (viewPosition view)))))
      (mk_config (mk_state s (activeFrame f (viewLocals view))) [admininstr_TRAP])).
  { unfold frameConfig. rewrite (skipn_at_instruction _ _ _ Hi).
    eapply spec_steps_transitive.
    - apply steps_in_value_labels; [exact Hready | ]. apply value_unreachable_steps; [exact Hz | exact Hvalues | ].
      apply forall_skipn; exact HcodeWf.
    - apply wrap_value_labels_trap; assumption. }
  pose proof (suspended_stack_ready _ _ Hstore Hwf Hvalid _ _ _ _ _ Hsusp) as HreadyCalls.
  eapply spec_steps_transitive.
  - apply steps_in_call_frame; [exact Hcaller | ]. exact (steps_in_plug_calls stack _ _ HreadyCalls Hinner).
  - apply plug_calls_trap; [exact HreadyCalls | constructor; assumption | inversion Hz; assumption].
Qed.

(* A finishing resolve row traps, and so does SpecTec: had the table held a
   function of the expected type at the index, every check would have passed. *)
Lemma resolve_trap_row : forall program s stack view f x y initial final row inner caller arity,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> tablesLoaded s initial -> FullRelation.satFull row ->
  row COL_IS_PROGRAM_ROW = 1 -> row COL_ACTIVATION_FINISHED_AFTER = 1 ->
  Replay initial (eventsForRow memories row) final -> wf_frame caller ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (CALL_INDIRECT x y) ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) stack ->
  row COL_TRAPPED_AFTER = 1 /\
  Steps (callContext caller arity (plugCalls stack inner)) (mk_config (mk_state s caller) [admininstr_TRAP]).
Proof.
  intros program s stack view f x y initial final row inner caller arity Hstore Hwf Hvalid Hprime Hloaded Htables
    Hfull Hprogram Hfinish Hreplay Hcaller Hstate Hf Hi Hsusp.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [Hpc [_ ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [_ [Hfbp0 [Hbase0 [Hbounds _]]]]]]. rewrite Hf in Hf'. injection Hf' as <-.
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hi, Hpath, Hbounds, Hcells, Hsp, Hpc |- *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] _]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup.
  destruct Hfacts as [[Hwi [Htable [params [results [Htype Hheight]]]]] _]. rewrite Nat.add_0_r in Hheight.
  inversion Hsup as [ | ? ? Hhead _].
  inversion Hhead as [ | | | | | | | | | | | | x' y' p r Htype' Hp Hr Hinteger Hready_ | | ].
  cbn [programCallees envTypes envTables indirectReady] in Htype, Htable, Hready_.
  destruct (split_top_values values 1 ltac:(lia)) as [prefix [top [-> Htop]]].
  destruct top as [ | tableIndex [ | ? ?]]; try discriminate. clear Htop.
  destruct (frame_condition_value _ _ _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe Hi
    eq_refl eq_refl) as [tableIndexRaw [HindexRange ->]].
  destruct (nth_error (TABLES (guestModule f)) (N.to_nat (x :> N))) as [ta | ] eqn:Hta;
    [ | apply nth_error_None in Hta; lia].
  destruct (encoded_path_resolve _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as [Himage [HtableRom HtypeRom]].
  rewrite Hta in HtableRom. rewrite Htype in HtypeRom. cbn [option_map romValue] in HtableRom, HtypeRom.
  unfold pathPC in Himage, HtableRom, HtypeRom.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage, HtableRom, HtypeRom.
  pose proof (control_row_static_edge _ _ row _ _ _ Hprogram Hreplay Logic.eq_refl Himage) as Hedge.
  destruct (BlockTrace.control_row _ _ row _ CODE_CALL_INDIRECT_RESOLVE COL_SEL_CALL_INDIRECT_RESOLVE _ Hprime Hfull
    Hprogram Hreplay Logic.eq_refl ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage)
    as [Hresolve _].
  pose proof (full_resolve_finish row Hprime Hfull Hresolve Hedge) as HfinishTrap. rewrite Hfinish in HfinishTrap.
  destruct (full_resolve_trap_finishes row Hprime Hfull Hresolve (eq_sym HfinishTrap)) as [Htrapped _].
  pose proof HfinishTrap as Htrapped'.
  split; [exact Htrapped | ].
  (* The index on top, and the checks' verdict. *)
  assert (HspPositive : 1 <= row COL_SP_BEFORE).
  { change (row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE +
      Z.of_nat (length (concat segments ++ prefix ++ [i32Value tableIndexRaw]))) in Hsp.
    rewrite !length_app in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_resolve_stack_shape row Hprime Hfull Hresolve HspPositive) as Hshape.
  destruct (replace_stack_replay row initial final Hshape Hreplay) as [Hread _].
  assert (Htop : initial stackId [2 * (row COL_SP_BEFORE - 1)] = Some tableIndexRaw).
  { change (row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE +
      Z.of_nat (length (concat segments ++ prefix ++ [i32Value tableIndexRaw]))) in Hsp.
    replace (row COL_SP_BEFORE - 1) with
      (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat segments ++ prefix)))
      by (rewrite Hsp, !length_app; cbn [length]; lia).
    apply cell_matches_i32_read; [exact (proj1 HindexRange) | ].
    apply Hcells. unfold viewOperands. cbn [viewSegments viewValues].
    rewrite app_assoc, nth_error_app2, Nat.sub_diag by lia. reflexivity. }
  rewrite (replace_read_address _ Hshape), Htop in Hread. injection Hread as HindexValue.
  destruct (full_resolve_checks row Hprime Hfull Hresolve) as [HidActive [HimmActive [HsizeActive [HindexCol
    [Hoob [HreadActive [Hnull [Heq [Hgate [HrefIs [Htrap _]]]]]]]]]]].
  destruct (resolve_rom_reads _ _ _ Hreplay HidActive HimmActive HsizeActive) as [HidRead [HexpRead HsizeRead]].
  rewrite HtableRom in HidRead. injection HidRead as HtableId.
  rewrite HtypeRom in HexpRead. injection HexpRead as Hexpected.
  assert (Hindex : row COL_TABLE_INDEX = tableIndexRaw) by (rewrite HindexCol; symmetry; exact HindexValue).
  assert (Hu32 : 0 <= tableIndexRaw < i32Modulus).
  { pose proof (full_read0_u32 row Hprime Hfull). normalize_columns. unfold i32Modulus. lia. }
  set (z := mk_state s (activeFrame f locals)).
  (* No function of the expected type sits at the index: otherwise no check fails. *)
  assert (Hnone : forall a, ~ indirectTarget z x y tableIndexRaw a).
  { intros a [Hentry [fi [Hfi HfiType]]].
    change (nth_error (REFS (lookup_total (store_TABLES s) (lookup_total (TABLES (guestModule f)) (x :> N))))
      (Z.to_nat tableIndexRaw) = Some (REF_FUNC_ADDR a)) in Hentry.
    rewrite (spec_lookup_from_nth_error _ _ _ _ _ Hta) in Hentry.
    destruct (function_module_ok program s Hvalid _ _ Hf) as [C HC].
    pose proof (module_table_in_store _ _ _ _ _ HC Hta) as HtaStore.
    destruct (nth_error (store_TABLES s) (N.to_nat ta)) as [t | ] eqn:Ht; [ | apply nth_error_None in Ht; lia].
    rewrite (spec_lookup_from_nth_error _ _ _ _ _ Ht) in Hentry.
    destruct (Htables _ _ Ht) as [Hsize Hentries].
    replace (Z.of_nat (N.to_nat ta)) with (Z.of_N ta) in Hsize, Hentries by lia.
    rewrite HtableId, HsizeRead in Hsize. injection Hsize as Hsize.
    assert (Hbound : (Z.to_nat tableIndexRaw < length (REFS t))%nat)
      by (apply nth_error_Some; rewrite Hentry; discriminate).
    assert (HoobZero : row COL_CI_OOB = 0)
      by (rewrite Hoob, (proj2 (Z.ltb_lt _ _)); [reflexivity | lia]).
    assert (HsetOff : row COL_SEL_TABLE_SET <> 1).
    { assert (Hin' : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
      rewrite (full_opcode_selector_exclusive row _ COL_SEL_TABLE_SET Hprime Hfull Hresolve Hin'
        ltac:(vm_compute; tauto) ltac:(discriminate)). discriminate. }
    pose proof (resolve_entry_read _ _ _ Hfull Hreplay HsetOff ltac:(rewrite HreadActive, HoobZero; reflexivity)) as HentryRead.
    pose proof (Hentries _ _ Hentry) as HentryStore. rewrite Z2Nat.id in HentryStore by lia.
    rewrite HtableId, <- Hindex, HentryRead in HentryStore. cbn in HentryStore. injection HentryStore as Hvalue.
    assert (HnullZero : row COL_CI_ENTRY_IS_NULL = 0)
      by (rewrite Hnull, (proj2 (Z.eqb_neq _ _)); [reflexivity | lia]).
    assert (HgateOne : row COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 1) by (rewrite Hgate, HoobZero, HnullZero; reflexivity).
    pose proof (HrefIs HgateOne) as Hreference. rewrite Hvalue in Hreference.
    pose proof (resolve_type_read _ _ _ Hreplay HgateOne) as HtypeRead.
    change (nth_error (store_FUNCS s) (N.to_nat a) = Some fi) in Hfi. rewrite Hstore in Hfi.
    destruct (Hready_ _ _ Hfi) as [g [Hg Hmodule]].
    pose proof (function_type_rom _ _ _ _ Hloaded Hg) as HtypeRom'.
    replace (Z.of_nat (N.to_nat a) + 1) with (row COL_FUNCTION_REF) in HtypeRom' by lia.
    rewrite HtypeRom' in HtypeRead. injection HtypeRead as HtypeId.
    destruct (function_at_typed _ _ _ Hg) as [Hgnth _]. rewrite Hgnth in Hfi. injection Hfi as <-.
    change (guestType g =
      lookup_total (TYPES (guestModule f)) (y :> N)) in HfiType.
    rewrite (spec_lookup_from_nth_error _ _ _ _ _ Htype) in HfiType.
    assert (HeqOne : row COL_CI_TYPE_EQ = 1).
    { rewrite Heq, (proj2 (Z.eqb_eq _ _)); [reflexivity | ].
      rewrite <- HtypeId, <- Hexpected, Hmodule, HfiType. reflexivity. }
    rewrite Htrap, HgateOne, HeqOne in Htrapped'. lia. }
  (* SpecTec traps at the call, then through the frame's labels and the callers. *)
  destruct (frame_ready _ _ Hstore Hwf Hvalid _ _ _ _ _ _ _ _ Hframe) as [Hready [Hvalues [HcodeWf Hz]]].
  cbn [viewPosition viewSegments viewValues viewLocals positionLabels] in Hready, Hvalues, HcodeWf, Hz.
  apply Forall_app in Hvalues. destruct Hvalues as [HprefixBounds _].
  assert (Hrest : Forall wf_instr (skipn (S offset) code)) by (apply forall_skipn; exact HcodeWf).
  assert (Hinner : Steps
      (frameConfig f (frameView (codePosition index labels offset) segments (prefix ++ [i32Value tableIndexRaw]) locals)
        (valueStackConfig z (prefix ++ [i32Value tableIndexRaw]) (skipn offset code)))
      (mk_config z [admininstr_TRAP])).
  { unfold frameConfig. cbn [viewPosition viewSegments positionLabels]. rewrite (skipn_at_instruction _ _ _ Hi).
    eapply spec_steps_transitive.
    - apply steps_in_value_labels; [exact Hready | ]. rewrite value_stack_config_split.
      apply steps_with_value_prefix; [exact HprefixBounds | ].
      apply steps_with_instruction_suffix; [exact Hrest | ].
      apply one_spec_step.
      + apply value_stack_config_wf; [exact Hz | constructor; [apply i32_value_wf; exact Hu32 | constructor] | constructor; [exact Hwi | constructor]].
      + apply trap_config_wf. exact Hz.
      + exact (indirect_call_traps z x y tableIndexRaw Hu32 Hnone).
    - eapply spec_steps_transitive with (b := wrapValueLabels (guestBody f) labels segments (mk_config z [admininstr_TRAP])).
      + apply steps_in_value_labels; [exact Hready | ].
        cbn [valueStackPrefix instructionSuffix]. rewrite app_assoc.
        apply value_trap_discards_context; assumption.
      + apply wrap_value_labels_trap; assumption. }
  pose proof (suspended_stack_ready _ _ Hstore Hwf Hvalid _ _ _ _ _ Hsusp) as HreadyCalls.
  eapply spec_steps_transitive.
  - apply steps_in_call_frame; [exact Hcaller | ]. exact (steps_in_plug_calls stack _ _ HreadyCalls Hinner).
  - apply plug_calls_trap; [exact HreadyCalls | constructor; assumption | inversion Hz; assumption].
Qed.
