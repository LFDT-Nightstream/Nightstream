(* A base-frame return captures the source function's results and drops its frame. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec ProgramSpec ProgramEncoding CodeLayout BlockEncoding
  Fragment CodeFacts ControlDecode ControlInvariant ControlBridge FrameFacts FrameRows CallRows SlideLayout ProgramBinding
  OpcodeCatalog Selectors CircuitFacts CallContext TrapFacts TerminalFacts FunctionResults.
Require Import IntegerMemory IntegerDecode.
Import ListNotations.
Open Scope Z_scope.

Lemma exit_distance : forall program s view f initial final row inner,
  store_FUNCS s = program -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_HOST_EVENT_EXIT_LATCH = 1 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f -> exitsFunction f view ->
  (guestResults f <= length (viewValues view))%nat ->
  row COL_EXIT_SLIDE_DISTANCE = Z.of_nat (frameLocals f + (length (viewOperands view) - guestResults f)).
Proof.
  intros program s view f initial final row inner Hstore Hvalid Hprime Hloaded Hfull Hprogram Hexit Hreplay
    Hstate Hf Hleaves Hfits.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [_ [Hpc _]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights _]]]. rewrite Hf in Hf'. injection Hf' as <-.
  pose proof (circuit_row_exit_distance_fetch _ _ _ Hreplay Hexit) as Hfetch.
  cbn [beforeBoundary boundaryPC] in Hpc.
  destruct Hleaves as [[Hlabels Hend] | [i [Hi Hreturns]]].
  - destruct (function_end_selected _ _ view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay
      Hstate Hf Hlabels Hend) as [Hsel _].
    destruct (encoded_end_slide _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf) as [_ Hrom].
    rewrite Hlabels, Hend in Hpc. cbn [pathOffset] in Hpc. rewrite code_offset_end in Hpc.
    change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (codeSize (guestBody f))) in Hpc.
    rewrite <- Hpc, Hfetch in Hrom.
    rewrite (full_selected_choice_pinned row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) Hsel
      ltac:(vm_compute; intuition discriminate)) in Hrom. injection Hrom as Hdistance.
    destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [Hresults _]]]].
    specialize (Hresults Hlabels). unfold innerCode in Hresults. rewrite Hlabels in Hresults. cbn in Hresults.
    rewrite Hlabels in Hheights. cbn [pathHeights] in Hheights. rewrite Hend, firstn_all in Hheights.
    destruct (sourceHeights _ (guestBody f) 0) as [height | ] eqn:Hheight; [ | discriminate].
    rewrite (Hresults _ Hheight) in Hheights. cbn [option_map] in Hheights.
    injection Hheights as Hsegments Hlength.
    assert (Hempty : viewSegments view = []).
    { destruct (viewSegments view); [reflexivity | discriminate]. }
    rewrite Nat.add_0_r in Hlength.
    unfold viewOperands. rewrite Hempty. cbn. rewrite <- Hlength, Nat.sub_diag, Nat.add_0_r.
    unfold frameLocals. exact Hdistance.
  - destruct (return_selected _ _ view f i initial final row inner Hprime Hloaded Hfull Hprogram Hreplay
      Hstate Hf Hi Hreturns) as [Hsel _].
    destruct (encoded_return_slide _ _ _ _ _ _ _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf Hpath Hi Hreturns Hheights)
      as [_ Hrom].
    rewrite <- Hpc, Hfetch in Hrom.
    rewrite (full_selected_choice_pinned row COL_SEL_RETURN Hprime Hfull ltac:(vm_compute; tauto) Hsel
      ltac:(vm_compute; intuition discriminate)) in Hrom.
    rewrite Nat.add_0_r, (proj2 (Nat.leb_le _ _) Hfits) in Hrom. cbn iota in Hrom.
    change ((0 =? 0) && true) with true in Hrom. injection Hrom as Hdistance.
    destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [_ Htotal]. rewrite !Nat.add_0_r in Htotal.
    rewrite Htotal in Hdistance. unfold slideWord, frameLocals in *. exact Hdistance.
Qed.

Lemma finish_function_row : forall program s view f initial final row inner caller,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 1 -> Replay initial (eventsForRow memories row) final -> wf_frame caller ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f -> exitsFunction f view ->
  exists result, row COL_TRAPPED_AFTER = 0 /\
    (match result with None => row COL_OUTPUT_CAPTURED = 0
      | Some value => row COL_OUTPUT_CAPTURED = 1 /\ row COL_OUTPUT_VALUE_LO_AFTER = value end) /\
    Steps (callContext caller (N.of_nat (guestResults f)) inner)
      (mk_config (mk_state s caller) (match result with None => [] | Some v => [i32Constant v] end)).
Proof.
  intros program s view f initial final row inner caller Hstore Hwf Hvalid Hprime Hloaded Hfull Hprogram
    Hfinish Hreplay Hcaller Hstate Hf Hleaves.
  assert (Htrap : row COL_TRAPPED_AFTER = 0).
  { destruct Hleaves as [[Hl He] | [i [Hi Hr]]].
    - destruct (function_end_selected _ _ view f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay
        Hstate Hf Hl He) as [Hsel _]. eapply full_nontrapping_untrapped; [exact Hprime | exact Hfull | | exact Hsel]. cbn; tauto.
    - destruct (return_selected _ _ view f i initial final row inner Hprime Hloaded Hfull Hprogram Hreplay
        Hstate Hf Hi Hr) as [Hsel _]. eapply full_nontrapping_untrapped; [exact Hprime | exact Hfull | | exact Hsel]. cbn; tauto. }
  destruct (full_clean_finish row Hprime Hfull Hprogram Hfinish Htrap) as [Hexit [_ [Hcapture _]]].
  destruct (function_result_values program s view f initial row inner Hstore Hwf Hvalid Hprime Hfull Hprogram Hstate Hf Hleaves)
    as [dropped [returned [Hsplit [Hlength [Hfits Hsteps]]]]].
  pose proof (exit_distance program s view f initial final row inner Hstore Hvalid Hprime Hloaded Hfull Hprogram
    Hexit Hreplay Hstate Hf Hleaves Hfits) as Hdistance.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp _]]]].
  pose proof Hframe as [Hf' [_ [_ [Hfbp0 [Hbase [Hbounds _]]]]]]. rewrite Hf in Hf'. injection Hf' as <-.
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  cbn [beforeBoundary boundarySP boundaryStackBase boundaryLocalsBase] in Hsp, Hbase, Hfbp0, Hcells.
  rewrite Hsplit, length_app, <- Hlength, Nat.add_sub, Nat2Z.inj_add in Hdistance.
  rewrite Hsplit, length_app, Nat2Z.inj_add in Hsp.
  assert (Hcount : row COL_OUTPUT_CAPTURED = Z.of_nat (length returned)) by lia.
  pose proof (Hsteps caller ltac:(constructor; assumption)) as Hreturn.
  destruct returned as [ | v [ | v' rest]].
  - exists None. split; [exact Htrap | ]. split; [cbn in Hcount; exact Hcount | exact Hreturn].
  - assert (Hv : i32Operand v).
    { apply (proj1 (Forall_forall _ _) Hbounds). rewrite Hsplit.
      apply in_or_app; right. left. reflexivity. }
    destruct Hv as [raw [Hraw ->]].
    exists (Some raw). split; [exact Htrap | ]. split; [split | exact Hreturn].
    + exact Hcount.
    + assert (Hcap : row COL_OUTPUT_CAPTURED = 1) by exact Hcount.
      assert (Haddr := full_capture_address row Hprime Hfull Hcap ltac:(cbn in Hsp; lia)).
      pose proof (captured_stack_read _ _ _ Hreplay Hcap) as Hread.
      assert (Hvalue : initial stackId [2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length dropped))] = Some raw).
      { apply cell_matches_i32_read; [exact (proj1 Hraw) | ]. apply Hcells. rewrite Hsplit, nth_error_app2, Nat.sub_diag by lia. reflexivity. }
      rewrite Haddr in Hread. replace (2 * row COL_SP_BEFORE - 2)
        with (2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length dropped))) in Hread by (cbn in Hsp; lia).
      rewrite Hvalue in Hread. injection Hread as Hread.
      rewrite (full_capture_value row Hfull Hcap). symmetry; exact Hread.
  - exfalso.
    assert (Hboolean : row COL_OUTPUT_CAPTURED = 0 \/ row COL_OUTPUT_CAPTURED = 1).
    { circuit_boolean COL_OUTPUT_CAPTURED Hcap. exact Hcap. }
    cbn in Hcount. lia.
Qed.
