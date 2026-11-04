(* A terminal row either traps through the activation or returns its captured
   result. The terminal opcode and the empty call stack on return are derived. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range FullRelation MemoryTrace MemoryCatalog Wasm2
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec ProgramSpec Fragment ControlDecode ControlInvariant
  FrameFacts FrameRows CallRows ControlBridge CallInitialization TrapRows FinishRows
  TerminalClassification TerminalFacts FunctionResults ExecutionSpec TableState OpcodeCatalog BlockEncoding
  ProgramBinding Selectors IndirectBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma terminal_control_row : forall program s baseDepth cursor root f initial final row inner caller,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  fieldPrime -> programLoaded program initial -> tablesLoaded s initial -> FullRelation.satFull row ->
  Replay initial (eventsForRow memories row) final -> terminalRow row -> row COL_HOST_FRAME_DEPTH_BEFORE = baseDepth -> wf_frame caller ->
  functionAt program root = Some f -> cursorRoot cursor = root ->
  ControlState program s baseDepth cursor initial (beforeBoundary row) inner ->
  exists outcome c1, Steps (callContext caller (N.of_nat (guestResults f)) inner) c1 /\
    TerminalConfig s caller row outcome c1.
Proof.
  intros program s baseDepth [stack position] root base initial final row inner caller Hstore Hwf Hvalid Hprime
    Hloaded Htables Hfull Hreplay [Hprogram Hfinish] Hhost Hcaller Hroot Hindex
    [f [active [Hf [Hstate [Hsusp [Hbase0 [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in Hf, Hstate, Hsusp, Hdepth.
  cbn [beforeBoundary boundaryCallDepth boundaryLocalsBase] in Hsusp, Hdepth.
  destruct position as [view | view callee | function arguments].
  2: {
    (* A resolved call's invocation row continues: its edge is dynamic. *)
    exfalso. cbn [activeFunction] in Hf.
    destruct Hstate as [f' [g [i [Hframe [Hi [Hind [_ [_ [_ [_ [Hpc _]]]]]]]]]]].
    pose proof Hframe as [Hf' [Hpath _]]. rewrite Hf in Hf'. injection Hf' as <-.
    destruct i; try discriminate Hind.
    destruct (encoded_path_invoke _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as [Hop [Hkind _]].
    unfold pathPC in Hop, Hkind. cbn [beforeBoundary boundaryPC] in Hpc. rewrite <- Hpc in Hop, Hkind.
    pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
    rewrite Hop in Hfetch. injection Hfetch as Hcode.
    pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as Hedge.
    rewrite Hkind in Hedge. injection Hedge as Hedge.
    assert (Hsel : row COL_SEL_CALL_INDIRECT = 1).
    { apply (full_opcode_selects row COL_SEL_CALL_INDIRECT CODE_CALL_INDIRECT Hprime Hfull Hprogram);
        [vm_compute; reflexivity | symmetry; exact Hcode]. }
    pose proof (full_invocation_continues row Hprime Hfull Hsel (eq_sym Hedge)). lia. }
  2: {
    destruct Hstate as [g [_ [_ [_ [_ [[Hmove Hzflag] [Hremaining [Hzero _]]]]]]]].
    cbn [beforeBoundary boundaryMoveRemaining boundaryZeroRemaining boundaryZeroActive] in Hremaining, Hzflag, Hzero.
    rewrite Hremaining in Hzflag. cbn in Hzflag.
    pose proof (full_row_kind_exclusive row COL_IS_PROGRAM_ROW COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram) as HzeroOff.
    destruct (row COL_LOCAL_ZERO_REMAINING_BEFORE =? 0) eqn:Htest;
      [apply Z.eqb_eq in Htest; lia | lia]. }
  cbn [activeFunction] in Hf.
  assert (HfinishReturn : exitsFunction f view -> exists outcome c1,
      Steps (callContext caller (N.of_nat (guestResults base)) (plugCalls stack active)) c1 /\
      TerminalConfig s caller row outcome c1).
  { intros Hleaves.
    destruct (finish_function_row program s view f initial final row active caller Hstore Hwf Hvalid Hprime
      Hloaded Hfull Hprogram Hfinish Hreplay Hcaller Hstate Hf Hleaves)
      as [result [Htrap [Hcapture Hsteps]]].
    destruct (full_clean_finish row Hprime Hfull Hprogram Hfinish Htrap) as [_ [HsameDepth _]].
    assert (Hnil : stack = []) by (destruct stack; [reflexivity | cbn in Hdepth; lia]). subst stack.
    change (positionFunction (viewPosition view) = root) in Hindex. rewrite Hindex, Hroot in Hf.
    injection Hf as <-.
    exists (returned result), (mk_config (mk_state s caller)
      (match result with None => [] | Some value => [i32Constant value] end)).
    split; [exact Hsteps | ]. split; [exact Hfinish | ]. split; [exact Htrap | ]. split; [exact Hcapture | reflexivity]. }
  destruct (nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view))) as [i | ] eqn:Hi.
  - destruct (terminal_instruction program s view f i initial final row active Hstore Hvalid Hprime Hloaded Hfull
      Hprogram Hfinish Hreplay Hstate Hf Hi) as [-> | [[x [y ->]] | Hreturns]].
    + destruct (unreachable_row program s stack view f initial final row active caller (N.of_nat (guestResults base))
        Hstore Hwf Hvalid Hprime Hloaded Hfull Hprogram Hreplay Hcaller Hstate Hf Hi Hsusp)
        as [Htrap [_ Hsteps]].
      exists trapped, (mk_config (mk_state s caller) [admininstr_TRAP]).
      split; [exact Hsteps | ]. split; [exact Hfinish | ]. split; [exact Htrap | reflexivity].
    + destruct (resolve_trap_row program s stack view f x y initial final row active caller
        (N.of_nat (guestResults base)) Hstore Hwf Hvalid Hprime Hloaded Htables Hfull Hprogram Hfinish Hreplay
        Hcaller Hstate Hf Hi Hsusp) as [Htrap Hsteps].
      exists trapped, (mk_config (mk_state s caller) [admininstr_TRAP]).
      split; [exact Hsteps | ]. split; [exact Hfinish | ]. split; [exact Htrap | reflexivity].
    + apply HfinishReturn. right; eauto.
  - apply nth_error_None in Hi.
    pose proof Hstate as [g [Hframe [Hoffset _]]].
    pose proof Hframe as [Hfg _]. rewrite Hf in Hfg. injection Hfg as <-.
    assert (Hend : positionOffset (viewPosition view) = length (innerCode f (viewPosition view))).
    { lia. }
    pose proof (terminal_end_is_function_end program s view f initial final row active Hstore Hvalid Hprime Hloaded Hfull
      Hprogram Hfinish Hreplay Hstate Hf Hend) as Hlabels.
    apply HfinishReturn. left. split; [exact Hlabels | ]. unfold innerCode in Hend. rewrite Hlabels in Hend. exact Hend.
Qed.
