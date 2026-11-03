(* An UNREACHABLE row traps through every active label and suspended call. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog Wasm2 WasmIntegers
  StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches CallerSpec ProgramSpec ProgramEncoding
  CodeLayout BlockEncoding Fragment ControlDecode ControlInvariant FrameFacts FrameRows CallRows
  ProgramBinding OpcodeCatalog Selectors TrapFacts SpecTecTraps.
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
      - apply steps_in_labels; [exact Hlabels | ].
        apply steps_with_stack_prefix; [exact Hvalues | ]. apply steps_with_instruction_suffix; [exact Hcode | ].
        apply call_context_traps; assumption.
      - eapply spec_steps_transitive.
        + apply steps_in_labels; [exact Hlabels | ].
          cbn [stackPrefix instructionSuffix]. rewrite app_assoc.
          apply trap_discards_context; assumption.
        + apply wrap_labels_trap; assumption. }
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
      (frameConfig f view (stackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
        (skipn (positionOffset (viewPosition view)) (innerCode f (viewPosition view)))))
      (mk_config (mk_state s (activeFrame f (viewLocals view))) [admininstr_TRAP])).
  { unfold frameConfig. rewrite (skipn_at_instruction _ _ _ Hi).
    eapply spec_steps_transitive.
    - apply steps_in_labels; [exact Hready | ]. apply unreachable_steps; [exact Hz | exact Hvalues | ].
      apply forall_skipn; exact HcodeWf.
    - apply wrap_labels_trap; assumption. }
  pose proof (suspended_stack_ready _ _ Hstore Hwf Hvalid _ _ _ _ _ Hsusp) as HreadyCalls.
  eapply spec_steps_transitive.
  - apply steps_in_call_frame; [exact Hcaller | ]. exact (steps_in_plug_calls stack _ _ HreadyCalls Hinner).
  - apply plug_calls_trap; [exact HreadyCalls | constructor; assumption | inversion Hz; assumption].
Qed.
