(* Real Rust traces through run_refinement, with an i64 value larger than the field modulus. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers WasmValues
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec FunctionValidation ProgramSpec ProgramEncoding CodeLayout
  ControlDecode ControlInvariant ControlBridge FrameFacts FrameRows IntegerMemory IntegerDecode
  ExecutionSpec ExecutionRefinement TableState
  tests.MixedPrograms tests.MixedCallTrace tests.MixedIndirectTrace tests.WitnessCheck.
Import ListNotations.
Open Scope Z_scope.

Definition wide := i64Value (joinLimbs 4294967295 4294967295).
Definition shape := controlShape [] (siteCode (codePosition 0 [] 0)).
Definition outerFrame module := MKframe [] module.

Lemma typed_start_config : forall program s index f locals memory boundary,
  store_FUNCS s = program -> wf_store s -> programValidated program s -> tablesLoaded s memory ->
  functionAt program index = Some f -> valuesTyped (guestFrameTypes f) locals ->
  (forall i v, nth_error locals i = Some v -> CellMatches memory (boundaryLocalsBase boundary + Z.of_nat i) v) ->
  0 <= boundaryLocalsBase boundary ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (frameLocals f) ->
  boundarySP boundary = boundaryStackBase boundary -> boundaryPC boundary = guestEntry f -> initializationIdle boundary ->
  0 <= boundaryCallDepth boundary ->
  CircuitConfig program s (boundaryCallDepth boundary) (controlShape [] (siteCode (codePosition index [] 0)))
    memory boundary (valueStackConfig (mk_state s (activeFrame f locals)) [] (guestBody f)).
Proof.
  intros program s index f locals memory boundary Hstore Hwf Hvalid Htables Hf Htypes Hcells Hfbp Hbase Hsp Hpc Hidle Hdepth.
  apply (control_state_circuit _ _ _ (controlCursor [] (atCode (frameView (codePosition index [] 0) [] [] locals))));
    [exact Hstore | exact Hwf | exact Hvalid | exact Htables | ].
  exists f, (valueStackConfig (mk_state s (activeFrame f locals)) [] (guestBody f)).
  split; [exact Hf | ]. split.
  - apply (FrameRows.idle_active_state program s memory boundary f (frameView (codePosition index [] 0) [] [] locals));
      try assumption; try (cbn; lia).
    + apply FrameRows.frame_at_unmoved_intro; unfold viewOperands;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset concat app];
        try assumption; try reflexivity.
      * constructor.
      * intros [ | i] v Hv; discriminate.
      * pose proof (Forall2_length Htypes) as Hlen. unfold guestFrameTypes in Hlen.
        rewrite length_app in Hlen. unfold frameLocals, guestParams, guestLocals. lia.
      * apply frame_typing_idle. reflexivity.
  - split; [exact I | ]. split; [exact Hdepth | ]. split; [cbn; lia | reflexivity].
Qed.

Lemma terminal_values : forall program root s caller depth memory boundary rows outcome c row f values,
  ExecutionConfig program s caller depth root memory boundary rows outcome c ->
  lastRow rows = Some row -> row COL_ACTIVATION_FINISHED_AFTER = 1 -> row COL_TRAPPED_AFTER = 0 ->
  functionAt program root = Some f -> capturedValues row (guestResultTypes f) = Some values ->
  c = mk_config (mk_state s caller) (map admininstr_val values).
Proof.
  intros program root s caller depth memory boundary rows outcome c row f values H Hlast Hfinish Htrap Hf Hvalues.
  destruct outcome as [shape' | | returned]; cbn [ExecutionConfig] in H.
  - destruct H as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct H as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
  - destruct H as [last [Hlast' [_ [_ [[g [Hg Hv]] ->]]]]].
    rewrite Hlast in Hlast'; injection Hlast' as <-. rewrite Hf in Hg; injection Hg as <-. congruence.
Qed.

Definition argumentMemory (condition : Z) : MemoryState := fun id address =>
  if Nat.eqb id stackId then
    match address with [0] | [1] => Some 4294967295 | [2] => Some condition | [3] | [4] | [5] => Some 0 | _ => None end
  else if Nat.eqb id exportEntryScheduleCountId then match address with [1] => Some 9 | _ => None end
  else if Nat.eqb id exportExitScheduleCountId then match address with [1] => Some 1 | _ => None end
  else None.

Module Calls.
Definition memory := loadProgram MixedCall.program (argumentMemory 1).
Definition rows := map rowWitness mixedCallRows.
Definition start := mkBoundary 0 3 3 0 0 0 0 0 0 0 0.
Definition initial := valueStackConfig
  (mk_state MixedCall.store (activeFrame MixedCall.guest [wide; i32Value 1; i64Value 0])) [] MixedCall.mainBody.
Definition caller := outerFrame MixedCall.module.

Lemma checks : validRunb memory rows = true.
Proof. vm_compute; reflexivity. Qed.
Lemma fragment : executionFragment start 0 rows.
Proof. vm_compute; intuition discriminate. Qed.
Lemma initial_config : CircuitConfig MixedCall.program MixedCall.store 0 shape memory start initial.
Proof.
  apply (typed_start_config MixedCall.program MixedCall.store 0 MixedCall.guest _ memory start);
    try reflexivity; try (apply no_tables_loaded; reflexivity); try (cbn; lia).
  - repeat constructor.
  - exact MixedCall.validated.
  - repeat first [apply Forall2_cons | apply Forall2_nil | split | reflexivity].
    + unfold wide. apply i64_value_wf; apply join_limbs_bound; unfold i32Modulus; lia.
    + apply i32_value_wf; unfold i32Modulus; lia.
    + apply i64_value_wf; unfold i64Modulus; lia.
  - intros [ | [ | [ | i]]] v Hv; cbn in Hv; [injection Hv as <- .. | destruct i; discriminate].
    + unfold wide. constructor; try (unfold i32Modulus; lia); reflexivity.
    + constructor; [unfold i32Modulus; lia | reflexivity].
    + change (CellMatches memory 2 (i64Value (joinLimbs 0 0))).
      constructor; try (unfold i32Modulus; lia); reflexivity.
  - repeat split.
Qed.

Lemma final_row_decodes : capturedValues (rowWitness (last mixedCallRows [])) [valtype_I64] = Some [wide].
Proof. vm_compute; reflexivity. Qed.

Example complete_mixed_call_refines : fieldPrime ->
  Steps (callContext caller 1 initial) (mk_config (mk_state MixedCall.store caller) [admininstr_val wide]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig MixedCall.program MixedCall.store caller 0 0 shape memory start (callContext caller 1 initial)).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ].
    exists MixedCall.guest, initial. split; [reflexivity | ]. split; [exact initial_config | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ checks) fragment Hstart)
    as [outcome [c [Hsteps [_ Hconfig]]]].
  assert (Hc : c = mk_config (mk_state MixedCall.store caller) [admininstr_val wide]).
  { eapply terminal_values with (row := rowWitness (last mixedCallRows [])) (f := MixedCall.guest) (values := [wide]);
      [exact Hconfig | reflexivity | reflexivity | reflexivity | reflexivity | exact final_row_decodes]. }
  rewrite <- Hc. exact Hsteps.
Qed.
End Calls.

Module Indirect.
Definition ram : MemoryState := fun id address =>
  if Nat.eqb id tableSizeId then match address with [0] => Some 1 | _ => None end
  else if Nat.eqb id tableElementId then match address with [0; 0] => Some 2 | _ => None end
  else argumentMemory 0 id address.
Definition memory := loadProgram MixedIndirect.program ram.
Definition rows := map rowWitness mixedIndirectRows.
Definition start := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.
Definition initial := valueStackConfig
  (mk_state MixedIndirect.store (activeFrame MixedIndirect.guest [wide; i32Value 0])) [] MixedIndirect.mainBody.
Definition caller := outerFrame MixedIndirect.module.

Lemma tables_loaded : tablesLoaded MixedIndirect.store memory.
Proof.
  intros [ | a] t Ht; cbn in Ht; [injection Ht as <- | destruct a; discriminate].
  split; [reflexivity | ]. intros [ | i] r Hr; cbn in Hr; [injection Hr as <- | destruct i; discriminate]. reflexivity.
Qed.
Lemma checks : validRunb memory rows = true.
Proof. vm_compute; reflexivity. Qed.
Lemma fragment : executionFragment start 0 rows.
Proof. vm_compute; intuition discriminate. Qed.
Lemma initial_config : CircuitConfig MixedIndirect.program MixedIndirect.store 0 shape memory start initial.
Proof.
  apply (typed_start_config MixedIndirect.program MixedIndirect.store 0 MixedIndirect.guest _ memory start);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact MixedIndirect.validated.
  - exact tables_loaded.
  - repeat first [apply Forall2_cons | apply Forall2_nil | split | reflexivity].
    + unfold wide. apply i64_value_wf; apply join_limbs_bound; unfold i32Modulus; lia.
    + apply i32_value_wf; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <- .. | destruct i; discriminate].
    + unfold wide. constructor; try (unfold i32Modulus; lia); reflexivity.
    + constructor; [unfold i32Modulus; lia | reflexivity].
  - repeat split.
Qed.
Lemma final_row_decodes : capturedValues (rowWitness (last mixedIndirectRows [])) [valtype_I64] = Some [wide].
Proof. vm_compute; reflexivity. Qed.
Example complete_mixed_indirect_refines : fieldPrime ->
  Steps (callContext caller 1 initial) (mk_config (mk_state MixedIndirect.store caller) [admininstr_val wide]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig MixedIndirect.program MixedIndirect.store caller 0 0 shape memory start (callContext caller 1 initial)).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ].
    exists MixedIndirect.guest, initial. split; [reflexivity | ]. split; [exact initial_config | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _) (validRunb_sound _ _ checks) fragment Hstart)
    as [outcome [c [Hsteps [_ Hconfig]]]].
  assert (Hc : c = mk_config (mk_state MixedIndirect.store caller) [admininstr_val wide]).
  { eapply terminal_values with (row := rowWitness (last mixedIndirectRows [])) (f := MixedIndirect.guest) (values := [wide]);
      [exact Hconfig | reflexivity | reflexivity | reflexivity | reflexivity | exact final_row_decodes]. }
  rewrite <- Hc. exact Hsteps.
Qed.

(* The high limb is part of the initial memory and the terminal capture. *)
Definition wrongInput := update memory stackId [1] 0.
Example changed_argument_high_rejected : validRunb wrongInput rows = false.
Proof. vm_compute; reflexivity. Qed.
Definition last := rowWitness (List.last mixedIndirectRows []).
Definition wrongOutput : Witness := {| entries := fun c =>
  if Nat.eqb c COL_OUTPUT_VALUE_HI_AFTER then field 0 else entries last c |}.
Example changed_output_high_rejected : satFullb wrongOutput = false.
Proof. vm_compute; reflexivity. Qed.
End Indirect.
