From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range Add TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation
  BlockSpec BlockValidation SourceInstructions CodeLayout ProgramSpec ProgramEncoding ProgramRefinement ControlSpec ControlState ProgramValidation SourceValidation tests.BlockValidation.
Import ListNotations.
Open Scope Z_scope.

Definition computedCode := [BLOCK (_RESULT None) computedBranches; LOCAL_GET (mk_uN 1)].
Definition computedModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition computedFunction := i32Function computedModule (mk_uN 0) 0 1 2 computedCode.
Definition computedProgram := [computedFunction].
Definition computedState condition := mk_state (MKstore computedProgram [] [] [] [] [])
  (MKframe [i32Value condition; i32Value 9] computedModule).
Definition computedRam condition : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then match address with [8] => Some 42 | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some condition | [7; 1] => Some 9 | _ => None end
  else None.
Definition computedMemory condition := loadProgram computedProgram (computedRam condition).

Definition computedContext := MKcontext [i32StackType 0 1] [i32StackType 0 1] [] [] [] [] [] [] [] None.
Definition computedBodyContext := functionContext computedContext computedFunction.

Example computed_code_validates : Instrs_ok computedBodyContext computedCode (i32StackType 0 1).
Proof.
  change (Instrs_ok computedBodyContext
    ([BLOCK (_RESULT None) computedBranches] ++ [LOCAL_GET (mk_uN 1)]) (i32StackType 0 1)).
  eapply res_seq with (t_2_lst := []).
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block_validates_context 2); try reflexivity; try solve [repeat constructor].
    + split; [reflexivity | ]. split; [exact computed_branches_supported | exact computed_branches_balanced].
    + exact computed_branches_validate.
  - apply Instrs_ok__instr; [apply local_get | | ]; try reflexivity; repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Example computed_module_validates : forall condition,
  Moduleinst_ok (fun_store (computedState condition)) computedModule computedContext.
Proof.
  intros. apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: try solve [constructor; [eapply Externaddr_ok__func with (v_funcinst := computedFunction);
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor | constructor]].
  all: repeat constructor.
Qed.

Example computed_function_validates : forall condition,
  Funcinst_ok (fun_store (computedState condition)) computedFunction (i32StackType 0 1).
Proof.
  intro condition. apply mk_Funcinst_ok with (C := computedContext).
  - repeat constructor.
  - apply computed_module_validates.
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
    + repeat constructor.
    + apply mk_Expr_ok; [exact computed_code_validates | repeat constructor | repeat constructor].
    + repeat constructor.
    + repeat constructor.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor.
Qed.

Example computed_initial_state_exists : forall condition,
  0 <= condition < i32Modulus ->
  programLoaded computedProgram (computedMemory condition) /\
  SourceState computedProgram (sourcePosition 0 0) (computedState condition)
    (computedMemory condition) (mkBoundary 0 5 4 7) computedCode
    (stackConfig (computedState condition) [42] computedCode).
Proof.
  intros condition Hcondition. split; [apply load_program_loaded | ].
  split; [repeat split; reflexivity | ]. split; [split; [cbn; lia | reflexivity] | ]. split.
  - split; [constructor; [apply computed_function_validates | constructor] | ]. split.
    + constructor.
      * split; [reflexivity | ]. split; [exact computed_branches_supported | exact computed_branches_balanced].
      * constructor; [constructor; [eexists; reflexivity | constructor] | constructor].
    + exists computedContext, 2%nat. split; [apply computed_module_validates | ].
      change (Instrs_ok computedBodyContext computedCode (i32StackType (1 + 0) (1 + 1))).
      unfold i32StackType. rewrite !repeat_app. apply Instrs_ok__frame;
        [exact computed_code_validates | repeat constructor | repeat constructor].
  - split; [ | reflexivity]. exists [42]. split.
    + unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split.
      * constructor; [unfold i32Modulus; lia | constructor].
      * intros [ | index] value H; [injection H as <-; reflexivity | destruct index; discriminate].
    + split; [reflexivity | ]. split.
      * apply stack_config_wf.
        -- constructor; [repeat constructor | ]. constructor; [ | repeat constructor].
           constructor; [apply val_case_0; apply spectec_i32_well_formed; exact Hcondition | repeat constructor].
        -- constructor; [unfold i32Modulus; lia | constructor].
        -- repeat constructor.
      * intros [ | [ | index]] value H; cbn [computedState stateLocals LOCALS nth_error] in H.
        -- injection H as <-. exists condition. repeat split; try reflexivity; lia.
        -- injection H as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
        -- destruct index; discriminate.
Qed.

(* Raw rows may now stop inside the label, without a sourceEndpoint premise. *)
Example computed_program_refines : forall condition rows final depth,
  fieldPrime -> 0 <= condition < i32Modulus ->
  validCircuitRun (computedMemory condition) rows final ->
  callerFragment (mkBoundary 0 5 4 7) depth rows ->
  exists cursor z c,
    Steps (stackConfig (computedState condition) [42] computedCode) c /\
    programLoaded computedProgram final /\
    ControlState computedProgram cursor z final (rowsFinalBoundary (mkBoundary 0 5 4 7) rows) c.
Proof.
  intros condition rows final depth Hprime Hcondition Hrun Hfragment.
  destruct (computed_initial_state_exists condition Hcondition) as [Hloaded Hsource].
  exact (run_refinement computedProgram (atInstruction (sourcePosition 0 0)) rows
    (computedState condition) (computedMemory condition) final (mkBoundary 0 5 4 7)
    depth _ Hprime Hloaded Hrun Hfragment ltac:(exists computedCode; exact Hsource)).
Qed.

Example computed_branch_layout :
  encodeProgram computedProgram pcRomId [4; 0] = Some 5 /\
  encodeProgram computedProgram pcRomId [4; 1] = Some 10 /\
  encodeProgram computedProgram pcRomId [6; 0] = Some 7 /\
  encodeProgram computedProgram pcRomId [6; 1] = Some 10 /\
  encodeProgram computedProgram programOpcodeId [9] = Some CODE_END.
Proof. repeat split; reflexivity. Qed.

(* After local.get 0; local.get 1; i32.sub, the value is 7 - 9 modulo 2^32.
   The enclosing stack's 42 remains below the block's label. *)
Definition computedInteriorMemory := loadProgram computedProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with
    | [8] => Some 42 | [10] => Some 4294967294 | _ => None end
  else computedRam 7 memory address).
Definition computedInteriorConfig := stackPrefix [42]
  (instructionSuffix [LOCAL_GET (mk_uN 1)]
    (SpecTecBranches.blockContext (stackConfig (computedState 7) [4294967294] (skipn 3 computedBranches)))).

Example computed_interior_state_exists :
  programLoaded computedProgram computedInteriorMemory /\
  ControlState computedProgram (inBlock (sourcePosition 0 0) 3 [42]) (computedState 7)
    computedInteriorMemory (mkBoundary 4 6 4 7) computedInteriorConfig.
Proof.
  split; [apply load_program_loaded | ].
  destruct (computed_initial_state_exists 7 ltac:(unfold i32Modulus; lia)) as [_ Hsource].
  destruct Hsource as [Hframe [Hcode [Hvalid [Hrep Hpc]]]].
  pose proof (represented_state_wf _ _ _ _ _ Hrep) as Hz.
  pose proof (validated_code_facts _ _ _ _ _ Hframe Hz Hvalid) as Hfacts.
  destruct Hfacts as [_ Hfacts].
  assert (Hcont : validatedCode computedProgram
    (mkGuestFunction 0 computedModule (mk_uN 0) 0 1 2 computedCode 0) (computedState 7) 1 [LOCAL_GET (mk_uN 1)]).
  { eapply validated_code_remaining with (executed := [BLOCK (_RESULT None) computedBranches]); try eassumption.
    repeat split; reflexivity. }
  exists (mkGuestFunction 0 computedModule (mk_uN 0) 0 1 2 computedCode 0), computedBranches,
    [LOCAL_GET (mk_uN 1)], [4294967294].
  split; [reflexivity | ]. split; [reflexivity | ]. split; [split; [cbn; lia | reflexivity] | ].
  split; [exact Hframe | ]. split; [exact Hcont | ]. split; [exact computed_branches_supported | ].
  split; [cbn; lia | ]. split; [exact remainder_after_sub_validates | ].
  split; [exact remainder_after_sub_balanced | ]. split; [exact Hz | ]. split.
  - unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split.
    + repeat constructor; unfold i32Modulus; lia.
    + intros [ | [ | index]] value H; try (destruct index; discriminate); injection H as <-; reflexivity.
  - split.
    + intros [ | [ | index]] value H; cbn in H.
      * injection H as <-. exists 7. repeat split; try reflexivity; unfold i32Modulus; lia.
      * injection H as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
      * destruct index; discriminate.
    + split; reflexivity.
Qed.

(* No source endpoint or selected branch outcome is needed, even when starting
   inside the block. The first run's output is precisely the second run's input. *)
Example computed_interior_two_runs : forall firstRows lastRows final depth,
  fieldPrime -> validCircuitRun computedInteriorMemory (firstRows ++ lastRows) final ->
  callerFragment (mkBoundary 4 6 4 7) depth firstRows ->
  callerFragment (rowsFinalBoundary (mkBoundary 4 6 4 7) firstRows) depth lastRows ->
  exists cursor z c, Steps computedInteriorConfig c /\ programLoaded computedProgram final /\
    ControlState computedProgram cursor z final
      (rowsFinalBoundary (mkBoundary 4 6 4 7) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final depth Hprime Hrun Hfirst Hlast.
  destruct computed_interior_state_exists as [Hloaded Hstate].
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram (inBlock (sourcePosition 0 0) 3 [42]) firstRows (computedState 7)
    computedInteriorMemory between (mkBoundary 4 6 4 7) depth _ Hprime Hloaded Hrun1 Hfirst Hstate)
    as [cursor1 [z1 [c1 [Hsteps1 [Hloaded1 Hstate1]]]]].
  destruct (run_refinement computedProgram cursor1 lastRows z1 between final
    (rowsFinalBoundary (mkBoundary 4 6 4 7) firstRows) depth c1 Hprime Hloaded1 Hrun2 Hlast Hstate1)
    as [cursor2 [z2 [c2 [Hsteps2 [Hloaded2 Hstate2]]]]].
  exists cursor2, z2, c2. split; [eapply spec_steps_transitive; eassumption | ].
  split; [exact Hloaded2 | ]. rewrite TraceSegmentation.rows_final_boundary_app. exact Hstate2.
Qed.

Lemma computed_pc_four_is_inside : forall cursor z memory boundary c,
  ControlState computedProgram cursor z memory boundary c -> boundaryPC boundary = 4 ->
  exists prefix, cursor = inBlock (sourcePosition 0 0) 3 prefix.
Proof.
  intros [[index offset] | [index outer] offset prefix] z memory boundary c Hstate Hpc.
  - destruct Hstate as [code Hstate]. unfold SourceState in Hstate.
    destruct index as [ | index]; [ | destruct index; contradiction].
    destruct Hstate as [_ [[Hbound _] [_ [_ Hpos]]]].
    change (offset <= 2)%nat in Hbound.
    destruct offset as [ | [ | [ | offset]]]; cbn in Hpos; lia.
  - destruct Hstate as [f [body [continuation [values [Hf [Hbody [_ [_ [_ [_ [Hbound [_ [_ [_ [_ [_ [Hpos _]]]]]]]]]]]]]]]]].
    destruct index as [ | index]; [ | destruct index; discriminate].
    injection Hf as <-.
    destruct outer as [ | [ | outer]]; [ | discriminate | destruct outer; discriminate].
    injection Hbody as <-. exists prefix. f_equal. change (boundaryPC boundary = 1 + Z.of_nat offset) in Hpos. lia.
Qed.

(* Splitting the raw trace before the first br_if yields an interior cursor,
   rather than requiring the theorem user to supply a block segment. *)
Example computed_split_before_branch : forall firstRows lastRows final depth,
  fieldPrime -> validCircuitRun (computedMemory 7) (firstRows ++ lastRows) final ->
  callerFragment (mkBoundary 0 5 4 7) depth firstRows ->
  callerFragment (rowsFinalBoundary (mkBoundary 0 5 4 7) firstRows) depth lastRows ->
  boundaryPC (rowsFinalBoundary (mkBoundary 0 5 4 7) firstRows) = 4 ->
  exists prefix between z1 c1 cursor z c,
    ControlState computedProgram (inBlock (sourcePosition 0 0) 3 prefix) z1 between
      (rowsFinalBoundary (mkBoundary 0 5 4 7) firstRows) c1 /\
    Steps (stackConfig (computedState 7) [42] computedCode) c1 /\ Steps c1 c /\
    programLoaded computedProgram final /\
    ControlState computedProgram cursor z final
      (rowsFinalBoundary (mkBoundary 0 5 4 7) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final depth Hprime Hrun Hfirst Hlast Hpc.
  destruct (computed_initial_state_exists 7 ltac:(unfold i32Modulus; lia)) as [Hloaded Hstate].
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram (atInstruction (sourcePosition 0 0)) firstRows (computedState 7)
    (computedMemory 7) between (mkBoundary 0 5 4 7) depth _ Hprime Hloaded Hrun1 Hfirst
    ltac:(exists computedCode; exact Hstate))
    as [cursor1 [z1 [c1 [Hsteps1 [Hloaded1 Hstate1]]]]].
  destruct (computed_pc_four_is_inside _ _ _ _ _ Hstate1 Hpc) as [prefix ->].
  destruct (run_refinement computedProgram (inBlock (sourcePosition 0 0) 3 prefix) lastRows z1 between final
    (rowsFinalBoundary (mkBoundary 0 5 4 7) firstRows) depth c1 Hprime Hloaded1 Hrun2 Hlast Hstate1)
    as [cursor2 [z2 [c2 [Hsteps2 [Hloaded2 Hstate2]]]]].
  exists prefix, between, z1, c1, cursor2, z2, c2.
  split; [exact Hstate1 | ]. split; [exact Hsteps1 | ]. split; [exact Hsteps2 | ].
  split; [exact Hloaded2 | ]. rewrite TraceSegmentation.rows_final_boundary_app. exact Hstate2.
Qed.
