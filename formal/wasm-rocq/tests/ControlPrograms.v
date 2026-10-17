From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range Add TraceRefinement MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation
  BlockSpec BlockValidation SourceInstructions CodeLayout ProgramSpec ProgramEncoding ProgramRefinement ControlDecode ControlInvariant ControlState ControlBridge ProgramValidation SourceValidation tests.BlockValidation.
Import ListNotations.
Open Scope Z_scope.

Definition computedCode := [BLOCK (_RESULT None) computedBranches; LOCAL_GET (mk_uN 1)].
Definition computedModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition computedFunction := i32Function computedModule (mk_uN 0) 0 1 2 computedCode.
Definition computedProgram := [computedFunction].
Definition computedStore := MKstore computedProgram [] [] [] [] [].
Definition computedState condition := mk_state computedStore
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
    (computedMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) computedCode
    (stackConfig (computedState condition) [] computedCode).
Proof.
  intros condition Hcondition. split; [apply load_program_loaded | ].
  split; [repeat split; reflexivity | ]. split; [split; [cbn; lia | reflexivity] | ]. split.
  - split; [constructor; [apply computed_function_validates | constructor] | ]. split.
    + constructor.
      * split; [reflexivity | ]. split; [exact computed_branches_supported | exact computed_branches_balanced].
      * constructor; [constructor; [eexists; reflexivity | constructor] | constructor].
    + exists computedContext. split; [apply computed_module_validates | ].
      change (Instrs_ok computedBodyContext computedCode (i32StackType 0 1)).
      exact computed_code_validates.
  - split; [ | reflexivity]. exists []. split.
    + unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split; [constructor | ].
      intros [ | index] value H; discriminate.
    + split; [reflexivity | ]. split.
      * apply stack_config_wf.
        -- constructor; [repeat constructor | ]. constructor; [ | repeat constructor].
           constructor; [apply val_case_0; apply spectec_i32_well_formed; exact Hcondition | repeat constructor].
        -- constructor.
        -- repeat constructor.
      * intros [ | [ | index]] value H; cbn [computedState stateLocals LOCALS nth_error] in H.
        -- injection H as <-. exists condition. repeat split; try reflexivity; lia.
        -- injection H as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
        -- destruct index; discriminate.
Qed.

Lemma computed_initial_control : forall condition,
  0 <= condition < i32Modulus ->
  ControlState computedProgram 0 (controlCursor [] (atInstruction (sourcePosition 0 0))) (computedState condition)
    (computedMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0)
    (stackConfig (computedState condition) [] computedCode).
Proof.
  intros condition Hcondition.
  destruct (computed_initial_state_exists condition Hcondition) as [_ Hsource].
  apply (base_control_state computedProgram (atInstruction (sourcePosition 0 0)) (computedState condition)
    (computedMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0)); [ | cbn; lia].
  split; [exists computedCode; exact Hsource | repeat split].
Qed.

Example computed_store_wf : wf_store computedStore.
Proof. repeat constructor. Qed.

Lemma computed_initial_config : forall condition,
  0 <= condition < i32Modulus ->
  CircuitConfig computedProgram computedStore 0 (controlShape [] (siteInstruction (sourcePosition 0 0)))
    (computedMemory condition) (mkBoundary 0 4 4 7 0 0 0 0 0) (stackConfig (computedState condition) [] computedCode).
Proof.
  intros condition Hcondition.
  apply (control_state_circuit _ _ _ (controlCursor [] (atInstruction (sourcePosition 0 0))) (computedState condition));
    [reflexivity | exact computed_store_wf | reflexivity | exact (computed_initial_control condition Hcondition)].
Qed.

(* Raw rows may stop anywhere inside the function, including inside the label. *)
Example computed_program_refines : forall condition rows final,
  fieldPrime -> 0 <= condition < i32Modulus ->
  validCircuitRun (computedMemory condition) rows final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 rows ->
  exists shape c,
    Steps (stackConfig (computedState condition) [] computedCode) c /\
    programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) rows) c.
Proof.
  intros condition rows final Hprime Hcondition Hrun Hfragment.
  exact (run_refinement computedProgram computedStore 0 _ rows _ final _ _ Hprime
    (proj1 (computed_initial_state_exists condition Hcondition)) Hrun Hfragment
    (computed_initial_config condition Hcondition)).
Qed.

Example computed_branch_layout :
  encodeProgram computedProgram pcRomId [4; 0] = Some 5 /\
  encodeProgram computedProgram pcRomId [4; 1] = Some 10 /\
  encodeProgram computedProgram pcRomId [6; 0] = Some 7 /\
  encodeProgram computedProgram pcRomId [6; 1] = Some 10 /\
  encodeProgram computedProgram programOpcodeId [9] = Some CODE_END.
Proof. repeat split; reflexivity. Qed.

(* After local.get 0; local.get 1; i32.sub, the value is 7 - 9 modulo 2^32.
   It is the only value inside the block's label. *)
Definition computedInteriorMemory := loadProgram computedProgram (fun memory address =>
  if Nat.eqb memory stackId then match address with
    | [8] => Some 4294967294 | _ => None end
  else computedRam 7 memory address).
Definition computedInteriorConfig := stackPrefix []
  (instructionSuffix [LOCAL_GET (mk_uN 1)]
    (SpecTecBranches.blockContext (stackConfig (computedState 7) [4294967294] (skipn 3 computedBranches)))).

Example computed_interior_state_exists :
  programLoaded computedProgram computedInteriorMemory /\
  ControlState computedProgram 0 (controlCursor [] (inBlock (sourcePosition 0 0) 3 [])) (computedState 7)
    computedInteriorMemory (mkBoundary 4 5 4 7 0 0 0 0 0) computedInteriorConfig.
Proof.
  split; [apply load_program_loaded | ].
  destruct (computed_initial_state_exists 7 ltac:(unfold i32Modulus; lia)) as [_ Hsource].
  destruct Hsource as [Hframe [Hcode [Hvalid [Hrep Hpc]]]].
  pose proof (represented_state_wf _ _ _ _ _ Hrep) as Hz.
  pose proof (validated_code_facts _ _ _ _ _ Hframe Hz Hvalid) as Hfacts.
  destruct Hfacts as [_ Hfacts].
  assert (Hcont : validatedCode computedProgram
    (mkGuestFunction 0 computedModule (mk_uN 0) 0 1 2 computedCode 0) (computedState 7) 0 [LOCAL_GET (mk_uN 1)]).
  { eapply validated_code_remaining with (executed := [BLOCK (_RESULT None) computedBranches]); try eassumption.
    all: try (repeat split; reflexivity); reflexivity. }
  destruct (validated_code_module _ _ _ _ _ Hcont) as [C HC].
  apply (base_control_state computedProgram (inBlock (sourcePosition 0 0) 3 []) (computedState 7)
    computedInteriorMemory (mkBoundary 4 5 4 7 0 0 0 0 0)); [ | cbn; lia].
  exists (mkGuestFunction 0 computedModule (mk_uN 0) 0 1 2 computedCode 0), computedBranches,
    [LOCAL_GET (mk_uN 1)], [4294967294].
  split; [reflexivity | ]. split; [reflexivity | ]. split; [split; [cbn; lia | reflexivity] | ].
  split; [exact Hframe | ]. split; [exact Hcont | ]. split; [exact computed_branches_supported | ].
  split; [cbn; lia | ]. split.
  { eapply block_rest_validated; [exact Hframe | | exact HC | exact remainder_after_sub_validates].
    apply supported_branch_suffix. exact computed_branches_supported. }
  split; [exact remainder_after_sub_balanced | ]. split; [exact Hz | ]. split.
  - unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split.
    + repeat constructor; unfold i32Modulus; lia.
    + intros [ | index] value H; [injection H as <-; reflexivity | destruct index; discriminate].
  - split.
    + intros [ | [ | index]] value H; cbn in H.
      * injection H as <-. exists 7. repeat split; try reflexivity; unfold i32Modulus; lia.
      * injection H as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
      * destruct index; discriminate.
    + split; [reflexivity | ]. split; [repeat split | reflexivity].
Qed.

(* Inside the label, the decoder splits the operand stack at the block's height. *)
Example computed_interior_decodes :
  option_map fst (decodeControl computedProgram computedStore 0
    (controlShape [] (siteBlock (sourcePosition 0 0) 3 0)) computedInteriorMemory (mkBoundary 4 5 4 7 0 0 0 0 0)) =
  Some computedInteriorConfig.
Proof. vm_compute. reflexivity. Qed.

Example computed_interior_config :
  CircuitConfig computedProgram computedStore 0 (controlShape [] (siteBlock (sourcePosition 0 0) 3 0))
    computedInteriorMemory (mkBoundary 4 5 4 7 0 0 0 0 0) computedInteriorConfig.
Proof.
  apply (control_state_circuit _ _ _ (controlCursor [] (inBlock (sourcePosition 0 0) 3 [])) (computedState 7));
    [reflexivity | exact computed_store_wf | reflexivity | exact (proj2 computed_interior_state_exists)].
Qed.

(* No branch outcome is needed, even when starting inside the block. The first
   run's output is precisely the second run's input. *)
Example computed_interior_two_runs : forall firstRows lastRows final,
  fieldPrime -> validCircuitRun computedInteriorMemory (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 4 5 4 7 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 4 5 4 7 0 0 0 0 0) firstRows) 0 lastRows ->
  exists shape c, Steps computedInteriorConfig c /\ programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final
      (rowsFinalBoundary (mkBoundary 4 5 4 7 0 0 0 0 0) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final Hprime Hrun Hfirst Hlast.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram computedStore 0 _ firstRows computedInteriorMemory between _ _ Hprime
    (proj1 computed_interior_state_exists) Hrun1 Hfirst computed_interior_config)
    as [shape1 [c1 [Hsteps1 [Hloaded1 Hconfig1]]]].
  destruct (run_refinement computedProgram computedStore 0 shape1 lastRows between final _ c1 Hprime Hloaded1 Hrun2
    Hlast Hconfig1) as [shape2 [c2 [Hsteps2 [Hloaded2 Hconfig2]]]].
  exists shape2, c2. split; [eapply spec_steps_transitive; eassumption | ].
  split; [exact Hloaded2 | ]. rewrite rows_final_boundary_app. exact Hconfig2.
Qed.

Lemma computed_pc_four_is_inside : forall cursor z memory boundary c,
  ActiveState computedProgram cursor z memory boundary c -> boundaryPC boundary = 4 ->
  exists prefix, cursor = inBlock (sourcePosition 0 0) 3 prefix.
Proof.
  intros [[index offset] | [index outer] offset prefix | function arguments] z memory boundary c Hstate Hpc.
  - destruct Hstate as [[code Hstate] _]. unfold SourceState in Hstate.
    destruct index as [ | index]; [ | destruct index; contradiction].
    destruct Hstate as [_ [[Hbound _] [_ [_ Hpos]]]].
    change (offset <= 2)%nat in Hbound.
    destruct offset as [ | [ | [ | offset]]]; cbn in Hpos; lia.
  - destruct Hstate as [f [body [continuation [values [Hf [Hbody [_ [_ [_ [_ [Hbound [_ [_ [_ [_ [_ [Hpos _]]]]]]]]]]]]]]]]].
    destruct index as [ | index]; [ | destruct index; discriminate].
    injection Hf as <-.
    destruct outer as [ | [ | outer]]; [ | discriminate | destruct outer; discriminate].
    injection Hbody as <-. exists prefix. f_equal. change (boundaryPC boundary = 1 + Z.of_nat offset) in Hpos. lia.
  - exfalso. destruct Hstate as [f [Hf [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [Hpos _]]]]]]]]]]]]].
    destruct function as [ | function]; [ | destruct function; discriminate].
    injection Hf as <-. cbn in Hpos. lia.
Qed.

(* Splitting the raw trace before the first br_if yields an interior shape,
   rather than requiring the theorem user to supply a block segment. *)
Example computed_split_before_branch : forall firstRows lastRows final,
  fieldPrime -> validCircuitRun (computedMemory 7) (firstRows ++ lastRows) final ->
  runFragment (mkBoundary 0 4 4 7 0 0 0 0 0) 0 firstRows ->
  runFragment (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) 0 lastRows ->
  boundaryPC (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) = 4 ->
  exists height between c1 shape c,
    CircuitConfig computedProgram computedStore 0 (controlShape [] (siteBlock (sourcePosition 0 0) 3 height))
      between (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) firstRows) c1 /\
    Steps (stackConfig (computedState 7) [] computedCode) c1 /\ Steps c1 c /\
    programLoaded computedProgram final /\
    CircuitConfig computedProgram computedStore 0 shape final
      (rowsFinalBoundary (mkBoundary 0 4 4 7 0 0 0 0 0) (firstRows ++ lastRows)) c.
Proof.
  intros firstRows lastRows final Hprime Hrun Hfirst Hlast Hpc.
  pose proof (proj1 (computed_initial_state_exists 7 ltac:(unfold i32Modulus; lia))) as Hloaded.
  destruct (valid_run_split _ _ _ _ Hrun) as [between [Hrun1 Hrun2]].
  destruct (run_refinement computedProgram computedStore 0 _ firstRows (computedMemory 7) between _ _ Hprime Hloaded
    Hrun1 Hfirst (computed_initial_config 7 ltac:(unfold i32Modulus; lia)))
    as [shape1 [c1 [Hsteps1 [Hloaded1 Hconfig1]]]].
  destruct (run_refinement computedProgram computedStore 0 shape1 lastRows between final _ c1 Hprime Hloaded1 Hrun2
    Hlast Hconfig1) as [shape2 [c2 [Hsteps2 [Hloaded2 Hconfig2]]]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig1) as [[stack active] [z1 [Hstore Hstate1]]].
  pose proof Hstate1 as [f [inner [Hf [Hactive [Hsusp _]]]]]. cbn [activePosition suspendedCalls] in *.
  destruct (computed_pc_four_is_inside _ _ _ _ _ Hactive Hpc) as [prefix ->].
  destruct stack as [ | top more].
  2: { exfalso. injection Hf as <-. pose proof (suspended_callee_straight _ _ _ _ _ _ _ _ _ Hsusp) as Hstraight.
       inversion Hstraight as [ | ? ? [op Hop] _]. discriminate. }
  exists (length prefix), between, c1, shape2, c2.
  split; [exact (control_state_circuit _ _ _ _ _ _ _ _ Logic.eq_refl computed_store_wf Hstore Hstate1) | ].
  split; [exact Hsteps1 | ]. split; [exact Hsteps2 | ].
  split; [exact Hloaded2 | ]. rewrite rows_final_boundary_app. exact Hconfig2.
Qed.
