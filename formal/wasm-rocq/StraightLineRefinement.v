(* Composition of arithmetic and local instructions within one active frame. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub Locals
  StackBindings StraightLineSpec StackSimulation LocalSimulation LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma binary_row_refinement : forall program z initial final row pc c0 op chosen,
  Range.fieldPrime -> loadedProgram program initial -> FullRelation.satFull row ->
  row COL_IS_PROGRAM_ROW = 1 -> Replay initial (eventsForRow MemoryCatalog.memories row) final ->
  nth_error program pc = Some (BINOP I32 op) ->
  binaryArithmeticSelector chosen -> row chosen = 1 ->
  Step_pure
    [i32Constant (row (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
     i32Constant (row (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)); admininstr_instr (BINOP I32 op)]
    [i32Constant (row COL_STACK_WRITE0_VALUE_LO)] ->
  represents program z initial (beforeBoundary row) pc c0 ->
  exists c1, Step c0 c1 /\ represents program z final (afterBoundary row) (S pc) c1.
Proof.
  intros program z initial final row pc c0 op chosen Hprime Hloaded Hfull Hprogram Hreplay
    Hlookup Hchoice Hchosen Hpure Hrep.
  destruct Hrep as [Hpc [HpcBound [values [Hstack [Htyped [Hconfig [Hwf Hlocals]]]]]]].
  change (row COL_PC_BEFORE = Z.of_nat pc) in Hpc.
  assert (HlocalChoice : straightLineSelector chosen).
  { destruct Hchoice as [-> | ->]; cbn; auto. }
  pose proof (skipn_at_instruction program pc (BINOP I32 op) Hlookup) as Hsuffix.
  rewrite Hsuffix in Htyped, Hconfig.
  assert (Hheight : (2 <= length values)%nat).
  { destruct (length values) as [ | [ | n]]; cbn [programTyped] in Htyped; tauto || lia. }
  destruct (stack_has_two_values values Hheight) as [prefix [a [b Hvalues]]].
  subst values c0.
  assert (Hdepth : 2 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]].
    cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_binary_stack_shape row chosen Hprime Hfull Hchoice Hchosen Hdepth) as Hshape.
  pose proof (Range.full_write0_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hresult.
  destruct (stack_matches_binary_step row initial final prefix a b Hshape Hreplay Hstack Hresult)
    as [Hread0 [Hread1 HstackAfter]].
  rewrite Hread0, Hread1 in Hpure.
  destruct (stack_config_binary_step z prefix a b (row COL_STACK_WRITE0_VALUE_LO)
    (BINOP I32 op) (skipn (S pc) program) Hpure Hresult Hwf) as [Hstep HwfAfter].
  exists (stackConfig z (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]) (skipn (S pc) program)).
  split; [exact Hstep | ].
  unfold represents. split.
  - change (row COL_PC_AFTER = Z.of_nat (S pc)).
    destruct (program_row_next_pc program initial final row pc
      Hprime Hfull Hloaded Hprogram Hpc Hreplay) as [_ [_ Hnext]].
    replace (S pc) with (pc + 1)%nat by lia. exact Hnext.
  - split.
    + apply nth_error_Some. rewrite Hlookup. discriminate.
    + exists (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]). split; [exact HstackAfter | ].
      split.
      * rewrite length_app in Htyped |- *. cbn [length] in Htyped |- *.
        replace (length prefix + 2)%nat with (S (S (length prefix))) in Htyped by lia.
        cbn [programTyped] in Htyped.
        replace (length prefix + 1)%nat with (S (length prefix)) by lia. exact Htyped.
      * split; [reflexivity | ]. split; [exact HwfAfter | ].
        eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z); [ | | exact Hlocals].
        -- exact (proj1 (full_straight_line_local_gates row chosen Hprime Hfull HlocalChoice Hchosen)).
        -- apply no_local_writes_preserve with (events := eventsForRow MemoryCatalog.memories row);
             [exact Hreplay | ].
           rewrite (full_straight_line_local_writes row chosen Hprime Hfull HlocalChoice Hchosen).
           destruct Hchoice as [-> | ->]; reflexivity.
Qed.

Lemma encoded_opcode_cases : forall instruction code,
  encodedOpcode instruction = Some code ->
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 ADD) /\ code = CODE_I32_ADD) \/
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 SUB) /\ code = CODE_I32_SUB) \/
  (exists x, instruction = LOCAL_GET x /\ code = CODE_LOCAL_GET) \/
  (exists x, instruction = LOCAL_SET x /\ code = CODE_LOCAL_SET).
Proof.
  intros instruction code Hcode.
  destruct instruction; try discriminate;
    repeat match goal with
    | x : numtype |- _ => destruct x; try discriminate
    | x : binop_ |- _ => destruct x; try discriminate
    | x : Inn |- _ => destruct x; try discriminate
    | x : binop_Inn |- _ => destruct x; try discriminate
    end; inversion Hcode; subst; eauto 8.
Qed.

Theorem straight_line_row_refinement : forall program z initial final row pc c0,
  Range.fieldPrime -> loadedProgram program initial -> FullRelation.satFull row ->
  row COL_IS_PROGRAM_ROW = 1 -> Replay initial (eventsForRow MemoryCatalog.memories row) final ->
  represents program z initial (beforeBoundary row) pc c0 ->
  exists z' c1, Step c0 c1 /\ represents program z' final (afterBoundary row) (S pc) c1.
Proof.
  intros program z initial final row pc c0 Hprime Hloaded Hfull Hprogram Hreplay Hrep.
  assert (Hpc : row COL_PC_BEFORE = Z.of_nat pc) by exact (proj1 Hrep).
  destruct (program_row_fetch program initial final row pc Hloaded Hprogram Hpc Hreplay)
    as [instruction [Hlookup Hcode]].
  destruct (encoded_opcode_cases instruction _ Hcode)
    as [[-> HcodeAdd] | [[-> HcodeSub] | [[x [-> HcodeGet]] | [x [-> HcodeSet]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1).
    { eapply full_opcode_selects; try eassumption; reflexivity. }
    destruct (binary_row_refinement program z initial final row pc c0 _ COL_SEL_I32_ADD
      Hprime Hloaded Hfull Hprogram Hreplay Hlookup ltac:(left; reflexivity) Hchosen
      (Add.full_relation_implies_spectec_add row Hprime Hchosen Hfull) Hrep) as [c1 Hresult].
    exists z, c1. exact Hresult.
  - assert (Hchosen : row COL_SEL_I32_SUB = 1).
    { eapply full_opcode_selects; try eassumption; reflexivity. }
    destruct (binary_row_refinement program z initial final row pc c0 _ COL_SEL_I32_SUB
      Hprime Hloaded Hfull Hprogram Hreplay Hlookup ltac:(right; reflexivity) Hchosen
      (Sub.full_relation_implies_spectec_sub row Hprime Hchosen Hfull) Hrep) as [c1 Hresult].
    exists z, c1. exact Hresult.
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1).
    { eapply full_opcode_selects; try eassumption; reflexivity. }
    destruct (local_get_row_refinement program z initial final row pc c0 x
      Hprime Hloaded Hfull Hprogram Hreplay Hlookup Hchosen Hrep) as [c1 Hresult].
    exists z, c1. exact Hresult.
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1).
    { eapply full_opcode_selects; try eassumption; reflexivity. }
    destruct (local_set_row_refinement program z initial final row pc c0 x
      Hprime Hloaded Hfull Hprogram Hreplay Hlookup Hchosen Hrep) as [c1 Hresult].
    exists (with_local z x (i32Value (row COL_LOCAL_VALUE))), c1. exact Hresult.
Qed.

Theorem straight_line_rows_refine_prefix :
  forall program z rest first pc initial final c0,
    Range.fieldPrime -> supportedProgram program -> loadedProgram program initial ->
    validCircuitRun initial (first :: rest) final ->
    Forall (fun (row : Witness) => row COL_IS_PROGRAM_ROW = 1) (first :: rest) ->
    represents program z initial (beforeBoundary first) pc c0 ->
    exists z' c1,
      Steps c0 c1 /\
      represents program z' final (afterBoundary (finalRow first rest))
        (pc + length (first :: rest))%nat c1.
Proof.
  intros program z rest. revert z.
  pose proof straight_line_row_refinement as Hrow.
  induction rest as [ | next rest IH];
    intros z first pc initial final c0 Hprime Hsupported Hloaded Hrun Hprogram Hbefore.
  - destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hsat [Hreplay Htail]]].
    destruct Htail as [_ [_ Hempty]].
    assert (Hend : middle = final) by (inversion Hempty; congruence).
    subst middle.
    destruct (Hrow program z initial final first pc c0 Hprime Hloaded
      Hsat (Forall_inv Hprogram) Hreplay Hbefore) as [z1 [c1 [Hstep Hafter]]].
    exists z1, c1. split.
    + pose proof (represents_wf _ _ _ _ _ _ Hbefore) as Hwf0.
      pose proof (represents_wf _ _ _ _ _ _ Hafter) as Hwf1.
      destruct c0, c1. eapply trans; try eassumption.
      apply Steps__refl. exact Hwf1.
    + simpl. replace (pc + 1)%nat with (S pc) by lia. exact Hafter.
  - destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hsat [Hreplay Htail]]].
    destruct (Hrow program z initial middle first pc c0 Hprime Hloaded
      Hsat (Forall_inv Hprogram) Hreplay Hbefore) as [z1 [c1 [Hstep Hafter]]].
    assert (HloadedMiddle : loadedProgram program middle).
    { eapply loaded_program_preserved; [exact Hloaded | exact Hreplay | ].
      intros memory address Hreadonly. apply read_only_row. exact Hreadonly. }
    assert (Hlink : Follows MemoryCatalog.continuityLinks first next).
    { destruct Hrun as [_ [Hchain _]]. inversion Hchain; subst; assumption. }
    pose proof Hafter as Hnext.
    rewrite (follows_boundary first next Hlink) in Hnext.
    destruct (IH z1 next (S pc) middle final c1 Hprime Hsupported HloadedMiddle Htail
      (Forall_inv_tail Hprogram) Hnext) as [z2 [c2 [Hsteps Hfinal]]].
    exists z2, c2. split.
    + pose proof (represents_wf _ _ _ _ _ _ Hbefore) as Hwf0.
      pose proof (represents_wf _ _ _ _ _ _ Hafter) as Hwf1.
      pose proof (represents_wf _ _ _ _ _ _ Hfinal) as Hwf2.
      destruct c0, c1, c2. eapply trans; eassumption.
    + simpl in Hfinal |- *.
      replace (pc + S (S (length rest)))%nat
        with (S pc + S (length rest))%nat by lia.
      exact Hfinal.
Qed.

(* Auxiliary rows are outside this fragment.
   Opcode support is checked on the source, not via selectors.
   The run can be a prefix: its final configuration contains the source suffix. *)
Theorem straight_line_run_refinement :
  forall program z initial final first rest c0,
    Range.fieldPrime -> supportedProgram program -> loadedProgram program initial ->
    validCircuitRun initial (first :: rest) final ->
    Forall (fun (row : Witness) => row COL_IS_PROGRAM_ROW = 1) (first :: rest) ->
    represents program z initial (beforeBoundary first) 0%nat c0 ->
    exists z' c1,
      Steps c0 c1 /\
      represents program z' final (afterBoundary (finalRow first rest))
        (length (first :: rest)) c1.
Proof.
  intros program z initial final first rest c0 Hprime Hsupported Hloaded
    Hrun Hprogram Hbefore.
  exact (straight_line_rows_refine_prefix program z rest first 0%nat
    initial final c0 Hprime Hsupported Hloaded Hrun Hprogram Hbefore).
Qed.

