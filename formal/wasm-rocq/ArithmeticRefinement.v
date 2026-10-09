(* Straight-line i32.add/sub refinement target. The observed circuit state is
   the operand stack and PC; the ambient SpecTec store/frame stays fixed.
   Row simulation derives the stack and control-flow effects from the circuit. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  OpcodeCatalog Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Sub StackBindings ArithmeticSpec.
Import ListNotations.
Open Scope Z_scope.

Lemma follows_boundary : forall previous next,
  Follows MemoryCatalog.continuityLinks previous next ->
  afterBoundary previous = beforeBoundary next.
Proof.
  intros previous next Hlinks.
  unfold afterBoundary, beforeBoundary. f_equal;
    apply Hlinks; vm_compute; tauto.
Qed.

Lemma represents_wf : forall program z memory boundary pc configuration,
  represents program z memory boundary pc configuration -> wf_config configuration.
Proof.
  intros program z memory boundary pc configuration [_ [_ [values [_ [_ [_ Hwf]]]]]].
  exact Hwf.
Qed.

Lemma loaded_program_preserved : forall program before events after,
  loadedProgram program before -> Replay before events after ->
  (forall memory address,
    readOnlyMemory memory MemoryCatalog.memories = true ->
    Forall (NoWriteTo memory address) events) ->
  loadedProgram program after.
Proof.
  intros program before events after [Hopcode [Hkind Hpc]] Hreplay Hreadonly.
  repeat split; intros.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ opcode_memory_read_only)). apply Hopcode.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ edge_kind_memory_read_only)). apply Hkind.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ pc_memory_read_only)). apply Hpc.
Qed.

Lemma arithmetic_row_fetch : forall program initial final (w : Witness) pc,
  loadedProgram program initial ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = Z.of_nat pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  exists instruction,
    nth_error program pc = Some instruction /\
    encodedOpcode instruction = Some (w COL_OPCODE_CODE).
Proof.
  intros program initial final w pc [Hopcode _] Hprogram Hpc Hreplay.
  pose proof (circuit_row_opcode_fetch initial final w Hreplay Hprogram) as Hfetch.
  rewrite Hpc, Hopcode in Hfetch. unfold programOpcodeAt in Hfetch.
  destruct (nth_error program pc) as [instruction | ] eqn:Hlookup; [ | discriminate].
  exists instruction. auto.
Qed.

Lemma arithmetic_row_next_pc : forall program initial final (w : Witness) pc,
  Range.fieldPrime -> FullRelation.satFull w -> loadedProgram program initial ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = Z.of_nat pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PC_ROM_ACTIVE = 1 /\ w COL_CONTROL_CHOICE = 0 /\
  w COL_PC_AFTER = Z.of_nat (pc + 1).
Proof.
  intros program initial final w pc Hprime Hfull [_ [Hkind HpcRom]] Hprogram Hpc Hreplay.
  pose proof (circuit_row_edge_kind_fetch initial final w Hreplay Hprogram) as HfetchKind.
  rewrite Hpc, Hkind in HfetchKind.
  destruct (pc <? length program)%nat eqn:Hinside; [ | discriminate].
  assert (Hstatic : w COL_PC_EDGE_KIND = 0) by congruence.
  pose proof (full_static_pc_rom_active w Hprime Hfull Hprogram Hstatic) as Hactive.
  pose proof (circuit_row_next_pc_fetch initial final w Hreplay Hactive) as HfetchPc.
  rewrite Hpc, HpcRom, Hinside in HfetchPc. cbn in HfetchPc.
  destruct (w COL_CONTROL_CHOICE =? 0) eqn:Hchoice; [ | discriminate].
  apply Z.eqb_eq in Hchoice. repeat split; congruence.
Qed.

Lemma encoded_opcode_cases : forall instruction code,
  encodedOpcode instruction = Some code ->
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 ADD) /\ code = CODE_I32_ADD) \/
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 SUB) /\ code = CODE_I32_SUB).
Proof.
  intros instruction code Hcode.
  destruct instruction; try discriminate.
  repeat match goal with
  | x : numtype |- _ => destruct x; try discriminate
  | x : binop_ |- _ => destruct x; try discriminate
  | x : Inn |- _ => destruct x; try discriminate
  | x : binop_Inn |- _ => destruct x; try discriminate
  end; inversion Hcode; subst; auto.
Qed.

(* No selector premise: fetch and decode choose the applicable arithmetic proof. *)
Theorem arithmetic_row_pure_step : forall program initial final (w : Witness) pc,
  Range.fieldPrime -> FullRelation.satFull w -> loadedProgram program initial ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = Z.of_nat pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  exists instruction,
    nth_error program pc = Some instruction /\
    Step_pure
      [i32Constant (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
       i32Constant (w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat));
       admininstr_instr instruction]
      [i32Constant (w COL_STACK_WRITE0_VALUE_LO)].
Proof.
  intros program initial final w pc Hprime Hfull Hloaded Hprogram Hpc Hreplay.
  destruct (arithmetic_row_fetch program initial final w pc Hloaded Hprogram Hpc Hreplay)
    as [instruction [Hlookup Hcode]].
  exists instruction. split; [exact Hlookup | ].
  destruct (encoded_opcode_cases instruction _ Hcode) as [[Hin Hop] | [Hin Hop]];
    subst instruction.
  - apply Add.full_relation_implies_spectec_add; try assumption.
    unfold Add.addSelector. eapply full_opcode_selects; try eassumption.
    exact (eq_refl : filter (fun entry => Z.eqb (snd entry) CODE_I32_ADD) selectorCodes =
      [(COL_SEL_I32_ADD, CODE_I32_ADD)]).
  - apply Sub.full_relation_implies_spectec_sub; try assumption.
    unfold Sub.subSelector. eapply full_opcode_selects; try eassumption.
    exact (eq_refl : filter (fun entry => Z.eqb (snd entry) CODE_I32_SUB) selectorCodes =
      [(COL_SEL_I32_SUB, CODE_I32_SUB)]).
Qed.

Lemma stack_matches_binary_step : forall w initial final prefix a b,
  BinaryStackShape w ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [a; b]) ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < i32Modulus ->
  w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = a /\
  w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat) = b /\
  stackMatches final (afterBoundary w) (prefix ++ [w COL_STACK_WRITE0_VALUE_LO]).
Proof.
  intros w initial final prefix a b Hshape Hreplay [Hbase [Hheight [Hbounds Hcells]]] Hresult.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hheight, Hcells.
  rewrite length_app in Hheight. cbn [length] in Hheight.
  rewrite Nat2Z.inj_add in Hheight.
  destruct (binary_stack_replay w initial final Hshape Hreplay) as [Hread0 [Hread1 Hfinal]].
  assert (Ha : nth_error (prefix ++ [a; b]) (length prefix) = Some a).
  { rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity. }
  assert (Hb : nth_error (prefix ++ [a; b]) (S (length prefix)) = Some b).
  { rewrite nth_error_app2 by lia.
    replace (S (length prefix) - length prefix)%nat with 1%nat by lia. reflexivity. }
  pose proof (Hcells _ _ Ha) as HcellA.
  pose proof (Hcells _ _ Hb) as HcellB.
  rewrite (stack_read0_address w Hshape) in Hread0.
  rewrite (stack_read1_address w Hshape) in Hread1.
  replace (2 * (w COL_SP_BEFORE - 2)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix))) in Hread0 by lia.
  replace (2 * (w COL_SP_BEFORE - 1)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (S (length prefix)))) in Hread1
    by (rewrite Nat2Z.inj_succ; lia).
  split; [congruence | ]. split; [congruence | ].
  unfold stackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (stack_base_after w Hshape), (stack_pointer_after w Hshape).
  split; [exact Hbase | ]. split.
  - rewrite length_app, Nat2Z.inj_add. cbn. lia.
  - split.
    + apply Forall_app in Hbounds. apply Forall_app. split; [exact (proj1 Hbounds) | ].
      constructor; [exact Hresult | constructor].
    + intros index value Hlookup. rewrite Hfinal.
      rewrite update_other.
      2: { right. rewrite (stack_write_hi_address w Hshape).
           intro Heq. injection Heq as Heq.
           change (2 * (w COL_SP_BEFORE - 2) + 1 =
             2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) in Heq. lia. }
      destruct (Nat.lt_ge_cases index (length prefix)) as [Hinside | Houtside].
      * rewrite update_other.
        2: { right. rewrite (stack_write_address w Hshape).
             intro Heq. injection Heq as Heq.
             change (2 * (w COL_SP_BEFORE - 2) =
               2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) in Heq.
             apply Nat2Z.inj_lt in Hinside. lia. }
        apply Hcells. rewrite nth_error_app1 by exact Hinside.
        rewrite nth_error_app1 in Hlookup by exact Hinside. exact Hlookup.
      * rewrite nth_error_app2 in Hlookup by exact Houtside.
        destruct (index - length prefix)%nat as [ |n] eqn:Hindex.
        -- cbn in Hlookup. inversion Hlookup; subst value.
           assert (HindexEq : index = length prefix) by lia. subst index.
           replace (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix)))
             with (w COL_STACK_WRITE0_ADDR_LO)
             by (rewrite (stack_write_address w Hshape); lia).
           apply update_same.
        -- destruct n; discriminate.
Qed.

Lemma skipn_at_instruction : forall (program : list instr) pc instruction,
  nth_error program pc = Some instruction ->
  skipn pc program = instruction :: skipn (S pc) program.
Proof.
  induction program as [ |head rest IH]; intros [ |pc] instruction Hlookup;
    cbn in *; try discriminate.
  - inversion Hlookup; reflexivity.
  - apply IH. exact Hlookup.
Qed.

Lemma stack_has_two_values : forall (values : list Z),
  (2 <= length values)%nat -> exists prefix a b, values = prefix ++ [a; b].
Proof.
  induction values as [ |a rest IH]; intros Hlength; [cbn in Hlength; lia | ].
  destruct rest as [ |b rest]; [cbn in Hlength; lia | ].
  destruct rest as [ |c rest].
  - exists [], a, b. reflexivity.
  - destruct (IH ltac:(cbn; lia)) as [prefix [x [y Heq]]].
    exists (a :: prefix), x, y. cbn. now rewrite Heq.
Qed.

Lemma i32_constant_wf : forall value,
  0 <= value < i32Modulus -> wf_admininstr (i32Constant value).
Proof.
  intros value Hbound. unfold i32Constant. constructor.
  apply Add.spectec_i32_well_formed. exact Hbound.
Qed.

Lemma stack_config_binary_step : forall z prefix a b result instruction code,
  Step_pure [i32Constant a; i32Constant b; admininstr_instr instruction]
    [i32Constant result] ->
  0 <= result < i32Modulus ->
  wf_config (stackConfig z (prefix ++ [a; b]) (instruction :: code)) ->
  Step (stackConfig z (prefix ++ [a; b]) (instruction :: code))
    (stackConfig z (prefix ++ [result]) code) /\
  wf_config (stackConfig z (prefix ++ [result]) code).
Proof.
  intros z prefix a b result instruction code Hpure Hresult Hwf.
  unfold stackConfig in *.
  rewrite !map_app in *. cbn [map] in *.
  repeat rewrite <- app_assoc in *.
  inversion Hwf as [state instructions Hz Hall]; subst.
  apply Forall_app in Hall. destruct Hall as [Hprefix Htail].
  cbn [app] in Htail.
  pose proof (Forall_inv Htail) as Ha.
  pose proof (Forall_inv (Forall_inv_tail Htail)) as Hb.
  pose proof (Forall_inv (Forall_inv_tail (Forall_inv_tail Htail))) as Hi.
  pose proof (Forall_inv_tail (Forall_inv_tail (Forall_inv_tail Htail))) as Hcode.
  pose proof (i32_constant_wf result Hresult) as Hr.
  assert (Hbefore : wf_config (mk_config z
    [i32Constant a; i32Constant b; admininstr_instr instruction])).
  { constructor; [exact Hz | ]. constructor; [exact Ha | ].
    constructor; [exact Hb | ]. constructor; [exact Hi | constructor]. }
  assert (Hafter : wf_config (mk_config z [i32Constant result])).
  { constructor; [exact Hz | ]. constructor; [exact Hr | constructor]. }
  split.
  - assert (Hmap : map admininstr_val
        (map (fun value => val_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N value)))) prefix) =
      map i32Constant prefix).
    { rewrite map_map. reflexivity. }
    rewrite <- Hmap.
    destruct prefix as [ |head prefix], code as [ |next code].
    + cbn [map app]. apply pure. exact Hpure.
    + eapply ctxt_instrs with
        (admininstr_lst := [i32Constant a; i32Constant b; admininstr_instr instruction])
        (admininstr'_lst := [i32Constant result]);
        [apply pure; exact Hpure | reflexivity | exact Hbefore | exact Hafter].
    + eapply ctxt_instrs with
        (admininstr_lst := [i32Constant a; i32Constant b; admininstr_instr instruction])
        (admininstr'_lst := [i32Constant result]);
        [apply pure; exact Hpure | reflexivity | exact Hbefore | exact Hafter].
    + eapply ctxt_instrs with
        (admininstr_lst := [i32Constant a; i32Constant b; admininstr_instr instruction])
        (admininstr'_lst := [i32Constant result]);
        [apply pure; exact Hpure | reflexivity | exact Hbefore | exact Hafter].
  - constructor; [exact Hz | ]. apply Forall_app. split; [exact Hprefix | ].
    cbn [app]. constructor; assumption.
Qed.

Fixpoint finalRow (first : Witness) (rest : list Witness) : Witness :=
  match rest with [] => first | next :: rest => finalRow next rest end.

(* Selector, port activation, addresses, stack update and next PC are derived. *)
Theorem arithmetic_row_refinement :
  forall program z initial final row pc c0,
    Range.fieldPrime -> supportedProgram program -> loadedProgram program initial ->
    FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
    Replay initial (eventsForRow MemoryCatalog.memories row) final ->
    represents program z initial (beforeBoundary row) pc c0 ->
    exists c1, Step c0 c1 /\
      represents program z final (afterBoundary row) (S pc) c1.

Proof.
  intros program z initial final row pc c0 Hprime Hsupported Hloaded Hfull Hprogram Hreplay Hrep.
  destruct Hrep as [Hpc [HpcBound [values [Hstack [Htyped [Hconfig Hwf]]]]]].
  change (row COL_PC_BEFORE = Z.of_nat pc) in Hpc.
  destruct (arithmetic_row_fetch program initial final row pc Hloaded Hprogram Hpc Hreplay)
    as [instruction [Hlookup Hcode]].
  pose proof (skipn_at_instruction program pc instruction Hlookup) as Hsuffix.
  rewrite Hsuffix in Htyped, Hconfig.
  assert (Hheight : (2 <= length values)%nat).
  { destruct (length values) as [ | [ | n]]; cbn [arithmeticStackTyped] in Htyped; tauto || lia. }
  destruct (stack_has_two_values values Hheight) as [prefix [a [b Hvalues]]].
  subst values c0.
  assert (Hdepth : 2 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]].
    cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  assert (Hselected : exists chosen, binaryArithmeticSelector chosen /\ row chosen = 1).
  { destruct (encoded_opcode_cases instruction _ Hcode) as [[_ Hop] | [_ Hop]].
    - exists COL_SEL_I32_ADD. split; [left; reflexivity | ].
      eapply full_opcode_selects; try eassumption.
      exact (eq_refl : filter (fun entry => Z.eqb (snd entry) CODE_I32_ADD) selectorCodes =
        [(COL_SEL_I32_ADD, CODE_I32_ADD)]).
    - exists COL_SEL_I32_SUB. split; [right; reflexivity | ].
      eapply full_opcode_selects; try eassumption.
      exact (eq_refl : filter (fun entry => Z.eqb (snd entry) CODE_I32_SUB) selectorCodes =
        [(COL_SEL_I32_SUB, CODE_I32_SUB)]). }
  destruct Hselected as [chosen [Hchoice Hchosen]].
  pose proof (full_binary_stack_shape row chosen Hprime Hfull Hchoice Hchosen Hdepth) as Hshape.
  pose proof (Range.full_write0_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hresult.
  destruct (stack_matches_binary_step row initial final prefix a b Hshape Hreplay Hstack Hresult)
    as [Hread0 [Hread1 HstackAfter]].
  destruct (arithmetic_row_pure_step program initial final row pc
    Hprime Hfull Hloaded Hprogram Hpc Hreplay) as [fetched [Hfetched Hpure]].
  assert (Hsame : fetched = instruction) by congruence. subst fetched.
  rewrite Hread0, Hread1 in Hpure.
  destruct (stack_config_binary_step z prefix a b (row COL_STACK_WRITE0_VALUE_LO)
    instruction (skipn (S pc) program) Hpure Hresult Hwf) as [Hstep HwfAfter].
  exists (stackConfig z (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]) (skipn (S pc) program)).
  split; [exact Hstep | ].
  unfold represents. split.
  - change (row COL_PC_AFTER = Z.of_nat (S pc)).
    destruct (arithmetic_row_next_pc program initial final row pc
      Hprime Hfull Hloaded Hprogram Hpc Hreplay) as [_ [_ Hnext]].
    replace (S pc) with (pc + 1)%nat by lia. exact Hnext.
  - split.
    + apply nth_error_Some. rewrite Hlookup. discriminate.
    + exists (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]). split; [exact HstackAfter | ].
      split.
      * rewrite length_app in Htyped |- *. cbn [length] in Htyped |- *.
        replace (length prefix + 2)%nat with (S (S (length prefix))) in Htyped by lia.
        cbn [arithmeticStackTyped] in Htyped.
        replace (length prefix + 1)%nat with (S (length prefix)) by lia. exact Htyped.
      * split; [reflexivity | exact HwfAfter].
Qed.

Theorem arithmetic_rows_refine_prefix :
  forall program z rest first pc initial final c0,
    Range.fieldPrime -> supportedProgram program -> loadedProgram program initial ->
    validCircuitRun initial (first :: rest) final ->
    Forall (fun (row : Witness) => row COL_IS_PROGRAM_ROW = 1) (first :: rest) ->
    represents program z initial (beforeBoundary first) pc c0 ->
    exists c1,
      Steps c0 c1 /\
      represents program z final (afterBoundary (finalRow first rest))
        (pc + length (first :: rest))%nat c1.
Proof.
  intros program z rest.
  pose proof arithmetic_row_refinement as Hrow.
  induction rest as [ | next rest IH];
    intros first pc initial final c0 Hprime Hsupported Hloaded Hrun Hprogram Hbefore.
  - destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hsat [Hreplay Htail]]].
    destruct Htail as [_ [_ Hempty]].
    assert (Hend : middle = final) by (inversion Hempty; congruence).
    subst middle.
    destruct (Hrow program z initial final first pc c0 Hprime Hsupported Hloaded
      Hsat (Forall_inv Hprogram) Hreplay Hbefore) as [c1 [Hstep Hafter]].
    exists c1. split.
    + pose proof (represents_wf _ _ _ _ _ _ Hbefore) as Hwf0.
      pose proof (represents_wf _ _ _ _ _ _ Hafter) as Hwf1.
      destruct c0, c1. eapply trans; try eassumption.
      apply Steps__refl. exact Hwf1.
    + simpl. replace (pc + 1)%nat with (S pc) by lia. exact Hafter.
  - destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hsat [Hreplay Htail]]].
    destruct (Hrow program z initial middle first pc c0 Hprime Hsupported Hloaded
      Hsat (Forall_inv Hprogram) Hreplay Hbefore) as [c1 [Hstep Hafter]].
    assert (HloadedMiddle : loadedProgram program middle).
    { eapply loaded_program_preserved; [exact Hloaded | exact Hreplay | ].
      intros memory address Hreadonly. apply read_only_row. exact Hreadonly. }
    assert (Hlink : Follows MemoryCatalog.continuityLinks first next).
    { destruct Hrun as [_ [Hchain _]]. inversion Hchain; subst; assumption. }
    pose proof Hafter as Hnext.
    rewrite (follows_boundary first next Hlink) in Hnext.
    destruct (IH next (S pc) middle final c1 Hprime Hsupported HloadedMiddle Htail
      (Forall_inv_tail Hprogram) Hnext) as [c2 [Hsteps Hfinal]].
    exists c2. split.
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
Theorem arithmetic_run_refinement :
  forall program z initial final first rest c0,
    Range.fieldPrime -> supportedProgram program -> loadedProgram program initial ->
    validCircuitRun initial (first :: rest) final ->
    Forall (fun (row : Witness) => row COL_IS_PROGRAM_ROW = 1) (first :: rest) ->
    represents program z initial (beforeBoundary first) 0%nat c0 ->
    exists c1,
      Steps c0 c1 /\
      represents program z final (afterBoundary (finalRow first rest))
        (length (first :: rest)) c1.
Proof.
  intros program z initial final first rest c0 Hprime Hsupported Hloaded
    Hrun Hprogram Hbefore.
  exact (arithmetic_rows_refine_prefix program z rest first 0%nat
    initial final c0 Hprime Hsupported Hloaded Hrun Hprogram Hbefore).
Qed.
