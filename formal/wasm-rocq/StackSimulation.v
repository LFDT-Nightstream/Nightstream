(* Relates stack memory and row boundaries to contextual SpecTec steps. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  OpcodeCatalog Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Locals StackBindings StraightLineSpec.
Import ListNotations.
Open Scope Z_scope.

Lemma instruction_stack_height_delta : forall instruction before after,
  instructionStackHeight instruction before = Some after ->
  instructionStackDelta instruction = Some (Z.of_nat after - Z.of_nat before).
Proof.
  intros instruction before after Hheight.
  destruct instruction; cbn [instructionStackHeight instructionStackDelta] in *; try discriminate;
    repeat match goal with
    | x : numtype |- _ => destruct x; try discriminate
    | x : binop_ |- _ => destruct x; try discriminate
    | x : Inn |- _ => destruct x; try discriminate
    | x : binop_Inn |- _ => destruct x; try discriminate
    end;
    destruct before as [ | [ | before]]; cbn in Hheight; try discriminate;
    inversion Hheight; subst; f_equal; lia.
Qed.

Lemma program_typed_binary : forall localCount height op rest,
  programTyped localCount height (BINOP I32 op :: rest) ->
  match height with S (S remaining) => programTyped localCount (S remaining) rest | _ => False end.
Proof.
  intros localCount height op rest Htyped.
  destruct op; repeat match goal with
    | x : Inn |- _ => destruct x
    | x : binop_Inn |- _ => destruct x
    end; cbn [programTyped instructionStackHeight] in Htyped; try contradiction;
    destruct height as [ | [ | remaining]]; exact Htyped.
Qed.

Lemma follows_boundary : forall previous next,
  Follows MemoryCatalog.continuityLinks previous next ->
  afterBoundary previous = beforeBoundary next.
Proof.
  intros previous next Hlinks.
  unfold afterBoundary, beforeBoundary. f_equal;
    apply Hlinks; vm_compute; tauto.
Qed.

Lemma stack_matches_height : forall memory boundary values,
  stackMatches memory boundary values -> stackHeight boundary = length values.
Proof.
  intros memory boundary values [_ [Hsp _]]. unfold stackHeight. rewrite Hsp.
  replace (boundaryStackBase boundary + Z.of_nat (length values) - boundaryStackBase boundary)
    with (Z.of_nat (length values)) by lia. apply Nat2Z.id.
Qed.

Lemma represented_values : forall (typed : nat -> list instr -> Prop) z memory boundary code c,
  (typed (stackHeight boundary) code /\ represents z memory boundary code c) <->
  exists values, stackMatches memory boundary values /\ typed (length values) code /\
    c = stackConfig z values code /\ wf_config c /\ localsMatch memory boundary z.
Proof.
  intros typed z memory boundary code c. split.
  - intros [Ht [values [Hs Hr]]]. exists values. split; [exact Hs | ].
    split; [rewrite <- (stack_matches_height _ _ _ Hs); exact Ht | exact Hr].
  - intros [values [Hs [Ht Hr]]]. split.
    + rewrite (stack_matches_height _ _ _ Hs). exact Ht.
    + exists values. auto.
Qed.

Lemma represents_wf : forall z memory boundary code c,
  represents z memory boundary code c -> wf_config c.
Proof. intros z memory boundary code c [values [_ [_ [Hwf _]]]]. exact Hwf. Qed.

Lemma loaded_program_preserved : forall program before events after,
  loadedProgram program before -> Replay before events after ->
  (forall memory address,
    readOnlyMemory memory MemoryCatalog.memories = true ->
    Forall (NoWriteTo memory address) events) ->
  loadedProgram program after.
Proof.
  intros program before events after [Hopcode [Hkind [Hpc Hlocal]]] Hreplay Hreadonly.
  repeat split; intros.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ opcode_memory_read_only)). apply Hopcode.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ edge_kind_memory_read_only)). apply Hkind.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ pc_memory_read_only)). apply Hpc.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay
      (Hreadonly _ _ local_index_memory_read_only)). apply Hlocal.
Qed.

Lemma program_row_fetch : forall program initial final (w : Witness) pc,
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

Lemma program_row_next_pc : forall program initial final (w : Witness) pc,
  Range.fieldPrime -> FullRelation.satFull w -> loadedProgram program initial ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_BEFORE = Z.of_nat pc ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PC_ROM_ACTIVE = 1 /\ w COL_CONTROL_CHOICE = 0 /\
  w COL_PC_AFTER = Z.of_nat (pc + 1).
Proof.
  intros program initial final w pc Hprime Hfull [_ [Hkind [HpcRom _]]] Hprogram Hpc Hreplay.
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

Lemma stack_config_wf : forall z values code,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values ->
  Forall wf_instr code -> wf_config (stackConfig z values code).
Proof.
  intros z values code Hz Hvalues Hcode. constructor; [exact Hz | ].
  apply Forall_app. split; apply Forall_map.
  - eapply Forall_impl; [ | exact Hvalues]. intros value H. apply i32_constant_wf. exact H.
  - eapply Forall_impl; [ | exact Hcode]. intros i H. destruct H; cbn; solve [constructor | econstructor; eassumption].
Qed.

Lemma stack_config_step : forall z z' prefix consumed produced instruction code,
  Step (stackConfig z consumed [instruction]) (stackConfig z' produced []) ->
  Forall (fun value => 0 <= value < i32Modulus) produced -> wf_state z' ->
  wf_config (stackConfig z (prefix ++ consumed) (instruction :: code)) ->
  Step (stackConfig z (prefix ++ consumed) (instruction :: code))
    (stackConfig z' (prefix ++ produced) code) /\
  wf_config (stackConfig z' (prefix ++ produced) code).
Proof.
  intros z z' prefix consumed produced instruction code Hstep Hbounds Hz' Hwf.
  unfold stackConfig in *. rewrite !map_app in *. cbn [map] in *.
  repeat rewrite <- app_assoc in *.
  inversion Hwf as [state instructions Hz Hall]; subst.
  apply Forall_app in Hall. destruct Hall as [Hprefix Htail].
  apply Forall_app in Htail. destruct Htail as [Hconsumed Htail].
  pose proof (Forall_inv Htail) as Hinstruction.
  pose proof (Forall_inv_tail Htail) as Hcode.
  assert (Hproduced : Forall wf_admininstr (map i32Constant produced)).
  { apply Forall_map. eapply Forall_impl; [ | exact Hbounds].
    intros value Hvalue. apply i32_constant_wf. exact Hvalue. }
  assert (HsmallBefore : wf_config
    (mk_config z (map i32Constant consumed ++ [admininstr_instr instruction]))).
  { constructor; [exact Hz | ]. apply Forall_app. split; [exact Hconsumed | ].
    constructor; [exact Hinstruction | constructor]. }
  assert (HsmallAfter : wf_config (mk_config z' (map i32Constant produced ++ []))).
  { constructor; [exact Hz' | ]. rewrite app_nil_r. exact Hproduced. }
  split.
  - assert (Hmap : map admininstr_val (map i32Value prefix) = map i32Constant prefix).
    { rewrite map_map. reflexivity. }
    rewrite <- Hmap.
    replace (map i32Constant consumed ++ admininstr_instr instruction :: map admininstr_instr code)
      with ((map i32Constant consumed ++ [admininstr_instr instruction]) ++ map admininstr_instr code)
      by (rewrite <- app_assoc; reflexivity).
    replace (map i32Constant produced ++ map admininstr_instr code)
      with ((map i32Constant produced ++ []) ++ map admininstr_instr code)
      by (rewrite app_nil_r; reflexivity).
    destruct prefix as [ |head prefix], code as [ |next code].
    + cbn [map app]. rewrite !app_nil_r in *. exact Hstep.
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
  - constructor; [exact Hz' | ]. apply Forall_app. split; [exact Hprefix | ].
    apply Forall_app. auto.
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
  apply (stack_config_step z z prefix [a; b] [result] instruction code).
  - apply pure. exact Hpure.
  - constructor; [exact Hresult | constructor].
  - inversion Hwf; assumption.
  - exact Hwf.
Qed.
