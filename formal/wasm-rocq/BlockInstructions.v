(* Straight-line rows under a block label. The protected stack prefix stays
   outside the label, while the instruction consumes only the inner values. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors Wasm2 WasmIntegers Add Sub Locals
  MemoryTrace MemoryCatalog ProgramBinding StraightLineSpec StackBindings StackSimulation LocalBindings LocalSimulation
  StraightLineInstructions CallSpec CallBody CallScope CallContext CallerRefinement SpecTecCalls.
Import ListNotations.
Open Scope Z_scope.

Definition FramedResult (prefix : list Z) (final : MemoryState) (row : Witness)
    (z : state) (values : list Z) (instruction : instr) (z' : state) (values' : list Z) : Prop :=
  Step (stackConfig z values [instruction]) (stackConfig z' values' []) /\
  wf_state z' /\ stackMatches final (afterBoundary row) (prefix ++ values') /\
  localsMatch final (afterBoundary row) z' /\ sameCaller z z' /\
  instructionStackHeight instruction (length values) = Some (length values').

Lemma framed_binary : forall prefix values z initial final row op chosen,
  fieldPrime -> FullRelation.satFull row -> Replay initial (eventsForRow memories row) final ->
  binaryArithmeticSelector chosen -> row chosen = 1 -> wf_state z -> wf_instr (BINOP I32 op) ->
  (2 <= length values)%nat -> stackMatches initial (beforeBoundary row) (prefix ++ values) ->
  localsMatch initial (beforeBoundary row) z ->
  boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
    boundaryStackBase (beforeBoundary row) ->
  (op = mk_binop__0 Inn_I32 ADD \/ op = mk_binop__0 Inn_I32 SUB) ->
  Step_pure [i32Constant (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
    i32Constant (row (nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)); admininstr_instr (BINOP I32 op)]
    [i32Constant (row COL_STACK_WRITE0_VALUE_LO)] ->
  exists values', FramedResult prefix final row z values (BINOP I32 op) z values'.
Proof.
  intros prefix values z initial final row op chosen Hprime Hfull Hreplay Hchoice Hchosen Hz Hi
    Hheight Hstack Hlocals Hlayout Hop Hpure.
  destruct (stack_has_two_values values Hheight) as [inner [a [b ->]]].
  rewrite app_assoc in Hstack.
  assert (Hdepth : row COL_STACK_FRAME_BASE_BEFORE + 2 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
    rewrite !length_app, !Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_binary_stack_shape row chosen Hprime Hfull Hchoice Hchosen
    ltac:(destruct Hstack as [Hbase _]; cbn in Hbase; lia)) as Hshape.
  pose proof (full_write0_u32 row Hprime Hfull) as Hbound.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hbound.
  destruct (stack_matches_binary_step row initial final (prefix ++ inner) a b Hshape Hreplay Hstack Hbound)
    as [Ha [Hb Hafter]].
  rewrite Ha, Hb in Hpure.
  assert (Hwf : wf_config (stackConfig z (inner ++ [a; b]) [BINOP I32 op])).
  { apply stack_config_wf; [exact Hz | | constructor; [exact Hi | constructor]].
    pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues.
    rewrite <- app_assoc in Hvalues. apply Forall_app in Hvalues. exact (proj2 Hvalues). }
  destruct (stack_config_binary_step z inner a b (row COL_STACK_WRITE0_VALUE_LO)
    (BINOP I32 op) [] Hpure Hbound Hwf) as [Hstep _].
  exists (inner ++ [row COL_STACK_WRITE0_VALUE_LO]).
  split; [exact Hstep | ]. split; [exact Hz | ]. split; [now rewrite app_assoc | ]. split.
  - assert (Hs : straightLineSelector chosen) by (destruct Hchoice as [-> | ->]; cbn; auto).
    eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z).
    + exact (proj1 (full_straight_line_local_gates row chosen Hprime Hfull Hs Hchosen)).
    + intros index Hindex. destruct (binary_stack_replay row initial final Hshape Hreplay) as [_ [_ Hwrites]].
      cbn [beforeBoundary boundaryLocalsBase boundaryStackBase] in *.
      rewrite Hwrites, (stack_write_address row Hshape), (stack_write_hi_address row Hshape).
      rewrite !update_other by cell_differs. reflexivity.
    + exact Hlocals.
  - split; [repeat split; reflexivity | ].
    rewrite !length_app. cbn [length]. destruct Hop as [-> | ->];
      replace (length inner + 2)%nat with (S (S (length inner))) by lia;
      cbn [instructionStackHeight]; f_equal; lia.
Qed.

Lemma framed_local_get : forall prefix values z initial final row x,
  fieldPrime -> FullRelation.satFull row -> Replay initial (eventsForRow memories row) final ->
  row COL_SEL_LOCAL_GET = 1 -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  wf_state z -> wf_instr (LOCAL_GET x) -> (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
    boundaryStackBase (beforeBoundary row) ->
  exists values', FramedResult prefix final row z values (LOCAL_GET x) z values'.
Proof.
  intros prefix values z initial final row x Hprime Hfull Hreplay Hchosen Hindex Hz Hi HindexBound Hstack Hlocals
    Hlayout.
  assert (Hchoice : straightLineSelector COL_SEL_LOCAL_GET) by (cbn; auto).
  destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [Hfbp _].
  pose proof (full_local_get_stack_shape row Hprime Hfull Hchosen
    ltac:(pose proof (value_bound row COL_SP_BEFORE); lia)) as Hshape.
  destruct (local_get_stack_replay row initial final Hshape Hreplay) as [Hread Hfinal].
  destruct (full_local_addresses row Hprime Hfull) as [HaddrLo HaddrHi].
  destruct (locals_match_lookup initial (beforeBoundary row) z x Hlocals HindexBound)
    as [value [Hlocal [Hbound Hcell]]].
  destruct Hstack as [Hbase [Hsp Hstackrest]].
  cbn [beforeBoundary boundaryLocalsBase boundaryStackBase boundarySP] in Hcell, Hbase, Hsp, Hlayout.
  rewrite (local_write_address row false true Hshape eq_refl), (local_write_hi_address row false true Hshape eq_refl),
    HaddrLo, Hindex in Hread.
  rewrite !update_other in Hread by cell_differs.
  rewrite Hcell in Hread. injection Hread as Hvalue.
  assert (Hstack : stackMatches initial (beforeBoundary row) (prefix ++ values)) by (split; [exact Hbase | split; assumption]).
  pose proof (full_relation_implies_local_get row z x Hchosen Hfull ltac:(congruence)) as Hsmall.
  change (Step (stackConfig z [] [LOCAL_GET x]) (stackConfig z [row COL_STACK_WRITE0_VALUE_LO] [])) in Hsmall.
  pose proof (full_write0_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hresult.
  pose proof (stack_matches_local_get row initial final (prefix ++ values) Hshape Hreplay Hstack Hresult) as Hafter.
  assert (Hwf : wf_config (stackConfig z (values ++ []) [LOCAL_GET x])).
  { rewrite app_nil_r. apply stack_config_wf; [exact Hz | | constructor; [exact Hi | constructor]].
    pose proof (proj1 (proj2 (proj2 Hstack))) as Hvalues.
    apply Forall_app in Hvalues. exact (proj2 Hvalues). }
  destruct (stack_config_step z z values [] [row COL_STACK_WRITE0_VALUE_LO] (LOCAL_GET x) []
    Hsmall ltac:(constructor; [exact Hresult | constructor]) Hz Hwf) as [Hstep _].
  rewrite app_nil_r in Hstep.
  exists (values ++ [row COL_STACK_WRITE0_VALUE_LO]).
  split; [exact Hstep | ]. split; [exact Hz | ]. split; [now rewrite app_assoc | ]. split.
  - eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z);
      [exact Hfbp | | exact Hlocals].
    intros index HindexLocal. cbn [beforeBoundary boundaryLocalsBase].
    rewrite Hfinal, (local_write_address row false true Hshape eq_refl),
      (local_write_hi_address row false true Hshape eq_refl).
    rewrite !update_other by cell_differs. reflexivity.
  - split; [repeat split; reflexivity | ]. cbn [instructionStackHeight]. rewrite length_app. cbn. f_equal; lia.
Qed.

Lemma framed_local_set : forall prefix values z initial final row x,
  fieldPrime -> FullRelation.satFull row -> Replay initial (eventsForRow memories row) final ->
  row COL_SEL_LOCAL_SET = 1 -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  wf_state z -> wf_instr (LOCAL_SET x) -> (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  (0 < length values)%nat ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
    boundaryStackBase (beforeBoundary row) ->
  exists values', FramedResult prefix final row z values (LOCAL_SET x)
    (with_local z x (i32Value (row COL_LOCAL_VALUE))) values'.
Proof.
  intros prefix values z initial final row x Hprime Hfull Hreplay Hchosen Hindex Hz Hi HindexBound Hheight Hstack Hlocals
    Hlayout.
  destruct (stack_has_last values Hheight) as [inner [value ->]]. rewrite app_assoc in Hstack.
  assert (Hdepth : 1 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_local_set_stack_shape row Hprime Hfull Hchosen Hdepth) as Hshape.
  destruct (full_local_addresses row Hprime Hfull) as [HaddrLo HaddrHi].
  assert (Hbelow : row COL_LOCAL_ADDR_LO < 2 * row COL_STACK_FRAME_BASE_BEFORE /\
    row COL_LOCAL_ADDR_HI < 2 * row COL_STACK_FRAME_BASE_BEFORE).
  { cbn [beforeBoundary boundaryLocalsBase boundaryStackBase] in Hlayout. rewrite HaddrHi, HaddrLo, Hindex.
    apply Nat2Z.inj_lt in HindexBound. rewrite N_nat_Z in HindexBound. lia. }
  destruct (stack_matches_pop row initial final (prefix ++ inner) value true Hshape Hreplay
    (fun _ => Hbelow) Hstack) as [Hread Hafter].
  assert (Hchoice : straightLineSelector COL_SEL_LOCAL_SET) by (cbn; auto).
  destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen) as [Hfbp _].
  pose proof (full_local_value_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_LOCAL_VALUE < i32Modulus) in Hresult.
  pose proof (full_relation_implies_local_set row z x Hchosen Hfull) as Hsmall. rewrite Hread in Hsmall.
  change (Step (stackConfig z [value] [LOCAL_SET x])
    (stackConfig (with_local z x (i32Value (row COL_LOCAL_VALUE))) [] [])) in Hsmall.
  pose proof (with_local_wf z x _ Hz (full_local_value_wf row Hprime Hfull)) as HzAfter.
  assert (Hwf : wf_config (stackConfig z (inner ++ [value]) [LOCAL_SET x])).
  { apply stack_config_wf; [exact Hz | | constructor; [exact Hi | constructor]].
    pose proof (proj1 (proj2 (proj2 Hstack))) as Hv. rewrite <- app_assoc in Hv.
    apply Forall_app in Hv. exact (proj2 Hv). }
  destruct (stack_config_step z (with_local z x (i32Value (row COL_LOCAL_VALUE)))
    inner [value] [] (LOCAL_SET x) [] Hsmall ltac:(constructor) HzAfter Hwf) as [Hstep _].
  rewrite app_nil_r in Hstep. exists inner.
  split; [exact Hstep | ]. split; [exact HzAfter | ]. split; [exact Hafter | ]. split.
  - eapply (locals_match_set initial final (beforeBoundary row) (afterBoundary row) z x (row COL_LOCAL_VALUE));
      [exact Hfbp | exact HindexBound | exact Hresult | | | exact Hlocals];
      destruct (local_set_stack_replay row initial final Hshape Hreplay) as [_ Hfinal].
    + cbn [beforeBoundary boundaryLocalsBase]. rewrite Hfinal, HaddrHi, HaddrLo, Hindex.
      rewrite update_other by cell_differs. apply update_same.
    + intros index HindexLocal Hneq. cbn [beforeBoundary boundaryLocalsBase].
      rewrite Hfinal, HaddrHi, HaddrLo, Hindex. rewrite !update_other; [reflexivity | | ].
      * right. intro Heq. apply (f_equal (@hd Z 0)) in Heq. cbn [hd] in Heq. lia.
      * right. intro Heq. apply (f_equal (@hd Z 0)) in Heq. cbn [hd] in Heq.
        apply Hneq. apply Nat2Z.inj. rewrite N_nat_Z. lia.
  - split.
    + split; [destruct z; reflexivity | ]. split; [destruct z; reflexivity | apply locals_length_after_set].
    + rewrite length_app. cbn [length]. replace (length inner + 1)%nat with (S (length inner)) by lia. reflexivity.
Qed.

Theorem framed_instruction : forall prefix values instruction z initial final row,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  instructionImage initial (row COL_PC_BEFORE) instruction ->
  wf_state z -> wf_instr instruction ->
  programTyped (length (stateLocals z)) (length values) [instruction] ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
    boundaryStackBase (beforeBoundary row) ->
  exists z' values', FramedResult prefix final row z values instruction z' values' /\
    row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\ frameRegistersPreserved row.
Proof.
  intros prefix values instruction z initial final row Hprime Hfull Hprogram Hreplay Himage Hz Hi Htyped Hstack Hlocals
    Hlayout.
  pose proof (instruction_row_next_pc _ _ _ _ Hprime Hfull Hprogram Hreplay Himage) as Hnext.
  destruct Himage as [Hop [_ [_ Hlocal]]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite Hop in Hfetch.
  assert (Hfinish : forall chosen,
    straightLineSelector chosen -> row chosen = 1 ->
    (exists z' values', FramedResult prefix final row z values instruction z' values') ->
    exists z' values', FramedResult prefix final row z values instruction z' values' /\
      row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\ frameRegistersPreserved row).
  { intros chosen Hchoice Hchosen [z' [values' Hresult]]. exists z', values'. split; [exact Hresult | ].
    split; [exact Hnext | ]. exact (proj2 (proj2 (full_body_context_quiet row chosen Hprime Hfull Hchoice Hchosen))). }
  destruct (encoded_opcode_cases instruction _ Hfetch) as [[-> Hcode] | [[-> Hcode] |
    [[x [-> Hcode]] | [x [-> Hcode]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_ADD); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_I32_ADD ltac:(cbn; auto) Hchosen).
    destruct (framed_binary prefix values z initial final row _ COL_SEL_I32_ADD Hprime Hfull Hreplay
      ltac:(left; reflexivity) Hchosen Hz Hi ltac:(destruct (length values) as [ | [ | n]]; try contradiction; lia)
      Hstack Hlocals Hlayout ltac:(left; reflexivity) (Add.full_relation_implies_spectec_add row Hprime Hchosen Hfull))
      as [values' Hresult]. eauto.
  - assert (Hchosen : row COL_SEL_I32_SUB = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_SUB); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_I32_SUB ltac:(cbn; auto) Hchosen).
    destruct (framed_binary prefix values z initial final row _ COL_SEL_I32_SUB Hprime Hfull Hreplay
      ltac:(right; reflexivity) Hchosen Hz Hi ltac:(destruct (length values) as [ | [ | n]]; try contradiction; lia)
      Hstack Hlocals Hlayout ltac:(right; reflexivity) (Sub.full_relation_implies_spectec_sub row Hprime Hchosen Hfull))
      as [values' Hresult]. eauto.
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_GET); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_LOCAL_GET ltac:(cbn; auto) Hchosen).
    pose proof (proj2 (proj2 (full_straight_line_local_gates row COL_SEL_LOCAL_GET Hprime Hfull ltac:(cbn; auto) Hchosen))) as Hgate.
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as Hindex. rewrite Hlocal in Hindex.
    destruct (framed_local_get prefix values z initial final row x Hprime Hfull Hreplay Hchosen ltac:(congruence)
      Hz Hi (proj1 Htyped) Hstack Hlocals Hlayout) as [values' Hresult]. eauto.
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_SET); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_LOCAL_SET ltac:(cbn; auto) Hchosen).
    pose proof (proj2 (proj2 (full_straight_line_local_gates row COL_SEL_LOCAL_SET Hprime Hfull ltac:(cbn; auto) Hchosen))) as Hgate.
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as Hindex. rewrite Hlocal in Hindex.
    destruct (framed_local_set prefix values z initial final row x Hprime Hfull Hreplay Hchosen ltac:(congruence)
      Hz Hi (proj1 Htyped) ltac:(destruct Htyped as [_ Ht]; destruct (length values); try contradiction; lia)
      Hstack Hlocals Hlayout) as [values' Hresult]. eauto.
Qed.
