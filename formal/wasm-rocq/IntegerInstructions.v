(* Ordinary row simulation with typed operand stacks and locals. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues R1CS Columns FullRelation Range Selectors Add Sub
  MemoryTrace MemoryCatalog StraightLineSpec StackBindings StackSimulation LocalBindings LocalSimulation
  IntegerMemory IntegerDecode IntegerStack IntegerLocals SpecTecCalls CallScope CallContext
  StraightLineInstructions ProgramBinding CallerRefinement StackTypes CallBody CallSpec.
Import ListNotations.
Open Scope Z_scope.

Definition valueLocalsMatch (memory : MemoryState) (fbp : Z) (locals : list val) : Prop :=
  forall index value, nth_error locals index = Some value ->
    CellMatches memory (fbp + Z.of_nat index) value.

Lemma value_locals_preserved : forall initial final fbp locals,
  (forall index limb, (index < length locals)%nat -> (limb = 0 \/ limb = 1) ->
    final stackId [2 * (fbp + Z.of_nat index) + limb] =
      initial stackId [2 * (fbp + Z.of_nat index) + limb]) ->
  valueLocalsMatch initial fbp locals -> valueLocalsMatch final fbp locals.
Proof.
  intros initial final fbp locals Hmemory Hlocals index value Hnth.
  eapply cell_matches_memory; [ | exact (Hlocals _ _ Hnth)].
  intros limb Hlimb. apply Hmemory; [apply nth_error_Some; rewrite Hnth; discriminate | exact Hlimb].
Qed.

Lemma value_locals_set : forall initial final fbp locals index value,
  (index < length locals)%nat ->
  CellMatches final (fbp + Z.of_nat index) value ->
  (forall other limb, (other < length locals)%nat -> other <> index -> (limb = 0 \/ limb = 1) ->
    final stackId [2 * (fbp + Z.of_nat other) + limb] =
      initial stackId [2 * (fbp + Z.of_nat other) + limb]) ->
  valueLocalsMatch initial fbp locals -> valueLocalsMatch final fbp (setAt locals index value).
Proof.
  intros initial final fbp locals index value Hindex Hcell Hmemory Hlocals other v Hnth.
  rewrite nth_error_set_at in Hnth by exact Hindex.
  destruct (Nat.eqb other index) eqn:Heq.
  - apply Nat.eqb_eq in Heq. subst other. injection Hnth as <-. exact Hcell.
  - apply Nat.eqb_neq in Heq. eapply cell_matches_memory; [ | exact (Hlocals _ _ Hnth)].
    intros limb Hlimb. apply Hmemory; [apply nth_error_Some; rewrite Hnth; discriminate | exact Heq | exact Hlimb].
Qed.

Lemma value_stack_step_prefix : forall z z' prefix consumed produced i,
  wf_state z -> wf_state z' -> Forall wf_val consumed -> Forall wf_val produced -> wf_instr i ->
  Step (valueStackConfig z consumed [i]) (valueStackConfig z' produced []) ->
  Step (valueStackConfig z (prefix ++ consumed) [i]) (valueStackConfig z' (prefix ++ produced) []).
Proof.
  intros z z' prefix consumed produced i Hz Hz' Hc Hp Hi Hstep.
  destruct prefix as [ | head rest]; [exact Hstep | ].
  assert (Hbefore : wf_config (valueStackConfig z consumed [i]))
    by (apply value_stack_config_wf; auto; constructor; auto; constructor).
  assert (Hafter : wf_config (valueStackConfig z' produced []))
    by (apply value_stack_config_wf; auto; constructor).
  pose proof (ctxt_instrs z (head :: rest)
    (map admininstr_val consumed ++ map admininstr_instr [i]) [] z'
    (map admininstr_val produced ++ []) Hstep eq_refl Hbefore Hafter) as H.
  unfold valueStackConfig. rewrite !map_app, <- !app_assoc.
  rewrite !app_nil_r in H. cbn [map] in H |- *. rewrite !app_nil_r. exact H.
Qed.

Lemma value_local_lookup : forall z x value,
  nth_error (stateLocals z) (N.to_nat (x :> N)) = Some value -> fun_local z x = value.
Proof.
  intros [s [locals module]] x value Hnth. unfold fun_local. cbn [stateLocals] in Hnth.
  exact (spec_lookup_from_nth_error _ _ _ _ _ Hnth).
Qed.

Lemma integer_local_get : forall row initial final prefix values z x value,
  fieldPrime -> FullRelation.satFull row -> row COL_SEL_LOCAL_GET = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  wf_state z -> wf_instr (LOCAL_GET x) ->
  nth_error (stateLocals z) (N.to_nat (x :> N)) = Some value ->
  valueStackMatches initial (beforeBoundary row) (prefix ++ values) ->
  valueLocalsMatch initial (row COL_LOCALS_FBP_BEFORE) (stateLocals z) ->
  row COL_LOCALS_FBP_BEFORE + Z.of_nat (length (stateLocals z)) <= row COL_STACK_FRAME_BASE_BEFORE ->
  Step (valueStackConfig z values [LOCAL_GET x]) (valueStackConfig z (values ++ [value]) []) /\
  valueStackMatches final (afterBoundary row) (prefix ++ (values ++ [value])) /\
  valueLocalsMatch final (row COL_LOCALS_FBP_BEFORE) (stateLocals z).
Proof.
  intros row initial final prefix values z x value Hprime Hfull Hget Hreplay Hindex Hz Hi Hnth Hstack Hlocals Hlayout.
  assert (HindexBound : (N.to_nat (x :> N) < length (stateLocals z))%nat)
    by (apply nth_error_Some; rewrite Hnth; discriminate).
  pose proof (Hlocals _ _ Hnth) as Hcell. rewrite N_nat_Z, <- Hindex in Hcell.
  assert (Hbelow : row COL_LOCALS_FBP_BEFORE + row COL_LOCAL_INDEX < row COL_STACK_FRAME_BASE_BEFORE).
  { apply Nat2Z.inj_lt in HindexBound. rewrite N_nat_Z, <- Hindex in HindexBound. lia. }
  split.
  - rewrite <- (app_nil_r values) at 1.
    apply value_stack_step_prefix; try assumption; [constructor | | ].
    + constructor; [eapply cell_matches_wf; exact Hcell | constructor].
    + change (Step (mk_config z [admininstr_LOCAL_GET x]) (mk_config z [admininstr_val value])).
      rewrite <- (value_local_lookup _ _ _ Hnth). apply read, Step_read__local_get.
  - split.
    + rewrite app_assoc. eapply local_get_value_stack; eassumption.
    + pose proof Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
      pose proof (full_local_get_stack_shape row Hprime Hfull Hget ltac:(lia)) as Hshape.
      destruct (local_get_stack_replay row initial final Hshape Hreplay) as [_ [_ Hwrites]].
      eapply value_locals_preserved; [ | exact Hlocals]. intros index limb Hbound Hlimb.
      assert (0 <= limb <= 1) by (destruct Hlimb; lia).
      rewrite Hwrites, (local_write_address _ _ _ Hshape eq_refl), (local_write_hi_address _ _ _ Hshape eq_refl).
      rewrite !update_other by cell_differs. reflexivity.
Qed.

Lemma integer_local_set : forall row initial final prefix values z x value,
  fieldPrime -> FullRelation.satFull row -> row COL_SEL_LOCAL_SET = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  wf_state z -> wf_instr (LOCAL_SET x) -> (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  valueStackMatches initial (beforeBoundary row) (prefix ++ (values ++ [value])) ->
  valueLocalsMatch initial (row COL_LOCALS_FBP_BEFORE) (stateLocals z) ->
  row COL_LOCALS_FBP_BEFORE + Z.of_nat (length (stateLocals z)) <= row COL_STACK_FRAME_BASE_BEFORE ->
  Step (valueStackConfig z (values ++ [value]) [LOCAL_SET x])
    (valueStackConfig (with_local z x value) values []) /\
  wf_state (with_local z x value) /\
  valueStackMatches final (afterBoundary row) (prefix ++ values) /\
  valueLocalsMatch final (row COL_LOCALS_FBP_BEFORE) (stateLocals (with_local z x value)).
Proof.
  intros row initial final prefix values z x value Hprime Hfull Hset Hreplay Hindex Hz Hi HindexBound Hstack Hlocals Hlayout.
  rewrite app_assoc in Hstack.
  assert (Hbelow : row COL_LOCALS_FBP_BEFORE + row COL_LOCAL_INDEX < row COL_STACK_FRAME_BASE_BEFORE).
  { apply Nat2Z.inj_lt in HindexBound. rewrite N_nat_Z, <- Hindex in HindexBound. lia. }
  destruct (local_set_value_stack _ _ _ _ _ Hprime Hfull Hset Hreplay Hbelow Hstack) as [Hafter Hcell].
  pose proof (cell_matches_wf _ _ _ Hcell) as Hv.
  pose proof (with_local_wf z x value Hz Hv) as Hz'.
  split.
  - rewrite <- (app_nil_r values) at 2.
    apply value_stack_step_prefix; try assumption; [constructor; [exact Hv | constructor] | constructor | ].
    apply Step__local_set.
  - split; [exact Hz' | ]. split; [exact Hafter | ].
    rewrite locals_after_set. apply value_locals_set with (initial := initial); try assumption.
    + now rewrite N_nat_Z, <- Hindex.
    + intros other limb Hbound Hneq Hlimb.
      pose proof Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
      rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
      pose proof (full_local_set_stack_shape row Hprime Hfull Hset ltac:(lia)) as Hshape.
      destruct (local_set_stack_replay row initial final Hshape Hreplay) as [_ [_ Hwrites]].
      destruct (full_local_addresses row Hprime Hfull) as [Hlo Hhi]. rewrite Hwrites, Hhi, Hlo.
      change (writeCell initial (row COL_LOCALS_FBP_BEFORE + row COL_LOCAL_INDEX)
        (row COL_LOCAL_VALUE) (row COL_LOCAL_VALUE_HI) stackId
        [2 * (row COL_LOCALS_FBP_BEFORE + Z.of_nat other) + limb] =
        initial stackId [2 * (row COL_LOCALS_FBP_BEFORE + Z.of_nat other) + limb]).
      apply write_cell_other; [ | exact Hlimb]. intros Heq. apply Hneq, Nat2Z.inj. rewrite N_nat_Z, <- Hindex. lia.
Qed.

Lemma integer_binary : forall row initial final prefix values z op chosen a b,
  fieldPrime -> FullRelation.satFull row -> binaryArithmeticSelector chosen -> row chosen = 1 ->
  Replay initial (eventsForRow memories row) final ->
  wf_state z -> wf_instr (BINOP I32 op) -> 0 <= a < i32Modulus -> 0 <= b < i32Modulus ->
  valueStackMatches initial (beforeBoundary row) (prefix ++ (values ++ [i32Value a; i32Value b])) ->
  valueLocalsMatch initial (row COL_LOCALS_FBP_BEFORE) (stateLocals z) ->
  row COL_LOCALS_FBP_BEFORE + Z.of_nat (length (stateLocals z)) <= row COL_STACK_FRAME_BASE_BEFORE ->
  Step_pure [i32Constant (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
    i32Constant (row (nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)); admininstr_instr (BINOP I32 op)]
    [i32Constant (row COL_STACK_WRITE0_VALUE_LO)] ->
  Step (valueStackConfig z (values ++ [i32Value a; i32Value b]) [BINOP I32 op])
    (valueStackConfig z (values ++ [i32Value (row COL_STACK_WRITE0_VALUE_LO)]) []) /\
  valueStackMatches final (afterBoundary row) (prefix ++ (values ++ [i32Value (row COL_STACK_WRITE0_VALUE_LO)])) /\
  valueLocalsMatch final (row COL_LOCALS_FBP_BEFORE) (stateLocals z).
Proof.
  intros row initial final prefix values z op chosen a b Hprime Hfull Hchoice Hchosen Hreplay Hz Hi Ha Hb
    Hstack Hlocals Hlayout Hpure.
  rewrite app_assoc in Hstack.
  pose proof Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_binary_stack_shape row chosen Hprime Hfull Hchoice Hchosen ltac:(lia)) as Hshape.
  pose proof (full_write0_u32 row Hprime Hfull) as Hr.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hr.
  destruct (value_stack_matches_binary _ _ _ _ _ _ Hshape Hreplay Hstack (proj1 Ha) (proj1 Hb) Hr)
    as [HreadA [HreadB Hafter]].
  rewrite HreadA, HreadB in Hpure.
  split.
  - apply value_stack_step_prefix; try assumption.
    + constructor; [apply i32_value_wf; exact Ha | constructor; [apply i32_value_wf; exact Hb | constructor]].
    + constructor; [apply i32_value_wf; exact Hr | constructor].
    + apply pure. exact Hpure.
  - split; [now rewrite app_assoc | ].
    destruct (binary_stack_replay row initial final Hshape Hreplay) as [_ [_ Hwrites]].
    eapply value_locals_preserved; [ | exact Hlocals]. intros index limb Hbound Hlimb.
    assert (0 <= limb <= 1) by (destruct Hlimb; lia).
    rewrite Hwrites, (stack_write_address _ Hshape), (stack_write_hi_address _ Hshape).
    rewrite !update_other by cell_differs. reflexivity.
Qed.

Lemma values_split_two : forall (values : list val),
  (2 <= length values)%nat -> exists prefix a b, values = prefix ++ [a; b].
Proof.
  intros values Hlen. assert (Hne : values <> []) by (intros ->; cbn in Hlen; lia).
  destruct (exists_last Hne) as [rest [b ->]].
  assert (Hrest : rest <> []) by (intros ->; cbn in Hlen; lia).
  destruct (exists_last Hrest) as [prefix [a ->]]. exists prefix, a, b.
  rewrite <- app_assoc. reflexivity.
Qed.

Lemma value_types_set : forall locals index old value,
  nth_error locals index = Some old -> valueType value = valueType old ->
  valueTypes (setAt locals index value) = valueTypes locals.
Proof.
  induction locals as [ | head rest IH]; intros [ | index] old value Hnth Htype; cbn in Hnth; try discriminate;
    cbn [setAt valueTypes map].
  - injection Hnth as <-. now rewrite Htype.
  - f_equal. eapply IH; eassumption.
Qed.

Lemma integer_binary_types : forall memory boundary below values op outputTypes callees locals,
  (op = mk_binop__0 Inn_I32 ADD \/ op = mk_binop__0 Inn_I32 SUB) ->
  valueStackMatches memory boundary (below ++ values) ->
  sourceTypes locals callees (BINOP I32 op) (valueTypes values) = Some outputTypes ->
  exists prefix a b,
    values = prefix ++ [i32Value a; i32Value b] /\
    0 <= a < i32Modulus /\ 0 <= b < i32Modulus /\
    outputTypes = valueTypes prefix ++ [valtype_I32].
Proof.
  intros memory boundary below values op outputTypes callees locals Hop Hstack Htypes.
  assert (Hreplace : replaceStackTypes [valtype_I32; valtype_I32] [valtype_I32]
      (valueTypes values) = Some outputTypes) by (destruct Hop as [-> | ->]; exact Htypes).
  assert (Hlen : (2 <= length values)%nat).
  { apply replace_stack_types_some in Hreplace as [prefix [Hinput _]].
    pose proof (f_equal (@length valtype) Hinput). unfold valueTypes in H. rewrite length_map, length_app in H.
    cbn [length] in H. lia. }
  destruct (values_split_two _ Hlen) as [prefix [a [b ->]]].
  unfold valueTypes in Hreplace. rewrite map_app in Hreplace. cbn [map] in Hreplace.
  destruct (replace_stack_types_suffix [valtype_I32; valtype_I32] [valtype_I32]
    (map valueType prefix) [valueType a; valueType b] outputTypes eq_refl Hreplace) as [Hinput Houtput].
  injection Hinput as Ha Hb.
  pose proof (proj2 (proj2 Hstack)) as Hcells.
  assert (HcellA : CellMatches memory (boundaryStackBase boundary + Z.of_nat (length (below ++ prefix))) a).
  { apply Hcells. rewrite app_assoc, nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity. }
  assert (HcellB : CellMatches memory (boundaryStackBase boundary + Z.of_nat (S (length (below ++ prefix)))) b).
  { apply Hcells. rewrite app_assoc, nth_error_app2 by lia.
    replace (S (length (below ++ prefix)) - length (below ++ prefix))%nat with 1%nat by lia. reflexivity. }
  destruct (cell_i32_operand _ _ _ HcellA Ha) as [rawA [HboundA ->]].
  destruct (cell_i32_operand _ _ _ HcellB Hb) as [rawB [HboundB ->]].
  exists prefix, rawA, rawB. auto.
Qed.

Definition IntegerResult (prefix : list val) (final : MemoryState) (row : Witness)
    (z : state) (values : list val) (i : instr) (types : list valtype) (z' : state) (values' : list val) : Prop :=
  Step (valueStackConfig z values [i]) (valueStackConfig z' values' []) /\
  wf_state z' /\ valueStackMatches final (afterBoundary row) (prefix ++ values') /\
  valueLocalsMatch final (row COL_LOCALS_FBP_BEFORE) (stateLocals z') /\
  sameCaller z z' /\ valueTypes (stateLocals z') = valueTypes (stateLocals z) /\ valueTypes values' = types.

Theorem integer_instruction : forall prefix values instruction z initial final row callees outputTypes,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  instructionImage initial (row COL_PC_BEFORE) instruction ->
  wf_state z -> wf_instr instruction ->
  sourceTypes (valueTypes (stateLocals z)) callees instruction (valueTypes values) = Some outputTypes ->
  valueStackMatches initial (beforeBoundary row) (prefix ++ values) ->
  valueLocalsMatch initial (row COL_LOCALS_FBP_BEFORE) (stateLocals z) ->
  row COL_LOCALS_FBP_BEFORE + Z.of_nat (length (stateLocals z)) <= row COL_STACK_FRAME_BASE_BEFORE ->
  exists z' values', IntegerResult prefix final row z values instruction outputTypes z' values' /\
    row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\ frameRegistersPreserved row.
Proof.
  intros prefix values instruction z initial final row callees outputTypes Hprime Hfull Hprogram Hreplay Himage Hz Hi
    Htyped Hstack Hlocals Hlayout.
  pose proof (instruction_row_next_pc _ _ _ _ Hprime Hfull Hprogram Hreplay Himage) as Hnext.
  destruct Himage as [Hop [_ [_ Hlocal]]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite Hop in Hfetch.
  assert (Hfinish : forall chosen,
    straightLineSelector chosen -> row chosen = 1 ->
    (exists z' values', IntegerResult prefix final row z values instruction outputTypes z' values') ->
    exists z' values', IntegerResult prefix final row z values instruction outputTypes z' values' /\
      row COL_PC_AFTER = row COL_PC_BEFORE + 1 /\ frameRegistersPreserved row).
  { intros chosen Hchoice Hchosen [z' [values' Hresult]]. exists z', values'. split; [exact Hresult | ].
    split; [exact Hnext | ]. exact (proj2 (proj2 (full_body_context_quiet row chosen Hprime Hfull Hchoice Hchosen))). }
  destruct (encoded_opcode_cases instruction _ Hfetch) as [[-> Hcode] | [[-> Hcode] |
    [[x [-> Hcode]] | [x [-> Hcode]]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_ADD); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_I32_ADD ltac:(cbn; auto) Hchosen).
    destruct (integer_binary_types _ _ _ _ _ _ _ _ (or_introl eq_refl) Hstack Htyped)
      as [inner [a [b [-> [Ha [Hb ->]]]]]].
    destruct (integer_binary _ _ _ _ _ _ _ COL_SEL_I32_ADD _ _ Hprime Hfull (or_introl eq_refl) Hchosen
      Hreplay Hz Hi Ha Hb Hstack Hlocals Hlayout (Add.full_relation_implies_spectec_add row Hprime Hchosen Hfull))
      as [Hstep [Hafter Hlocals']].
    exists z, (inner ++ [i32Value (row COL_STACK_WRITE0_VALUE_LO)]).
    split; [exact Hstep | ]. split; [exact Hz | ]. split; [exact Hafter | ]. split; [exact Hlocals' | ].
    split; [repeat split; reflexivity | ]. split; [reflexivity | unfold valueTypes; rewrite map_app; reflexivity].
  - assert (Hchosen : row COL_SEL_I32_SUB = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_SUB); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_I32_SUB ltac:(cbn; auto) Hchosen).
    destruct (integer_binary_types _ _ _ _ _ _ _ _ (or_intror eq_refl) Hstack Htyped)
      as [inner [a [b [-> [Ha [Hb ->]]]]]].
    destruct (integer_binary _ _ _ _ _ _ _ COL_SEL_I32_SUB _ _ Hprime Hfull (or_intror eq_refl) Hchosen
      Hreplay Hz Hi Ha Hb Hstack Hlocals Hlayout (Sub.full_relation_implies_spectec_sub row Hprime Hchosen Hfull))
      as [Hstep [Hafter Hlocals']].
    exists z, (inner ++ [i32Value (row COL_STACK_WRITE0_VALUE_LO)]).
    split; [exact Hstep | ]. split; [exact Hz | ]. split; [exact Hafter | ]. split; [exact Hlocals' | ].
    split; [repeat split; reflexivity | ]. split; [reflexivity | unfold valueTypes; rewrite map_app; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_GET); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_LOCAL_GET ltac:(cbn; auto) Hchosen).
    pose proof (proj2 (proj2 (full_straight_line_local_gates row COL_SEL_LOCAL_GET Hprime Hfull ltac:(cbn; auto) Hchosen))) as Hgate.
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as Hindex. rewrite Hlocal in Hindex.
    unfold sourceTypes, instructionTypes, valueTypes in Htyped. rewrite nth_error_map in Htyped.
    destruct (nth_error (stateLocals z) (N.to_nat (x :> N))) as [value | ] eqn:Hnth; [ | discriminate].
    cbn [option_map] in Htyped.
    change (replaceStackTypes [] [valueType value] (map valueType values) = Some outputTypes) in Htyped.
    rewrite <- (app_nil_r (map valueType values)) in Htyped at 1.
    rewrite replace_stack_types_exact in Htyped. injection Htyped as <-.
    destruct (integer_local_get row initial final prefix values z x value Hprime Hfull Hchosen Hreplay ltac:(congruence)
      Hz Hi Hnth Hstack Hlocals Hlayout) as [Hstep [Hafter Hlocals']].
    exists z, (values ++ [value]).
    split; [exact Hstep | ]. split; [exact Hz | ]. split; [exact Hafter | ]. split; [exact Hlocals' | ].
    split; [repeat split; reflexivity | ]. split; [reflexivity | unfold valueTypes; rewrite map_app; reflexivity].
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_SET); try eassumption; reflexivity).
    apply (Hfinish COL_SEL_LOCAL_SET ltac:(cbn; auto) Hchosen).
    pose proof (proj2 (proj2 (full_straight_line_local_gates row COL_SEL_LOCAL_SET Hprime Hfull ltac:(cbn; auto) Hchosen))) as Hgate.
    pose proof (circuit_row_local_index_fetch _ _ _ Hreplay Hgate) as Hindex. rewrite Hlocal in Hindex.
    unfold sourceTypes, instructionTypes, valueTypes in Htyped. rewrite nth_error_map in Htyped.
    destruct (nth_error (stateLocals z) (N.to_nat (x :> N))) as [old | ] eqn:Hnth; [ | discriminate].
    cbn [option_map] in Htyped.
    assert (Hne : values <> []).
    { intros ->. discriminate Htyped. }
    destruct (exists_last Hne) as [inner [value ->]]. rewrite map_app in Htyped. cbn [map] in Htyped.
    destruct (replace_stack_types_suffix [valueType old] [] (map valueType inner) [valueType value]
      outputTypes eq_refl Htyped) as [Htype Houtput].
    injection Htype as Htype. rewrite app_nil_r in Houtput. subst outputTypes.
    assert (Hbound : (N.to_nat (x :> N) < length (stateLocals z))%nat)
      by (apply nth_error_Some; rewrite Hnth; discriminate).
    destruct (integer_local_set row initial final prefix inner z x value Hprime Hfull Hchosen Hreplay ltac:(congruence)
      Hz Hi Hbound Hstack Hlocals Hlayout) as [Hstep [Hz' [Hafter Hlocals']]].
    exists (with_local z x value), inner.
    split; [exact Hstep | ]. split; [exact Hz' | ]. split; [exact Hafter | ]. split; [exact Hlocals' | ].
    split.
    + split; [destruct z; reflexivity | ]. split; [destruct z; reflexivity | apply locals_length_after_set].
    + split; [rewrite locals_after_set; eapply value_types_set; eassumption | reflexivity].
Qed.

Lemma written_cell_frame_effect : forall initial final row chosen target lo hi,
  fieldPrime -> FullRelation.satFull row -> straightLineSelector chosen -> row chosen = 1 ->
  Replay initial (eventsForRow memories row) final -> row COL_LOCALS_FBP_BEFORE <= target ->
  (forall address, final stackId address = writeCell initial target lo hi stackId address) ->
  FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros initial final row chosen target lo hi Hprime Hfull Hchoice Hchosen Hreplay Htarget Hmemory.
  constructor; [ | exact (body_row_call_context _ _ _ _ Hprime Hfull Hchoice Hchosen Hreplay)].
  intros index limb Hindex Hlimb. rewrite Hmemory. apply write_cell_other; auto; lia.
Qed.

Lemma integer_row_scope : forall row initial final prefix values i callees locals types,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> encodedOpcode i = Some (row COL_OPCODE_CODE) ->
  sourceTypes locals callees i (valueTypes values) = Some types ->
  valueStackMatches initial (beforeBoundary row) (prefix ++ values) ->
  row COL_LOCALS_FBP_BEFORE <= row COL_STACK_FRAME_BASE_BEFORE ->
  FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros row initial final prefix values i callees locals types Hprime Hfull Hprogram Hreplay Hcode Htypes
    [Hbase [Hsp Hcells]] Hlayout.
  cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
  rewrite length_app, Nat2Z.inj_add in Hsp.
  pose proof (source_types_height _ _ _ _ _ Htypes) as Hheight.
  unfold valueTypes in Hheight. rewrite length_map in Hheight.
  destruct (encoded_opcode_cases _ _ Hcode) as [[-> Hcode'] | [[-> Hcode'] | [[x [-> Hcode']] | [x [-> Hcode']]]]].
  - assert (Hchosen : row COL_SEL_I32_ADD = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_ADD); try eassumption; reflexivity).
    assert (Hdepth : (2 <= length values)%nat)
      by (destruct (length values) as [ | [ | n]]; try discriminate; lia).
    pose proof (full_binary_stack_shape row COL_SEL_I32_ADD Hprime Hfull (or_introl eq_refl) Hchosen ltac:(lia)) as Hshape.
    destruct (binary_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hmemory]].
    eapply written_cell_frame_effect with (chosen := COL_SEL_I32_ADD) (target := row COL_SP_BEFORE - 2); try eassumption; try solve [cbn; auto].
    + lia.
    + intros address. rewrite Hmemory, (stack_write_address _ Hshape), (stack_write_hi_address _ Hshape). reflexivity.
  - assert (Hchosen : row COL_SEL_I32_SUB = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_I32_SUB); try eassumption; reflexivity).
    assert (Hdepth : (2 <= length values)%nat)
      by (destruct (length values) as [ | [ | n]]; try discriminate; lia).
    pose proof (full_binary_stack_shape row COL_SEL_I32_SUB Hprime Hfull (or_intror eq_refl) Hchosen ltac:(lia)) as Hshape.
    destruct (binary_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hmemory]].
    eapply written_cell_frame_effect with (chosen := COL_SEL_I32_SUB) (target := row COL_SP_BEFORE - 2); try eassumption; try solve [cbn; auto].
    + lia.
    + intros address. rewrite Hmemory, (stack_write_address _ Hshape), (stack_write_hi_address _ Hshape). reflexivity.
  - assert (Hchosen : row COL_SEL_LOCAL_GET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_GET); try eassumption; reflexivity).
    pose proof (full_local_get_stack_shape row Hprime Hfull Hchosen ltac:(lia)) as Hshape.
    destruct (local_get_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hmemory]].
    eapply written_cell_frame_effect with (chosen := COL_SEL_LOCAL_GET) (target := row COL_SP_BEFORE); try eassumption; try solve [cbn; auto].
    + lia.
    + intros address. rewrite Hmemory, (local_write_address _ _ _ Hshape eq_refl),
        (local_write_hi_address _ _ _ Hshape eq_refl). reflexivity.
  - assert (Hchosen : row COL_SEL_LOCAL_SET = 1)
      by (eapply full_opcode_selects with (code := OpcodeCatalog.CODE_LOCAL_SET); try eassumption; reflexivity).
    assert (Hdepth : (1 <= length values)%nat)
      by (destruct (length values) as [ | n]; try discriminate; lia).
    pose proof (full_local_set_stack_shape row Hprime Hfull Hchosen ltac:(lia)) as Hshape.
    destruct (local_set_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hmemory]].
    destruct (full_local_addresses row Hprime Hfull) as [Hlo Hhi].
    eapply written_cell_frame_effect with (chosen := COL_SEL_LOCAL_SET) (target := row COL_LOCALS_FBP_BEFORE + row COL_LOCAL_INDEX); try eassumption; try solve [cbn; auto].
    + pose proof (value_bound row COL_LOCAL_INDEX). lia.
    + intros address. rewrite Hmemory, Hhi, Hlo. reflexivity.
Qed.
