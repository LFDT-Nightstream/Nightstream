(* Local instruction simulation over the complete stack and active frame. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Selectors Wasm2 WasmIntegers Locals
  MemoryTrace MemoryCatalog StraightLineSpec StackSimulation
  LocalBindings StackBindings.
Import ListNotations.
Open Scope Z_scope.

Fixpoint setAt {A : Type} (values : list A) (index : nat) (value : A) : list A :=
  match values, index with
  | [], _ => []
  | _ :: rest, O => value :: rest
  | head :: rest, S index => head :: setAt rest index value
  end.

Lemma list_update_set_at : forall (A : Type) (values : list A) n value,
  list_update_func values n (fun _ => value) = setAt values (N.to_nat n) value.
Proof.
  intros A values. induction values as [ | head rest IH]; intros n value.
  - reflexivity.
  - destruct n as [ |p]; [reflexivity | ].
    cbn [list_update_func].
    replace (N.to_nat (Npos p)) with (S (N.to_nat (N.pred (Npos p))))
      by (rewrite <- N2Nat.inj_succ; rewrite N.succ_pred_pos by lia; reflexivity).
    cbn [setAt]. f_equal. apply IH.
Qed.

Lemma length_set_at : forall (A : Type) (values : list A) index value,
  length (setAt values index value) = length values.
Proof. intros A values. induction values; intros [ |index] value; cbn; auto. Qed.

Lemma nth_error_set_at : forall (A : Type) (values : list A) index value target,
  (index < length values)%nat ->
  nth_error (setAt values index value) target =
    if Nat.eqb target index then Some value else nth_error values target.
Proof.
  intros A values. induction values as [ |head rest IH]; intros [ |index] value [ |target] Hbound;
    cbn in *; try lia; try reflexivity.
  apply IH. lia.
Qed.

Lemma locals_after_set : forall z x value,
  stateLocals (with_local z x value) = setAt (stateLocals z) (N.to_nat (x :> N)) value.
Proof. intros [s f] x value. cbn [stateLocals with_local]. apply list_update_set_at. Qed.

Lemma locals_length_after_set : forall z x value,
  length (stateLocals (with_local z x value)) = length (stateLocals z).
Proof. intros. rewrite locals_after_set. apply length_set_at. Qed.

Lemma locals_match_preserved : forall initial final before after z,
  boundaryLocalsBase after = boundaryLocalsBase before ->
  (forall address, final MemoryCatalog.localLoId address = initial MemoryCatalog.localLoId address) ->
  localsMatch initial before z -> localsMatch final after z.
Proof.
  intros initial final before after z Hbase Hmemory Hlocals index v Hlookup.
  destruct (Hlocals index v Hlookup) as [value [Hv [Hbound Hcell]]].
  exists value. split; [exact Hv | ]. split; [exact Hbound | ].
  rewrite Hbase, Hmemory. exact Hcell.
Qed.

Lemma locals_match_set : forall initial final before after z x value,
  boundaryLocalsBase after = boundaryLocalsBase before ->
  (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  0 <= value < i32Modulus ->
  (forall address, final MemoryCatalog.localLoId address =
    update initial MemoryCatalog.localLoId [boundaryLocalsBase before; Z.of_N (x :> N)] value
      MemoryCatalog.localLoId address) ->
  localsMatch initial before z ->
  localsMatch final after (with_local z x (i32Value value)).
Proof.
  intros initial final before after z x value Hbase Hindex Hvalue Hmemory Hlocals index v Hlookup.
  rewrite locals_after_set, nth_error_set_at in Hlookup by exact Hindex.
  destruct (Nat.eqb index (N.to_nat (x :> N))) eqn:Hsame.
  - apply Nat.eqb_eq in Hsame. subst index.
    injection Hlookup as <-. exists value. split; [reflexivity | ]. split; [exact Hvalue | ].
    rewrite Hbase, Hmemory, N_nat_Z. apply update_same.
  - apply Nat.eqb_neq in Hsame.
    destruct (Hlocals index v Hlookup) as [old [Hv [Hbound Hcell]]].
    exists old. split; [exact Hv | ]. split; [exact Hbound | ].
    rewrite Hbase, Hmemory, update_other; [exact Hcell | ].
    right. intro Heq. injection Heq as Heq.
    apply Hsame. apply Nat2Z.inj. rewrite N_nat_Z. symmetry. exact Heq.
Qed.

Lemma seq_nth_list_nth : forall (A : Type) (values : list A) index default,
  seq.nth default values index = List.nth index values default.
Proof. intros A values. induction values; intros [ |index] default; cbn; auto. Qed.

Lemma locals_match_lookup : forall memory boundary z x,
  localsMatch memory boundary z -> (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  exists value, fun_local z x = i32Value value /\ 0 <= value < i32Modulus /\
    memory MemoryCatalog.localLoId [boundaryLocalsBase boundary; Z.of_N (x :> N)] = Some value.
Proof.
  intros memory boundary z x Hlocals Hbound.
  destruct (nth_error (stateLocals z) (N.to_nat (x :> N))) as [v | ] eqn:Hlookup.
  2: { apply nth_error_None in Hlookup. lia. }
  destruct (Hlocals _ _ Hlookup) as [value [Hv [Hvalue Hcell]]].
  exists value. rewrite N_nat_Z in Hcell. split; [ | auto].
  destruct z as [s f]. unfold fun_local, lookup_total. cbn [stateLocals] in Hlookup.
  rewrite <- Hv. apply nth_error_nth with (d := default_val) in Hlookup.
  rewrite seq_nth_list_nth. exact Hlookup.
Qed.

Lemma forall_set_at : forall (A : Type) (P : A -> Prop) values index value,
  Forall P values -> P value -> Forall P (setAt values index value).
Proof.
  intros A P values. induction values; intros [ |index] value Hall Hv;
    cbn; inversion Hall; subst; constructor; auto.
Qed.

Lemma with_local_wf : forall z x value,
  wf_state z -> wf_val value -> wf_state (with_local z x value).
Proof.
  intros [s [locals module]] x value Hstate Hvalue.
  inversion Hstate; subst. match goal with H : wf_frame _ |- _ => inversion H; subst end.
  unfold with_local. cbn. constructor; [assumption | ]. constructor; [ | assumption].
  rewrite list_update_set_at.
  eapply forall_set_at; eassumption.
Qed.

Lemma stack_matches_local_set : forall w initial final prefix value,
  LocalStackShape w true -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [value]) ->
  w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = value /\
  stackMatches final (afterBoundary w) prefix.
Proof.
  intros w initial final prefix value Hshape Hreplay [Hbase [Hsp [Hbounds Hcells]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  destruct (local_set_stack_replay w initial final Hshape Hreplay) as [Hread Hfinal].
  assert (Hlookup : nth_error (prefix ++ [value]) (length prefix) = Some value).
  { rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity. }
  pose proof (Hcells _ _ Hlookup) as Hcell.
  rewrite (local_read_address w true Hshape eq_refl) in Hread.
  replace (2 * (w COL_SP_BEFORE - 1)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix))) in Hread by lia.
  split; [congruence | ].
  unfold stackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after w true Hshape), (local_stack_pointer_after w true Hshape).
  split; [exact Hbase | ]. split; [lia | ].
  apply Forall_app in Hbounds. split; [exact (proj1 Hbounds) | ].
  intros index v HlookupPrefix. rewrite Hfinal.
  apply Hcells. rewrite nth_error_app1; [exact HlookupPrefix | ].
  apply nth_error_Some. rewrite HlookupPrefix. discriminate.
Qed.

Lemma stack_matches_local_get : forall w initial final values,
  LocalStackShape w false -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) values ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < i32Modulus ->
  stackMatches final (afterBoundary w) (values ++ [w COL_STACK_WRITE0_VALUE_LO]).
Proof.
  intros w initial final values Hshape Hreplay [Hbase [Hsp [Hbounds Hcells]]] Hresult.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  pose proof (local_get_stack_replay w initial final Hshape Hreplay) as Hfinal.
  unfold stackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after w false Hshape), (local_stack_pointer_after w false Hshape).
  split; [exact Hbase | ]. split.
  - rewrite length_app, Nat2Z.inj_add. cbn. lia.
  - split.
    + apply Forall_app. split; [exact Hbounds | ]. constructor; [exact Hresult | constructor].
    + intros index value Hlookup. rewrite Hfinal.
      rewrite update_other.
      2: { right. rewrite (local_write_hi_address w false Hshape eq_refl).
           intro Heq. injection Heq as Heq.
           change (2 * w COL_SP_BEFORE + 1 =
             2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) in Heq. lia. }
      destruct (Nat.lt_ge_cases index (length values)) as [Hinside | Houtside].
      * rewrite update_other.
        2: { right. rewrite (local_write_address w false Hshape eq_refl).
             intro Heq. injection Heq as Heq.
             change (2 * w COL_SP_BEFORE =
               2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)) in Heq.
             apply Nat2Z.inj_lt in Hinside. lia. }
        apply Hcells. rewrite nth_error_app1 in Hlookup by exact Hinside. exact Hlookup.
      * rewrite nth_error_app2 in Hlookup by exact Houtside.
        destruct (index - length values)%nat as [ |n] eqn:Hindex.
        -- cbn in Hlookup. inversion Hlookup; subst value.
           assert (HindexEq : index = length values) by lia. subst index.
           replace (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length values)))
             with (w COL_STACK_WRITE0_ADDR_LO)
             by (rewrite (local_write_address w false Hshape eq_refl); lia).
           apply update_same.
        -- destruct n; discriminate.
Qed.

Lemma local_get_frame_refinement : forall z initial final row c0 x rest,
  Range.fieldPrime -> FullRelation.satFull row ->
  Replay initial (eventsForRow MemoryCatalog.memories row) final ->
  row COL_SEL_LOCAL_GET = 1 -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (LOCAL_GET x :: rest) /\
    represents z initial (beforeBoundary row) (LOCAL_GET x :: rest) c0) ->
  exists c1, Step c0 c1 /\
    (programTyped (length (stateLocals z)) (stackHeight (afterBoundary row)) rest /\
    represents z final (afterBoundary row) rest c1).
Proof.
  intros z initial final row c0 x rest Hprime Hfull Hreplay Hchosen Hindex Hrep.
  apply represented_values in Hrep. destruct Hrep as [values [Hstack [Htyped [Hconfig [Hwf Hlocals]]]]].
  cbn [programTyped instructionStackHeight] in Htyped. destruct Htyped as [HindexBound Htyped].
  assert (Hchoice : straightLineSelector COL_SEL_LOCAL_GET) by (cbn; auto).
  destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen)
    as [Hfbp [Hwrite HindexGate]].
  pose proof (full_straight_line_local_writes row _ Hprime Hfull Hchoice Hchosen) as Hwrites.
  cbn [Nat.eqb COL_SEL_LOCAL_GET COL_SEL_LOCAL_SET] in Hwrites.
  assert (Hno : Forall (NoWriteTo MemoryCatalog.localLoId
      [row COL_LOCALS_FBP_BEFORE; row COL_LOCAL_INDEX]) (eventsForRow MemoryCatalog.memories row)).
  { pose proof (filter_no_local_writes _ Hwrites) as Hfilter.
    eapply Forall_impl; [ | exact Hfilter].
    intros [mem addr v | mem addr v old] H; cbn in *; auto. }
  pose proof (replay_read_from_initial _ _ _ _ _ _ Hreplay Hno
    (local_get_event_in_row row Hchosen)) as Hread.
  destruct (locals_match_lookup initial (beforeBoundary row) z x Hlocals HindexBound)
    as [value [Hlocal [Hbound Hcell]]].
  cbn [beforeBoundary boundaryLocalsBase] in Hcell.
  rewrite Hindex, Hcell in Hread. injection Hread as Hvalue.
  assert (HlocalValue : fun_local z x = i32Value (row COL_LOCAL_VALUE)) by congruence.
  assert (Hdepth : 0 <= row COL_SP_BEFORE) by (pose proof (value_bound row COL_SP_BEFORE); lia).
  pose proof (full_local_get_stack_shape row Hprime Hfull Hchosen Hdepth) as Hshape.
  pose proof (Range.full_write0_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hresult.
  pose proof (stack_matches_local_get row initial final values Hshape Hreplay Hstack Hresult) as HstackAfter.
  subst c0.
  assert (Hz : wf_state z) by (inversion Hwf; assumption).
  pose proof (full_relation_implies_local_get row z x Hchosen Hfull HlocalValue) as Hsmall.
  change (Step (stackConfig z [] [LOCAL_GET x])
    (stackConfig z [row COL_STACK_WRITE0_VALUE_LO] [])) in Hsmall.
  rewrite <- (app_nil_r values) in Hwf at 1.
  destruct (stack_config_step z z values [] [row COL_STACK_WRITE0_VALUE_LO] (LOCAL_GET x)
    rest Hsmall ltac:(constructor; [exact Hresult | constructor]) Hz Hwf)
    as [Hstep HwfAfter].
  rewrite app_nil_r in Hstep.
  exists (stackConfig z (values ++ [row COL_STACK_WRITE0_VALUE_LO]) rest).
  split; [exact Hstep | ].
  apply represented_values. exists (values ++ [row COL_STACK_WRITE0_VALUE_LO]). split; [exact HstackAfter | ].
  split.
  - rewrite length_app. cbn [length]. replace (length values + 1)%nat with (S (length values)) by lia.
    exact Htyped.
  - split; [reflexivity | ]. split; [exact HwfAfter | ].
    eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z); [exact Hfbp | | exact Hlocals].
    eapply no_local_writes_preserve; eassumption.
Qed.

Lemma stack_has_last : forall (values : list Z),
  (0 < length values)%nat -> exists prefix value, values = prefix ++ [value].
Proof.
  induction values as [ |a rest IH]; intros Hlength; [cbn in Hlength; lia | ].
  destruct rest as [ |b rest].
  - exists [], a. reflexivity.
  - destruct (IH ltac:(cbn; lia)) as [prefix [value Hrest]].
    exists (a :: prefix), value. cbn. now rewrite Hrest.
Qed.

Lemma local_set_frame_refinement : forall z initial final row c0 x rest,
  Range.fieldPrime -> FullRelation.satFull row ->
  Replay initial (eventsForRow MemoryCatalog.memories row) final ->
  row COL_SEL_LOCAL_SET = 1 -> row COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (LOCAL_SET x :: rest) /\
    represents z initial (beforeBoundary row) (LOCAL_SET x :: rest) c0) ->
  exists c1, Step c0 c1 /\
    (programTyped (length (stateLocals (with_local z x (i32Value (row COL_LOCAL_VALUE))))) (stackHeight (afterBoundary row)) rest /\
    represents (with_local z x (i32Value (row COL_LOCAL_VALUE))) final (afterBoundary row) rest c1).
Proof.
  intros z initial final row c0 x rest Hprime Hfull Hreplay Hchosen Hindex Hrep.
  apply represented_values in Hrep. destruct Hrep as [values [Hstack [Htyped [Hconfig [Hwf Hlocals]]]]].
  cbn [programTyped instructionStackHeight] in Htyped. destruct Htyped as [HindexBound Htyped].
  assert (Hheight : (0 < length values)%nat).
  { destruct (length values); [contradiction | lia]. }
  destruct (stack_has_last values Hheight) as [prefix [value Hvalues]]. subst values c0.
  assert (Hdepth : 1 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]]. cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_local_set_stack_shape row Hprime Hfull Hchosen Hdepth) as Hshape.
  destruct (stack_matches_local_set row initial final prefix value Hshape Hreplay Hstack)
    as [Hread HstackAfter].
  assert (Hchoice : straightLineSelector COL_SEL_LOCAL_SET) by (cbn; auto).
  destruct (full_straight_line_local_gates row _ Hprime Hfull Hchoice Hchosen)
    as [Hfbp [Hwrite HindexGate]].
  pose proof (full_straight_line_local_writes row _ Hprime Hfull Hchoice Hchosen) as Hwrites.
  cbn [Nat.eqb COL_SEL_LOCAL_SET] in Hwrites.
  pose proof (Range.full_local_value_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_LOCAL_VALUE < i32Modulus) in Hresult.
  pose proof (full_relation_implies_local_set row z x Hchosen Hfull) as Hsmall.
  rewrite Hread in Hsmall.
  change (Step (stackConfig z [value] [LOCAL_SET x])
    (stackConfig (with_local z x (i32Value (row COL_LOCAL_VALUE))) [] [])) in Hsmall.
  assert (Hz : wf_state z) by (inversion Hwf; assumption).
  pose proof (with_local_wf z x _ Hz (full_local_value_wf row Hprime Hfull)) as HzAfter.
  destruct (stack_config_step z (with_local z x (i32Value (row COL_LOCAL_VALUE)))
    prefix [value] [] (LOCAL_SET x) rest Hsmall ltac:(constructor) HzAfter Hwf)
    as [Hstep HwfAfter].
  rewrite app_nil_r in Hstep, HwfAfter.
  exists (stackConfig (with_local z x (i32Value (row COL_LOCAL_VALUE))) prefix rest).
  split; [exact Hstep | ].
  apply represented_values. exists prefix. split; [exact HstackAfter | ]. split.
  - rewrite locals_length_after_set.
    rewrite length_app in Htyped. cbn [length] in Htyped.
    replace (length prefix + 1)%nat with (S (length prefix)) in Htyped by lia. exact Htyped.
  - split; [reflexivity | ]. split; [exact HwfAfter | ].
    eapply (locals_match_set initial final (beforeBoundary row) (afterBoundary row) z x (row COL_LOCAL_VALUE)); [exact Hfbp | exact HindexBound | exact Hresult | | exact Hlocals].
    pose proof (one_local_write_effect _ _ _ _ _ _ Hreplay Hwrites) as Heffect.
    rewrite Hindex in Heffect. exact Heffect.
Qed.
