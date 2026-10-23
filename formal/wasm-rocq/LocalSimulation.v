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

(* Locals are preserved when their cells are. *)
Lemma locals_match_preserved : forall initial final before after z,
  boundaryLocalsBase after = boundaryLocalsBase before ->
  (forall index, (index < length (stateLocals z))%nat ->
    final MemoryCatalog.stackId [2 * (boundaryLocalsBase before + Z.of_nat index)] =
    initial MemoryCatalog.stackId [2 * (boundaryLocalsBase before + Z.of_nat index)]) ->
  localsMatch initial before z -> localsMatch final after z.
Proof.
  intros initial final before after z Hbase Hmemory Hlocals index v Hlookup.
  destruct (Hlocals index v Hlookup) as [value [Hv [Hbound Hcell]]].
  exists value. split; [exact Hv | ]. split; [exact Hbound | ].
  assert (Hindex : (index < length (stateLocals z))%nat).
  { apply nth_error_Some. rewrite Hlookup. discriminate. }
  rewrite Hbase, Hmemory by exact Hindex. exact Hcell.
Qed.

Lemma locals_match_set : forall initial final before after z x value,
  boundaryLocalsBase after = boundaryLocalsBase before ->
  (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  0 <= value < i32Modulus ->
  final MemoryCatalog.stackId [2 * (boundaryLocalsBase before + Z.of_N (x :> N))] = Some value ->
  (forall index, (index < length (stateLocals z))%nat -> index <> N.to_nat (x :> N) ->
    final MemoryCatalog.stackId [2 * (boundaryLocalsBase before + Z.of_nat index)] =
    initial MemoryCatalog.stackId [2 * (boundaryLocalsBase before + Z.of_nat index)]) ->
  localsMatch initial before z ->
  localsMatch final after (with_local z x (i32Value value)).
Proof.
  intros initial final before after z x value Hbase Hindex Hvalue Hset Hother Hlocals index v Hlookup.
  assert (Hbound : (index < length (stateLocals z))%nat).
  { rewrite <- (locals_length_after_set z x (i32Value value)).
    apply nth_error_Some. rewrite Hlookup. discriminate. }
  rewrite locals_after_set, nth_error_set_at in Hlookup by exact Hindex.
  destruct (Nat.eqb index (N.to_nat (x :> N))) eqn:Hsame.
  - apply Nat.eqb_eq in Hsame. subst index.
    injection Hlookup as <-. exists value. split; [reflexivity | ]. split; [exact Hvalue | ].
    rewrite Hbase, N_nat_Z. exact Hset.
  - apply Nat.eqb_neq in Hsame.
    destruct (Hlocals index v Hlookup) as [old [Hv [Hold Hcell]]].
    exists old. split; [exact Hv | ]. split; [exact Hold | ].
    rewrite Hbase, Hother by assumption. exact Hcell.
Qed.

Lemma seq_nth_list_nth : forall (A : Type) (values : list A) index default,
  seq.nth default values index = List.nth index values default.
Proof. intros A values. induction values; intros [ |index] default; cbn; auto. Qed.

Lemma locals_match_lookup : forall memory boundary z x,
  localsMatch memory boundary z -> (N.to_nat (x :> N) < length (stateLocals z))%nat ->
  exists value, fun_local z x = i32Value value /\ 0 <= value < i32Modulus /\
    memory MemoryCatalog.stackId [2 * (boundaryLocalsBase boundary + Z.of_N (x :> N))] = Some value.
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

(* A pop, possibly into a local whose cell lies below the operand base. *)
Lemma stack_matches_pop : forall w initial final prefix value local,
  UnaryStackShape w true local -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  (local = true -> w COL_LOCAL_ADDR_LO < 2 * w COL_STACK_FRAME_BASE_BEFORE /\
    w COL_LOCAL_ADDR_HI < 2 * w COL_STACK_FRAME_BASE_BEFORE) ->
  stackMatches initial (beforeBoundary w) (prefix ++ [value]) ->
  w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = value /\
  stackMatches final (afterBoundary w) prefix.
Proof.
  intros w initial final prefix value local Hshape Hreplay Hbelow [Hbase [Hsp [Hbounds Hcells]]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  assert (Hread : initial MemoryCatalog.stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
      Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
    forall index, (index < length prefix)%nat ->
      final MemoryCatalog.stackId [2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)] =
      initial MemoryCatalog.stackId [2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index)]).
  { destruct local.
    - destruct (local_set_stack_replay w initial final Hshape Hreplay) as [Hread Hfinal].
      split; [exact Hread | ]. intros index Hindex. rewrite Hfinal.
      destruct (Hbelow eq_refl). rewrite !update_other by cell_differs. reflexivity.
    - destruct (pop_stack_replay w initial final Hshape Hreplay) as [Hread Hfinal].
      split; [exact Hread | ]. intros index _. apply Hfinal. }
  destruct Hread as [Hread Hfinal].
  assert (Hlookup : nth_error (prefix ++ [value]) (length prefix) = Some value).
  { rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity. }
  pose proof (Hcells _ _ Hlookup) as Hcell.
  rewrite (local_read_address w true local Hshape eq_refl) in Hread.
  replace (2 * (w COL_SP_BEFORE - 1)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix))) in Hread by lia.
  split; [congruence | ].
  unfold stackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after w true local Hshape), (local_stack_pointer_after w true local Hshape).
  split; [exact Hbase | ]. split; [lia | ].
  apply Forall_app in Hbounds. split; [exact (proj1 Hbounds) | ].
  intros index v HlookupPrefix.
  assert (Hindex : (index < length prefix)%nat).
  { apply nth_error_Some. rewrite HlookupPrefix. discriminate. }
  rewrite Hfinal by exact Hindex.
  apply Hcells. rewrite nth_error_app1; [exact HlookupPrefix | exact Hindex].
Qed.

Lemma stack_matches_local_get : forall w initial final values,
  UnaryStackShape w false true -> Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) values ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < i32Modulus ->
  stackMatches final (afterBoundary w) (values ++ [w COL_STACK_WRITE0_VALUE_LO]).
Proof.
  intros w initial final values Hshape Hreplay [Hbase [Hsp [Hbounds Hcells]]] Hresult.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  destruct (local_get_stack_replay w initial final Hshape Hreplay) as [_ Hfinal].
  unfold stackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after w false true Hshape), (local_stack_pointer_after w false true Hshape).
  split; [exact Hbase | ]. split.
  - rewrite length_app, Nat2Z.inj_add. cbn. lia.
  - split.
    + apply Forall_app. split; [exact Hbounds | ]. constructor; [exact Hresult | constructor].
    + intros index value Hlookup. rewrite Hfinal.
      rewrite update_other.
      2: { rewrite (local_write_hi_address w false true Hshape eq_refl). cell_differs. }
      destruct (Nat.lt_ge_cases index (length values)) as [Hinside | Houtside].
      * rewrite update_other.
        2: { rewrite (local_write_address w false true Hshape eq_refl).
             apply Nat2Z.inj_lt in Hinside. cell_differs. }
        apply Hcells. rewrite nth_error_app1 in Hlookup by exact Hinside. exact Hlookup.
      * rewrite nth_error_app2 in Hlookup by exact Houtside.
        destruct (index - length values)%nat as [ |n] eqn:Hindex.
        -- cbn in Hlookup. inversion Hlookup; subst value.
           assert (HindexEq : index = length values) by lia. subst index.
           replace (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length values)))
             with (w COL_STACK_WRITE0_ADDR_LO)
             by (rewrite (local_write_address w false true Hshape eq_refl); lia).
           apply update_same.
        -- destruct n; discriminate.
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
