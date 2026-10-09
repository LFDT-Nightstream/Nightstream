(* Connects exported local memory ports, trace replay, and SpecTec i32 locals. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Wasm2 Locals
  MemoryTrace MemoryCatalog Selectors TraceRefinement.
Import ListNotations.
Open Scope Z_scope.

Definition localSpec : MemorySpec :=
  nth MemoryCatalog.localLoId MemoryCatalog.memories (mkMemory 0%nat Ram []).

Definition localGetPort : PortSpec :=
  mkPort [COL_LOCALS_FBP_BEFORE; COL_LOCAL_INDEX] COL_LOCAL_VALUE Read
    (When COL_SEL_LOCAL_GET).

Definition localSetPort : PortSpec :=
  mkPort [COL_LOCALS_FBP_BEFORE; COL_LOCAL_INDEX] COL_LOCAL_VALUE (Write None)
    (When COL_LOCAL_WRITE_ENABLED).

Lemma local_spec_in : In localSpec MemoryCatalog.memories.
Proof.
  unfold localSpec.
  apply nth_In.
  vm_compute. lia.
Qed.

Lemma local_get_port_in_catalog :
  nth_error (ports localSpec) 0%nat = Some localGetPort.
Proof. vm_compute. reflexivity. Qed.

Lemma local_set_port_in_catalog :
  nth_error (ports localSpec) 2%nat = Some localSetPort.
Proof. vm_compute. reflexivity. Qed.

Lemma local_get_event : forall (w : Witness),
  w COL_SEL_LOCAL_GET = 1 ->
  portEvent MemoryCatalog.localLoId w localGetPort =
    Some (ReadEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX] (w COL_LOCAL_VALUE)).
Proof.
  intros w Hgate.
  unfold portEvent, active, localGetPort.
  simpl. rewrite Hgate. reflexivity.
Qed.

Lemma local_set_event : forall (w : Witness),
  w COL_LOCAL_WRITE_ENABLED = 1 ->
  portEvent MemoryCatalog.localLoId w localSetPort =
    Some (WriteEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX] (w COL_LOCAL_VALUE) None).
Proof.
  intros w Hgate.
  unfold portEvent, active, localSetPort.
  simpl. rewrite Hgate. reflexivity.
Qed.

Lemma local_get_event_in_row : forall (w : Witness),
  w COL_SEL_LOCAL_GET = 1 ->
  In (ReadEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX] (w COL_LOCAL_VALUE))
    (eventsForRow MemoryCatalog.memories w).
Proof.
  intros w Hgate.
  eapply port_event_in_row with (specification := localSpec) (port := localGetPort).
  - exact local_spec_in.
  - eapply nth_error_In. exact local_get_port_in_catalog.
  - exact (local_get_event w Hgate).
Qed.

Lemma local_set_event_in_row : forall (w : Witness),
  w COL_LOCAL_WRITE_ENABLED = 1 ->
  In (WriteEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX] (w COL_LOCAL_VALUE) None)
    (eventsForRow MemoryCatalog.memories w).
Proof.
  intros w Hgate.
  eapply port_event_in_row with (specification := localSpec) (port := localSetPort).
  - exact local_spec_in.
  - eapply nth_error_In. exact local_set_port_in_catalog.
  - exact (local_set_event w Hgate).
Qed.

Lemma local_set_get_event_order : forall (wSet wGet : Witness),
  wSet COL_LOCAL_WRITE_ENABLED = 1 ->
  wGet COL_SEL_LOCAL_GET = 1 ->
  exists prefix middle suffix,
    eventsForTrace MemoryCatalog.memories [wSet; wGet] =
      prefix ++
        WriteEvent MemoryCatalog.localLoId
          [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]
          (wSet COL_LOCAL_VALUE) None ::
        middle ++
        ReadEvent MemoryCatalog.localLoId
          [wGet COL_LOCALS_FBP_BEFORE; wGet COL_LOCAL_INDEX]
          (wGet COL_LOCAL_VALUE) :: suffix.
Proof.
  intros wSet wGet Hwrite Hread.
  apply events_two_rows_order.
  - apply local_set_event_in_row. exact Hwrite.
  - apply local_get_event_in_row. exact Hread.
Qed.

Lemma fbp_link_in_catalog :
  In (COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE)
    MemoryCatalog.continuityLinks.
Proof. vm_compute. tauto. Qed.

Lemma follows_links_fbp : forall previous next,
  Follows MemoryCatalog.continuityLinks previous next ->
  previous COL_LOCALS_FBP_AFTER = next COL_LOCALS_FBP_BEFORE.
Proof.
  intros previous next Hfollows.
  apply Hfollows. exact fbp_link_in_catalog.
Qed.

Theorem local_readback_from_trace : forall initial final rows
    (wSet wGet : Witness) prefix middle suffix,
  validCircuitRun initial rows final ->
  wSet COL_LOCALS_FBP_BEFORE = wGet COL_LOCALS_FBP_BEFORE ->
  wSet COL_LOCAL_INDEX = wGet COL_LOCAL_INDEX ->
  eventsForTrace MemoryCatalog.memories rows =
    prefix ++
      WriteEvent MemoryCatalog.localLoId
        [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]
        (wSet COL_LOCAL_VALUE) None ::
      middle ++
      ReadEvent MemoryCatalog.localLoId
        [wGet COL_LOCALS_FBP_BEFORE; wGet COL_LOCAL_INDEX]
        (wGet COL_LOCAL_VALUE) :: suffix ->
  Forall (NoWriteTo MemoryCatalog.localLoId
    [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]) middle ->
  wGet COL_LOCAL_VALUE = wSet COL_LOCAL_VALUE.
Proof.
  intros initial final rows wSet wGet prefix middle suffix
    [_ [_ Hreplay]] Hfbp Hindex Hevents Hno.
  rewrite Hevents in Hreplay.
  rewrite <- Hfbp, <- Hindex in Hreplay.
  eapply replay_last_write_read; eassumption.
Qed.

Definition localAddress (base : Z) (x : localidx) : Address :=
  [base; Z.of_N (x :> N)].

Definition localCellAgrees (memory : MemoryState) (z : state)
    (base : Z) (x : localidx) : Prop :=
  exists value,
    fun_local z x = i32Value value /\
    memory MemoryCatalog.localLoId (localAddress base x) = Some value.

Lemma lookup_after_local_update : forall (locals : list val) (n : N) (v : val),
  (N.to_nat n < length locals)%nat ->
  lookup_total (list_update_func locals n (fun _ => v)) n = v.
Proof.
  induction locals; intros n v Hbound.
  - cbn in Hbound. lia.
  - rename a into h.
    rename locals into t.
    rename IHlocals into IH.
    destruct n.
    + reflexivity.
    + unfold lookup_total in *.
      cbn [list_update_func].
      change (seq.nth default_val
        (h :: list_update_func t (N.pred (Npos p)) (fun _ => v))
        (N.to_nat (Npos p)) = v).
      replace (N.to_nat (Npos p)) with
        (S (N.to_nat (N.pred (Npos p))))
        by (rewrite <- N2Nat.inj_succ; rewrite N.succ_pred_pos by lia; reflexivity).
      cbn [seq.nth].
      apply IH.
      rewrite N2Nat.inj_pred.
      pose proof (Pos2Nat.is_pos p) as Hpos.
      change (0 < N.to_nat (Npos p))%nat in Hpos.
      cbn [length] in Hbound.
      lia.
Qed.

Lemma local_after_set : forall (s : store) (f : frame) (x : localidx) (v : val),
  (N.to_nat (x :> N) < length (LOCALS f))%nat ->
  fun_local (with_local (mk_state s f) x v) x = v.
Proof.
  intros s f x v Hbound.
  unfold fun_local, with_local.
  cbn.
  apply lookup_after_local_update.
  exact Hbound.
Qed.

Theorem local_set_establishes_agreement :
  forall (w : Witness) (memory after : MemoryState)
    (s : store) (f : frame) (x : localidx) (base : Z),
  Range.fieldPrime ->
  w COL_SEL_LOCAL_SET = 1 -> FullRelation.satFull w ->
  w COL_LOCALS_FBP_BEFORE = base ->
  w COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  (N.to_nat (x :> N) < length (LOCALS f))%nat ->
  EventStep memory
    (WriteEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX]
      (w COL_LOCAL_VALUE) None) after ->
  let z := mk_state s f in
  let z' := with_local z x (i32Value (w COL_LOCAL_VALUE)) in
  Step (mk_config z
      ((admininstr_val (i32Value (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))) ::
        admininstr_LOCAL_SET x :: nil)%list))
    (mk_config z' nil) /\
  wf_val (i32Value (w COL_LOCAL_VALUE)) /\
  localCellAgrees after z' base x.
Proof.
  intros w memory after s f x base Hprime Hsel Hfull Hbase Hindex Hbound Hevent.
  cbn beta.
  split.
  - exact (full_relation_implies_local_set w (mk_state s f) x Hsel Hfull).
  - split.
    + exact (full_local_value_wf w Hprime Hfull).
    + exists (w COL_LOCAL_VALUE). split.
      * apply local_after_set. exact Hbound.
      * inversion Hevent; subst.
        unfold localAddress.
        rewrite <- Hindex.
        apply update_same.
Qed.

Theorem local_get_reads_agreed_cell :
  forall (w : Witness) (memory after : MemoryState) (z : state)
    (x : localidx) (base : Z),
  w COL_SEL_LOCAL_GET = 1 -> FullRelation.satFull w ->
  w COL_LOCALS_FBP_BEFORE = base ->
  w COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  localCellAgrees memory z base x ->
  EventStep memory
    (ReadEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX]
      (w COL_LOCAL_VALUE)) after ->
  Step (mk_config z ((admininstr_LOCAL_GET x :: nil)%list))
    (mk_config z ((admininstr_val (i32Value (w COL_STACK_WRITE0_VALUE_LO)) :: nil)%list)).
Proof.
  intros w memory after z x base Hsel Hfull Hbase Hindex
    [value [Hlocal Hcell]] Hevent.
  inversion Hevent as [st mem addr val Hread | st mem addr val prior Hprior]; subst.
  apply full_relation_implies_local_get; try assumption.
  unfold localAddress in Hcell.
  rewrite <- Hindex in Hcell.
  rewrite Hcell in Hread.
  injection Hread as Hvalue.
  rewrite Hvalue in Hlocal.
  exact Hlocal.
Qed.

Lemma local_cell_agrees_preserved : forall before after z base x events,
  Replay before events after ->
  Forall (NoWriteTo MemoryCatalog.localLoId (localAddress base x)) events ->
  localCellAgrees before z base x ->
  localCellAgrees after z base x.
Proof.
  intros before after z base x events Hreplay Hno [value [Hlocal Hcell]].
  exists value. split.
  - exact Hlocal.
  - rewrite (replay_preserves _ _ _ _ _ Hreplay Hno).
    exact Hcell.
Qed.

Definition guestEntryRow : Row :=
  ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1);
    (COL_SEL_RETURN_CALL, 1); (COL_SEL_RETURN_CALL_INDIRECT, 1)],
   [(COL_TARGET_FUNCTION_IS_GUEST, 1)],
   [(COL_GUEST_ENTRY_ACTIVE, 1)]).

Definition popAllowedRow : Row :=
  ([(COL_CALL_STACK_POP_PRESENT, 1)],
   [(COL_ONE, 1); (COL_SEL_RETURN, -1); (COL_SEL_END, -1)], []).

Definition localsFbpPreserveRow : Row :=
  ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1);
    (COL_CALL_STACK_POP_PRESENT, -1)],
   [(COL_LOCALS_FBP_AFTER, 1); (COL_LOCALS_FBP_BEFORE, -1)], []).

Lemma guest_entry_row_in_full :
  nth_error FullRelation.rows 63%nat = Some guestEntryRow.
Proof. vm_compute. reflexivity. Qed.

Lemma pop_allowed_row_in_full :
  nth_error FullRelation.rows 99%nat = Some popAllowedRow.
Proof. vm_compute. reflexivity. Qed.

Lemma locals_fbp_preserve_row_in_full :
  nth_error FullRelation.rows 117%nat = Some localsFbpPreserveRow.
Proof. vm_compute. reflexivity. Qed.

Lemma full_local_set_fbp_stable : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_LOCAL_SET = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE.
Proof.
  intros w Hprime Hfull Hset.
  assert (Hother : forall column,
    In column [COL_SEL_CALL; COL_SEL_CALL_INDIRECT;
      COL_SEL_RETURN_CALL; COL_SEL_RETURN_CALL_INDIRECT;
      COL_SEL_RETURN; COL_SEL_END] -> w column = 0).
  { intros column Hin.
    assert (Hcatalog : In column Selectors.selectorColumns).
    { vm_compute in Hin |- *. tauto. }
    assert (Hneq : COL_SEL_LOCAL_SET <> column).
    { vm_compute in Hin |- *. lia. }
    eapply Selectors.full_opcode_selector_exclusive;
      [exact Hprime | exact Hfull | exact Hset | vm_compute; tauto |
       exact Hcatalog | exact Hneq]. }
  destruct Hfull as [Hone Hall].
  pose proof (Range.sat_row w 63%nat _ Hall guest_entry_row_in_full) as HguestRow.
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { unfold sat, guestEntryRow in HguestRow.
    cbn [eval fold_right] in HguestRow.
    repeat rewrite Hother in HguestRow by (cbn; tauto).
    apply Z.mod_divide in HguestRow; try (unfold modulus; lia).
    destruct HguestRow as [k Hk].
    pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE) as [Hlo Hhi].
    unfold modulus in *; lia. }
  pose proof (Range.sat_row w 99%nat _ Hall pop_allowed_row_in_full) as HpopRow.
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { unfold sat, popAllowedRow in HpopRow.
    cbn [eval fold_right] in HpopRow.
    assert (Hreturn : w COL_SEL_RETURN = 0).
    { apply Hother. cbn. tauto. }
    assert (Hend : w COL_SEL_END = 0).
    { apply Hother. cbn. tauto. }
    rewrite Hreturn, Hend in HpopRow.
    change (w COL_ONE = 1) in Hone.
    rewrite Hone in HpopRow.
    apply Z.mod_divide in HpopRow; try (unfold modulus; lia).
    destruct HpopRow as [k Hk].
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT) as [Hlo Hhi].
    unfold modulus in *; nia. }
  pose proof (Range.sat_row w 117%nat _ Hall locals_fbp_preserve_row_in_full)
    as HfbpRow.
  unfold sat, localsFbpPreserveRow in HfbpRow.
  cbn [eval fold_right] in HfbpRow.
  change (w COL_ONE = 1) in Hone.
  rewrite Hone, Hguest, Hpop in HfbpRow.
  apply Z.mod_divide in HfbpRow; try (unfold modulus; lia).
  destruct HfbpRow as [k Hk].
  pose proof (value_bound w COL_LOCALS_FBP_AFTER) as [Ha0 Ha1].
  pose proof (value_bound w COL_LOCALS_FBP_BEFORE) as [Hb0 Hb1].
  unfold modulus in *; nia.
Qed.

Definition gatherLocalLoRow : Row :=
  ([(537%nat, 1)], [(COL_ONE, 1); (440%nat, -1)],
   [(COL_GATHER_LOCAL_WRITE_LO, 1)]).

Lemma gather_local_lo_row_in_full :
  nth_error FullRelation.rows 176%nat = Some gatherLocalLoRow.
Proof. vm_compute. reflexivity. Qed.

Lemma full_no_gather_local_write : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_GATHER_ACTIVE = 0 ->
  w COL_GATHER_LOCAL_WRITE_LO = 0.
Proof.
  intros w Hprime Hfull Hgather.
  pose proof (Selectors.selector_boolean_from_block w 3687%nat 10%nat
    gatherKindColumns Hprime Hfull gather_kind_boolean_block) as Hbits.
  pose proof (Selectors.selector_sum_boolean_bound w gatherKindColumns Hbits)
    as HsumBound.
  change (0 <= Selectors.selectorSum w gatherKindColumns <= 10) in HsumBound.
  destruct Hfull as [Hone Hall].
  pose proof (Range.sat_row w 154%nat _ Hall gather_kind_row_in_full) as HkindRow.
  unfold sat, gatherKindRow in HkindRow.
  rewrite Selectors.eval_app in HkindRow.
  rewrite Selectors.selector_sum_eval in HkindRow.
  cbn [eval fold_right] in HkindRow.
  change (w COL_ONE = 1) in Hone.
  rewrite Hone, Hgather in HkindRow.
  assert (Hsum : Selectors.selectorSum w gatherKindColumns = 0).
  { apply Z.mod_divide in HkindRow; try (unfold modulus; lia).
    destruct HkindRow as [k Hk]. unfold modulus in *; lia. }
  assert (Hkind : w 537%nat = 0).
  { pose proof (Selectors.selector_sum_one w gatherKindColumns 537%nat)
      as Hselected.
    assert (Hin : In 537%nat gatherKindColumns).
    { vm_compute. tauto. }
    specialize (Hselected Hin).
    rewrite Hsum in Hselected.
    pose proof (value_bound w 537%nat) as [Hlo _]. lia. }
  pose proof (Range.sat_row w 176%nat _ Hall gather_local_lo_row_in_full)
    as HloRow.
  unfold sat, gatherLocalLoRow in HloRow.
  cbn [eval fold_right] in HloRow.
  rewrite Hkind in HloRow.
  apply Z.mod_divide in HloRow; try (unfold modulus; lia).
  destruct HloRow as [k Hk].
  pose proof (value_bound w COL_GATHER_LOCAL_WRITE_LO) as [Hlo Hhi].
  unfold modulus in *; lia.
Qed.

Definition localWriteGateRow : Row :=
  ([(COL_LOCAL_WRITE_ENABLED, 1); (COL_SEL_LOCAL_SET, -1);
    (COL_SEL_LOCAL_TEE, -1)], [(COL_ONE, 1)], []).

Lemma local_write_gate_row_in_full :
  nth_error FullRelation.rows 2%nat = Some localWriteGateRow.
Proof. vm_compute. reflexivity. Qed.

Lemma full_local_write_gate : forall (w : Witness),
  FullRelation.satFull w ->
  w COL_SEL_LOCAL_SET + w COL_SEL_LOCAL_TEE < modulus ->
  w COL_LOCAL_WRITE_ENABLED = w COL_SEL_LOCAL_SET + w COL_SEL_LOCAL_TEE.
Proof.
  intros w [Hone Hall] HsumBound.
  pose proof (Range.sat_row w 2%nat _ Hall local_write_gate_row_in_full) as Hrow.
  unfold sat, localWriteGateRow in Hrow.
  cbn [eval fold_right] in Hrow.
  change (w COL_ONE = 1) in Hone.
  rewrite Hone in Hrow.
  apply Z.mod_divide in Hrow; try (unfold modulus; lia).
  destruct Hrow as [k Hk].
  pose proof (value_bound w COL_LOCAL_WRITE_ENABLED) as [Hlo Hhi].
  pose proof (value_bound w COL_SEL_LOCAL_SET) as [Hs0 _].
  pose proof (value_bound w COL_SEL_LOCAL_TEE) as [Ht0 _].
  unfold modulus in *; lia.
Qed.

Lemma full_local_set_write_enabled : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_LOCAL_SET = 1 -> w COL_LOCAL_WRITE_ENABLED = 1.
Proof.
  intros w Hprime Hfull Hset.
  assert (Htee : w COL_SEL_LOCAL_TEE = 0).
  { eapply Selectors.full_opcode_selector_exclusive;
      [exact Hprime | exact Hfull | exact Hset |
       vm_compute; tauto | vm_compute; tauto | vm_compute; discriminate]. }
  rewrite (full_local_write_gate w Hfull).
  - rewrite Hset, Htee. lia.
  - rewrite Hset, Htee. unfold modulus. lia.
Qed.

Definition isLocalWrite (event : Event) : bool :=
  match event with
  | WriteEvent memory _ _ _ => Nat.eqb memory MemoryCatalog.localLoId
  | ReadEvent _ _ _ => false
  end.

Lemma filter_other_memory_ports : forall w memory ports,
  memory <> MemoryCatalog.localLoId ->
  filter isLocalWrite
    (flat_map (fun p => match portEvent memory w p with
       | Some event => [event] | None => [] end) ports) = [].
Proof.
  intros w memory ports Hother.
  induction ports.
  - reflexivity.
  - cbn [flat_map]. rewrite filter_app, IHports.
    unfold portEvent.
    destruct (active w (activation a)); destruct (portKind a);
      cbn [filter isLocalWrite].
    all: try reflexivity.
    apply Nat.eqb_neq in Hother. rewrite Hother. reflexivity.
Qed.

Lemma filter_other_memories : forall w catalog,
  Forall (fun specification =>
    memoryId specification <> MemoryCatalog.localLoId) catalog ->
  filter isLocalWrite (eventsForRow catalog w) = [].
Proof.
  intros w catalog Hother.
  induction Hother.
  - reflexivity.
  - unfold eventsForRow in *.
    cbn [flat_map]. rewrite filter_app.
    rewrite (filter_other_memory_ports w (memoryId x) (ports x) H).
    exact IHHother.
Qed.

Lemma local_catalog_split :
  MemoryCatalog.memories =
    firstn 5%nat MemoryCatalog.memories ++
      localSpec :: skipn 6%nat MemoryCatalog.memories /\
  Forall (fun specification =>
    memoryId specification <> MemoryCatalog.localLoId)
    (firstn 5%nat MemoryCatalog.memories) /\
  Forall (fun specification =>
    memoryId specification <> MemoryCatalog.localLoId)
    (skipn 6%nat MemoryCatalog.memories).
Proof. vm_compute. repeat split; repeat constructor; discriminate. Qed.

Lemma events_for_row_app : forall first second w,
  eventsForRow (first ++ second) w =
    eventsForRow first w ++ eventsForRow second w.
Proof. intros. unfold eventsForRow. apply flat_map_app. Qed.

Lemma events_for_row_cons : forall specification rest w,
  eventsForRow (specification :: rest) w =
    flat_map (fun p => match portEvent (memoryId specification) w p with
      | Some event => [event] | None => [] end) (ports specification) ++
    eventsForRow rest w.
Proof. reflexivity. Qed.

Lemma local_writes_in_row : forall (w : Witness),
  w COL_PARAM_INIT_ACTIVE_BEFORE = 0 ->
  w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0 ->
  w COL_GATHER_LOCAL_WRITE_LO = 0 ->
  filter isLocalWrite (eventsForRow MemoryCatalog.memories w) =
    if Z.eqb (w COL_LOCAL_WRITE_ENABLED) 1 then
      [WriteEvent MemoryCatalog.localLoId
      [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX]
      (w COL_LOCAL_VALUE) None] else [].
Proof.
  intros w Hparam Hzero Hgather.
  unfold COL_LOCAL_WRITE_ENABLED.
  destruct local_catalog_split as [Hsplit [Hbefore Hafter]].
  rewrite Hsplit.
  rewrite events_for_row_app, events_for_row_cons.
  repeat rewrite filter_app.
  rewrite (filter_other_memories w _ Hbefore).
  rewrite (filter_other_memories w _ Hafter).
  cbn [app].
  unfold localSpec.
  unfold MemoryCatalog.localLoId.
  cbn [nth MemoryCatalog.memories ports memoryId flat_map filter].
  cbv [portEvent active].
  cbn [flat_map filter map Nat.eqb memoryId ports
    activation portKind addressColumns valueColumn].
  change (w 29%nat = 0) in Hparam.
  change (w 23%nat = 0) in Hzero.
  change (w 446%nat = 0) in Hgather.
  rewrite Hparam, Hzero, Hgather.
  unfold isLocalWrite.
  repeat rewrite filter_app.
  destruct (w 130%nat =? 1); destruct (w 550%nat =? 1);
    destruct (w 146%nat =? 1);
    cbn [Z.eqb filter flat_map app Nat.eqb]; reflexivity.
Qed.

Lemma local_writes_in_set_get_rows : forall (wSet wGet : Witness),
  Range.fieldPrime ->
  FullRelation.satFull wSet -> FullRelation.satFull wGet ->
  wSet COL_SEL_LOCAL_SET = 1 -> wGet COL_SEL_LOCAL_GET = 1 ->
  filter isLocalWrite
    (eventsForTrace MemoryCatalog.memories [wSet; wGet]) =
    [WriteEvent MemoryCatalog.localLoId
      [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]
      (wSet COL_LOCAL_VALUE) None].
Proof.
  intros wSet wGet Hprime HfullSet HfullGet Hset Hget.
  assert (HgetSet : wGet COL_SEL_LOCAL_SET = 0).
  { eapply Selectors.full_opcode_selector_exclusive;
      [exact Hprime | exact HfullGet | exact Hget |
       vm_compute; tauto | vm_compute; tauto | vm_compute; discriminate]. }
  assert (HgetTee : wGet COL_SEL_LOCAL_TEE = 0).
  { eapply Selectors.full_opcode_selector_exclusive;
      [exact Hprime | exact HfullGet | exact Hget |
       vm_compute; tauto | vm_compute; tauto | vm_compute; discriminate]. }
  pose proof (full_local_set_write_enabled wSet Hprime HfullSet Hset)
    as HwriteSet.
  assert (HwriteGet : wGet COL_LOCAL_WRITE_ENABLED = 0).
  { rewrite (full_local_write_gate wGet HfullGet).
    - rewrite HgetSet, HgetTee. lia.
    - rewrite HgetSet, HgetTee. unfold modulus. lia. }
  pose proof (full_opcode_aux_gates_zero wSet COL_SEL_LOCAL_SET Hprime
    HfullSet ltac:(vm_compute; tauto) Hset) as [HsParam [HsZero [HsGather _]]].
  pose proof (full_opcode_aux_gates_zero wGet COL_SEL_LOCAL_GET Hprime
    HfullGet ltac:(vm_compute; tauto) Hget) as [HgParam [HgZero [HgGather _]]].
  pose proof (full_no_gather_local_write wSet Hprime HfullSet HsGather)
    as HsGatherLo.
  pose proof (full_no_gather_local_write wGet Hprime HfullGet HgGather)
    as HgGatherLo.
  unfold eventsForTrace. cbn [flat_map]. rewrite app_nil_r.
  rewrite filter_app.
  rewrite (local_writes_in_row wSet HsParam HsZero HsGatherLo).
  rewrite (local_writes_in_row wGet HgParam HgZero HgGatherLo).
  rewrite HwriteSet, HwriteGet.
  reflexivity.
Qed.

Lemma filter_no_local_writes : forall events,
  filter isLocalWrite events = [] ->
  Forall (fun event => match event with
    | ReadEvent _ _ _ => True
    | WriteEvent memory _ _ _ => memory <> MemoryCatalog.localLoId
    end) events.
Proof.
  intros events Hfilter.
  induction events.
  - constructor.
  - cbn [filter] in Hfilter.
    destruct (isLocalWrite a) eqn:Hwrite; try discriminate.
    constructor.
    + destruct a as [memory address value | memory address value prior].
      * exact I.
      * unfold isLocalWrite in Hwrite.
        apply Nat.eqb_neq in Hwrite.
        exact Hwrite.
    + apply IHevents. exact Hfilter.
Qed.

Lemma no_local_write_between_set_get : forall (wSet wGet : Witness)
    prefix middle suffix,
  Range.fieldPrime ->
  FullRelation.satFull wSet -> FullRelation.satFull wGet ->
  wSet COL_SEL_LOCAL_SET = 1 -> wGet COL_SEL_LOCAL_GET = 1 ->
  eventsForTrace MemoryCatalog.memories [wSet; wGet] =
    prefix ++
      WriteEvent MemoryCatalog.localLoId
        [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]
        (wSet COL_LOCAL_VALUE) None ::
      middle ++
      ReadEvent MemoryCatalog.localLoId
        [wGet COL_LOCALS_FBP_BEFORE; wGet COL_LOCAL_INDEX]
        (wGet COL_LOCAL_VALUE) :: suffix ->
  Forall (NoWriteTo MemoryCatalog.localLoId
    [wSet COL_LOCALS_FBP_BEFORE; wSet COL_LOCAL_INDEX]) middle.
Proof.
  intros wSet wGet prefix middle suffix Hprime HfullSet HfullGet
    Hset Hget Hevents.
  pose proof (local_writes_in_set_get_rows wSet wGet Hprime
    HfullSet HfullGet Hset Hget) as Hfilter.
  rewrite Hevents in Hfilter.
  repeat rewrite filter_app in Hfilter.
  unfold isLocalWrite, MemoryCatalog.localLoId in Hfilter.
  cbn [filter Nat.eqb] in Hfilter.
  repeat rewrite filter_app in Hfilter.
  cbn [filter] in Hfilter.
  cbv [MemoryCatalog.localLoId isLocalWrite] in Hfilter.
  cbn in Hfilter.
  pose proof (f_equal (@length Event) Hfilter) as Hlength.
  repeat rewrite length_app in Hlength.
  cbn [length] in Hlength.
  change ((length (filter isLocalWrite prefix) +
    S (length (filter isLocalWrite middle ++
      filter isLocalWrite suffix)))%nat = 1%nat) in Hlength.
  rewrite length_app in Hlength.
  assert (HmiddleLength : length (filter isLocalWrite middle) = 0%nat)
    by lia.
  assert (Hempty : filter isLocalWrite middle = []).
  { destruct (filter isLocalWrite middle).
    - reflexivity.
    - cbn [length] in HmiddleLength. lia. }
  pose proof (filter_no_local_writes middle Hempty) as Hno.
  eapply Forall_impl.
  2: exact Hno.
  intros event HnoEvent.
  destruct event as [memory address value | memory address value prior].
  - exact I.
  - left. exact HnoEvent.
Qed.

Theorem local_set_get_from_trace :
  forall initial final (wSet wGet : Witness) (s : store) (f : frame)
    (x : localidx),
  validCircuitRun initial [wSet; wGet] final ->
  Range.fieldPrime ->
  wSet COL_SEL_LOCAL_SET = 1 ->
  wGet COL_SEL_LOCAL_GET = 1 ->
  wSet COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  wGet COL_LOCAL_INDEX = Z.of_N (x :> N) ->
  (N.to_nat (x :> N) < length (LOCALS f))%nat ->
  let z := mk_state s f in
  let z' := with_local z x (i32Value (wSet COL_LOCAL_VALUE)) in
  Step (mk_config z
      ((admininstr_val (i32Value (wSet (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))) ::
        admininstr_LOCAL_SET x :: nil)%list))
    (mk_config z' nil) /\
  Step (mk_config z' ((admininstr_LOCAL_GET x :: nil)%list))
    (mk_config z'
      ((admininstr_val (i32Value (wGet COL_STACK_WRITE0_VALUE_LO)) :: nil)%list)) /\
  wGet COL_STACK_WRITE0_VALUE_LO =
    wSet (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) /\
  wf_val (i32Value (wSet COL_LOCAL_VALUE)).
Proof.
  intros initial final wSet wGet s f x
    Htrace Hprime Hset Hget
    HsetIndex HgetIndex Hbound.
  destruct Htrace as [Hrows [Hchain Hmemory]].
  pose proof (Forall_inv Hrows) as HfullSet.
  pose proof (Forall_inv (Forall_inv_tail Hrows)) as HfullGet.
  pose proof (full_local_set_write_enabled wSet Hprime HfullSet Hset)
    as HwriteGate.
  destruct (local_set_get_event_order wSet wGet HwriteGate Hget)
    as [prefix [middle [suffix Hevents]]].
  pose proof (no_local_write_between_set_get wSet wGet prefix middle suffix
    Hprime HfullSet HfullGet Hset Hget Hevents) as Hno.
  pose proof (full_local_set_fbp_stable wSet Hprime HfullSet Hset) as Hstable.
  assert (Hfollows : Follows MemoryCatalog.continuityLinks wSet wGet).
  { inversion Hchain; subst; assumption. }
  pose proof (follows_links_fbp wSet wGet Hfollows) as HfbpLink.
  assert (Hfbp : wSet COL_LOCALS_FBP_BEFORE = wGet COL_LOCALS_FBP_BEFORE)
    by lia.
  assert (Hindex : wSet COL_LOCAL_INDEX = wGet COL_LOCAL_INDEX) by lia.
  pose proof (local_readback_from_trace initial final [wSet; wGet] wSet wGet
    prefix middle suffix (conj Hrows (conj Hchain Hmemory))
    Hfbp Hindex Hevents Hno) as Hreadback.
  rewrite Hevents in Hmemory.
  destruct (replay_app_inv prefix _ _ _ Hmemory)
    as [beforeWrite [_ Htail]].
  inversion Htail as [s0 | before0 afterWrite after0 event0 events0 Hwrite HafterWrite]; subst.
  destruct (replay_app_inv middle _ _ _ HafterWrite)
    as [beforeRead [Hmiddle HreadTail]].
  inversion HreadTail as [s1 | before1 after1 after2 event1 events1 Hread Hsuffix]; subst.
  pose proof (local_set_establishes_agreement wSet beforeWrite afterWrite
    s f x (wSet COL_LOCALS_FBP_BEFORE) Hprime Hset HfullSet
    eq_refl HsetIndex Hbound Hwrite) as [HsetStep [Hwf HagreeWrite]].
  assert (HnoCell : Forall (NoWriteTo MemoryCatalog.localLoId
      (localAddress (wSet COL_LOCALS_FBP_BEFORE) x)) middle).
  { unfold localAddress. rewrite <- HsetIndex. exact Hno. }
  pose proof (local_cell_agrees_preserved afterWrite beforeRead
    (with_local (mk_state s f) x (i32Value (wSet COL_LOCAL_VALUE)))
    (wSet COL_LOCALS_FBP_BEFORE) x middle Hmiddle HnoCell HagreeWrite)
    as HagreeRead.
  pose proof (full_local_set_value wSet Hset HfullSet) as HsetValue.
  pose proof (full_local_get_value wGet Hget HfullGet) as HgetValue.
  cbn beta.
  split.
  - exact HsetStep.
  - split.
    + eapply local_get_reads_agreed_cell with
        (memory := beforeRead) (base := wSet COL_LOCALS_FBP_BEFORE).
      * exact Hget.
      * exact HfullGet.
      * symmetry. exact Hfbp.
      * exact HgetIndex.
      * exact HagreeRead.
      * exact Hread.
    + split.
      * lia.
      * exact Hwf.
Qed.
