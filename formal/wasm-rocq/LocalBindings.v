(* Circuit-derived local memory gates and frame-base preservation. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

Definition straightLineSelector (chosen : Column) : Prop :=
  In chosen [COL_SEL_I32_ADD; COL_SEL_I32_SUB; COL_SEL_LOCAL_GET; COL_SEL_LOCAL_SET].

Lemma straight_line_selector_in : forall chosen,
  straightLineSelector chosen -> In chosen selectorColumns.
Proof. intros chosen H; destruct H as [<- | [<- | [<- | [<- | []]]]]; vm_compute; tauto. Qed.

Lemma full_straight_line_local_gates : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  straightLineSelector chosen -> w chosen = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_LOCAL_WRITE_ENABLED = (if Nat.eqb chosen COL_SEL_LOCAL_SET then 1 else 0) /\
  w COL_PROGRAM_LOCAL_INDEX_ACTIVE =
    (if Nat.eqb chosen COL_SEL_LOCAL_GET || Nat.eqb chosen COL_SEL_LOCAL_SET then 1 else 0).
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen.
  pose proof (straight_line_selector_in chosen Hchoice) as Hin.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct Hchoice as [<- | [<- | [<- | [<- | []]]]];
    cbn [COL_SEL_I32_ADD COL_SEL_I32_SUB COL_SEL_LOCAL_GET COL_SEL_LOCAL_SET Nat.eqb orb];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0) by
        (selected_row chosen guestEntryFlagRow Hrow;
         pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE); field_linear Hrow);
      assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0) by
        (selected_row chosen callStackPopSupportRow Hrow; rewrite Hone in Hrow;
         pose proof (value_bound w COL_CALL_STACK_POP_PRESENT); field_linear Hrow);
      split;
      [ selected_row chosen localsBasePreservedRow Hrow; rewrite Hone, Hguest, Hpop in Hrow;
        pose proof (value_bound w COL_LOCALS_FBP_BEFORE);
        pose proof (value_bound w COL_LOCALS_FBP_AFTER); field_linear Hrow
      | split;
        [ selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
          pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow
        | selected_row chosen localIndexGateRow Hrow; rewrite Hone in Hrow;
          pose proof (value_bound w COL_PROGRAM_LOCAL_INDEX_ACTIVE); field_linear Hrow ] ]
    end.
Qed.

Definition localSpec : MemorySpec :=
  nth MemoryCatalog.localLoId MemoryCatalog.memories (mkMemory 0%nat Ram []).

Definition localGetPort : PortSpec :=
  mkPort [COL_LOCALS_FBP_BEFORE; COL_LOCAL_INDEX] COL_LOCAL_VALUE Read
    (When COL_SEL_LOCAL_GET).

Lemma local_spec_in : In localSpec MemoryCatalog.memories.
Proof.
  unfold localSpec.
  apply nth_In.
  vm_compute. lia.
Qed.

Lemma local_get_port_in_catalog :
  nth_error (ports localSpec) 0%nat = Some localGetPort.
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

Lemma full_no_gather_local_write : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  In chosen selectorColumns -> w chosen = 1 ->
  w COL_GATHER_LOCAL_WRITE_LO = 0.
Proof.
  intros w chosen Hprime Hfull Hin Hchosen.
  pose proof (quiet_gather_kinds w
    (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen)) as Hkinds.
  selected_row chosen gatherLocalLowWriteRow Hrow.
  zero_gather_kinds Hkinds Hrow.
  pose proof (value_bound w COL_GATHER_LOCAL_WRITE_LO). field_linear Hrow.
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
    firstn MemoryCatalog.localLoId MemoryCatalog.memories ++
      localSpec :: skipn (S MemoryCatalog.localLoId) MemoryCatalog.memories /\
  Forall (fun specification =>
    memoryId specification <> MemoryCatalog.localLoId)
    (firstn MemoryCatalog.localLoId MemoryCatalog.memories) /\
  Forall (fun specification =>
    memoryId specification <> MemoryCatalog.localLoId)
    (skipn (S MemoryCatalog.localLoId) MemoryCatalog.memories).
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
  normalize_columns.
  rewrite Hparam, Hzero, Hgather.
  unfold isLocalWrite.
  repeat rewrite filter_app.
  let get := eval compute in COL_SEL_LOCAL_GET in
  let gather := eval compute in GMEM_LOCAL in
  let write := eval compute in COL_LOCAL_WRITE_ENABLED in
  destruct (w get =? 1); destruct (w gather =? 1); destruct (w write =? 1);
    cbn [Z.eqb filter flat_map app Nat.eqb]; reflexivity.
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

Lemma full_straight_line_local_writes : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  straightLineSelector chosen -> w chosen = 1 ->
  filter isLocalWrite (eventsForRow MemoryCatalog.memories w) =
    if Nat.eqb chosen COL_SEL_LOCAL_SET then
      [WriteEvent MemoryCatalog.localLoId
        [w COL_LOCALS_FBP_BEFORE; w COL_LOCAL_INDEX] (w COL_LOCAL_VALUE) None]
    else [].
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen.
  pose proof (straight_line_selector_in chosen Hchoice) as Hin.
  pose proof (full_opcode_aux_gates_zero w chosen Hprime Hfull Hin Hchosen)
    as [Hparam [Hzero _]].
  pose proof (full_no_gather_local_write w chosen Hprime Hfull Hin Hchosen) as Hlo.
  rewrite (local_writes_in_row w Hparam Hzero Hlo).
  destruct (full_straight_line_local_gates w chosen Hprime Hfull Hchoice Hchosen)
    as [_ [Hwrite _]].
  rewrite Hwrite. destruct (Nat.eqb chosen COL_SEL_LOCAL_SET); reflexivity.
Qed.

Lemma no_local_writes_preserve : forall initial events final,
  Replay initial events final -> filter isLocalWrite events = [] ->
  forall address, final MemoryCatalog.localLoId address = initial MemoryCatalog.localLoId address.
Proof.
  intros initial events final Hreplay Hfilter address.
  eapply replay_preserves; [exact Hreplay | ].
  pose proof (filter_no_local_writes events Hfilter) as Hno.
  eapply Forall_impl; [ | exact Hno].
  intros [memory addr value | memory addr value prior] H; cbn in *; auto.
Qed.

Lemma one_local_write_effect : forall initial events final address value prior,
  Replay initial events final ->
  filter isLocalWrite events = [WriteEvent MemoryCatalog.localLoId address value prior] ->
  forall target, final MemoryCatalog.localLoId target =
    update initial MemoryCatalog.localLoId address value MemoryCatalog.localLoId target.
Proof.
  intros initial events final address value prior Hreplay.
  induction Hreplay; intros Hfilter target; [discriminate | ].
  cbn [filter] in Hfilter.
  destruct (isLocalWrite event) eqn:His.
  - destruct event as [mem addr val | mem addr val old]; [discriminate | ].
    unfold isLocalWrite in His. apply Nat.eqb_eq in His. subst mem.
    injection Hfilter as Haddr Hval Hold Htail. subst addr val old.
    rewrite (no_local_writes_preserve _ _ _ Hreplay Htail target).
    inversion H; subst. reflexivity.
  - rewrite (IHHreplay Hfilter target).
    destruct (list_eq_dec Z.eq_dec target address) as [<- | Hneq].
    + rewrite !update_same. reflexivity.
    + rewrite !update_other by (right; congruence).
      eapply event_step_preserves; [exact H | ].
      destruct event as [mem addr val | mem addr val old]; cbn; auto.
      unfold isLocalWrite in His. apply Nat.eqb_neq in His. auto.
Qed.
