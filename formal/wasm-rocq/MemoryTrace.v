(* Ordered memory events extracted from the circuit's declared ports. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS.
Import ListNotations.
Open Scope Z_scope.

Inductive PortKind := Read | Write (priorColumn : option Column).
Inductive Activation := Always | When (column : Column) | Unless (column : Column).
Inductive MemoryKind := Ram | Rom.

Record PortSpec := mkPort {
  addressColumns : list Column;
  valueColumn : Column;
  portKind : PortKind;
  activation : Activation
}.

Record MemorySpec := mkMemory {
  memoryId : nat;
  memoryKind : MemoryKind;
  ports : list PortSpec
}.

Definition Address := list Z.

Inductive Event :=
| ReadEvent (memory : nat) (address : Address) (value : Z)
| WriteEvent (memory : nat) (address : Address) (value : Z) (prior : option Z).

Definition active (w : Witness) (a : Activation) : bool :=
  match a with
  | Always => true
  | When c => Z.eqb (w c) 1
  | Unless c => Z.eqb (w c) 0
  end.

Definition portEvent (memory : nat) (w : Witness) (p : PortSpec) : option Event :=
  if active w (activation p) then
    let address := map w (addressColumns p) in
    let value := w (valueColumn p) in
    Some (match portKind p with
          | Read => ReadEvent memory address value
          | Write prior => WriteEvent memory address value (option_map w prior)
          end)
  else None.

Definition eventsForRow (catalog : list MemorySpec) (w : Witness) : list Event :=
  flat_map (fun m =>
    flat_map (fun p => match portEvent (memoryId m) w p with
                       | Some event => [event] | None => [] end) (ports m)) catalog.

Definition eventsForTrace (catalog : list MemorySpec) (rows : list Witness) : list Event :=
  flat_map (eventsForRow catalog) rows.

Definition lookupPort (catalog : list MemorySpec) (memory port : nat) : option PortSpec :=
  match nth_error catalog memory with
  | Some specification => nth_error (ports specification) port
  | None => None
  end.

Lemma port_event_in_row : forall catalog w specification port event,
  In specification catalog ->
  In port (ports specification) ->
  portEvent (memoryId specification) w port = Some event ->
  In event (eventsForRow catalog w).
Proof.
  intros catalog w specification port event Hmemory Hport Hevent.
  unfold eventsForRow.
  apply in_flat_map.
  exists specification. split; [exact Hmemory |].
  apply in_flat_map.
  exists port. split; [exact Hport |].
  rewrite Hevent. simpl. auto.
Qed.

Lemma events_two_rows_order : forall catalog first second writeEvent readEvent,
  In writeEvent (eventsForRow catalog first) ->
  In readEvent (eventsForRow catalog second) ->
  exists prefix middle suffix,
    eventsForTrace catalog [first; second] =
      prefix ++ writeEvent :: middle ++ readEvent :: suffix.
Proof.
  intros catalog first second writeEvent readEvent Hwrite Hread.
  apply in_split in Hwrite.
  apply in_split in Hread.
  destruct Hwrite as [beforeWrite [afterWrite Hwrite]].
  destruct Hread as [beforeRead [afterRead Hread]].
  exists beforeWrite, (afterWrite ++ beforeRead), afterRead.
  unfold eventsForTrace. simpl.
  rewrite Hwrite, Hread.
  rewrite app_nil_r.
  repeat rewrite <- app_assoc.
  reflexivity.
Qed.

Definition MemoryState := nat -> Address -> option Z.

Definition update (state : MemoryState) (memory : nat) (address : Address) (value : Z)
    : MemoryState :=
  fun m a =>
    if Nat.eq_dec m memory then
      if list_eq_dec Z.eq_dec a address then Some value else state m a
    else state m a.

Definition priorMatches (state : MemoryState) (memory : nat)
    (address : Address) (prior : option Z) : Prop :=
  match prior with
  | None => True
  | Some value => state memory address = Some value
  end.

Inductive EventStep : MemoryState -> Event -> MemoryState -> Prop :=
| stepRead : forall state memory address value,
    state memory address = Some value ->
    EventStep state (ReadEvent memory address value) state
| stepWrite : forall state memory address value prior,
    priorMatches state memory address prior ->
    EventStep state (WriteEvent memory address value prior)
      (update state memory address value).

Inductive Replay : MemoryState -> list Event -> MemoryState -> Prop :=
| replayNil : forall state, Replay state [] state
| replayCons : forall before middle after event events,
    EventStep before event middle ->
    Replay middle events after ->
    Replay before (event :: events) after.

Definition MemoryConsistent (catalog : list MemorySpec) (initial : MemoryState)
    (rows : list Witness) : Prop :=
  exists final, Replay initial (eventsForTrace catalog rows) final.

Definition ContinuityLink := (Column * Column)%type.

Definition Follows (links : list ContinuityLink) (previous next : Witness) : Prop :=
  forall previousColumn nextColumn,
    In (previousColumn, nextColumn) links ->
    previous previousColumn = next nextColumn.

Inductive Chain (links : list ContinuityLink) : list Witness -> Prop :=
| chainNil : Chain links []
| chainOne : forall w, Chain links [w]
| chainCons : forall first second rest,
    Follows links first second ->
    Chain links (second :: rest) ->
    Chain links (first :: second :: rest).

Lemma update_same : forall state memory address value,
  update state memory address value memory address = Some value.
Proof.
  intros. unfold update.
  destruct (Nat.eq_dec memory memory) as [_ | Hneq]; [| contradiction].
  destruct (list_eq_dec Z.eq_dec address address) as [_ | Hneq]; [reflexivity | contradiction].
Qed.

Definition NoWriteTo (memory : nat) (address : Address) (event : Event) : Prop :=
  match event with
  | ReadEvent _ _ _ => True
  | WriteEvent otherMemory otherAddress _ _ =>
      otherMemory <> memory \/ otherAddress <> address
  end.

Lemma update_other : forall state memory address value targetMemory targetAddress,
  memory <> targetMemory \/ address <> targetAddress ->
  update state memory address value targetMemory targetAddress =
    state targetMemory targetAddress.
Proof.
  intros state memory address value targetMemory targetAddress Hother.
  unfold update.
  destruct (Nat.eq_dec targetMemory memory) as [Hmem | Hmem].
  - subst targetMemory.
    destruct (list_eq_dec Z.eq_dec targetAddress address) as [Haddr | Haddr].
    + subst targetAddress. destruct Hother as [Hbad | Hbad]; contradiction.
    + reflexivity.
  - reflexivity.
Qed.

Lemma event_step_preserves : forall before event after memory address,
  EventStep before event after ->
  NoWriteTo memory address event ->
  after memory address = before memory address.
Proof.
  intros before event after memory address Hstep Hother.
  inversion Hstep; subst; simpl in *.
  - reflexivity.
  - apply update_other. exact Hother.
Qed.

Lemma replay_preserves : forall before events after memory address,
  Replay before events after ->
  Forall (NoWriteTo memory address) events ->
  after memory address = before memory address.
Proof.
  intros before events after memory address Hreplay.
  induction Hreplay; intros Hno.
  - reflexivity.
  - inversion Hno; subst.
    rewrite (IHHreplay H3).
    eapply event_step_preserves; eassumption.
Qed.

Lemma read_after_last_write : forall initial middleState final middle
    memory address written observed,
  Replay (update initial memory address written) middle middleState ->
  Forall (NoWriteTo memory address) middle ->
  EventStep middleState (ReadEvent memory address observed) final ->
  observed = written.
Proof.
  intros initial middleState final middle memory address written observed
    Hmiddle Hno Hread.
  pose proof (replay_preserves _ _ _ _ _ Hmiddle Hno) as Hpreserve.
  inversion Hread; subst.
  match goal with H : _ memory address = Some observed |- _ =>
    rewrite Hpreserve in H;
    rewrite update_same in H;
    now injection H end.
Qed.

Lemma replay_app_inv : forall prefix suffix initial final,
  Replay initial (prefix ++ suffix) final ->
  exists middle, Replay initial prefix middle /\ Replay middle suffix final.
Proof.
  induction prefix as [|event prefix IH]; intros suffix initial final Hreplay.
  - exists initial. split; [constructor | exact Hreplay].
  - simpl in Hreplay.
    inversion Hreplay as [| ? ? ? ? ? Hstep Htail]; subst.
    destruct (IH suffix _ _ Htail) as [bridge [Hprefix Hsuffix]].
    exists bridge. split; [econstructor; eassumption | exact Hsuffix].
Qed.

Definition readOnlyMemory (memory : nat) (catalog : list MemorySpec) : bool :=
  forallb (fun specification =>
    if Nat.eqb (memoryId specification) memory then
      forallb (fun port => match portKind port with Read => true | _ => false end)
        (ports specification)
    else true) catalog.

Lemma read_only_row : forall catalog memory address w,
  readOnlyMemory memory catalog = true ->
  Forall (NoWriteTo memory address) (eventsForRow catalog w).
Proof.
  intros catalog memory address w Hreadonly.
  apply Forall_forall. intros event Hin.
  unfold eventsForRow in Hin.
  apply in_flat_map in Hin. destruct Hin as [specification [Hspec Hin]].
  apply in_flat_map in Hin. destruct Hin as [port [Hport Hin]].
  unfold readOnlyMemory in Hreadonly.
  apply forallb_forall with (x := specification) in Hreadonly; [|exact Hspec].
  unfold portEvent in Hin.
  destruct (active w (activation port)); [|contradiction].
  destruct (Nat.eqb (memoryId specification) memory) eqn:Hid.
  - apply forallb_forall with (x := port) in Hreadonly; [|exact Hport].
    destruct (portKind port); [|discriminate].
    destruct Hin as [Hevent | []]. subst event. exact I.
  - apply Nat.eqb_neq in Hid.
    destruct (portKind port); destruct Hin as [Hevent | []]; subst event.
    + exact I.
    + left. exact Hid.
Qed.

Lemma read_only_trace : forall catalog memory address rows,
  readOnlyMemory memory catalog = true ->
  Forall (NoWriteTo memory address) (eventsForTrace catalog rows).
Proof.
  intros catalog memory address rows Hreadonly.
  induction rows as [|w rows IH]; [constructor |].
  simpl. apply Forall_app. split; [apply read_only_row |]; assumption.
Qed.

Lemma replay_read_from_initial : forall initial final events memory address value,
  Replay initial events final ->
  Forall (NoWriteTo memory address) events ->
  In (ReadEvent memory address value) events ->
  initial memory address = Some value.
Proof.
  intros initial final events memory address value Hreplay Hno Hin.
  apply in_split in Hin. destruct Hin as [prefix [suffix Hevents]].
  subst events.
  apply Forall_app in Hno. destruct Hno as [Hprefix _].
  destruct (replay_app_inv prefix _ _ _ Hreplay)
    as [middle [HreplayPrefix Htail]].
  inversion Htail; subst.
  match goal with H : EventStep _ (ReadEvent _ _ _) _ |- _ =>
    inversion H; subst end.
  pose proof (replay_preserves _ _ _ _ _ HreplayPrefix Hprefix) as Hpreserve.
  congruence.
Qed.

Lemma replay_last_write_read : forall initial final prefix middle suffix
    memory address written observed prior,
  Replay initial
    (prefix ++ WriteEvent memory address written prior ::
      middle ++ ReadEvent memory address observed :: suffix) final ->
  Forall (NoWriteTo memory address) middle ->
  observed = written.
Proof.
  intros initial final prefix middle suffix memory address written observed prior
    Hreplay Hno.
  destruct (replay_app_inv prefix _ _ _ Hreplay)
    as [beforeWrite [_ Htail]].
  inversion Htail; subst.
  match goal with H : EventStep _ (WriteEvent _ _ _ _) _ |- _ =>
    inversion H; subst; clear H end.
  match goal with H : Replay _ (middle ++ ReadEvent _ _ _ :: suffix) _ |- _ =>
    destruct (replay_app_inv middle _ _ _ H)
      as [beforeRead [Hmiddle HreadTail]] end.
  inversion HreadTail; subst.
  match goal with H : EventStep _ (ReadEvent _ _ _) _ |- _ =>
    eapply read_after_last_write; [exact Hmiddle | exact Hno | exact H] end.
Qed.

Lemma replay_last_write_write_prior : forall initial final prefix middle suffix
    memory address written observed nextValue firstPrior,
  Replay initial
    (prefix ++ WriteEvent memory address written firstPrior ::
      middle ++ WriteEvent memory address nextValue (Some observed) :: suffix) final ->
  Forall (NoWriteTo memory address) middle ->
  observed = written.
Proof.
  intros initial final prefix middle suffix memory address
    written observed nextValue firstPrior Hreplay Hno.
  destruct (replay_app_inv prefix _ _ _ Hreplay)
    as [beforeWrite [_ Htail]].
  inversion Htail; subst.
  match goal with H : EventStep _ (WriteEvent _ _ _ _) _ |- _ =>
    inversion H; subst; clear H end.
  match goal with H : Replay _ (middle ++ WriteEvent _ _ _ (Some _) :: suffix) _ |- _ =>
    destruct (replay_app_inv middle _ _ _ H)
      as [beforeNext [Hmiddle HnextTail]] end.
  inversion HnextTail; subst.
  match goal with H : EventStep _ (WriteEvent _ _ _ (Some _)) _ |- _ =>
    inversion H; subst; clear H end.
  match goal with H : priorMatches _ memory address (Some observed) |- _ =>
    simpl in H;
    rewrite (replay_preserves _ _ _ _ _ Hmiddle Hno) in H;
    rewrite update_same in H;
    now injection H end.
Qed.
