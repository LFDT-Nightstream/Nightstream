(* ROM sharing at the event boundary; physical slot allocation stays separate. *)
From Stdlib Require Import List ZArith Bool.
Require Import MemoryTrace.
Import ListNotations.

Definition eventMemory (e : Event) : nat :=
  match e with ReadEvent m _ _ | WriteEvent m _ _ _ => m end.
Definition ramEvents (rom : nat -> bool) (events : list Event) :=
  filter (fun e => negb (rom (eventMemory e))) events.
Definition readsOnly (rom : nat -> bool) (events : list Event) : Prop :=
  Forall (fun e => match e with ReadEvent _ _ _ => True
    | WriteEvent m _ _ _ => rom m = false end) events.

Lemma erase_rom_reads : forall rom events before after,
  Replay before events after -> readsOnly rom events ->
  Replay before (ramEvents rom events) after.
Proof.
  intros rom events before after H. induction H; intros Hro; [constructor | ].
  inversion Hro as [ | ? ? Hhead Htail]; subst.
  unfold ramEvents in *. cbn [filter].
  destruct event as [m a v | m a v prior]; cbn [eventMemory] in *.
  - destruct (rom m); cbn; inversion H; subst; [apply IHReplay | econstructor]; eauto.
  - rewrite Hhead. cbn. econstructor; eauto.
Qed.

(* ROM stays fixed; RAM agrees with the shared execution. Pointwise equality
   avoids a functional-extensionality assumption. *)
Definition imageRelation (rom : nat -> bool) (initial logical shared : MemoryState) : Prop :=
  forall m a, logical m a = if rom m then initial m a else shared m a.

Lemma update_image_relation : forall rom initial logical shared m a v,
  imageRelation rom initial logical shared -> rom m = false ->
  imageRelation rom initial (update logical m a v) (update shared m a v).
Proof.
  intros rom initial logical shared m a v H Hram target address.
  unfold update. destruct (Nat.eq_dec target m) as [-> | Hne].
  - rewrite Hram. destruct (list_eq_dec Z.eq_dec address a); [reflexivity | ].
    specialize (H m address). rewrite Hram in H. exact H.
  - apply H.
Qed.

Lemma restore_rom_reads : forall rom initial events shared final logical,
  readsOnly rom events ->
  (forall m a v, In (ReadEvent m a v) events -> rom m = true -> initial m a = Some v) ->
  Replay shared (ramEvents rom events) final ->
  imageRelation rom initial logical shared ->
  exists last, Replay logical events last /\ imageRelation rom initial last final.
Proof.
  intros rom initial events. induction events as [ | event events IH];
    intros shared final logical Hro Hreads Hreplay Himage.
  - inversion Hreplay; subst. exists logical. split; [constructor | assumption].
  - inversion Hro as [ | ? ? Hhead Htail]; subst.
    assert (Hrest : forall m a v, In (ReadEvent m a v) events -> rom m = true -> initial m a = Some v)
      by (intros; apply Hreads; auto using in_cons).
    destruct event as [m a v | m a v prior].
    + destruct (rom m) eqn:Hrom.
      * cbn [ramEvents filter eventMemory] in Hreplay. rewrite Hrom in Hreplay. cbn in Hreplay.
        destruct (IH shared final logical Htail Hrest Hreplay Himage) as [last [Hrun Hlast]].
        exists last. split; [ | assumption]. econstructor; [constructor | exact Hrun].
        rewrite Himage, Hrom. apply Hreads; [left; reflexivity | exact Hrom].
      * cbn [ramEvents filter eventMemory] in Hreplay. rewrite Hrom in Hreplay. cbn in Hreplay.
        inversion Hreplay as [ | ? middle ? ? ? Hstep Hrun]; subst.
        inversion Hstep; subst.
        destruct (IH _ final logical Htail Hrest Hrun Himage) as [last [Hlogical Hlast]].
        exists last. split; [ | assumption]. econstructor; [constructor | exact Hlogical].
        rewrite Himage, Hrom. assumption.
    + cbn in Hhead. cbn [ramEvents filter eventMemory] in Hreplay. rewrite Hhead in Hreplay. cbn in Hreplay.
      inversion Hreplay as [ | ? middle ? ? ? Hstep Hrun]; subst.
      inversion Hstep; subst.
      assert (Hprior : priorMatches logical m a prior).
      { destruct prior; [ | exact I]. cbn in *. rewrite Himage, Hhead. assumption. }
      destruct (IH _ final (update logical m a v) Htail Hrest Hrun
        (update_image_relation _ _ _ _ _ _ _ Himage Hhead)) as [last [Hlogical Hlast]].
      exists last. split; [ | assumption]. econstructor; [constructor; exact Hprior | exact Hlogical].
Qed.

(* Ownership and opcode-bound gates establish agreement only at active reads.
   Unused logical ROM cells may differ. RAM IDs are unchanged (or normalized
   into the logical namespace before applying this theorem). *)
Definition ImagesAgree (rom : nat -> bool) (table : nat -> nat)
    (logical shared : MemoryState) (events : list Event) : Prop :=
  forall m a v, In (ReadEvent m a v) events -> rom m = true ->
    logical m a = shared (table m) a.
Definition SharedTrace (rom : nat -> bool) (table : nat -> nat)
    (logical shared : list Event) : Prop :=
  ramEvents rom logical = ramEvents rom shared /\
  forall m a v, In (ReadEvent m a v) logical -> rom m = true ->
    rom (table m) = true /\ In (ReadEvent (table m) a v) shared.

Theorem shared_rom_replay : forall rom table logicalEvents sharedEvents logicalInitial sharedInitial sharedFinal,
  readsOnly rom logicalEvents -> readsOnly rom sharedEvents ->
  SharedTrace rom table logicalEvents sharedEvents ->
  ImagesAgree rom table logicalInitial sharedInitial logicalEvents ->
  (forall m a, rom m = false -> logicalInitial m a = sharedInitial m a) ->
  Replay sharedInitial sharedEvents sharedFinal ->
  exists logicalFinal, Replay logicalInitial logicalEvents logicalFinal /\
    imageRelation rom logicalInitial logicalFinal sharedFinal.
Proof.
  intros rom table logicalEvents sharedEvents logicalInitial sharedInitial sharedFinal
    Hlogical Hshared [Hram Htrace] Himages Hinitial Hrun.
  eapply restore_rom_reads; [exact Hlogical | | | ].
  - intros m a v Hin Hrom. rewrite (Himages m a v Hin Hrom).
    destruct (Htrace m a v Hin Hrom) as [Htable Hread].
    eapply replay_read_from_initial; [exact Hrun | | exact Hread].
    apply Forall_forall. intros event Hevent.
    apply (proj1 (Forall_forall _ _) Hshared) in Hevent.
    destruct event as [other address value | other address value prior]; [exact I | ].
    left. intro Heq. subst other. cbn in Hevent. congruence.
  - rewrite Hram. exact (erase_rom_reads rom sharedEvents sharedInitial sharedFinal Hrun Hshared).
  - intros m a. destruct (rom m) eqn:Hrom; [reflexivity | apply Hinitial; assumption].
Qed.
