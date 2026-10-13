(* Circuit trace validity, with an explicit final memory state. *)
From Stdlib Require Import List.
Require Import R1CS FullRelation MemoryTrace MemoryCatalog.
Import ListNotations.

Definition validCircuitRun (initial : MemoryState) (rows : list Witness)
    (final : MemoryState) : Prop :=
  Forall FullRelation.satFull rows /\
  Chain MemoryCatalog.continuityLinks rows /\
  Replay initial (eventsForTrace MemoryCatalog.memories rows) final.

Lemma valid_run_head : forall initial final first rest,
  validCircuitRun initial (first :: rest) final ->
  exists middle,
    FullRelation.satFull first /\
    Replay initial (eventsForRow MemoryCatalog.memories first) middle /\
    validCircuitRun middle rest final.
Proof.
  intros initial final first rest [Hsat [Hchain Hreplay]].
  destruct (replay_app_inv (eventsForRow MemoryCatalog.memories first)
    (eventsForTrace MemoryCatalog.memories rest) initial final Hreplay)
    as [middle [Hhead Htail]].
  assert (HtailChain : Chain MemoryCatalog.continuityLinks rest).
  { inversion Hchain; subst; auto using chainNil. }
  exists middle. split; [exact (Forall_inv Hsat) |].
  split; [exact Hhead |].
  split; [exact (Forall_inv_tail Hsat) |].
  split; assumption.
Qed.

Lemma chain_prefix : forall links prefix suffix,
  Chain links (prefix ++ suffix) -> Chain links prefix.
Proof.
  intros links prefix. induction prefix as [ | first rest IH]; intros suffix Hchain.
  - constructor.
  - destruct rest as [ | next rest]; [constructor | ].
    inversion Hchain; subst. constructor; [assumption | ]. eapply IH; eassumption.
Qed.

Lemma chain_suffix : forall links prefix suffix,
  Chain links (prefix ++ suffix) -> Chain links suffix.
Proof.
  intros links prefix. induction prefix as [ | first rest IH]; intros suffix Hchain.
  - exact Hchain.
  - apply IH. inversion Hchain; subst; try constructor; assumption.
Qed.

Lemma valid_run_split : forall prefix suffix initial final,
  validCircuitRun initial (prefix ++ suffix) final ->
  exists middle, validCircuitRun initial prefix middle /\ validCircuitRun middle suffix final.
Proof.
  intros prefix suffix initial final [Hsat [Hchain Hreplay]].
  apply Forall_app in Hsat. destruct Hsat as [Hprefix Hsuffix].
  unfold eventsForTrace in Hreplay. rewrite flat_map_app in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [Hfirst Hrest]].
  exists middle. split; repeat split; try assumption.
  - eapply chain_prefix; exact Hchain.
  - eapply chain_suffix; exact Hchain.
Qed.

Lemma replay_app : forall initial prefix middle suffix final,
  Replay initial prefix middle -> Replay middle suffix final ->
  Replay initial (prefix ++ suffix) final.
Proof.
  intros initial prefix middle suffix final Hprefix Hsuffix.
  induction Hprefix; cbn; [exact Hsuffix | econstructor; eauto].
Qed.
