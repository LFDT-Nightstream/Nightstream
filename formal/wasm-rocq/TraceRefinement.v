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
