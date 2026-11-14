(* Two logical ROMs share disjoint cells. Grouped ROM reads move across a RAM
   read-modify-write, but the RAM event order cannot change. *)
From Stdlib Require Import List ZArith Bool.
Require Import MemoryTrace RomSharing.
Import ListNotations.
Open Scope Z_scope.

Definition rom (m : nat) := negb (Nat.eqb m 0).
Definition table (m : nat) := if Nat.eqb m 0 then 0%nat else 3%nat.
Definition logical : MemoryState := fun m a =>
  match m, a with 1%nat, [0] => Some 7 | 2%nat, [1] => Some 9 | _, _ => Some 0 end.
Definition shared : MemoryState := fun m a =>
  match m, a with 3%nat, [0] => Some 7 | 3%nat, [1] => Some 9 | _, _ => Some 0 end.
Definition logicalEvents := [ReadEvent 1 [0] 7; WriteEvent 0 [0] 5 (Some 0);
  ReadEvent 2 [1] 9; ReadEvent 0 [0] 5].
Definition sharedEvents := [ReadEvent 3 [1] 9; WriteEvent 0 [0] 5 (Some 0);
  ReadEvent 0 [0] 5; ReadEvent 3 [0] 7].

Example shared_run : Replay shared sharedEvents (update shared 0 [0] 5).
Proof.
  unfold sharedEvents. repeat (econstructor; try reflexivity).
Qed.

Example bridge_with_reordered_reads : exists final,
  Replay logical logicalEvents final /\ final 0%nat [0] = Some 5 /\ final 1%nat [1] = Some 0.
Proof.
  assert (Himages : ImagesAgree rom table logical shared logicalEvents).
  { intros m a v Hin Hrom. cbn in Hin.
    destruct Hin as [H | [H | [H | [H | []]]]]; inversion H; subst; reflexivity. }
  assert (Htrace : SharedTrace rom table logicalEvents sharedEvents).
  { split; [reflexivity | ]. intros m a v Hin Hrom. cbn in Hin.
    destruct Hin as [H | [H | [H | [H | []]]]]; inversion H; subst; try discriminate;
      split; try reflexivity; cbn; tauto. }
  destruct (shared_rom_replay rom table logicalEvents sharedEvents logical shared
    (update shared 0 [0] 5)) as [final [Hrun Himage]].
  - repeat constructor.
  - repeat constructor.
  - exact Htrace.
  - exact Himages.
  - intros [ | m] a H; [reflexivity | discriminate].
  - exact shared_run.
  - exists final. split; [exact Hrun | ]. split; rewrite Himage; reflexivity.
Qed.

Example changed_owned_cell_rejected :
  ~ ImagesAgree rom table logical (update shared 3 [0] 8) logicalEvents.
Proof.
  intro H. specialize (H 1%nat [0] 7 (or_introl eq_refl) eq_refl). discriminate.
Qed.

Example ram_order_must_be_preserved :
  ~ SharedTrace rom table logicalEvents
    [ReadEvent 0 [0] 5; WriteEvent 0 [0] 5 (Some 0); ReadEvent 3 [0] 7; ReadEvent 3 [1] 9].
Proof. intros [H _]. discriminate. Qed.

Example wrong_prior_value_rejected :
  ~ Replay shared [WriteEvent 0 [0] 5 (Some 1)] (update shared 0 [0] 5).
Proof. intro H. inversion H; subst. match goal with H : EventStep _ _ _ |- _ => inversion H; subst end.
  match goal with H : priorMatches _ _ _ _ |- _ => discriminate H end.
Qed.

Example conflicting_owners_rejected : forall image,
  ~ ImagesAgree rom table
    (fun m a => if Nat.eqb m 1 then Some 7 else Some 9) image
    [ReadEvent 1 [0] 7; ReadEvent 2 [0] 9].
Proof.
  intros image H.
  pose proof (H 1%nat [0] 7 (or_introl eq_refl) eq_refl) as Hfirst.
  pose proof (H 2%nat [0] 9 (or_intror (or_introl eq_refl)) eq_refl) as Hsecond.
  cbn in Hfirst, Hsecond. congruence.
Qed.
