(* The Lean copy of the memory-event model (`Spec/Nebula/MemoryTrace.lean` in
   formal/nightstream-fprime) against MemoryTrace.v on the golden traces of the Lean
   parity file: the same events, the same replay verdict and final values, and the
   same bounded-event verdict for each row. `replayb` and the bound check decide the
   Rocq predicates, so a verdict of the checks is a verdict of `Replay` and
   `boundedMemoryEvents`. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS MemoryTrace tests.WitnessCheck tests.LeanParityTrace.
Import ListNotations.
Open Scope Z_scope.

Lemma replayb_complete : forall state events final,
  Replay state events final -> replayb state events = Some final.
Proof.
  intros state events final Hreplay. induction Hreplay as [ | before middle after event events Hstep _ IH]; [reflexivity | ].
  inversion Hstep as [? memory address value Hheld | ? memory address value prior Hprior]; subst; cbn [replayb].
  - unfold holds. rewrite Hheld, Z.eqb_refl. exact IH.
  - destruct prior as [before' | ]; [ | exact IH].
    cbn in Hprior. unfold holds. rewrite Hprior, Z.eqb_refl. exact IH.
Qed.

Lemma bounded_events_check_complete : forall events,
  boundedMemoryEvents events -> forallb eventValuesBoundedb events = true.
Proof.
  intros events Hbounds. apply forallb_forall. intros event Hin.
  pose proof (proj1 (Forall_forall _ _) Hbounds event Hin) as Hevent.
  assert (Hword : forall v, u32Value v -> u32Valueb v = true).
  { intros v [Hlo Hhi]. unfold u32Valueb. apply andb_true_intro.
    split; [apply Z.leb_le | apply Z.ltb_lt]; assumption. }
  destruct event as [memory address value | memory address value [before | ]]; cbn in *.
  - exact (Hword _ Hevent).
  - destruct Hevent as [Hvalue Hbefore]. rewrite (Hword _ Hvalue), (Hword _ Hbefore). reflexivity.
  - destruct Hevent as [Hvalue _]. rewrite (Hword _ Hvalue). reflexivity.
Qed.

Definition stateOf (cells : list (nat * list Z * Z)) : MemoryState :=
  fold_left (fun state '(memory, address, value) => update state memory address value) cells
    (fun _ _ => None).

Definition event_eq_dec : forall e f : Event, {e = f} + {e <> f}.
Proof. decide equality; repeat decide equality; apply Z.eq_dec. Defined.

Definition sameList {A : Type} (eq_dec : forall x y : A, {x = y} + {x <> y}) (xs ys : list A) : bool :=
  if list_eq_dec eq_dec xs ys then true else false.

Definition option_eq_dec : forall x y : option Z, {x = y} + {x <> y}.
Proof. decide equality; apply Z.eq_dec. Defined.

Definition agrees (c : list MemorySpec * list (list Z) * list (nat * list Z * Z) *
    list (nat * list Z) * (list Event * bool * list (option Z) * list bool)) : bool :=
  let '(catalog, rows, initial, queries, (events, replays, finals, bounded)) := c in
  let witnesses := map rowWitness rows in
  let trace := eventsForTrace catalog witnesses in
  sameList event_eq_dec trace events &&
  match replayb (stateOf initial) trace with
  | Some final => Bool.eqb replays true &&
      sameList option_eq_dec (map (fun '(memory, address) => final memory address) queries) finals
  | None => Bool.eqb replays false && sameList option_eq_dec finals []
  end &&
  sameList bool_dec (map (fun w => forallb eventValuesBoundedb (eventsForRow catalog w)) witnesses) bounded.

Example lean_parity : forallb agrees cases = true.
Proof. vm_compute. reflexivity. Qed.
