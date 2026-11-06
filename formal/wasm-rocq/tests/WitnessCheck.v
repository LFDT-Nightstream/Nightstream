(* Computable checks for a concrete circuit run: a witness from its nonzero entries,
   and boolean row satisfaction, continuity and replay, each sound for its Prop. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import R1CS Columns FullRelation MemoryTrace MemoryCatalog TraceRefinement ProgramSpec.
Import ListNotations.
Open Scope Z_scope.

Lemma modulus_pos : 0 < modulus.
Proof. reflexivity. Qed.

Definition field (z : Z) : FieldElement := exist _ (z mod modulus) (Z.mod_pos_bound z modulus modulus_pos).

(* A row's witness from its values, column by column. *)
Definition rowWitness (values : list Z) : Witness :=
  {| entries := fun column => field (nth column values 0) |}.

Definition satb (w : Witness) (row : Row) : bool :=
  let '(a, b, c) := row in (eval w a * eval w b - eval w c) mod modulus =? 0.

Definition satFullb (w : Witness) : bool := (w oneColumn =? 1) && forallb (satb w) rows.

Lemma satFullb_sound : forall w, satFullb w = true -> satFull w.
Proof.
  intros w H. apply andb_prop in H as [Hone Hrows]. split; [apply Z.eqb_eq; exact Hone | ].
  apply Forall_forall. intros [[a b] c] Hin. apply (proj1 (forallb_forall _ _) Hrows) in Hin.
  apply Z.eqb_eq. exact Hin.
Qed.

Definition followsb (links : list ContinuityLink) (previous next : Witness) : bool :=
  forallb (fun link => previous (fst link) =? next (snd link)) links.

Fixpoint chainb (links : list ContinuityLink) (rows : list Witness) : bool :=
  match rows with
  | first :: ((second :: _) as rest) => followsb links first second && chainb links rest
  | _ => true
  end.

Lemma chainb_sound : forall links rows, chainb links rows = true -> Chain links rows.
Proof.
  intros links rows. induction rows as [ | first [ | second rest] IH]; intros H.
  - constructor.
  - constructor.
  - cbn [chainb] in H. apply andb_prop in H as [Hfollows Hrest]. constructor; [ | exact (IH Hrest)].
    intros a b Hin. apply (proj1 (forallb_forall _ _) Hfollows) in Hin. apply Z.eqb_eq. exact Hin.
Qed.

Definition holds (state : MemoryState) (memory : nat) (address : Address) (value : Z) : bool :=
  match state memory address with Some current => current =? value | None => false end.

Fixpoint replayb (state : MemoryState) (events : list Event) : option MemoryState :=
  match events with
  | [] => Some state
  | ReadEvent memory address value :: rest =>
      if holds state memory address value then replayb state rest else None
  | WriteEvent memory address value prior :: rest =>
      if match prior with None => true | Some before => holds state memory address before end
      then replayb (update state memory address value) rest else None
  end.

Lemma holds_sound : forall state memory address value,
  holds state memory address value = true -> state memory address = Some value.
Proof.
  intros state memory address value. unfold holds.
  destruct (state memory address); [intros H; apply Z.eqb_eq in H; congruence | discriminate].
Qed.

Lemma replayb_sound : forall events state final, replayb state events = Some final -> Replay state events final.
Proof.
  induction events as [ | [memory address value | memory address value prior] rest IH]; intros state final H.
  - injection H as <-. constructor.
  - cbn [replayb] in H. destruct (holds state memory address value) eqn:Hholds; [ | discriminate].
    econstructor; [constructor; exact (holds_sound _ _ _ _ Hholds) | exact (IH _ _ H)].
  - cbn [replayb] in H.
    destruct (match prior with None => true | Some before => holds state memory address before end) eqn:Hprior;
      [ | discriminate].
    econstructor; [constructor | exact (IH _ _ H)].
    destruct prior; [exact (holds_sound _ _ _ _ Hprior) | exact I].
Qed.

(* A run checked by computation is a valid circuit run. *)
Definition replayed (initial : MemoryState) (rows : list Witness) : MemoryState :=
  match replayb initial (eventsForTrace memories rows) with Some final => final | None => initial end.

Definition validRunb (initial : MemoryState) (rows : list Witness) : bool :=
  forallb satFullb rows && chainb continuityLinks rows &&
  match replayb initial (eventsForTrace memories rows) with Some _ => true | None => false end.

Lemma validRunb_sound : forall initial rows,
  validRunb initial rows = true -> validCircuitRun initial rows (replayed initial rows).
Proof.
  intros initial rows H. unfold validRunb in H. apply andb_prop in H as [H Hreplay].
  apply andb_prop in H as [Hsat Hchain].
  split; [ | split; [exact (chainb_sound _ _ Hchain) | ]].
  - apply Forall_forall. intros w Hin. apply satFullb_sound. exact (proj1 (forallb_forall _ _) Hsat w Hin).
  - unfold replayed. destruct (replayb initial (eventsForTrace memories rows)) eqn:Hrun; [ | discriminate].
    exact (replayb_sound _ _ _ Hrun).
Qed.

(* The run's row kinds, as runFragment requires of each row. *)
Definition callerRowb (depth : Z) (row : Witness) : bool :=
  ((row COL_IS_PROGRAM_ROW =? 1) || (row COL_LOCAL_ZERO_ACTIVE_BEFORE =? 1) ||
    (row COL_FRAME_MOVE_ACTIVE_BEFORE =? 1) || (row COL_PADDING_ACTIVE =? 1)) &&
  (row COL_ACTIVATION_FINISHED_AFTER =? 0) &&
  (negb (row COL_CALL_STACK_DEPTH_BEFORE =? depth) || (row COL_CALL_STACK_POP_PRESENT =? 0)).

Lemma callerRowb_sound : forall depth rows, forallb (callerRowb depth) rows = true -> Forall (callerRow depth) rows.
Proof.
  intros depth rows H. apply Forall_forall. intros row Hin.
  pose proof (proj1 (forallb_forall _ _) H row Hin) as Hrow. unfold callerRowb in Hrow.
  apply andb_prop in Hrow as [Hrow Hbase]. apply andb_prop in Hrow as [Hkind Hlive].
  repeat split.
  - repeat (apply orb_prop in Hkind as [Hkind | Hkind]); apply Z.eqb_eq in Hkind; tauto.
  - apply Z.eqb_eq. exact Hlive.
  - intros Hdepth. rewrite Hdepth, Z.eqb_refl in Hbase. apply Z.eqb_eq. exact Hbase.
Qed.
