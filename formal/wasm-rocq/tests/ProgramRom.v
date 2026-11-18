(* The program ROMs that Rust preloads for the corpus of
   crates/neo-wasm/tests/wasm_rocq_program_rom_export.rs are the ROMs of
   `encodeProgram`, so each corpus program is loaded in its Rust preload. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import MemoryTrace ProgramSpec tests.ProgramRomTrace.
Import ListNotations.
Open Scope Z_scope.

Definition Cells := list (nat * Address * Z).

Definition cellAt (cells : Cells) (id : nat) (address : Address) : Z :=
  match find (fun cell => Nat.eqb (fst (fst cell)) id &&
      if list_eq_dec Z.eq_dec (snd (fst cell)) address then true else false) cells with
  | Some cell => snd cell
  | None => 0
  end.

Definition preloaded (cells : Cells) : MemoryState := fun id address => Some (cellAt cells id address).

Definition programSize (program : Program) : nat :=
  fold_right (fun f size => functionSize f + size)%nat 0%nat program.

Definition programKeys (program : Program) : nat := Nat.max (programSize program) (S (length program)).

Definition inRange (program : Program) (address : Address) : Prop :=
  match address with [key] => 0 <= key < Z.of_nat (programKeys program) | _ => False end.

Definition programRom (id : nat) : bool := existsb (Nat.eqb id) programRomIds.

Definition cellInRange (program : Program) (cell : nat * Address * Z) : bool :=
  negb (programRom (fst (fst cell))) ||
  match snd (fst cell) with
  | [key] => (0 <=? key) && (key <? Z.of_nat (programKeys program))
  | _ => false
  end.

Definition agrees (program : Program) (cells : Cells) : bool :=
  forallb (cellInRange program) cells &&
  forallb (fun id => forallb (fun key =>
      romValue (encodeProgram program id [Z.of_nat key]) =? cellAt cells id [Z.of_nat key])
    (seq 0 (programKeys program))) programRomIds.

Lemma locate_past : forall program pc, (programSize program <= pc)%nat -> locatePC program pc = None.
Proof.
  unfold programSize. induction program as [ | f rest IH]; intros pc Hpc; [reflexivity | ].
  cbn [fold_right] in Hpc. cbn [locatePC].
  destruct (pc <? functionSize f)%nat eqn:Hlt; [apply Nat.ltb_lt in Hlt; lia | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma encode_outside : forall program id address,
  ~ inRange program address -> encodeProgram program id address = None.
Proof.
  intros program id [ | key [ | next rest]] Hout; try reflexivity.
  cbn [inRange] in Hout. unfold programKeys in Hout.
  assert (Hslide : slideRom program id key = None).
  { unfold slideRom. destruct (key <? 0) eqn:Hneg; [reflexivity | ]. apply Z.ltb_ge in Hneg.
    rewrite locate_past by lia. reflexivity. }
  assert (Hfunction : functionRom program id key = None).
  { unfold functionRom. destruct (key <=? 0) eqn:Hpos; [reflexivity | ]. apply Z.leb_gt in Hpos.
    unfold functionAt. rewrite (proj2 (nth_error_None program (Z.to_nat (key - 1)))) by lia. reflexivity. }
  assert (Hinstruction : instructionRom program id key = None).
  { unfold instructionRom, instructionAtPC. destruct (key <? 0) eqn:Hneg; [reflexivity | ].
    apply Z.ltb_ge in Hneg. rewrite locate_past by lia. reflexivity. }
  unfold encodeProgram.
  repeat match goal with |- context [if ?b then _ else _] => destruct b end; assumption.
Qed.

Lemma cell_outside : forall program cells id address,
  forallb (cellInRange program) cells = true -> programRom id = true -> ~ inRange program address ->
  cellAt cells id address = 0.
Proof.
  intros program cells id address Hcells Hrom Hout. unfold cellAt.
  match goal with |- context [find ?matches cells] =>
    destruct (find matches cells) as [cell | ] eqn:Hfind; [ | reflexivity] end.
  apply find_some in Hfind as [Hin Hmatch].
  destruct cell as [[cellId cellAddress] value]. cbn [fst snd] in Hmatch.
  apply andb_true_iff in Hmatch as [Hid Haddress]. apply Nat.eqb_eq in Hid. subst cellId.
  destruct (list_eq_dec Z.eq_dec cellAddress address) as [-> | ]; [ | discriminate].
  pose proof (proj1 (forallb_forall _ _) Hcells _ Hin) as Hcell.
  unfold cellInRange in Hcell. cbn [fst snd] in Hcell. rewrite Hrom in Hcell. cbn [negb orb] in Hcell.
  destruct address as [ | key [ | next rest]]; try discriminate.
  apply andb_true_iff in Hcell as [Hlow Hhigh]. apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  exfalso. apply Hout. cbn [inRange]. lia.
Qed.

Theorem agrees_loaded : forall program cells,
  agrees program cells = true -> programLoaded program (preloaded cells).
Proof.
  intros program cells Hagree id Hid address. unfold preloaded. f_equal.
  apply andb_true_iff in Hagree as [Hcells Hvalues].
  assert (Hrom : programRom id = true)
    by (apply existsb_exists; exists id; split; [exact Hid | apply Nat.eqb_refl]).
  assert (Hdec : inRange program address \/ ~ inRange program address).
  { destruct address as [ | key [ | next rest]]; cbn [inRange]; lia. }
  destruct Hdec as [Hin | Hout].
  - destruct address as [ | key [ | next rest]]; try contradiction. cbn [inRange] in Hin.
    pose proof (proj1 (forallb_forall _ _) Hvalues id Hid) as Hkeys.
    pose proof (proj1 (forallb_forall _ _) Hkeys (Z.to_nat key)
      ltac:(apply in_seq; lia)) as Hkey.
    cbv beta in Hkey. rewrite Z2Nat.id in Hkey by lia. apply Z.eqb_eq in Hkey. symmetry. exact Hkey.
  - rewrite encode_outside, (cell_outside program) by assumption. reflexivity.
Qed.

Example corpus_agrees : forallb (fun entry => agrees (fst entry) (snd entry)) corpus = true.
Proof. vm_compute. reflexivity. Qed.

Theorem corpus_loaded : Forall (fun entry => programLoaded (fst entry) (preloaded (snd entry))) corpus.
Proof.
  apply Forall_forall. intros entry Hin. apply agrees_loaded.
  exact (proj1 (forallb_forall _ _) corpus_agrees entry Hin).
Qed.
