(* Flatten blocks into instruction addresses, recursively: a BLOCK, its body, then
   its END, as the Rust preprocessing numbers operators. A branch inside a block
   body stores its distance to the instruction after that block's END (the depth-0
   target; deeper branches are outside the fragment). *)
From Stdlib Require Import List Arith Lia.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StackSimulation.
Import ListNotations.

Record FlatInstruction := flatInstruction {
  flatOpcode : option instr;
  flatBranchDistance : option nat
}.

Fixpoint flattenInstruction (i : instr) : list FlatInstruction :=
  match i with
  | BLOCK _ body =>
      flatInstruction (Some i) None ::
        (fix blockCode (code : list instr) : list FlatInstruction :=
           match code with
           | [] => []
           | j :: rest =>
               let tail := blockCode rest in
               match j with
               | BR_IF _ => flatInstruction (Some j) (Some (S (S (length tail)))) :: tail
               | _ => flattenInstruction j ++ tail
               end
           end) body ++ [flatInstruction None None]
  | _ => [flatInstruction (Some i) None]
  end.

(* A block body, as flattenInstruction lays it out. *)
Fixpoint blockCode (code : list instr) : list FlatInstruction :=
  match code with
  | [] => []
  | j :: rest =>
      let tail := blockCode rest in
      match j with
      | BR_IF _ => flatInstruction (Some j) (Some (S (S (length tail)))) :: tail
      | _ => flattenInstruction j ++ tail
      end
  end.

Definition flattenCode (code : list instr) := flat_map flattenInstruction code.
Definition instructionSize (i : instr) := length (flattenInstruction i).
Definition codeSize (code : list instr) := length (flattenCode code).
Definition codeOffset (code : list instr) (offset : nat) := codeSize (firstn offset code).

(* A position inside nested blocks: the offsets of the enclosing BLOCKs, outermost
   first. pathCode is the innermost body; pathOffset is the flattened address of an
   offset in it, relative to the start of code. *)
Fixpoint pathCode (code : list instr) (labels : list nat) : option (list instr) :=
  match labels with
  | [] => Some code
  | b :: rest => match nth_error code b with
      | Some (BLOCK _ body) => pathCode body rest | _ => None end
  end.

Fixpoint pathOffset (code : list instr) (labels : list nat) (offset : nat) : nat :=
  match labels with
  | [] => codeOffset code offset
  | b :: rest => match nth_error code b with
      | Some (BLOCK _ body) => codeOffset code b + S (pathOffset body rest offset) | _ => 0 end
  end.

Lemma flatten_block : forall bt body,
  flattenInstruction (BLOCK bt body) =
    flatInstruction (Some (BLOCK bt body)) None :: blockCode body ++ [flatInstruction None None].
Proof. reflexivity. Qed.

Lemma code_size_app : forall a b, codeSize (a ++ b) = codeSize a + codeSize b.
Proof. intros; unfold codeSize, flattenCode; rewrite flat_map_app, length_app; reflexivity. Qed.
Lemma code_size_cons : forall i rest, codeSize (i :: rest) = instructionSize i + codeSize rest.
Proof. intros; unfold codeSize, flattenCode; cbn [flat_map]; apply length_app. Qed.
Lemma ordinary_instruction_size : forall i,
  (exists op, encodedOpcode i = Some op) -> instructionSize i = 1.
Proof.
  intros i [op H]. destruct (encoded_opcode_cases _ _ H) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity.
Qed.
Lemma firstn_at_instruction : forall (code : list instr) offset i,
  nth_error code offset = Some i -> firstn (S offset) code = firstn offset code ++ [i].
Proof.
  induction code as [ | a code IH]; intros [ | offset] i H; try discriminate.
  - injection H as <-. reflexivity.
  - change (a :: firstn (S offset) code = a :: (firstn offset code ++ [i])).
    f_equal. apply IH. exact H.
Qed.
Lemma code_offset_next : forall code offset i,
  nth_error code offset = Some i -> codeOffset code (S offset) = codeOffset code offset + instructionSize i.
Proof.
  intros code offset i H. unfold codeOffset.
  rewrite (firstn_at_instruction code offset i H), code_size_app, code_size_cons.
  change (codeSize (firstn offset code) + (instructionSize i + 0) = codeSize (firstn offset code) + instructionSize i).
  lia.
Qed.
Lemma code_offset_end : forall code, codeOffset code (length code) = codeSize code.
Proof. intros; unfold codeOffset; now rewrite firstn_all. Qed.

Lemma flatten_within_instruction : forall code offset i inner slot,
  nth_error code offset = Some i -> nth_error (flattenInstruction i) inner = Some slot ->
  nth_error (flattenCode code) (codeOffset code offset + inner) = Some slot.
Proof.
  intros code offset i inner slot Hsource Hslot.
  rewrite <- (firstn_skipn offset code) at 1. unfold flattenCode at 1. rewrite flat_map_app.
  rewrite nth_error_app2 by (unfold codeOffset, codeSize, flattenCode; lia).
  replace (codeOffset code offset + inner - length (flat_map flattenInstruction (firstn offset code)))
    with inner by (unfold codeOffset, codeSize, flattenCode; lia).
  rewrite (skipn_at_instruction code offset i Hsource). cbn [flat_map].
  rewrite nth_error_app1; [exact Hslot | apply nth_error_Some; rewrite Hslot; discriminate].
Qed.

(* One instruction of a block body: a branch carries its distance to the END. *)
Definition blockSlot (i : instr) (rest : list instr) : list FlatInstruction :=
  match i with
  | BR_IF _ => [flatInstruction (Some i) (Some (S (S (codeSize rest))))]
  | _ => flattenInstruction i
  end.

Lemma block_slot_length : forall i rest, length (blockSlot i rest) = instructionSize i.
Proof. intros i rest. destruct i; reflexivity. Qed.

Lemma block_slot_head : forall i rest, exists more,
  blockSlot i rest = flatInstruction (Some i) (match i with BR_IF _ => Some (S (S (codeSize rest))) | _ => None end) :: more.
Proof. intros i rest. destruct i; eexists; reflexivity. Qed.

Lemma block_code_length : forall code, length (blockCode code) = codeSize code.
Proof.
  induction code as [ | j rest IH]; [reflexivity | ]. rewrite code_size_cons, <- IH.
  destruct j; cbn [blockCode]; rewrite ?length_app; reflexivity.
Qed.

Lemma block_code_cons : forall j rest, blockCode (j :: rest) = blockSlot j rest ++ blockCode rest.
Proof.
  intros j rest. destruct j; try reflexivity. cbn [blockCode blockSlot]. rewrite block_code_length. reflexivity.
Qed.

Lemma block_code_at : forall code offset i k s,
  nth_error code offset = Some i -> nth_error (blockSlot i (skipn (S offset) code)) k = Some s ->
  nth_error (blockCode code) (codeOffset code offset + k) = Some s.
Proof.
  induction code as [ | j rest IH]; intros [ | offset] i k s Hi Hs; try discriminate.
  - injection Hi as <-. cbn [skipn] in Hs. rewrite block_code_cons. change (codeOffset (j :: rest) 0) with 0%nat.
    cbn [Nat.add].
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hs; discriminate). exact Hs.
  - rewrite block_code_cons. unfold codeOffset. cbn [firstn]. rewrite code_size_cons.
    rewrite nth_error_app2 by (rewrite block_slot_length; lia). rewrite block_slot_length.
    replace (instructionSize j + codeSize (firstn offset rest) + k - instructionSize j)
      with (codeOffset rest offset + k) by (unfold codeOffset; lia).
    exact (IH _ _ _ _ Hi ltac:(cbn [skipn] in Hs |- *; exact Hs)).
Qed.

Lemma flatten_block_inner : forall bt body inner,
  nth_error (flattenInstruction (BLOCK bt body)) (S inner) =
    nth_error (blockCode body ++ [flatInstruction None None]) inner.
Proof. reflexivity. Qed.

(* Slots of a body at any depth of nesting, within the enclosing block's layout. *)
Lemma block_path_slot : forall labels body inner offset i k s,
  pathCode body labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (blockSlot i (skipn (S offset) inner)) k = Some s ->
  nth_error (blockCode body) (pathOffset body labels offset + k) = Some s.
Proof.
  induction labels as [ | b rest IH]; intros body inner offset i k s Hpath Hi Hs.
  - injection Hpath as <-. exact (block_code_at _ _ _ _ _ Hi Hs).
  - cbn [pathCode pathOffset] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with |- context [codeOffset body b + S (pathOffset ?nested rest offset) + k] =>
      replace (codeOffset body b + S (pathOffset nested rest offset) + k)
        with (codeOffset body b + S (pathOffset nested rest offset + k)) by lia end.
    apply (block_code_at _ _ _ _ _ Hb). cbn [blockSlot]. rewrite flatten_block_inner.
    pose proof (IH _ _ _ _ _ _ Hpath Hi Hs) as Hinner.
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hinner; discriminate). exact Hinner.
Qed.

Lemma block_path_end : forall labels body inner,
  pathCode body labels = Some inner ->
  nth_error (blockCode body ++ [flatInstruction None None]) (pathOffset body labels (length inner)) =
    Some (flatInstruction None None).
Proof.
  induction labels as [ | b rest IH]; intros body inner Hpath.
  - injection Hpath as <-. cbn [pathOffset]. rewrite code_offset_end, <- block_code_length.
    rewrite nth_error_app2, Nat.sub_diag by lia. reflexivity.
  - cbn [pathCode pathOffset] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with |- context [codeOffset body b + S (pathOffset ?nested rest (length inner))] =>
      assert (Hslot : nth_error (blockCode body) (codeOffset body b + S (pathOffset nested rest (length inner))) =
          Some (flatInstruction None None)) end.
    { apply (block_code_at _ _ _ _ _ Hb). cbn [blockSlot]. rewrite flatten_block_inner. exact (IH _ _ Hpath). }
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hslot; discriminate). exact Hslot.
Qed.

(* The same slots within a function body, below a top-level BLOCK. *)
Lemma flatten_in_block : forall code b bt body labels inner offset i k s,
  nth_error code b = Some (BLOCK bt body) -> pathCode body labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (blockSlot i (skipn (S offset) inner)) k = Some s ->
  nth_error (flattenCode code) (pathOffset code (b :: labels) offset + k) = Some s.
Proof.
  intros code b bt body labels inner offset i k s Hb Hpath Hi Hs. cbn [pathOffset]. rewrite Hb.
  replace (codeOffset code b + S (pathOffset body labels offset) + k)
    with (codeOffset code b + S (pathOffset body labels offset + k)) by lia.
  apply (flatten_within_instruction _ _ _ _ _ Hb). rewrite flatten_block_inner.
  pose proof (block_path_slot _ _ _ _ _ _ _ Hpath Hi Hs) as Hinner.
  rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hinner; discriminate). exact Hinner.
Qed.

Lemma flatten_block_end : forall code b bt body labels inner,
  nth_error code b = Some (BLOCK bt body) -> pathCode body labels = Some inner ->
  nth_error (flattenCode code) (pathOffset code (b :: labels) (length inner)) = Some (flatInstruction None None).
Proof.
  intros code b bt body labels inner Hb Hpath. cbn [pathOffset]. rewrite Hb.
  apply (flatten_within_instruction _ _ _ _ _ Hb). rewrite flatten_block_inner. exact (block_path_end _ _ _ Hpath).
Qed.

Lemma path_offset_shift : forall labels code inner offset,
  pathCode code labels = Some inner -> pathOffset code labels offset = pathOffset code labels 0 + codeOffset inner offset.
Proof.
  induction labels as [ | b rest IH]; intros code inner offset Hpath.
  - injection Hpath as <-. reflexivity.
  - cbn [pathCode pathOffset] in *. destruct (nth_error code b) as [j | ]; [ | discriminate].
    destruct j; try discriminate. rewrite (IH _ _ offset Hpath), (IH _ _ 0 Hpath). cbn. lia.
Qed.
