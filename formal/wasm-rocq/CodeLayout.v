(* Flatten blocks into instruction addresses, recursively: a BLOCK, its body, then
   its END, as the Rust preprocessing numbers operators. A branch inside a block
   body stores its distance to the instruction after its target block's END. A
   block is laid out knowing, for each block around it, the distance from the
   instruction after its own END to the instruction after that block's END. *)
From Stdlib Require Import List Arith NArith Lia.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StackSimulation.
Import ListNotations.

Record FlatInstruction := flatInstruction {
  flatOpcode : option instr;
  flatBranchDistance : option nat
}.

(* A branch tail instructions before its block's END, to label l: the enclosing
   blocks' distances, `outer`, continue past this block's. *)
Definition branchDistance (outer : list nat) (l : labelidx) (tail : nat) : option nat :=
  Datatypes.option_map (Nat.add (S (S tail))) (nth_error (0 :: outer) (N.to_nat (l :> N))).

Fixpoint flattenInstruction (outer : list nat) (i : instr) : list FlatInstruction :=
  match i with
  | BLOCK _ body =>
      flatInstruction (Some i) None ::
        (fix blockCode (code : list instr) : list FlatInstruction :=
           match code with
           | [] => []
           | j :: rest =>
               let tail := blockCode rest in
               match j with
               | BR_IF l => flatInstruction (Some j) (branchDistance outer l (length tail)) :: tail
               | _ => flattenInstruction (map (Nat.add (S (length tail))) (0 :: outer)) j ++ tail
               end
           end) body ++ [flatInstruction None None]
  | _ => [flatInstruction (Some i) None]
  end.

(* A block body, as flattenInstruction lays it out. *)
Definition blockCode (outer : list nat) : list instr -> list FlatInstruction :=
  fix blockCode (code : list instr) : list FlatInstruction :=
    match code with
    | [] => []
    | j :: rest =>
        let tail := blockCode rest in
        match j with
        | BR_IF l => flatInstruction (Some j) (branchDistance outer l (length tail)) :: tail
        | _ => flattenInstruction (map (Nat.add (S (length tail))) (0 :: outer)) j ++ tail
        end
    end.

(* A function body has no enclosing block. *)
Definition flattenCode (code : list instr) := flat_map (flattenInstruction []) code.
Definition instructionSize (i : instr) := length (flattenInstruction [] i).
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

Lemma flatten_block : forall outer bt body,
  flattenInstruction outer (BLOCK bt body) =
    flatInstruction (Some (BLOCK bt body)) None :: blockCode outer body ++ [flatInstruction None None].
Proof. reflexivity. Qed.

Lemma block_code_unfold : forall outer j rest, blockCode outer (j :: rest) =
  match j with
  | BR_IF l => flatInstruction (Some j) (branchDistance outer l (length (blockCode outer rest))) :: blockCode outer rest
  | _ => flattenInstruction (map (Nat.add (S (length (blockCode outer rest)))) (0 :: outer)) j ++ blockCode outer rest
  end.
Proof. reflexivity. Qed.

(* Distances never change how many addresses an instruction takes. *)
Fixpoint flatten_length_any (i : instr) {struct i} :
  forall o o', length (flattenInstruction o i) = length (flattenInstruction o' i).
Proof.
  destruct i; intros outer outer'; try reflexivity.
  rewrite !flatten_block. cbn [length]. rewrite !length_app. f_equal. f_equal.
  match goal with |- context [blockCode _ ?body] => revert outer outer'; generalize body end.
  refine (fix go (code : list instr) : forall o o', length (blockCode o code) = length (blockCode o' code) := _).
  destruct code as [ | j rest]; intros o o'; [reflexivity | ].
  pose proof (flatten_length_any j) as Hj. pose proof (go rest o o') as Hrest.
  rewrite !block_code_unfold. destruct j; cbn [length]; rewrite ?length_app, ?Hrest; try reflexivity.
  all: f_equal; apply Hj.
Qed.

Lemma flatten_length : forall outer i, length (flattenInstruction outer i) = instructionSize i.
Proof. intros. apply flatten_length_any. Qed.

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
  nth_error code offset = Some i -> nth_error (flattenInstruction [] i) inner = Some slot ->
  nth_error (flattenCode code) (codeOffset code offset + inner) = Some slot.
Proof.
  intros code offset i inner slot Hsource Hslot.
  rewrite <- (firstn_skipn offset code) at 1. unfold flattenCode at 1. rewrite flat_map_app.
  rewrite nth_error_app2 by (unfold codeOffset, codeSize, flattenCode; lia).
  replace (codeOffset code offset + inner - length (flat_map (flattenInstruction []) (firstn offset code)))
    with inner by (unfold codeOffset, codeSize, flattenCode; lia).
  rewrite (skipn_at_instruction code offset i Hsource). cbn [flat_map].
  rewrite nth_error_app1; [exact Hslot | apply nth_error_Some; rewrite Hslot; discriminate].
Qed.

Lemma code_size_split : forall code offset, codeSize code = codeOffset code offset + codeSize (skipn offset code).
Proof. intros. unfold codeOffset. rewrite <- code_size_app, firstn_skipn. reflexivity. Qed.

(* One instruction of a block body: a branch carries its distance to its target;
   a nested block learns the distances around it. *)
Definition blockSlot (outer : list nat) (i : instr) (rest : list instr) : list FlatInstruction :=
  match i with
  | BR_IF l => [flatInstruction (Some i) (branchDistance outer l (codeSize rest))]
  | _ => flattenInstruction (map (Nat.add (S (codeSize rest))) (0 :: outer)) i
  end.

Lemma block_slot_length : forall outer i rest, length (blockSlot outer i rest) = instructionSize i.
Proof. intros outer i rest. destruct i; try reflexivity; apply flatten_length. Qed.

Lemma block_slot_head : forall outer i rest, exists more,
  blockSlot outer i rest = flatInstruction (Some i)
    (match i with BR_IF l => branchDistance outer l (codeSize rest) | _ => None end) :: more.
Proof. intros outer i rest. destruct i; eexists; reflexivity. Qed.

Lemma block_code_length : forall outer code, length (blockCode outer code) = codeSize code.
Proof.
  intros outer. induction code as [ | j rest IH]; [reflexivity | ].
  rewrite block_code_unfold, code_size_cons, <- IH.
  destruct j; cbn [length]; rewrite ?length_app, ?flatten_length; reflexivity.
Qed.

Lemma block_code_cons : forall outer j rest,
  blockCode outer (j :: rest) = blockSlot outer j rest ++ blockCode outer rest.
Proof. intros outer j rest. rewrite block_code_unfold, block_code_length. destruct j; reflexivity. Qed.

Lemma block_instruction_size : forall bt body, instructionSize (BLOCK bt body) = (codeSize body + 2)%nat.
Proof.
  intros. unfold instructionSize. rewrite flatten_block. cbn [length]. rewrite length_app, block_code_length. cbn. lia.
Qed.

Lemma block_code_at : forall outer code offset i k s,
  nth_error code offset = Some i -> nth_error (blockSlot outer i (skipn (S offset) code)) k = Some s ->
  nth_error (blockCode outer code) (codeOffset code offset + k) = Some s.
Proof.
  intros outer. induction code as [ | j rest IH]; intros [ | offset] i k s Hi Hs; try discriminate.
  - injection Hi as <-. cbn [skipn] in Hs. rewrite block_code_cons. change (codeOffset (j :: rest) 0) with 0%nat.
    cbn [Nat.add].
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hs; discriminate). exact Hs.
  - rewrite block_code_cons. unfold codeOffset. cbn [firstn]. rewrite code_size_cons.
    rewrite nth_error_app2 by (rewrite block_slot_length; lia). rewrite block_slot_length.
    replace (instructionSize j + codeSize (firstn offset rest) + k - instructionSize j)
      with (codeOffset rest offset + k) by (unfold codeOffset; lia).
    exact (IH _ _ _ _ Hi ltac:(cbn [skipn] in Hs |- *; exact Hs)).
Qed.

Lemma flatten_block_inner : forall outer bt body inner,
  nth_error (flattenInstruction outer (BLOCK bt body)) (S inner) =
    nth_error (blockCode outer body ++ [flatInstruction None None]) inner.
Proof. reflexivity. Qed.

(* The distances a body at a path is laid out with. *)
Fixpoint bodyOuter (outer : list nat) (body : list instr) (labels : list nat) : list nat :=
  match labels with
  | [] => outer
  | b :: rest => match nth_error body b with
      | Some (BLOCK _ nested) => bodyOuter (map (Nat.add (S (codeSize (skipn (S b) body)))) (0 :: outer)) nested rest
      | _ => outer
      end
  end.

(* Slots of a body at any depth of nesting, within the enclosing block's layout. *)
Lemma block_path_slot : forall labels body outer inner offset i k s,
  pathCode body labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (blockSlot (bodyOuter outer body labels) i (skipn (S offset) inner)) k = Some s ->
  nth_error (blockCode outer body) (pathOffset body labels offset + k) = Some s.
Proof.
  induction labels as [ | b rest IH]; intros body outer inner offset i k s Hpath Hi Hs.
  - injection Hpath as <-. exact (block_code_at _ _ _ _ _ _ Hi Hs).
  - cbn [pathCode pathOffset bodyOuter] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with |- context [codeOffset body b + S (pathOffset ?nested rest offset) + k] =>
      replace (codeOffset body b + S (pathOffset nested rest offset) + k)
        with (codeOffset body b + S (pathOffset nested rest offset + k)) by lia end.
    apply (block_code_at _ _ _ _ _ _ Hb). cbn [blockSlot]. rewrite flatten_block_inner.
    pose proof (IH _ _ _ _ _ _ _ Hpath Hi Hs) as Hinner.
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hinner; discriminate). exact Hinner.
Qed.

Lemma block_path_end : forall labels body outer inner,
  pathCode body labels = Some inner ->
  nth_error (blockCode outer body ++ [flatInstruction None None]) (pathOffset body labels (length inner)) =
    Some (flatInstruction None None).
Proof.
  induction labels as [ | b rest IH]; intros body outer inner Hpath.
  - injection Hpath as <-. cbn [pathOffset]. rewrite code_offset_end, <- (block_code_length outer).
    rewrite nth_error_app2, Nat.sub_diag by lia. reflexivity.
  - cbn [pathCode pathOffset] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with |- context [codeOffset body b + S (pathOffset ?nested rest (length inner))] =>
      assert (Hslot : nth_error (blockCode outer body) (codeOffset body b + S (pathOffset nested rest (length inner))) =
          Some (flatInstruction None None)) end.
    { apply (block_code_at _ _ _ _ _ _ Hb). cbn [blockSlot]. rewrite flatten_block_inner. exact (IH _ _ _ Hpath). }
    rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hslot; discriminate). exact Hslot.
Qed.

(* The distances the innermost body at a function's code path is laid out with. *)
Definition pathOuter (code : list instr) (labels : list nat) : list nat :=
  match labels with
  | [] => []
  | b :: rest => match nth_error code b with Some (BLOCK _ body) => bodyOuter [] body rest | _ => [] end
  end.

(* The same slots within a function body, below a top-level BLOCK. *)
Lemma flatten_in_block : forall code b bt body labels inner offset i k s,
  nth_error code b = Some (BLOCK bt body) -> pathCode body labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (blockSlot (bodyOuter [] body labels) i (skipn (S offset) inner)) k = Some s ->
  nth_error (flattenCode code) (pathOffset code (b :: labels) offset + k) = Some s.
Proof.
  intros code b bt body labels inner offset i k s Hb Hpath Hi Hs. cbn [pathOffset]. rewrite Hb.
  replace (codeOffset code b + S (pathOffset body labels offset) + k)
    with (codeOffset code b + S (pathOffset body labels offset + k)) by lia.
  apply (flatten_within_instruction _ _ _ _ _ Hb). rewrite flatten_block_inner.
  pose proof (block_path_slot _ _ _ _ _ _ _ _ Hpath Hi Hs) as Hinner.
  rewrite nth_error_app1 by (apply nth_error_Some; rewrite Hinner; discriminate). exact Hinner.
Qed.

Lemma flatten_block_end : forall code b bt body labels inner,
  nth_error code b = Some (BLOCK bt body) -> pathCode body labels = Some inner ->
  nth_error (flattenCode code) (pathOffset code (b :: labels) (length inner)) = Some (flatInstruction None None).
Proof.
  intros code b bt body labels inner Hb Hpath. cbn [pathOffset]. rewrite Hb.
  apply (flatten_within_instruction _ _ _ _ _ Hb). rewrite flatten_block_inner. exact (block_path_end _ _ _ _ Hpath).
Qed.

Lemma path_offset_shift : forall labels code inner offset,
  pathCode code labels = Some inner -> pathOffset code labels offset = pathOffset code labels 0 + codeOffset inner offset.
Proof.
  induction labels as [ | b rest IH]; intros code inner offset Hpath.
  - injection Hpath as <-. reflexivity.
  - cbn [pathCode pathOffset] in *. destruct (nth_error code b) as [j | ]; [ | discriminate].
    destruct j; try discriminate. rewrite (IH _ _ offset Hpath), (IH _ _ 0 Hpath). cbn. lia.
Qed.

(* Where a branch from a body at a path can land, innermost block first, relative
   to the body's start: after each enclosing block's END, then after the body's
   own block's END, then the distances around it. *)
Fixpoint bodyExits (outer : list nat) (body : list instr) (labels : list nat) : list nat :=
  match labels with
  | [] => map (Nat.add (S (codeSize body))) (0 :: outer)
  | b :: rest => match nth_error body b with
      | Some (BLOCK _ nested) =>
          map (Nat.add (S (codeOffset body b))) (bodyExits (map (Nat.add (S (codeSize (skipn (S b) body)))) (0 :: outer)) nested rest)
      | _ => []
      end
  end.

Lemma block_branch_distance : forall labels body outer inner offset l t,
  pathCode body labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  nth_error (bodyExits outer body labels) (N.to_nat (l :> N)) = Some t ->
  exists d, branchDistance (bodyOuter outer body labels) l (codeSize (skipn (S offset) inner)) = Some d /\
    pathOffset body labels offset + d = t.
Proof.
  induction labels as [ | b rest IH]; intros body outer inner offset l t Hpath Hi Ht.
  - injection Hpath as <-. cbn [bodyExits bodyOuter pathOffset] in *. unfold branchDistance.
    rewrite nth_error_map in Ht.
    destruct (nth_error (0 :: outer) (N.to_nat (l :> N))) as [e | ]; cbn [Datatypes.option_map] in Ht |- *; [ | discriminate].
    injection Ht as <-. eexists. split; [reflexivity | ].
    pose proof (code_size_split body (S offset)) as Hsplit. rewrite (code_offset_next _ _ _ Hi) in Hsplit.
    change (instructionSize (BR_IF l)) with 1 in Hsplit. lia.
  - cbn [pathCode bodyExits bodyOuter pathOffset] in *.
    destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate]. destruct j; try discriminate.
    rewrite nth_error_map in Ht.
    match goal with H : Datatypes.option_map _ (nth_error ?exits _) = _ |- _ =>
      destruct (nth_error exits (N.to_nat (l :> N))) as [t2 | ] eqn:Ht2; cbn in H; [ | discriminate];
      injection H as <- end.
    destruct (IH _ _ _ _ _ _ Hpath Hi Ht2) as [d [Hd Hsum]]. exists d. split; [exact Hd | lia].
Qed.

Lemma body_exits_beyond : forall labels body outer inner k,
  pathCode body labels = Some inner ->
  nth_error (bodyExits outer body labels) (length labels + k) =
    Datatypes.option_map (Nat.add (S (codeSize body))) (nth_error (0 :: outer) k).
Proof.
  induction labels as [ | b rest IH]; intros body outer inner k Hpath.
  - cbn [bodyExits length Nat.add]. apply nth_error_map.
  - cbn [pathCode bodyExits length] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate. rewrite nth_error_map.
    replace (S (length rest) + k)%nat with (length rest + S k)%nat by lia.
    rewrite (IH _ _ _ _ Hpath). cbn [nth_error]. rewrite nth_error_map.
    pose proof (code_size_split body b) as Hsplit. rewrite (skipn_at_instruction _ _ _ Hb), code_size_cons,
      block_instruction_size in Hsplit.
    destruct (nth_error (0 :: outer) k) as [e | ]; cbn [Datatypes.option_map]; [f_equal; lia | reflexivity].
Qed.

Lemma body_exits_prefix : forall pre b post body outer inner,
  pathCode body (pre ++ b :: post) = Some inner ->
  nth_error (bodyExits outer body (pre ++ b :: post)) (length post) = Some (pathOffset body pre (S b)).
Proof.
  induction pre as [ | b0 pre IH]; intros b post body outer inner Hpath; cbn [app] in *.
  - cbn [pathCode bodyExits] in *. destruct (nth_error body b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate. rewrite nth_error_map.
    rewrite <- (Nat.add_0_r (length post)), (body_exits_beyond _ _ _ _ 0 Hpath). cbn [nth_error Datatypes.option_map].
    cbn [pathOffset]. rewrite (code_offset_next _ _ _ Hb), block_instruction_size. f_equal. lia.
  - cbn [pathCode bodyExits pathOffset] in *. destruct (nth_error body b0) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate. rewrite nth_error_map, (IH _ _ _ _ _ Hpath). cbn [Datatypes.option_map]. f_equal. lia.
Qed.

(* A branch at a function's code path, to the block at b after pre: its distance
   lands just after that block. *)
Lemma flatten_branch_target : forall code labels inner offset l pre b post,
  pathCode code labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  labels = pre ++ b :: post -> N.to_nat (l :> N) = length post ->
  exists d, branchDistance (pathOuter code labels) l (codeSize (skipn (S offset) inner)) = Some d /\
    pathOffset code labels offset + d = pathOffset code pre (S b).
Proof.
  intros code labels inner offset l pre b post Hpath Hi -> Hl.
  destruct pre as [ | b0 pre]; cbn [app pathCode pathOuter pathOffset] in *.
  - destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate]. destruct j; try discriminate.
    pose proof (body_exits_beyond _ _ [] _ 0 Hpath) as Ht. rewrite Nat.add_0_r, <- Hl in Ht. cbn in Ht.
    destruct (block_branch_distance _ _ _ _ _ _ _ Hpath Hi Ht) as [d [Hd Hsum]].
    exists d. split; [exact Hd | ]. rewrite (code_offset_next _ _ _ Hb), block_instruction_size. lia.
  - destruct (nth_error code b0) as [j | ] eqn:Hb; [ | discriminate]. destruct j; try discriminate.
    pose proof (body_exits_prefix _ _ _ _ [] _ Hpath) as Ht. rewrite <- Hl in Ht.
    destruct (block_branch_distance _ _ _ _ _ _ _ Hpath Hi Ht) as [d [Hd Hsum]].
    exists d. split; [exact Hd | lia].
Qed.
