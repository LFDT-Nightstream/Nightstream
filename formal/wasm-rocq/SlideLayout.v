(* Where the slide encoder's branch drops sit: aligned with the flattened layout, a
   br_if's slot holds the drop that the decoder's static heights give, and the
   program ROM serves it at the br_if's pc. *)
From Stdlib Require Import List ZArith Arith Bool Lia.
Require Import Wasm2 MemoryTrace MemoryCatalog CallerSpec CodeLayout Fragment ProgramSpec ProgramEncoding
  BlockEncoding ControlDecode.
Import ListNotations.

Definition nextHeight (callees : CalleeEnvironment) (h : option nat) (j : instr) : option nat :=
  match h with Some h => sourceHeight callees j h | None => None end.

Definition heightsFrom (callees : CalleeEnvironment) (code : list instr) (h : option nat) : option nat :=
  match h with Some h => sourceHeights callees code h | None => None end.

(* A block body's drops, as instructionDrops lays them out. *)
Definition bodyDrops (callees : CalleeEnvironment) (labels : list nat) : list instr -> option nat -> list (option nat) :=
  fix bodyDrops (code : list instr) (h : option nat) : list (option nat) :=
    match code with
    | [] => []
    | j :: rest => instructionDrops callees labels h j ++ bodyDrops rest (nextHeight callees h j)
    end.

(* The drop of a taken br_if at operand height `height` to label `l`. *)
Definition branchDrop (labels : list nat) (height l : nat) : option nat :=
  match height with
  | S below => if (l <? length labels)%nat then Some (below + fold_right Nat.add 0 (firstn l labels))%nat else None
  | O => None
  end.

Lemma code_drops_body : forall callees code h, codeDrops callees code h = bodyDrops callees [] code h.
Proof. reflexivity. Qed.

Lemma body_drops_cons : forall callees labels j rest h,
  bodyDrops callees labels (j :: rest) h =
    instructionDrops callees labels h j ++ bodyDrops callees labels rest (nextHeight callees h j).
Proof. reflexivity. Qed.

Lemma instruction_drops_block : forall callees labels height bt body,
  instructionDrops callees labels height (BLOCK bt body) =
    None :: bodyDrops callees (match height with Some h => h | None => 0 end :: labels)%nat body
      (option_map (fun _ => 0%nat) height) ++ [None].
Proof. reflexivity. Qed.

Lemma instruction_drops_br_if : forall callees labels h l,
  instructionDrops callees labels (Some h) (BR_IF l) = [branchDrop labels h (N.to_nat (l :> N))].
Proof. intros callees labels [ | h] l; reflexivity. Qed.

Fixpoint instruction_drops_length (callees : CalleeEnvironment) (i : instr) {struct i} :
  forall labels h, length (instructionDrops callees labels h i) = instructionSize i.
Proof.
  destruct i; intros labels h; try reflexivity.
  rewrite instruction_drops_block, block_instruction_size. cbn [length]. rewrite length_app. cbn [length].
  match goal with |- context [codeSize ?body] =>
    enough (Hbody : forall labels' h', length (bodyDrops callees labels' body h') = codeSize body)
      by (rewrite Hbody; lia);
    clear labels h; generalize body end.
  refine (fix go (code : list instr) : forall labels' h',
    length (bodyDrops callees labels' code h') = codeSize code := _).
  destruct code as [ | j rest]; intros labels' h'; [reflexivity | ].
  rewrite body_drops_cons, length_app, code_size_cons, (instruction_drops_length callees j), go. reflexivity.
Qed.

Lemma body_drops_length : forall callees labels code h, length (bodyDrops callees labels code h) = codeSize code.
Proof.
  induction code as [ | j rest IH]; intros h; [reflexivity | ].
  rewrite body_drops_cons, length_app, code_size_cons, instruction_drops_length, IH. reflexivity.
Qed.

Lemma body_drops_at : forall callees labels code h offset i k,
  nth_error code offset = Some i -> (k < instructionSize i)%nat ->
  nth_error (bodyDrops callees labels code h) (codeOffset code offset + k) =
    nth_error (instructionDrops callees labels (heightsFrom callees (firstn offset code) h) i) k.
Proof.
  intros callees labels code. induction code as [ | a rest IH]; intros h [ | offset] i k Hi Hk; try discriminate.
  - injection Hi as <-. rewrite body_drops_cons. unfold codeOffset. cbn [firstn codeSize flattenCode flat_map length].
    rewrite nth_error_app1 by (rewrite instruction_drops_length; exact Hk).
    destruct h; reflexivity.
  - cbn [nth_error] in Hi. rewrite body_drops_cons.
    replace (codeOffset (a :: rest) (S offset) + k)%nat with
      (length (instructionDrops callees labels h a) + (codeOffset rest offset + k))%nat
      by (rewrite instruction_drops_length; unfold codeOffset; cbn [firstn]; rewrite code_size_cons; lia).
    rewrite nth_error_app2, Nat.add_comm, Nat.add_sub by lia.
    rewrite (IH (nextHeight callees h a) offset i k Hi Hk).
    destruct h as [h | ]; [ | reflexivity]. cbn [firstn heightsFrom sourceHeights nextHeight].
    destruct (sourceHeight callees a h); reflexivity.
Qed.

(* At a br_if on a code path, the drop of its taken edge. `labels0` holds the
   heights below the blocks around `code`, innermost first. *)
Lemma drops_at_path : forall callees labels code inner offset l hs top labels0,
  pathCode code labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  pathHeights callees code labels offset = Some (hs, top) ->
  nth_error (bodyDrops callees labels0 code (Some 0%nat)) (pathOffset code labels offset) =
    Some (branchDrop (rev hs ++ labels0) top (N.to_nat (l :> N))).
Proof.
  intros callees labels. induction labels as [ | b rest IH];
    intros code inner offset l hs top labels0 Hpath Hi Hheights.
  - injection Hpath as <-. cbn [pathHeights] in Hheights.
    destruct (sourceHeights callees (firstn offset code) 0) as [h | ] eqn:Hh; [ | discriminate].
    injection Hheights as <- <-. cbn [pathOffset].
    rewrite <- (Nat.add_0_r (codeOffset code offset)), (body_drops_at _ _ _ _ _ _ 0 Hi ltac:(cbn; lia)).
    cbn [heightsFrom]. rewrite Hh, instruction_drops_br_if. reflexivity.
  - cbn [pathCode] in Hpath. cbn [pathHeights] in Hheights. cbn [pathOffset].
    destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with Hb : nth_error code b = Some (BLOCK _ ?inside) |- _ => rename inside into body end.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs' top'] | ] eqn:Hinner; [ | discriminate].
    injection Hheights as <- <-.
    pose proof (IH body inner offset l hs' top' (below :: labels0) Hpath Hi Hinner) as Hdrop.
    assert (Hp : (pathOffset body rest offset < codeSize body)%nat).
    { rewrite <- (body_drops_length callees (below :: labels0) body (Some 0%nat)).
      apply nth_error_Some. rewrite Hdrop. discriminate. }
    rewrite (body_drops_at _ _ _ _ _ _ (S (pathOffset body rest offset)) Hb
      ltac:(rewrite block_instruction_size; lia)).
    cbn [heightsFrom]. rewrite Hbelow, instruction_drops_block. cbn [nth_error option_map].
    rewrite nth_error_app1 by (rewrite body_drops_length; exact Hp).
    rewrite Hdrop. cbn [rev]. rewrite <- app_assoc. reflexivity.
Qed.

(* The ROM's slide at a br_if: the fall-through edge drops nothing, the taken
   edge drops what the static heights give, and neither keeps an operand. *)
Lemma encoded_branch_slide : forall program memory index f labels inner offset l hs top choice,
  programLoaded program memory -> functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset = Some (hs, top) ->
  let pc := guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) in
  memory frameSlideRemainingId [pc; choice] = (if (choice =? 0)%Z || (choice =? 1)%Z then Some 0%Z else None) /\
  memory frameSlideDistanceId [pc; choice] =
    (if (choice =? 0)%Z then Some 0%Z
     else if (choice =? 1)%Z then option_map Z.of_nat (branchDrop (rev hs) top (N.to_nat (l :> N))) else None).
Proof.
  intros program memory index f labels inner offset l hs top choice Hloaded Hf Hpath Hi Hheights pc.
  destruct (path_slot_head (guestBody f) labels (BR_IF l) (skipn (S offset) inner)) as [more Hhead].
  pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
  rewrite Nat.add_0_r in Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    unfold codeSize. lia. }
  pose proof (drops_at_path _ _ _ _ _ _ _ _ [] Hpath Hi Hheights) as Hdrop.
  rewrite app_nil_r, <- code_drops_body in Hdrop.
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ _ choice Hf Hbound), Hslot; cbn [Nat.eqb frameSlideRemainingId
      frameSlideDistanceId].
  - reflexivity.
  - rewrite Hdrop. destruct (choice =? 0)%Z; [reflexivity | ].
    destruct (choice =? 1)%Z; [ | reflexivity]. destruct (branchDrop _ _ _); reflexivity.
Qed.
