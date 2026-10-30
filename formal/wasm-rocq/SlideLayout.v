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

(* A block body's slides, as instructionDrops lays them out. *)
Definition bodyDrops (callees : CalleeEnvironment) (heights arities : list nat)
    : list instr -> option nat -> list (option (nat * nat)) :=
  fix bodyDrops (code : list instr) (h : option nat) : list (option (nat * nat)) :=
    match code with
    | [] => []
    | j :: rest => instructionDrops callees heights arities h j ++ bodyDrops rest (nextHeight callees h j)
    end.

Lemma code_drops_body : forall callees results code h,
  codeDrops callees results code h = bodyDrops callees [0%nat] [results] code h.
Proof. reflexivity. Qed.

Lemma body_drops_cons : forall callees heights arities j rest h,
  bodyDrops callees heights arities (j :: rest) h =
    instructionDrops callees heights arities h j ++ bodyDrops callees heights arities rest (nextHeight callees h j).
Proof. reflexivity. Qed.

Lemma instruction_drops_block : forall callees heights arities height bt body,
  instructionDrops callees heights arities height (BLOCK bt body) =
    None :: bodyDrops callees (match height with Some h => h | None => 0 end :: heights)%nat
      (blockResults bt :: arities) body (option_map (fun _ => 0%nat) height) ++ [None].
Proof. reflexivity. Qed.

Fixpoint instruction_drops_length (callees : CalleeEnvironment) (i : instr) {struct i} :
  forall heights arities h, length (instructionDrops callees heights arities h i) = instructionSize i.
Proof.
  destruct i; intros heights arities h; try reflexivity.
  rewrite instruction_drops_block, block_instruction_size. cbn [length]. rewrite length_app. cbn [length].
  match goal with |- context [codeSize ?body] =>
    enough (Hbody : forall heights' arities' h', length (bodyDrops callees heights' arities' body h') = codeSize body)
      by (rewrite Hbody; lia);
    clear heights arities h; generalize body end.
  refine (fix go (code : list instr) : forall heights' arities' h',
    length (bodyDrops callees heights' arities' code h') = codeSize code := _).
  destruct code as [ | j rest]; intros heights' arities' h'; [reflexivity | ].
  rewrite body_drops_cons, length_app, code_size_cons, (instruction_drops_length callees j), go. reflexivity.
Qed.

Lemma body_drops_length : forall callees heights arities code h,
  length (bodyDrops callees heights arities code h) = codeSize code.
Proof.
  induction code as [ | j rest IH]; intros h; [reflexivity | ].
  rewrite body_drops_cons, length_app, code_size_cons, instruction_drops_length, IH. reflexivity.
Qed.

Lemma body_drops_at : forall callees heights arities code h offset i k,
  nth_error code offset = Some i -> (k < instructionSize i)%nat ->
  nth_error (bodyDrops callees heights arities code h) (codeOffset code offset + k) =
    nth_error (instructionDrops callees heights arities (heightsFrom callees (firstn offset code) h) i) k.
Proof.
  intros callees heights arities code. induction code as [ | a rest IH]; intros h [ | offset] i k Hi Hk;
    try discriminate.
  - injection Hi as <-. rewrite body_drops_cons. unfold codeOffset. cbn [firstn codeSize flattenCode flat_map length].
    rewrite nth_error_app1 by (rewrite instruction_drops_length; exact Hk).
    destruct h; reflexivity.
  - cbn [nth_error] in Hi. rewrite body_drops_cons.
    replace (codeOffset (a :: rest) (S offset) + k)%nat with
      (length (instructionDrops callees heights arities h a) + (codeOffset rest offset + k))%nat
      by (rewrite instruction_drops_length; unfold codeOffset; cbn [firstn]; rewrite code_size_cons; lia).
    rewrite nth_error_app2, Nat.add_comm, Nat.add_sub by lia.
    rewrite (IH (nextHeight callees h a) offset i k Hi Hk).
    destruct h as [h | ]; [ | reflexivity]. cbn [firstn heightsFrom sourceHeights nextHeight].
    destruct (sourceHeight callees a h); reflexivity.
Qed.

(* At an instruction other than a block on a code path, its first slot's slide at
   the heights and arities the code path gives. `heights0` and `arities0` describe
   the labels around `code`, innermost first. *)
Lemma drops_at_path : forall callees labels code inner offset i hs top heights0 arities0,
  pathCode code labels = Some inner -> nth_error inner offset = Some i -> (forall bt body, i <> BLOCK bt body) ->
  pathHeights callees code labels offset = Some (hs, top) ->
  nth_error (bodyDrops callees heights0 arities0 code (Some 0%nat)) (pathOffset code labels offset) =
    nth_error (instructionDrops callees (rev hs ++ heights0) (rev (pathArities code labels) ++ arities0) (Some top) i) 0.
Proof.
  intros callees labels. induction labels as [ | b rest IH];
    intros code inner offset i hs top heights0 arities0 Hpath Hi Hflat Hheights.
  - injection Hpath as <-. cbn [pathHeights] in Hheights.
    destruct (sourceHeights callees (firstn offset code) 0) as [h | ] eqn:Hh; [ | discriminate].
    injection Hheights as <- <-. cbn [pathOffset].
    assert (Hsize : instructionSize i = 1%nat).
    { destruct i; try reflexivity. exfalso. eapply Hflat. reflexivity. }
    rewrite <- (Nat.add_0_r (codeOffset code offset)), (body_drops_at _ _ _ _ _ _ _ 0 Hi ltac:(lia)).
    cbn [heightsFrom]. rewrite Hh. reflexivity.
  - cbn [pathCode] in Hpath. cbn [pathHeights] in Hheights. cbn [pathOffset].
    destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate.
    match goal with Hb : nth_error code b = Some (BLOCK _ ?inside) |- _ => rename inside into body end.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs' top'] | ] eqn:Hinner; [ | discriminate].
    injection Hheights as <- <-.
    match goal with Hb : nth_error code b = Some (BLOCK ?blocktype _) |- _ => rename blocktype into bt end.
    pose proof (IH body inner offset i hs' top' (below :: heights0) (blockResults bt :: arities0) Hpath Hi Hflat Hinner)
      as Hdrop.
    assert (Hslot : nth_error (instructionDrops callees (rev hs' ++ below :: heights0)
        (rev (pathArities body rest) ++ blockResults bt :: arities0) (Some top') i) 0 <> None).
    { assert (Hsize : instructionSize i = 1%nat).
      { destruct i; try reflexivity. exfalso. eapply Hflat. reflexivity. }
      apply nth_error_Some. rewrite instruction_drops_length. lia. }
    assert (Hp : (pathOffset body rest offset < codeSize body)%nat).
    { rewrite <- (body_drops_length callees (below :: heights0) (blockResults bt :: arities0) body (Some 0%nat)).
      apply nth_error_Some. rewrite Hdrop. exact Hslot. }
    rewrite (body_drops_at _ _ _ _ _ _ _ (S (pathOffset body rest offset)) Hb
      ltac:(rewrite block_instruction_size; lia)).
    cbn [heightsFrom]. rewrite Hbelow, instruction_drops_block. cbn [nth_error option_map].
    rewrite nth_error_app1 by (rewrite body_drops_length; exact Hp).
    rewrite Hdrop. cbn [pathArities]. rewrite Hb. cbn [rev]. rewrite <- !app_assoc. reflexivity.
Qed.

Lemma sum_rev : forall hs top, (top + fold_right Nat.add 0 (rev hs))%nat = totalHeight (hs, top).
Proof.
  intros hs top. unfold totalHeight. cbn [fst snd].
  assert (Hshift : forall a xs, fold_right Nat.add a xs = (a + fold_right Nat.add 0 xs)%nat).
  { intros a xs. induction xs as [ | x xs IH]; cbn [fold_right]; lia. }
  induction hs as [ | h hs IH]; [cbn; lia | ].
  cbn [rev fold_right]. rewrite fold_right_app, Hshift. cbn [fold_right]. rewrite Hshift in IH. lia.
Qed.

Lemma path_arities_length : forall callees labels code offset hs top,
  pathHeights callees code labels offset = Some (hs, top) -> length (pathArities code labels) = length hs.
Proof.
  intros callees. induction labels as [ | b rest IH]; intros code offset hs top H; cbn [pathHeights] in H.
  - destruct (sourceHeights callees (firstn offset code) 0); cbn in H; [ | discriminate]. now injection H as <- _.
  - cbn [pathArities]. destruct (nth_error code b) as [i | ]; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0); [ | discriminate].
    match goal with H : option_map _ (pathHeights callees ?nested rest offset) = _ |- _ =>
      destruct (pathHeights callees nested rest offset) as [[hs' t] | ] eqn:Hin; cbn in H; [ | discriminate];
      injection H as <- _; cbn; f_equal; exact (IH _ _ _ _ Hin) end.
Qed.

(* A slide ROM word, for a slot that may not slide. *)
Definition slideEntry (remaining : bool) (slide : option (nat * nat)) : Z :=
  match slide with Some (kept, dropped) => slideWord remaining kept dropped | None => 0%Z end.

(* The ROM's slide at a br_if: the fall-through edge slides nothing; the taken
   edge keeps its target's arity and drops what the static heights give. *)
Lemma encoded_branch_slide : forall program memory index f labels inner offset l hs top choice,
  programLoaded program memory -> functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset = Some (hs, top) ->
  let pc := guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) in
  let slide := branchSlide (rev hs ++ [0%nat]) (rev (pathArities (guestBody f) labels) ++ [guestResults f]) top
    (N.to_nat (l :> N)) in
  memory frameSlideRemainingId [pc; choice] = Some (if (choice =? 1)%Z then slideEntry true slide else 0%Z) /\
  memory frameSlideDistanceId [pc; choice] = Some (if (choice =? 1)%Z then slideEntry false slide else 0%Z).
Proof.
  intros program memory index f labels inner offset l hs top choice Hloaded Hf Hpath Hi Hheights pc slide.
  destruct (path_slot_head (guestBody f) labels (BR_IF l) (skipn (S offset) inner)) as [more Hhead].
  pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
  rewrite Nat.add_0_r in Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    unfold codeSize. lia. }
  pose proof (drops_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi ltac:(discriminate) Hheights) as Hdrop.
  rewrite <- code_drops_body in Hdrop. cbn [instructionDrops nth_error] in Hdrop.
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ _ choice Hf Hbound); unfold functionSlide; cbv zeta;
    rewrite Hslot, Hdrop; fold slide; cbn [Nat.eqb frameSlideRemainingId frameSlideDistanceId].
  all: destruct (choice =? 0)%Z eqn:H0;
    [apply Z.eqb_eq in H0; subst choice; reflexivity | destruct (choice =? 1)%Z; [ | reflexivity]].
  all: destruct slide as [[kept dropped] | ]; reflexivity.
Qed.

(* The ROM's slide at a return: it keeps the function's results and drops its
   locals and every other operand of its frame, when the return's own block holds
   the results. *)
Lemma encoded_return_slide : forall program memory index f labels inner offset hs top choice,
  programLoaded program memory -> functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some RETURN ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset = Some (hs, top) ->
  let pc := guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) in
  let total := totalHeight (hs, top) in
  let distance := (guestParams f + guestLocals f + (total - guestResults f))%nat in
  let fits := (choice =? 0)%Z && Nat.leb (guestResults f) top in
  memory frameSlideRemainingId [pc; choice] =
    Some (if fits then slideWord true (guestResults f) distance else 0%Z) /\
  memory frameSlideDistanceId [pc; choice] =
    Some (if fits then slideWord false (guestResults f) distance else 0%Z).
Proof.
  intros program memory index f labels inner offset hs top choice Hloaded Hf Hpath Hi Hheights pc total distance fits.
  destruct (path_slot_head (guestBody f) labels RETURN (skipn (S offset) inner)) as [more Hhead].
  pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
  rewrite Nat.add_0_r in Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    unfold codeSize. lia. }
  pose proof (drops_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi ltac:(discriminate) Hheights) as Hdrop.
  rewrite <- code_drops_body in Hdrop. cbn [instructionDrops nth_error] in Hdrop.
  (* The function's label is the outermost one: below it nothing, and it keeps the results. *)
  assert (Hlen : length (pathArities (guestBody f) labels) = length hs) by exact (path_arities_length _ _ _ _ _ _ Hheights).
  assert (Hslide : labelSlide (rev hs ++ [0%nat]) (rev (pathArities (guestBody f) labels) ++ [guestResults f]) top
      (length (rev (pathArities (guestBody f) labels) ++ [guestResults f]) - 1) =
      if Nat.leb (guestResults f) top
      then Some (guestResults f, top - guestResults f + fold_right Nat.add 0 (rev hs))%nat else None).
  { unfold labelSlide. rewrite length_app, length_rev, Hlen. cbn [length]. rewrite Nat.add_sub.
    rewrite nth_error_app2 by (rewrite length_rev, Hlen; lia). rewrite length_rev, Hlen, Nat.sub_diag. cbn [nth_error].
    rewrite firstn_app, length_rev, Nat.sub_diag, <- (length_rev hs), firstn_all. cbn [firstn]. rewrite app_nil_r. reflexivity. }
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ _ choice Hf Hbound); unfold functionSlide; cbv zeta;
    rewrite Hslot, Hdrop, Hslide; unfold fits.
  all: destruct (Nat.leb (guestResults f) top) eqn:Hfits; destruct (choice =? 0)%Z; try reflexivity.
  all: apply Nat.leb_le in Hfits; unfold distance, total; rewrite <- sum_rev; cbn [andb].
  all: replace (top + fold_right Nat.add 0 (rev hs) - guestResults f)%nat
    with (top - guestResults f + fold_right Nat.add 0 (rev hs))%nat by lia; reflexivity.
Qed.
