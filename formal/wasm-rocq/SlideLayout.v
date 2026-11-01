(* Where the slide encoder's slot contexts sit: aligned with the flattened layout,
   an instruction's slot holds the label context that the decoder's static heights
   and the code path's arities give, and the program ROM serves its slides at the
   instruction's pc. *)
From Stdlib Require Import List ZArith Arith Bool Lia.
Require Import Wasm2 MemoryTrace MemoryCatalog CallerSpec CodeLayout Fragment ProgramSpec ProgramEncoding
  BlockEncoding ControlDecode.
Import ListNotations.

Definition nextHeight (callees : CalleeEnvironment) (h : option nat) (j : instr) : option nat :=
  match h with Some h => sourceHeight callees j h | None => None end.

Definition heightsFrom (callees : CalleeEnvironment) (code : list instr) (h : option nat) : option nat :=
  match h with Some h => sourceHeights callees code h | None => None end.

(* A body's slot contexts, as instructionContexts lays them out. *)
Fixpoint bodyContexts (callees : CalleeEnvironment) (heights arities : list nat) (code : list instr) (h : option nat)
    : list (option SlotContext) :=
  match code with
  | [] => []
  | j :: rest => instructionContexts callees heights arities h j ++
      bodyContexts callees heights arities rest (nextHeight callees h j)
  end.

Lemma code_contexts_body : forall callees results code h,
  codeContexts callees results code h = bodyContexts callees [0%nat] [results] code h.
Proof. intros callees results code. induction code as [ | j rest IH]; intros h; [reflexivity | ].
  cbn [bodyContexts]. rewrite <- IH. reflexivity. Qed.

Lemma body_contexts_cons : forall callees heights arities j rest h,
  bodyContexts callees heights arities (j :: rest) h =
    instructionContexts callees heights arities h j ++ bodyContexts callees heights arities rest (nextHeight callees h j).
Proof. reflexivity. Qed.

(* A structured instruction's slots: each arm's body runs inside the instruction's label. *)
Lemma instruction_contexts_arms : forall callees heights arities height i,
  let inner := (labelBase i (match height with Some h => h | None => 0 end) :: heights)%nat in
  let entered := option_map (fun _ => 0%nat) height in
  (forall bt body, i = BLOCK bt body \/ i = LOOP bt body ->
    instructionContexts callees heights arities height i =
      None :: bodyContexts callees inner (labelArity i :: arities) body entered ++ [None]) /\
  (forall bt t e, i = IFELSE bt t e ->
    instructionContexts callees heights arities height i =
      None :: bodyContexts callees inner (labelArity i :: arities) t entered ++
        match e with [] => [] | _ => None :: bodyContexts callees inner (labelArity i :: arities) e entered end ++
        [None]).
Proof.
  intros callees heights arities height i inner entered.
  assert (Hfix : forall hs ars code h, (fix bodyContexts (heights arities : list nat) (code : list instr)
      (h : option nat) : list (option SlotContext) :=
      match code with
      | [] => []
      | j :: rest => instructionContexts callees heights arities h j ++
          bodyContexts heights arities rest (match h with Some h => sourceHeight callees j h | None => None end)
      end) hs ars code h = bodyContexts callees hs ars code h).
  { intros hs ars code. induction code as [ | j rest IH]; intros h; [reflexivity | ].
    cbn [bodyContexts]. rewrite <- IH. reflexivity. }
  split.
  - intros bt body [-> | ->]; cbn [instructionContexts]; cbv zeta; rewrite Hfix; reflexivity.
  - intros bt t e ->. cbn [instructionContexts]. cbv zeta. rewrite !Hfix. reflexivity.
Qed.

Lemma contexts_body_length : forall callees n,
  (forall i heights arities h, (instructionSize i <= n)%nat ->
    length (instructionContexts callees heights arities h i) = instructionSize i) /\
  (forall code heights arities h, (codeSize code <= n)%nat ->
    length (bodyContexts callees heights arities code h) = codeSize code).
Proof.
  intros callees. induction n as [ | n [IHi IHc]].
  - split; [intros i heights arities h H; pose proof (instruction_size_pos i); lia | ].
    intros [ | j rest] heights arities h H; [reflexivity | ]. pose proof (instruction_size_pos j). cbn [codeSize] in H. lia.
  - assert (Hi : forall i heights arities h, (instructionSize i <= S n)%nat ->
        length (instructionContexts callees heights arities h i) = instructionSize i).
    { intros i heights arities h H. destruct (instruction_contexts_arms callees heights arities h i) as [Hblock Hif].
      destruct i; try reflexivity.
      - rewrite (Hblock _ _ (or_introl Logic.eq_refl)). rewrite block_instruction_size in *. cbn [length].
        rewrite length_app, IHc by lia. cbn [length]. lia.
      - rewrite (Hblock _ _ (or_intror Logic.eq_refl)). rewrite loop_instruction_size in *. cbn [length].
        rewrite length_app, IHc by lia. cbn [length]. lia.
      - rewrite (Hif _ _ _ Logic.eq_refl). rewrite if_instruction_size in *. cbn [length].
        rewrite !length_app, IHc by lia.
        match goal with |- context [match ?e with [] => _ | _ :: _ => _ end] => destruct e end;
          cbn [length codeSize] in *; rewrite ?IHc by (cbn [codeSize]; lia); cbn [codeSize]; lia. }
    split; [exact Hi | ].
    induction code as [ | j rest IH]; intros heights arities h H; [reflexivity | ].
    rewrite body_contexts_cons, length_app. cbn [codeSize] in H |- *. pose proof (instruction_size_pos j).
    rewrite Hi, IH by lia. reflexivity.
Qed.

Lemma instruction_contexts_length : forall callees i heights arities h,
  length (instructionContexts callees heights arities h i) = instructionSize i.
Proof. intros. exact (proj1 (contexts_body_length callees (instructionSize i)) i heights arities h (le_n _)). Qed.

Lemma body_contexts_length : forall callees heights arities code h,
  length (bodyContexts callees heights arities code h) = codeSize code.
Proof. intros. exact (proj2 (contexts_body_length callees (codeSize code)) code heights arities h (le_n _)). Qed.

Lemma body_contexts_at : forall callees heights arities code h offset i k,
  nth_error code offset = Some i -> (k < instructionSize i)%nat ->
  nth_error (bodyContexts callees heights arities code h) (codeOffset code offset + k) =
    nth_error (instructionContexts callees heights arities (heightsFrom callees (firstn offset code) h) i) k.
Proof.
  intros callees heights arities code. induction code as [ | a rest IH]; intros h [ | offset] i k Hi Hk;
    try discriminate.
  - injection Hi as <-. rewrite body_contexts_cons. change (codeOffset (a :: rest) 0) with 0%nat.
    rewrite nth_error_app1 by (rewrite instruction_contexts_length; exact Hk).
    destruct h; reflexivity.
  - cbn [nth_error] in Hi. rewrite body_contexts_cons.
    replace (codeOffset (a :: rest) (S offset) + k)%nat with
      (length (instructionContexts callees heights arities h a) + (codeOffset rest offset + k))%nat
      by (rewrite instruction_contexts_length; unfold codeOffset; cbn [firstn codeSize]; lia).
    rewrite nth_error_app2 by lia.
    replace (length (instructionContexts callees heights arities h a) + (codeOffset rest offset + k) -
      length (instructionContexts callees heights arities h a))%nat with (codeOffset rest offset + k)%nat by lia.
    rewrite (IH (nextHeight callees h a) offset i k Hi Hk).
    destruct h as [h | ]; [ | reflexivity]. cbn [firstn heightsFrom sourceHeights nextHeight].
    destruct (sourceHeight callees a h); reflexivity.
Qed.

(* An arm's slot contexts inside its instruction's. *)
Lemma instruction_contexts_arm : forall callees heights arities height i arm body k,
  instrArm i arm = Some body -> (k < codeSize body)%nat ->
  nth_error (instructionContexts callees heights arities height i) (armStart i arm + k) =
    nth_error (bodyContexts callees (labelBase i (match height with Some h => h | None => 0 end) :: heights)%nat
      (labelArity i :: arities) body (option_map (fun _ => 0%nat) height)) k.
Proof.
  intros callees heights arities height i arm body k Harm Hk.
  destruct (instruction_contexts_arms callees heights arities height i) as [Hblock Hif].
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; cbn [instrArm] in Harm.
  - injection Harm as <-. rewrite (Hblock _ _ (or_introl Logic.eq_refl)). cbn [armStart].
    replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    rewrite nth_error_app1 by (rewrite body_contexts_length; exact Hk). reflexivity.
  - injection Harm as <-. rewrite (Hblock _ _ (or_intror Logic.eq_refl)). cbn [armStart].
    replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    rewrite nth_error_app1 by (rewrite body_contexts_length; exact Hk). reflexivity.
  - injection Harm as <-. rewrite (Hif _ _ _ Logic.eq_refl). cbn [armStart].
    replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    rewrite nth_error_app1 by (rewrite body_contexts_length; exact Hk). reflexivity.
  - match type of Harm with match ?e with [] => _ | _ :: _ => _ end = _ => destruct e as [ | e0 e] end;
      [discriminate | ].
    injection Harm as <-. rewrite (Hif _ _ _ Logic.eq_refl). cbn [armStart].
    match goal with |- context [S (S (codeSize ?t))] =>
      replace (S (S (codeSize t)) + k)%nat with (S (codeSize t + S k))%nat by lia end.
    cbn [nth_error]. rewrite nth_error_app2 by (rewrite body_contexts_length; lia).
    rewrite body_contexts_length.
    match goal with |- context [(codeSize ?t + S k - codeSize ?t)%nat] =>
      replace (codeSize t + S k - codeSize t)%nat with (S k) by lia end.
    cbn [nth_error app]. rewrite nth_error_app1 by (rewrite body_contexts_length; exact Hk). reflexivity.
Qed.

(* At a one-slot instruction on a code path, its slot context: the heights and
   arities the code path gives. `heights0` and `arities0` describe the labels
   around `code`, innermost first. *)
Lemma contexts_at_path : forall callees labels code inner offset i hs top heights0 arities0,
  pathCode code labels = Some inner -> nth_error inner offset = Some i -> instructionSize i = 1%nat ->
  pathHeights callees code labels offset = Some (hs, top) ->
  nth_error (bodyContexts callees heights0 arities0 code (Some 0%nat)) (pathOffset code labels offset) =
    nth_error (instructionContexts callees (rev hs ++ heights0) (rev (pathArities code labels) ++ arities0) (Some top) i) 0.
Proof.
  intros callees labels. induction labels as [ | s rest IH];
    intros code inner offset i hs top heights0 arities0 Hpath Hi Hsize Hheights.
  - injection Hpath as Heq. subst code. cbn [pathHeights] in Hheights.
    destruct (sourceHeights callees (firstn offset inner) 0) as [h | ] eqn:Hh; [ | discriminate].
    injection Hheights as <- <-. cbn [pathOffset].
    rewrite <- (Nat.add_0_r (codeOffset inner offset)), (body_contexts_at _ _ _ _ _ _ _ 0 Hi ltac:(lia)).
    cbn [heightsFrom]. rewrite Hh. reflexivity.
  - cbn [pathCode pathHeights pathOffset pathArities] in *.
    destruct (siteEntry code s) as [[j body] | ] eqn:Hentry; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0) as [height | ] eqn:Hheight; [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs' top'] | ] eqn:Hinner; [ | discriminate].
    injection Hheights as <- <-.
    unfold siteEntry in Hentry. destruct (nth_error code (siteOffset s)) as [j' | ] eqn:Hj; [ | discriminate].
    destruct (instrArm j' (siteArm s)) as [body' | ] eqn:Harm; [ | discriminate].
    cbn [option_map] in Hentry. injection Hentry as Hjj Hbb. subst j' body'.
    pose proof (IH body inner offset i hs' top' (labelBase j height :: heights0) (labelArity j :: arities0)
      Hpath Hi Hsize Hinner) as Hctx.
    assert (Hp : (pathOffset body rest offset < codeSize body)%nat).
    { rewrite <- (body_contexts_length callees (labelBase j height :: heights0) (labelArity j :: arities0) body
        (Some 0%nat)).
      apply nth_error_Some. rewrite Hctx. apply nth_error_Some. rewrite instruction_contexts_length. lia. }
    pose proof (arm_inside _ _ _ Harm) as Hinside.
    rewrite <- Nat.add_assoc, (body_contexts_at _ _ _ _ _ _ _ (armStart j (siteArm s) + pathOffset body rest offset) Hj ltac:(lia)).
    cbn [heightsFrom]. rewrite Hheight, (instruction_contexts_arm _ _ _ _ _ _ _ _ Harm Hp).
    cbn [option_map]. rewrite Hctx. cbn [rev]. rewrite <- !app_assoc. reflexivity.
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
  intros callees. induction labels as [ | s rest IH]; intros code offset hs top H; cbn [pathHeights pathArities] in *.
  - destruct (sourceHeights callees (firstn offset code) 0); cbn in H; [ | discriminate]. now injection H as <- _.
  - destruct (siteEntry code s) as [[i body] | ]; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0); [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs' t] | ] eqn:Hin; cbn in H; [ | discriminate].
    injection H as <- _. cbn [length]. f_equal. exact (IH _ _ _ _ Hin).
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
  pose proof (flatten_at_path _ _ _ _ _ Hpath Hi) as Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { rewrite <- flatten_code_length.
    assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    lia. }
  pose proof (contexts_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi Logic.eq_refl Hheights) as Hdrop.
  rewrite <- code_contexts_body in Hdrop. cbn [instructionContexts nth_error option_map] in Hdrop.
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
  pose proof (flatten_at_path _ _ _ _ _ Hpath Hi) as Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { rewrite <- flatten_code_length.
    assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    lia. }
  pose proof (contexts_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi Logic.eq_refl Hheights) as Hdrop.
  rewrite <- code_contexts_body in Hdrop. cbn [instructionContexts nth_error option_map] in Hdrop.
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

(* The ROM's slide at a br: its one edge keeps its target's arity and drops what
   the static heights give. *)
Lemma encoded_br_slide : forall program memory index f labels inner offset l hs top choice,
  programLoaded program memory -> functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR l) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset = Some (hs, top) ->
  let pc := guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) in
  let slide := labelSlide (rev hs ++ [0%nat]) (rev (pathArities (guestBody f) labels) ++ [guestResults f]) top
    (N.to_nat (l :> N)) in
  memory frameSlideRemainingId [pc; choice] = Some (if (choice =? 0)%Z then slideEntry true slide else 0%Z) /\
  memory frameSlideDistanceId [pc; choice] = Some (if (choice =? 0)%Z then slideEntry false slide else 0%Z).
Proof.
  intros program memory index f labels inner offset l hs top choice Hloaded Hf Hpath Hi Hheights pc slide.
  pose proof (flatten_at_path _ _ _ _ _ Hpath Hi) as Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { rewrite <- flatten_code_length.
    assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    lia. }
  pose proof (contexts_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi Logic.eq_refl Hheights) as Hdrop.
  rewrite <- code_contexts_body in Hdrop. cbn [instructionContexts nth_error option_map] in Hdrop.
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ _ choice Hf Hbound); unfold functionSlide; cbv zeta;
    rewrite Hslot, Hdrop; fold slide; cbn [Nat.eqb frameSlideRemainingId frameSlideDistanceId].
  all: destruct slide as [[kept dropped] | ]; destruct (choice =? 0)%Z; reflexivity.
Qed.

(* The ROM's slide at a br_table: each edge, the default at choice 0 and arm k at
   choice k + 1, pops the index, keeps its target's arity and drops what the
   static heights give. *)
Lemma encoded_br_table_slide : forall program memory index f labels inner offset ls l hs top k target,
  programLoaded program memory -> functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_TABLE ls l) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset = Some (hs, top) ->
  nth_error (l :: ls) k = Some target ->
  let pc := guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) in
  let slide := branchSlide (rev hs ++ [0%nat]) (rev (pathArities (guestBody f) labels) ++ [guestResults f]) top
    (N.to_nat (target :> N)) in
  memory frameSlideRemainingId [pc; Z.of_nat k] = Some (slideEntry true slide) /\
  memory frameSlideDistanceId [pc; Z.of_nat k] = Some (slideEntry false slide).
Proof.
  intros program memory index f labels inner offset ls l hs top k target Hloaded Hf Hpath Hi Hheights Hk pc slide.
  pose proof (flatten_at_path _ _ _ _ _ Hpath Hi) as Hslot.
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { rewrite <- flatten_code_length.
    assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate).
    lia. }
  pose proof (contexts_at_path _ _ _ _ _ _ _ _ [0%nat] [guestResults f] Hpath Hi Logic.eq_refl Hheights) as Hdrop.
  rewrite <- code_contexts_body in Hdrop. cbn [instructionContexts nth_error option_map] in Hdrop.
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ _ (Z.of_nat k) Hf Hbound); unfold functionSlide; cbv zeta;
    rewrite Hslot, Hdrop; cbn [Nat.eqb frameSlideRemainingId frameSlideDistanceId].
  all: replace (Z.of_nat k <? 0) with false by (symmetry; apply Z.ltb_ge; lia); rewrite Nat2Z.id, Hk; fold slide.
  all: destruct slide as [[kept dropped] | ]; reflexivity.
Qed.
