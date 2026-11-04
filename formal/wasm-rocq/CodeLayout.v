(* Flatten structured code into instruction addresses, recursively, as the Rust
   preprocessing numbers operators: a block or loop, its body, then its END; an
   if, its then-body, an ELSE when it has an else-body, that body, then its END.
   Each slot records, per edge, the function-relative address it jumps to: a
   branch lands where its target label continues (after a block's or if's END, at
   a loop's body), and an if's false edge at its else-body, or after its END. *)
From Stdlib Require Import List Arith NArith Lia.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StackSimulation.
Import ListNotations.

(* A slot holds an instruction, one of the markers that delimit structured code,
   or an indirect call's invocation, which follows its resolve slot. *)
Inductive FlatOp := opInstr (i : instr) | opElse | opEnd | opInvoke.

Record FlatInstruction := flatInstruction {
  flatOpcode : FlatOp;
  flatTargets : list (option nat)
}.

(* A branch's targets, one per edge: br_table's default first, then its arms, as
   the circuit numbers its control choices. *)
Definition branchTargets (i : instr) : list labelidx :=
  match i with BR_IF l | BR l => [l] | BR_TABLE ls l => l :: ls | _ => [] end.

Fixpoint instructionSize (i : instr) : nat :=
  let codeSize := fix codeSize (code : list instr) : nat :=
    match code with [] => 0 | j :: rest => instructionSize j + codeSize rest end in
  match i with
  | BLOCK _ body | LOOP _ body => S (S (codeSize body))
  | IFELSE _ t e => S (S (codeSize t + match e with [] => 0 | _ => S (codeSize e) end))
  | CALL_INDIRECT _ _ => 2
  | _ => 1
  end.

Fixpoint codeSize (code : list instr) : nat :=
  match code with [] => 0 | j :: rest => instructionSize j + codeSize rest end.

Definition codeOffset (code : list instr) (offset : nat) := codeSize (firstn offset code).

(* Where branches to an instruction's label land, relative to the instruction: a
   loop's body, otherwise after its END. *)
Definition labelTarget (i : instr) : nat := match i with LOOP _ _ => 1 | _ => instructionSize i end.

(* An instruction's slots at function address `base`; `labels` holds, innermost
   first, where each enclosing label's branches land. An if's false edge lands
   past its then-body and ELSE: at its else-body, or after its END. *)
Fixpoint flattenAt (labels : list nat) (base : nat) (i : instr) {struct i} : list FlatInstruction :=
  let bodyAt := fix bodyAt (labels : list nat) (base : nat) (code : list instr) : list FlatInstruction :=
    match code with
    | [] => []
    | j :: rest => flattenAt labels base j ++ bodyAt labels (base + instructionSize j) rest
    end in
  let inner := (base + labelTarget i :: labels)%nat in
  match i with
  | BLOCK _ body | LOOP _ body =>
      flatInstruction (opInstr i) [] :: bodyAt inner (S base) body ++ [flatInstruction opEnd []]
  | IFELSE _ t e =>
      flatInstruction (opInstr i) [Some (base + S (S (codeSize t)))] ::
        bodyAt inner (S base) t ++
        match e with
        | [] => []
        | _ => flatInstruction opElse [Some (base + instructionSize i)] :: bodyAt inner (base + S (S (codeSize t))) e
        end ++ [flatInstruction opEnd []]
  | BR_IF _ | BR _ | BR_TABLE _ _ =>
      [flatInstruction (opInstr i) (map (fun l => nth_error labels (N.to_nat (l :> N))) (branchTargets i))]
  | CALL_INDIRECT _ _ => [flatInstruction (opInstr i) []; flatInstruction opInvoke []]
  | _ => [flatInstruction (opInstr i) []]
  end.

Fixpoint bodyAt (labels : list nat) (base : nat) (code : list instr) : list FlatInstruction :=
  match code with
  | [] => []
  | j :: rest => flattenAt labels base j ++ bodyAt labels (base + instructionSize j) rest
  end.

(* A function body has no enclosing label. *)
Definition flattenCode (code : list instr) := bodyAt [] 0 code.

(* An instruction's first slot. *)
Definition slotTargets (labels : list nat) (base : nat) (i : instr) : list (option nat) :=
  match i with
  | IFELSE _ t _ => [Some (base + S (S (codeSize t)))]
  | BR_IF _ | BR _ | BR_TABLE _ _ => map (fun l => nth_error labels (N.to_nat (l :> N))) (branchTargets i)
  | _ => []
  end.

Lemma flatten_head : forall labels base i, exists more,
  flattenAt labels base i = flatInstruction (opInstr i) (slotTargets labels base i) :: more.
Proof. intros labels base i. destruct i; eexists; reflexivity. Qed.

Lemma body_at_cons : forall labels base j rest,
  bodyAt labels base (j :: rest) = flattenAt labels base j ++ bodyAt labels (base + instructionSize j) rest.
Proof. reflexivity. Qed.

Lemma code_size_cons : forall i rest, codeSize (i :: rest) = instructionSize i + codeSize rest.
Proof. reflexivity. Qed.

Lemma code_size_app : forall a b, codeSize (a ++ b) = codeSize a + codeSize b.
Proof. induction a as [ | i a IH]; intros b; cbn [app codeSize]; [reflexivity | rewrite IH; lia]. Qed.

Lemma block_instruction_size : forall bt body, instructionSize (BLOCK bt body) = (codeSize body + 2)%nat.
Proof. intros. change (S (S (codeSize body)) = codeSize body + 2)%nat. lia. Qed.

Lemma loop_instruction_size : forall bt body, instructionSize (LOOP bt body) = (codeSize body + 2)%nat.
Proof. intros. change (S (S (codeSize body)) = codeSize body + 2)%nat. lia. Qed.

Lemma if_instruction_size : forall bt t e,
  instructionSize (IFELSE bt t e) = (S (S (codeSize t)) + match e with [] => 0 | _ => S (codeSize e) end)%nat.
Proof. intros. change (S (S (codeSize t + match e with [] => 0 | _ => S (codeSize e) end)) =
  S (S (codeSize t)) + match e with [] => 0 | _ => S (codeSize e) end)%nat. lia. Qed.

Lemma flatten_block : forall labels base bt body,
  flattenAt labels base (BLOCK bt body) = flatInstruction (opInstr (BLOCK bt body)) [] ::
    bodyAt (base + labelTarget (BLOCK bt body) :: labels) (S base) body ++ [flatInstruction opEnd []].
Proof. reflexivity. Qed.

Lemma flatten_loop : forall labels base bt body,
  flattenAt labels base (LOOP bt body) = flatInstruction (opInstr (LOOP bt body)) [] ::
    bodyAt (base + labelTarget (LOOP bt body) :: labels) (S base) body ++ [flatInstruction opEnd []].
Proof. reflexivity. Qed.

Lemma flatten_if : forall labels base bt t e,
  flattenAt labels base (IFELSE bt t e) =
    flatInstruction (opInstr (IFELSE bt t e)) [Some (base + S (S (codeSize t)))] ::
      bodyAt (base + labelTarget (IFELSE bt t e) :: labels) (S base) t ++
      match e with
      | [] => []
      | _ => flatInstruction opElse [Some (base + instructionSize (IFELSE bt t e))] ::
          bodyAt (base + labelTarget (IFELSE bt t e) :: labels) (base + S (S (codeSize t))) e
      end ++ [flatInstruction opEnd []].
Proof. reflexivity. Qed.

Lemma instruction_size_pos : forall i, (1 <= instructionSize i)%nat.
Proof. destruct i; cbn; lia. Qed.

(* By the size of the code: a body's instructions are smaller than it. *)
Lemma flatten_body_length : forall n,
  (forall i labels base, (instructionSize i <= n)%nat -> length (flattenAt labels base i) = instructionSize i) /\
  (forall code labels base, (codeSize code <= n)%nat -> length (bodyAt labels base code) = codeSize code).
Proof.
  induction n as [ | n [IHi IHc]].
  - split; [intros i labels base H; pose proof (instruction_size_pos i); lia | ].
    intros [ | j rest] labels base H; [reflexivity | ]. pose proof (instruction_size_pos j). cbn [codeSize] in H. lia.
  - assert (Hi : forall i labels base, (instructionSize i <= S n)%nat ->
        length (flattenAt labels base i) = instructionSize i).
    { intros i labels base H. destruct i; try reflexivity.
      - rewrite flatten_block. rewrite block_instruction_size in *. cbn [length].
        rewrite length_app, IHc by lia. cbn [length]. lia.
      - rewrite flatten_loop. rewrite loop_instruction_size in *. cbn [length].
        rewrite length_app, IHc by lia. cbn [length]. lia.
      - rewrite flatten_if. rewrite if_instruction_size in *. cbn [length].
        rewrite !length_app, IHc by lia.
        match goal with |- context [match ?e with [] => _ | _ => _ end] => destruct e end;
          cbn [length codeSize] in *; rewrite ?IHc by (cbn [codeSize]; lia); cbn [codeSize]; lia. }
    split; [exact Hi | ].
    induction code as [ | j rest IH]; intros labels base H; [reflexivity | ].
    rewrite body_at_cons, length_app. cbn [codeSize] in H |- *. pose proof (instruction_size_pos j).
    rewrite Hi, IH by lia. reflexivity.
Qed.

Lemma flatten_length : forall i labels base, length (flattenAt labels base i) = instructionSize i.
Proof. intros. exact (proj1 (flatten_body_length (instructionSize i)) i labels base (le_n _)). Qed.

Lemma body_length : forall code labels base, length (bodyAt labels base code) = codeSize code.
Proof. intros. exact (proj2 (flatten_body_length (codeSize code)) code labels base (le_n _)). Qed.

Lemma flatten_code_length : forall code, length (flattenCode code) = codeSize code.
Proof. intros. apply body_length. Qed.

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
  rewrite (firstn_at_instruction code offset i H), code_size_app. cbn [codeSize]. lia.
Qed.

Lemma code_offset_end : forall code, codeOffset code (length code) = codeSize code.
Proof. intros; unfold codeOffset; now rewrite firstn_all. Qed.

Lemma code_size_split : forall code offset, codeSize code = codeOffset code offset + codeSize (skipn offset code).
Proof. intros. unfold codeOffset. rewrite <- code_size_app, firstn_skipn. reflexivity. Qed.

(* A body's instruction's slots, at its offset in the body. *)
Lemma body_at_instruction : forall labels code base offset i k,
  nth_error code offset = Some i -> (k < instructionSize i)%nat ->
  nth_error (bodyAt labels base code) (codeOffset code offset + k) =
    nth_error (flattenAt labels (base + codeOffset code offset) i) k.
Proof.
  intros labels. induction code as [ | j rest IH]; intros base [ | offset] i k Hi Hk; try discriminate.
  - injection Hi as <-. rewrite body_at_cons. change (codeOffset (j :: rest) 0) with 0%nat.
    rewrite nth_error_app1 by (rewrite flatten_length; exact Hk). rewrite !Nat.add_0_r. reflexivity.
  - cbn [nth_error] in Hi. rewrite body_at_cons.
    replace (codeOffset (j :: rest) (S offset) + k)%nat with
      (length (flattenAt labels base j) + (codeOffset rest offset + k))%nat
      by (rewrite flatten_length; unfold codeOffset; cbn [firstn codeSize]; lia).
    rewrite nth_error_app2 by lia.
    replace (length (flattenAt labels base j) + (codeOffset rest offset + k) - length (flattenAt labels base j))%nat
      with (codeOffset rest offset + k) by lia.
    rewrite (IH _ _ _ _ Hi Hk). f_equal. f_equal. unfold codeOffset. cbn [firstn codeSize]. lia.
Qed.

(* A position inside nested structured code: the enclosing labels, outermost first,
   each the structured instruction at `siteOffset` entered through arm `siteArm`. A
   block's or loop's body is arm 0; an if's then- and else-bodies are arms 0 and 1.
   An empty else-body is never entered: the if's false edge skips it. *)
Record LabelSite := labelSite { siteOffset : nat; siteArm : nat }.

Definition instrArm (i : instr) (arm : nat) : option (list instr) :=
  match i with
  | BLOCK _ body | LOOP _ body => match arm with O => Some body | _ => None end
  | IFELSE _ t e => match arm with O => Some t | 1 => match e with [] => None | _ => Some e end | _ => None end
  | _ => None
  end.

(* Where an arm's body starts, relative to its instruction. *)
Definition armStart (i : instr) (arm : nat) : nat :=
  match i, arm with IFELSE _ t _, 1 => S (S (codeSize t)) | _, _ => 1 end.

(* An arm's last slot: an if's then-body ends at its ELSE when it has an
   else-body; every other arm ends at its instruction's END. *)
Definition armEnd (i : instr) (arm : nat) (after : nat) : FlatInstruction :=
  match i, arm with
  | IFELSE _ _ (_ :: _), O => flatInstruction opElse [Some after]
  | _, _ => flatInstruction opEnd []
  end.

(* An arm's body and last slot lie inside its instruction. *)
Lemma arm_inside : forall i arm body,
  instrArm i arm = Some body -> (armStart i arm + codeSize body < instructionSize i)%nat.
Proof.
  intros i arm body Harm.
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; try (injection Harm as Hbody; subst);
    rewrite ?block_instruction_size, ?loop_instruction_size, ?if_instruction_size; cbn [armStart].
  - lia.
  - lia.
  - match goal with |- context [match ?e with [] => _ | _ => _ end] => destruct e end; cbv iota; lia.
  - cbn [instrArm] in Harm.
    match type of Harm with context [match ?e with [] => _ | _ :: _ => _ end] => destruct e end; [discriminate | ].
    injection Harm as Hbody; subst. lia.
Qed.

(* The structured instruction at a site and the body its arm enters. *)
Definition siteEntry (code : list instr) (s : LabelSite) : option (instr * list instr) :=
  match nth_error code (siteOffset s) with
  | Some i => option_map (fun body => (i, body)) (instrArm i (siteArm s))
  | None => None
  end.

Fixpoint pathCode (code : list instr) (labels : list LabelSite) : option (list instr) :=
  match labels with
  | [] => Some code
  | s :: rest => match siteEntry code s with Some (_, body) => pathCode body rest | None => None end
  end.

(* The address of an offset in the innermost body, relative to code's start. *)
Fixpoint pathOffset (code : list instr) (labels : list LabelSite) (offset : nat) : nat :=
  match labels with
  | [] => codeOffset code offset
  | s :: rest => match siteEntry code s with
      | Some (i, body) => codeOffset code (siteOffset s) + armStart i (siteArm s) + pathOffset body rest offset
      | None => 0
      end
  end.

(* Where each enclosing label's branches land, innermost first, relative to code's start. *)
Fixpoint pathTargets (code : list instr) (labels : list LabelSite) : list nat :=
  match labels with
  | [] => []
  | s :: rest => match siteEntry code s with
      | Some (i, body) =>
          map (Nat.add (codeOffset code (siteOffset s) + armStart i (siteArm s))) (pathTargets body rest) ++
            [codeOffset code (siteOffset s) + labelTarget i]
      | None => []
      end
  end.

(* An arm's slots inside its instruction's, then the arm's last slot. *)
Lemma flatten_arm : forall labels base i arm body k,
  instrArm i arm = Some body -> (k <= codeSize body)%nat ->
  nth_error (flattenAt labels base i) (armStart i arm + k) =
    nth_error (bodyAt (base + labelTarget i :: labels) (base + armStart i arm) body ++
      [armEnd i arm (base + instructionSize i)]) k.
Proof.
  intros labels base i arm body k Harm Hk.
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; try (injection Harm as Hbody; subst).
  - rewrite flatten_block. cbn [armStart armEnd]. replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    cbn [labelTarget]. rewrite ?Nat.add_1_r. reflexivity.
  - rewrite flatten_loop. cbn [armStart armEnd]. replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    cbn [labelTarget]. rewrite ?Nat.add_1_r. reflexivity.
  - rewrite flatten_if. cbn [armStart]. replace (1 + k)%nat with (S k) by lia. cbn [nth_error].
    cbn [labelTarget]. rewrite ?Nat.add_1_r.
    match goal with |- context [armEnd (IFELSE _ _ ?e)] => destruct e as [ | e0 e] end; cbv beta iota; cbn [armEnd].
    + reflexivity.
    + destruct (Nat.lt_ge_cases k (codeSize body)) as [Hlt | Hge].
      * rewrite !nth_error_app1 by (rewrite body_length; lia). reflexivity.
      * replace k with (codeSize body) by lia.
        assert (Hat : forall (xs ys : list FlatInstruction), nth_error (xs ++ ys) (length xs) = nth_error ys 0)
          by (intros; rewrite nth_error_app2, Nat.sub_diag by lia; reflexivity).
        match goal with |- context [bodyAt ?ls (S base) body] => rewrite <- (body_length body ls (S base)) end.
        rewrite !Hat. reflexivity.
  - cbn [instrArm] in Harm.
    match type of Harm with context [match ?e with [] => _ | _ :: _ => _ end] => destruct e as [ | e0 e] end;
      [discriminate | ].
    injection Harm as Hbody; subst.
    rewrite flatten_if. cbv beta iota. cbn [armStart armEnd].
    match goal with |- context [S (S (codeSize ?t))] =>
      replace (S (S (codeSize t)) + k)%nat with (S (codeSize t + S k))%nat by lia end.
    cbn [nth_error]. rewrite nth_error_app2 by (rewrite body_length; lia).
    rewrite body_length. match goal with |- context [(codeSize ?t + S k - codeSize ?t)%nat] =>
      replace (codeSize t + S k - codeSize t)%nat with (S k) by lia end. reflexivity.
Qed.

(* An instruction's slots at a code path. *)
Lemma path_slots : forall code labels inner offset i targets base k,
  pathCode code labels = Some inner -> nth_error inner offset = Some i -> (k < instructionSize i)%nat ->
  nth_error (bodyAt targets base code) (pathOffset code labels offset + k) =
    nth_error (flattenAt (map (Nat.add base) (pathTargets code labels) ++ targets)
      (base + pathOffset code labels offset) i) k.
Proof.
  intros code labels. revert code.
  induction labels as [ | s rest IH]; intros code inner offset i targets base k Hpath Hi Hk.
  - injection Hpath as Heq. subst code. cbn [pathOffset pathTargets map app].
    exact (body_at_instruction _ _ _ _ _ _ Hi Hk).
  - cbn [pathCode pathOffset pathTargets] in *. unfold siteEntry in *.
    destruct (nth_error code (siteOffset s)) as [j | ] eqn:Hj; [ | discriminate].
    destruct (instrArm j (siteArm s)) as [body | ] eqn:Harm; [ | discriminate]. cbn [option_map] in *.
    pose proof (IH _ _ _ _ (base + codeOffset code (siteOffset s) + labelTarget j :: targets)
      (base + codeOffset code (siteOffset s) + armStart j (siteArm s)) _ Hpath Hi Hk) as Hinner.
    assert (Hlen : (pathOffset body rest offset + k < codeSize body)%nat).
    { rewrite <- (body_length body (base + codeOffset code (siteOffset s) + labelTarget j :: targets)
        (base + codeOffset code (siteOffset s) + armStart j (siteArm s))).
      apply nth_error_Some. rewrite Hinner. apply nth_error_Some. rewrite flatten_length. exact Hk. }
    assert (Hsize : (armStart j (siteArm s) + (pathOffset body rest offset + k) < instructionSize j)%nat).
    { pose proof (arm_inside _ _ _ Harm). lia. }
    match goal with |- nth_error _ ?n = _ =>
      replace n with (codeOffset code (siteOffset s) + (armStart j (siteArm s) + (pathOffset body rest offset + k)))%nat
        by lia end.
    rewrite (body_at_instruction _ _ _ _ _ _ Hj Hsize).
    rewrite (flatten_arm _ _ _ _ _ (pathOffset body rest offset + k) Harm ltac:(lia)), nth_error_app1
      by (rewrite body_length; exact Hlen).
    rewrite Hinner. f_equal. f_equal; [ | lia].
    rewrite map_app, <- app_assoc, map_map. cbn [map app].
    f_equal; [apply map_ext; intros; lia | f_equal; lia].
Qed.

(* An instruction's head slot at a code path. *)
Lemma path_slot : forall code labels inner offset i targets base,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (bodyAt targets base code) (pathOffset code labels offset) =
    Some (flatInstruction (opInstr i) (slotTargets
      (map (Nat.add base) (pathTargets code labels) ++ targets) (base + pathOffset code labels offset) i)).
Proof.
  intros code labels inner offset i targets base Hpath Hi.
  rewrite <- (Nat.add_0_r (pathOffset code labels offset)),
    (path_slots _ _ _ _ _ _ _ 0 Hpath Hi (instruction_size_pos i)), Nat.add_0_r.
  destruct (flatten_head (map (Nat.add base) (pathTargets code labels) ++ targets)
    (base + pathOffset code labels offset) i) as [more ->].
  reflexivity.
Qed.

(* The last slot of the innermost arm at a code path. *)
Lemma path_arm_end : forall code pre s inner targets base,
  pathCode code (pre ++ [s]) = Some inner ->
  exists i, siteEntry (match pathCode code pre with Some outer => outer | None => [] end) s = Some (i, inner) /\
    nth_error (bodyAt targets base code) (pathOffset code (pre ++ [s]) (length inner)) =
      Some (armEnd i (siteArm s) (base + pathOffset code pre (siteOffset s) + instructionSize i)).
Proof.
  intros code pre. revert code. induction pre as [ | s0 pre IH]; intros code s inner targets base Hpath;
    cbn [app pathCode pathOffset] in *.
  - unfold siteEntry in *. destruct (nth_error code (siteOffset s)) as [i | ] eqn:Hi; [ | discriminate].
    destruct (instrArm i (siteArm s)) as [body | ] eqn:Harm; [ | discriminate]. cbn [option_map] in *.
    injection Hpath as <-. exists i. split; [reflexivity | ].
    rewrite code_offset_end.
    assert (Hsize : (armStart i (siteArm s) + codeSize body < instructionSize i)%nat).
    { pose proof (arm_inside _ _ _ Harm). lia. }
    rewrite <- Nat.add_assoc, (body_at_instruction _ _ _ _ _ _ Hi Hsize).
    rewrite (flatten_arm _ _ _ _ _ (codeSize body) Harm ltac:(lia)), nth_error_app2, body_length, Nat.sub_diag
      by (rewrite body_length; lia).
    cbn [pathOffset]. reflexivity.
  - unfold siteEntry in Hpath at 1 |- *. destruct (nth_error code (siteOffset s0)) as [j | ] eqn:Hj; [ | discriminate].
    destruct (instrArm j (siteArm s0)) as [body | ] eqn:Harm; [ | discriminate]. cbn [option_map] in *.
    destruct (IH _ _ _ (base + codeOffset code (siteOffset s0) + labelTarget j :: targets)
      (base + codeOffset code (siteOffset s0) + armStart j (siteArm s0)) Hpath) as [i [Hentry Hend]].
    exists i. split; [exact Hentry | ].
    assert (Hlen : (pathOffset body (pre ++ [s]) (length inner) < codeSize body)%nat).
    { rewrite <- (body_length body (base + codeOffset code (siteOffset s0) + labelTarget j :: targets)
        (base + codeOffset code (siteOffset s0) + armStart j (siteArm s0))).
      apply nth_error_Some. rewrite Hend. discriminate. }
    assert (Hsize : (armStart j (siteArm s0) + pathOffset body (pre ++ [s]) (length inner) < instructionSize j)%nat).
    { pose proof (arm_inside _ _ _ Harm). lia. }
    rewrite <- Nat.add_assoc, (body_at_instruction _ _ _ _ _ _ Hj Hsize).
    rewrite (flatten_arm _ _ _ _ _ (pathOffset body (pre ++ [s]) (length inner)) Harm ltac:(lia)), nth_error_app1
      by (rewrite body_length; exact Hlen).
    rewrite Hend. f_equal. f_equal. lia.
Qed.

(* A function's slots: an instruction's head slot at a code path, carrying where
   each of its edges lands, and the last slot of the innermost arm. *)
Lemma flatten_at_path : forall code labels inner offset i,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (flattenCode code) (pathOffset code labels offset) =
    Some (flatInstruction (opInstr i) (slotTargets (pathTargets code labels) (pathOffset code labels offset) i)).
Proof.
  intros code labels inner offset i Hpath Hi. unfold flattenCode.
  rewrite (path_slot _ _ _ _ _ [] 0 Hpath Hi), app_nil_r.
  replace (map (Nat.add 0) (pathTargets code labels)) with (pathTargets code labels)
    by (rewrite <- (map_id (pathTargets code labels)) at 1; apply map_ext; reflexivity).
  reflexivity.
Qed.

(* An indirect call's invocation follows its resolve slot. *)
Lemma flatten_at_invoke : forall code labels inner offset x y,
  pathCode code labels = Some inner -> nth_error inner offset = Some (CALL_INDIRECT x y) ->
  nth_error (flattenCode code) (S (pathOffset code labels offset)) = Some (flatInstruction opInvoke []).
Proof.
  intros code labels inner offset x y Hpath Hi. unfold flattenCode.
  rewrite <- Nat.add_1_r. rewrite (path_slots _ _ _ _ _ [] 0 1 Hpath Hi ltac:(cbn; lia)). reflexivity.
Qed.

Lemma flatten_arm_end : forall code pre s inner,
  pathCode code (pre ++ [s]) = Some inner ->
  exists i, siteEntry (match pathCode code pre with Some outer => outer | None => [] end) s = Some (i, inner) /\
    nth_error (flattenCode code) (pathOffset code (pre ++ [s]) (length inner)) =
      Some (armEnd i (siteArm s) (pathOffset code pre (siteOffset s) + instructionSize i)).
Proof. intros code pre s inner Hpath. exact (path_arm_end code pre s inner [] 0 Hpath). Qed.

Lemma path_offset_shift : forall labels code inner offset,
  pathCode code labels = Some inner -> pathOffset code labels offset = pathOffset code labels 0 + codeOffset inner offset.
Proof.
  induction labels as [ | s rest IH]; intros code inner offset Hpath.
  - injection Hpath as <-. reflexivity.
  - cbn [pathCode pathOffset] in *. destruct (siteEntry code s) as [[i body] | ]; [ | discriminate].
    rewrite (IH _ _ offset Hpath), (IH _ _ 0 Hpath). lia.
Qed.

Lemma path_targets_length : forall labels code inner,
  pathCode code labels = Some inner -> length (pathTargets code labels) = length labels.
Proof.
  induction labels as [ | s rest IH]; intros code inner Hpath; [reflexivity | ].
  cbn [pathCode pathTargets] in *. destruct (siteEntry code s) as [[i body] | ]; [ | discriminate].
  rewrite length_app, length_map, (IH _ _ Hpath). cbn [length]. lia.
Qed.

(* The branches of a body at a path to its label at `s` after `pre` land at the
   label's target. *)
Lemma path_target : forall code pre s post inner outer i body,
  pathCode code (pre ++ s :: post) = Some inner -> pathCode code pre = Some outer ->
  siteEntry outer s = Some (i, body) ->
  nth_error (pathTargets code (pre ++ s :: post)) (length post) =
    Some (pathOffset code pre (siteOffset s) + labelTarget i).
Proof.
  intros code pre. revert code. induction pre as [ | s0 pre IH]; intros code s post inner outer i body Hpath Hpre Hentry;
    cbn [app pathCode pathTargets pathOffset] in *.
  - injection Hpre as <-. rewrite Hentry in Hpath |- *.
    rewrite nth_error_app2; rewrite length_map, (path_targets_length _ _ _ Hpath); [ | lia].
    rewrite Nat.sub_diag. reflexivity.
  - destruct (siteEntry code s0) as [[j nested] | ]; [ | discriminate].
    rewrite nth_error_app1.
    + rewrite nth_error_map, (IH _ _ _ _ _ _ _ Hpath Hpre Hentry). cbn [Datatypes.option_map]. f_equal. lia.
    + rewrite length_map, (path_targets_length _ _ _ Hpath), length_app. cbn [length]. lia.
Qed.

Lemma path_offset_next : forall code labels inner offset i,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  pathOffset code labels (S offset) = pathOffset code labels offset + instructionSize i.
Proof.
  intros code labels inner offset i Hpath Hi.
  rewrite (path_offset_shift _ _ _ (S offset) Hpath), (path_offset_shift _ _ _ offset Hpath),
    (code_offset_next _ _ _ Hi). lia.
Qed.

(* A path one label deeper. *)
Lemma path_snoc : forall labels code s outer i body,
  pathCode code labels = Some outer -> siteEntry outer s = Some (i, body) ->
  pathCode code (labels ++ [s]) = Some body /\
  forall offset, pathOffset code (labels ++ [s]) offset =
    pathOffset code labels (siteOffset s) + armStart i (siteArm s) + codeOffset body offset.
Proof.
  induction labels as [ | s0 rest IH]; intros code s outer i body Hpath Hentry; cbn [app pathCode pathOffset] in *.
  - injection Hpath as <-. rewrite Hentry. split; [reflexivity | intros; reflexivity].
  - destruct (siteEntry code s0) as [[j nested] | ]; [ | discriminate].
    destruct (IH _ _ _ _ _ Hpath Hentry) as [Hcode Hoffset]. split; [exact Hcode | ].
    intros offset. rewrite Hoffset. lia.
Qed.

Lemma site_entry_split : forall code s i body,
  siteEntry code s = Some (i, body) -> nth_error code (siteOffset s) = Some i /\ instrArm i (siteArm s) = Some body.
Proof.
  intros code s i body H. unfold siteEntry in H. destruct (nth_error code (siteOffset s)) as [j | ]; [ | discriminate].
  destruct (instrArm j (siteArm s)) as [arm | ] eqn:Harm; [ | discriminate]. cbn in H. injection H as <- <-.
  split; [reflexivity | exact Harm].
Qed.

Lemma arm_wf : forall i arm body, wf_instr i -> instrArm i arm = Some body -> Forall wf_instr body.
Proof.
  intros i arm body Hw Harm.
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; cbn [instrArm] in Harm;
    inversion Hw; subst.
  - injection Harm as <-. assumption.
  - injection Harm as <-. assumption.
  - injection Harm as <-. assumption.
  - match type of Harm with match ?e with [] => _ | _ :: _ => _ end = _ => destruct e end; [discriminate | ].
    injection Harm as <-. assumption.
Qed.

Lemma site_wf : forall code s i body, Forall wf_instr code -> siteEntry code s = Some (i, body) -> Forall wf_instr body.
Proof.
  intros code s i body Hwf Hentry. destruct (site_entry_split _ _ _ _ Hentry) as [Hi Harm].
  exact (arm_wf _ _ _ (proj1 (Forall_forall _ _) Hwf _ (nth_error_In _ _ Hi)) Harm).
Qed.
