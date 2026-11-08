(* The types at a source-code position. These are source types, not witness tags.
   This computes fallthrough effects; SpecTec validation separately checks label
   targets and structured bodies. Unconditional transfers have no fallthrough. *)
From Stdlib Require Import List Arith NArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 CallerSpec Fragment StraightLineInstructions LocalSimulation CodeLayout.
Import ListNotations.
Open Scope nat_scope.

(* "replace" here means pop (inputs) + push (outputs) *)
Definition replaceStackTypes (inputs outputs stack : list valtype) : option (list valtype) :=
  (* the stack needs to have enough arguments to pop *)
  if length inputs <=? length stack then
    let below := length stack - length inputs in
    if list_eq_dec valtype_eq_dec (skipn below stack) inputs
    then Some (firstn below stack ++ outputs) else None
  else None.

Definition instructionTypes (locals : list valtype) (callees : CalleeEnvironment) (i : instr)
    : option (list valtype * list valtype) :=
  match i with
  | BINOP I32 (mk_binop__0 Inn_I32 ADD)
  | BINOP I32 (mk_binop__0 Inn_I32 SUB) => Some ([valtype_I32; valtype_I32], [valtype_I32])
  | LOCAL_GET x => option_map (fun t => ([], [t])) (nth_error locals (N.to_nat (x :> N)))
  | LOCAL_SET x => option_map (fun t => ([t], [])) (nth_error locals (N.to_nat (x :> N)))
  | CALL x => option_map (fun f => (guestParamTypes f, guestResultTypes f)) (callees (x :> N))
  | CALL_INDIRECT _ y =>
      match nth_error (envTypes callees) (N.to_nat (y :> N)) with
      | Some (mk_functype (mk_list inputs) (mk_list outputs)) => Some (inputs ++ [valtype_I32], outputs)
      | None => None
      end
  | BLOCK (_RESULT result) _ | LOOP (_RESULT result) _ => Some ([], option_to_list result)
  | IFELSE (_RESULT result) _ _ => Some ([valtype_I32], option_to_list result)
  | BR_IF _ => Some ([valtype_I32], [])
  | NOP => Some ([], [])
  | _ => None
  end.

Definition sourceTypes (locals : list valtype) (callees : CalleeEnvironment) (i : instr)
    (stack : list valtype) : option (list valtype) :=
  match instructionTypes locals callees i with
  | Some (inputs, outputs) => replaceStackTypes inputs outputs stack
  | None => None
  end.

Fixpoint sourceTypeStacks (locals : list valtype) (callees : CalleeEnvironment)
    (code : list instr) (stack : list valtype) : option (list valtype) :=
  match code with
  | [] => Some stack
  | i :: rest => match sourceTypes locals callees i stack with
      | Some next => sourceTypeStacks locals callees rest next
      | None => None
      end
  end.

Lemma replace_stack_types_exact : forall inputs outputs prefix,
  replaceStackTypes inputs outputs (prefix ++ inputs) = Some (prefix ++ outputs).
Proof.
  intros. unfold replaceStackTypes. rewrite length_app.
  replace (length inputs <=? length prefix + length inputs) with true by (symmetry; apply Nat.leb_le; lia).
  replace (length prefix + length inputs - length inputs) with (length prefix) by lia.
  rewrite skipn_app, skipn_all, Nat.sub_diag. cbn [skipn app].
  destruct (list_eq_dec valtype_eq_dec inputs inputs); [ | contradiction].
  rewrite firstn_app, firstn_all, Nat.sub_diag. cbn [firstn]. rewrite app_nil_r. reflexivity.
Qed.

Lemma replace_stack_types_some : forall inputs outputs stack next,
  replaceStackTypes inputs outputs stack = Some next <->
  exists prefix, stack = prefix ++ inputs /\ next = prefix ++ outputs.
Proof.
  intros inputs outputs stack next. split.
  - unfold replaceStackTypes. destruct (length inputs <=? length stack) eqn:Hlength; [ | discriminate].
    apply Nat.leb_le in Hlength.
    destruct (list_eq_dec valtype_eq_dec (skipn (length stack - length inputs) stack) inputs)
      as [Hsuffix | ]; [ | discriminate].
    intros H. injection H as <-. exists (firstn (length stack - length inputs) stack).
    split; [ | reflexivity].
    pose proof (firstn_skipn (length stack - length inputs) stack) as Hsplit.
    rewrite Hsuffix in Hsplit. symmetry. exact Hsplit.
  - intros [prefix [-> ->]]. apply replace_stack_types_exact.
Qed.

Lemma source_type_stacks_app : forall locals callees prefix suffix stack,
  sourceTypeStacks locals callees (prefix ++ suffix) stack =
    match sourceTypeStacks locals callees prefix stack with
    | Some middle => sourceTypeStacks locals callees suffix middle | None => None end.
Proof.
  intros locals callees prefix. induction prefix as [ | i prefix IH]; intros suffix stack; cbn; [reflexivity | ].
  destruct (sourceTypes locals callees i stack); [apply IH | reflexivity].
Qed.

Lemma local_get_types : forall locals callees x t prefix,
  nth_error locals (N.to_nat (x :> N)) = Some t ->
  sourceTypes locals callees (LOCAL_GET x) prefix = Some (prefix ++ [t]).
Proof.
  intros. unfold sourceTypes, instructionTypes. rewrite H. cbn [option_map].
  rewrite <- (app_nil_r prefix) at 1. apply replace_stack_types_exact.
Qed.

Lemma local_set_types : forall locals callees x t prefix,
  nth_error locals (N.to_nat (x :> N)) = Some t ->
  sourceTypes locals callees (LOCAL_SET x) (prefix ++ [t]) = Some prefix.
Proof.
  intros. unfold sourceTypes, instructionTypes. rewrite H. cbn [option_map].
  rewrite replace_stack_types_exact, app_nil_r. reflexivity.
Qed.

Lemma call_types : forall locals (callees : CalleeEnvironment) x f prefix,
  callees (x :> N) = Some f ->
  sourceTypes locals callees (CALL x) (prefix ++ guestParamTypes f) = Some (prefix ++ guestResultTypes f).
Proof.
  intros. unfold sourceTypes, instructionTypes. rewrite H. cbn [option_map]. apply replace_stack_types_exact.
Qed.

(* For ordinary instructions, the computed types are exactly SpecTec's rule,
   including when the preserved prefix or the named local contains i64 values. *)
Lemma validated_ordinary_types : forall C i inputs outputs callees prefix,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  (exists code, StraightLineSpec.encodedOpcode i = Some code) ->
  sourceTypes (context_LOCALS C) callees i (prefix ++ inputs) = Some (prefix ++ outputs).
Proof.
  intros C i inputs outputs callees prefix Hvalid [code Hcode].
  destruct (StraightLineInstructions.encoded_opcode_cases _ _ Hcode)
    as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; inversion Hvalid; subst.
  - apply replace_stack_types_exact.
  - apply replace_stack_types_exact.
  all: assert (Hlocal : nth_error (context_LOCALS C) (N.to_nat (x :> N)) = Some t).
  all: try (
    match goal with Hbound : is_true (N.ltb _ _) |- _ =>
      apply N.ltb_lt in Hbound;
      change ((x :> N) < N.of_nat (length (context_LOCALS C)))%N in Hbound end;
    match goal with Hlookup : is_true (_ == ?t0) |- _ => apply (elimT eqP) in Hlookup end;
    destruct (nth_error (context_LOCALS C) (N.to_nat (x :> N))) as [actual | ] eqn:Hnth;
    [ match goal with Hlookup : lookup_total _ _ = ?t0 |- _ =>
        rewrite (SpecTecCalls.spec_lookup_from_nth_error _ _ _ _ _ Hnth) in Hlookup;
        now subst actual end
    | apply nth_error_None in Hnth; lia ]).
  - rewrite app_nil_r. apply local_get_types. exact Hlocal.
  - rewrite app_nil_r. apply local_set_types. exact Hlocal.
Qed.

Lemma instruction_types_height : forall locals callees i inputs outputs height,
  instructionTypes locals callees i = Some (inputs, outputs) ->
  sourceHeight callees i height =
    if length inputs <=? height then Some (height - length inputs + length outputs) else None.
Proof.
  intros locals callees i inputs outputs height H.
  destruct i; cbn [instructionTypes] in H; try discriminate.
  all: unfold option_map in H.
  all: repeat match type of H with
  | context [match ?x with _ => _ end] => destruct x eqn:?; cbn [option_map] in H; try discriminate
  end.
  all: injection H as <- <-; cbn [sourceHeight StraightLineSpec.instructionStackHeight blockResults length option_to_list];
    try (rewrite ?Heqo, ?Heqo0; cbn [length]);
    unfold guestParams, guestResults; cbn;
    try (destruct height as [ | [ | height]]; cbn; f_equal; lia).
  all: try reflexivity.
  all: try match goal with result : option valtype |- _ =>
    destruct result; destruct height; cbn; f_equal; lia end.
  rewrite length_app. cbn [length]. rewrite Nat.add_1_r. reflexivity.
Qed.

Lemma source_types_height : forall locals callees i stack next,
  sourceTypes locals callees i stack = Some next ->
  sourceHeight callees i (length stack) = Some (length next).
Proof.
  intros locals callees i stack next H. unfold sourceTypes in H.
  destruct (instructionTypes locals callees i) as [[inputs outputs] | ] eqn:Htypes; [ | discriminate].
  apply replace_stack_types_some in H as [prefix [-> ->]].
  rewrite (instruction_types_height _ _ _ _ _ _ Htypes), !length_app.
  replace (length inputs <=? length prefix + length inputs) with true by (symmetry; apply Nat.leb_le; lia).
  f_equal. lia.
Qed.

Lemma source_type_stacks_heights : forall locals callees code stack next,
  sourceTypeStacks locals callees code stack = Some next ->
  sourceHeights callees code (length stack) = Some (length next).
Proof.
  intros locals callees code. induction code as [ | i rest IH]; intros stack next H; cbn in H |- *.
  - now injection H as <-.
  - destruct (sourceTypes locals callees i stack) as [middle | ] eqn:Hstep; [ | discriminate].
    rewrite (source_types_height _ _ _ _ _ Hstep). exact (IH _ _ H).
Qed.

(* The operand height below each enclosing label, and in the innermost code at
   the offset, from the code's stack effects. *)
Fixpoint pathHeights (callees : CalleeEnvironment) (code : list instr) (labels : list LabelSite) (offset : nat)
    : option (list nat * nat) :=
  match labels with
  | [] => option_map (fun top => ([], top)) (sourceHeights callees (firstn offset code) 0)
  | s :: rest =>
      match siteEntry code s, sourceHeights callees (firstn (siteOffset s) code) 0 with
      | Some (i, body), Some height =>
          option_map (fun heights => (labelBase i height :: fst heights, snd heights))
            (pathHeights callees body rest offset)
      | _, _ => None
      end
  end.

Lemma path_heights_length : forall callees labels code offset heights top,
  pathHeights callees code labels offset = Some (heights, top) -> length heights = length labels.
Proof.
  intros callees. induction labels as [ | s rest IH]; intros code offset heights top H; cbn [pathHeights] in H.
  - destruct (sourceHeights callees (firstn offset code) 0); cbn in H; [ | discriminate]. now injection H as <- _.
  - destruct (siteEntry code s) as [[i body] | ]; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0); [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate].
    injection H as <- _. cbn [length]. f_equal. exact (IH _ _ _ _ Hin).
Qed.

(* Operand types below each enclosing label, then in the innermost code. *)
Fixpoint pathTypes (locals : list valtype) (callees : CalleeEnvironment) (code : list instr)
    (labels : list LabelSite) (offset : nat) : option (list (list valtype) * list valtype) :=
  match labels with
  | [] => option_map (fun top => ([], top)) (sourceTypeStacks locals callees (firstn offset code) [])
  | site :: rest =>
      match siteEntry code site, sourceTypeStacks locals callees (firstn (siteOffset site) code) [] with
      | Some (i, body), Some stack =>
          option_map (fun inside => (firstn (labelBase i (length stack)) stack :: fst inside, snd inside))
            (pathTypes locals callees body rest offset)
      | _, _ => None
      end
  end.

Lemma label_base_bound : forall i height, labelBase i height <= height.
Proof. intros i height. destruct i; cbn [labelBase]; lia. Qed.

Lemma path_types_heights : forall locals callees labels code offset segments top,
  pathTypes locals callees code labels offset = Some (segments, top) ->
  pathHeights callees code labels offset = Some (map (@length valtype) segments, length top).
Proof.
  intros locals callees labels. induction labels as [ | site rest IH]; intros code offset segments top H;
    cbn [pathTypes pathHeights] in *.
  - destruct (sourceTypeStacks locals callees (firstn offset code) []) as [stack | ] eqn:Hstack;
      cbn in H; [ | discriminate].
    injection H as <- <-.
    pose proof (source_type_stacks_heights _ _ _ _ _ Hstack) as Hheight.
    cbn [length] in Hheight. rewrite Hheight. reflexivity.
  - destruct (siteEntry code site) as [[i body] | ]; [ | discriminate].
    destruct (sourceTypeStacks locals callees (firstn (siteOffset site) code) []) as [stack | ] eqn:Hstack;
      [ | discriminate].
    destruct (pathTypes locals callees body rest offset) as [[nested inner] | ] eqn:Hinner;
      cbn in H; [ | discriminate].
    injection H as <- <-.
    pose proof (source_type_stacks_heights _ _ _ _ _ Hstack) as Hheight.
    cbn [length] in Hheight. rewrite Hheight, (IH _ _ _ _ Hinner).
    cbn [option_map map fst snd].
    rewrite length_firstn, Nat.min_l by apply label_base_bound. reflexivity.
Qed.
