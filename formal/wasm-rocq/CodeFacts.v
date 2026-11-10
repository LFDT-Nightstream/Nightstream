(* What SpecTec validation gives the row proofs about supported code: each
   instruction's facts at the operand height it runs at, down into nested blocks,
   and the code's stack heights. Validation stays a premise; these are its
   consequences for the supported fragment. *)
From Stdlib Require Import List ZArith NArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StraightLineValidation SpecTecCalls
  CallerSpec CallerValidation FunctionValidation CodeLayout Fragment.
Import ListNotations.
Open Scope nat_scope.

(* If control falls through the whole of code from height h, it ends at out. *)
Definition endsAt (callees : CalleeEnvironment) (code : list instr) (h out : nat) : Prop :=
  forall h', sourceHeights callees code h = Some h' -> h' = out.

(* An instruction's facts at the height it runs at, for a frame with localCount
   locals and resultCount results; `arities` holds the enclosing labels' arities,
   innermost first, the function's own label last. A block's body runs from height
   0 and ends at the block's arity. Code nothing falls through to is only
   well-formed. *)
Fixpoint instrFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (i : instr)
    (height : nat) : Prop :=
  match i with
  | BLOCK bt body => wf_instr i /\ endsAt callees body 0 (blockResults bt) /\
      (fix bodyFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (h : nat)
          (code : list instr) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrFacts localCount resultCount arities callees j h /\
             match sourceHeight callees j h with
             | Some next => bodyFacts localCount resultCount arities callees next rest | None => Forall wf_instr rest end
         end) localCount resultCount (blockResults bt :: arities) callees 0 body
  (* A loop's body ends at its results, but branches to its label keep nothing. *)
  | LOOP bt body => wf_instr i /\ endsAt callees body 0 (blockResults bt) /\
      (fix bodyFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (h : nat)
          (code : list instr) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrFacts localCount resultCount arities callees j h /\
             match sourceHeight callees j h with
             | Some next => bodyFacts localCount resultCount arities callees next rest | None => Forall wf_instr rest end
         end) localCount resultCount (0 :: arities) callees 0 body
  (* An if pops its condition; each arm runs like a block's body. *)
  | IFELSE bt t e => wf_instr i /\ (1 <= height)%nat /\
      endsAt callees t 0 (blockResults bt) /\ endsAt callees e 0 (blockResults bt) /\
      (fix bodyFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (h : nat)
          (code : list instr) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrFacts localCount resultCount arities callees j h /\
             match sourceHeight callees j h with
             | Some next => bodyFacts localCount resultCount arities callees next rest | None => Forall wf_instr rest end
         end) localCount resultCount (blockResults bt :: arities) callees 0 t /\
      (fix bodyFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (h : nat)
          (code : list instr) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrFacts localCount resultCount arities callees j h /\
             match sourceHeight callees j h with
             | Some next => bodyFacts localCount resultCount arities callees next rest | None => Forall wf_instr rest end
         end) localCount resultCount (blockResults bt :: arities) callees 0 e
  | CALL x => wf_instr i /\ exists g, callees (x :> N) = Some g /\ (guestParams g <= height)%nat
  (* An indirect call's table exists; its arguments sit below the table index. *)
  | CALL_INDIRECT x y => wf_instr i /\ (N.to_nat (x :> N) < length (envTables callees))%nat /\
      exists params results, nth_error (envTypes callees) (N.to_nat (y :> N)) = Some (i32StackType params results) /\
        (S params <= height)%nat
  (* A return consumes the results from its own block's operands. *)
  | RETURN => wf_instr i /\ (resultCount <= height)%nat
  (* A br_if's operands hold its target's results below the condition. *)
  | BR_IF l => wf_instr i /\ exists n, nth_error arities (N.to_nat (l :> N)) = Some n /\ (S n <= height)%nat
  (* A br_table's operands hold its targets' common results below the index. *)
  | BR_TABLE ls l => wf_instr i /\ exists n,
      Forall (fun target => nth_error arities (N.to_nat (target :> N)) = Some n) (l :: ls) /\ (S n <= height)%nat
  (* A br's operands hold its target's results. *)
  | BR l => wf_instr i /\ exists n, nth_error arities (N.to_nat (l :> N)) = Some n /\ (n <= height)%nat
  | NOP | UNREACHABLE => wf_instr i
  | _ => wf_instr i /\ programTyped localCount height [i]
  end.

Fixpoint codeFacts (localCount resultCount : nat) (arities : list nat) (callees : CalleeEnvironment) (height : nat)
    (code : list instr) : Prop :=
  match code with
  | [] => True
  | i :: rest => instrFacts localCount resultCount arities callees i height /\
      match sourceHeight callees i height with
      | Some next => codeFacts localCount resultCount arities callees next rest | None => Forall wf_instr rest end
  end.

Lemma block_facts : forall localCount resultCount arities callees bt body height,
  instrFacts localCount resultCount arities callees (BLOCK bt body) height <->
  wf_instr (BLOCK bt body) /\ endsAt callees body 0 (blockResults bt) /\
    codeFacts localCount resultCount (blockResults bt :: arities) callees 0 body.
Proof. reflexivity. Qed.

Lemma loop_facts : forall localCount resultCount arities callees bt body height,
  instrFacts localCount resultCount arities callees (LOOP bt body) height <->
  wf_instr (LOOP bt body) /\ endsAt callees body 0 (blockResults bt) /\
    codeFacts localCount resultCount (0 :: arities) callees 0 body.
Proof. reflexivity. Qed.

Definition i32Results (n : nat) : res_list valtype := mk_list _ (repeat valtype_I32 n).

Lemma if_facts : forall localCount resultCount arities callees bt t e height,
  instrFacts localCount resultCount arities callees (IFELSE bt t e) height <->
  wf_instr (IFELSE bt t e) /\ (1 <= height)%nat /\
    endsAt callees t 0 (blockResults bt) /\ endsAt callees e 0 (blockResults bt) /\
    codeFacts localCount resultCount (blockResults bt :: arities) callees 0 t /\
    codeFacts localCount resultCount (blockResults bt :: arities) callees 0 e.
Proof. reflexivity. Qed.

Definition resultArity (types : res_list valtype) : nat :=
  match types with mk_list types => length types end.

(* Height facts use arities, not the types occupying those positions. The
   SpecTec validation judgement retains the actual local and result types. *)
Definition contextFits (C : context) (localCount resultCount : nat) (arities : list nat)
    (callees : CalleeEnvironment) : Prop :=
  length (context_LOCALS C) = localCount /\ option_map resultArity (context_RETURN C) = Some resultCount /\
  map resultArity (LABELS C) = arities /\ calleeTypes C callees /\
  context_TYPES C = envTypes callees /\ length (context_TABLES C) = length (envTables callees).

Lemma source_heights_app : forall callees a b h,
  sourceHeights callees (a ++ b) h =
    match sourceHeights callees a h with Some next => sourceHeights callees b next | None => None end.
Proof. intros callees a; induction a; intros; cbn; [reflexivity | ]. destruct (sourceHeight callees a h); auto. Qed.

Lemma block_size : forall bt body, codeSize [BLOCK bt body] = (codeSize body + 2)%nat.
Proof. intros. rewrite code_size_cons, block_instruction_size. cbn [codeSize]. lia. Qed.

Lemma loop_size : forall bt body, codeSize [LOOP bt body] = (codeSize body + 2)%nat.
Proof. intros. rewrite code_size_cons, loop_instruction_size. cbn [codeSize]. lia. Qed.

Lemma if_size : forall bt t e, (codeSize t < codeSize [IFELSE bt t e] /\ codeSize e < codeSize [IFELSE bt t e])%nat.
Proof. intros. rewrite code_size_cons, if_instruction_size. cbn [codeSize]. destruct e; cbn [codeSize]; lia. Qed.

Lemma label_fits : forall C localCount resultCount arities callees n,
  contextFits C localCount resultCount arities callees ->
  contextFits (labelContext (repeat valtype_I32 n) C) localCount resultCount (n :: arities) callees.
Proof.
  intros C localCount resultCount arities callees n [Hlocals [Hreturn [Hlabels [Htypes [HtypeList Htables]]]]].
  split; [rewrite label_context_locals; exact Hlocals | ]. split; [exact Hreturn | ].
  split; [change (length (repeat valtype_I32 n) :: map resultArity (LABELS C) = n :: arities);
    rewrite repeat_length, Hlabels; reflexivity | ].
  split; [intros x f Hf; exact (Htypes x f Hf) | ]. split; [exact HtypeList | exact Htables].
Qed.

Lemma label_arity_lookup : forall labels index types,
  (N.to_nat index < length labels)%nat ->
  proj_list_0 valtype (lookup_total labels index) = types ->
  nth_error (map resultArity labels) (N.to_nat index) = Some (length types).
Proof.
  intros labels index types Hbound Hlookup.
  destruct (nth_error labels (N.to_nat index)) as [[actual] | ] eqn:Hnth;
    [ | apply nth_error_None in Hnth; lia].
  rewrite (spec_lookup_from_nth_error _ _ _ _ _ Hnth) in Hlookup.
  cbn [proj_list_0] in Hlookup. subst actual.
  rewrite nth_error_map.
  exact (f_equal (option_map resultArity) Hnth).
Qed.

(* One instruction other than a block: its facts and its height step. *)
Lemma flat_instruction_facts : forall C i inputs outputs localCount resultCount arities callees,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) -> wf_instr i ->
  contextFits C localCount resultCount arities callees -> Supported callees i ->
  (forall bt body, i <> BLOCK bt body) -> (forall bt t e, i <> IFELSE bt t e) -> (forall bt body, i <> LOOP bt body) ->
  (forall extra, instrFacts localCount resultCount arities callees i (extra + length inputs)) /\
  (forall extra, sourceHeight callees i (extra + length inputs) = Some (extra + length outputs)%nat \/
    sourceHeight callees i (extra + length inputs) = None).
Proof.
  intros C i inputs outputs localCount resultCount arities callees Hi Hw [Hlocals [Hreturn [Hlabels [Htypes [HtypeList Htables]]]]] Hsupport
    Hnot Hnot' Hnot''.
  assert (HlabelLookup : forall index, nth_error arities index = option_map resultArity (nth_error (LABELS C) index)).
  { intro index. rewrite <- Hlabels, nth_error_map. reflexivity. }
  clear Hlabels. rename HlabelLookup into Hlabels.
  assert (Hordinary : (exists op, encodedOpcode i = Some op) -> forall extra,
      wf_instr i /\ programTyped localCount (extra + length inputs) [i]).
  { intros Hop extra. split; [exact Hw | ].
    pose proof (proj1 (validated_instruction_height _ _ _ _ Hi Hop) extra [] I) as H.
    rewrite Hlocals in H. exact H. }
  inversion Hsupport as [ | | | | x f Hf Hp Hr | result body | l | l | ls l | | result t e | result body
    | x y params results Htype Hp Hr Hready | | ]; subst.
  - split; [exact (Hordinary ltac:(eexists; reflexivity)) | ].
    inversion Hi; subst. intros extra. left. cbn [length]. rewrite Nat.add_comm. cbn. f_equal; lia.
  - split; [exact (Hordinary ltac:(eexists; reflexivity)) | ].
    inversion Hi; subst. intros extra. left. cbn [length]. rewrite Nat.add_comm. cbn. f_equal; lia.
  - split; [exact (Hordinary ltac:(eexists; reflexivity)) | ].
    inversion Hi; subst. intros extra. left. cbn [length]. rewrite Nat.add_comm. cbn. f_equal; lia.
  - split; [exact (Hordinary ltac:(eexists; reflexivity)) | ].
    inversion Hi; subst. intros extra. left. cbn [length]. rewrite Nat.add_comm. cbn. f_equal; lia.
  - inversion Hi; subst. pose proof (Htypes _ _ Hf) as Htype.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H end.
    match goal with H : ?lhs = mk_functype _ _ |- _ =>
      assert (Hlookup : lhs = guestType f)
        by (apply spec_lookup_from_nth_error; exact Htype); rewrite Hlookup in H; inversion H; subst end.
    split; [intros extra; split; [exact Hw | exists f; split; [exact Hf | unfold guestParams; lia]] | ].
    intros extra. left. cbn [sourceHeight]. rewrite Hf. fold (guestParams f) (guestResults f).
    replace (Nat.leb (guestParams f) (extra + guestParams f)) with true by (symmetry; apply Nat.leb_le; lia).
    f_equal; lia.
  - exfalso. eapply Hnot. reflexivity.
  - inversion Hi; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H; rename H into Hlabel end.
    match type of Hlabel with _ = ?types => set (kept := types) in * end.
    match goal with H : is_true (N.ltb _ _) |- _ => apply N.ltb_lt in H; rename H into Hbound end.
    assert (Hn : nth_error arities (N.to_nat (l :> N)) = Some (length kept)).
    { rewrite Hlabels, <- nth_error_map. apply label_arity_lookup; [ | exact Hlabel].
      change (N.lt (l :> N) (N.of_nat (length (LABELS C)))) in Hbound.
      change (Nat.lt (N.to_nat (l :> N)) (@length resulttype (LABELS C))). lia. }
    split; [intros extra; split; [exact Hw | exists (length kept); split; [exact Hn | ]] | ].
    { rewrite length_app. cbn [length]. lia. }
    intros extra. left. rewrite length_app. cbn [length].
    replace (extra + (length kept + 1))%nat with (S (extra + length kept)) by lia. reflexivity.
  - inversion Hi; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H; rename H into Hlabel end.
    match type of Hlabel with _ = ?types => set (kept := types) in * end.
    match goal with H : is_true (N.ltb _ _) |- _ => apply N.ltb_lt in H; rename H into Hbound end.
    assert (Hn : nth_error arities (N.to_nat (l :> N)) = Some (length kept)).
    { rewrite Hlabels, <- nth_error_map. apply label_arity_lookup; [ | exact Hlabel].
      change (N.lt (l :> N) (N.of_nat (length (LABELS C)))) in Hbound.
      change (Nat.lt (N.to_nat (l :> N)) (@length resulttype (LABELS C))). lia. }
    split; [intros extra; split; [exact Hw | exists (length kept); split; [exact Hn | ]] | intros extra; right; reflexivity].
    assert (Hlength : forall (xs ys : list valtype), length (seq.cat xs ys) = length xs + length ys)
      by (intros xs ys; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    rewrite Hlength. lia.
  - inversion Hi; subst.
    assert (Hsize : forall (A : Type) (xs : list A), seq.size xs = length xs)
      by (intros A xs; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    assert (Hlength : forall (xs ys : list valtype), length (seq.cat xs ys) = (length xs + length ys)%nat)
      by (intros xs ys; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    (* Each target's label holds as many results as the operands the table keeps. *)
    match goal with Hs : Resulttype_sub (mk_list _ ?ts) _ |- _ => set (kept := ts) in * end.
    match goal with Hs : Resulttype_sub _ (@lookup_total _ ?inst _ _) |- _ =>
      assert (Htarget : forall target : labelidx, is_true ((target :> N) <? (|LABELS C|))%BN ->
        Resulttype_sub (mk_list _ kept) (@lookup_total _ inst (LABELS C) (target :> N)) ->
        nth_error arities (N.to_nat (target :> N)) = Some (length kept)) end.
    { intros target Hb Hs. apply N.ltb_lt in Hb.
      rewrite Hsize in Hb.
      destruct (nth_error (LABELS C) (N.to_nat (target :> N))) as [[actual] | ] eqn:Hm;
        [ | apply nth_error_None in Hm; lia].
      erewrite spec_lookup_from_nth_error in Hs by exact Hm.
      pose proof (result_sub_length _ _ Hs) as Hlen.
      rewrite Hlabels, Hm. cbn [option_map resultArity]. now rewrite Hlen. }
    split; [intros extra; split; [exact Hw | exists (length kept); split] | intros extra; right; reflexivity].
    + constructor; [apply (Htarget l); assumption | ].
      apply Forall_forall. intros target Hin.
      apply (Htarget target).
      * match goal with HF : Forall (fun _ => is_true _) ls |- _ => exact (proj1 (Forall_forall _ _) HF _ Hin) end.
      * match goal with HF : Forall (fun _ => Resulttype_sub _ _) ls |- _ =>
          exact (proj1 (Forall_forall _ _) HF _ Hin) end.
    + rewrite !Hlength. cbn [length]. lia.
  - inversion Hi; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H end.
    match goal with H : context_RETURN C = Some _ |- _ => rewrite H in Hreturn end.
    cbn [option_map resultArity] in Hreturn. injection Hreturn as Hresults.
    split; [intros extra; split; [exact Hw | ] | intros extra; right; reflexivity].
    assert (Hlength : forall (xs ys : list valtype), length (seq.cat xs ys) = (length xs + length ys)%nat)
      by (intros xs ys; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    rewrite Hlength. lia.
  - exfalso. eapply Hnot'. reflexivity.
  - exfalso. eapply Hnot''. reflexivity.
  - inversion Hi; subst.
    repeat match goal with H : is_true (N.ltb _ _) |- _ => apply N.ltb_lt in H end.
    match goal with H : is_true (_ == mk_functype _ _), H' : is_true (_ == mk_functype _ _) |- _ => idtac | _ => idtac end.
    match goal with H : is_true (@lookup_total _ _ (context_TYPES C) _ == _) |- _ =>
      apply (elimT eqP) in H; rewrite HtypeList in H;
      erewrite spec_lookup_from_nth_error in H by exact Htype; inversion H; subst end.
    assert (Hsize : forall (A : Type) (xs : list A), seq.size xs = length xs)
      by (intros A xs; induction xs as [ | z xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    assert (Hlength : forall (xs ys : list valtype), length (seq.cat xs ys) = (length xs + length ys)%nat)
      by (intros xs ys; induction xs as [ | z xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    split.
    + intros extra. split; [exact Hw | ]. split.
      * match goal with H : context [seq.size (context_TABLES C)], H' : is_true (_ == mk_tabletype _ _) |- _ =>
          rewrite Hsize in H end.
        match goal with |- context [@coerce tableidx N ?c x] => change (@coerce tableidx N c x) with (@coerce idx N c x) end.
        lia.
      * exists params, results. split; [exact Htype | ]. rewrite Hlength, repeat_length. cbn [length]. lia.
    + intros extra. left. cbn [sourceHeight]. rewrite Htype. cbn.
      rewrite Hlength, !repeat_length. cbn [length].
      replace (extra + (params + 1))%nat with (S (extra + params)) by lia.
      replace (params <=? extra + params) with true by (symmetry; apply Nat.leb_le; lia).
      f_equal. lia.
  - split; [intros extra; exact Hw | ]. inversion Hi; subst. intros extra. left. reflexivity.
  - split; [intros extra; exact Hw | intros extra; right; reflexivity].
Qed.

Lemma instr_structured_cases : forall i, (exists bt body, i = BLOCK bt body) \/ (exists bt t e, i = IFELSE bt t e) \/
  (exists bt body, i = LOOP bt body) \/
  ((forall bt body, i <> BLOCK bt body) /\ (forall bt t e, i <> IFELSE bt t e) /\ (forall bt body, i <> LOOP bt body)).
Proof.
  destruct i; try (right; right; right; split; [ | split]; intros; discriminate);
    [left | right; right; left | right; left]; eauto.
Qed.

Lemma instr_facts_wf : forall localCount resultCount arities callees i h, instrFacts localCount resultCount arities callees i h -> wf_instr i.
Proof. intros localCount resultCount arities callees i h H. destruct i; try exact (proj1 H); exact H. Qed.

Lemma code_facts_wf : forall localCount resultCount arities callees code h, codeFacts localCount resultCount arities callees h code -> Forall wf_instr code.
Proof.
  intros localCount resultCount arities callees code. induction code as [ | i rest IH]; intros h H; [constructor | ].
  destruct H as [Hi Hrest]. constructor; [eapply instr_facts_wf; exact Hi | ].
  destruct (sourceHeight callees i h) as [next | ]; [eapply IH; exact Hrest | exact Hrest].
Qed.

Lemma code_validation_facts : forall n C code ft, Instrs_ok C code ft -> (codeSize code < n)%nat ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
  forall localCount resultCount arities callees, contextFits C localCount resultCount arities callees -> supportedCode callees code ->
    (forall extra suffix, codeFacts localCount resultCount arities callees (extra + length outputs) suffix ->
      codeFacts localCount resultCount arities callees (extra + length inputs) (code ++ suffix)) /\
    (forall extra, endsAt callees code (extra + length inputs) (extra + length outputs)%nat) /\
    Forall wf_instr code
  end.
Proof.
  induction n as [ | n IHn]; intros C code ft Hok Hsize; [lia | ]. revert Hsize.
  induction Hok as [C HC | C i inputs outputs Hi HC Hw | C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb
    | C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw | C code prefix inputs outputs Hv IH HC Hw].
  - intros Hsize localCount resultCount arities callees _ _. split; [intros; assumption | ].
    split; [intros extra h' H; injection H as <-; reflexivity | constructor].
  - intros Hsize localCount resultCount arities callees Hfits Hsupport.
    pose proof (Forall_inv Hsupport) as Hsup.
    destruct (instr_structured_cases i) as [[bt [body ->]] | [[bt [t [e ->]]] | [[bt [body ->]] | [Hnot [Hnot' Hnot'']]]]].
    + inversion Hsup as [ | | | | | result body' Hbody' Hresult | | | | | | |  | | ]; subst. inversion Hi; subst.
      match goal with Hbt : Blocktype_ok _ _ _ |- _ => inversion Hbt; subst end.
      rewrite block_size in Hsize.
      match goal with Hbody : Instrs_ok _ body _ |- _ => pose proof (IHn _ _ _ Hbody ltac:(lia)) as Hinner end.
      (* The block's label holds its results: none, or one i32. *)
      assert (Hlabel : option_to_list result = repeat valtype_I32 (blockResults (_RESULT result)))
        by (destruct Hresult as [-> | ->]; reflexivity).
      cbn [option_to_list] in Hinner. rewrite Hlabel in Hinner. cbn [length] in Hinner |- *. rewrite repeat_length in Hinner.
      destruct (Hinner localCount resultCount (blockResults (_RESULT result) :: arities) callees
        (label_fits _ _ _ _ _ _ Hfits) ltac:(assumption)) as [Hbody [Hheights _]].
      split; [ | split; [ | constructor; [exact Hw | constructor]]].
      * intros extra suffix Hsuffix. cbn [app codeFacts sourceHeight]. cbn [length option_to_list] in Hsuffix |- *.
        rewrite Hlabel, repeat_length in Hsuffix. rewrite Nat.add_0_r.
        split; [ | exact Hsuffix]. apply block_facts. split; [assumption | ]. split.
        -- exact (Hheights 0%nat).
        -- pose proof (Hbody 0%nat [] I) as Hb. rewrite app_nil_r in Hb. exact Hb.
      * intros extra h' H. cbn [sourceHeights sourceHeight] in H. injection H as <-.
        cbn [length option_to_list]. rewrite Hlabel, repeat_length. destruct result; cbn; lia.
    + (* An if: each arm like a block's body, above the condition. *)
      inversion Hsup as [ | | | | | | | | | | result t' e' Ht He Hresult | |  | | ]; subst. inversion Hi; subst.
      match goal with Hbt : Blocktype_ok _ _ _ |- _ => inversion Hbt; subst end.
      destruct (if_size (_RESULT result) t e) as [Hst Hse].
      assert (Hlabel : option_to_list result = repeat valtype_I32 (blockResults (_RESULT result)))
        by (destruct Hresult as [-> | ->]; reflexivity).
      match goal with Hthen : Instrs_ok _ t _, Helse : Instrs_ok _ e _ |- _ =>
        pose proof (IHn _ _ _ Hthen ltac:(lia)) as Hinner1; pose proof (IHn _ _ _ Helse ltac:(lia)) as Hinner2 end.
      cbn [option_to_list] in Hinner1, Hinner2. rewrite Hlabel in Hinner1, Hinner2.
      cbn [length] in Hinner1, Hinner2. rewrite repeat_length in Hinner1, Hinner2.
      destruct (Hinner1 localCount resultCount (blockResults (_RESULT result) :: arities) callees
        (label_fits _ _ _ _ _ _ Hfits) Ht) as [Hbody1 [Hheights1 _]].
      destruct (Hinner2 localCount resultCount (blockResults (_RESULT result) :: arities) callees
        (label_fits _ _ _ _ _ _ Hfits) He) as [Hbody2 [Hheights2 _]].
      assert (Hin : length (seq.cat [] [valtype_I32]) = 1%nat) by reflexivity.
      split; [ | split; [ | constructor; [exact Hw | constructor]]].
      * intros extra suffix Hsuffix. rewrite Hin. cbn [app codeFacts]. rewrite Nat.add_1_r.
        cbn [length option_to_list] in Hsuffix. rewrite Hlabel, repeat_length in Hsuffix.
        split.
        -- apply if_facts. split; [assumption | ]. split; [lia | ].
           split; [exact (Hheights1 0%nat) | ]. split; [exact (Hheights2 0%nat) | ].
           pose proof (Hbody1 0%nat [] I) as Hb1. pose proof (Hbody2 0%nat [] I) as Hb2.
           rewrite app_nil_r in Hb1, Hb2. split; assumption.
        -- exact Hsuffix.
      * intros extra h' H. rewrite Hin, Nat.add_1_r in H. cbn [sourceHeights sourceHeight] in H.
        injection H as <-. cbn [length option_to_list]. rewrite Hlabel, repeat_length. destruct result; cbn; lia.
    + (* A loop: its body like a block's, under a label that keeps nothing. *)
      inversion Hsup as [ | | | | | | | | | | | result body' Hbody' Hresult |  | | ]; subst. inversion Hi; subst.
      match goal with Hbt : Blocktype_ok _ _ _ |- _ => inversion Hbt; subst end.
      rewrite loop_size in Hsize.
      match goal with Hbody : Instrs_ok _ body _ |- _ => pose proof (IHn _ _ _ Hbody ltac:(lia)) as Hinner end.
      assert (Hlabel : option_to_list result = repeat valtype_I32 (blockResults (_RESULT result)))
        by (destruct Hresult as [-> | ->]; reflexivity).
      cbn [option_to_list] in Hinner. rewrite Hlabel in Hinner. cbn [length] in Hinner |- *. rewrite repeat_length in Hinner.
      destruct (Hinner localCount resultCount (0 :: arities) callees
        (label_fits _ _ _ _ _ 0 Hfits) ltac:(assumption)) as [Hbody [Hheights _]].
      split; [ | split; [ | constructor; [exact Hw | constructor]]].
      * intros extra suffix Hsuffix. cbn [app codeFacts sourceHeight]. cbn [length option_to_list] in Hsuffix |- *.
        rewrite Hlabel, repeat_length in Hsuffix. rewrite Nat.add_0_r.
        split; [ | exact Hsuffix]. apply loop_facts. split; [assumption | ]. split.
        -- exact (Hheights 0%nat).
        -- pose proof (Hbody 0%nat [] I) as Hb. rewrite app_nil_r in Hb. exact Hb.
      * intros extra h' H. cbn [sourceHeights sourceHeight] in H. injection H as <-.
        cbn [length option_to_list]. rewrite Hlabel, repeat_length. destruct result; cbn; lia.
    + destruct (flat_instruction_facts C i inputs outputs localCount resultCount arities callees Hi Hw Hfits Hsup
        Hnot Hnot' Hnot'') as [Hfact Hheight].
      split; [ | split; [ | constructor; [exact Hw | constructor]]].
      * intros extra suffix Hsuffix. cbn [app codeFacts]. split; [exact (Hfact extra) | ].
        destruct (Hheight extra) as [-> | ->]; [exact Hsuffix | exact (code_facts_wf _ _ _ _ _ _ Hsuffix)].
      * intros extra h' H. cbn [sourceHeights] in H.
        destruct (Hheight extra) as [Heq | Heq]; rewrite Heq in H; [injection H as <-; reflexivity | discriminate].
  - intros Hsize localCount resultCount arities callees Hfits Hsupport.
    rewrite code_size_app in Hsize. apply Forall_app in Hsupport.
    destruct (IHa ltac:(lia) localCount resultCount arities callees Hfits (proj1 Hsupport)) as [Hfa [Hha Hwfa]].
    destruct (IHb ltac:(lia) localCount resultCount arities callees Hfits (proj2 Hsupport)) as [Hfb [Hhb Hwfb]].
    split; [ | split; [ | apply Forall_app; split; assumption]].
    + intros extra suffix Hsuffix. rewrite <- app_assoc. apply Hfa. now apply Hfb.
    + intros extra h' H. rewrite source_heights_app in H.
      destruct (sourceHeights callees a (extra + length inputs)) as [mid | ] eqn:Hmid; [ | discriminate].
      rewrite (Hha extra mid Hmid) in H. exact (Hhb extra h' H).
  - intros Hsize localCount resultCount arities callees Hfits Hsupport.
    rewrite (result_sub_length _ _ Hin), <- (result_sub_length _ _ Hout).
    exact (IH Hsize localCount resultCount arities callees Hfits Hsupport).
  - intros Hsize localCount resultCount arities callees Hfits Hsupport.
    destruct (IH Hsize localCount resultCount arities callees Hfits Hsupport) as [Hfacts [Hheight Hwf]]. rewrite !length_app.
    split; [ | split; [ | exact Hwf]].
    + intros extra suffix Hsuffix. rewrite Nat.add_assoc in Hsuffix |- *. now apply Hfacts.
    + intro extra. rewrite !Nat.add_assoc. apply Hheight.
Qed.

(* Walking along code: facts, heights, balance and support hold for what remains. *)
Lemma code_facts_skip : forall localCount resultCount arities callees code k h h',
  codeFacts localCount resultCount arities callees h code -> sourceHeights callees (firstn k code) h = Some h' ->
  codeFacts localCount resultCount arities callees h' (skipn k code).
Proof.
  intros localCount resultCount arities callees code. induction code as [ | i rest IH]; intros k h h' Hfacts Hheights.
  - destruct k; cbn in Hheights; injection Hheights as <-; exact I.
  - destruct k as [ | k]; cbn in Hheights |- *; [injection Hheights as <-; exact Hfacts | ].
    destruct Hfacts as [_ Hrest]. destruct (sourceHeight callees i h) as [next | ]; [ | discriminate].
    exact (IH k next h' Hrest Hheights).
Qed.

Lemma code_branches_skip : forall callees exits code k h h',
  codeBranches callees exits code h -> sourceHeights callees (firstn k code) h = Some h' ->
  codeBranches callees exits (skipn k code) h'.
Proof.
  intros callees exits code. induction code as [ | i rest IH]; intros k h h' Hbranches Hheights.
  - destruct k; cbn in Hheights; injection Hheights as <-; exact I.
  - destruct k as [ | k]; cbn in Hheights |- *; [injection Hheights as <-; exact Hbranches | ].
    destruct Hbranches as [_ Hrest]. destruct (sourceHeight callees i h) as [next | ]; [ | discriminate].
    exact (IH k next h' Hrest Hheights).
Qed.

Lemma block_branches : forall callees bt body height exits,
  instrBranches callees (BLOCK bt body) height exits <->
  codeBranches callees (S exits) body 0.
Proof. reflexivity. Qed.

Lemma supported_skip : forall callees code k, supportedCode callees code -> supportedCode callees (skipn k code).
Proof.
  intros callees code k H. unfold supportedCode in *. apply Forall_forall. intros x Hx.
  apply (proj1 (Forall_forall _ _) H). rewrite <- (firstn_skipn k code). apply in_or_app. right. exact Hx.
Qed.

