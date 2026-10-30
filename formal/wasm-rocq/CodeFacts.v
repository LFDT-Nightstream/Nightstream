(* What SpecTec validation gives the row proofs about supported code: each
   instruction's facts at the operand height it runs at, down into nested blocks,
   and the code's stack heights. Validation stays a premise; these are its
   consequences for the i32 fragment. *)
From Stdlib Require Import List ZArith Lia.
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
  | CALL x => wf_instr i /\ exists g, callees (x :> N) = Some g /\ (guestParams g <= height)%nat
  (* A return consumes the results from its own block's operands. *)
  | RETURN => wf_instr i /\ (resultCount <= height)%nat
  (* A br_if's operands hold its target's results below the condition. *)
  | BR_IF l => wf_instr i /\ exists n, nth_error arities (N.to_nat (l :> N)) = Some n /\ (S n <= height)%nat
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

Definition i32Results (n : nat) : res_list valtype := mk_list _ (repeat valtype_I32 n).

(* The typing context matches the frame's locals, results, labels and callees. *)
Definition contextFits (C : context) (localCount resultCount : nat) (arities : list nat)
    (callees : CalleeEnvironment) : Prop :=
  context_LOCALS C = repeat valtype_I32 localCount /\ context_RETURN C = Some (i32Results resultCount) /\
  LABELS C = map i32Results arities /\ calleeTypes C callees.

Lemma source_heights_app : forall callees a b h,
  sourceHeights callees (a ++ b) h =
    match sourceHeights callees a h with Some next => sourceHeights callees b next | None => None end.
Proof. intros callees a; induction a; intros; cbn; [reflexivity | ]. destruct (sourceHeight callees a h); auto. Qed.

Lemma block_size : forall bt body, codeSize [BLOCK bt body] = (codeSize body + 2)%nat.
Proof.
  intros. rewrite code_size_cons. unfold instructionSize. rewrite flatten_block. cbn [length].
  rewrite length_app, block_code_length. cbn [length codeSize flattenCode flat_map]. lia.
Qed.

Lemma label_fits : forall C localCount resultCount arities callees n,
  contextFits C localCount resultCount arities callees ->
  contextFits (labelContext (repeat valtype_I32 n) C) localCount resultCount (n :: arities) callees.
Proof.
  intros C localCount resultCount arities callees n [Hlocals [Hreturn [Hlabels Htypes]]].
  split; [rewrite label_context_locals; exact Hlocals | ]. split; [exact Hreturn | ].
  split; [cbn; rewrite Hlabels; reflexivity | ].
  intros x f Hf. exact (Htypes x f Hf).
Qed.

(* One instruction other than a block: its facts and its height step. *)
Lemma flat_instruction_facts : forall C i inputs outputs localCount resultCount arities callees,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) -> wf_instr i ->
  contextFits C localCount resultCount arities callees -> Supported callees i ->
  (forall bt body, i <> BLOCK bt body) ->
  (forall extra, instrFacts localCount resultCount arities callees i (extra + length inputs)) /\
  (forall extra, sourceHeight callees i (extra + length inputs) = Some (extra + length outputs)%nat \/
    sourceHeight callees i (extra + length inputs) = None).
Proof.
  intros C i inputs outputs localCount resultCount arities callees Hi Hw [Hlocals [Hreturn [Hlabels Htypes]]] Hsupport Hnot.
  assert (Hordinary : (exists op, encodedOpcode i = Some op) -> forall extra,
      wf_instr i /\ programTyped localCount (extra + length inputs) [i]).
  { intros Hop extra. split; [exact Hw | ].
    pose proof (proj1 (validated_instruction_height _ _ _ _ Hi Hop) extra [] I) as H.
    rewrite Hlocals, repeat_length in H. exact H. }
  inversion Hsupport as [ | | | | x f Hf Hp Hr | result body | l | ]; subst.
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
      assert (Hlookup : lhs = i32StackType (guestParams f) (guestResults f))
        by (apply spec_lookup_from_nth_error; exact Htype); rewrite Hlookup in H; inversion H; subst end.
    split; [intros extra; split; [exact Hw | exists f; split; [exact Hf | rewrite repeat_length; lia]] | ].
    intros extra. left. cbn [sourceHeight]. rewrite Hf, !repeat_length.
    replace (Nat.leb (guestParams f) (extra + guestParams f)) with true by (symmetry; apply Nat.leb_le; lia).
    f_equal; lia.
  - exfalso. eapply Hnot. reflexivity.
  - inversion Hi; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H; rename H into Hlabel end.
    match goal with H : is_true (N.ltb _ _) |- _ => apply N.ltb_lt in H; rename H into Hbound end.
    rewrite Hlabels in Hlabel, Hbound.
    assert (Hsize : forall (xs : list (res_list valtype)), seq.size xs = length xs)
      by (induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    rewrite Hsize, length_map in Hbound.
    destruct (nth_error arities (N.to_nat (l :> N))) as [n | ] eqn:Hn; [ | apply nth_error_None in Hn; lia].
    erewrite spec_lookup_from_nth_error in Hlabel; [ | rewrite nth_error_map, Hn; reflexivity].
    cbn [i32Results proj_list_0] in Hlabel. subst.
    split; [intros extra; split; [exact Hw | exists n; split; [exact Hn | ]] | ].
    { rewrite length_app, repeat_length. cbn [length]. lia. }
    intros extra. left. rewrite length_app. cbn [length].
    match goal with |- sourceHeight _ _ (extra + (length ?ts + 1)) = _ =>
      replace (extra + (length ts + 1))%nat with (S (extra + length ts)) by lia end. reflexivity.
  - inversion Hi; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H end.
    rewrite Hreturn in *. match goal with H : Some _ = Some _ |- _ => injection H as Hresults end.
    split; [intros extra; split; [exact Hw | ] | intros extra; right; reflexivity].
    assert (Hlength : forall (xs ys : list valtype), length (seq.cat xs ys) = (length xs + length ys)%nat)
      by (intros xs ys; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]).
    rewrite Hlength, <- Hresults, repeat_length. lia.
Qed.

Lemma instr_block_cases : forall i, (exists bt body, i = BLOCK bt body) \/ (forall bt body, i <> BLOCK bt body).
Proof. destruct i; try (right; intros; discriminate). left; eauto. Qed.

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
    destruct (instr_block_cases i) as [[bt [body ->]] | Hnot].
    + inversion Hsup as [ | | | | | result body' Hbody' Hresult | | ]; subst. inversion Hi; subst.
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
    + destruct (flat_instruction_facts C i inputs outputs localCount resultCount arities callees Hi Hw Hfits Hsup Hnot)
        as [Hfact Hheight].
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

