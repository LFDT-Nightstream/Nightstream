(* SpecTec validation determines the computed operand types. BOT is used by
   validation for unreachable code; it is not the type of a decoded value. *)
From Stdlib Require Import List NArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout Fragment StackTypes StackSimulation.
Import ListNotations.
Open Scope nat_scope.

Definition concreteTypes (types : list valtype) : Prop := Forall (fun t => t <> BOT) types.

Lemma concrete_sub_equal : forall types other,
  concreteTypes types -> Resulttype_sub (mk_list _ types) (mk_list _ other) -> types = other.
Proof.
  intros types other Htypes Hsub. inversion Hsub as [xs ys _ Hpair]; subst.
  clear Hsub. revert Htypes. induction Hpair; intros Htypes; [reflexivity | ]. inversion Htypes; subst.
  inversion H; subst; [f_equal; auto | contradiction].
Qed.

Lemma instruction_outputs_concrete : forall locals (callees : CalleeEnvironment) i inputs outputs,
  concreteTypes locals ->
  (forall x f, callees x = Some f -> concreteTypes (guestResultTypes f)) ->
  Supported callees i -> instructionTypes locals callees i = Some (inputs, outputs) ->
  concreteTypes outputs.
Proof.
  intros locals callees i inputs outputs Hlocals Hcallees Hsupported Hsignature.
  inversion Hsupported as [ | | x | x | x f Hf Hp Hr | result body Hbody Hresult | l | l | ls l | | result t e Ht He Hresult | result body Hbody Hresult
    | x y params results Htype Hp Hr Hready | | ]; subst;
    cbn [instructionTypes] in Hsignature; try discriminate.
  - injection Hsignature as <- <-. repeat constructor; discriminate.
  - injection Hsignature as <- <-. repeat constructor; discriminate.
  - destruct (nth_error locals (N.to_nat (x :> N))) as [t | ] eqn:Hlocal; [ | discriminate].
    injection Hsignature as <- <-. constructor; [ | constructor].
    exact (proj1 (Forall_forall _ _) Hlocals _ (nth_error_In _ _ Hlocal)).
  - destruct (nth_error locals (N.to_nat (x :> N))); [ | discriminate].
    injection Hsignature as <- <-. constructor.
  - rewrite Hf in Hsignature. injection Hsignature as <- <-. exact (Hcallees _ _ Hf).
  - injection Hsignature as <- <-. destruct Hresult as [-> | ->]; repeat constructor; discriminate.
  - injection Hsignature as <- <-. constructor.
  - injection Hsignature as <- <-. destruct Hresult as [-> | ->]; repeat constructor; discriminate.
  - injection Hsignature as <- <-. destruct Hresult as [-> | ->]; repeat constructor; discriminate.
  - rewrite Htype in Hsignature. injection Hsignature as <- <-.
    apply Forall_forall. intros t Ht. apply repeat_spec in Ht. subst t. discriminate.
  - injection Hsignature as <- <-. constructor.
Qed.

Lemma source_types_concrete : forall locals (callees : CalleeEnvironment) i stack next,
  concreteTypes locals ->
  (forall x f, callees x = Some f -> concreteTypes (guestResultTypes f)) ->
  Supported callees i -> concreteTypes stack ->
  sourceTypes locals callees i stack = Some next -> concreteTypes next.
Proof.
  intros locals callees i stack next Hlocals Hcallees Hsupported Hstack Hstep.
  unfold sourceTypes in Hstep.
  destruct (instructionTypes locals callees i) as [[inputs outputs] | ] eqn:Hsignature; [ | discriminate].
  apply replace_stack_types_some in Hstep as [prefix [-> ->]].
  apply Forall_app in Hstack as [Hprefix _]. apply Forall_app. split; [exact Hprefix | ].
  exact (instruction_outputs_concrete locals callees i inputs outputs Hlocals Hcallees Hsupported Hsignature).
Qed.

(* A supported instruction either has no fallthrough, or its computed effect
   is precisely the type rule, under any preserved operand prefix. *)
Lemma validated_instruction_types : forall C i inputs outputs callees prefix,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  Supported callees i -> calleeTypes C callees -> context_TYPES C = envTypes callees ->
  sourceTypes (context_LOCALS C) callees i (prefix ++ inputs) = None \/
  sourceTypes (context_LOCALS C) callees i (prefix ++ inputs) = Some (prefix ++ outputs).
Proof.
  intros C i inputs outputs callees prefix Hvalid Hsupported Hcallees Htypes.
  inversion Hsupported as [ | | x | x | x f Hf Hp Hr | result body Hbody Hresult | l | l | ls l | | result t e Ht He Hresult | result body Hbody Hresult
    | x y params results Htype Hp Hr Hready | | ]; subst.
  1-4: right; eapply validated_ordinary_types; [exact Hvalid | eexists; reflexivity].
  - right. inversion Hvalid; subst.
    match goal with H : is_true (_ == _) |- _ => apply (elimT eqP) in H end.
    match goal with H : ?lhs = mk_functype _ _ |- _ =>
      assert (Hlookup : lhs = guestType f)
        by (apply spec_lookup_from_nth_error; exact (Hcallees _ _ Hf));
      rewrite Hlookup in H; inversion H; subst end.
    apply call_types. exact Hf.
  - right. inversion Hvalid; subst.
    match goal with H : Blocktype_ok _ _ _ |- _ => inversion H; subst end.
    apply replace_stack_types_exact.
  - right. inversion Hvalid; subst.
    match goal with |- _ = Some (prefix ++ ?types) =>
      change (sourceTypes (context_LOCALS C) callees (BR_IF l) (prefix ++ (types ++ [valtype_I32])) =
        Some (prefix ++ types)) end.
    rewrite app_assoc. unfold sourceTypes, instructionTypes.
    rewrite replace_stack_types_exact, app_nil_r. reflexivity.
  - left. reflexivity.
  - left. reflexivity.
  - left. reflexivity.
  - right. inversion Hvalid; subst.
    match goal with H : Blocktype_ok _ _ _ |- _ => inversion H; subst end.
    apply replace_stack_types_exact.
  - right. inversion Hvalid; subst.
    match goal with H : Blocktype_ok _ _ _ |- _ => inversion H; subst end.
    apply replace_stack_types_exact.
  - right. inversion Hvalid; subst.
    match goal with H : is_true (@lookup_total _ _ (context_TYPES C) _ == _) |- _ =>
      apply (elimT eqP) in H; rewrite Htypes in H;
      erewrite spec_lookup_from_nth_error in H by exact Htype; inversion H; subst end.
    unfold sourceTypes, instructionTypes. rewrite Htype. apply replace_stack_types_exact.
  - right. inversion Hvalid; subst. apply replace_stack_types_exact.
  - left. reflexivity.
Qed.

Lemma empty_concrete_types : forall C code ft,
  Instrs_ok C code ft -> code = [] ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
    concreteTypes inputs -> inputs = outputs
  end.
Proof.
  intros C code ft Hvalid.
  induction Hvalid as [C HC | C i inputs outputs Hi HC Hw | C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb
    | C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw | C code below inputs outputs Hv IH HC Hw];
    intros Hcode Hconcrete; try discriminate.
  - reflexivity.
  - apply app_eq_nil in Hcode as [Hca Hcb].
    rewrite (IHa Hca Hconcrete) in Hconcrete |- *. exact (IHb Hcb Hconcrete).
  - pose proof (concrete_sub_equal _ _ Hconcrete Hin) as Heq. subst inputs'.
    pose proof (IH Hcode Hconcrete) as Heq. subst outputs.
    exact (concrete_sub_equal _ _ Hconcrete Hout).
  - f_equal. apply IH; [exact Hcode | exact (proj2 (proj1 (Forall_app _ _ _) Hconcrete))].
Qed.

Lemma result_sub_reflexive : forall types, Resulttype_sub (mk_list _ types) (mk_list _ types).
Proof.
  intros types. constructor; [apply (introT eqP); reflexivity | ].
  induction types; constructor; [apply refl | exact IHtypes].
Qed.

(* On a reachable stack, framing and subtyping still expose an actual type rule
   for the next instruction and a validation judgement for its continuation. *)
Lemma validation_concrete_head : forall C code ft,
  Instrs_ok C code ft ->
  match ft with mk_functype (mk_list inputs) (mk_list outputs) =>
    forall i rest, code = i :: rest -> concreteTypes inputs ->
    exists prefix consumed produced,
      inputs = prefix ++ consumed /\
      Instr_ok C i (mk_functype (mk_list _ consumed) (mk_list _ produced)) /\
      Instrs_ok C rest (mk_functype (mk_list _ (prefix ++ produced)) (mk_list _ outputs))
  end.
Proof.
  intros C code ft Hvalid.
  induction Hvalid as [C HC | C i inputs outputs Hi HC Hw | C a b inputs outputs middle Ha IHa Hb IHb HC Hwa Hwb
    | C code inputs' outputs' inputs outputs Hv IH Hin Hout HC Hw | C code below inputs outputs Hv IH HC Hw];
    intros instruction rest Hcode Hconcrete; try discriminate.
  - injection Hcode as <- <-. exists [], inputs, outputs. split; [reflexivity | ]. split; [exact Hi | ].
    change (Instrs_ok C [] (mk_functype (mk_list _ outputs) (mk_list _ outputs))).
    replace outputs with (outputs ++ []) at 1 2 by apply app_nil_r.
    apply Instrs_ok__frame; [apply empty; exact HC | exact HC | constructor].
  - destruct a as [ | first tail].
    + pose proof (empty_concrete_types _ _ _ Ha Logic.eq_refl Hconcrete) as Heq. subst middle.
      apply IHb; assumption.
    + cbn [app] in Hcode. injection Hcode as <- <-.
      destruct (IHa first tail Logic.eq_refl Hconcrete) as [prefix [consumed [produced [Hinput [Hhead Htail]]]]].
      exists prefix, consumed, produced. split; [exact Hinput | ]. split; [exact Hhead | ].
      eapply res_seq; [exact Htail | exact Hb | exact HC | exact (Forall_inv_tail Hwa) | exact Hwb].
  - pose proof (concrete_sub_equal _ _ Hconcrete Hin) as Heq. subst inputs'.
    destruct (IH instruction rest Hcode Hconcrete) as [prefix [consumed [produced [Hinput [Hhead Htail]]]]].
    exists prefix, consumed, produced. split; [exact Hinput | ]. split; [exact Hhead | ].
    subst code. eapply sub; [exact Htail | apply result_sub_reflexive | exact Hout | exact HC |
      exact (Forall_inv_tail Hw)].
  - destruct (IH instruction rest Hcode (proj2 (proj1 (Forall_app _ _ _) Hconcrete)))
      as [prefix [consumed [produced [Hinput [Hhead Htail]]]]].
    exists (below ++ prefix), consumed, produced. split; [now rewrite <- app_assoc, <- Hinput | ].
    split; [exact Hhead | ]. rewrite <- app_assoc. subst code.
    apply Instrs_ok__frame; [exact Htail | exact HC | exact (Forall_inv_tail Hw)].
Qed.

Lemma validated_type_step : forall C callees i rest inputs outputs next,
  Instrs_ok C (i :: rest) (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  concreteTypes inputs -> Supported callees i ->
  calleeTypes C callees -> context_TYPES C = envTypes callees ->
  sourceTypes (context_LOCALS C) callees i inputs = Some next ->
  Instrs_ok C rest (mk_functype (mk_list _ next) (mk_list _ outputs)).
Proof.
  intros C callees i rest inputs outputs next Hvalid Hconcrete Hsupported Hcallees Htypes Hstep.
  destruct (validation_concrete_head _ _ _ Hvalid i rest Logic.eq_refl Hconcrete)
    as [prefix [consumed [produced [-> [Hhead Htail]]]]].
  destruct (validated_instruction_types _ _ _ _ _ prefix Hhead Hsupported Hcallees Htypes)
    as [Hnone | Hsome]; [congruence | ].
  rewrite Hstep in Hsome. injection Hsome as ->. exact Htail.
Qed.

(* The computed stack at a reachable offset is also a valid input stack for
   the remaining source code, in the original SpecTec context. *)
Lemma validated_type_prefix : forall C callees offset code inputs outputs middle,
  Instrs_ok C code (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  supportedCode callees code -> calleeTypes C callees -> context_TYPES C = envTypes callees ->
  concreteTypes (context_LOCALS C) ->
  (forall x f, callees x = Some f -> concreteTypes (guestResultTypes f)) -> concreteTypes inputs ->
  sourceTypeStacks (context_LOCALS C) callees (firstn offset code) inputs = Some middle ->
  Instrs_ok C (skipn offset code) (mk_functype (mk_list _ middle) (mk_list _ outputs)) /\
  concreteTypes middle.
Proof.
  intros C callees offset. induction offset as [ | offset IH];
    intros code inputs outputs middle Hvalid Hsupported Hcallees Htypes Hlocals Hresults Hinputs Hcomputed.
  - cbn in Hcomputed |- *. injection Hcomputed as <-. auto.
  - destruct code as [ | i rest].
    + cbn in Hcomputed |- *. injection Hcomputed as <-. auto.
    + cbn [firstn sourceTypeStacks] in Hcomputed. cbn [skipn].
      destruct (sourceTypes (context_LOCALS C) callees i inputs) as [next | ] eqn:Hstep; [ | discriminate].
      inversion Hsupported as [ | ? ? Hi Hrest]; subst.
      apply (IH rest next outputs middle); try assumption.
      * exact (validated_type_step _ _ _ _ _ _ _ Hvalid Hinputs Hi Hcallees Htypes Hstep).
      * exact (source_types_concrete _ _ _ _ _ Hlocals Hresults Hi Hinputs Hstep).
Qed.

(* These types are used only at a structured instruction accepted by siteEntry. *)
Definition structuredResults (i : instr) : list valtype :=
  match i with
  | BLOCK (_RESULT result) _ | LOOP (_RESULT result) _ | IFELSE (_RESULT result) _ _ => option_to_list result
  | _ => []
  end.

Definition structuredLabel (i : instr) : list valtype :=
  match i with LOOP _ _ => [] | _ => structuredResults i end.

Fixpoint pathValidationContext (C : context) (results : list valtype) (code : list instr) (labels : list LabelSite)
    : option (context * list valtype) :=
  match labels with
  | [] => Some (C, results)
  | site :: rest =>
      match siteEntry code site with
      | Some (i, body) => pathValidationContext (labelContext (structuredLabel i) C) (structuredResults i) body rest
      | None => None
      end
  end.

Lemma supported_arm_validation : forall C callees i inputs outputs arm body,
  Instr_ok C i (mk_functype (mk_list _ inputs) (mk_list _ outputs)) ->
  Supported callees i -> instrArm i arm = Some body ->
  Instrs_ok (labelContext (structuredLabel i) C) body
    (mk_functype (mk_list _ []) (mk_list _ (structuredResults i))).
Proof.
  intros C callees i inputs outputs arm body Hvalid Hsupported Harm.
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; cbn [instrArm] in Harm.
  all: try match type of Harm with match ?code with [] => _ | _ :: _ => _ end = _ =>
    destruct code; [discriminate | ] end.
  all: injection Harm as <-; inversion Hsupported; subst; inversion Hvalid; subst.
  all: match goal with H : Blocktype_ok _ _ _ |- _ => inversion H; subst end.
  all: assumption.
Qed.

(* Descending into a label uses the context in SpecTec's structured instruction
   rule. At an offset, the computed types validate the remaining inner code. *)
Lemma validated_path_types : forall callees labels C results code offset segments top,
  Instrs_ok C code (mk_functype (mk_list _ []) (mk_list _ results)) ->
  supportedCode callees code -> calleeTypes C callees -> context_TYPES C = envTypes callees ->
  concreteTypes (context_LOCALS C) ->
  (forall x f, callees x = Some f -> concreteTypes (guestResultTypes f)) ->
  pathTypes (context_LOCALS C) callees code labels offset = Some (segments, top) ->
  exists inner current outputs,
    pathCode code labels = Some inner /\
    pathValidationContext C results code labels = Some (current, outputs) /\
    context_LOCALS current = context_LOCALS C /\ concreteTypes top /\
    Instrs_ok current (skipn offset inner) (mk_functype (mk_list _ top) (mk_list _ outputs)).
Proof.
  intros callees labels. induction labels as [ | site rest IH];
    intros C results code offset segments top Hvalid Hsupported Hcallees Htypes Hlocals Hresults Hpath.
  - cbn [pathTypes] in Hpath.
    destruct (sourceTypeStacks (context_LOCALS C) callees (firstn offset code) []) as [stack | ] eqn:Hstack;
      [ | discriminate]. injection Hpath as <- <-.
    exists code, C, results. split; [reflexivity | ]. split; [reflexivity | ]. split; [reflexivity | ].
    destruct (validated_type_prefix _ _ _ _ _ _ _ Hvalid Hsupported Hcallees Htypes Hlocals Hresults
      (Forall_nil _) Hstack) as [Htail Hconcrete]. auto.
  - cbn [pathTypes] in Hpath.
    destruct (siteEntry code site) as [[i body] | ] eqn:Hsite; [ | discriminate].
    destruct (sourceTypeStacks (context_LOCALS C) callees (firstn (siteOffset site) code) []) as [stack | ] eqn:Hstack;
      [ | discriminate].
    destruct (pathTypes (context_LOCALS C) callees body rest offset) as [[nested values] | ] eqn:Hnested;
      [ | discriminate]. injection Hpath as <- <-.
    destruct (validated_type_prefix _ _ _ _ _ _ _ Hvalid Hsupported Hcallees Htypes Hlocals Hresults
      (Forall_nil _) Hstack) as [Hsuffix Hconcrete].
    destruct (site_entry_split _ _ _ _ Hsite) as [Hi Harm].
    rewrite (skipn_at_instruction _ _ _ Hi) in Hsuffix.
    destruct (validation_concrete_head _ _ _ Hsuffix i _ Logic.eq_refl Hconcrete)
      as [prefix [consumed [produced [_ [Hhead _]]]]].
    pose proof (proj1 (Forall_forall _ _) Hsupported _ (nth_error_In _ _ Hi)) as Hsi.
    pose proof (supported_arm_validation _ _ _ _ _ _ _ Hhead Hsi Harm) as Hbody.
    assert (Hsbody : supportedCode callees body).
    { destruct i; try discriminate; destruct (siteArm site) as [ | [ | arm]]; try discriminate;
        cbn [instrArm] in Harm.
      all: try match type of Harm with match ?code with [] => _ | _ :: _ => _ end = _ =>
        destruct code; [discriminate | ] end.
      all: injection Harm as <-; inversion Hsi; assumption. }
    destruct (IH (labelContext (structuredLabel i) C) (structuredResults i) body offset nested values
      Hbody Hsbody Hcallees Htypes Hlocals Hresults Hnested)
      as [inner [current [outputs [Hinner [Hcontext [Hlocal Htail]]]]]].
    exists inner, current, outputs. cbn [pathCode pathValidationContext]. rewrite Hsite.
    split; [exact Hinner | ]. split; [exact Hcontext | ]. split; assumption.
Qed.
