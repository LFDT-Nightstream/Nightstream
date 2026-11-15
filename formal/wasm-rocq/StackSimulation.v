(* Relates stack memory and row boundaries to contextual SpecTec steps. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog
  OpcodeCatalog Selectors TraceRefinement ProgramBinding Wasm2 WasmIntegers Add Locals StackBindings StraightLineSpec IntegerStack.
Import ListNotations.
Open Scope Z_scope.

Lemma follows_boundary : forall previous next,
  Follows MemoryCatalog.continuityLinks previous next ->
  afterBoundary previous = beforeBoundary next.
Proof.
  intros previous next Hlinks.
  unfold afterBoundary, beforeBoundary. f_equal;
    apply Hlinks; vm_compute; tauto.
Qed.

Lemma stack_matches_height : forall memory boundary values,
  stackMatches memory boundary values -> stackHeight boundary = length values.
Proof.
  intros memory boundary values [_ [Hsp _]]. unfold stackHeight. rewrite Hsp.
  replace (boundaryStackBase boundary + Z.of_nat (length values) - boundaryStackBase boundary)
    with (Z.of_nat (length values)) by lia. apply Nat2Z.id.
Qed.

Lemma represented_values : forall (typed : nat -> list instr -> Prop) z memory boundary code c,
  (typed (stackHeight boundary) code /\ represents z memory boundary code c) <->
  exists values, stackMatches memory boundary values /\ typed (length values) code /\
    c = stackConfig z values code /\ wf_config c /\ localsMatch memory boundary z.
Proof.
  intros typed z memory boundary code c. split.
  - intros [Ht [values [Hs Hr]]]. exists values. split; [exact Hs | ].
    split; [rewrite <- (stack_matches_height _ _ _ Hs); exact Ht | exact Hr].
  - intros [values [Hs [Ht Hr]]]. split.
    + rewrite (stack_matches_height _ _ _ Hs). exact Ht.
    + exists values. auto.
Qed.

Lemma stack_matches_binary_step : forall w initial final prefix a b,
  BinaryStackShape w ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  stackMatches initial (beforeBoundary w) (prefix ++ [a; b]) ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < i32Modulus ->
  w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = a /\
  w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat) = b /\
  stackMatches final (afterBoundary w) (prefix ++ [w COL_STACK_WRITE0_VALUE_LO]).
Proof.
  intros w initial final prefix a b Hshape Hreplay Hstack Hresult.
  pose proof (proj1 (proj2 (proj2 Hstack))) as Hbounds.
  pose proof (proj2 (value_stack_matches_i32 _ _ _ Hbounds) Hstack) as Htyped.
  rewrite map_app in Htyped. cbn [map] in Htyped.
  apply Forall_app in Hbounds as [Hprefix Hargs].
  inversion Hargs; subst. match goal with H : Forall _ [b] |- _ => inversion H; subst end.
  destruct (value_stack_matches_binary _ _ _ _ _ _ Hshape Hreplay Htyped ltac:(lia) ltac:(lia) Hresult)
    as [Hread0 [Hread1 Hafter]].
  split; [exact Hread0 | ]. split; [exact Hread1 | ].
  assert (Hout : Forall (fun value => 0 <= value < i32Modulus) (prefix ++ [w COL_STACK_WRITE0_VALUE_LO])).
  { apply Forall_app. split; [exact Hprefix | constructor; [exact Hresult | constructor]]. }
  apply (proj1 (value_stack_matches_i32 _ _ _ Hout)). rewrite map_app. exact Hafter.
Qed.

Lemma skipn_at_instruction : forall (program : list instr) pc instruction,
  nth_error program pc = Some instruction ->
  skipn pc program = instruction :: skipn (S pc) program.
Proof.
  induction program as [ |head rest IH]; intros [ |pc] instruction Hlookup;
    cbn in *; try discriminate.
  - inversion Hlookup; reflexivity.
  - apply IH. exact Hlookup.
Qed.

Lemma stack_has_two_values : forall (values : list Z),
  (2 <= length values)%nat -> exists prefix a b, values = prefix ++ [a; b].
Proof.
  induction values as [ |a rest IH]; intros Hlength; [cbn in Hlength; lia | ].
  destruct rest as [ |b rest]; [cbn in Hlength; lia | ].
  destruct rest as [ |c rest].
  - exists [], a, b. reflexivity.
  - destruct (IH ltac:(cbn; lia)) as [prefix [x [y Heq]]].
    exists (a :: prefix), x, y. cbn. now rewrite Heq.
Qed.

Lemma i32_constant_wf : forall value,
  0 <= value < i32Modulus -> wf_admininstr (i32Constant value).
Proof.
  intros value Hbound. unfold i32Constant. constructor.
  apply WasmValues.spectec_i32_well_formed. exact Hbound.
Qed.

Lemma stack_config_wf : forall z values code,
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus) values ->
  Forall wf_instr code -> wf_config (stackConfig z values code).
Proof.
  intros z values code Hz Hvalues Hcode. constructor; [exact Hz | ].
  apply Forall_app. split; apply Forall_map.
  - eapply Forall_impl; [ | exact Hvalues]. intros value H. apply i32_constant_wf. exact H.
  - eapply Forall_impl; [ | exact Hcode]. intros i H. destruct H; cbn; solve [constructor | econstructor; eassumption].
Qed.

Lemma stack_config_step : forall z z' prefix consumed produced instruction code,
  Step (stackConfig z consumed [instruction]) (stackConfig z' produced []) ->
  Forall (fun value => 0 <= value < i32Modulus) produced -> wf_state z' ->
  wf_config (stackConfig z (prefix ++ consumed) (instruction :: code)) ->
  Step (stackConfig z (prefix ++ consumed) (instruction :: code))
    (stackConfig z' (prefix ++ produced) code) /\
  wf_config (stackConfig z' (prefix ++ produced) code).
Proof.
  intros z z' prefix consumed produced instruction code Hstep Hbounds Hz' Hwf.
  unfold stackConfig in *. rewrite !map_app in *. cbn [map] in *.
  repeat rewrite <- app_assoc in *.
  inversion Hwf as [state instructions Hz Hall]; subst.
  apply Forall_app in Hall. destruct Hall as [Hprefix Htail].
  apply Forall_app in Htail. destruct Htail as [Hconsumed Htail].
  pose proof (Forall_inv Htail) as Hinstruction.
  pose proof (Forall_inv_tail Htail) as Hcode.
  assert (Hproduced : Forall wf_admininstr (map i32Constant produced)).
  { apply Forall_map. eapply Forall_impl; [ | exact Hbounds].
    intros value Hvalue. apply i32_constant_wf. exact Hvalue. }
  assert (HsmallBefore : wf_config
    (mk_config z (map i32Constant consumed ++ [admininstr_instr instruction]))).
  { constructor; [exact Hz | ]. apply Forall_app. split; [exact Hconsumed | ].
    constructor; [exact Hinstruction | constructor]. }
  assert (HsmallAfter : wf_config (mk_config z' (map i32Constant produced ++ []))).
  { constructor; [exact Hz' | ]. rewrite app_nil_r. exact Hproduced. }
  split.
  - assert (Hmap : map admininstr_val (map i32Value prefix) = map i32Constant prefix).
    { rewrite map_map. reflexivity. }
    rewrite <- Hmap.
    replace (map i32Constant consumed ++ admininstr_instr instruction :: map admininstr_instr code)
      with ((map i32Constant consumed ++ [admininstr_instr instruction]) ++ map admininstr_instr code)
      by (rewrite <- app_assoc; reflexivity).
    replace (map i32Constant produced ++ map admininstr_instr code)
      with ((map i32Constant produced ++ []) ++ map admininstr_instr code)
      by (rewrite app_nil_r; reflexivity).
    destruct prefix as [ |head prefix], code as [ |next code].
    + cbn [map app]. rewrite !app_nil_r in *. exact Hstep.
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
    + eapply ctxt_instrs; [exact Hstep | reflexivity | exact HsmallBefore | exact HsmallAfter].
  - constructor; [exact Hz' | ]. apply Forall_app. split; [exact Hprefix | ].
    apply Forall_app. auto.
Qed.

Lemma stack_config_binary_step : forall z prefix a b result instruction code,
  Step_pure [i32Constant a; i32Constant b; admininstr_instr instruction]
    [i32Constant result] ->
  0 <= result < i32Modulus ->
  wf_config (stackConfig z (prefix ++ [a; b]) (instruction :: code)) ->
  Step (stackConfig z (prefix ++ [a; b]) (instruction :: code))
    (stackConfig z (prefix ++ [result]) code) /\
  wf_config (stackConfig z (prefix ++ [result]) code).
Proof.
  intros z prefix a b result instruction code Hpure Hresult Hwf.
  apply (stack_config_step z z prefix [a; b] [result] instruction code).
  - apply pure. exact Hpure.
  - constructor; [exact Hresult | constructor].
  - inversion Hwf; assumption.
  - exact Hwf.
Qed.
