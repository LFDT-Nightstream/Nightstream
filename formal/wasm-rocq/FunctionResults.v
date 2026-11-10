(* Values returned by a function's END, explicit RETURN, or branch to its own
   label. This is the SpecTec half of both nested and terminal returns. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range FullRelation MemoryTrace Wasm2 WasmIntegers Locals StraightLineSpec
  StackSimulation SpecTecCalls SpecTecBranches CallerSpec ProgramSpec ProgramEncoding CodeLayout Fragment CodeFacts
  ControlDecode ControlInvariant FrameFacts FrameRows SlideLayout.
Require Import IntegerDecode TypeFacts.
Import ListNotations.
Open Scope Z_scope.

Definition exitsFunction (f : GuestFunction) (view : FrameView) : Prop :=
  (positionLabels (viewPosition view) = [] /\ positionOffset (viewPosition view) = length (guestBody f)) \/
  exists i, nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i /\
    returnsFrom (length (positionLabels (viewPosition view))) i.

Lemma function_return_types : forall program s view f dropped returned,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  FrameTyping program f view 0 -> exitsFunction f view ->
  viewOperands view = dropped ++ returned -> length returned = guestResults f ->
  valueTypes returned = guestResultTypes f.
Proof.
  intros program s view f dropped returned Hstore Hvalid Hf Htyping Hexit Hsplit Hlength.
  apply frame_typing_idle in Htyping.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hpath [Hcontext [Hlocals [Hconcrete Hrest]]]]]]]]].
  assert (HinnerTypes : exists prefix, valueTypes (viewValues view) = prefix ++ guestResultTypes f).
  { destruct Hexit as [[Hlabels Hend] | [i [Hi Hreturn]]].
    - rewrite Hlabels in Hcontext, Hpath. cbn [pathValidationContext] in Hcontext.
      injection Hcontext as <- <-. cbn [pathCode] in Hpath. injection Hpath as <-.
      rewrite Hend, skipn_all in Hrest. exists [].
      exact (empty_concrete_types _ _ _ Hrest eq_refl Hconcrete).
    - unfold innerCode in Hi. rewrite Hpath in Hi. rewrite (skipn_at_instruction _ _ _ Hi) in Hrest.
      destruct (validation_concrete_head _ _ _ Hrest i _ eq_refl Hconcrete)
        as [prefix [inputs [results [Hinput [Hhead Htail]]]]].
      assert (Hinputs : exists before, inputs = before ++ guestResultTypes f).
      { destruct Hreturn as [-> | [l [-> Hlevel]]].
        - eapply validated_return_suffix; [exact Hhead | ].
          rewrite (path_context_return _ _ _ _ _ _ Hcontext). reflexivity.
        - assert (Hlabel : nth_error (LABELS current) (N.to_nat (l :> N)) =
              Some (mk_list _ (guestResultTypes f))).
          { rewrite Hlevel. replace (length (positionLabels (viewPosition view))) with
              (length (positionLabels (viewPosition view)) + 0)%nat by lia.
            eapply path_context_outer_label; [exact Hcontext | reflexivity]. }
          destruct (validated_branch_suffix _ _ _ _ _ _ Hhead (or_introl eq_refl) Hlabel
            ltac:(rewrite Hinput in Hconcrete; exact (proj2 (proj1 (Forall_app _ _ _) Hconcrete)))) as [before Hbefore].
          cbn [branchConditionTypes] in Hbefore. rewrite app_nil_r in Hbefore. eauto. }
      destruct Hinputs as [before ->]. exists (prefix ++ before). rewrite Hinput, app_assoc. reflexivity. }
  destruct HinnerTypes as [prefix Htypes].
  assert (Hwhole : valueTypes (viewOperands view) = (valueTypes (concat (viewSegments view)) ++ prefix) ++ guestResultTypes f).
  { unfold viewOperands, valueTypes. rewrite map_app. fold (valueTypes (viewValues view)). rewrite Htypes, app_assoc.
    reflexivity. }
  rewrite Hsplit in Hwhole. unfold valueTypes in Hwhole. rewrite map_app in Hwhole.
  apply (app_suffix_eq _ (map valueType dropped) (map valueType returned) (map valueType (concat (viewSegments view)) ++ prefix) (guestResultTypes f));
    [unfold guestResults in Hlength; rewrite length_map; exact Hlength | exact Hwhole].
Qed.

Lemma function_result_values : forall program s view f memory row inner,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  ActiveState program s memory (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f -> exitsFunction f view ->
  exists dropped returned,
    viewOperands view = dropped ++ returned /\ length returned = guestResults f /\
    (guestResults f <= length (viewValues view))%nat /\
    forall caller, wf_state (mk_state s caller) ->
      Steps (callContext caller (N.of_nat (guestResults f)) inner)
        (mk_config (mk_state s caller) (List.map admininstr_val returned)).
Proof.
  intros program s view f memory row inner Hstore Hwf Hvalid Hprime Hfull Hprogram Hstate Hf Hexit.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [_ [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [_ [_ [Hbounds [_ [_ [Hlb _]]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe)
    as [Hfacts [_ [_ [Hresults [Hready Hwfcode]]]]].
  assert (Hz : wf_state (mk_state s (activeFrame f (viewLocals view))))
    by (eapply frame_state_wf; [eassumption | eassumption | eassumption | exact (values_typed_wf _ _ Hlb)]).
  assert (Hcallee : wf_frame (activeFrame f (viewLocals view))) by (inversion Hz; assumption).
  unfold exitsFunction in Hexit.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset]
    in Hf, Hpath, Hheights, Hfacts, Hresults, Hready, Hwfcode, Hz, Hcallee, Hexit |- *.
  destruct Hexit as [[Hlabels Hend] | [i [Hi Hreturn]]].
  - subst labels offset.
    specialize (Hresults Logic.eq_refl). unfold innerCode in Hresults. cbn [pathCode positionLabels] in Hresults.
    cbn [pathHeights] in Hheights. rewrite firstn_all in Hheights.
    destruct (sourceHeights _ (guestBody f) 0) as [height | ] eqn:Hheight; [ | discriminate].
    rewrite (Hresults _ Hheight) in Hheights. cbn [option_map] in Hheights.
    injection Hheights as Hsegments Hlength. destruct segments; [ | discriminate].
    rewrite Nat.add_0_r in Hlength.
    exists [], values. split; [reflexivity | ]. split; [lia | ]. split; [lia | ]. intros caller Hcaller.
    unfold frameConfig, innerCode. cbn [wrapValueLabels viewPosition positionLabels viewSegments pathCode].
    rewrite skipn_all. unfold callContext, valueStackConfig. cbn [List.map]. rewrite app_nil_r.
    replace (guestResults f) with (length values) by lia.
    apply value_function_returns; assumption.
  - rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts. cbn [codeFacts] in Hfacts. destruct Hfacts as [Hinstr _].
    assert (Hfits : (guestResults f <= length values)%nat).
    { assert (Hlabels : length (pathArities (guestBody f) labels) = length labels)
        by (rewrite (path_arities_length _ _ _ _ _ _ Hheights); exact (path_heights_length _ _ _ _ _ _ Hheights)).
      destruct Hreturn as [-> | [l [-> Hl]]]; cbn [instrFacts] in Hinstr; [lia | ].
      destruct Hinstr as [_ [n [Hn Hle]]].
      rewrite nth_error_app2, length_rev, Hlabels, Hl, Nat.sub_diag in Hn by (rewrite length_rev; lia).
      cbn [nth_error] in Hn. injection Hn as <-. lia. }
    destruct (split_top_values (concat segments ++ values) (guestResults f) ltac:(rewrite length_app; lia))
      as [dropped [returned [Hsplit Hlength]]].
    exists dropped, returned. split; [exact Hsplit | ]. split; [exact Hlength | ]. split; [exact Hfits | ]. intros caller Hcaller.
    unfold frameConfig. cbn [viewPosition positionLabels viewSegments].
    rewrite (skipn_at_instruction _ _ _ Hi).
    assert (Hsegments : length segments = length labels)
      by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
    destruct Hreturn as [-> | [l [-> Hl]]].
    + destruct (wrap_value_labels_return labels (guestBody f) _ segments (mk_state s (activeFrame f locals)) values
        (skipn (S offset) (innerCode f (codePosition index labels offset))) Hpath Hsegments
        Hready Hz Hbounds (forall_skipn _ _ _ _ Hwfcode)) as [rest' [Hrest' Hreturn]].
      eapply spec_steps_transitive;
        [exact (steps_in_call_frame caller _ _ _ ltac:(inversion Hcaller; assumption) Hreturn) | ].
      rewrite Hsplit, <- Hlength. apply value_function_returns_early; try assumption.
      rewrite <- Hsplit. exact Hbounds.
    + assert (Hindex : l = mk_uN (N.of_nat (length labels + 0))).
      { destruct l as [ln]. cbn in Hl |- *. f_equal. lia. }
      assert (Hwfl : wf_uN 32%N (mk_uN (N.of_nat (length labels + 0)))).
      { rewrite <- Hindex. pose proof (proj1 (Forall_forall _ _) Hwfcode _ (nth_error_In _ _ Hi)) as H.
        inversion H; assumption. }
      rewrite Hindex.
      destruct (wrap_value_labels_exit labels (guestBody f) _ segments (mk_state s (activeFrame f locals)) values 0
        (skipn (S offset) (innerCode f (codePosition index labels offset))) Hpath Hsegments
        Hready Hz Hbounds Hwfl (forall_skipn _ _ _ _ Hwfcode)) as [rest' [Hrest' Hexit]].
      eapply spec_steps_transitive;
        [exact (steps_in_call_frame caller _ _ _ ltac:(inversion Hcaller; assumption) Hexit) | ].
      rewrite Hsplit, <- Hlength. apply value_function_branches_out; try assumption.
      rewrite <- Hsplit. exact Hbounds.
Qed.
