(* What a frame at a code path inherits from the program's validation, and how
   code paths, heights and enclosing labels compose. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerValidation FunctionValidation ProgramSpec ProgramEncoding CodeLayout Fragment CodeFacts ControlDecode
  ControlInvariant TypeFacts.
Import ListNotations.
Open Scope nat_scope.

(* Paths extended by one more enclosing label. *)
Lemma path_code_snoc : forall labels code s,
  pathCode code (labels ++ [s]) =
    match pathCode code labels with Some inner => option_map snd (siteEntry inner s) | None => None end.
Proof.
  induction labels as [ | s' rest IH]; intros code s; cbn [app pathCode].
  - destruct (siteEntry code s) as [[i body] | ]; reflexivity.
  - destruct (siteEntry code s') as [[i body] | ]; [apply IH | reflexivity].
Qed.

(* The arities along a path split at an enclosing label. *)
Lemma path_arities_app : forall pre post code outer s i body,
  pathCode code pre = Some outer -> siteEntry outer s = Some (i, body) ->
  pathArities code (pre ++ s :: post) = pathArities code pre ++ labelArity i :: pathArities body post.
Proof.
  induction pre as [ | s' rest IH]; intros post code outer s i body Hpath Hs; cbn [app pathCode pathArities] in *.
  - injection Hpath as <-. rewrite Hs. reflexivity.
  - destruct (siteEntry code s') as [[j nested] | ]; [ | discriminate].
    cbn [app]. f_equal. exact (IH _ _ _ _ _ _ Hpath Hs).
Qed.

Lemma path_offset_snoc : forall labels code inner s i body offset,
  pathCode code labels = Some inner -> siteEntry inner s = Some (i, body) ->
  pathOffset code (labels ++ [s]) offset =
    (pathOffset code labels (siteOffset s) + armStart i (siteArm s) + codeOffset body offset)%nat.
Proof. intros labels code inner s i body offset Hpath Hs. exact (proj2 (path_snoc _ _ _ _ _ _ Hpath Hs) offset). Qed.

Lemma path_heights_snoc : forall callees labels code inner s i body heights height offset,
  pathCode code labels = Some inner -> siteEntry inner s = Some (i, body) ->
  pathHeights callees code labels (siteOffset s) = Some (heights, height) ->
  pathHeights callees code (labels ++ [s]) offset =
    option_map (fun top => (heights ++ [labelBase i height], top)) (sourceHeights callees (firstn offset body) 0).
Proof.
  intros callees. induction labels as [ | s' rest IH];
    intros code inner s i body heights height offset Hpath Hs Hheights; cbn [app pathCode pathHeights] in *.
  - injection Hpath as <-. rewrite Hs.
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0) as [h | ]; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-. cbn. destruct (sourceHeights callees (firstn offset body) 0); reflexivity.
  - destruct (siteEntry code s') as [[j nested] | ]; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s') code) 0) as [h | ]; [ | discriminate].
    destruct (pathHeights callees nested rest (siteOffset s)) as [[hs top] | ] eqn:Hinner; cbn in Hheights;
      [ | discriminate].
    injection Hheights as <- <-. rewrite (IH _ _ _ _ _ _ _ offset Hpath Hs Hinner).
    destruct (sourceHeights callees (firstn offset body) 0); reflexivity.
Qed.

Lemma wrap_value_labels_snoc : forall labels code inner s i body segments segment c,
  pathCode code labels = Some inner -> siteEntry inner s = Some (i, body) ->
  length segments = length labels ->
  wrapValueLabels code (labels ++ [s]) (segments ++ [segment]) c =
    wrapValueLabels code labels segments
      (valueStackPrefix segment (instructionSuffix (skipn (S (siteOffset s)) inner)
        (blockContext (labelArity i) (labelContinuation i) c))).
Proof.
  induction labels as [ | s' rest IH]; intros code inner s i body segments segment c Hpath Hs Hlen.
  - destruct segments; [ | discriminate]. injection Hpath as <-. cbn [app wrapValueLabels]. rewrite Hs. reflexivity.
  - destruct segments as [ | seg more]; [discriminate | ]. cbn [app wrapValueLabels pathCode] in *.
    destruct (siteEntry code s') as [[j nested] | ]; [ | discriminate].
    rewrite (IH _ _ _ _ _ more segment c Hpath Hs ltac:(cbn in Hlen; lia)). reflexivity.
Qed.

(* Paths split at an enclosing label: the labels before it, it, and those inside. *)
Lemma path_code_app : forall pre post code,
  pathCode code (pre ++ post) = match pathCode code pre with Some inner => pathCode inner post | None => None end.
Proof.
  induction pre as [ | s rest IH]; intros post code; cbn [app pathCode]; [reflexivity | ].
  destruct (siteEntry code s) as [[i body] | ]; [apply IH | reflexivity].
Qed.

Lemma wrap_value_labels_app : forall pre post code outer segmentsPre segmentsPost c,
  pathCode code pre = Some outer -> length segmentsPre = length pre ->
  wrapValueLabels code (pre ++ post) (segmentsPre ++ segmentsPost) c =
    wrapValueLabels code pre segmentsPre (wrapValueLabels outer post segmentsPost c).
Proof.
  induction pre as [ | s rest IH]; intros post code outer segmentsPre segmentsPost c Hpath Hlen.
  - destruct segmentsPre; [ | discriminate]. injection Hpath as <-. reflexivity.
  - destruct segmentsPre as [ | seg more]; [discriminate | ]. cbn [app wrapValueLabels pathCode] in *.
    destruct (siteEntry code s) as [[i body] | ]; [ | discriminate].
    rewrite (IH _ _ _ more _ c Hpath ltac:(cbn in Hlen; lia)). reflexivity.
Qed.

Lemma path_heights_split : forall callees pre s post code offset heights top,
  pathHeights callees code (pre ++ s :: post) offset = Some (heights, top) ->
  exists outer i body headHeights height tailHeights,
    pathCode code pre = Some outer /\ siteEntry outer s = Some (i, body) /\
    pathHeights callees code pre (siteOffset s) = Some (headHeights, height) /\
    heights = headHeights ++ labelBase i height :: tailHeights /\
    pathHeights callees body post offset = Some (tailHeights, top).
Proof.
  intros callees. induction pre as [ | s0 rest IH]; intros s post code offset heights top H;
    cbn [app pathHeights] in H.
  - destruct (siteEntry code s) as [[i body] | ] eqn:Hs; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0) as [height | ] eqn:Hheight; [ | discriminate].
    destruct (pathHeights callees body post offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate].
    injection H as <- <-. exists code, i, body, [], height, hs.
    cbn [pathCode pathHeights]. rewrite Hheight. repeat split; assumption || reflexivity.
  - destruct (siteEntry code s0) as [[j nested] | ] eqn:Hs0; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s0) code) 0) as [height0 | ] eqn:Hheight0; [ | discriminate].
    destruct (pathHeights callees nested (rest ++ s :: post) offset) as [[hs t] | ] eqn:Hin; cbn in H;
      [ | discriminate].
    injection H as <- <-.
    destruct (IH _ _ _ _ _ _ Hin) as [outer [i [body [hs' [height [hs'' [Hpath [Hs [Hheights [-> Htail]]]]]]]]]].
    exists outer, i, body, (labelBase j height0 :: hs'), height, hs''.
    cbn [pathCode pathHeights]. rewrite Hs0, Hheight0, Hheights. repeat split; assumption || reflexivity.
Qed.

(* The operands below each label, then the innermost ones. *)
Lemma split_concat : forall segments values,
  splitSegments (map (@length val) segments) (concat segments ++ values) = (segments, values).
Proof.
  induction segments as [ | seg more IH]; intros values; [reflexivity | ].
  cbn [map concat splitSegments]. rewrite <- app_assoc, firstn_app, Nat.sub_diag, firstn_all, skipn_app,
    skipn_all, Nat.sub_diag. cbn [firstn skipn app]. rewrite app_nil_r, IH. reflexivity.
Qed.

Lemma total_height_operands : forall segments values pending,
  totalHeight (map (@length val) segments, (length values + pending)%nat) =
    (length (concat segments ++ values) + pending)%nat.
Proof.
  intros. unfold totalHeight. cbn [fst snd]. rewrite length_app.
  induction segments as [ | seg more IH]; cbn; [reflexivity | ]. rewrite length_app. lia.
Qed.

Lemma split_lengths : forall heights stack,
  (fold_right Nat.add 0 heights <= length stack)%nat ->
  map (@length val) (fst (splitSegments heights stack)) = heights /\
  concat (fst (splitSegments heights stack)) ++ snd (splitSegments heights stack) = stack.
Proof.
  induction heights as [ | h rest IH]; intros stack Hlen; [split; reflexivity | ].
  cbn [fold_right] in Hlen. cbn [splitSegments fst snd map concat].
  destruct (IH (skipn h stack) ltac:(rewrite length_skipn; lia)) as [Hmap Hconcat].
  split; [rewrite length_firstn, Hmap; f_equal; lia | ].
  rewrite <- app_assoc, Hconcat. apply firstn_skipn.
Qed.

Lemma wrap_value_labels_i32 : forall labels code segments c,
  wrapValueLabels code labels (map (map i32Value) segments) c = wrapLabels code labels segments c.
Proof.
  induction labels as [ | site rest IH]; intros code [ | segment more] c; try reflexivity.
  cbn [map wrapValueLabels wrapLabels]. destruct (siteEntry code site) as [[i body] | ]; [ | reflexivity].
  rewrite IH, value_stack_prefix_i32. reflexivity.
Qed.

(* Enclosing labels with well-formed operands and continuations. *)
Fixpoint ValueLabelsReady (code : list instr) (labels : list LabelSite) (segments : list (list val)) : Prop :=
  match labels, segments with
  | s :: rest, segment :: more =>
      Forall wf_val segment /\ Forall wf_instr (skipn (S (siteOffset s)) code) /\
      match siteEntry code s with
      | Some (i, body) => Forall wf_instr (labelContinuation i) /\ ValueLabelsReady body rest more
      | None => True
      end
  | _, _ => True
  end.

(* Scalar wrappers used by the existing i32-only helper lemmas. *)
Definition LabelsReady code labels segments :=
  ValueLabelsReady code labels (map (map i32Value) segments).

Lemma concat_map_values : forall segments,
  concat (map (map i32Value) segments) = map i32Value (concat segments).
Proof.
  induction segments as [ | segment rest IH]; [reflexivity | ].
  cbn [map concat]. now rewrite IH, map_app.
Qed.

Lemma forall_skipn : forall (A : Type) (P : A -> Prop) k xs, Forall P xs -> Forall P (skipn k xs).
Proof.
  intros A P k xs H. apply Forall_forall. intros x Hx. apply (proj1 (Forall_forall _ _) H).
  rewrite <- (firstn_skipn k xs). apply in_or_app. right. exact Hx.
Qed.

Lemma path_code_wf : forall labels code inner,
  Forall wf_instr code -> pathCode code labels = Some inner -> Forall wf_instr inner.
Proof.
  induction labels as [ | s rest IH]; intros code inner Hwf Hpath; cbn [pathCode] in Hpath.
  - injection Hpath as <-. exact Hwf.
  - destruct (siteEntry code s) as [[i body] | ] eqn:Hs; [ | discriminate].
    exact (IH _ _ (site_wf _ _ _ _ Hwf Hs) Hpath).
Qed.

Lemma continuation_wf : forall code s i body,
  Forall wf_instr code -> siteEntry code s = Some (i, body) -> Forall wf_instr (labelContinuation i).
Proof.
  intros code s i body Hwf Hentry. destruct (site_entry_split _ _ _ _ Hentry) as [Hi _].
  pose proof (proj1 (Forall_forall _ _) Hwf _ (nth_error_In _ _ Hi)) as Hw.
  destruct i; cbn [labelContinuation]; constructor; [exact Hw | constructor].
Qed.

Lemma value_labels_ready : forall labels code inner segments,
  Forall wf_instr code -> pathCode code labels = Some inner -> length segments = length labels ->
  Forall wf_val (concat segments) -> ValueLabelsReady code labels segments.
Proof.
  induction labels as [ | s rest IH]; intros code inner segments Hwf Hpath Hlen Hvalues; [exact I | ].
  destruct segments as [ | seg more]; [discriminate | ]. cbn [concat] in Hvalues. apply Forall_app in Hvalues.
  cbn [pathCode] in Hpath. cbn [ValueLabelsReady].
  split; [exact (proj1 Hvalues) | ]. split; [apply forall_skipn; exact Hwf | ].
  destruct (siteEntry code s) as [[i body] | ] eqn:Hs; [ | discriminate].
  split; [exact (continuation_wf _ _ _ _ Hwf Hs) | ].
  exact (IH _ _ more (site_wf _ _ _ _ Hwf Hs) Hpath ltac:(cbn in Hlen; lia) (proj2 Hvalues)).
Qed.

Lemma labels_ready : forall labels code inner segments,
  Forall wf_instr code -> pathCode code labels = Some inner -> length segments = length labels ->
  Forall (fun value => 0 <= value < i32Modulus)%Z (concat segments) -> LabelsReady code labels segments.
Proof.
  intros labels code inner segments Hwf Hpath Hlen Hvalues.
  apply value_labels_ready with (inner := inner); try assumption.
  - now rewrite length_map.
  - rewrite concat_map_values. now apply i32_values_wf.
Qed.

Lemma steps_in_value_labels : forall labels code segments before after,
  ValueLabelsReady code labels segments -> Steps before after ->
  Steps (wrapValueLabels code labels segments before) (wrapValueLabels code labels segments after).
Proof.
  induction labels as [ | s rest IH]; intros code segments before after Hready Hsteps; [exact Hsteps | ].
  destruct segments as [ | seg more]; [exact Hsteps | ]. destruct Hready as [Hseg [Hcont Hinner]].
  cbn [wrapValueLabels]. destruct (siteEntry code s) as [[i body] | ]; [ | exact Hsteps]. destruct Hinner as [Hk Hinner].
  apply steps_with_value_prefix; [exact Hseg | ]. apply steps_with_instruction_suffix; [exact Hcont | ].
  apply steps_in_block; [exact Hk | ]. exact (IH _ _ _ _ Hinner Hsteps).
Qed.

Lemma steps_in_labels : forall labels code segments before after,
  LabelsReady code labels segments -> Steps before after ->
  Steps (wrapLabels code labels segments before) (wrapLabels code labels segments after).
Proof.
  intros labels code segments before after Hready Hsteps.
  pose proof (steps_in_value_labels labels code (map (map i32Value) segments) before after Hready Hsteps) as H.
  rewrite !wrap_value_labels_i32 in H. exact H.
Qed.

Lemma wrap_value_labels_wf : forall labels code segments c,
  ValueLabelsReady code labels segments -> wf_config c -> wf_config (wrapValueLabels code labels segments c).
Proof.
  induction labels as [ | s rest IH]; intros code segments c Hready Hwf; [exact Hwf | ].
  destruct segments as [ | seg more]; [exact Hwf | ]. destruct Hready as [Hseg [Hcont Hinner]].
  cbn [wrapValueLabels]. destruct (siteEntry code s) as [[i body] | ]; [ | exact Hwf]. destruct Hinner as [Hk Hinner].
  apply value_stack_prefix_wf; [exact Hseg | ]. apply instruction_suffix_wf; [exact Hcont | ].
  pose proof (IH _ _ _ Hinner Hwf) as H.
  match type of H with wf_config ?inner => destruct inner as [z code'] end.
  apply block_context_wf; [exact Hk | exact H].
Qed.

Lemma wrap_labels_wf : forall labels code segments c,
  LabelsReady code labels segments -> wf_config c -> wf_config (wrapLabels code labels segments c).
Proof.
  intros labels code segments c Hready Hwf.
  pose proof (wrap_value_labels_wf labels code (map (map i32Value) segments) c Hready Hwf) as H.
  rewrite wrap_value_labels_i32 in H. exact H.
Qed.

(* A branch out of labels: br k+n inside n labels leaves them as br k, carrying
   their operands out and dropping the code after each one's END. *)
Lemma wrap_value_labels_exit : forall labels code inner segments z values m rest,
  pathCode code labels = Some inner -> length segments = length labels -> ValueLabelsReady code labels segments ->
  wf_state z -> Forall wf_val (concat segments ++ values) ->
  wf_uN 32%N (mk_uN (N.of_nat (length labels + m))) -> Forall wf_instr rest ->
  exists rest', Forall wf_instr rest' /\
    Steps (wrapValueLabels code labels segments (valueStackConfig z values (BR (mk_uN (N.of_nat (length labels + m))) :: rest)))
      (valueStackConfig z (concat segments ++ values) (BR (mk_uN (N.of_nat m)) :: rest')).
Proof.
  induction labels as [ | s more IH]; intros code inner segments z values m rest Hpath Hlen Hready Hz Hbounds Hl Hrest.
  - destruct segments; [ | discriminate]. exists rest. split; [exact Hrest | ]. cbn [wrapValueLabels length Nat.add concat app].
    apply Steps__refl. apply value_stack_config_wf; [exact Hz | exact Hbounds | ].
    constructor; [constructor; exact Hl | exact Hrest].
  - destruct segments as [ | seg segments]; [discriminate | ].
    destruct Hready as [Hseg [Hcont Hinner]]. cbn [pathCode wrapValueLabels concat] in *.
    destruct (siteEntry code s) as [[i body] | ]; [ | discriminate]. destruct Hinner as [Hk Hinner].
    replace (length (s :: more) + m) with (length more + S m) in * by (cbn; lia).
    rewrite <- app_assoc in Hbounds. apply Forall_app in Hbounds as [_ Hcarried].
    destruct (IH _ _ segments z values (S m) rest Hpath ltac:(cbn in Hlen; lia) Hinner Hz Hcarried Hl Hrest)
      as [rest' [Hrest' Hsteps]].
    replace (N.of_nat (S m)) with (N.of_nat m + 1)%N in Hsteps by lia.
    assert (Hk' : wf_uN 32%N (mk_uN (N.of_nat m + 1)%N)) by (eapply wf_label_le; [ | exact Hl]; lia).
    exists (skipn (S (siteOffset s)) code). split; [exact Hcont | ].
    rewrite <- app_assoc, <- value_stack_prefix_config.
    replace (BR (mk_uN (N.of_nat m)) :: skipn (S (siteOffset s)) code)
      with ([BR (mk_uN (N.of_nat m))] ++ skipn (S (siteOffset s)) code) by reflexivity.
    rewrite <- value_instruction_suffix_config.
    apply (steps_with_value_prefix seg _ _ Hseg). apply (steps_with_instruction_suffix _ _ _ Hcont).
    eapply spec_steps_transitive; [exact (steps_in_block _ _ _ _ Hk Hsteps) | ].
    exact (value_br_exit_label_carrying _ _ z _ (N.of_nat m) rest' Hk Hz Hcarried Hk' Hrest').
Qed.

Lemma wrap_labels_exit : forall labels code inner segments z values m rest,
  pathCode code labels = Some inner -> length segments = length labels -> LabelsReady code labels segments ->
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus)%Z (concat segments ++ values) ->
  wf_uN 32%N (mk_uN (N.of_nat (length labels + m))) -> Forall wf_instr rest ->
  exists rest', Forall wf_instr rest' /\
    Steps (wrapLabels code labels segments (stackConfig z values (BR (mk_uN (N.of_nat (length labels + m))) :: rest)))
      (stackConfig z (concat segments ++ values) (BR (mk_uN (N.of_nat m)) :: rest')).
Proof.
  intros labels code inner segments z values m rest Hp Hl Hr Hz Hv Hlabel Hrest.
  assert (Hvalues : Forall wf_val (concat (map (map i32Value) segments) ++ map i32Value values)).
  { rewrite concat_map_values, <- map_app. now apply i32_values_wf. }
  destruct (wrap_value_labels_exit labels code inner (map (map i32Value) segments) z (map i32Value values) m rest Hp ltac:(now rewrite length_map)
    Hr Hz Hvalues Hlabel Hrest) as [rest' [Hrest' Hsteps]].
  exists rest'. split; [exact Hrest' | ].
  rewrite concat_map_values, <- map_app, !value_stack_config_i32, wrap_value_labels_i32 in Hsteps. exact Hsteps.
Qed.

(* A return inside labels leaves them all, carrying their operands out and
   dropping the code after each one's END. *)
Lemma wrap_value_labels_return : forall labels code inner segments z values rest,
  pathCode code labels = Some inner -> length segments = length labels -> ValueLabelsReady code labels segments ->
  wf_state z -> Forall wf_val (concat segments ++ values) -> Forall wf_instr rest ->
  exists rest', Forall wf_instr rest' /\
    Steps (wrapValueLabels code labels segments (valueStackConfig z values (RETURN :: rest)))
      (valueStackConfig z (concat segments ++ values) (RETURN :: rest')).
Proof.
  induction labels as [ | s more IH]; intros code inner segments z values rest Hpath Hlen Hready Hz Hbounds Hrest.
  - destruct segments; [ | discriminate]. exists rest. split; [exact Hrest | ]. cbn [wrapValueLabels concat app].
    apply Steps__refl. apply value_stack_config_wf; [exact Hz | exact Hbounds | ]. constructor; [constructor | exact Hrest].
  - destruct segments as [ | seg segments]; [discriminate | ].
    destruct Hready as [Hseg [Hcont Hinner]]. cbn [pathCode wrapValueLabels concat] in *.
    destruct (siteEntry code s) as [[i body] | ]; [ | discriminate]. destruct Hinner as [Hk Hinner].
    rewrite <- app_assoc in Hbounds. apply Forall_app in Hbounds as [_ Hcarried].
    destruct (IH _ _ segments z values rest Hpath ltac:(cbn in Hlen; lia) Hinner Hz Hcarried Hrest)
      as [rest' [Hrest' Hsteps]].
    exists (skipn (S (siteOffset s)) code). split; [exact Hcont | ].
    rewrite <- app_assoc, <- value_stack_prefix_config.
    replace (RETURN :: skipn (S (siteOffset s)) code) with ([RETURN] ++ skipn (S (siteOffset s)) code) by reflexivity.
    rewrite <- value_instruction_suffix_config.
    apply (steps_with_value_prefix seg _ _ Hseg). apply (steps_with_instruction_suffix _ _ _ Hcont).
    eapply spec_steps_transitive; [exact (steps_in_block _ _ _ _ Hk Hsteps) | ].
    exact (value_return_label_carrying _ _ z _ rest' Hk Hz Hcarried Hrest').
Qed.

Lemma wrap_labels_return : forall labels code inner segments z values rest,
  pathCode code labels = Some inner -> length segments = length labels -> LabelsReady code labels segments ->
  wf_state z -> Forall (fun value => 0 <= value < i32Modulus)%Z (concat segments ++ values) -> Forall wf_instr rest ->
  exists rest', Forall wf_instr rest' /\
    Steps (wrapLabels code labels segments (stackConfig z values (RETURN :: rest)))
      (stackConfig z (concat segments ++ values) (RETURN :: rest')).
Proof.
  intros labels code inner segments z values rest Hp Hl Hr Hz Hv Hrest.
  assert (Hvalues : Forall wf_val (concat (map (map i32Value) segments) ++ map i32Value values)).
  { rewrite concat_map_values, <- map_app. now apply i32_values_wf. }
  destruct (wrap_value_labels_return labels code inner (map (map i32Value) segments) z (map i32Value values) rest Hp ltac:(now rewrite length_map)
    Hr Hz Hvalues Hrest) as [rest' [Hrest' Hsteps]].
  exists rest'. split; [exact Hrest' | ].
  rewrite concat_map_values, <- map_app, !value_stack_config_i32, wrap_value_labels_i32 in Hsteps. exact Hsteps.
Qed.

(* Callee types and validation come from the module context SpecTec instantiated. *)
Lemma program_callee_types : forall program s module C,
  store_FUNCS s = program -> Moduleinst_ok s module C ->
  calleeTypes C (programCallees program module).
Proof.
  intros program s module C Hstore Hmodule x f Hf.
  unfold programCallees in Hf; cbn [calleeAt] in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ] eqn:Ha; [ | discriminate].
  destruct (function_at_typed _ _ _ Hf) as [Hfi _].
  exact (module_function_type _ _ _ _ _ _ Hmodule Ha ltac:(rewrite Hstore; exact Hfi)).
Qed.

Lemma program_callee_validates : forall program s module x f,
  Forall (fun fi => Funcinst_ok s fi (funcinst_TYPE fi)) program ->
  programCallees program module x = Some f ->
  Funcinst_ok s (typedFunction (guestModule f) (guestTypeIndex f)
    (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f))
    (guestType f).
Proof.
  intros program s module x f Hvalid Hf. unfold programCallees in Hf; cbn [calleeAt] in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ]; [ | discriminate].
  destruct (function_at_typed _ _ _ Hf) as [Hfi _].
  rewrite Forall_forall in Hvalid. apply Hvalid. eapply nth_error_In; exact Hfi.
Qed.

(* Keep the declared types when extracting the body judgement. *)
Lemma function_body_validation : forall program s index f,
  programValidated program s -> functionAt program index = Some f ->
  exists C, Moduleinst_ok s (guestModule f) C /\
    Instrs_ok (functionContext C (typedFunction (guestModule f) (guestTypeIndex f)
      (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f))) (guestBody f)
      (mk_functype (mk_list _ []) (mk_list _ (guestResultTypes f))).
Proof.
  intros program s index f [Hfuncs _] Hf.
  destruct (function_at_typed _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) Hfuncs _ (nth_error_In _ _ Hnth)) as Hinst.
  exact (typed_funcinst_body_validated _ _ _ _ _ _ _ Hinst).
Qed.

Lemma function_concrete_types : forall program index f,
  functionAt program index = Some f ->
  concreteTypes (guestFrameTypes f) /\ concreteTypes (guestResultTypes f).
Proof.
  intros program index f Hf. destruct (function_integer_types _ _ _ Hf) as [Hp [Hr Hl]].
  unfold guestFrameTypes. split; [apply Forall_app; split | ];
    eapply Forall_impl; try eassumption; intros t [-> | ->]; discriminate.
Qed.

(* The decoder's computed types also validate the continuation in SpecTec's
   actual function and label contexts. No synthetic all-i32 context is used. *)
Lemma function_path_validation : forall program s index f labels offset segments top,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f))
    (guestBody f) labels offset = Some (segments, top) ->
  exists C inner current outputs,
    Moduleinst_ok s (guestModule f) C /\
    pathCode (guestBody f) labels = Some inner /\
    pathValidationContext
      (functionContext C (typedFunction (guestModule f) (guestTypeIndex f)
        (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f)))
      (guestResultTypes f) (guestBody f) labels = Some (current, outputs) /\
    context_LOCALS current = guestFrameTypes f /\ concreteTypes top /\
    Instrs_ok current (skipn offset inner) (mk_functype (mk_list _ top) (mk_list _ outputs)).
Proof.
  intros program s index f labels offset segments top Hstore Hvalid Hf Hpath.
  destruct (function_body_validation _ _ _ _ Hvalid Hf) as [C [HC Hbody]].
  set (FC := functionContext C (typedFunction (guestModule f) (guestTypeIndex f)
    (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f))) in *.
  assert (Hlocals : context_LOCALS FC = guestFrameTypes f).
  { unfold FC, guestFrameTypes. apply typed_function_context_locals.
    exact (module_context_locals _ _ _ HC). }
  assert (Hcallees : calleeTypes FC (programCallees program (guestModule f))).
  { exact (program_callee_types _ _ _ _ Hstore HC). }
  assert (Htypes : context_TYPES FC = envTypes (programCallees program (guestModule f))).
  { exact (proj1 (module_context_tables _ _ _ HC)). }
  destruct (function_concrete_types _ _ _ Hf) as [HlocalTypes _]. rewrite <- Hlocals in HlocalTypes, Hpath.
  assert (HresultTypes : forall x g, programCallees program (guestModule f) x = Some g ->
    concreteTypes (guestResultTypes g)).
  { intros x g Hg. unfold programCallees in Hg; cbn [calleeAt] in Hg.
    destruct (nth_error (FUNCS (guestModule f)) (N.to_nat x)); [ | discriminate].
    exact (proj2 (function_concrete_types _ _ _ Hg)). }
  destruct (validated_path_types _ _ _ _ _ _ _ _ Hbody (proj1 (proj2 Hvalid index f Hf))
    Hcallees Htypes HlocalTypes HresultTypes Hpath)
    as [inner [current [outputs [Hinner [Hcontext [Hcurrent [Hconcrete Htail]]]]]]].
  exists C, inner, current, outputs. rewrite Hcurrent, Hlocals.
  repeat split; assumption.
Qed.

(* A function's body: its facts from height 0, its result height, its support,
   and its branch sites. *)
Lemma function_facts : forall program s index f,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  let callees := programCallees program (guestModule f) in
  codeFacts (frameLocals f) (guestResults f) [guestResults f] callees 0 (guestBody f) /\
  endsAt callees (guestBody f) 0 (guestResults f) /\
  supportedCode callees (guestBody f) /\ codeBranches callees 0 (guestBody f) 0.
Proof.
  intros program s index f Hstore Hvalid Hf callees.
  destruct (function_body_validation _ _ _ _ Hvalid Hf) as [C [HC Hbody]].
  destruct Hvalid as [_ Hsupported].
  pose proof (code_validation_facts (S (codeSize (guestBody f))) _ _ _ Hbody ltac:(lia)) as H.
  destruct (Hsupported index f Hf) as [Hsupport Hbranches].
  assert (Hfits : contextFits (functionContext C (typedFunction (guestModule f) (guestTypeIndex f)
      (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f))) (frameLocals f) (guestResults f)
      [guestResults f] callees).
  { split; [rewrite typed_function_context_locals by (eapply module_context_locals; exact HC);
      rewrite length_app; reflexivity | ].
    split; [inversion HC; subst; reflexivity | ].
    split; [inversion HC; subst; reflexivity | ].
    split; [intros x g Hg; exact (program_callee_types _ _ _ _ Hstore HC x g Hg) | ].
    destruct (module_context_tables _ _ _ HC) as [Htypes Htables]. exact (conj Htypes Htables). }
  destruct (H (frameLocals f) (guestResults f) _ _ Hfits Hsupport) as [Hfacts [Hheights _]].
  cbn [length] in Hfacts, Hheights. fold (guestResults f) in Hfacts, Hheights.
  split; [pose proof (Hfacts 0%nat [] I) as Hf0; rewrite app_nil_r in Hf0; exact Hf0 | ].
  split; [exact (Hheights 0%nat) | split; assumption].
Qed.

(* An arm's facts, from its instruction's: its body runs inside the instruction's
   label from height 0 and ends at the instruction's results. *)
Lemma arm_facts : forall localCount resultCount arities callees i h exits arm body,
  instrFacts localCount resultCount arities callees i h -> instrBranches callees i h exits ->
  Supported callees i -> instrArm i arm = Some body ->
  codeFacts localCount resultCount (labelArity i :: arities) callees 0 body /\
  endsAt callees body 0 (labelResults i) /\ codeBranches callees (S exits) body 0 /\ supportedCode callees body.
Proof.
  intros localCount resultCount arities callees i h exits arm body Hfacts Hbr Hsup Harm.
  destruct i; try discriminate; destruct arm as [ | [ | arm]]; try discriminate; cbn [instrArm] in Harm.
  - injection Harm as <-. destruct (proj1 (block_facts _ _ _ _ _ _ _) Hfacts) as [_ [Hend Hbody]].
    apply (proj1 (block_branches _ _ _ _ _)) in Hbr.
    inversion Hsup; subst. cbn [labelArity]. unfold supportedCode. split; [ | split; [ | split]]; assumption.
  - injection Harm as <-. destruct (proj1 (loop_facts _ _ _ _ _ _ _) Hfacts) as [_ [Hend Hbody]].
    inversion Hsup; subst. cbn [labelArity labelResults]. unfold supportedCode. split; [ | split; [ | split]]; assumption.
  - injection Harm as <-. destruct (proj1 (if_facts _ _ _ _ _ _ _ _) Hfacts) as [_ [_ [Hend [_ [Hbody _]]]]].
    destruct Hbr as [Hbr _]. inversion Hsup; subst. cbn [labelArity]. unfold supportedCode. split; [ | split; [ | split]]; assumption.
  - match type of Harm with match ?e with [] => _ | _ :: _ => _ end = _ => destruct e end; [discriminate | ].
    injection Harm as <-. destruct (proj1 (if_facts _ _ _ _ _ _ _ _) Hfacts) as [_ [_ [_ [Hend [_ Hbody]]]]].
    destruct Hbr as [_ Hbr]. inversion Hsup; subst. cbn [labelArity]. unfold supportedCode. split; [ | split; [ | split]]; assumption.
Qed.

(* The results of the innermost label's instruction at a path. *)
Fixpoint pathResults (code : list instr) (labels : list LabelSite) : nat :=
  match labels with
  | [] => 0
  | s :: rest => match siteEntry code s with
      | Some (i, body) => match rest with [] => labelResults i | _ => pathResults body rest end
      | None => 0
      end
  end.

Lemma path_results_snoc : forall pre code outer s i body,
  pathCode code pre = Some outer -> siteEntry outer s = Some (i, body) -> pathResults code (pre ++ [s]) = labelResults i.
Proof.
  induction pre as [ | s' rest IH]; intros code outer s i body Hpath Hs; cbn [app pathCode pathResults] in *.
  - injection Hpath as <-. rewrite Hs. reflexivity.
  - destruct (siteEntry code s') as [[j nested] | ]; [ | discriminate].
    destruct (rest ++ [s]) eqn:Hrest; [destruct rest; discriminate | ]. rewrite <- Hrest. exact (IH _ _ _ _ _ Hpath Hs).
Qed.

(* Facts of the code at a path, from the enclosing code's. *)
Lemma path_facts : forall localCount resultCount arities callees labels code exits inner offset heights top,
  codeFacts localCount resultCount arities callees 0 code -> supportedCode callees code ->
  codeBranches callees exits code 0 ->
  pathCode code labels = Some inner -> pathHeights callees code labels offset = Some (heights, top) ->
  codeFacts localCount resultCount (rev (pathArities code labels) ++ arities) callees top (skipn offset inner) /\
  supportedCode callees (skipn offset inner) /\
  codeBranches callees (exits + length heights) (skipn offset inner) top /\
  (labels <> [] -> endsAt callees inner 0 (pathResults code labels)).
Proof.
  intros localCount resultCount arities callees labels. revert arities. induction labels as [ | s rest IH];
    intros arities code exits inner offset heights top Hfacts Hsupport Hbranches Hpath Hheights;
    cbn [pathCode pathHeights pathArities] in *.
  - injection Hpath as <-.
    destruct (sourceHeights callees (firstn offset code) 0) as [h | ] eqn:Hh; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    split; [exact (code_facts_skip _ _ _ _ _ _ _ _ Hfacts Hh) | ]. split; [apply supported_skip; exact Hsupport | ].
    split; [rewrite Nat.add_0_r; exact (code_branches_skip _ _ _ _ _ _ Hbranches Hh) | ]. intros Hne; contradiction.
  - destruct (siteEntry code s) as [[i body] | ] eqn:Hs; [ | discriminate].
    destruct (site_entry_split _ _ _ _ Hs) as [Hb Harm].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs t] | ] eqn:Hin; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    pose proof (code_facts_skip _ _ _ _ _ _ _ _ Hfacts Hbelow) as Hat.
    rewrite (skipn_at_instruction _ _ _ Hb) in Hat. cbn [codeFacts] in Hat. destruct Hat as [Hinstr _].
    pose proof (code_branches_skip _ _ _ _ _ _ Hbranches Hbelow) as Hbr.
    rewrite (skipn_at_instruction _ _ _ Hb) in Hbr. destruct Hbr as [Hbr _].
    pose proof (proj1 (Forall_forall _ _) Hsupport _ (nth_error_In _ _ Hb)) as Hsup.
    destruct (arm_facts _ _ _ _ _ _ _ _ _ Hinstr Hbr Hsup Harm) as [Hbody [Hend [Hbranches' Hsupport']]].
    destruct (IH _ body _ inner offset hs t Hbody Hsupport' Hbranches' Hpath Hin) as [Hf [Hsp [Hbranch Hdeeper]]].
    cbn [rev]. rewrite <- app_assoc.
    split; [exact Hf | ]. split; [exact Hsp | ]. split; [cbn [length]; rewrite <- Nat.add_succ_comm; exact Hbranch | ].
    intros _. cbn [pathResults]. rewrite Hs. destruct rest as [ | s2 rest2].
    + cbn [pathCode] in Hpath. injection Hpath as <-. exact Hend.
    + exact (Hdeeper ltac:(discriminate)).
Qed.

(* What a frame at its position inherits from the program's validation. *)
Lemma view_facts : forall program s memory base fbp pending remaining distance f view,
  store_FUNCS s = program -> programValidated program s ->
  FrameAt program memory base fbp pending remaining distance f view ->
  let position := viewPosition view in
  let callees := programCallees program (guestModule f) in
  let rest := skipn (positionOffset position) (innerCode f position) in
  let arities := rev (pathArities (guestBody f) (positionLabels position)) ++ [guestResults f] in
  codeFacts (frameLocals f) (guestResults f) arities callees (length (viewValues view) + pending) rest /\
  (supportedCode callees rest /\
    codeBranches callees (length (viewSegments view)) rest (length (viewValues view) + pending)) /\
  (positionLabels position <> [] ->
    endsAt callees (innerCode f position) 0 (pathResults (guestBody f) (positionLabels position))) /\
  (positionLabels position = [] -> endsAt callees (innerCode f position) 0 (guestResults f)) /\
  ValueLabelsReady (guestBody f) (positionLabels position) (viewSegments view) /\
  Forall wf_instr (innerCode f position).
Proof.
  intros program s memory base fbp pending remaining distance f view Hstore Hvalid
    [Hf [Hpath [Hheights [_ [_ [Hoperands _]]]]]] position callees rest arities.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts [Hresults [Hsupport Hbranches]]].
  destruct (path_facts _ _ _ _ _ _ _ _ _ _ _ Hfacts Hsupport Hbranches Hpath Hheights) as [Hrest [Hsup [Hbr Hblock]]].
  pose proof (code_facts_wf _ _ _ _ _ _ Hfacts) as Hwf.
  split; [exact Hrest | ]. split; [split; [exact Hsup | rewrite length_map in Hbr; exact Hbr] | ].
  split; [exact Hblock | ].
  split.
  { intros Hnil. unfold innerCode. fold position in Hpath |- *. rewrite Hnil in Hpath |- *.
    cbn in Hpath |- *. exact Hresults. }
  split; [ | exact (path_code_wf _ _ _ Hwf Hpath)].
  apply (value_labels_ready _ _ _ _ Hwf Hpath).
  - pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H. rewrite length_map in H. exact H.
  - unfold viewOperands in Hoperands. apply Forall_app in Hoperands. exact (proj1 Hoperands).
Qed.

Lemma frame_typing : forall program memory base fbp pending remaining distance f view,
  FrameAt program memory base fbp pending remaining distance f view -> FrameTyping program f view pending.
Proof. intros. unfold FrameAt in H. tauto. Qed.

Lemma frame_typing_idle : forall program f view,
  FrameTyping program f view 0 <->
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f)
    (positionLabels (viewPosition view)) (positionOffset (viewPosition view)) =
    Some (map valueTypes (viewSegments view), valueTypes (viewValues view)).
Proof.
  intros program f view. split.
  - intros [pending [Hlen Hpath]]. apply length_zero_iff_nil in Hlen. subst pending.
    now rewrite app_nil_r in Hpath.
  - intros Hpath. exists []. split; [reflexivity | now rewrite app_nil_r].
Qed.

Lemma frame_typing_heights : forall program f view pending,
  FrameTyping program f view pending ->
  pathHeights (programCallees program (guestModule f)) (guestBody f)
    (positionLabels (viewPosition view)) (positionOffset (viewPosition view)) =
    Some (map (@length val) (viewSegments view), (length (viewValues view) + pending)%nat).
Proof.
  intros program f view pending [extra [Hlen Htypes]].
  pose proof (path_types_heights _ _ _ _ _ _ _ Htypes) as H.
  unfold valueTypes in H. rewrite map_map, length_app, length_map, Hlen in H.
  erewrite map_ext in H; [exact H | ]. intros values. apply length_map.
Qed.

Lemma frame_typing_decode : forall program f view pending,
  FrameTyping program f view pending -> exists types,
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f)
    (positionLabels (viewPosition view)) (positionOffset (viewPosition view)) = Some types /\
  typeHeights types = (map (@length val) (viewSegments view), (length (viewValues view) + pending)%nat) /\
  firstn (length (operandTypes types) - pending) (operandTypes types) = valueTypes (viewOperands view).
Proof.
  intros program f view pending [extra [Hlen Hpath]]. eexists. split; [exact Hpath | ]. split.
  - unfold typeHeights, valueTypes. cbn [fst snd]. rewrite map_map, length_app, length_map, Hlen.
    f_equal. apply map_ext. intros values. apply length_map.
  - unfold operandTypes. cbn [fst snd].
    assert (Hconcat : concat (map valueTypes (viewSegments view)) = valueTypes (concat (viewSegments view))).
    { clear Hpath. unfold valueTypes. induction (viewSegments view) as [ | segment rest IH]; cbn [map concat]; [reflexivity | ].
      now rewrite map_app, IH. }
    rewrite Hconcat. unfold valueTypes, viewOperands. rewrite app_assoc, <- map_app, length_app, Hlen.
    rewrite Nat.add_sub, firstn_app, firstn_all, Nat.sub_diag. cbn [firstn]. apply app_nil_r.
Qed.

Lemma frame_ordinary_types : forall program s f view i,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  FrameTyping program f view 0 ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  (exists opcode, encodedOpcode i = Some opcode) ->
  exists next, sourceTypes (guestFrameTypes f) (programCallees program (guestModule f)) i
    (valueTypes (viewValues view)) = Some next.
Proof.
  intros program s f view i Hstore Hvalid Hf Htyping Hi Hop.
  apply frame_typing_idle in Htyping.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hpath [Hcontext [Hlocals [Hconcrete HvalidRest]]]]]]]]].
  unfold innerCode in Hi. rewrite Hpath in Hi.
  rewrite (skipn_at_instruction _ _ _ Hi) in HvalidRest.
  destruct (validation_concrete_head _ _ _ HvalidRest i _ eq_refl Hconcrete)
    as [prefix [consumed [produced [Hinput [Hhead Htail]]]]].
  exists (prefix ++ produced). rewrite Hinput, <- Hlocals.
  eapply validated_ordinary_types; eassumption.
Qed.

Lemma frame_condition_type : forall program s f view i prefix value,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  FrameTyping program f view 0 ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  consumesI32Condition i = true -> viewValues view = prefix ++ [value] -> valueType value = valtype_I32.
Proof.
  intros program s f view i prefix value Hstore Hvalid Hf Htyping Hi Hcondition Hvalues.
  apply frame_typing_idle in Htyping.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hpath [Hcontext [Hlocals [Hconcrete HvalidRest]]]]]]]]].
  unfold innerCode in Hi. rewrite Hpath in Hi.
  rewrite (skipn_at_instruction _ _ _ Hi) in HvalidRest.
  destruct (validation_concrete_head _ _ _ HvalidRest i _ eq_refl Hconcrete)
    as [below [consumed [produced [Hinput [Hhead Htail]]]]].
  destruct (validated_condition_suffix _ _ _ _ Hhead Hcondition) as [args ->].
  rewrite Hvalues in Hinput. unfold valueTypes in Hinput. rewrite map_app, app_assoc in Hinput.
  pose proof (f_equal (@rev valtype) Hinput) as Hrev.
  rewrite !rev_app_distr in Hrev. cbn [map rev app] in Hrev. now injection Hrev.
Qed.

Lemma split_stack_types : forall segments top stack,
  valueTypes stack = concat segments ++ top ->
  map valueTypes (fst (splitSegments (map (@length valtype) segments) stack)) = segments /\
  valueTypes (snd (splitSegments (map (@length valtype) segments) stack)) = top.
Proof.
  induction segments as [ | types rest IH]; intros top stack Htypes; cbn [map splitSegments fst snd] in *.
  - split; [reflexivity | exact Htypes].
  - cbn [concat] in Htypes. rewrite <- app_assoc in Htypes.
    assert (Hprefix : valueTypes (firstn (length types) stack) = types).
    { unfold valueTypes. rewrite <- firstn_map. fold (valueTypes stack). rewrite Htypes.
      rewrite firstn_app, firstn_all, Nat.sub_diag. cbn [firstn]. apply app_nil_r. }
    assert (Hsuffix : valueTypes (skipn (length types) stack) = concat rest ++ top).
    { unfold valueTypes. rewrite <- skipn_map. fold (valueTypes stack). rewrite Htypes.
      rewrite skipn_app, skipn_all, Nat.sub_diag. reflexivity. }
    destruct (splitSegments (map (@length valtype) rest) (skipn (length types) stack)) as [parts values] eqn:Hsplit.
    destruct (IH _ _ Hsuffix) as [Hparts Hvalues]. rewrite Hsplit in Hparts, Hvalues. cbn [fst snd] in Hparts, Hvalues.
    cbn [fst snd map]. split; [now rewrite Hprefix, Hparts | exact Hvalues].
Qed.

Lemma operand_types_length : forall segments top,
  length (concat segments ++ top) = totalHeight (map (@length valtype) segments, length top).
Proof.
  intros segments top. unfold totalHeight. cbn [fst snd]. rewrite length_app.
  induction segments as [ | types rest IH]; cbn [concat map fold_right]; [reflexivity | ].
  rewrite length_app. lia.
Qed.

Lemma operand_types_without_pending : forall segments top pending,
  (pending <= length top)%nat ->
  firstn (totalHeight (map (@length valtype) segments, length top) - pending) (concat segments ++ top) =
    concat segments ++ firstn (length top - pending) top.
Proof.
  intros segments top pending Hpending. rewrite <- operand_types_length, length_app, firstn_app.
  rewrite firstn_all2 by lia. f_equal. f_equal. lia.
Qed.

(* A condition's i32 type comes from SpecTec validation; its numeric range
   comes from the cell representation. Values below it can have either type. *)
Lemma frame_condition_value : forall program s memory base fbp remaining distance f view i prefix value,
  store_FUNCS s = program -> programValidated program s ->
  FrameAt program memory base fbp 0 remaining distance f view ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  consumesI32Condition i = true -> viewValues view = prefix ++ [value] ->
  exists raw, (0 <= raw < i32Modulus)%Z /\ value = i32Value raw.
Proof.
  intros program s memory base fbp remaining distance f view i prefix value Hstore Hvalid Hframe Hi Hcondition Hvalues.
  pose proof Hframe as [Hf [_ [_ [_ [_ [_ [Hcells _]]]]]]].
  pose proof (frame_condition_type _ _ _ _ _ _ _ Hstore Hvalid Hf
    (frame_typing _ _ _ _ _ _ _ _ _ Hframe) Hi Hcondition Hvalues) as Htype.
  assert (Hnth : nth_error (viewOperands view) (length (concat (viewSegments view) ++ prefix)) = Some value).
  { unfold viewOperands. rewrite Hvalues, app_assoc, nth_error_app2 by lia.
    rewrite Nat.sub_diag. reflexivity. }
  exact (cell_i32_operand _ _ _ (Hcells _ _ Hnth) Htype).
Qed.

Lemma path_code_supported : forall callees labels code inner,
  supportedCode callees code -> pathCode code labels = Some inner -> supportedCode callees inner.
Proof.
  intros callees labels. induction labels as [ | site rest IH]; intros code inner Hsupported Hpath;
    cbn [pathCode] in Hpath.
  - injection Hpath as <-. exact Hsupported.
  - destruct (siteEntry code site) as [[i body] | ] eqn:Hsite; [ | discriminate].
    apply (IH body inner); [ | exact Hpath].
    destruct (site_entry_split _ _ _ _ Hsite) as [Hi Harm].
    pose proof (proj1 (Forall_forall _ _) Hsupported _ (nth_error_In _ _ Hi)) as Hsi.
    destruct i; try discriminate; destruct (siteArm site) as [ | [ | arm]]; try discriminate;
      cbn [instrArm] in Harm.
    all: try match type of Harm with match ?code with [] => _ | _ :: _ => _ end = _ =>
      destruct code; [discriminate | ] end.
    all: injection Harm as <-; inversion Hsi; assumption.
Qed.

Lemma function_structured_resume_types : forall program s index f labels offset segments stack inner i arm body,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f) labels offset =
    Some (segments, stack) ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  instrArm i arm = Some body ->
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f) labels (S offset) =
    Some (segments, firstn (labelBase i (length stack)) stack ++ structuredResults i).
Proof.
  intros program s index f labels offset segments stack inner i arm body Hstore Hvalid Hf Htypes Hpath Hi Harm.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htypes)
    as [C [inner' [current [outputs [HC [Hpath' [Hcontext [Hlocals [Hconcrete Hrest]]]]]]]]].
  rewrite Hpath in Hpath'. injection Hpath' as <-.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hrest.
  destruct (validation_concrete_head _ _ _ Hrest i _ eq_refl Hconcrete)
    as [prefix [inputs [results [Hstack [Hhead Htail]]]]].
  pose proof (proj1 (proj2 Hvalid index f Hf)) as Hsupported.
  pose proof (path_code_supported _ _ _ _ Hsupported Hpath) as HinnerSupported.
  eapply path_types_step; [exact Hpath | exact Hi | exact Htypes | ]. rewrite Hstack.
  eapply structured_instruction_types; [exact Hhead | | exact Harm].
  exact (proj1 (Forall_forall _ _) HinnerSupported _ (nth_error_In _ _ Hi)).
Qed.

(* Falling out of a label keeps exactly its declared result types. *)
Lemma frame_label_end_typing : forall program s index f labels site segments segment values locals offset body outer i,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  FrameTyping program f (frameView (codePosition index (labels ++ [site]) offset)
    (segments ++ [segment]) values locals) 0 ->
  pathCode (guestBody f) labels = Some outer -> siteEntry outer site = Some (i, body) ->
  offset = length body ->
  FrameTyping program f (frameView (codePosition index labels (S (siteOffset site)))
    segments (segment ++ values) locals) 0.
Proof.
  intros program s index f labels site segments segment values locals offset body outer i
    Hstore Hvalid Hf Htyping Hpath Hsite Hend.
  apply frame_typing_idle in Htyping. apply frame_typing_idle.
  cbn [viewPosition positionLabels positionOffset viewSegments viewValues] in Htyping |- *.
  destruct (path_types_split _ _ labels site [] _ _ _ _ Htyping)
    as [outer' [i' [body' [heads [stack [tails [Hpath' [Hsite' [Hentry [Hsegments Htail]]]]]]]]]].
  rewrite Hpath in Hpath'. injection Hpath' as <-. rewrite Hsite in Hsite'. injection Hsite' as <- <-.
  cbn [pathTypes] in Htail.
  destruct (sourceTypeStacks _ _ _ []) as [top | ] eqn:Htop; [ | discriminate]. injection Htail as Htails Hvalues. subst tails.
  rewrite map_app in Hsegments. cbn [map] in Hsegments.
  apply app_inj_tail in Hsegments as [Hheads Hsegment]. subst heads.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hinner [Hcontext [Hlocal [Hconcrete Hrest]]]]]]]]].
  rewrite (path_context_snoc _ _ _ _ _ _ _ _ Hpath Hsite) in Hcontext.
  destruct (pathValidationContext _ _ _ labels) as [[parent resultTypes] | ]; [ | discriminate].
  injection Hcontext as <- <-.
  rewrite path_code_snoc, Hpath, Hsite in Hinner. cbn [pathCode] in Hinner. injection Hinner as <-.
  rewrite Hend, skipn_all in Hrest.
  pose proof (empty_concrete_types _ _ _ Hrest eq_refl Hconcrete) as Hresults.
  destruct (site_entry_split _ _ _ _ Hsite) as [Hi Harm].
  pose proof (function_structured_resume_types _ _ _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hf Hentry Hpath Hi Harm)
    as Hresume.
  unfold valueTypes at 2. rewrite map_app. fold (valueTypes segment) (valueTypes values).
  rewrite Hsegment, Hresults. exact Hresume.
Qed.

Lemma app_split_eq : forall (A : Type) (a b c d : list A), length a = length c -> a ++ b = c ++ d -> a = c /\ b = d.
Proof.
  intros A a b c d Hlen Heq. pose proof (f_equal (firstn (length a)) Heq) as H1.
  pose proof (f_equal (skipn (length a)) Heq) as H2.
  rewrite firstn_app, Nat.sub_diag, firstn_all, app_nil_r in H1.
  rewrite Hlen, firstn_app, Nat.sub_diag, firstn_all, app_nil_r in H1.
  rewrite skipn_app, Nat.sub_diag, skipn_all in H2. rewrite Hlen, skipn_app, Nat.sub_diag, skipn_all in H2.
  split; [exact H1 | exact H2].
Qed.

Lemma app_suffix_eq : forall (A : Type) (prefix suffix before after : list A),
  length suffix = length after -> prefix ++ suffix = before ++ after -> suffix = after.
Proof.
  intros A prefix suffix before after Hlen Heq.
  assert (Hprefix : length prefix = length before).
  { pose proof (f_equal (@length A) Heq) as H. repeat rewrite length_app in H. lia. }
  exact (proj2 (app_split_eq _ _ _ _ _ Hprefix Heq)).
Qed.

(* The saved operands below a label retain the types at its entry. *)
Definition LabelTyping (program : Program) (f : GuestFunction) (pre : list LabelSite) (site : LabelSite)
    (segments : list (list val)) (below : list val) (i : instr) : Prop :=
  exists stack,
    pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f) pre (siteOffset site) =
      Some (map valueTypes segments, stack) /\
    firstn (labelBase i (length stack)) stack = valueTypes below.

Lemma frame_label_typing : forall program f index pre site post segments below tails values locals offset outer i body,
  FrameTyping program f (frameView (codePosition index (pre ++ site :: post) offset)
    (segments ++ below :: tails) values locals) 0 ->
  length segments = length pre -> pathCode (guestBody f) pre = Some outer ->
  siteEntry outer site = Some (i, body) -> LabelTyping program f pre site segments below i.
Proof.
  intros program f index pre site post segments below tails values locals offset outer i body
    Htyping Hlen Hpath Hsite.
  apply frame_typing_idle in Htyping.
  cbn [viewPosition positionLabels positionOffset viewSegments viewValues] in Htyping.
  destruct (path_types_split _ _ pre site post _ _ _ _ Htyping)
    as [outer' [i' [body' [heads [stack [more [Hpath' [Hsite' [Hentry [Hsegments Htail]]]]]]]]]].
  rewrite Hpath in Hpath'. injection Hpath' as <-. rewrite Hsite in Hsite'. injection Hsite' as <- <-.
  pose proof (path_types_heights _ _ _ _ _ _ _ Hentry) as Hheights.
  pose proof (path_heights_length _ _ _ _ _ _ Hheights) as Hheads. rewrite length_map in Hheads.
  rewrite map_app in Hsegments. cbn [map] in Hsegments.
  destruct (app_split_eq _ (map valueTypes segments) _ heads _ ltac:(rewrite length_map; lia) Hsegments) as [Heq Hrest].
  injection Hrest as Hbelow _. subst heads. exists stack. split; [exact Hentry | symmetry; exact Hbelow].
Qed.

Lemma label_typing_resume : forall program s index f pre site segments below i outer body kept,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  LabelTyping program f pre site segments below i ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer site = Some (i, body) ->
  valueTypes kept = structuredResults i ->
  forall locals, FrameTyping program f (frameView (codePosition index pre (S (siteOffset site)))
    segments (below ++ kept) locals) 0.
Proof.
  intros program s index f pre site segments below i outer body kept Hstore Hvalid Hf [stack [Htypes Hbelow]]
    Hpath Hsite Hkept locals.
  destruct (site_entry_split _ _ _ _ Hsite) as [Hi Harm].
  apply frame_typing_idle. cbn [viewPosition positionLabels positionOffset viewSegments viewValues].
  unfold valueTypes at 2. rewrite map_app. fold (valueTypes below) (valueTypes kept).
  rewrite Hkept, <- Hbelow. eapply function_structured_resume_types; eassumption.
Qed.

Lemma frame_branch_types : forall program s index f pre site post segments below tails values locals offset outer ti body i l
    dropped kept condition,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  FrameTyping program f (frameView (codePosition index (pre ++ site :: post) offset)
    (segments ++ below :: tails) values locals) 0 ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer site = Some (ti, body) ->
  nth_error (innerCode f (codePosition index (pre ++ site :: post) offset)) offset = Some i ->
  branchTargets i l -> N.to_nat (l :> N) = length post ->
  values = (dropped ++ kept) ++ condition -> valueTypes condition = branchConditionTypes i ->
  length kept = length (structuredLabel ti) -> valueTypes kept = structuredLabel ti.
Proof.
  intros program s index f pre site post segments below tails values locals offset outer ti body i l
    dropped kept condition Hstore Hvalid Hf Htyping Hpath Hsite Hi Htarget Hlevel Hvalues Hcondition Hlength.
  apply frame_typing_idle in Htyping.
  cbn [viewPosition positionLabels positionOffset viewSegments viewValues] in Htyping.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hinner [Hcontext [Hlocal [Hconcrete Hrest]]]]]]]]].
  unfold innerCode in Hi. cbn [positionLabels] in Hi. rewrite Hinner in Hi.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hrest.
  destruct (validation_concrete_head _ _ _ Hrest i _ eq_refl Hconcrete)
    as [prefix [inputs [results [Hinput [Hhead Htail]]]]].
  pose proof (path_context_target _ _ _ _ _ _ _ _ _ _ _ Hpath Hsite Hcontext) as Hlabel.
  rewrite <- Hlevel in Hlabel.
  assert (HconcreteInput : concreteTypes inputs)
    by (rewrite Hinput in Hconcrete; exact (proj2 (proj1 (Forall_app _ _ _) Hconcrete))).
  destruct (validated_branch_suffix _ _ _ _ _ _ Hhead Htarget Hlabel HconcreteInput) as [discarded Hinputs].
  rewrite Hinputs, Hvalues in Hinput. unfold valueTypes in Hinput. rewrite !map_app in Hinput.
  fold (valueTypes dropped) (valueTypes kept) (valueTypes condition) in Hinput.
  rewrite Hcondition in Hinput. rewrite !app_assoc in Hinput.
  apply app_inv_tail in Hinput.
  assert (HprefixLength : length (valueTypes dropped) = length (prefix ++ discarded)).
  { pose proof (f_equal (@length valtype) Hinput) as Hlen.
    unfold valueTypes in Hlen |- *. repeat rewrite length_app in Hlen. repeat rewrite length_map in Hlen.
    rewrite length_map, length_app. lia. }
  exact (proj2 (app_split_eq _ _ _ _ _ HprefixLength Hinput)).
Qed.

(* The source call still carries the index type while an indirect call is
   resolved. Only the arguments remain in its suspended operand view. *)
Lemma frame_call_argument_types : forall program s f view pending i callee prefix arguments,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  FrameTyping program f view pending ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  callsInto (programCallees program (guestModule f)) i callee ->
  pending = length (callIndexTypes i) -> viewValues view = prefix ++ arguments ->
  length arguments = guestParams callee -> valueTypes arguments = guestParamTypes callee.
Proof.
  intros program s f view pending i callee prefix arguments Hstore Hvalid Hf
    [pendingTypes [Hpending Htyping]] Hi Hcall Hindex Hvalues Hlength.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hpath [Hcontext [Hlocals [Hconcrete Hrest]]]]]]]]].
  unfold innerCode in Hi. rewrite Hpath in Hi. rewrite (skipn_at_instruction _ _ _ Hi) in Hrest.
  destruct (validation_concrete_head _ _ _ Hrest i _ eq_refl Hconcrete)
    as [below [inputs [results [Hinput [Hhead Htail]]]]].
  destruct (path_context_functions _ _ _ _ _ _ Hcontext) as [Hfuncs Htypes].
  assert (Hcallees : calleeTypes current (programCallees program (guestModule f))).
  { unfold calleeTypes. rewrite Hfuncs. exact (program_callee_types _ _ _ _ Hstore HC). }
  assert (HmoduleTypes : context_TYPES current = envTypes (programCallees program (guestModule f))).
  { rewrite Htypes. exact (proj1 (module_context_tables _ _ _ HC)). }
  destruct (validated_call_signature _ _ _ _ _ _ Hhead Hcall Hcallees HmoduleTypes) as [Hinputs _].
  rewrite Hinputs, Hvalues in Hinput. unfold valueTypes in Hinput. rewrite map_app, <- app_assoc in Hinput.
  assert (HprefixLength : length (valueTypes prefix) = length below).
  { pose proof (f_equal (@length valtype) Hinput) as Hlen.
    repeat rewrite length_app in Hlen. repeat rewrite length_map in Hlen.
    unfold valueTypes. rewrite length_map. unfold guestParams in Hlength. lia. }
  destruct (app_split_eq _ _ _ _ _ HprefixLength Hinput) as [_ Hargs].
  apply (proj1 (app_split_eq _ _ _ _ _ ltac:(unfold valueTypes; rewrite length_map; exact Hlength) Hargs)).
Qed.

Lemma frame_call_resume_types : forall program s f view i callee returned,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  FrameTyping program f view (callOperands i callee) ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  callsInto (programCallees program (guestModule f)) i callee ->
  valueTypes returned = guestResultTypes callee ->
  FrameTyping program f (frameView
    (codePosition (positionFunction (viewPosition view)) (positionLabels (viewPosition view))
      (S (positionOffset (viewPosition view))))
    (viewSegments view) (viewValues view ++ returned) (viewLocals view)) 0.
Proof.
  intros program s f view i callee returned Hstore Hvalid Hf [pending [Hpending Htyping]] Hi Hcall Hreturned.
  destruct (function_path_validation _ _ _ _ _ _ _ _ Hstore Hvalid Hf Htyping)
    as [C [inner [current [outputs [HC [Hpath [Hcontext [Hlocals [Hconcrete Hrest]]]]]]]]].
  unfold innerCode in Hi. rewrite Hpath in Hi. rewrite (skipn_at_instruction _ _ _ Hi) in Hrest.
  destruct (validation_concrete_head _ _ _ Hrest i _ eq_refl Hconcrete)
    as [below [inputs [results [Hinput [Hhead Htail]]]]].
  destruct (path_context_functions _ _ _ _ _ _ Hcontext) as [Hfuncs Htypes].
  assert (Hcallees : calleeTypes current (programCallees program (guestModule f))).
  { unfold calleeTypes. rewrite Hfuncs. exact (program_callee_types _ _ _ _ Hstore HC). }
  assert (HmoduleTypes : context_TYPES current = envTypes (programCallees program (guestModule f))).
  { rewrite Htypes. exact (proj1 (module_context_tables _ _ _ HC)). }
  destruct (validated_call_signature _ _ _ _ _ _ Hhead Hcall Hcallees HmoduleTypes) as [Hinputs _].
  rewrite Hinputs in Hinput.
  assert (HprefixLength : length (valueTypes (viewValues view)) = length below).
  { pose proof (f_equal (@length valtype) Hinput) as Hlen. repeat rewrite length_app in Hlen.
    rewrite <- (call_index_arity i callee) in Hpending. unfold guestParams in Hpending. lia. }
  destruct (app_split_eq _ _ _ _ _ HprefixLength Hinput) as [Hbelow Hargs].
  apply frame_typing_idle. cbn [viewPosition positionLabels positionOffset viewSegments viewValues].
  unfold valueTypes at 2. rewrite map_app. fold (valueTypes (viewValues view)) (valueTypes returned).
  rewrite Hreturned. eapply path_types_step; [exact Hpath | exact Hi | exact Htyping | ].
  rewrite Hargs. apply calls_into_types. exact Hcall.
Qed.

Lemma function_integer_defaults : forall program index f,
  functionAt program index = Some f -> exists defaults, integerZeros (guestLocalTypes f) = Some defaults.
Proof.
  intros program index f Hf. destruct (function_integer_types _ _ _ Hf) as [_ [_ Hlocals]].
  induction Hlocals as [ | t types Ht Htypes [defaults Hdefaults]].
  - exists []; reflexivity.
  - destruct Ht as [-> | ->]; cbn [integerZeros]; rewrite Hdefaults; eexists; reflexivity.
Qed.
