(* What a frame at a code path inherits from the program's validation, and how
   code paths, heights and enclosing labels compose. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerValidation FunctionValidation ProgramSpec ProgramEncoding CodeLayout Fragment CodeFacts ControlDecode
  ControlInvariant.
Import ListNotations.
Open Scope nat_scope.

(* Paths extended by one more enclosing block. *)
Lemma path_code_snoc : forall labels code b,
  pathCode code (labels ++ [b]) =
    match pathCode code labels with
    | Some inner => match nth_error inner b with Some (BLOCK _ body) => Some body | _ => None end
    | None => None
    end.
Proof.
  induction labels as [ | b' rest IH]; intros code b; cbn [app pathCode].
  - destruct (nth_error code b) as [i | ]; [destruct i | ]; reflexivity.
  - destruct (nth_error code b') as [i | ]; [destruct i | ]; try reflexivity. apply IH.
Qed.

Lemma path_offset_snoc : forall labels code inner b bt body offset,
  pathCode code labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  pathOffset code (labels ++ [b]) offset = (pathOffset code labels b + S (codeOffset body offset))%nat.
Proof.
  induction labels as [ | b' rest IH]; intros code inner b bt body offset Hpath Hb; cbn [app pathCode pathOffset] in *.
  - injection Hpath as <-. rewrite Hb. reflexivity.
  - destruct (nth_error code b') as [i | ]; [ | discriminate]. destruct i; try discriminate.
    rewrite (IH _ _ _ _ _ offset Hpath Hb). lia.
Qed.

Lemma path_heights_snoc : forall callees labels code inner b bt body heights below offset,
  pathCode code labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  pathHeights callees code labels b = Some (heights, below) ->
  pathHeights callees code (labels ++ [b]) offset =
    option_map (fun top => (heights ++ [below], top)) (sourceHeights callees (firstn offset body) 0).
Proof.
  intros callees. induction labels as [ | b' rest IH];
    intros code inner b bt body heights below offset Hpath Hb Hheights; cbn [app pathCode pathHeights] in *.
  - injection Hpath as <-. rewrite Hb.
    destruct (sourceHeights callees (firstn b code) 0) as [h | ]; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-. cbn. destruct (sourceHeights callees (firstn offset body) 0); reflexivity.
  - destruct (nth_error code b') as [i | ]; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b' code) 0) as [h | ]; [ | discriminate].
    match goal with Hp : pathCode ?nested rest = Some inner |- _ =>
      destruct (pathHeights callees nested rest b) as [[hs top] | ] eqn:Hinner; cbn in Hheights;
        [ | discriminate] end.
    injection Hheights as <- <-. rewrite (IH _ _ _ _ _ _ _ offset Hpath Hb Hinner).
    destruct (sourceHeights callees (firstn offset body) 0); reflexivity.
Qed.

Lemma wrap_labels_snoc : forall labels code inner b bt body segments segment c,
  pathCode code labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  length segments = length labels ->
  wrapLabels code (labels ++ [b]) (segments ++ [segment]) c =
    wrapLabels code labels segments
      (stackPrefix segment (instructionSuffix (skipn (S b) inner) (blockContext c))).
Proof.
  induction labels as [ | b' rest IH]; intros code inner b bt body segments segment c Hpath Hb Hlen.
  - destruct segments; [ | discriminate]. injection Hpath as <-. cbn [app wrapLabels]. rewrite Hb. reflexivity.
  - destruct segments as [ | seg more]; [discriminate | ]. cbn [app wrapLabels pathCode] in *.
    destruct (nth_error code b') as [i | ]; [ | discriminate]. destruct i; try discriminate.
    rewrite (IH _ _ _ _ _ more segment c Hpath Hb ltac:(cbn in Hlen; lia)). reflexivity.
Qed.

(* Paths split at an enclosing block: the labels before it, it, and those inside. *)
Lemma path_code_app : forall pre post code,
  pathCode code (pre ++ post) = match pathCode code pre with Some inner => pathCode inner post | None => None end.
Proof.
  induction pre as [ | b rest IH]; intros post code; cbn [app pathCode]; [reflexivity | ].
  destruct (nth_error code b) as [i | ]; [destruct i | ]; try reflexivity. apply IH.
Qed.

Lemma wrap_labels_app : forall pre post code outer segmentsPre segmentsPost c,
  pathCode code pre = Some outer -> length segmentsPre = length pre ->
  wrapLabels code (pre ++ post) (segmentsPre ++ segmentsPost) c =
    wrapLabels code pre segmentsPre (wrapLabels outer post segmentsPost c).
Proof.
  induction pre as [ | b rest IH]; intros post code outer segmentsPre segmentsPost c Hpath Hlen.
  - destruct segmentsPre; [ | discriminate]. injection Hpath as <-. reflexivity.
  - destruct segmentsPre as [ | seg more]; [discriminate | ]. cbn [app wrapLabels pathCode] in *.
    destruct (nth_error code b) as [i | ]; [ | discriminate]. destruct i; try discriminate.
    rewrite (IH _ _ _ more _ c Hpath ltac:(cbn in Hlen; lia)). reflexivity.
Qed.

Lemma path_heights_split : forall callees pre b post code offset heights top,
  pathHeights callees code (pre ++ b :: post) offset = Some (heights, top) ->
  exists outer bt body headHeights below tailHeights,
    pathCode code pre = Some outer /\ nth_error outer b = Some (BLOCK bt body) /\
    pathHeights callees code pre b = Some (headHeights, below) /\ heights = headHeights ++ below :: tailHeights /\
    pathHeights callees body post offset = Some (tailHeights, top).
Proof.
  intros callees. induction pre as [ | b0 rest IH]; intros b post code offset heights top H;
    cbn [app pathHeights] in H.
  - destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    match goal with Hb : nth_error code b = Some (BLOCK ?bt ?body) |- _ =>
      destruct (pathHeights callees body post offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate];
      injection H as <- <-; exists code, bt, body, [], below, hs end.
    cbn [pathCode pathHeights]. rewrite Hbelow. repeat split; assumption || reflexivity.
  - destruct (nth_error code b0) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b0 code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    match goal with H : option_map _ (pathHeights callees ?nested (rest ++ b :: post) offset) = _ |- _ =>
      destruct (pathHeights callees nested (rest ++ b :: post) offset) as [[hs t] | ] eqn:Hin; cbn in H;
        [ | discriminate];
      injection H as <- <-;
      destruct (IH _ _ _ _ _ _ Hin) as [outer [bt [body [hs' [below' [hs'' [Hpath [Hblock [Hheights [-> Htail]]]]]]]]]];
      exists outer, bt, body, (below :: hs'), below', hs'' end.
    cbn [pathCode pathHeights]. rewrite Hb, Hbelow, Hheights. repeat split; assumption || reflexivity.
Qed.

(* The operands below each label, then the innermost ones. *)
Lemma split_concat : forall segments values,
  splitSegments (map (@length Z) segments) (concat segments ++ values) = (segments, values).
Proof.
  induction segments as [ | seg more IH]; intros values; [reflexivity | ].
  cbn [map concat splitSegments]. rewrite <- app_assoc, firstn_app, Nat.sub_diag, firstn_all, skipn_app,
    skipn_all, Nat.sub_diag. cbn [firstn skipn app]. rewrite app_nil_r, IH. reflexivity.
Qed.

Lemma total_height_operands : forall segments values pending,
  totalHeight (map (@length Z) segments, (length values + pending)%nat) =
    (length (concat segments ++ values) + pending)%nat.
Proof.
  intros. unfold totalHeight. cbn [fst snd]. rewrite length_app.
  induction segments as [ | seg more IH]; cbn; [reflexivity | ]. rewrite length_app. lia.
Qed.

Lemma split_lengths : forall heights stack,
  (fold_right Nat.add 0 heights <= length stack)%nat ->
  map (@length Z) (fst (splitSegments heights stack)) = heights /\
  concat (fst (splitSegments heights stack)) ++ snd (splitSegments heights stack) = stack.
Proof.
  induction heights as [ | h rest IH]; intros stack Hlen; [split; reflexivity | ].
  cbn [fold_right] in Hlen. cbn [splitSegments fst snd map concat].
  destruct (IH (skipn h stack) ltac:(rewrite length_skipn; lia)) as [Hmap Hconcat].
  split; [rewrite length_firstn, Hmap; f_equal; lia | ].
  rewrite <- app_assoc, Hconcat. apply firstn_skipn.
Qed.

(* Enclosing labels that are ready to run their inner code: their operands are
   i32 values and the code after each END is well formed. *)
Fixpoint LabelsReady (code : list instr) (labels : list nat) (segments : list (list Z)) : Prop :=
  match labels, segments with
  | b :: rest, segment :: more =>
      Forall (fun value => 0 <= value < i32Modulus)%Z segment /\ Forall wf_instr (skipn (S b) code) /\
      match nth_error code b with Some (BLOCK _ body) => LabelsReady body rest more | _ => True end
  | _, _ => True
  end.

Lemma forall_skipn : forall (A : Type) (P : A -> Prop) k xs, Forall P xs -> Forall P (skipn k xs).
Proof.
  intros A P k xs H. apply Forall_forall. intros x Hx. apply (proj1 (Forall_forall _ _) H).
  rewrite <- (firstn_skipn k xs). apply in_or_app. right. exact Hx.
Qed.

Lemma path_code_wf : forall labels code inner,
  Forall wf_instr code -> pathCode code labels = Some inner -> Forall wf_instr inner.
Proof.
  induction labels as [ | b rest IH]; intros code inner Hwf Hpath; cbn [pathCode] in Hpath.
  - injection Hpath as <-. exact Hwf.
  - destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    eapply IH; [ | exact Hpath].
    pose proof (proj1 (Forall_forall _ _) Hwf _ (nth_error_In _ _ Hb)) as Hblock. inversion Hblock; assumption.
Qed.

Lemma labels_ready : forall labels code inner segments,
  Forall wf_instr code -> pathCode code labels = Some inner -> length segments = length labels ->
  Forall (fun value => 0 <= value < i32Modulus)%Z (concat segments) -> LabelsReady code labels segments.
Proof.
  induction labels as [ | b rest IH]; intros code inner segments Hwf Hpath Hlen Hvalues; [exact I | ].
  destruct segments as [ | seg more]; [discriminate | ]. cbn [concat] in Hvalues. apply Forall_app in Hvalues.
  cbn [pathCode] in Hpath. cbn [LabelsReady].
  split; [exact (proj1 Hvalues) | ]. split; [apply forall_skipn; exact Hwf | ].
  destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
  eapply IH; [ | exact Hpath | cbn in Hlen; lia | exact (proj2 Hvalues)].
  pose proof (proj1 (Forall_forall _ _) Hwf _ (nth_error_In _ _ Hb)) as Hblock. inversion Hblock; assumption.
Qed.

Lemma steps_in_labels : forall labels code segments before after,
  LabelsReady code labels segments -> Steps before after ->
  Steps (wrapLabels code labels segments before) (wrapLabels code labels segments after).
Proof.
  induction labels as [ | b rest IH]; intros code segments before after Hready Hsteps; [exact Hsteps | ].
  destruct segments as [ | seg more]; [exact Hsteps | ]. destruct Hready as [Hseg [Hcont Hinner]].
  cbn [wrapLabels]. destruct (nth_error code b) as [i | ]; [ | exact Hsteps]. destruct i; try exact Hsteps.
  apply steps_with_stack_prefix; [exact Hseg | ]. apply steps_with_instruction_suffix; [exact Hcont | ].
  apply steps_in_block. exact (IH _ _ _ _ Hinner Hsteps).
Qed.

Lemma wrap_labels_wf : forall labels code segments c,
  LabelsReady code labels segments -> wf_config c -> wf_config (wrapLabels code labels segments c).
Proof.
  induction labels as [ | b rest IH]; intros code segments c Hready Hwf; [exact Hwf | ].
  destruct segments as [ | seg more]; [exact Hwf | ]. destruct Hready as [Hseg [Hcont Hinner]].
  cbn [wrapLabels]. destruct (nth_error code b) as [i | ]; [ | exact Hwf]. destruct i; try exact Hwf.
  apply stack_prefix_wf; [exact Hseg | ]. apply instruction_suffix_wf; [exact Hcont | ].
  pose proof (IH _ _ _ Hinner Hwf) as H.
  match type of H with wf_config ?inner => destruct inner as [z code'] end.
  apply block_context_wf. exact H.
Qed.

(* A branch out of labels whose operands are all gone: br k+n inside n labels
   leaves them as br k, dropping the code after each one's END. *)
Lemma wrap_labels_exit : forall labels code inner segments z m rest,
  pathCode code labels = Some inner -> length segments = length labels ->
  Forall (fun segment : list Z => segment = []) segments -> LabelsReady code labels segments ->
  wf_state z -> wf_uN 32%N (mk_uN (N.of_nat (length labels + m))) -> Forall wf_instr rest ->
  exists rest', Forall wf_instr rest' /\
    Steps (wrapLabels code labels segments (stackConfig z [] (BR (mk_uN (N.of_nat (length labels + m))) :: rest)))
      (stackConfig z [] (BR (mk_uN (N.of_nat m)) :: rest')).
Proof.
  induction labels as [ | b more IH]; intros code inner segments z m rest Hpath Hlen Hempty Hready Hz Hl Hrest.
  - destruct segments; [ | discriminate]. exists rest. split; [exact Hrest | ]. cbn [wrapLabels length Nat.add].
    apply Steps__refl. apply stack_config_wf; [exact Hz | constructor | ]. constructor; [constructor; exact Hl | exact Hrest].
  - destruct segments as [ | seg segments]; [discriminate | ]. inversion Hempty as [ | ? ? -> Hempties]; subst.
    destruct Hready as [_ [Hcont Hinner]]. cbn [pathCode wrapLabels] in *.
    destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    replace (length (b :: more) + m) with (length more + S m) in * by (cbn; lia).
    destruct (IH _ _ segments z (S m) rest Hpath ltac:(cbn in Hlen; lia) Hempties Hinner Hz Hl Hrest)
      as [rest' [Hrest' Hsteps]].
    replace (N.of_nat (S m)) with (N.of_nat m + 1)%N in Hsteps by lia.
    assert (Hk : wf_uN 32%N (mk_uN (N.of_nat m + 1)%N)) by (eapply wf_label_le; [ | exact Hl]; lia).
    exists (skipn (S b) code). split; [exact Hcont | ].
    change (stackConfig z [] (BR (mk_uN (N.of_nat m)) :: skipn (S b) code)) with
      (stackPrefix [] (instructionSuffix (skipn (S b) code) (stackConfig z [] [BR (mk_uN (N.of_nat m))]))).
    apply (steps_with_stack_prefix [] _ _ (Forall_nil _)). apply (steps_with_instruction_suffix _ _ _ Hcont).
    eapply spec_steps_transitive; [exact (steps_in_block _ _ Hsteps) | ].
    exact (br_exit_label z (N.of_nat m) rest' Hz Hk Hrest').
Qed.

(* The blocks a branch may leave without discarding operands, from the operand
   heights at which the enclosing blocks were entered, outermost first. *)
Definition labelExits (exits : nat) (heights : list nat) : nat :=
  fold_left (fun exits height => if Nat.eqb height 0 then S exits else 1) heights exits.

(* A branch may leave the labels whose operands below are empty, innermost first. *)
Lemma label_exits_split : forall heights n, n < labelExits 0 heights ->
  exists headHeights height tailHeights, heights = headHeights ++ height :: tailHeights /\
    length tailHeights = n /\ Forall (fun h => h = 0) tailHeights.
Proof.
  induction heights as [ | height heights IH] using rev_ind; intros n Hn; [cbn in Hn; lia | ].
  unfold labelExits in Hn. rewrite fold_left_app in Hn. cbn in Hn. fold (labelExits 0 heights) in Hn.
  destruct (Nat.eqb height 0) eqn:Hzero.
  - destruct n as [ | n].
    + exists heights, height, []. repeat split; constructor.
    + destruct (IH n ltac:(lia)) as [hs [h [tail [-> [Htail Hzeros]]]]].
      exists hs, h, (tail ++ [height]). rewrite <- app_assoc. split; [reflexivity | ].
      split; [rewrite length_app; cbn; lia | ]. apply Forall_app. split; [exact Hzeros | ].
      constructor; [apply Nat.eqb_eq; exact Hzero | constructor].
  - assert (n = 0) by lia. subst n. exists heights, height, []. repeat split; constructor.
Qed.

Lemma path_heights_length : forall callees labels code offset heights top,
  pathHeights callees code labels offset = Some (heights, top) -> length heights = length labels.
Proof.
  intros callees. induction labels as [ | b rest IH]; intros code offset heights top H; cbn [pathHeights] in H.
  - destruct (sourceHeights callees (firstn offset code) 0); cbn in H; [ | discriminate]. now injection H as <- _.
  - destruct (nth_error code b) as [i | ]; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0); [ | discriminate].
    match goal with H : option_map _ (pathHeights callees ?nested rest offset) = _ |- _ =>
      destruct (pathHeights callees nested rest offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate];
      injection H as <- _; cbn; f_equal; exact (IH _ _ _ _ Hin) end.
Qed.

(* Callee types and validation come from the module context SpecTec instantiated. *)
Lemma program_callee_types : forall program s module C,
  store_FUNCS s = program -> Moduleinst_ok s module C ->
  calleeTypes C (programCallees program module).
Proof.
  intros program s module C Hstore Hmodule x f Hf.
  unfold programCallees in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ] eqn:Ha; [ | discriminate].
  destruct (function_at_sound _ _ _ Hf) as [Hfi _].
  exact (module_function_type _ _ _ _ _ _ Hmodule Ha ltac:(rewrite Hstore; exact Hfi)).
Qed.

Lemma program_callee_validates : forall program s module x f,
  Forall (fun fi => Funcinst_ok s fi (funcinst_TYPE fi)) program ->
  programCallees program module x = Some f ->
  Funcinst_ok s (i32Function (guestModule f) (guestTypeIndex f)
    (guestParams f) (guestResults f) (guestLocals f) (guestBody f))
    (i32StackType (guestParams f) (guestResults f)).
Proof.
  intros program s module x f Hvalid Hf. unfold programCallees in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ]; [ | discriminate].
  destruct (function_at_sound _ _ _ Hf) as [Hfi _].
  rewrite Forall_forall in Hvalid. apply Hvalid. eapply nth_error_In; exact Hfi.
Qed.

(* A function's body: its facts from height 0, its result height, its support,
   and its branch sites. *)
Lemma function_facts : forall program s index f,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  let callees := programCallees program (guestModule f) in
  codeFacts (frameLocals f) callees 0 (guestBody f) /\
  sourceHeights callees (guestBody f) 0 = Some (guestResults f) /\
  supportedCode callees (guestBody f) /\ codeBranches callees 0 (guestBody f) 0.
Proof.
  intros program s index f Hstore [Hfuncs Hsupported] Hf callees.
  destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) Hfuncs _ (nth_error_In _ _ Hnth)) as Hinst.
  destruct (funcinst_body_validated _ _ _ _ _ _ _ Hinst) as [C [HC Hbody]].
  pose proof (code_validation_facts (S (codeSize (guestBody f))) _ _ _ Hbody ltac:(lia)) as H.
  cbn [i32StackType] in H.
  destruct (Hsupported index f Hf) as [Hsupport Hbranches].
  assert (Hfits : contextFits (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) (frameLocals f) callees).
  { split; [rewrite function_context_locals by (eapply module_context_locals; exact HC); reflexivity | ].
    intros x g Hg. exact (program_callee_types _ _ _ _ Hstore HC x g Hg). }
  destruct (H (frameLocals f) _ Hfits Hsupport) as [Hfacts Hheights].
  rewrite !repeat_length in Hfacts, Hheights.
  split; [pose proof (Hfacts 0%nat [] I) as Hf0; rewrite app_nil_r in Hf0; exact Hf0 | ].
  split; [exact (Hheights 0%nat) | split; assumption].
Qed.

(* Facts of the code at a path, from the enclosing code's. *)
Lemma path_facts : forall localCount callees labels code exits inner offset heights top,
  codeFacts localCount callees 0 code -> supportedCode callees code -> codeBranches callees exits code 0 ->
  pathCode code labels = Some inner -> pathHeights callees code labels offset = Some (heights, top) ->
  codeFacts localCount callees top (skipn offset inner) /\ supportedCode callees (skipn offset inner) /\
  codeBranches callees (labelExits exits heights) (skipn offset inner) top /\
  (labels <> [] -> sourceHeights callees inner 0 = Some 0).
Proof.
  intros localCount callees. induction labels as [ | b rest IH];
    intros code exits inner offset heights top Hfacts Hsupport Hbranches Hpath Hheights;
    cbn [pathCode pathHeights] in *.
  - injection Hpath as <-.
    destruct (sourceHeights callees (firstn offset code) 0) as [h | ] eqn:Hh; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    split; [exact (code_facts_skip _ _ _ _ _ _ Hfacts Hh) | ]. split; [apply supported_skip; exact Hsupport | ].
    split; [exact (code_branches_skip _ _ _ _ _ _ Hbranches Hh) | ]. intros Hne; contradiction.
  - destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    match goal with Hp : pathCode ?nested rest = Some inner |- _ => rename nested into body end.
    destruct (pathHeights callees body rest offset) as [[hs t] | ] eqn:Hin; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    pose proof (code_facts_skip _ _ _ _ _ _ Hfacts Hbelow) as Hat.
    rewrite (skipn_at_instruction _ _ _ Hb) in Hat. cbn [codeFacts] in Hat. destruct Hat as [Hblock _].
    destruct (proj1 (block_facts _ _ _ _ _) Hblock) as [_ [Hzero Hbody]].
    pose proof (code_branches_skip _ _ _ _ _ _ Hbranches Hbelow) as Hbr.
    rewrite (skipn_at_instruction _ _ _ Hb) in Hbr. destruct Hbr as [Hbr _].
    apply (proj1 (block_branches _ _ _ _ _)) in Hbr.
    pose proof (proj1 (Forall_forall _ _) Hsupport _ (nth_error_In _ _ Hb)) as Hsup.
    inversion Hsup as [ | | | | x f Hf Hp Hr | body' Hbs | l ]; subst.
    destruct (IH body _ inner offset hs t Hbody Hbs Hbr Hpath Hin) as [Hf [Hs [Hbranch Hdeeper]]].
    split; [exact Hf | ]. split; [exact Hs | ]. split; [exact Hbranch | ].
    intros _. destruct rest as [ | b2 rest2].
    + cbn [pathCode] in Hpath. injection Hpath as <-. exact Hzero.
    + apply Hdeeper. discriminate.
Qed.

(* What a frame at its position inherits from the program's validation. *)
Lemma view_facts : forall program s memory base fbp pending moved distance f view,
  store_FUNCS s = program -> programValidated program s ->
  FrameAt program memory base fbp pending moved distance f view ->
  let position := viewPosition view in
  let callees := programCallees program (guestModule f) in
  let rest := skipn (positionOffset position) (innerCode f position) in
  codeFacts (frameLocals f) callees (length (viewValues view) + pending) rest /\
  (supportedCode callees rest /\
    codeBranches callees (labelExits 0 (map (@length Z) (viewSegments view))) rest (length (viewValues view) + pending)) /\
  (positionLabels position <> [] -> sourceHeights callees (innerCode f position) 0 = Some 0) /\
  (positionLabels position = [] -> sourceHeights callees (innerCode f position) 0 = Some (guestResults f)) /\
  LabelsReady (guestBody f) (positionLabels position) (viewSegments view) /\
  Forall wf_instr (innerCode f position).
Proof.
  intros program s memory base fbp pending moved distance f view Hstore Hvalid
    [Hf [Hpath [Hheights [_ [_ [Hoperands _]]]]]] position callees rest.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts [Hresults [Hsupport Hbranches]]].
  destruct (path_facts _ _ _ _ _ _ _ _ _ Hfacts Hsupport Hbranches Hpath Hheights) as [Hrest [Hsup [Hbr Hblock]]].
  pose proof (code_facts_wf _ _ _ _ Hfacts) as Hwf.
  split; [exact Hrest | ]. split; [split; [exact Hsup | exact Hbr] | ]. split; [exact Hblock | ].
  split.
  { intros Hnil. unfold innerCode. fold position in Hpath |- *. rewrite Hnil in Hpath |- *.
    cbn in Hpath |- *. exact Hresults. }
  split; [ | exact (path_code_wf _ _ _ Hwf Hpath)].
  apply (labels_ready _ _ _ _ Hwf Hpath).
  - pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H. rewrite length_map in H. exact H.
  - unfold viewOperands in Hoperands. apply Forall_app in Hoperands. exact (proj1 Hoperands).
Qed.
