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

(* A function's body: its facts from height 0, its result height, and its support. *)
Lemma function_facts : forall program s index f,
  store_FUNCS s = program -> programValidated program s -> functionAt program index = Some f ->
  codeFacts (frameLocals f) (programCallees program (guestModule f)) 0 (guestBody f) /\
  sourceHeights (programCallees program (guestModule f)) (guestBody f) 0 = Some (guestResults f) /\
  supportedCode (codeSite (programCallees program (guestModule f)) false) (guestBody f).
Proof.
  intros program s index f Hstore [Hfuncs Hsupported] Hf.
  destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) Hfuncs _ (nth_error_In _ _ Hnth)) as Hinst.
  destruct (funcinst_body_validated _ _ _ _ _ _ _ Hinst) as [C [HC Hbody]].
  pose proof (code_validation_facts (S (codeSize (guestBody f))) _ _ _ Hbody ltac:(lia)) as H.
  cbn [i32StackType] in H.
  assert (Hfits : contextFits (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) (frameLocals f)
      (programCallees program (guestModule f)) false).
  { split; [rewrite function_context_locals by (eapply module_context_locals; exact HC); reflexivity | ].
    split; [ | discriminate]. intros x g Hg. exact (program_callee_types _ _ _ _ Hstore HC x g Hg). }
  destruct (H (frameLocals f) _ false Hfits (Hsupported index f Hf)) as [Hfacts Hheights].
  rewrite !repeat_length in Hfacts, Hheights.
  split; [pose proof (Hfacts 0%nat [] I) as Hf0; rewrite app_nil_r in Hf0; exact Hf0 | ].
  split; [exact (Hheights 0%nat) | exact (Hsupported index f Hf)].
Qed.

(* Facts of the code at a path, from the enclosing code's. *)
Lemma path_facts : forall localCount callees labels code inside inner offset heights top,
  codeFacts localCount callees 0 code -> supportedCode (codeSite callees inside) code ->
  pathCode code labels = Some inner -> pathHeights callees code labels offset = Some (heights, top) ->
  codeFacts localCount callees top (skipn offset inner) /\
  supportedCode (codeSite callees (match labels with [] => inside | _ => true end)) (skipn offset inner) /\
  (labels <> [] -> sourceHeights callees inner 0 = Some 0 /\
    branchSitesBalanced callees (skipn offset inner) top).
Proof.
  intros localCount callees. induction labels as [ | b rest IH];
    intros code inside inner offset heights top Hfacts Hsupport Hpath Hheights; cbn [pathCode pathHeights] in *.
  - injection Hpath as <-.
    destruct (sourceHeights callees (firstn offset code) 0) as [h | ] eqn:Hh; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    split; [exact (code_facts_skip _ _ _ _ _ _ Hfacts Hh) | ]. split; [apply supported_skip; exact Hsupport | ].
    intros Hne; contradiction.
  - destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    match goal with Hp : pathCode ?nested rest = Some inner |- _ => rename nested into body end.
    destruct (pathHeights callees body rest offset) as [[hs t] | ] eqn:Hin; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-.
    pose proof (code_facts_skip _ _ _ _ _ _ Hfacts Hbelow) as Hat.
    rewrite (skipn_at_instruction _ _ _ Hb) in Hat. cbn [codeFacts] in Hat. destruct Hat as [Hblock _].
    destruct (proj1 (block_facts _ _ _ _ _) Hblock) as [_ [Hzero Hbody]].
    pose proof (proj1 (Forall_forall _ _) Hsupport _ (nth_error_In _ _ Hb)) as Hsup.
    inversion Hsup as [ | | | | | callees' inside' body' Hbs Hbal | ]; subst.
    destruct (IH body true inner offset hs t Hbody Hbs Hpath Hin) as [Hf [Hs Hdeeper]].
    split; [exact Hf | ]. split; [destruct rest; exact Hs | ].
    intros _. destruct rest as [ | b2 rest2].
    + cbn [pathCode] in Hpath. injection Hpath as <-. split; [exact Hzero | ].
      cbn [pathHeights] in Hin.
      destruct (sourceHeights callees (firstn offset body) 0) as [h | ] eqn:Hh; cbn in Hin; [ | discriminate].
      injection Hin as _ <-. exact (balanced_skip _ _ _ _ _ Hbal Hh).
    + apply Hdeeper. discriminate.
Qed.

(* What a frame at its position inherits from the program's validation. *)
Lemma view_facts : forall program s memory base fbp pending f view,
  store_FUNCS s = program -> programValidated program s -> FrameAt program memory base fbp pending f view ->
  let position := viewPosition view in
  let callees := programCallees program (guestModule f) in
  let rest := skipn (positionOffset position) (innerCode f position) in
  codeFacts (frameLocals f) callees (length (viewValues view) + pending) rest /\
  supportedCode (codeSite callees (match positionLabels position with [] => false | _ => true end)) rest /\
  (positionLabels position <> [] -> sourceHeights callees (innerCode f position) 0 = Some 0 /\
    branchSitesBalanced callees rest (length (viewValues view) + pending)) /\
  (positionLabels position = [] -> sourceHeights callees (innerCode f position) 0 = Some (guestResults f)) /\
  LabelsReady (guestBody f) (positionLabels position) (viewSegments view) /\
  Forall wf_instr (innerCode f position).
Proof.
  intros program s memory base fbp pending f view Hstore Hvalid
    [Hf [Hpath [Hheights [_ [Hoperands _]]]]] position callees rest.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts [Hresults Hsupport]].
  destruct (path_facts _ _ _ _ _ _ _ _ _ Hfacts Hsupport Hpath Hheights) as [Hrest [Hsup Hblock]].
  pose proof (code_facts_wf _ _ _ _ Hfacts) as Hwf.
  split; [exact Hrest | ]. split; [exact Hsup | ]. split; [exact Hblock | ].
  split.
  { intros Hnil. unfold innerCode. fold position in Hpath |- *. rewrite Hnil in Hpath |- *.
    cbn in Hpath |- *. exact Hresults. }
  split; [ | exact (path_code_wf _ _ _ Hwf Hpath)].
  apply (labels_ready _ _ _ _ Hwf Hpath).
  - pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H. rewrite length_map in H. exact H.
  - unfold viewOperands in Hoperands. apply Forall_app in Hoperands. exact (proj1 Hoperands).
Qed.
