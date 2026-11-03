(* Rows that branch to an enclosing label: br_if, br and br_table. Each leaves the
   labels inside its target, keeping the target's arity of operands above the
   target's base; a block's or an if's label is left too (the exit state is in
   FrameRows), a loop's runs the loop again. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers Add R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog SlideLayout
  TraceRefinement StraightLineSpec StackSimulation LocalSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerRefinement ProgramSpec ProgramEncoding CodeLayout BlockSpec BlockBindings BlockEncoding BlockInstructions
  BlockTrace BranchBindings StackBindings CallInitialization Selectors CallScope CallContext CallMemory LocalBindings
  StraightLineInstructions ProgramBinding Fragment CodeFacts ControlDecode ControlInvariant FrameFacts ControlBridge
  CircuitFacts FrameRows.
Import ListNotations.
Open Scope Z_scope.

Section Rows.
Variables (program : Program) (s : store).
Hypothesis Hstore : store_FUNCS s = program.
Hypothesis Hwf : wf_store s.
Hypothesis Hvalid : programValidated program s.

Local Notation RowResult := (RowResult program s).
Local Notation exit_label_state := (exit_label_state program s).

(* A br out of `post`'s labels to the block at b: it carries the operands out of
   the inner labels; the block's label keeps its results and drops the rest, and
   the code after the block's END runs. *)
Lemma branch_exit_steps : forall code pre b post outer i body inner segsHead segT segsTail z dropped kept rest,
  pathCode code pre = Some outer -> siteEntry outer b = Some (i, body) -> pathCode body post = Some inner ->
  length segsHead = length pre -> length segsTail = length post ->
  LabelsReady code pre segsHead -> LabelsReady body post segsTail -> Forall wf_instr outer -> wf_state z ->
  Forall (fun value => 0 <= value < i32Modulus) segT ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segsTail ++ dropped ++ kept) ->
  wf_uN 32%N (mk_uN (N.of_nat (length post + 0))) -> Forall wf_instr rest -> length kept = labelArity i ->
  Steps (wrapLabels code (pre ++ b :: post) (segsHead ++ segT :: segsTail)
      (stackConfig z (dropped ++ kept) (BR (mk_uN (N.of_nat (length post + 0))) :: rest)))
    (wrapLabels code pre segsHead
      (stackPrefix segT (instructionSuffix (skipn (S (siteOffset b)) outer) (stackConfig z kept (labelContinuation i))))).
Proof.
  intros code pre b post outer i body inner segsHead segT segsTail z dropped kept rest Hpath Hb Hinner HlenHead
    HlenTail HreadyHead HreadyTail Hwfouter Hz Hsegment Hvalues Hl Hrest Hlength.
  destruct (wrap_labels_exit post body _ segsTail z (dropped ++ kept) 0 _ Hinner HlenTail HreadyTail Hz Hvalues Hl Hrest)
    as [rest' [Hrest' Hexit]].
  rewrite (wrap_labels_app pre (b :: post) _ _ segsHead (segT :: segsTail) _ Hpath HlenHead).
  cbn [wrapLabels]. rewrite Hb.
  apply steps_in_labels; [exact HreadyHead | ].
  pose proof (continuation_wf _ _ _ _ Hwfouter Hb) as Hcont.
  apply steps_with_stack_prefix; [exact Hsegment | ].
  apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
  eapply spec_steps_transitive; [apply steps_in_block; [exact Hcont | exact Hexit] | ].
  rewrite <- Hlength, app_assoc. apply br_exit_target; [exact Hcont | exact Hz | rewrite <- app_assoc; exact Hvalues |
    exact Hrest'].
Qed.

(* A branch to label l drops the operands of the l labels inside its target; the
   function's label lies further out. *)
Lemma branch_drop_labels : forall (segsHead : list (list Z)) (segT : list Z) (segsTail : list (list Z)),
  fold_right Nat.add 0%nat
    (firstn (length segsTail) (rev (map (@length Z) (segsHead ++ segT :: segsTail)) ++ [0%nat])) =
    length (concat segsTail).
Proof.
  intros segsHead segT segsTail.
  rewrite map_app, rev_app_distr. cbn [map rev]. rewrite <- !app_assoc.
  rewrite firstn_app, length_rev, length_map, Nat.sub_diag, firstn_O, app_nil_r.
  rewrite firstn_all2 by (rewrite length_rev, length_map; lia).
  assert (Hshift : forall a xs, fold_right Nat.add a xs = (a + fold_right Nat.add 0 xs)%nat).
  { intros a xs. induction xs as [ | x xs IH]; cbn [fold_right]; lia. }
  induction segsTail as [ | segment rest IH]; [reflexivity | ].
  cbn [map rev concat]. rewrite fold_right_app, Hshift, IH, length_app. cbn [fold_right]. lia.
Qed.

(* A branch to label `depth` from a frame at a code path: the path splits at the
   target's site b after `pre`, with `post` the `depth` labels inside it, and the
   frame's segments split alongside. Past the target, its results join the
   operands below its label. *)
Lemma branch_target_split : forall memory base fbp f index labels offset segments values locals depth,
  FrameAt program memory base fbp 0 0 0 f (frameView (codePosition index labels offset) segments values locals) ->
  (depth < length segments)%nat ->
  exists pre b post segsHead segT segsTail outer i body height,
    labels = pre ++ b :: post /\ segments = segsHead ++ segT :: segsTail /\
    length post = depth /\ length segsTail = depth /\ length segsHead = length pre /\
    pathCode (guestBody f) pre = Some outer /\ siteEntry outer b = Some (i, body) /\
    pathCode body post = Some (innerCode f (codePosition index labels offset)) /\
    pathHeights (programCallees program (guestModule f)) (guestBody f) pre (siteOffset b) =
      Some (map (@length Z) segsHead, height) /\ labelBase i height = length segT /\
    pathHeights (programCallees program (guestModule f)) (guestBody f) pre (S (siteOffset b)) =
      Some (map (@length Z) segsHead, length segT + labelResults i)%nat /\
    Supported (programCallees program (guestModule f)) i /\
    nth_error (rev (pathArities (guestBody f) labels) ++ [guestResults f]) depth = Some (labelArity i) /\
    LabelsReady (guestBody f) pre segsHead /\ LabelsReady body post segsTail /\ Forall wf_instr outer.
Proof.
  intros memory base fbp f index labels offset segments values locals depth Hframe Hdepth.
  pose proof Hframe as [Hf [Hpath [Hheights [_ [_ [Hbounds _]]]]]].
  cbn [viewPosition viewSegments viewValues positionFunction positionLabels positionOffset] in Hf, Hpath, Hheights.
  set (inner := innerCode f (codePosition index labels offset)) in *.
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  destruct (list_split_at _ segments (length segments - S depth) ltac:(lia))
    as [segsHead [segT [segsTail [-> HsegHead]]]].
  rewrite !length_app in Hlen, Hdepth, HsegHead. cbn [length] in Hlen, Hdepth, HsegHead.
  destruct (list_split_at _ labels (length segsHead) ltac:(lia)) as [pre [b [post [-> Hpre]]]].
  rewrite !length_app in Hlen. cbn [length] in Hlen.
  destruct (path_heights_split _ _ _ _ _ _ _ _ Hheights)
    as [outer [i [body [hsHead' [below [hsTail' [Hpath' [Hb [Hheights' [Hhs' HheightsTail]]]]]]]]]].
  pose proof (path_heights_length _ _ _ _ _ _ Hheights') as HheadLen.
  rewrite map_app in Hhs'. cbn [map] in Hhs'.
  destruct (app_split_eq _ (map (@length Z) segsHead) _ hsHead' _ ltac:(rewrite length_map; lia) Hhs')
    as [Hhead' Htail''].
  injection Htail'' as Hbelow _.
  rewrite <- Hhead' in Hheights'.
  pose proof Hpath as Hinside. rewrite path_code_app, Hpath' in Hinside. cbn [pathCode] in Hinside.
  rewrite Hb in Hinside.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts [_ [Hsup0 Hbr0]]].
  pose proof (code_facts_wf _ _ _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  destruct (site_entry_split _ _ _ _ Hb) as [Hnth Harm].
  assert (Hwfblock : Forall wf_instr body)
    by exact (arm_wf _ _ _ (proj1 (Forall_forall _ _) Hwfouter _ (nth_error_In _ _ Hnth)) Harm).
  (* The target's typing: where its label resumes. *)
  destruct (path_facts _ _ _ _ _ _ _ _ _ _ _ Hfacts Hsup0 Hbr0 Hpath' Hheights') as [Hat [Hsupat _]].
  rewrite (skipn_at_instruction _ _ _ Hnth) in Hat, Hsupat. destruct Hat as [Hinstr _].
  pose proof (path_heights_step _ _ _ _ _ _ _ _ _ Hpath' Hnth Hheights'
    (structured_resume _ _ _ _ _ _ _ _ Hinstr Harm)) as HheightsNext.
  rewrite <- Hbelow in HheightsNext.
  unfold viewOperands in Hbounds. cbn [viewSegments viewValues] in Hbounds.
  rewrite concat_app in Hbounds. cbn [concat] in Hbounds.
  apply Forall_app in Hbounds as [Hvb _]. apply Forall_app in Hvb as [Hhead Hvb]. apply Forall_app in Hvb as [_ Htail].
  exists pre, b, post, segsHead, segT, segsTail, outer, i, body, below.
  split; [reflexivity | ]. split; [reflexivity | ]. split; [lia | ]. split; [lia | ]. split; [lia | ].
  split; [exact Hpath' | ]. split; [exact Hb | ]. split; [exact Hinside | ]. split; [exact Hheights' | ].
  split; [exact (eq_sym Hbelow) | ]. split; [exact HheightsNext | ]. split; [exact (Forall_inv Hsupat) | ].
  split.
  { assert (Htail' : length (pathArities body post) = depth)
      by (rewrite (path_arities_length _ _ _ _ _ _ HheightsTail), (path_heights_length _ _ _ _ _ _ HheightsTail); lia).
    rewrite (path_arities_app _ _ _ _ _ _ _ Hpath' Hb), rev_app_distr. cbn [rev]. rewrite <- !app_assoc.
    rewrite nth_error_app2 by (rewrite length_rev; lia). rewrite length_rev, Htail', Nat.sub_diag. reflexivity. }
  split; [exact (labels_ready pre _ _ segsHead Hwfbody Hpath' ltac:(lia) Hhead) | ].
  split; [apply (labels_ready _ _ _ _ Hwfblock Hinside); [lia | exact Htail] | exact Hwfouter].
Qed.

(* A taken branch to the label of ti at b after `pre`, out of the labels `post`
   inside it, keeping `kept` and dropping the operands below them down to the
   label's base. A block's or an if's label resumes after its instruction, with
   `kept` still `distance` cells higher while the slide's move is pending; a
   loop's label keeps nothing and runs the loop's body again. *)
Lemma taken_branch_result : forall initial final row f inner view index pre b post segsHead segT segsTail outer ti
    body target height locals dropped kept rest distance,
  functionAt program index = Some f ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer b = Some (ti, body) -> pathCode body post = Some target ->
  length segsHead = length pre -> length segsTail = length post ->
  LabelsReady (guestBody f) pre segsHead -> LabelsReady body post segsTail -> Forall wf_instr outer ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) pre (siteOffset b) =
    Some (map (@length Z) segsHead, height) -> labelBase ti height = length segT ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) pre (S (siteOffset b)) =
    Some (map (@length Z) segsHead, length segT + labelResults ti)%nat ->
  Supported (programCallees program (guestModule f)) ti -> length kept = labelArity ti ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segsTail ++ dropped ++ kept) ->
  wf_uN 32%N (mk_uN (N.of_nat (length post + 0))) -> Forall wf_instr rest ->
  Steps inner (wrapLabels (guestBody f) (pre ++ b :: post) (segsHead ++ segT :: segsTail)
    (stackConfig (mk_state s (activeFrame f locals)) (dropped ++ kept)
      (BR (mk_uN (N.of_nat (length post + 0))) :: rest))) ->
  0 <= boundaryLocalsBase (afterBoundary row) ->
  boundaryStackBase (afterBoundary row) = boundaryLocalsBase (afterBoundary row) + Z.of_nat (frameLocals f) ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segsHead ++ segT ++ kept) ->
  (forall index' value, nth_error (concat segsHead ++ segT) index' = Some value ->
    final stackId [2 * (boundaryStackBase (afterBoundary row) + Z.of_nat index')] = Some value) ->
  (forall j value, nth_error kept j = Some value ->
    final stackId [2 * (boundaryStackBase (afterBoundary row) +
      Z.of_nat (length (concat segsHead ++ segT) + distance + j))] = Some value) ->
  boundaryMoveRemaining (afterBoundary row) = slideWord true (length kept) distance ->
  (distance <> 0%nat -> boundaryMoveDistance (afterBoundary row) = Z.of_nat distance) ->
  boundarySP (afterBoundary row) =
    boundaryStackBase (afterBoundary row) + Z.of_nat (length (concat segsHead ++ segT ++ kept)) ->
  length locals = frameLocals f -> Forall (fun value => 0 <= value < i32Modulus) locals ->
  (forall index' value, nth_error locals index' = Some value ->
    final stackId [2 * (boundaryLocalsBase (afterBoundary row) + Z.of_nat index')] = Some value) ->
  boundaryPC (afterBoundary row) =
    guestEntry f + Z.of_nat (pathOffset (guestBody f) pre (siteOffset b) + labelTarget ti) ->
  initializationFlags (afterBoundary row) -> boundaryZeroRemaining (afterBoundary row) = 0 ->
  frameRegistersPreserved row -> FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final ->
  RowResult initial final row f inner view.
Proof.
  intros initial final row f inner view index pre b post segsHead segT segsTail outer ti body target height locals
    dropped kept rest distance Hf Hpath Hb Hinside HheadLen HtailLen HreadyHead HreadyTail Hwfouter Hheights Hbelow
    HheightsNext Hsup Hlength Hcarried Hl Hrest Hsteps Hfbp Hbase Hbounds Hcells Hkept Hremaining Hdistance Hsp
    Hllen Hlb Hlc Hpc Hflags Hzero Hregs Heffect.
  set (z := mk_state s (activeFrame f locals)) in *.
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  destruct (site_entry_split _ _ _ _ Hb) as [Hnth Harm].
  assert (Hsegment : Forall (fun value => 0 <= value < i32Modulus) segT)
    by (apply Forall_app in Hbounds as [_ H]; apply Forall_app in H as [H _]; exact H).
  assert (Hhead : Forall (fun value => 0 <= value < i32Modulus) (concat segsHead))
    by (apply Forall_app in Hbounds as [H _]; exact H).
  pose proof (branch_exit_steps _ _ _ _ _ _ _ _ _ _ _ z dropped kept rest Hpath Hb Hinside HheadLen HtailLen
    HreadyHead HreadyTail Hwfouter Hz Hsegment Hcarried Hl Hrest Hlength) as Hexit.
  assert (Hcase : (exists bt lbody, ti = LOOP bt lbody) \/
      (labelContinuation ti = [] /\ labelArity ti = labelResults ti /\ labelTarget ti = instructionSize ti))
    by (destruct ti; try (right; split; [ | split]; reflexivity); left; eauto).
  destruct Hcase as [[bt [lbody ->]] | [Hcont [Harity Htarget]]].
  - (* A loop: the slide keeps nothing, and the body runs again under the label. *)
    destruct kept as [ | ? ?]; [ | discriminate Hlength].
    destruct (siteArm b) as [ | arm] eqn:Hsitearm; [ | destruct arm; discriminate Harm].
    injection Harm as Hlbody. subst lbody.
    inversion Hsup as [ | | | | | | | | | | | result ? Hbody Hresult |  | ]; try subst bt.
    assert (Hwbody : Forall wf_instr body) by exact (site_wf _ _ _ _ Hwfouter Hb).
    assert (Hpath' : pathCode (guestBody f) (pre ++ [b]) = Some body)
      by (rewrite path_code_snoc, Hpath, Hb; reflexivity).
    assert (Hinner' : innerCode f (codePosition index (pre ++ [b]) 0) = body)
      by (unfold innerCode; cbn [positionLabels]; rewrite Hpath'; reflexivity).
    rewrite !app_nil_r in Hbounds, Hsp.
    exists (frameView (codePosition index (pre ++ [b]) 0) (segsHead ++ [segT]) [] locals),
      (frameConfig f (frameView (codePosition index (pre ++ [b]) 0) (segsHead ++ [segT]) [] locals)
        (stackConfig z [] body)).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + eapply spec_steps_transitive; [exact Hsteps | ]. eapply spec_steps_transitive; [exact Hexit | ].
      unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      rewrite (wrap_labels_snoc _ _ _ _ _ _ segsHead segT _ Hpath Hb HheadLen).
      apply steps_in_labels; [exact HreadyHead | ].
      apply steps_with_stack_prefix; [exact Hsegment | ].
      apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
      exact (enter_loop z result body Hresult Hz Hwbody).
    + split; [ | split; [exact Hregs | exact Heffect]].
      pose proof (idle_active_state program s final (afterBoundary row) f
        (frameView (codePosition index (pre ++ [b]) 0) (segsHead ++ [segT]) [] locals)) as Hafter.
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
      rewrite Hinner' in Hafter. apply Hafter; clear Hafter.
      * apply frame_at_unmoved_intro;
          cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
          unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner'.
        -- exact Hf.
        -- exact Hpath'.
        -- rewrite (path_heights_snoc _ _ _ _ _ _ _ _ _ 0 Hpath Hb Hheights), map_app, Hbelow. reflexivity.
        -- exact Hfbp.
        -- exact Hbase.
        -- rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hbounds.
        -- rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hcells.
        -- exact Hllen.
        -- exact Hlb.
        -- exact Hlc.
      * cbn. lia.
      * unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
        exact Hsp.
      * rewrite Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath Hb), Hsitearm. unfold codeOffset. cbn. lia.
      * assert (Hmove : boundaryMoveRemaining (afterBoundary row) = 0)
          by (rewrite Hremaining; unfold slideWord; destruct (Nat.eqb distance 0); reflexivity).
        destruct Hflags as [Hma Hza]. rewrite Hmove in Hma, Hza. rewrite Hzero in Hza.
        split; [exact Hzero | ]. split; [exact Hmove | ]. split; [exact Hza | exact Hma].
  - (* A block or an if: resume after it. *)
    rewrite Hcont in Hexit.
    exists (frameView (codePosition index pre (S (siteOffset b))) segsHead (segT ++ kept) locals),
      (frameConfig f (frameView (codePosition index pre (S (siteOffset b))) segsHead (segT ++ kept) locals)
        (stackConfig z (segT ++ kept) (skipn (S (siteOffset b)) outer))).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + eapply spec_steps_transitive; [exact Hsteps | ].
      unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      replace (stackConfig z (segT ++ kept) (skipn (S (siteOffset b)) outer)) with
        (stackPrefix segT (instructionSuffix (skipn (S (siteOffset b)) outer) (stackConfig z kept [])))
        by (rewrite instruction_suffix_config, stack_prefix_config; reflexivity).
      exact Hexit.
    + split; [ | split; [exact Hregs | exact Heffect]].
      rewrite <- Harity, <- Hlength, <- length_app in HheightsNext.
      apply (exit_label_state _ _ _ _ _ _ _ _ distance _ _ _ Hf Hpath
        ltac:(apply nth_error_Some; rewrite Hnth; discriminate) HheightsNext); try assumption.
      rewrite Hpc, Htarget, (path_offset_step _ _ _ _ _ Hpath Hnth). reflexivity.
Qed.

(* A br_if l: on zero it falls through; otherwise it leaves the l blocks inside
   its target and the target itself, keeping the target's results and discarding
   every other operand above the target's base, and resumes after the target's
   END, with the results still to move down. *)
Lemma code_branch_row : forall view f l initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (BR_IF l) ->
  RowResult initial final row f inner view.
Proof.
  intros view f l initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe)
    as [Hfacts [[_ Hbranches] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hbranches, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hbranches. destruct Hbranches as [Hexits _].
  destruct Hfacts as [[_ [n [Harity Hkeep]]] _].
  (* The condition is the top operand; `under` are the operands below it. *)
  assert (Hnonempty : values <> []) by (intros ->; cbn in Hkeep; lia).
  destruct (exists_last Hnonempty) as [under [condition ->]]. clear Hnonempty.
  rewrite length_app, Nat.add_0_r in Hkeep. cbn [length] in Hkeep.
  assert (Hwfl : wf_uN 32%N l).
  { pose proof (proj1 (Forall_forall _ _) Hwfcode _ (nth_error_In _ _ Hi)) as H. inversion H; assumption. }
  cbn [instrBranches] in Hexits.
  match type of Hexits with (?target < _)%nat => set (depth := target) in * end.
  (* The target block, and the `depth` labels inside it. *)
  destruct (branch_target_split _ _ _ _ _ _ _ _ _ _ _ Hframe Hexits)
    as [pre [b [post [segsHead [segT [segsTail [outer [ti [body [height [-> [-> [Hpost [HtailLen [HheadLen
      [Hpath' [Hb [Hinside [Hheights0 [Hbelow [Hheights' [Hsupti [HtargetArity [HreadyHead [HreadyTail Hwfouter]]]]]]]]]]]]]]]]]]]]]]]]].
  rewrite HtargetArity in Harity. injection Harity as <-.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite concat_app in Hvb. cbn [concat] in Hvb.
  apply Forall_app in Hvb as [Hvb Hvalues]. apply Forall_app in Hvalues as [Hunder Hcond]. apply Forall_inv in Hcond.
  apply Forall_app in Hvb as [Hhead Hvb]. apply Forall_app in Hvb as [Hsegment Htail].
  (* The circuit: pop the condition, then fall through or slide past the target. *)
  pose proof (encoded_path_branch _ _ _ _ Hloaded Hf (pre ++ b :: post) code offset l pre b post outer ti body Hpath Hi
    Logic.eq_refl (Logic.eq_sym Hpost) Hpath' Hb) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (pre ++ b :: post) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (circuit_br_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram Logic.eq_refl Himage) as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (branch_depth_preserved row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hdepth.
  destruct (full_block_selector_quiet row COL_SEL_BR_IF_EQZ Hprime Hfull (or_intror (or_introl Logic.eq_refl)) Hchosen)
    as [Hpush Hpop].
  pose proof (frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack.
  (* Falling through slides nothing. Taken, the slide keeps the target's results
     and drops the operands below them, down to the target's base. *)
  set (distance := (length under - labelArity ti + length (concat segsTail))%nat).
  assert (HslideAfter : row COL_FRAME_MOVE_REMAINING_AFTER =
        (if Z.eqb (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then 0
         else slideWord true (labelArity ti) distance) /\
      row COL_FRAME_MOVE_DISTANCE_AFTER =
        if Z.eqb (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then 0 else Z.of_nat distance).
  { assert (Hslide : row COL_FRAME_SLIDE_ACTIVE = 1).
    { rewrite (full_frame_slide_gate row Hprime Hfull), Hpop, Hchosen.
      repeat rewrite (full_opcode_selector_exclusive row COL_SEL_BR_IF_EQZ) by
        (assumption || vm_compute; tauto || discriminate). reflexivity. }
    destruct (circuit_row_slide_fetch _ _ _ Hreplay Hslide) as [Hremaining Hdistance].
    destruct (encoded_branch_slide _ _ _ _ _ _ _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf Hpath Hi Hheights)
      as [Hkept Hrom].
    rewrite <- Hpc, Hremaining, (full_br_if_choice row Hprime Hfull Hchosen) in Hkept.
    rewrite <- Hpc, Hdistance, (full_br_if_choice row Hprime Hfull Hchosen) in Hrom.
    destruct (Z.eqb (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0).
    { change ((0 =? 1)%Z) with false in Hkept, Hrom. cbv iota in Hkept, Hrom.
      injection Hkept as ->. injection Hrom as ->. split; reflexivity. }
    change ((1 =? 1)%Z) with true in Hkept, Hrom. cbv iota in Hkept, Hrom.
    match type of Hkept with context [branchSlide ?hs ?ars ?top ?target] =>
      assert (Hcompute : branchSlide hs ars top target = Some (labelArity ti, distance));
      [replace target with depth by reflexivity | rewrite Hcompute in Hkept, Hrom] end.
    { rewrite length_app, Nat.add_0_r. cbn [length]. rewrite Nat.add_1_r. unfold branchSlide, labelSlide.
      rewrite HtargetArity. replace (Nat.leb (labelArity ti) (length under)) with true
        by (symmetry; apply Nat.leb_le; lia).
      rewrite <- HtailLen, branch_drop_labels. reflexivity. }
    unfold slideEntry in Hkept, Hrom. injection Hkept as ->. injection Hrom as ->. split; reflexivity. }
  destruct HslideAfter as [HkeptAfter HdistAfter].
  assert (Hshape : UnaryStackShape row true false).
  { apply full_br_if_stack_shape; [exact Hprime | exact Hfull | exact Hchosen | ].
    destruct Hstack as [Hb0 [Hsp0 _]]. cbn [beforeBoundary boundarySP boundaryStackBase] in Hb0, Hsp0.
    rewrite !length_app in Hsp0. cbn [length] in Hsp0. lia. }
  destruct (pop_stack_replay row initial final Hshape Hreplay) as [HreadCell HstackMem].
  assert (Hread : row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = condition).
  { assert (Hlookup : nth_error (concat (segsHead ++ segT :: segsTail) ++ under ++ [condition])
        (length (concat (segsHead ++ segT :: segsTail) ++ under)) = Some condition).
    { rewrite app_assoc, nth_error_app2, Nat.sub_diag by lia. reflexivity. }
    pose proof (Hcells _ _ Hlookup) as Hcell.
    rewrite (local_read_address _ _ _ Hshape Logic.eq_refl) in HreadCell.
    cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp. unfold viewOperands in Hsp.
    cbn [viewSegments viewValues] in Hsp.
    replace (2 * (row COL_SP_BEFORE - 1)) with
      (2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat (segsHead ++ segT :: segsTail) ++ under))))
      in HreadCell by (rewrite Hsp, !length_app; cbn [length]; lia).
    congruence. }
  rewrite Hread in Hnext, HkeptAfter, HdistAfter.
  pose proof (full_branch_fbp_unchanged row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hfbp.
  assert (Hwrites : filter isStackWrite (eventsForRow memories row) = []).
  { rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite (local_write_active _ _ _ Hshape), (local_write_hi_active _ _ _ Hshape), (local_write_gate _ _ _ Hshape),
      (local_gather_lo_inactive _ _ _ Hshape), (local_gather_inactive _ _ _ Hshape). reflexivity. }
  assert (Hregs : frameRegistersPreserved row)
    by (split; [exact Hdepth | split; [exact (local_stack_base_after _ _ _ Hshape) | exact Hfbp]]).
  assert (Heffect : FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final)
    by (eapply quiet_row_frame_effect; [exact Hreplay | exact Hwrites | exact Hpush | exact Hsuspend |
      exact HcontextRead]).
  (* SpecTec: the same choice. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  pose proof (forall_skipn _ _ (S offset) _ Hwfcode) as Hwfrest.
  pose proof (br_if_steps_below z under condition l (skipn (S offset) code) Hz Hunder Hcond Hwfl Hwfrest) as Hsteps.
  assert (Hbefore : frameConfig f (frameView (codePosition index (pre ++ b :: post) offset)
      (segsHead ++ segT :: segsTail) (under ++ [condition]) locals)
      (stackConfig z (under ++ [condition]) (skipn offset code)) =
      wrapLabels (guestBody f) (pre ++ b :: post) (segsHead ++ segT :: segsTail)
        (stackConfig z (under ++ [condition]) (BR_IF l :: skipn (S offset) code))).
  { unfold frameConfig. cbn [viewPosition viewSegments positionLabels]. rewrite (skipn_at_instruction _ _ _ Hi).
    reflexivity. }
  destruct (Z.eqb condition 0) eqn:Hzero.
  - (* Fall through: every label stays, and the condition is gone. *)
    cbv iota in HkeptAfter.
    pose proof (branch_keeps_idle row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen
      ltac:(vm_compute; tauto) HkeptAfter Hidle) as HidleAfter.
    rewrite app_assoc in Hstack.
    destruct (stack_matches_pop row initial final _ condition false Hshape Hreplay (fun _ => HdistAfter)
      ltac:(intro H; discriminate H) Hstack) as [_ HstackAfter].
    exists (frameView (codePosition index (pre ++ b :: post) (S offset)) (segsHead ++ segT :: segsTail) under locals),
      (frameConfig f (frameView (codePosition index (pre ++ b :: post) (S offset)) (segsHead ++ segT :: segsTail)
        under locals) (stackConfig z under (skipn (S offset) code))).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + rewrite Hbefore. unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      exact (steps_in_labels _ _ _ _ _ Hready Hsteps).
    + split; [ | split; [exact Hregs | exact Heffect]].
      assert (Hinner : innerCode f (codePosition index (pre ++ b :: post) (S offset)) = code) by reflexivity.
      destruct HstackAfter as [Hbase1 [Hsp1 [Hbounds1 Hcells1]]].
      pose proof (idle_active_state program s final (afterBoundary row) f
        (frameView (codePosition index (pre ++ b :: post) (S offset)) (segsHead ++ segT :: segsTail) under locals))
        as Hafter.
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
      rewrite Hinner in Hafter. apply Hafter; clear Hafter.
      * apply frame_at_unmoved_intro;
          cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
          unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner.
        -- exact Hf.
        -- exact Hpath.
        -- refine (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights _).
           rewrite length_app, !Nat.add_0_r. cbn [length sourceHeight]. rewrite Nat.add_1_r. reflexivity.
        -- cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
        -- pose proof (local_stack_base_after _ _ _ Hshape).
           cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
        -- exact Hbounds1.
        -- exact Hcells1.
        -- exact Hllen.
        -- exact Hlb.
        -- intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HstackMem. exact (Hlc _ _ Hv).
      * assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate). cbn. lia.
      * unfold viewOperands. cbn [viewSegments viewValues]. exact Hsp1.
      * cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi).
        change (instructionSize (BR_IF l)) with 1%nat. lia.
      * exact HidleAfter.
  - (* Taken: carry the operands out of the labels inside the target; the target's
       label keeps its results, `kept`, and drops the operands below them. The
       circuit's slide leaves `kept` to move down by `distance`. *)
    cbv iota in HkeptAfter, HdistAfter.
    assert (Hindex : l = mk_uN (N.of_nat (length post + 0))).
    { destruct l as [ln]. cbn in Hpost |- *. rewrite Nat.add_0_r, Hpost. f_equal. unfold depth. cbn. lia. }
    rewrite Hindex in Hsteps, Hwfl, Hbefore.
    destruct (split_top_values under (labelArity ti) ltac:(lia)) as [dropped [kept [-> Hlength]]].
    apply Forall_app in Hunder as [Hdropped Hkept].
    assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
      by (intros xs ys k; rewrite nth_error_app2 by lia; f_equal; lia).
    cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
    unfold viewOperands in Hsp, Hcells. cbn [viewSegments viewValues] in Hsp, Hcells.
    rewrite concat_app in Hsp, Hcells. cbn [concat] in Hsp, Hcells.
    replace ((concat segsHead ++ segT ++ concat segsTail) ++ (dropped ++ kept) ++ [condition]) with
      ((concat segsHead ++ segT) ++ (concat segsTail ++ dropped) ++ kept ++ [condition])
      in Hsp, Hcells by (rewrite <- !app_assoc; reflexivity).
    apply (taken_branch_result _ _ _ _ _ _ index pre b post segsHead segT segsTail outer ti body _ height locals
      dropped kept (skipn (S offset) code) distance Hf Hpath' Hb Hinside HheadLen (eq_trans HtailLen (eq_sym Hpost)) HreadyHead HreadyTail
      Hwfouter Hheights0 Hbelow Hheights' Hsupti Hlength).
    * apply Forall_app. split; [exact Htail | apply Forall_app; split; assumption].
    * exact Hwfl.
    * exact Hwfrest.
    * rewrite Hbefore. unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      exact (steps_in_labels _ _ _ _ _ Hready Hsteps).
    * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
    * pose proof (local_stack_base_after _ _ _ Hshape).
      cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
    * rewrite app_assoc. exact (proj2 (Forall_app _ _ _) (conj
        (proj2 (Forall_app _ _ _) (conj Hhead Hsegment)) Hkept)).
    * intros index' value Hv. cbn [afterBoundary boundaryStackBase].
      rewrite (local_stack_base_after _ _ _ Hshape), HstackMem. apply Hcells.
      rewrite nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate].
    * intros j value Hv. cbn [afterBoundary boundaryStackBase].
      rewrite (local_stack_base_after _ _ _ Hshape), HstackMem.
      replace (length (concat segsHead ++ segT) + distance + j)%nat with
        (length (concat segsHead ++ segT) + (length (concat segsTail ++ dropped) + j))%nat
        by (unfold distance; rewrite !length_app; lia).
      apply Hcells. rewrite !Hshift, nth_error_app1 by (apply nth_error_Some; rewrite Hv; discriminate). exact Hv.
    * rewrite Hlength. exact HkeptAfter.
    * intros _. exact HdistAfter.
    * cbn [afterBoundary boundarySP boundaryStackBase].
      rewrite (local_stack_pointer_after _ _ _ Hshape), (local_stack_base_after _ _ _ Hshape), HdistAfter, Hsp.
      unfold distance. rewrite !length_app. cbn [length]. lia.
    * exact Hllen.
    * exact Hlb.
    * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HstackMem. exact (Hlc _ _ Hv).
    * cbn [afterBoundary boundaryPC]. rewrite Hnext. reflexivity.
    * exact (full_initialization_flags row Hprime Hfull).
    * exact (branch_no_zeroing row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen
        ltac:(vm_compute; tauto) Hidle).
    * exact Hregs.
    * exact Heffect.
Qed.

(* A br l: it leaves the l blocks inside its target and the target itself,
   keeping the target's results and discarding every other operand above the
   target's base, and resumes after the target's END, with the results still to
   move down. *)
Lemma code_br_row : forall view f l initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (BR l) ->
  N.to_nat (l :> N) <> length (positionLabels (viewPosition view)) ->
  RowResult initial final row f inner view.
Proof.
  intros view f l initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi Hnotfunction.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe)
    as [Hfacts [[_ Hbranches] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hbranches, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hbranches. destruct Hbranches as [Hexits _].
  destruct Hfacts as [[_ [n [Harity Hkeep]]] _]. rewrite Nat.add_0_r in Hkeep.
  assert (Hwfl : wf_uN 32%N l).
  { pose proof (proj1 (Forall_forall _ _) Hwfcode _ (nth_error_In _ _ Hi)) as H. inversion H; assumption. }
  cbn [instrBranches] in Hexits.
  (* Not the function's label (a return), so an enclosing one. *)
  assert (Hlabels : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  assert (Htarget : (N.to_nat (l :> N) < length segments)%nat) by lia.
  clear Hexits. rename Htarget into Hexits.
  match type of Hexits with (?target < _)%nat => set (depth := target) in * end.
  (* The target block, and the `depth` labels inside it. *)
  destruct (branch_target_split _ _ _ _ _ _ _ _ _ _ _ Hframe Hexits)
    as [pre [b [post [segsHead [segT [segsTail [outer [ti [body [height [-> [-> [Hpost [HtailLen [HheadLen
      [Hpath' [Hb [Hinside [Hheights0 [Hbelow [Hheights' [Hsupti [HtargetArity [HreadyHead [HreadyTail Hwfouter]]]]]]]]]]]]]]]]]]]]]]]]].
  rewrite HtargetArity in Harity. injection Harity as <-.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite concat_app in Hvb. cbn [concat] in Hvb.
  apply Forall_app in Hvb as [Hvb Hvalues].
  apply Forall_app in Hvb as [Hhead Hvb]. apply Forall_app in Hvb as [Hsegment Htail].
  (* The circuit: the row's one edge slides past the target. *)
  pose proof (encoded_path_br _ _ _ _ Hloaded Hf (pre ++ b :: post) code offset l pre b post outer ti body Hpath Hi
    Logic.eq_refl (Logic.eq_sym Hpost) Hpath' Hb) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (pre ++ b :: post) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (BlockTrace.control_row _ _ row _ CODE_BR COL_SEL_BR _ Hprime Hfull Hprogram Hreplay Logic.eq_refl
    ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage) as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row COL_SEL_BR Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (branch_depth_preserved row COL_SEL_BR Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hdepth.
  destruct (full_branch_frame row COL_SEL_BR Hprime Hfull ltac:(vm_compute; tauto) Hchosen ltac:(cbn; tauto))
    as [Hpop Hpush _ _ _].
  pose proof (full_branch_fbp_unchanged row COL_SEL_BR Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hfbp.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
  destruct (full_br_stack row Hprime Hfull Hchosen ltac:(rewrite Hsp; cbn [beforeBoundary boundaryStackBase] in Hbase0;
    lia)) as [HspAfter [HbaseAfter Hwrites]].
  pose proof (no_stack_writes_preserve _ _ _ Hreplay Hwrites) as HstackMem.
  (* The slide keeps the target's results and drops the operands below them, down
     to the target's base. *)
  set (distance := (length values - labelArity ti + length (concat segsTail))%nat).
  assert (HslideAfter : row COL_FRAME_MOVE_REMAINING_AFTER = slideWord true (labelArity ti) distance /\
      row COL_FRAME_MOVE_DISTANCE_AFTER = Z.of_nat distance).
  { assert (Hslide : row COL_FRAME_SLIDE_ACTIVE = 1).
    { rewrite (full_frame_slide_gate row Hprime Hfull), Hpop, Hchosen.
      repeat rewrite (full_opcode_selector_exclusive row COL_SEL_BR) by
        (assumption || vm_compute; tauto || discriminate). reflexivity. }
    destruct (circuit_row_slide_fetch _ _ _ Hreplay Hslide) as [Hremaining Hdistance].
    destruct (encoded_br_slide _ _ _ _ _ _ _ _ _ _ (row COL_CONTROL_CHOICE) Hloaded Hf Hpath Hi
      ltac:(rewrite length_app; cbn [length]; lia) Hheights)
      as [Hkept Hrom].
    pose proof (full_selected_choice_pinned row COL_SEL_BR Hprime Hfull ltac:(vm_compute; tauto) Hchosen
      ltac:(vm_compute; intuition discriminate)) as Hchoice.
    rewrite <- Hpc, Hremaining, Hchoice in Hkept. rewrite <- Hpc, Hdistance, Hchoice in Hrom.
    change ((0 =? 0)%Z) with true in Hkept, Hrom. cbv iota in Hkept, Hrom.
    match type of Hkept with context [labelSlide ?hs ?ars ?top ?target] =>
      assert (Hcompute : labelSlide hs ars top target = Some (labelArity ti, distance));
      [replace target with depth by reflexivity | rewrite Hcompute in Hkept, Hrom] end.
    { rewrite Nat.add_0_r. unfold labelSlide. rewrite HtargetArity.
      replace (Nat.leb (labelArity ti) (length values)) with true by (symmetry; apply Nat.leb_le; lia).
      rewrite <- HtailLen, branch_drop_labels. reflexivity. }
    unfold slideEntry in Hkept, Hrom. injection Hkept as ->. injection Hrom as ->. split; reflexivity. }
  destruct HslideAfter as [HkeptAfter HdistAfter].
  assert (Hregs : frameRegistersPreserved row) by (split; [exact Hdepth | split; [exact HbaseAfter | exact Hfbp]]).
  assert (Heffect : FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final)
    by (eapply quiet_row_frame_effect; [exact Hreplay | exact Hwrites | exact Hpush | exact Hsuspend |
      exact HcontextRead]).
  (* SpecTec: the same branch. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  pose proof (forall_skipn _ _ (S offset) _ Hwfcode) as Hwfrest.
  assert (Hindex : l = mk_uN (N.of_nat (length post + 0))).
  { destruct l as [ln]. cbn in Hpost |- *. rewrite Nat.add_0_r, Hpost. f_equal. unfold depth. cbn. lia. }
  rewrite Hindex in Hwfl.
  destruct (split_top_values values (labelArity ti) ltac:(lia)) as [dropped [kept [-> Hlength]]].
  apply Forall_app in Hvalues as [Hdropped Hkept].
  assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
    by (intros xs ys k; rewrite nth_error_app2 by lia; f_equal; lia).
  unfold viewOperands in Hsp, Hcells. cbn [viewSegments viewValues] in Hsp, Hcells.
  rewrite concat_app in Hsp, Hcells. cbn [concat] in Hsp, Hcells.
  replace ((concat segsHead ++ segT ++ concat segsTail) ++ dropped ++ kept) with
    ((concat segsHead ++ segT) ++ (concat segsTail ++ dropped) ++ kept)
    in Hsp, Hcells by (rewrite <- !app_assoc; reflexivity).
  apply (taken_branch_result _ _ _ _ _ _ index pre b post segsHead segT segsTail outer ti body _ height locals
    dropped kept (skipn (S offset) code) distance Hf Hpath' Hb Hinside HheadLen (eq_trans HtailLen (eq_sym Hpost)) HreadyHead HreadyTail
    Hwfouter Hheights0 Hbelow Hheights' Hsupti Hlength).
  + apply Forall_app. split; [exact Htail | apply Forall_app; split; assumption].
  + exact Hwfl.
  + exact Hwfrest.
  + unfold frameConfig. cbn [viewPosition viewSegments viewValues positionLabels positionOffset].
    rewrite (skipn_at_instruction _ _ _ Hi), Hindex. apply steps_refl_wf. apply wrap_labels_wf; [exact Hready | ].
    apply stack_config_wf; [exact Hz | apply Forall_app; split; assumption | ].
    constructor; [constructor; exact Hwfl | exact Hwfrest].
  + cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
  + cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
  + rewrite app_assoc. exact (proj2 (Forall_app _ _ _) (conj
      (proj2 (Forall_app _ _ _) (conj Hhead Hsegment)) Hkept)).
  + intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite HbaseAfter, HstackMem. apply Hcells.
    rewrite nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate].
  + intros j value Hv. cbn [afterBoundary boundaryStackBase]. rewrite HbaseAfter, HstackMem.
    replace (length (concat segsHead ++ segT) + distance + j)%nat with
      (length (concat segsHead ++ segT) + (length (concat segsTail ++ dropped) + j))%nat
      by (unfold distance; rewrite !length_app; lia).
    apply Hcells. rewrite !Hshift. exact Hv.
  + rewrite Hlength. exact HkeptAfter.
  + intros _. exact HdistAfter.
  + cbn [afterBoundary boundarySP boundaryStackBase]. rewrite HspAfter, HbaseAfter, HdistAfter, Hsp.
    unfold distance. rewrite !length_app. lia.
  + exact Hllen.
  + exact Hlb.
  + intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HstackMem. exact (Hlc _ _ Hv).
  + cbn [afterBoundary boundaryPC]. rewrite Hnext. reflexivity.
  + exact (full_initialization_flags row Hprime Hfull).
  + exact (branch_no_zeroing row COL_SEL_BR Hprime Hfull ltac:(vm_compute; tauto) Hchosen
      ltac:(vm_compute; tauto) Hidle).
  + exact Hregs.
  + exact Heffect.
Qed.

(* A br_table: its index picks an arm, or the default beyond the arms. Like a br
   to that target, it keeps the target's results, discards every other operand
   above the target's base, and resumes after the target's END, with the results
   still to move down. *)
Lemma code_br_table_row : forall view f ls l initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (BR_TABLE ls l) ->
  RowResult initial final row f inner view.
Proof.
  intros view f ls l initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe)
    as [Hfacts [[_ Hbranches] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hbranches, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hbranches. destruct Hbranches as [Hexits _].
  destruct Hfacts as [[Hwi [n [Hall Hkeep]]] _]. rewrite Nat.add_0_r in Hkeep.
  (* The index is the top operand; `under` are the operands below it. *)
  assert (Hnonempty : values <> []) by (intros ->; cbn in Hkeep; lia).
  destruct (exists_last Hnonempty) as [under [tableIndex ->]]. clear Hnonempty.
  rewrite length_app in Hkeep. cbn [length] in Hkeep.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  apply Forall_app in Hvb as [Hsegs Hvalues]. apply Forall_app in Hvalues as [Hunder Hcond]. apply Forall_inv in Hcond.
  (* The circuit reads the table and the index, and picks the edge. *)
  pose proof (encoded_path_br_table _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as [Hop [Hkind [Hlen Hedges]]].
  unfold pathPC in Hop, Hkind, Hlen, Hedges.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Hop, Hkind, Hlen, Hedges.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite Hop in Hfetch.
  injection Hfetch as Hcode.
  assert (Hchosen : row COL_SEL_BR_TABLE = 1)
    by (apply (full_opcode_selects row COL_SEL_BR_TABLE CODE_BR_TABLE Hprime Hfull Hprogram);
        [vm_compute; reflexivity | congruence]).
  pose proof (circuit_row_br_table_len_fetch _ _ _ Hreplay Hchosen) as HlenRead.
  rewrite Hlen in HlenRead. injection HlenRead as HlenValue.
  pose proof (circuit_row_edge_kind_fetch _ _ _ Hreplay Hprogram) as HkindFetch.
  rewrite Hkind in HkindFetch. injection HkindFetch as HkindValue.
  pose proof (full_static_pc_rom_active row Hprime Hfull Hprogram ltac:(congruence)) as Hgate.
  pose proof (circuit_row_next_pc_fetch _ _ _ Hreplay Hgate) as Hnext.
  destruct (full_selected_no_host_switch row COL_SEL_BR_TABLE Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (branch_depth_preserved row COL_SEL_BR_TABLE Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hdepth.
  destruct (full_branch_frame row COL_SEL_BR_TABLE Hprime Hfull ltac:(vm_compute; tauto) Hchosen ltac:(cbn; tauto))
    as [Hpop Hpush _ _ _].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp. unfold viewOperands in Hsp.
  cbn [viewSegments viewValues] in Hsp.
  assert (Hshape : UnaryStackShape row true false).
  { apply full_br_table_stack_shape; [exact Hprime | exact Hfull | exact Hchosen | ].
    cbn [beforeBoundary boundaryStackBase] in Hbase0. rewrite Hsp, !length_app. cbn [length]. lia. }
  destruct (pop_stack_replay row initial final Hshape Hreplay) as [HreadCell HstackMem].
  assert (Hread : row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = tableIndex).
  { assert (Hlookup : nth_error (concat segments ++ under ++ [tableIndex]) (length (concat segments ++ under)) =
        Some tableIndex) by (rewrite app_assoc, nth_error_app2, Nat.sub_diag by lia; reflexivity).
    pose proof (Hcells _ _ Hlookup) as Hcell.
    rewrite (local_read_address _ _ _ Hshape Logic.eq_refl) in HreadCell.
    replace (2 * (row COL_SP_BEFORE - 1)) with
      (2 * (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat segments ++ under))))
      in HreadCell by (rewrite Hsp, !length_app; cbn [length]; lia).
    congruence. }
  pose proof (full_br_table_choice row Hprime Hfull Hchosen) as Hchoice. rewrite Hread, <- HlenValue in Hchoice.
  set (k := if tableIndex <? Z.of_nat (length ls) then S (Z.to_nat tableIndex) else 0%nat).
  assert (HchoiceK : row COL_CONTROL_CHOICE = Z.of_nat k).
  { rewrite Hchoice. unfold k. destruct (tableIndex <? Z.of_nat (length ls)); [lia | reflexivity]. }
  assert (Hk : nth_error (l :: ls) k = Some (brTableTarget ls l tableIndex)).
  { unfold k, brTableTarget. destruct (tableIndex <? Z.of_nat (length ls)) eqn:Hin; [ | reflexivity].
    apply Z.ltb_lt in Hin. cbn [nth_error]. apply nth_error_nth'. lia. }
  set (target := brTableTarget ls l tableIndex) in *.
  assert (Hwft : wf_uN 32%N target).
  { assert (Hls : Forall (wf_uN 32%N) (l :: ls)) by (inversion Hwi; constructor; assumption).
    exact (proj1 (Forall_forall _ _) Hls _ (nth_error_In _ _ Hk)). }
  pose proof (proj1 (Forall_forall _ _) Hexits _ (nth_error_In _ _ Hk)) as Hexit. cbn beta in Hexit.
  pose proof (proj1 (Forall_forall _ _) Hall _ (nth_error_In _ _ Hk)) as Harity. cbn beta in Harity.
  set (depth := N.to_nat (target :> N)) in *.
  (* The target block, and the `depth` labels inside it. *)
  destruct (branch_target_split _ _ _ _ _ _ _ _ _ _ _ Hframe Hexit)
    as [pre [b [post [segsHead [segT [segsTail [outer [ti [body [height [-> [-> [Hpost [HtailLen [HheadLen
      [Hpath' [Hb [Hinside [Hheights0 [Hbelow [Hheights' [Hsupti [HtargetArity [HreadyHead [HreadyTail Hwfouter]]]]]]]]]]]]]]]]]]]]]]]]].
  rewrite HtargetArity in Harity. injection Harity as <-.
  rewrite concat_app in Hsegs. cbn [concat] in Hsegs.
  apply Forall_app in Hsegs as [Hhead Hsegs]. apply Forall_app in Hsegs as [Hsegment Htail].
  rewrite HchoiceK, (Hedges k target pre b post outer ti body Hk Logic.eq_refl (Logic.eq_sym Hpost) Hpath' Hb) in Hnext.
  injection Hnext as Hnext.
  pose proof (full_branch_fbp_unchanged row COL_SEL_BR_TABLE Hprime Hfull ltac:(cbn; tauto) Hchosen) as Hfbp.
  (* The slide keeps the target's results and drops the operands below them, down
     to the target's base. *)
  set (distance := (length under - labelArity ti + length (concat segsTail))%nat).
  assert (HslideAfter : row COL_FRAME_MOVE_REMAINING_AFTER = slideWord true (labelArity ti) distance /\
      row COL_FRAME_MOVE_DISTANCE_AFTER = Z.of_nat distance).
  { assert (Hslide : row COL_FRAME_SLIDE_ACTIVE = 1).
    { rewrite (full_frame_slide_gate row Hprime Hfull), Hpop, Hchosen.
      repeat rewrite (full_opcode_selector_exclusive row COL_SEL_BR_TABLE) by
        (assumption || vm_compute; tauto || discriminate). reflexivity. }
    destruct (circuit_row_slide_fetch _ _ _ Hreplay Hslide) as [Hremaining Hdistance].
    destruct (encoded_br_table_slide _ _ _ _ _ _ _ _ _ _ _ k target Hloaded Hf Hpath Hi Hheights Hk)
      as [Hkept Hrom].
    rewrite <- Hpc, <- HchoiceK, Hremaining in Hkept. rewrite <- Hpc, <- HchoiceK, Hdistance in Hrom.
    match type of Hkept with context [branchSlide ?hs ?ars ?top ?t] =>
      assert (Hcompute : branchSlide hs ars top t = Some (labelArity ti, distance));
      [replace t with depth by reflexivity | rewrite Hcompute in Hkept, Hrom] end.
    { rewrite length_app, Nat.add_0_r. cbn [length]. rewrite Nat.add_1_r. unfold branchSlide, labelSlide.
      rewrite HtargetArity. replace (Nat.leb (labelArity ti) (length under)) with true
        by (symmetry; apply Nat.leb_le; lia).
      rewrite <- HtailLen, branch_drop_labels. reflexivity. }
    unfold slideEntry in Hkept, Hrom. injection Hkept as ->. injection Hrom as ->. split; reflexivity. }
  destruct HslideAfter as [HkeptAfter HdistAfter].
  assert (Hwrites : filter isStackWrite (eventsForRow memories row) = []).
  { rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite (local_write_active _ _ _ Hshape), (local_write_hi_active _ _ _ Hshape), (local_write_gate _ _ _ Hshape),
      (local_gather_lo_inactive _ _ _ Hshape), (local_gather_inactive _ _ _ Hshape). reflexivity. }
  assert (Hregs : frameRegistersPreserved row)
    by (split; [exact Hdepth | split; [exact (local_stack_base_after _ _ _ Hshape) | exact Hfbp]]).
  assert (Heffect : FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final)
    by (eapply quiet_row_frame_effect; [exact Hreplay | exact Hwrites | exact Hpush | exact Hsuspend |
      exact HcontextRead]).
  (* SpecTec: the table picks the same target, then branches to it. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  pose proof (forall_skipn _ _ (S offset) _ Hwfcode) as Hwfrest.
  pose proof (br_table_steps z tableIndex ls l (skipn (S offset) code) Hz Hcond Hwi Hwfrest) as Hsteps.
  assert (Hindex : target = mk_uN (N.of_nat (length post + 0))).
  { clearbody target. destruct target as [tn]. cbn in Hpost |- *. rewrite Nat.add_0_r, Hpost. f_equal.
    unfold depth. cbn. lia. }
  fold target in Hsteps. rewrite Hindex in Hsteps, Hwft.
  destruct (split_top_values under (labelArity ti) ltac:(lia)) as [dropped [kept [-> Hlength]]].
  apply Forall_app in Hunder as [Hdropped Hkept].
  assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
    by (intros xs ys k'; rewrite nth_error_app2 by lia; f_equal; lia).
  unfold viewOperands in Hcells. cbn [viewSegments viewValues] in Hcells.
  rewrite concat_app in Hsp, Hcells. cbn [concat] in Hsp, Hcells.
  replace ((concat segsHead ++ segT ++ concat segsTail) ++ (dropped ++ kept) ++ [tableIndex]) with
    ((concat segsHead ++ segT) ++ (concat segsTail ++ dropped) ++ kept ++ [tableIndex])
    in Hsp, Hcells by (rewrite <- !app_assoc; reflexivity).
  apply (taken_branch_result _ _ _ _ _ _ index pre b post segsHead segT segsTail outer ti body _ height locals
    dropped kept (skipn (S offset) code) distance Hf Hpath' Hb Hinside HheadLen (eq_trans HtailLen (eq_sym Hpost)) HreadyHead HreadyTail
    Hwfouter Hheights0 Hbelow Hheights' Hsupti Hlength).
  + apply Forall_app. split; [exact Htail | apply Forall_app; split; assumption].
  + exact Hwft.
  + exact Hwfrest.
  + unfold frameConfig. cbn [viewPosition viewSegments positionLabels]. rewrite (skipn_at_instruction _ _ _ Hi).
    apply steps_in_labels; [exact Hready | ].
    replace (stackConfig z ((dropped ++ kept) ++ [tableIndex]) (BR_TABLE ls l :: skipn (S offset) code))
      with (stackPrefix (dropped ++ kept) (stackConfig z [tableIndex] (BR_TABLE ls l :: skipn (S offset) code)))
      by (rewrite stack_prefix_config; reflexivity).
    match goal with |- Steps _ (stackConfig ?z' (dropped ++ kept) ?code') =>
      replace (stackConfig z' (dropped ++ kept) code') with (stackPrefix (dropped ++ kept) (stackConfig z' [] code'))
        by (rewrite stack_prefix_config, app_nil_r; reflexivity) end.
    eapply steps_with_stack_prefix; [apply Forall_app; split; assumption | exact Hsteps].
  + cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
  + pose proof (local_stack_base_after _ _ _ Hshape).
    cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
  + rewrite app_assoc. exact (proj2 (Forall_app _ _ _) (conj
      (proj2 (Forall_app _ _ _) (conj Hhead Hsegment)) Hkept)).
  + intros index' value Hv. cbn [afterBoundary boundaryStackBase].
    rewrite (local_stack_base_after _ _ _ Hshape), HstackMem. apply Hcells.
    rewrite nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate].
  + intros j value Hv. cbn [afterBoundary boundaryStackBase].
    rewrite (local_stack_base_after _ _ _ Hshape), HstackMem.
    replace (length (concat segsHead ++ segT) + distance + j)%nat with
      (length (concat segsHead ++ segT) + (length (concat segsTail ++ dropped) + j))%nat
      by (unfold distance; rewrite !length_app; lia).
    apply Hcells. rewrite !Hshift, nth_error_app1 by (apply nth_error_Some; rewrite Hv; discriminate). exact Hv.
  + rewrite Hlength. exact HkeptAfter.
  + intros _. exact HdistAfter.
  + cbn [afterBoundary boundarySP boundaryStackBase].
    rewrite (local_stack_pointer_after _ _ _ Hshape), (local_stack_base_after _ _ _ Hshape), HdistAfter, Hsp.
    unfold distance. rewrite !length_app. cbn [length]. lia.
  + exact Hllen.
  + exact Hlb.
  + intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HstackMem. exact (Hlc _ _ Hv).
  + cbn [afterBoundary boundaryPC]. rewrite <- Hnext. reflexivity.
  + exact (full_initialization_flags row Hprime Hfull).
  + exact (branch_no_zeroing row COL_SEL_BR_TABLE Hprime Hfull ltac:(vm_compute; tauto) Hchosen
      ltac:(vm_compute; tauto) Hidle).
  + exact Hregs.
  + exact Heffect.
Qed.

End Rows.
