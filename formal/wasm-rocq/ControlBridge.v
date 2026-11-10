(* For validated programs in the supported integer fragment, the typed decoder and
   the relational invariant agree: a decoded state is a ControlState, and every
   ControlState decodes at its own shape. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog Add Locals StraightLineSpec StackSimulation
  SpecTecCalls SpecTecBranches CallerSpec ProgramSpec TableState ProgramEncoding CodeLayout FrameInitialization
  Fragment ControlDecode ControlInvariant FrameFacts IntegerMemory IntegerDecode.
Import ListNotations.
Open Scope Z_scope.

Definition siteOf (position : ControlPosition) : ActiveSite :=
  match position with
  | atCode view => siteCode (viewPosition view)
  | resolved view _ => siteResolved (viewPosition view)
  | entering function _ => siteEntering function
  end.

Definition shapeOf (cursor : ControlCursor) : ControlShape :=
  controlShape (map (fun suspended => viewPosition (callView suspended)) (suspendedCalls cursor))
    (siteOf (activePosition cursor)).

Ltac bools := repeat match goal with
  | H : (_ && _) = true |- _ => apply andb_true_iff in H; destruct H
  | H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H
  | H : Z.leb _ _ = true |- _ => apply Z.leb_le in H
  | H : Z.ltb _ _ = true |- _ => apply Z.ltb_lt in H
  | H : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in H
  | H : Nat.leb _ _ = true |- _ => apply Nat.leb_le in H
  end.

Ltac refute_checks := repeat match goal with
  | H : (_ && _) = false |- _ => apply andb_false_iff in H; destruct H as [H | H]
  | H : Z.eqb _ _ = false |- _ => apply Z.eqb_neq in H
  | H : Z.leb _ _ = false |- _ => apply Z.leb_gt in H
  | H : Z.ltb _ _ = false |- _ => apply Z.ltb_ge in H
  | H : Nat.eqb _ _ = false |- _ => apply Nat.eqb_neq in H
  | H : Nat.leb _ _ = false |- _ => apply Nat.leb_gt in H
  end.

(* The decoder's checks, run on a branch that the facts in context make true. *)
Ltac pass_check :=
  match goal with |- context [if ?b then _ else None] =>
    let Hb := fresh "Hcheck" in
    destruct b eqn:Hb; [ | exfalso; refute_checks; unfold frameLocals in *; first [lia | congruence | contradiction]]
  end.

Lemma flags_check : forall boundary, initializationFlagsb boundary = true <-> initializationFlags boundary.
Proof.
  intros boundary. unfold initializationFlagsb, initializationFlags. rewrite andb_true_iff, !Z.eqb_eq. tauto.
Qed.

Lemma read_locals_cells : forall memory fbp types locals,
  readLocals memory fbp types = Some locals <->
  Forall2 (fun t value => integerValueType value = Some t) types locals /\
  forall index value, nth_error locals index = Some value ->
    IntegerMemory.CellMatches memory (fbp + Z.of_nat index) value.
Proof. intros. apply read_integer_cells_some. Qed.

Lemma nth_error_app_cases : forall (A : Type) (low high : list A) index value,
  nth_error (low ++ high) index = Some value ->
  (index < length low /\ nth_error low index = Some value)%nat \/
  (length low <= index /\ nth_error high (index - length low) = Some value)%nat.
Proof.
  intros A low high index value H.
  destruct (Nat.lt_ge_cases index (length low)) as [Hlt | Hge].
  - left. rewrite nth_error_app1 in H by exact Hlt. auto.
  - right. rewrite nth_error_app2 in H by exact Hge. auto.
Qed.

Lemma operand_cell_unmoved : forall base count distance index,
  (index < count)%nat -> operandCell base count 0 distance index = base + Z.of_nat index.
Proof.
  intros base count distance index Hindex. unfold operandCell.
  replace (index <? count - 0)%nat with true by (symmetry; apply Nat.ltb_lt; lia). lia.
Qed.

Lemma read_integer_operands_unmoved : forall memory base types distance,
  readIntegerOperands memory base types 0 distance =
    readIntegerCells memory (fun index => base + Z.of_nat index) types.
Proof.
  intros. unfold readIntegerOperands. apply read_integer_cells_addresses.
  intros. now apply operand_cell_unmoved.
Qed.

Lemma operand_types_height : forall types,
  length (operandTypes types) = totalHeight (typeHeights types).
Proof. intros [segments top]. apply operand_types_length. Qed.

Lemma read_operands_length : forall memory base types remaining distance values,
  readIntegerOperands memory base types remaining distance = Some values -> length values = length types.
Proof. intros. apply read_integer_operands_cells in H as [H _]. symmetry. exact (Forall2_length H). Qed.

Lemma concat_length : forall (segments : list (list val)),
  length (concat segments) = fold_right Nat.add 0%nat (map (@length val) segments).
Proof. induction segments as [ | seg more IH]; cbn; [reflexivity | ]. rewrite length_app, IH. reflexivity. Qed.

Lemma total_height_sum : forall heights top,
  totalHeight (heights, top) = (fold_right Nat.add 0%nat heights + top)%nat.
Proof. intros heights top. unfold totalHeight. cbn. induction heights as [ | h rest IH]; cbn; lia. Qed.

(* Operands read at a frame's heights split into its label segments and its values. *)
Lemma split_stack : forall heights top pending stack,
  (pending <= top)%nat -> length stack = (totalHeight (heights, top) - pending)%nat ->
  map (@length val) (fst (splitSegments heights stack)) = heights /\
  (length (snd (splitSegments heights stack)) + pending)%nat = top /\
  concat (fst (splitSegments heights stack)) ++ snd (splitSegments heights stack) = stack.
Proof.
  intros heights top pending stack Hpending Hlen. rewrite total_height_sum in Hlen.
  destruct (split_lengths heights stack ltac:(lia)) as [Hmap Hconcat].
  split; [exact Hmap | ]. split; [ | exact Hconcat].
  pose proof (f_equal (@length val) Hconcat) as H. rewrite length_app, concat_length, Hmap in H. lia.
Qed.

Lemma frame_reads : forall program memory base fbp pending remaining distance f view,
  FrameAt program memory base fbp pending remaining distance f view ->
  (remaining <= length (viewOperands view))%nat ->
  readIntegerOperands memory base (valueTypes (viewOperands view)) remaining distance = Some (viewOperands view) /\
  readLocals memory fbp (guestFrameTypes f) = Some (viewLocals view).
Proof.
  intros program memory base fbp pending remaining distance f view
    [Hf [_ [_ [_ [_ [Hbounds [Hcells [Hlen [Hlb [Hlc Htyping]]]]]]]]]] HremainingBound.
  split.
  - apply read_integer_operands_cells. split; [eapply cell_values_typed; exact Hcells | exact Hcells].
  -
  apply read_locals_cells. split; [exact (values_typed_types _ _ Hlb) | exact Hlc].
Qed.

Lemma frame_of_reads : forall program memory base fbp pending remaining distance f position inner types stack locals,
  functionAt program (positionFunction position) = Some f ->
  pathCode (guestBody f) (positionLabels position) = Some inner ->
  pathTypes (guestFrameTypes f) (programCallees program (guestModule f)) (guestBody f) (positionLabels position)
    (positionOffset position) = Some types ->
  0 <= fbp -> base = fbp + Z.of_nat (frameLocals f) -> (pending <= length (snd types))%nat ->
  readIntegerOperands memory base (firstn (length (operandTypes types) - pending) (operandTypes types))
    remaining distance = Some stack ->
  readLocals memory fbp (guestFrameTypes f) = Some locals ->
  let heights := typeHeights types in
  let view := frameView position (fst (splitSegments (fst heights) stack))
    (snd (splitSegments (fst heights) stack)) locals in
  FrameAt program memory base fbp pending remaining distance f view /\ innerCode f position = inner /\
  viewOperands view = stack.
Proof.
  intros program memory base fbp pending remaining distance f position inner [segments top] stack locals
    Hf Hpath Htypes Hfbp Hbase Hpending Hstack Hlocals heights view.
  cbn [operandTypes snd fst] in Hpending, Hstack.
  pose proof (path_types_heights _ _ _ _ _ _ _ Htypes) as Hheights.
  assert (Hlength : length stack = (length (concat segments ++ top) - pending)%nat).
  { apply read_integer_operands_cells in Hstack as [Htyped _].
    pose proof (Forall2_length Htyped) as H. rewrite length_firstn, Nat.min_l in H by lia. symmetry; exact H. }
  apply read_integer_operands_cells in Hstack as [Htyped Hcells].
  pose proof (values_typed_projection _ _ Htyped) as HstackTypes.
  change (valueTypes stack = firstn (length (concat segments ++ top) - pending) (concat segments ++ top)) in HstackTypes.
  rewrite operand_types_length, operand_types_without_pending in HstackTypes by exact Hpending.
  destruct (split_stack_types segments (firstn (length top - pending) top) stack HstackTypes) as [Hsegments Hvalues].
  assert (Hwf : Forall wf_val stack) by exact (values_typed_wf _ _ Htyped).
  destruct (proj1 (read_locals_cells _ _ _ _) Hlocals) as [Hlt Hlc].
  assert (Hlb : valuesTyped (guestFrameTypes f) locals).
  { apply values_typed_intro; [exact Hlt | eapply read_integer_cells_wf; exact Hlocals]. }
  assert (Hllen : length locals = frameLocals f).
  { rewrite <- (Forall2_length Hlt). unfold guestFrameTypes. rewrite length_app. reflexivity. }
  cbn [operandTypes fst snd] in Hlength. rewrite operand_types_length in Hlength.
  destruct (split_stack (map (@length valtype) segments) (length top) pending stack Hpending Hlength)
    as [Hmap [Htop Hconcat]].
  assert (Hinner : innerCode f position = inner) by (unfold innerCode; rewrite Hpath; reflexivity).
  assert (Hoperands : viewOperands view = stack) by exact Hconcat.
  split; [ | split; [exact Hinner | exact Hoperands]].
  unfold FrameAt, view. cbn [viewPosition viewSegments viewValues viewLocals].
  unfold heights, typeHeights. cbn [fst snd].
  split; [exact Hf | ]. split; [rewrite Hinner; exact Hpath | ].
  split; [rewrite Hmap, Htop; exact Hheights | ]. split; [exact Hfbp | ]. split; [exact Hbase | ].
  match goal with |- Forall wf_val ?operands /\ _ =>
    replace operands with stack by (symmetry; exact Hoperands) end.
  split; [exact Hwf | ]. split; [exact Hcells | ]. split; [exact Hllen | ]. split; [exact Hlb | ].
  split; [exact Hlc | ].
  exists (skipn (length top - pending) top). split; [rewrite length_skipn; lia | ].
  cbn [viewPosition positionLabels positionOffset viewSegments viewValues].
  unfold heights, typeHeights. cbn [fst snd]. rewrite Hsegments, Hvalues, firstn_skipn. exact Htypes.
Qed.

Lemma frame_heights : forall program memory base fbp pending remaining distance f view,
  FrameAt program memory base fbp pending remaining distance f view ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels (viewPosition view))
    (positionOffset (viewPosition view)) = Some (map (@length val) (viewSegments view), (length (viewValues view) + pending)%nat) /\
  totalHeight (map (@length val) (viewSegments view), (length (viewValues view) + pending)%nat) =
    (length (viewOperands view) + pending)%nat.
Proof.
  intros program memory base fbp pending remaining distance f view Hframe.
  split; [exact (proj1 (proj2 (proj2 Hframe))) | ].
  apply total_height_operands.
Qed.

Lemma function_index_unique : forall program index index' f,
  functionAt program index = Some f -> functionAt program index' = Some f -> index = index'.
Proof.
  intros program index index' f H H'.
  destruct (function_at_typed _ _ _ H) as [_ [Ha _]]. destruct (function_at_typed _ _ _ H') as [_ [Ha' _]].
  rewrite Ha in Ha'. lia.
Qed.

Lemma calls_into_check : forall program module i callee f,
  functionAt program callee = Some f ->
  callsIntob module i callee f = true <-> callsInto (programCallees program module) i f.
Proof.
  intros program module i callee f Hf.
  destruct i; cbn [callsIntob callsInto]; try (split; [discriminate | contradiction]);
    unfold programCallees; cbn [calleeAt envTypes].
  - destruct (nth_error (FUNCS module) _) as [address | ]; [ | split; discriminate].
    split; intros H.
    + apply Nat.eqb_eq in H. rewrite H. exact Hf.
    + apply Nat.eqb_eq. eapply function_index_unique; eassumption.
  - destruct (nth_error (TYPES module) _) as [t | ]; [ | split; discriminate].
    destruct (functypeEqDec t (guestType f)) as [-> | Hne]; split; intros H; try reflexivity; try discriminate.
    injection H as ->. contradiction.
Qed.

Lemma decode_active_function : forall program s memory boundary site r,
  decodeActive program s memory boundary site = Some r -> exists f, functionAt program (siteFunction site) = Some f.
Proof.
  intros program s memory boundary [position | position | function] r H; cbn [siteFunction];
    unfold decodeActive in H; destruct (functionAt program _) as [f | ]; eauto; discriminate.
Qed.

Lemma active_function_site : forall program position,
  activeFunction program position = functionAt program (siteFunction (siteOf position)).
Proof. intros program [view | view callee | function arguments]; reflexivity. Qed.

(* Decoded active frames satisfy the invariant. *)
Lemma decoded_active : forall program s memory boundary site c obligations,
  store_FUNCS s = program -> programValidated program s ->
  decodeActive program s memory boundary site = Some (c, obligations) -> obligations ->
  exists position, siteOf position = site /\ ActiveState program s memory boundary position c.
Proof.
  intros program s memory boundary [position | position | function] c obligations Hstore Hvalid Hdecode Hobligations;
    unfold decodeActive in Hdecode; cbv zeta in Hdecode.
  - destruct (functionAt program (positionFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (pathCode (guestBody f) (positionLabels position)) as [inner | ] eqn:Hpath; [ | discriminate].
    destruct (pathTypes _ _ _ _ _) as [types | ] eqn:Htypes; [ | discriminate].
    set (heights := typeHeights types) in *.
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    destruct (readIntegerOperands _ _ _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals memory (boundaryLocalsBase boundary) (guestFrameTypes f)) as [locals | ] eqn:Hlocals;
      [ | discriminate].
    injection Hdecode as <- <-.
    pose proof (read_operands_length _ _ _ _ _ _ Hstack) as Hlen. rewrite operand_types_height in Hlen.
    fold heights in Hlen.
    assert (Hstack' : readIntegerOperands memory (boundaryStackBase boundary)
      (firstn (length (operandTypes types) - 0) (operandTypes types))
      (Z.to_nat (boundaryMoveRemaining boundary)) (boundaryMoveDistance boundary) = Some stack).
    { now rewrite Nat.sub_0_r, firstn_all. }
    destruct (frame_of_reads program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0%nat
      _ _ _ _ _ _ _ _ Hf Hpath Htypes ltac:(assumption) ltac:(assumption) ltac:(lia) Hstack' Hlocals)
      as [Hframe [Hinner Hoperands]].
    pose proof (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hframeheights _].
    assert (Hheights : pathHeights (programCallees program (guestModule f)) (guestBody f)
      (positionLabels position) (positionOffset position) = Some heights).
    { destruct types as [segments top]. exact (path_types_heights _ _ _ _ _ _ _ Htypes). }
    cbn [viewPosition] in Hframeheights. rewrite Hheights in Hframeheights.
    assert (Hheightseq : snd heights = (length (snd (splitSegments (fst heights) stack)) + 0)%nat).
    { change (Some heights = Some (map (@length val) (fst (splitSegments (fst heights) stack)),
        (length (snd (splitSegments (fst heights) stack)) + 0)%nat)) in Hframeheights.
      pose proof (f_equal (fun result : option (list nat * nat) =>
        match result with Some heightPair => snd heightPair | None => 0%nat end) Hframeheights) as Htop.
      exact Htop. }
    exists (atCode (frameView position (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals)).
    split; [reflexivity | ]. exists f. cbn [viewPosition viewLocals viewValues].
    split; [exact Hframe | ]. rewrite Hinner. split; [assumption | ].
    split; [fold heights in Hoperands; rewrite Hoperands, Hlen; assumption | ].
    split; [assumption | ]. split; [assumption | ]. split; [assumption | ]. split; [lia | reflexivity].
  - destruct (functionAt program (positionFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (pathCode (guestBody f) (positionLabels position)) as [inner | ] eqn:Hpath; [ | discriminate].
    destruct (nth_error inner (positionOffset position)) as [i | ] eqn:Hi; [ | discriminate].
    destruct (pathTypes _ _ _ _ _) as [types | ] eqn:Htypes; [ | discriminate].
    set (heights := typeHeights types) in *.
    destruct (memory stackId _) as [reference | ] eqn:Href; [ | discriminate].
    destruct (functionAt program (Z.to_nat (reference - 1))) as [g | ] eqn:Hg; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    destruct (readIntegerOperands _ _ _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals memory (boundaryLocalsBase boundary) (guestFrameTypes f)) as [locals | ] eqn:Hlocals;
      [ | discriminate].
    injection Hdecode as <- <-.
    assert (Htop : (snd heights <= totalHeight heights)%nat)
      by (destruct heights as [hs top]; rewrite total_height_sum; cbn [snd]; lia).
    pose proof (read_operands_length _ _ _ _ _ _ Hstack) as Hlen.
    rewrite length_firstn, operand_types_height in Hlen. fold heights in Hlen.
    rewrite Nat.min_l in Hlen by lia.
    unfold heights in Hstack. rewrite <- operand_types_height in Hstack.
    destruct (frame_of_reads program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 1%nat
      0%nat 0 _ _ _ _ _ _ Hf Hpath Htypes ltac:(assumption) ltac:(assumption)
      ltac:(change (1 <= snd heights)%nat; assumption) Hstack Hlocals) as [Hframe [Hinner Hoperands]].
    exists (resolved (frameView position (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals) (Z.to_nat (reference - 1))).
    split; [reflexivity | ]. exists f, g, i. cbn [viewPosition viewLocals viewValues].
    split; [exact Hframe | ]. rewrite Hinner.
    split; [exact Hi | ]. split; [assumption | ]. split; [exact Hg | ].
    split; [apply (proj1 (calls_into_check _ _ _ _ _ Hg)); assumption | ].
    fold heights in Hoperands. rewrite Hoperands, Hlen.
    split; [rewrite Href; f_equal; lia | ].
    split; [lia | ]. split; [assumption | ]. split; [assumption | ]. split; [assumption | ].
    split; [assumption | reflexivity].
  - destruct (functionAt program function) as [f | ] eqn:Hf; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    destruct (readLocals _ _ _) as [arguments | ] eqn:Hstack; [ | discriminate].
    destruct (integerZeros _) as [defaults | ] eqn:Hdefaults; [ | discriminate].
    injection Hdecode as <- <-.
    apply read_integer_cells_some in Hstack as [Htypes _].
    exists (entering function arguments). split; [reflexivity | ]. exists f, defaults.
    split; [exact Hf | ]. split; [exact Htypes | ]. split; [exact Hdefaults | ]. split; [exact Hobligations | ].
    split; [assumption | ]. split; [assumption | ]. split; [lia | ]. split; [assumption | reflexivity].
Qed.

(* Active frames of the invariant decode at their own site. *)
Lemma active_decodes : forall program s position memory boundary c,
  store_FUNCS s = program -> programValidated program s ->
  ActiveState program s memory boundary position c ->
  exists obligations, decodeActive program s memory boundary (siteOf position) = Some (c, obligations) /\ obligations.
Proof.
  intros program s [view | view callee | function arguments] memory boundary c Hstore Hvalid Hstate; cbn [siteOf];
    unfold decodeActive; cbv zeta.
  - destruct Hstate as [f [Hframe [Hoffset [Hsp [Hpc [Hflags [Hzero [HremainingBound ->]]]]]]]].
    pose proof Hframe as [Hf [Hpath [_ [Hfbp [Hbase _]]]]].
    destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
    destruct (frame_typing_decode _ _ _ _ (frame_typing _ _ _ _ _ _ _ _ _ Hframe))
      as [types [Htypes [HtypeHeights Hkept]]].
    rewrite Nat.sub_0_r, firstn_all in Hkept.
    assert (HremainingBoundOps : (Z.to_nat (boundaryMoveRemaining boundary) <= length (viewOperands view))%nat).
    { unfold viewOperands. rewrite length_app. lia. }
    destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe HremainingBoundOps) as [Hstack Hlocals].
    rewrite Hf, Hpath, Htypes, HtypeHeights, Htotal, Hkept.
    cbn [snd]. rewrite !Nat.add_0_r.
    apply flags_check in Hflags. pass_check.
    rewrite Hstack, Hlocals. cbn [fst snd].
    unfold viewOperands. rewrite split_concat.
    eexists. split; [reflexivity | exact I].
  - destruct Hstate as [f [g [i [Hframe [Hi [Hind [Hg [Hcalls [Href [Hsp [Hpc [Hflags [Hzero [Hmove ->]]]]]]]]]]]]]].
    pose proof Hframe as [Hf [Hpath [_ [Hfbp [Hbase _]]]]].
    destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
    destruct (frame_typing_decode _ _ _ _ (frame_typing _ _ _ _ _ _ _ _ _ Hframe))
      as [types [Htypes [HtypeHeights Hkept]]].
    pose proof (operand_types_height types) as Hlength. rewrite HtypeHeights, Htotal in Hlength.
    rewrite Hlength, Nat.add_sub in Hkept.
    destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe (Nat.le_0_l _)) as [Hstack Hlocals].
    unfold innerCode in Hi. rewrite Hpath in Hi.
    rewrite Hf, Hpath, Hi, Htypes, HtypeHeights, Htotal, Nat.add_sub, Href.
    replace (Z.to_nat (Z.of_nat callee + 1 - 1)) with callee by lia. rewrite Hg.
    pose proof (proj2 (calls_into_check _ _ _ _ _ Hg) Hcalls) as Hcallsb.
    cbn [snd]. apply flags_check in Hflags. pass_check.
    rewrite Hkept, Hstack, Hlocals. cbn [fst snd].
    unfold viewOperands. rewrite split_concat.
    eexists. split; [ | exact I]. unfold frameConfig, innerCode. rewrite Hpath. reflexivity.
  - destruct Hstate as [f [defaults [Hf [Htypes [Hdefaults [Hinit [Hflags [Hmove [Hpending [Hpc ->]]]]]]]]]].
    rewrite Hf. apply flags_check in Hflags. pass_check.
    assert (Hargs : readLocals memory (boundaryLocalsBase boundary) (guestParamTypes f) = Some arguments).
    { apply read_integer_cells_some. split; [exact Htypes | exact (argument_cells _ _ _ _ _ Hinit)]. }
    rewrite Hargs, Hdefaults. eexists. split; [reflexivity | exact Hinit].
Qed.

(* Decoded callers form the invariant's suspended stack. *)
Lemma decoded_calls : forall program s memory sites callee f fbp depth inner obligations c result,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program callee = Some f ->
  decodeCalls program memory callee fbp depth sites inner obligations = Some (c, result) ->
  result = obligations /\ exists stack, map (fun suspended => viewPosition (callView suspended)) stack = sites /\
    SuspendedStack program memory f fbp depth stack /\ c = plugCalls stack inner.
Proof.
  intros program s memory sites. induction sites as [ | site sites IH];
    intros callee f fbp depth inner obligations c result Hstore Hvalid Hf Hdecode; cbn [decodeCalls] in Hdecode.
  - injection Hdecode as <- <-. split; [reflexivity | exists []; repeat split].
  - destruct (functionAt program (positionFunction site)) as [g | ] eqn:Hg; [ | discriminate].
    rewrite Hf in Hdecode.
    destruct (pathCode (guestBody g) (positionLabels site)) as [code | ] eqn:Hpath; [ | discriminate].
    destruct (nth_error code (positionOffset site)) as [i | ] eqn:Hcall; [ | discriminate].
    destruct (pathTypes _ _ _ _ _) as [types | ] eqn:Htypes; [ | discriminate].
    set (heights := typeHeights types) in *.
    destruct (memory callStackReturnPcId _) as [returnPC | ] eqn:Hret; [ | discriminate].
    destruct (memory callStackCallerFbpId _) as [savedFbp | ] eqn:HsavedFbp; [ | discriminate].
    destruct (memory callStackCallerSpBaseId _) as [savedBase | ] eqn:HsavedBase; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    destruct (readIntegerCells _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals memory savedFbp (guestFrameTypes g)) as [locals | ] eqn:Hlocals; [ | discriminate].
    destruct (IH _ _ _ _ _ _ _ _ Hstore Hvalid Hg Hdecode) as [Hresult [stack' [Hsites [Hsusp ->]]]].
    split; [exact Hresult | ].
    rewrite <- (read_integer_operands_unmoved memory savedBase _ 0) in Hstack.
    pose proof (read_operands_length _ _ _ _ _ _ Hstack) as Hlen.
    rewrite length_firstn, operand_types_height in Hlen. fold heights in Hlen.
    rewrite Nat.min_l in Hlen by lia.
    unfold heights in Hstack. rewrite <- operand_types_height in Hstack.
    destruct (frame_of_reads program memory savedBase savedFbp (callOperands i f) 0%nat 0 _ _ _ _ _ _ Hg Hpath Htypes
      ltac:(lia) ltac:(lia) ltac:(change (callOperands i f <= snd heights)%nat; assumption)
      Hstack Hlocals) as [Hframe [Hinner Hoperands]].
    exists (suspendedCall (frameView site (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals) g (guestResults f) savedBase savedFbp :: stack').
    split; [cbn; rewrite Hsites; reflexivity | ].
    split; [ | cbn [plugCalls]; unfold callerConfig, frameConfig;
      cbn [callView callerFunction viewPosition viewSegments viewValues viewLocals callResults]; rewrite Hinner;
      reflexivity].
    exists i. cbn [callView callerFunction viewPosition callerBase callerFbp callResults].
    split; [exact Hframe | ]. split; [rewrite Hinner; exact Hcall | ].
    split; [apply (proj1 (calls_into_check _ _ _ _ _ Hf)); assumption | ]. split; [reflexivity | ].
    split; [fold heights in Hoperands; rewrite Hoperands, Hlen; assumption | ].
    split; [rewrite Hret; f_equal; assumption | ].
    split; [exact HsavedFbp | ]. split; [exact HsavedBase | exact Hsusp].
Qed.

(* The invariant's suspended stack decodes at its own call sites. *)
Lemma calls_decode : forall program s memory stack callee f fbp depth inner obligations,
  store_FUNCS s = program -> programValidated program s ->
  functionAt program callee = Some f -> SuspendedStack program memory f fbp depth stack ->
  decodeCalls program memory callee fbp depth
    (map (fun suspended => viewPosition (callView suspended)) stack) inner obligations =
    Some (plugCalls stack inner, obligations).
Proof.
  intros program s memory stack. induction stack as [ | top stack IH];
    intros callee f fbp depth inner obligations Hstore Hvalid Hf Hsusp; cbn [map decodeCalls plugCalls]; [reflexivity | ].
  destruct Hsusp as [x [Hframe [Hcall [Hcallee [Hres [Hfbp [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]].
  pose proof Hframe as [Hg [Hpath [_ [Hfbp0 [Hbase0 _]]]]].
  destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
  destruct (frame_typing_decode _ _ _ _ (frame_typing _ _ _ _ _ _ _ _ _ Hframe))
    as [types [Htypes [HtypeHeights Hkept]]].
  pose proof (operand_types_height types) as Hlength. rewrite HtypeHeights, Htotal in Hlength.
  rewrite Hlength, Nat.add_sub in Hkept.
  destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe ltac:(lia)) as [Hstack Hlocals].
  rewrite read_integer_operands_unmoved in Hstack.
  rewrite Hg, Hf, Hpath, Hcall.
  pose proof (proj2 (calls_into_check _ _ _ _ _ Hf) Hcallee) as Hcallsb.
  rewrite Htypes, HtypeHeights, Hret, HsavedFbp, HsavedBase.
  rewrite Htotal, Nat.add_sub. cbn [fst snd]. pass_check.
  rewrite Hkept, Hstack, Hlocals. cbn [fst snd].
  unfold viewOperands. rewrite split_concat. cbn [fst snd].
  rewrite <- Hres. exact (IH _ _ _ _ _ _ Hstore Hvalid Hg Hrest).
Qed.

(* The bridge: what the audited statement says, in the invariant's terms. *)
Theorem circuit_config_state : forall program s baseDepth shape memory boundary c,
  CircuitConfig program s baseDepth shape memory boundary c ->
  exists cursor, shapeOf cursor = shape /\ ControlState program s baseDepth cursor memory boundary c.
Proof.
  intros program s baseDepth shape memory boundary c [Hstore [_ [Hvalid [_ [obligations [Hdecode Hobligations]]]]]].
  unfold decodeControl in Hdecode.
  destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
  destruct (decodeActive program s memory boundary (activeSite shape)) as [[inner active] | ] eqn:Hactive;
    [ | discriminate]. cbn [fst snd] in Hdecode.
  destruct (decode_active_function _ _ _ _ _ _ Hactive) as [f Hf].
  destruct (decoded_calls _ _ _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hf Hdecode) as [Hresult [stack [Hsites [Hsusp ->]]]].
  subst obligations.
  destruct (decoded_active _ _ _ _ _ _ _ Hstore Hvalid Hactive Hobligations) as [position [Hsite Hstate]].
  exists (controlCursor stack position). split.
  { unfold shapeOf. cbn [suspendedCalls activePosition]. rewrite Hsites, Hsite. destruct shape; reflexivity. }
  exists f, inner. cbn [activePosition suspendedCalls].
  split; [rewrite active_function_site, Hsite; exact Hf | ]. split; [exact Hstate | ].
  split; [exact Hsusp | ]. split; [assumption | ].
  split; [rewrite <- Hsites, length_map in *; assumption | reflexivity].
Qed.

Theorem control_state_circuit : forall program s baseDepth cursor memory boundary c,
  store_FUNCS s = program -> wf_store s -> programValidated program s -> tablesLoaded s memory ->
  ControlState program s baseDepth cursor memory boundary c ->
  CircuitConfig program s baseDepth (shapeOf cursor) memory boundary c.
Proof.
  intros program s baseDepth [stack position] memory boundary c Hfuncs Hwf Hvalid Htables
    [f [inner [Hf [Hstate [Hsusp [Hbase [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in *.
  split; [exact Hfuncs | split; [exact Hwf | split; [exact Hvalid | split; [exact Htables | ]]]].
  unfold decodeControl, shapeOf. cbn [callSites activeSite activePosition suspendedCalls]. rewrite length_map.
  pass_check.
  destruct (active_decodes _ _ _ _ _ _ Hfuncs Hvalid Hstate) as [active [Hactive Hobligations]].
  rewrite Hactive. cbn [fst snd]. rewrite active_function_site in Hf.
  exists active. split; [exact (calls_decode _ _ _ _ _ _ _ _ _ _ Hfuncs Hvalid Hf Hsusp) | exact Hobligations].
Qed.
