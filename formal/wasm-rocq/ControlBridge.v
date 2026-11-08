(* For validated programs in the current i32 fragment, the typed decoder and
   the relational invariant agree: a decoded state is a ControlState, and every
   ControlState decodes at its own shape. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog Add Locals StraightLineSpec StackSimulation
  SpecTecCalls SpecTecBranches CallerSpec ProgramSpec TableState ProgramEncoding CodeLayout FrameInitialization
  Fragment ControlDecode ControlInvariant FrameFacts.
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

Lemma read_stack_cells : forall memory first count values,
  readStack memory first count = Some values <->
  length values = count /\ Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value -> memory stackId [2 * (first + Z.of_nat index)] = Some value.
Proof. intros. exact (read_cells_some _ _ _ _). Qed.

Lemma read_locals_cells : forall memory fbp types locals,
  readLocals memory fbp types = Some locals <->
  Forall2 (fun t value => integerValueType value = Some t) types locals /\
  forall index value, nth_error locals index = Some value ->
    IntegerMemory.CellMatches memory (fbp + Z.of_nat index) value.
Proof. intros. apply read_integer_cells_some. Qed.

Lemma read_locals_i32 : forall program index f memory fbp,
  functionAt program index = Some f ->
  readLocals memory fbp (guestFrameTypes f) =
    option_map (map i32Value) (readStack memory fbp (frameLocals f)).
Proof.
  intros program index f memory fbp Hf.
  destruct (function_at_i32_types _ _ _ Hf) as [Hp [_ Hl]].
  unfold readLocals, guestFrameTypes. rewrite Hp, Hl, <- repeat_app.
  apply read_integer_cells_i32.
Qed.

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

Lemma read_operands_cells : forall memory base count remaining distance values,
  (remaining <= count)%nat ->
  readOperands memory base count remaining distance = Some values <->
  length values = count /\ Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value ->
    memory stackId [2 * operandCell base count remaining distance index] = Some value.
Proof.
  intros memory base count remaining distance values HremainingBound. unfold readOperands, operandCell.
  split.
  - destruct (readStack memory base (count - remaining)) as [low | ] eqn:Hlow; [ | discriminate].
    destruct (readStack memory _ remaining) as [high | ] eqn:Hhigh; [ | discriminate].
    intros H. injection H as <-.
    destruct (proj1 (read_stack_cells _ _ _ _) Hlow) as [Hl [Hlb Hlc]].
    destruct (proj1 (read_stack_cells _ _ _ _) Hhigh) as [Hh [Hhb Hhc]].
    split; [rewrite length_app; lia | split; [apply Forall_app; auto | ]].
    intros index value Hindex.
    destruct (nth_error_app_cases _ _ _ _ _ Hindex) as [[Hlt Hv] | [Hge Hv]].
    + replace (index <? count - remaining)%nat with true by (symmetry; apply Nat.ltb_lt; lia).
      rewrite Z.add_0_r. exact (Hlc _ _ Hv).
    + replace (index <? count - remaining)%nat with false by (symmetry; apply Nat.ltb_ge; lia).
      pose proof (Hhc _ _ Hv) as Hcell.
      replace (base + Z.of_nat (count - remaining) + distance + Z.of_nat (index - length low))
        with (base + Z.of_nat index + distance) in Hcell by lia. exact Hcell.
  - intros [Hlen [Hbounds Hcells]].
    rewrite <- (firstn_skipn (count - remaining) values) in Hbounds, Hcells |- *.
    apply Forall_app in Hbounds as [Hlb Hhb].
    assert (Hlowlen : length (firstn (count - remaining) values) = (count - remaining)%nat)
      by (rewrite length_firstn; lia).
    rewrite (proj2 (read_stack_cells memory base (count - remaining) (firstn (count - remaining) values))).
    + rewrite (proj2 (read_stack_cells memory (base + Z.of_nat (count - remaining) + distance) remaining
        (skipn (count - remaining) values))); [reflexivity | ].
      split; [rewrite length_skipn; lia | split; [exact Hhb | ]].
      intros index value Hv.
      pose proof (Hcells ((count - remaining) + index)%nat value) as Hcell.
      rewrite nth_error_app2 in Hcell by lia. rewrite Hlowlen in Hcell.
      replace (count - remaining + index - (count - remaining))%nat with index in Hcell by lia.
      specialize (Hcell Hv).
      replace (count - remaining + index <? count - remaining)%nat with false in Hcell by (symmetry; apply Nat.ltb_ge; lia).
      replace (base + Z.of_nat (count - remaining) + distance + Z.of_nat index)
        with (base + Z.of_nat (count - remaining + index) + distance) by lia. exact Hcell.
    + split; [exact Hlowlen | split; [exact Hlb | ]].
      intros index value Hv.
      assert (Hlt : (index < count - remaining)%nat).
      { rewrite <- Hlowlen. apply nth_error_Some. rewrite Hv. discriminate. }
      pose proof (Hcells index value) as Hcell. rewrite nth_error_app1 in Hcell by lia.
      specialize (Hcell Hv).
      replace (index <? count - remaining)%nat with true in Hcell by (symmetry; apply Nat.ltb_lt; lia).
      rewrite Z.add_0_r in Hcell. exact Hcell.
Qed.

Lemma read_operands_unmoved : forall memory base count distance,
  readOperands memory base count 0 distance = readStack memory base count.
Proof.
  intros memory base count distance. unfold readOperands. rewrite Nat.sub_0_r.
  destruct (readStack memory base count) as [low | ]; [ | reflexivity].
  cbn [readStack readCells]. rewrite app_nil_r. reflexivity.
Qed.

Lemma read_integer_operands_i32 : forall memory base count remaining distance,
  (remaining <= count)%nat ->
  readIntegerOperands memory base (repeat valtype_I32 count) remaining distance =
    option_map (map i32Value) (readOperands memory base count remaining distance).
Proof.
  intros memory base count remaining distance Hremaining.
  unfold readIntegerOperands. rewrite repeat_length.
  destruct (readOperands memory base count remaining distance) as [raw | ] eqn:Hraw.
  - apply (read_operands_cells _ _ _ _ _ _ Hremaining) in Hraw as [Hlen [Hbounds Hcells]].
    pose proof (read_integer_cells_i32_from _ _ _ Hbounds Hcells) as Hread.
    rewrite Hlen in Hread. exact Hread.
  - destruct (readIntegerCells _ _ _) as [values | ] eqn:Hread; [ | reflexivity].
    destruct (read_integer_cells_i32_inv _ _ _ _ Hread) as [raw [_ [Hlen [Hbounds Hcells]]]].
    assert (Hsome : readOperands memory base count remaining distance = Some raw).
    { apply read_operands_cells; [exact Hremaining | auto]. }
    congruence.
Qed.

Lemma type_heights_i32 : forall heights,
  typeHeights (map (repeat valtype_I32) (fst heights), repeat valtype_I32 (snd heights)) = heights.
Proof.
  intros [hs top]. unfold typeHeights. cbn [fst snd]. rewrite repeat_length.
  f_equal. induction hs as [ | h rest IH]; [reflexivity | ]. cbn [map]. rewrite repeat_length, IH. reflexivity.
Qed.

Lemma operand_types_i32 : forall heights,
  operandTypes (map (repeat valtype_I32) (fst heights), repeat valtype_I32 (snd heights)) =
    repeat valtype_I32 (totalHeight heights).
Proof.
  intros [hs top]. unfold operandTypes, totalHeight. cbn [fst snd].
  induction hs as [ | h rest IH]; cbn [map concat fold_right app]; [reflexivity | ].
  rewrite <- app_assoc, IH, <- repeat_app. reflexivity.
Qed.

Lemma value_stack_config_i32 : forall z values code,
  valueStackConfig z (map i32Value values) code = stackConfig z values code.
Proof. intros. unfold valueStackConfig, stackConfig. rewrite map_map. reflexivity. Qed.

Lemma value_stack_prefix_i32 : forall values c,
  valueStackPrefix (map i32Value values) c = stackPrefix values c.
Proof. intros values [z code]. unfold valueStackPrefix, stackPrefix. rewrite map_map. reflexivity. Qed.

Lemma wrap_value_labels_i32 : forall labels code segments c,
  wrapValueLabels code labels (map (map i32Value) segments) c = wrapLabels code labels segments c.
Proof.
  induction labels as [ | site rest IH]; intros code [ | segment more] c; try reflexivity.
  cbn [map wrapValueLabels wrapLabels]. destruct (siteEntry code site) as [[i body] | ]; [ | reflexivity].
  rewrite IH, value_stack_prefix_i32. reflexivity.
Qed.

Lemma split_segments_map : forall (A B : Type) (f : A -> B) heights values,
  splitSegments heights (map f values) =
    (map (map f) (fst (splitSegments heights values)), map f (snd (splitSegments heights values))).
Proof.
  intros A B f heights. induction heights as [ | h rest IH]; intros values; [reflexivity | ].
  cbn [splitSegments]. rewrite firstn_map, skipn_map, IH. reflexivity.
Qed.

Lemma operand_cell_unmoved : forall base count distance index,
  (index < count)%nat -> operandCell base count 0 distance index = base + Z.of_nat index.
Proof.
  intros base count distance index Hindex. unfold operandCell.
  replace (index <? count - 0)%nat with true by (symmetry; apply Nat.ltb_lt; lia). lia.
Qed.

Lemma concat_length : forall (segments : list (list Z)),
  length (concat segments) = fold_right Nat.add 0%nat (map (@length Z) segments).
Proof. induction segments as [ | seg more IH]; cbn; [reflexivity | ]. rewrite length_app, IH. reflexivity. Qed.

Lemma total_height_sum : forall heights top,
  totalHeight (heights, top) = (fold_right Nat.add 0%nat heights + top)%nat.
Proof. intros heights top. unfold totalHeight. cbn. induction heights as [ | h rest IH]; cbn; lia. Qed.

(* Operands read at a frame's heights split into its label segments and its values. *)
Lemma split_stack : forall heights top pending stack,
  (pending <= top)%nat -> length stack = (totalHeight (heights, top) - pending)%nat ->
  map (@length Z) (fst (splitSegments heights stack)) = heights /\
  (length (snd (splitSegments heights stack)) + pending)%nat = top /\
  concat (fst (splitSegments heights stack)) ++ snd (splitSegments heights stack) = stack.
Proof.
  intros heights top pending stack Hpending Hlen. rewrite total_height_sum in Hlen.
  destruct (split_lengths heights stack ltac:(lia)) as [Hmap Hconcat].
  split; [exact Hmap | ]. split; [ | exact Hconcat].
  pose proof (f_equal (@length Z) Hconcat) as H. rewrite length_app, concat_length, Hmap in H. lia.
Qed.

Lemma frame_reads : forall program memory base fbp pending remaining distance f view,
  FrameAt program memory base fbp pending remaining distance f view ->
  (remaining <= length (viewOperands view))%nat ->
  readOperands memory base (length (viewOperands view)) remaining distance = Some (viewOperands view) /\
  readLocals memory fbp (guestFrameTypes f) = Some (map i32Value (viewLocals view)).
Proof.
  intros program memory base fbp pending remaining distance f view
    [Hf [_ [_ [_ [_ [Hbounds [Hcells [Hlen [Hlb Hlc]]]]]]]]] HremainingBound.
  split; [apply read_operands_cells; auto | ].
  rewrite (read_locals_i32 _ _ _ _ _ Hf), (proj2 (read_stack_cells _ _ _ _) (conj Hlen (conj Hlb Hlc))).
  reflexivity.
Qed.

Lemma frame_of_reads : forall program memory base fbp pending remaining distance f position inner heights stack locals,
  functionAt program (positionFunction position) = Some f ->
  pathCode (guestBody f) (positionLabels position) = Some inner ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels position)
    (positionOffset position) = Some heights ->
  0 <= fbp -> base = fbp + Z.of_nat (frameLocals f) -> (pending <= snd heights)%nat ->
  (remaining <= totalHeight heights - pending)%nat ->
  readOperands memory base (totalHeight heights - pending) remaining distance = Some stack ->
  readStack memory fbp (frameLocals f) = Some locals ->
  let view := frameView position (fst (splitSegments (fst heights) stack))
    (snd (splitSegments (fst heights) stack)) locals in
  FrameAt program memory base fbp pending remaining distance f view /\ innerCode f position = inner /\
  viewOperands view = stack.
Proof.
  intros program memory base fbp pending remaining distance f position inner [hs top] stack locals Hf Hpath Hheights
    Hfbp Hbase Hpending HremainingBound Hstack Hlocals view. cbn [fst snd] in *.
  destruct (proj1 (read_operands_cells _ _ _ _ _ _ HremainingBound) Hstack) as [Hlen [Hbounds Hcells]].
  destruct (proj1 (read_stack_cells _ _ _ _) Hlocals) as [Hllen [Hlb Hlc]].
  destruct (split_stack hs top pending stack Hpending Hlen) as [Hmap [Hvalues Hconcat]].
  assert (Hinner : innerCode f position = inner) by (unfold innerCode; rewrite Hpath; reflexivity).
  assert (Hoperands : viewOperands view = stack) by exact Hconcat.
  split; [ | split; [exact Hinner | exact Hoperands]].
  unfold FrameAt, view. cbn [viewPosition viewSegments viewValues viewLocals].
  split; [exact Hf | ]. split; [rewrite Hinner; exact Hpath | ].
  split; [rewrite Hmap, Hvalues; exact Hheights | ]. split; [exact Hfbp | ]. split; [exact Hbase | ].
  fold view. rewrite Hoperands, Hlen. split; [exact Hbounds | ]. split; [exact Hcells | ].
  split; [exact Hllen | ]. split; [exact Hlb | exact Hlc].
Qed.

Lemma frame_heights : forall program memory base fbp pending remaining distance f view,
  FrameAt program memory base fbp pending remaining distance f view ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels (viewPosition view))
    (positionOffset (viewPosition view)) = Some (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) /\
  totalHeight (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) =
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
  destruct (function_at_sound _ _ _ H) as [_ [Ha _]]. destruct (function_at_sound _ _ _ H') as [_ [Ha' _]].
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
    rewrite (function_path_types _ _ _ _ _ _ Hstore Hvalid Hf) in Hdecode.
    destruct (pathHeights _ _ _ _) as [heights | ] eqn:Hheights; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite type_heights_i32, operand_types_i32 in Hdecode.
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    assert (HreadBound : (Z.to_nat (boundaryMoveRemaining boundary) <= totalHeight heights)%nat).
    { destruct heights as [hs top]. rewrite total_height_sum. cbn [snd] in *. lia. }
    rewrite read_integer_operands_i32 in Hdecode by exact HreadBound.
    destruct (readOperands _ _ _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    rewrite (read_locals_i32 _ _ _ _ _ Hf) in Hdecode.
    destruct (readStack memory (boundaryLocalsBase boundary) (frameLocals f)) as [locals | ] eqn:Hlocals; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite split_segments_map in Hdecode. cbn [fst snd] in Hdecode.
    rewrite value_stack_config_i32, wrap_value_labels_i32 in Hdecode.
    injection Hdecode as <- <-.
    rewrite <- (Nat.sub_0_r (totalHeight heights)) in Hstack.
    assert (HremainingBound : (Z.to_nat (boundaryMoveRemaining boundary) <= totalHeight heights - 0)%nat).
    { destruct heights as [hs top]. rewrite total_height_sum. cbn [snd] in *. lia. }
    destruct (frame_of_reads program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0%nat
      _ _ _ _ _ _ _ _ Hf Hpath Hheights ltac:(assumption) ltac:(assumption) ltac:(lia) HremainingBound Hstack Hlocals)
      as [Hframe [Hinner Hoperands]].
    pose proof (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hframeheights _].
    cbn [viewPosition] in Hframeheights. rewrite Hheights in Hframeheights. injection Hframeheights as Hheightseq.
    apply (f_equal snd) in Hheightseq. cbn [snd viewValues] in Hheightseq.
    exists (atCode (frameView position (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals)).
    split; [reflexivity | ]. exists f. cbn [viewPosition viewLocals viewValues].
    split; [exact Hframe | ]. rewrite Hinner. split; [assumption | ].
    split; [rewrite Hoperands; destruct (proj1 (read_operands_cells _ _ _ _ _ _ HremainingBound) Hstack) as [Hlen _];
      rewrite Hlen, Nat.sub_0_r; assumption | ].
    split; [assumption | ]. split; [assumption | ]. split; [assumption | ].
    split; [lia | reflexivity].
  - destruct (functionAt program (positionFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (pathCode (guestBody f) (positionLabels position)) as [inner | ] eqn:Hpath; [ | discriminate].
    destruct (nth_error inner (positionOffset position)) as [i | ] eqn:Hi; [ | discriminate].
    rewrite (function_path_types _ _ _ _ _ _ Hstore Hvalid Hf) in Hdecode.
    destruct (pathHeights _ _ _ _) as [heights | ] eqn:Hheights; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite type_heights_i32, operand_types_i32 in Hdecode.
    destruct (memory stackId _) as [reference | ] eqn:Href; [ | discriminate].
    destruct (functionAt program (Z.to_nat (reference - 1))) as [g | ] eqn:Hg; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    rewrite firstn_repeat_le in Hdecode by lia.
    rewrite read_integer_operands_i32 in Hdecode by lia.
    destruct (readOperands _ _ _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    rewrite (read_locals_i32 _ _ _ _ _ Hf) in Hdecode.
    destruct (readStack memory (boundaryLocalsBase boundary) (frameLocals f)) as [locals | ] eqn:Hlocals; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite split_segments_map in Hdecode. cbn [fst snd] in Hdecode.
    rewrite value_stack_prefix_i32, wrap_value_labels_i32 in Hdecode.
    injection Hdecode as <- <-.
    assert (Htop : (snd heights <= totalHeight heights)%nat)
      by (destruct heights as [hs top]; rewrite total_height_sum; cbn [snd]; lia).
    destruct (frame_of_reads program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 1%nat
      0%nat 0 _ _ _ _ _ _ Hf Hpath Hheights ltac:(assumption) ltac:(assumption) ltac:(assumption) ltac:(lia)
      Hstack Hlocals) as [Hframe [Hinner Hoperands]].
    destruct (proj1 (read_operands_cells _ _ _ _ _ _ (Nat.le_0_l _)) Hstack) as [Hlen _].
    exists (resolved (frameView position (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals) (Z.to_nat (reference - 1))).
    split; [reflexivity | ]. exists f, g, i. cbn [viewPosition viewLocals viewValues].
    split; [exact Hframe | ]. rewrite Hinner.
    split; [exact Hi | ]. split; [assumption | ]. split; [exact Hg | ].
    split; [apply (proj1 (calls_into_check _ _ _ _ _ Hg)); assumption | ].
    rewrite Hoperands, Hlen.
    split; [rewrite Href; f_equal; lia | ].
    split; [lia | ]. split; [assumption | ]. split; [assumption | ]. split; [assumption | ].
    split; [assumption | reflexivity].
  - destruct (functionAt program function) as [f | ] eqn:Hf; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    repeat match goal with H : initializationFlagsb _ = true |- _ => apply flags_check in H end.
    destruct (readStack _ _ _) as [arguments | ] eqn:Hstack; [ | discriminate].
    injection Hdecode as <- <-.
    destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen [Hbounds _]].
    exists (entering function arguments). split; [reflexivity | ]. exists f.
    split; [exact Hf | ]. split; [exact Hlen | ]. split; [exact Hbounds | ]. split; [exact Hobligations | ].
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
    assert (HremainingBoundOps : (Z.to_nat (boundaryMoveRemaining boundary) <= length (viewOperands view))%nat).
    { unfold viewOperands. rewrite length_app. lia. }
    destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe HremainingBoundOps) as [Hstack Hlocals].
    rewrite Hf, Hpath, (function_path_types _ _ _ _ _ _ Hstore Hvalid Hf), Hheights.
    cbn [option_map]. rewrite type_heights_i32, operand_types_i32, Htotal.
    cbn [snd]. rewrite !Nat.add_0_r.
    apply flags_check in Hflags. pass_check.
    rewrite read_integer_operands_i32 by exact HremainingBoundOps.
    rewrite Hstack, Hlocals. cbn [option_map]. rewrite split_segments_map. cbn [fst snd].
    rewrite value_stack_config_i32, wrap_value_labels_i32.
    unfold viewOperands. rewrite split_concat.
    eexists. split; [reflexivity | exact I].
  - destruct Hstate as [f [g [i [Hframe [Hi [Hind [Hg [Hcalls [Href [Hsp [Hpc [Hflags [Hzero [Hmove ->]]]]]]]]]]]]]].
    pose proof Hframe as [Hf [Hpath [_ [Hfbp [Hbase _]]]]].
    destruct (frame_heights _ _ _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
    destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe (Nat.le_0_l _)) as [Hstack Hlocals].
    unfold innerCode in Hi. rewrite Hpath in Hi.
    rewrite Hf, Hpath, Hi, (function_path_types _ _ _ _ _ _ Hstore Hvalid Hf), Hheights.
    cbn [option_map]. rewrite type_heights_i32, operand_types_i32, Htotal, Nat.add_sub, Href.
    replace (Z.to_nat (Z.of_nat callee + 1 - 1)) with callee by lia. rewrite Hg.
    pose proof (proj2 (calls_into_check _ _ _ _ _ Hg) Hcalls) as Hcallsb.
    cbn [snd]. apply flags_check in Hflags. pass_check.
    rewrite firstn_repeat_le by lia.
    rewrite read_integer_operands_i32 by lia.
    rewrite Hstack, Hlocals. cbn [option_map]. rewrite split_segments_map. cbn [fst snd].
    rewrite value_stack_prefix_i32, wrap_value_labels_i32.
    unfold viewOperands. rewrite split_concat.
    eexists. split; [ | exact I]. unfold frameConfig, innerCode. rewrite Hpath. reflexivity.
  - destruct Hstate as [f [Hf [Hlen [Hbounds [Hinit [Hflags [Hmove [Hpending [Hpc ->]]]]]]]]].
    rewrite Hf. apply flags_check in Hflags. pass_check.
    assert (Hargs : readStack memory (boundaryLocalsBase boundary) (guestParams f) = Some arguments).
    { apply read_stack_cells. split; [exact Hlen | split; [exact Hbounds | ]].
      intros index value Hi. exact (argument_cells _ _ _ _ _ Hinit index value Hi). }
    rewrite Hargs. eexists. split; [reflexivity | exact Hinit].
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
    rewrite (function_path_types _ _ _ _ _ _ Hstore Hvalid Hg) in Hdecode.
    destruct (pathHeights _ _ _ _) as [heights | ] eqn:Hheights; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite type_heights_i32, operand_types_i32 in Hdecode.
    destruct (memory callStackReturnPcId _) as [returnPC | ] eqn:Hret; [ | discriminate].
    destruct (memory callStackCallerFbpId _) as [savedFbp | ] eqn:HsavedFbp; [ | discriminate].
    destruct (memory callStackCallerSpBaseId _) as [savedBase | ] eqn:HsavedBase; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    rewrite firstn_repeat_le in Hdecode by lia.
    rewrite read_integer_cells_i32 in Hdecode.
    fold (readStack memory savedBase (totalHeight heights - callOperands i f)) in Hdecode.
    destruct (readStack _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    rewrite (read_locals_i32 _ _ _ _ _ Hg) in Hdecode.
    destruct (readStack memory savedFbp (frameLocals g)) as [locals | ] eqn:Hlocals; [ | discriminate].
    cbn [option_map] in Hdecode. rewrite split_segments_map in Hdecode. cbn [fst snd] in Hdecode.
    rewrite value_stack_prefix_i32, wrap_value_labels_i32 in Hdecode.
    destruct (IH _ _ _ _ _ _ _ _ Hstore Hvalid Hg Hdecode) as [Hresult [stack' [Hsites [Hsusp ->]]]].
    split; [exact Hresult | ].
    rewrite <- (read_operands_unmoved memory savedBase _ 0) in Hstack.
    destruct (frame_of_reads program memory savedBase savedFbp (callOperands i f) 0%nat 0 _ _ _ _ _ _ Hg Hpath Hheights
      ltac:(lia) ltac:(lia) ltac:(lia) ltac:(lia) Hstack Hlocals) as [Hframe [Hinner Hoperands]].
    destruct (proj1 (read_operands_cells memory savedBase (totalHeight heights - callOperands i f) 0 0 stack
      (Nat.le_0_l _)) Hstack) as [Hlen _].
    exists (suspendedCall (frameView site (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals) g (guestResults f) savedBase savedFbp :: stack').
    split; [cbn; rewrite Hsites; reflexivity | ].
    split; [ | cbn [plugCalls]; unfold callerConfig, frameConfig;
      cbn [callView callerFunction viewPosition viewSegments viewValues viewLocals callResults]; rewrite Hinner;
      reflexivity].
    exists i. cbn [callView callerFunction viewPosition callerBase callerFbp callResults].
    split; [exact Hframe | ]. split; [rewrite Hinner; exact Hcall | ].
    split; [apply (proj1 (calls_into_check _ _ _ _ _ Hf)); assumption | ]. split; [reflexivity | ].
    split; [rewrite Hoperands, Hlen; assumption | ].
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
  destruct (frame_reads _ _ _ _ _ _ _ _ _ Hframe ltac:(lia)) as [Hstack Hlocals].
  rewrite read_operands_unmoved in Hstack.
  rewrite Hg, Hf, Hpath, Hcall.
  pose proof (proj2 (calls_into_check _ _ _ _ _ Hf) Hcallee) as Hcallsb.
  rewrite (function_path_types _ _ _ _ _ _ Hstore Hvalid Hg), Hheights.
  cbn [option_map]. rewrite type_heights_i32, operand_types_i32, Hret, HsavedFbp, HsavedBase.
  rewrite Htotal, Nat.add_sub. cbn [fst snd]. pass_check.
  rewrite firstn_repeat_le by lia. rewrite read_integer_cells_i32.
  fold (readStack memory (callerBase top) (length (viewOperands (callView top)))).
  rewrite Hstack, Hlocals. cbn [option_map]. rewrite split_segments_map. cbn [fst snd].
  rewrite value_stack_prefix_i32, wrap_value_labels_i32.
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
