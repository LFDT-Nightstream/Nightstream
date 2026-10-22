(* The decoder and the relational invariant agree: a decoded circuit state is a
   ControlState, and every ControlState decodes at its own shape. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog Add Locals StraightLineSpec StackSimulation
  SpecTecCalls SpecTecBranches CallerSpec ProgramSpec ProgramEncoding CodeLayout FrameInitialization
  Fragment ControlDecode ControlInvariant FrameFacts.
Import ListNotations.
Open Scope Z_scope.

Definition siteOf (position : ControlPosition) : ActiveSite :=
  match position with atCode view => siteCode (viewPosition view) | entering function _ => siteEntering function end.

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

Lemma idle_check : forall boundary, initializationIdleb boundary = true <-> initializationIdle boundary.
Proof.
  intros boundary. unfold initializationIdleb, initializationIdle. rewrite !andb_true_iff, !Z.eqb_eq. tauto.
Qed.

Lemma flags_check : forall boundary, initializationFlagsb boundary = true <-> initializationFlags boundary.
Proof.
  intros boundary. unfold initializationFlagsb, initializationFlags. rewrite andb_true_iff, !Z.eqb_eq. tauto.
Qed.

Lemma i32_cell_some : forall cell value, i32Cell cell = Some value <-> cell = Some value /\ 0 <= value < i32Modulus.
Proof.
  intros [v | ] value; cbn; [ | split; [discriminate | intros [H _]; discriminate]].
  destruct ((0 <=? v) && (v <? i32Modulus)) eqn:Hb.
  - apply andb_true_iff in Hb as [H0 H1]. apply Z.leb_le in H0. apply Z.ltb_lt in H1.
    split; [intros H; injection H as <-; auto | intros [H _]; injection H as <-; reflexivity].
  - split; [discriminate | ]. intros [Hv Hr]. injection Hv as <-.
    apply andb_false_iff in Hb as [Hb | Hb]; [apply Z.leb_gt in Hb | apply Z.ltb_ge in Hb]; lia.
Qed.

Lemma read_cells_some : forall cell count first values,
  readCells cell first count = Some values <->
  length values = count /\ Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value -> cell (first + Z.of_nat index) = Some value.
Proof.
  intros cell count. induction count as [ | count IH]; intros first values; cbn [readCells].
  - split.
    + intros H. injection H as <-. split; [reflexivity | split; [constructor | intros [ | ] ? H; discriminate]].
    + intros [H _]. destruct values; [reflexivity | discriminate].
  - split.
    + destruct (i32Cell (cell first)) as [value | ] eqn:Hcell; [ | discriminate].
      destruct (readCells cell (first + 1) count) as [rest | ] eqn:Hrest; [ | discriminate].
      intros H. injection H as <-. apply i32_cell_some in Hcell as [Hcell Hrange].
      destruct (proj1 (IH _ _) Hrest) as [Hlen [Hbounds Hcells]].
      split; [cbn; lia | split; [constructor; assumption | ]].
      intros [ | index] v Hv; cbn in Hv.
      * injection Hv as <-. rewrite Z.add_0_r. exact Hcell.
      * replace (first + Z.of_nat (S index)) with (first + 1 + Z.of_nat index) by lia. apply Hcells. exact Hv.
    + intros [Hlen [Hbounds Hcells]]. destruct values as [ | value rest]; [discriminate | ].
      inversion Hbounds as [ | ? ? Hrange Hrest]; subst.
      assert (Hfirst : cell first = Some value)
        by (rewrite <- (Z.add_0_r first); exact (Hcells 0%nat value eq_refl)).
      rewrite (proj2 (i32_cell_some _ _) (conj Hfirst Hrange)).
      rewrite (proj2 (IH (first + 1) rest)); [reflexivity | ].
      split; [cbn in Hlen; lia | split; [exact Hrest | ]].
      intros index v Hv. replace (first + 1 + Z.of_nat index) with (first + Z.of_nat (S index)) by lia.
      apply Hcells. exact Hv.
Qed.

Lemma read_cells_total : forall cell count first,
  (forall index, (index < count)%nat -> exists value,
    cell (first + Z.of_nat index) = Some value /\ 0 <= value < i32Modulus) ->
  exists values, readCells cell first count = Some values.
Proof.
  intros cell count. induction count as [ | count IH]; intros first Hcells; [exists []; reflexivity | ].
  destruct (Hcells 0%nat ltac:(lia)) as [value [Hvalue Hrange]]. rewrite Z.add_0_r in Hvalue.
  destruct (IH (first + 1)) as [rest Hrest].
  { intros index Hindex. destruct (Hcells (S index) ltac:(lia)) as [v Hv].
    exists v. replace (first + 1 + Z.of_nat index) with (first + Z.of_nat (S index)) by lia. exact Hv. }
  exists (value :: rest). cbn [readCells].
  rewrite (proj2 (i32_cell_some _ _) (conj Hvalue Hrange)), Hrest. reflexivity.
Qed.

Lemma read_stack_cells : forall memory first count values,
  readStack memory first count = Some values <->
  length values = count /\ Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value -> memory stackId [2 * (first + Z.of_nat index)] = Some value.
Proof. intros. exact (read_cells_some _ _ _ _). Qed.

Lemma read_locals_cells : forall memory fbp count locals,
  readLocals memory fbp count = Some locals <->
  length locals = count /\ Forall (fun value => 0 <= value < i32Modulus) locals /\
  forall index value, nth_error locals index = Some value -> memory localLoId [fbp; Z.of_nat index] = Some value.
Proof.
  intros memory fbp count locals. unfold readLocals. rewrite read_cells_some.
  split; intros [Hlen [Hb Hc]]; (split; [exact Hlen | split; [exact Hb | ]]);
    intros index value H; [rewrite <- (Z.add_0_l (Z.of_nat index)) | rewrite Z.add_0_l]; exact (Hc _ _ H).
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

Lemma frame_reads : forall program memory base fbp pending f view,
  FrameAt program memory base fbp pending f view ->
  readStack memory base (length (viewOperands view)) = Some (viewOperands view) /\
  readLocals memory fbp (frameLocals f) = Some (viewLocals view).
Proof.
  intros program memory base fbp pending f view [_ [_ [_ [_ [Hbounds [Hcells [Hlen [Hlb Hlc]]]]]]]].
  split; [apply read_stack_cells; auto | apply read_locals_cells; auto].
Qed.

Lemma frame_of_reads : forall program memory base fbp pending f position inner heights stack locals,
  functionAt program (positionFunction position) = Some f ->
  pathCode (guestBody f) (positionLabels position) = Some inner ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels position)
    (positionOffset position) = Some heights ->
  0 <= base -> (pending <= snd heights)%nat ->
  readStack memory base (totalHeight heights - pending) = Some stack ->
  readLocals memory fbp (frameLocals f) = Some locals ->
  let view := frameView position (fst (splitSegments (fst heights) stack))
    (snd (splitSegments (fst heights) stack)) locals in
  FrameAt program memory base fbp pending f view /\ innerCode f position = inner /\
  viewOperands view = stack.
Proof.
  intros program memory base fbp pending f position inner [hs top] stack locals Hf Hpath Hheights Hbase
    Hpending Hstack Hlocals view. cbn [fst snd] in *.
  destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen [Hbounds Hcells]].
  destruct (proj1 (read_locals_cells _ _ _ _) Hlocals) as [Hllen [Hlb Hlc]].
  destruct (split_stack hs top pending stack Hpending Hlen) as [Hmap [Hvalues Hconcat]].
  assert (Hinner : innerCode f position = inner) by (unfold innerCode; rewrite Hpath; reflexivity).
  assert (Hoperands : viewOperands view = stack) by exact Hconcat.
  split; [ | split; [exact Hinner | exact Hoperands]].
  unfold FrameAt, view. cbn [viewPosition viewSegments viewValues viewLocals].
  split; [exact Hf | ]. split; [rewrite Hinner; exact Hpath | ].
  split; [rewrite Hmap, Hvalues; exact Hheights | ]. split; [exact Hbase | ].
  fold view. rewrite Hoperands. split; [exact Hbounds | ]. split; [exact Hcells | ].
  split; [exact Hllen | ]. split; [exact Hlb | exact Hlc].
Qed.

Lemma frame_heights : forall program memory base fbp pending f view,
  FrameAt program memory base fbp pending f view ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels (viewPosition view))
    (positionOffset (viewPosition view)) = Some (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) /\
  totalHeight (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) =
    (length (viewOperands view) + pending)%nat.
Proof.
  intros program memory base fbp pending f view Hframe. split; [exact (proj1 (proj2 (proj2 Hframe))) | ].
  apply total_height_operands.
Qed.

Lemma decode_active_function : forall program s memory boundary site r,
  decodeActive program s memory boundary site = Some r -> exists f, functionAt program (siteFunction site) = Some f.
Proof.
  intros program s memory boundary [position | function] r H; cbn [siteFunction];
    unfold decodeActive in H; destruct (functionAt program _) as [f | ]; eauto; discriminate.
Qed.

Lemma active_function_site : forall program position,
  activeFunction program position = functionAt program (siteFunction (siteOf position)).
Proof. intros program [view | function arguments]; reflexivity. Qed.

(* Decoded active frames satisfy the invariant. *)
Lemma decoded_active : forall program s memory boundary site c obligations,
  decodeActive program s memory boundary site = Some (c, obligations) -> obligations ->
  exists position, siteOf position = site /\ ActiveState program s memory boundary position c.
Proof.
  intros program s memory boundary [position | function] c obligations Hdecode Hobligations;
    unfold decodeActive in Hdecode; cbv zeta in Hdecode.
  - destruct (functionAt program (positionFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (pathCode (guestBody f) (positionLabels position)) as [inner | ] eqn:Hpath; [ | discriminate].
    destruct (pathHeights _ _ _ _) as [heights | ] eqn:Hheights; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools. apply idle_check in H.
    destruct (readStack _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals _ _ _) as [locals | ] eqn:Hlocals; [ | discriminate].
    injection Hdecode as <- <-.
    rewrite <- (Nat.sub_0_r (totalHeight heights)) in Hstack.
    destruct (frame_of_reads program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0%nat
      _ _ _ _ _ _ Hf Hpath Hheights ltac:(lia) ltac:(lia) Hstack Hlocals)
      as [Hframe [Hinner Hoperands]].
    exists (atCode (frameView position (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals)).
    split; [reflexivity | ]. exists f. cbn [viewPosition viewLocals viewValues].
    split; [exact Hframe | ]. rewrite Hinner. split; [assumption | ].
    split; [rewrite Hoperands; destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen _];
      rewrite Hlen, Nat.sub_0_r; assumption | ].
    split; [assumption | ]. split; [exact H | reflexivity].
  - destruct (functionAt program function) as [f | ] eqn:Hf; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools. apply flags_check in H.
    destruct (readStack _ _ _) as [arguments | ] eqn:Hstack; [ | discriminate].
    injection Hdecode as <- <-.
    destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen [Hbounds _]].
    exists (entering function arguments). split; [reflexivity | ]. exists f.
    split; [exact Hf | ]. split; [exact Hlen | ]. split; [exact Hbounds | ]. split; [exact Hobligations | ].
    split; [exact H | ]. split; [lia | ]. split; [assumption | reflexivity].
Qed.

(* Active frames of the invariant decode at their own site. *)
Lemma active_decodes : forall program s position memory boundary c,
  ActiveState program s memory boundary position c ->
  exists obligations, decodeActive program s memory boundary (siteOf position) = Some (c, obligations) /\ obligations.
Proof.
  intros program s [view | function arguments] memory boundary c Hstate; cbn [siteOf]; unfold decodeActive; cbv zeta.
  - destruct Hstate as [f [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
    pose proof Hframe as [Hf [Hpath _]]. destruct (frame_heights _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
    destruct (frame_reads _ _ _ _ _ _ _ Hframe) as [Hstack Hlocals].
    rewrite Hf, Hpath, Hheights, Htotal, Nat.add_0_r.
    pose proof (proj1 (proj2 (proj2 (proj2 Hframe)))) as Hbase.
    apply idle_check in Hidle. pass_check.
    rewrite Hstack, Hlocals. cbn [fst snd]. unfold viewOperands. rewrite split_concat.
    eexists. split; [reflexivity | exact I].
  - destruct Hstate as [f [Hf [Hlen [Hbounds [Hinit [Hflags [Hpending [Hpc ->]]]]]]]].
    rewrite Hf. apply flags_check in Hflags. pass_check.
    assert (Hargs : readStack memory (boundaryStackBase boundary) (guestParams f) = Some arguments).
    { apply read_stack_cells. split; [exact Hlen | split; [exact Hbounds | ]].
      intros index value Hi. exact (argument_cells _ _ _ _ _ _ Hinit index value Hi). }
    rewrite Hargs. eexists. split; [reflexivity | exact Hinit].
Qed.

Lemma function_index_unique : forall program index index' f,
  functionAt program index = Some f -> functionAt program index' = Some f -> index = index'.
Proof.
  intros program index index' f H H'.
  destruct (function_at_sound _ _ _ H) as [_ [Ha _]]. destruct (function_at_sound _ _ _ H') as [_ [Ha' _]].
  rewrite Ha in Ha'. lia.
Qed.

Lemma call_at_some : forall code offset x, callAt code offset = Some x <-> nth_error code offset = Some (CALL x).
Proof.
  intros code offset x. unfold callAt. split; [ | intros ->; reflexivity].
  destruct (nth_error code offset) as [i | ]; [ | discriminate].
  destruct i; try discriminate. intros H. injection H as ->. reflexivity.
Qed.

(* Decoded callers form the invariant's suspended stack. *)
Lemma decoded_calls : forall program memory sites callee f base fbp depth inner obligations c result,
  functionAt program callee = Some f ->
  decodeCalls program memory callee base fbp depth sites inner obligations = Some (c, result) ->
  result = obligations /\ exists stack, map (fun suspended => viewPosition (callView suspended)) stack = sites /\
    SuspendedStack program memory f base fbp depth stack /\ c = plugCalls stack inner.
Proof.
  intros program memory sites. induction sites as [ | site sites IH];
    intros callee f base fbp depth inner obligations c result Hf Hdecode; cbn [decodeCalls] in Hdecode.
  - injection Hdecode as <- <-. split; [reflexivity | exists []; repeat split].
  - destruct (functionAt program (positionFunction site)) as [g | ] eqn:Hg; [ | discriminate].
    rewrite Hf in Hdecode.
    destruct (pathCode (guestBody g) (positionLabels site)) as [code | ] eqn:Hpath; [ | discriminate].
    destruct (callAt code (positionOffset site)) as [x | ] eqn:Hcall; [ | discriminate].
    apply call_at_some in Hcall.
    destruct (nth_error (FUNCS (guestModule g)) _) as [address | ] eqn:Haddress; [ | discriminate].
    destruct (pathHeights _ _ _ _) as [heights | ] eqn:Hheights; [ | discriminate].
    destruct (memory callStackReturnPcId _) as [returnPC | ] eqn:Hret; [ | discriminate].
    destruct (memory callStackCallerFbpId _) as [savedFbp | ] eqn:HsavedFbp; [ | discriminate].
    destruct (memory callStackCallerSpBaseId _) as [savedBase | ] eqn:HsavedBase; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    destruct (readStack _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals _ _ _) as [locals | ] eqn:Hlocals; [ | discriminate].
    destruct (IH _ _ _ _ _ _ _ _ _ Hg Hdecode) as [Hresult [stack' [Hsites [Hsusp ->]]]].
    split; [exact Hresult | ].
    destruct (frame_of_reads program memory savedBase savedFbp (guestParams f) _ _ _ _ _ _ Hg Hpath Hheights
      ltac:(lia) ltac:(lia) Hstack Hlocals) as [Hframe [Hinner Hoperands]].
    destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen _].
    exists (suspendedCall (frameView site (fst (splitSegments (fst heights) stack))
      (snd (splitSegments (fst heights) stack)) locals) g (guestResults f) savedBase savedFbp :: stack').
    split; [cbn; rewrite Hsites; reflexivity | ].
    split; [ | cbn [plugCalls]; unfold callerConfig, frameConfig;
      cbn [callView callerFunction viewPosition viewSegments viewValues viewLocals callResults]; rewrite Hinner;
      reflexivity].
    exists x. cbn [callView callerFunction viewPosition callerBase callerFbp callResults].
    split; [exact Hframe | ]. split; [rewrite Hinner; exact Hcall | ].
    split; [unfold programCallees; rewrite Haddress; subst callee; exact Hf | ]. split; [reflexivity | ].
    split; [rewrite Hoperands, Hlen; assumption | ]. split; [assumption | ].
    split; [rewrite Hret; f_equal; assumption | ].
    split; [exact HsavedFbp | ]. split; [exact HsavedBase | exact Hsusp].
Qed.

(* The invariant's suspended stack decodes at its own call sites. *)
Lemma calls_decode : forall program memory stack callee f base fbp depth inner obligations,
  functionAt program callee = Some f -> SuspendedStack program memory f base fbp depth stack ->
  decodeCalls program memory callee base fbp depth
    (map (fun suspended => viewPosition (callView suspended)) stack) inner obligations =
    Some (plugCalls stack inner, obligations).
Proof.
  intros program memory stack. induction stack as [ | top stack IH];
    intros callee f base fbp depth inner obligations Hf Hsusp; cbn [map decodeCalls plugCalls]; [reflexivity | ].
  destruct Hsusp as [x [Hframe [Hcall [Hcallee [Hres [Hbase [Hfbp [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]]].
  pose proof Hframe as [Hg [Hpath _]]. destruct (frame_heights _ _ _ _ _ _ _ Hframe) as [Hheights Htotal].
  destruct (frame_reads _ _ _ _ _ _ _ Hframe) as [Hstack Hlocals].
  rewrite Hg, Hf, Hpath, (proj2 (call_at_some _ _ _) Hcall).
  unfold programCallees in Hcallee.
  destruct (nth_error (FUNCS (guestModule (callerFunction top))) (N.to_nat (x :> N))) as [address | ];
    [ | discriminate].
  pose proof (function_index_unique _ _ _ _ Hcallee Hf) as Haddress.
  rewrite Hheights, Hret, HsavedFbp, HsavedBase.
  pose proof (proj1 (proj2 (proj2 (proj2 Hframe)))) as Hbase0.
  rewrite Htotal, Nat.add_sub. cbn [fst snd]. pass_check.
  rewrite Hstack, Hlocals.
  unfold viewOperands. rewrite split_concat. cbn [fst snd].
  rewrite <- Hres. exact (IH _ _ _ _ _ _ _ Hg Hrest).
Qed.

(* The bridge: what the audited statement says, in the invariant's terms. *)
Theorem circuit_config_state : forall program s baseDepth shape memory boundary c,
  CircuitConfig program s baseDepth shape memory boundary c ->
  exists cursor, ControlState program s baseDepth cursor memory boundary c.
Proof.
  intros program s baseDepth shape memory boundary c [_ [_ [_ [obligations [Hdecode Hobligations]]]]].
  unfold decodeControl in Hdecode.
  destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
  destruct (decodeActive program s memory boundary (activeSite shape)) as [[inner active] | ] eqn:Hactive;
    [ | discriminate]. cbn [fst snd] in Hdecode.
  destruct (decode_active_function _ _ _ _ _ _ Hactive) as [f Hf].
  destruct (decoded_calls _ _ _ _ _ _ _ _ _ _ _ _ Hf Hdecode) as [Hresult [stack [Hsites [Hsusp ->]]]].
  subst obligations.
  destruct (decoded_active _ _ _ _ _ _ _ Hactive Hobligations) as [position [Hsite Hstate]].
  exists (controlCursor stack position). exists f, inner. cbn [activePosition suspendedCalls].
  split; [rewrite active_function_site, Hsite; exact Hf | ]. split; [exact Hstate | ].
  split; [exact Hsusp | ]. split; [assumption | ].
  split; [rewrite <- Hsites, length_map in *; assumption | reflexivity].
Qed.

Theorem control_state_circuit : forall program s baseDepth cursor memory boundary c,
  store_FUNCS s = program -> wf_store s -> programValidated program s ->
  ControlState program s baseDepth cursor memory boundary c ->
  CircuitConfig program s baseDepth (shapeOf cursor) memory boundary c.
Proof.
  intros program s baseDepth [stack position] memory boundary c Hfuncs Hwf Hvalid
    [f [inner [Hf [Hstate [Hsusp [Hbase [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in *.
  split; [exact Hfuncs | split; [exact Hwf | split; [exact Hvalid | ]]].
  unfold decodeControl, shapeOf. cbn [callSites activeSite activePosition suspendedCalls]. rewrite length_map.
  pass_check.
  destruct (active_decodes _ _ _ _ _ _ Hstate) as [active [Hactive Hobligations]].
  rewrite Hactive. cbn [fst snd]. rewrite active_function_site in Hf.
  exists active. split; [exact (calls_decode _ _ _ _ _ _ _ _ _ _ Hf Hsusp) | exact Hobligations].
Qed.
