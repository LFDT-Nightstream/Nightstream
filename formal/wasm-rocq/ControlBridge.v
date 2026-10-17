(* The decoder and the relational invariant agree: a decoded circuit state is a
   ControlState, and every ControlState decodes at its own shape. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog Add Locals StraightLineSpec StackSimulation
  SpecTecCalls SpecTecBranches CallerSpec ProgramSpec ProgramEncoding CodeLayout BlockSpec FrameInitialization
  ControlDecode ControlInvariant ControlState.
Import ListNotations.
Open Scope Z_scope.

Definition siteOf (position : ControlPosition) : ActiveSite :=
  match position with
  | atInstruction position => siteInstruction position
  | inBlock position offset prefix => siteBlock position offset (length prefix)
  | entering function _ => siteEntering function
  end.

Definition shapeOf (cursor : ControlCursor) : ControlShape :=
  controlShape (map callPosition (suspendedCalls cursor)) (siteOf (activePosition cursor)).

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

Lemma stack_matches_read : forall memory boundary values,
  stackMatches memory boundary values <->
  0 <= boundaryStackBase boundary /\ boundaryStackBase boundary <= boundarySP boundary /\
  readStack memory (boundaryStackBase boundary) (Z.to_nat (boundarySP boundary - boundaryStackBase boundary)) =
    Some values.
Proof.
  intros memory boundary values. split.
  - intros [H0 [Hsp [Hb Hc]]]. split; [exact H0 | split; [lia | ]].
    replace (Z.to_nat (boundarySP boundary - boundaryStackBase boundary)) with (length values) by lia.
    apply read_stack_cells. auto.
  - intros [H0 [Hle Hr]]. apply read_stack_cells in Hr as [Hlen [Hb Hc]].
    split; [exact H0 | split; [lia | auto]].
Qed.

Lemma read_locals_frame : forall memory fbp count locals s module,
  readLocals memory fbp count = Some locals ->
  length locals = count /\ Forall (fun value => 0 <= value < i32Modulus) locals /\
  localsAt memory fbp (mk_state s (MKframe (map i32Value locals) module)).
Proof.
  intros memory fbp count locals s module H. apply read_cells_some in H as [Hlen [Hb Hc]].
  split; [exact Hlen | split; [exact Hb | ]].
  intros index v Hv. cbn [stateLocals LOCALS] in Hv. rewrite nth_error_map in Hv.
  destruct (nth_error locals index) as [value | ] eqn:Hi; cbn in Hv; [injection Hv as <- | discriminate].
  exists value. split; [reflexivity | ].
  split; [exact (proj1 (Forall_forall _ _) Hb _ (nth_error_In _ _ Hi)) | ].
  rewrite <- (Z.add_0_l (Z.of_nat index)). exact (Hc _ _ Hi).
Qed.

Lemma locals_read : forall memory fbp z, localsAt memory fbp z ->
  exists locals, readLocals memory fbp (length (stateLocals z)) = Some locals /\
    stateLocals z = map i32Value locals.
Proof.
  intros memory fbp z Hlocals.
  destruct (read_cells_total (fun index => memory localLoId [fbp; index]) (length (stateLocals z)) 0)
    as [locals Hread].
  { intros index Hindex. destruct (nth_error (stateLocals z) index) as [v | ] eqn:Hv;
      [ | apply nth_error_None in Hv; lia].
    destruct (Hlocals index v Hv) as [value [_ [Hrange Hcell]]]. exists value. split; [exact Hcell | exact Hrange]. }
  exists locals. split; [exact Hread | ].
  apply read_cells_some in Hread as [Hlen [_ Hc]].
  apply nth_error_ext. intros index. rewrite nth_error_map.
  destruct (nth_error (stateLocals z) index) as [v | ] eqn:Hv.
  - destruct (Hlocals index v Hv) as [value [-> [_ Hcell]]].
    destruct (nth_error locals index) as [w | ] eqn:Hw.
    + pose proof (Hc _ _ Hw) as Hw'. cbn beta in Hw'. rewrite Z.add_0_l, Hcell in Hw'.
      injection Hw' as ->. reflexivity.
    + apply nth_error_None in Hw. assert (index < length (stateLocals z))%nat
        by (apply nth_error_Some; rewrite Hv; discriminate). lia.
  - apply nth_error_None in Hv. replace (nth_error locals index) with (@None Z); [reflexivity | ].
    symmetry. apply nth_error_None. lia.
Qed.

Lemma function_module_wf : forall program s index f,
  store_FUNCS s = program -> wf_store s -> functionAt program index = Some f -> wf_moduleinst (guestModule f).
Proof.
  intros program s index f Hfuncs Hwf Hf.
  destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  inversion Hwf as [funcs ? ? ? ? ? Hall]; subst. cbn in Hnth.
  pose proof (proj1 (Forall_forall _ _) Hall _ (nth_error_In _ _ Hnth)) as Hfi.
  inversion Hfi; assumption.
Qed.

Lemma active_frame_wf : forall program s index f locals,
  store_FUNCS s = program -> wf_store s -> functionAt program index = Some f ->
  Forall (fun value => 0 <= value < i32Modulus) locals -> wf_state (mk_state s (activeFrame f locals)).
Proof.
  intros program s index f locals Hfuncs Hwf Hf Hlocals. constructor; [exact Hwf | constructor].
  - apply Forall_map. eapply Forall_impl; [ | exact Hlocals].
    intros value Hvalue. apply val_case_0. apply spectec_i32_well_formed. exact Hvalue.
  - eapply function_module_wf; eassumption.
Qed.

Lemma active_frame_matches : forall program s f locals,
  store_FUNCS s = program -> length locals = frameLocals f ->
  frameMatches program f (mk_state s (activeFrame f locals)).
Proof.
  intros program s f locals Hfuncs Hlen. split; [exact Hfuncs | split; [reflexivity | ]].
  cbn. rewrite length_map. exact Hlen.
Qed.

(* A frame the invariant describes is the one the decoder rebuilds. *)
Lemma state_decodes : forall program f z memory fbp,
  frameMatches program f z -> localsAt memory fbp z ->
  exists locals, readLocals memory fbp (frameLocals f) = Some locals /\
    z = mk_state (fun_store z) (activeFrame f locals).
Proof.
  intros program f z memory fbp [_ [Hmodule Hlen]] Hlocals.
  destruct (locals_read _ _ _ Hlocals) as [locals [Hread Hz]].
  exists locals. split; [unfold frameLocals; rewrite <- Hlen; exact Hread | ].
  destruct z as [s [values module]]. cbn in Hmodule, Hz |- *. unfold activeFrame. congruence.
Qed.

Lemma block_at_some : forall f offset body,
  blockAt f offset = Some body <-> nth_error (guestBody f) offset = Some (BLOCK (_RESULT None) body).
Proof.
  intros f offset body. unfold blockAt. split; [ | intros ->; reflexivity].
  destruct (nth_error (guestBody f) offset) as [i | ]; [ | discriminate].
  destruct i; try discriminate.
  match goal with |- context [BLOCK ?bt _] => destruct bt as [[t | ] | x]; try discriminate end.
  intros H. injection H as ->. reflexivity.
Qed.

Lemma call_at_some : forall f offset x,
  callAt f offset = Some x <-> nth_error (guestBody f) offset = Some (CALL x).
Proof.
  intros f offset x. unfold callAt. split; [ | intros ->; reflexivity].
  destruct (nth_error (guestBody f) offset) as [i | ]; [ | discriminate].
  destruct i; try discriminate. intros H. injection H as ->. reflexivity.
Qed.

Lemma nth_bound : forall (A : Type) (xs : list A) index x, nth_error xs index = Some x -> (index < length xs)%nat.
Proof. intros A xs index x H. apply nth_error_Some. rewrite H. discriminate. Qed.

Lemma decode_active_function : forall program s memory boundary site r,
  decodeActive program s memory boundary site = Some r -> exists f, functionAt program (siteFunction site) = Some f.
Proof.
  intros program s memory boundary [position | position offset height | function] r H; cbn [siteFunction];
    unfold decodeActive in H; destruct (functionAt program _) as [f | ]; eauto; discriminate.
Qed.

Lemma active_function_site : forall program position,
  activeFunction program position = functionAt program (siteFunction (siteOf position)).
Proof. intros program [p | p o prefix | function arguments]; reflexivity. Qed.

(* Decoded active frames satisfy the invariant. *)
Lemma decoded_active : forall program s memory boundary site c obligations,
  store_FUNCS s = program -> wf_store s ->
  decodeActive program s memory boundary site = Some (c, obligations) -> obligations ->
  exists position z, siteOf position = site /\ fun_store z = s /\ ActiveState program position z memory boundary c.
Proof.
  intros program s memory boundary [position | position offset height | function] c obligations
    Hfuncs Hwf Hdecode Hobligations; unfold decodeActive in Hdecode; cbv zeta in Hdecode.
  - destruct (functionAt program (sourceFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools. apply idle_check in H.
    destruct (readStack _ _ _) as [values | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals _ _ _) as [locals | ] eqn:Hlocals; [ | discriminate].
    injection Hdecode as <- <-.
    destruct (read_locals_frame _ _ _ _ s (guestModule f) Hlocals) as [Hlen [Hbounds Hat]].
    pose proof (active_frame_wf _ _ _ _ _ Hfuncs Hwf Hf Hbounds) as Hz.
    exists (atInstruction position), (mk_state s (activeFrame f locals)).
    split; [reflexivity | split; [reflexivity | split; [ | exact H]]].
    exists (skipn (sourceOffset position) (guestBody f)). unfold SourceState. rewrite Hf.
    split; [apply active_frame_matches; assumption | ].
    split; [split; [assumption | reflexivity] | ]. split; [exact Hobligations | ]. split.
    + exists values. split; [apply stack_matches_read; auto | ]. split; [reflexivity | ]. split; [ | exact Hat].
      apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj1 (read_stack_cells _ _ _ _) Hstack))) | ].
      eapply validated_instructions_wf. exact Hobligations.
    + unfold sourceAddress in *. assumption.
  - destruct (functionAt program (sourceFunction position)) as [f | ] eqn:Hf; [ | discriminate].
    destruct (blockAt f (sourceOffset position)) as [body | ] eqn:Hblock; [ | discriminate].
    apply block_at_some in Hblock.
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools. apply idle_check in H.
    destruct (readStack _ _ _) as [stack | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals _ _ _) as [locals | ] eqn:Hlocals; [ | discriminate].
    injection Hdecode as <- <-.
    destruct Hobligations as [Hcontinuation [Hsupport [Hrest Hbalanced]]].
    destruct (read_locals_frame _ _ _ _ s (guestModule f) Hlocals) as [Hlen [Hbounds Hat]].
    pose proof (active_frame_wf _ _ _ _ _ Hfuncs Hwf Hf Hbounds) as Hz.
    pose proof (proj1 (proj1 (read_stack_cells _ _ _ _) Hstack)) as Hheight.
    assert (Hprefix : length (firstn height stack) = height) by (rewrite length_firstn; lia).
    exists (inBlock position offset (firstn height stack)), (mk_state s (activeFrame f locals)).
    split; [cbn; rewrite Hprefix; reflexivity | split; [reflexivity | ]].
    exists f, body, (skipn (S (sourceOffset position)) (guestBody f)), (skipn height stack).
    split; [exact Hf | ]. split; [exact Hblock | ].
    split; [split; [pose proof (nth_bound _ _ _ _ Hblock); lia | reflexivity] | ].
    split; [apply active_frame_matches; assumption | ]. split; [rewrite Hprefix; exact Hcontinuation | ].
    split; [exact Hsupport | ]. split; [assumption | ]. split; [exact Hrest | ]. split; [exact Hbalanced | ].
    split; [exact Hz | ]. split; [rewrite firstn_skipn; apply stack_matches_read; auto | ].
    split; [exact Hat | ]. split; [unfold sourceAddress in *; assumption | ]. split; [exact H | reflexivity].
  - destruct (functionAt program function) as [f | ] eqn:Hf; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools. apply flags_check in H.
    destruct (readStack _ _ _) as [arguments | ] eqn:Hstack; [ | discriminate].
    injection Hdecode as <- <-. destruct Hobligations as [Hvalid Hinit].
    destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hlen [Hbounds _]].
    assert (Hz : wf_state (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f)))).
    { constructor; [exact Hwf | apply entered_frame_wf; [eapply function_module_wf; eassumption | exact Hbounds]]. }
    exists (entering function arguments), (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f))).
    split; [reflexivity | split; [reflexivity | ]]. exists f.
    split; [exact Hf | ]. split; [exact Hlen | ]. split; [exact Hbounds | ]. split; [reflexivity | ].
    split.
    { split; [exact Hfuncs | split; [reflexivity | ]].
      cbn [stateLocals enteredFrame LOCALS]. rewrite length_app, length_map, repeat_length. lia. }
    split; [exact Hz | ]. split; [exact Hvalid | ].
    split; [apply stack_config_wf; [exact Hz | constructor | eapply validated_instructions_wf; exact Hvalid] | ].
    split; [exact Hinit | ]. split; [exact H | ]. split; [lia | ]. split; [assumption | reflexivity].
Qed.

(* Active frames of the invariant decode at their own site. *)
Lemma active_decodes : forall program s position z memory boundary c,
  fun_store z = s -> ActiveState program position z memory boundary c ->
  exists obligations, decodeActive program s memory boundary (siteOf position) = Some (c, obligations) /\ obligations.
Proof.
  intros program s [position | position offset prefix | function arguments] z memory boundary c Hstore Hstate;
    cbn [siteOf]; unfold decodeActive; cbv zeta.
  - destruct Hstate as [[code Hsource] Hidle]. unfold SourceState in Hsource.
    destruct (functionAt program (sourceFunction position)) as [f | ] eqn:Hf; [ | contradiction].
    destruct Hsource as [Hframe [[Hbound Hcode] [Hvalid [[values [Hstack [-> [_ Hlocals]]]] Hpc]]]].
    destruct (state_decodes _ _ _ _ _ Hframe Hlocals) as [locals [Hread Hz]].
    rewrite Hstore in Hz. subst z code.
    pose proof (proj1 (stack_matches_read _ _ _) Hstack) as [Hbase [Hle Hvalues]].
    apply idle_check in Hidle. unfold sourceAddress. pass_check.
    rewrite Hvalues, Hread. eexists. split; [reflexivity | exact Hvalid].
  - destruct Hstate as [f [body [continuation [values [Hf [Hblock [[Hbound Hcont] [Hframe [Hvalid [Hsupport
      [Hoffset [Hrest [Hbalanced [_ [Hstack [Hlocals [Hpc [Hidle ->]]]]]]]]]]]]]]]]]].
    rewrite Hf, (proj2 (block_at_some _ _ _) Hblock).
    destruct (state_decodes _ _ _ _ _ Hframe Hlocals) as [locals [Hread Hz]].
    rewrite Hstore in Hz. subst z continuation.
    pose proof (stack_matches_height _ _ _ Hstack) as Hheight. unfold stackHeight in Hheight.
    rewrite length_app in Hheight.
    pose proof (proj1 (stack_matches_read _ _ _) Hstack) as [Hbase [Hle Hvalues]].
    apply idle_check in Hidle. unfold sourceAddress. pass_check.
    rewrite Hvalues, Hread, firstn_app, skipn_app, Nat.sub_diag, firstn_all, skipn_all. cbn [firstn skipn app].
    rewrite app_nil_r. eexists. split; [reflexivity | ].
    split; [exact Hvalid | split; [exact Hsupport | split; assumption]].
  - destruct Hstate as [f [Hf [Hlen [Hbounds [Hz [Hframe [_ [Hvalid [_ [Hinit [Hflags [Hpending [Hpc ->]]]]]]]]]]]]].
    rewrite Hf. rewrite Hstore in Hz. subst z.
    apply flags_check in Hflags. pass_check.
    assert (Hargs : readStack memory (boundaryStackBase boundary) (guestParams f) = Some arguments).
    { apply read_stack_cells. split; [exact Hlen | split; [exact Hbounds | ]].
      intros index value Hi. exact (argument_cells _ _ _ _ _ _ Hinit index value Hi). }
    rewrite Hargs. eexists. split; [reflexivity | split; assumption].
Qed.

Lemma function_index_unique : forall program index index' f,
  functionAt program index = Some f -> functionAt program index' = Some f -> index = index'.
Proof.
  intros program index index' f H H'.
  destruct (function_at_sound _ _ _ H) as [_ [Ha _]]. destruct (function_at_sound _ _ _ H') as [_ [Ha' _]].
  rewrite Ha in Ha'. lia.
Qed.

(* Decoded callers form the invariant's suspended stack. *)
Lemma decoded_calls : forall program s memory sites callee f base fbp depth inner obligations c result,
  store_FUNCS s = program -> wf_store s -> functionAt program callee = Some f ->
  decodeCalls program s memory callee base fbp depth sites inner obligations = Some (c, result) -> result ->
  obligations /\ exists stack, map callPosition stack = sites /\
    SuspendedStack program f base fbp depth s memory stack /\ c = plugCalls stack inner.
Proof.
  intros program s memory sites. induction sites as [ | site sites IH];
    intros callee f base fbp depth inner obligations c result Hfuncs Hwf Hf Hdecode Hresult; cbn [decodeCalls] in Hdecode.
  - injection Hdecode as <- <-. split; [exact Hresult | exists []; repeat split].
  - destruct (functionAt program (sourceFunction site)) as [g | ] eqn:Hg; [ | discriminate].
    rewrite Hf in Hdecode.
    destruct (callAt g (sourceOffset site)) as [x | ] eqn:Hcall; [ | discriminate]. apply call_at_some in Hcall.
    destruct (nth_error (FUNCS (guestModule g)) _) as [address | ] eqn:Haddress; [ | discriminate].
    destruct (memory callStackReturnPcId _) as [returnPC | ] eqn:Hret; [ | discriminate].
    destruct (memory callStackCallerFbpId _) as [savedFbp | ] eqn:HsavedFbp; [ | discriminate].
    destruct (memory callStackCallerSpBaseId _) as [savedBase | ] eqn:HsavedBase; [ | discriminate].
    destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
    destruct (readStack _ _ _) as [values | ] eqn:Hstack; [ | discriminate].
    destruct (readLocals _ _ _) as [locals | ] eqn:Hlocals; [ | discriminate].
    destruct (IH _ _ _ _ _ _ _ _ _ Hfuncs Hwf Hg Hdecode Hresult)
      as [[Hobligations [Hstraight Hvalid]] [stack [Hsites [Hsusp ->]]]].
    split; [exact Hobligations | ].
    destruct (read_locals_frame _ _ _ _ s (guestModule g) Hlocals) as [Hlen [Hbounds Hat]].
    destruct (proj1 (read_stack_cells _ _ _ _) Hstack) as [Hcount [Hvalues Hcells]].
    exists (suspendedCall site (activeFrame g locals) values savedBase savedFbp
      (skipn (S (sourceOffset site)) (guestBody g)) (guestResults f) :: stack).
    split; [cbn; rewrite Hsites; reflexivity | ]. split; [ | reflexivity].
    exists g, x. cbn [callPosition callerFrame callerValues callerBase callerFbp callContinuation callResults].
    split; [exact Hg | ]. split; [exact Hcall | ].
    split; [split; [pose proof (nth_bound _ _ _ _ Hcall); lia | reflexivity] | ].
    split; [unfold programCallees; rewrite Haddress; subst callee; exact Hf | ]. split; [reflexivity | ].
    split; [exact Hstraight | ]. split; [apply active_frame_matches; assumption | ].
    split; [pose proof (active_frame_wf _ _ _ _ _ Hfuncs Hwf Hg Hbounds) as Hz; inversion Hz; assumption | ].
    split; [exact Hvalid | ]. split; [exact Hvalues | ]. split; [assumption | ]. split; [lia | ].
    split; [exact Hcells | ]. split; [cbn; rewrite length_map, Hlen; unfold frameLocals in *; lia | ].
    split; [exact Hat | ]. split; [unfold sourceAddress in *; congruence | ].
    split; [exact HsavedFbp | ]. split; [exact HsavedBase | exact Hsusp].
Qed.

(* The invariant's suspended stack decodes at its own call sites. *)
Lemma calls_decode : forall program s memory stack callee f base fbp depth inner obligations,
  functionAt program callee = Some f -> SuspendedStack program f base fbp depth s memory stack -> obligations ->
  exists result, decodeCalls program s memory callee base fbp depth (map callPosition stack) inner obligations =
    Some (plugCalls stack inner, result) /\ result.
Proof.
  intros program s memory stack. induction stack as [ | top stack IH];
    intros callee f base fbp depth inner obligations Hf Hsusp Hobligations; cbn [map decodeCalls plugCalls].
  - eexists. split; [reflexivity | exact Hobligations].
  - destruct Hsusp as [g [x [Hg [Hcall [[Hbound Hcont] [Hcallee [Hres [Hstraight [Hframe [_ [Hvalid [Hvalues
      [Hbase0 [Hbase [Hcells [Hfbp [Hlocals [Hret [HsavedFbp [HsavedBase Hrest]]]]]]]]]]]]]]]]]]]].
    rewrite Hg, Hf, (proj2 (call_at_some _ _ _) Hcall).
    unfold programCallees in Hcallee.
    destruct (nth_error (FUNCS (guestModule g)) (N.to_nat (x :> N))) as [address | ]; [ | discriminate].
    pose proof (function_index_unique _ _ _ _ Hcallee Hf) as Haddress.
    rewrite Hret, HsavedFbp, HsavedBase.
    destruct (state_decodes _ _ _ _ _ Hframe Hlocals) as [locals [Hread Hz]].
    cbn [fun_store] in Hz. injection Hz as Hz.
    pose proof (proj2 (proj2 Hframe)) as Hlen. cbn [stateLocals] in Hlen.
    unfold sourceAddress. pass_check.
    assert (Hstack : readStack memory (callerBase top) (Z.to_nat (base - callerBase top)) = Some (callerValues top)).
    { apply read_stack_cells. split; [lia | split; assumption]. }
    rewrite Hstack, Hread, <- Hz, <- Hres, Hcont.
    destruct (IH (sourceFunction (callPosition top)) g (callerBase top) (callerFbp top) (depth - 1)
      (stackPrefix (callerValues top) (instructionSuffix (callContinuation top)
        (callContext (callerFrame top) (N.of_nat (callResults top)) inner)))
      (obligations /\ supportedStraightLineProgram (guestBody f) /\
        validatedCode program g (mk_state s (callerFrame top)) (length (callerValues top) + callResults top)
          (callContinuation top)) Hg Hrest) as [result [Hdecode Hresult]].
    { split; [exact Hobligations | split; [exact Hstraight | exact Hvalid]]. }
    exists result. split; [exact Hdecode | exact Hresult].
Qed.

(* The bridge: what the audited statement says, in the invariant's terms. *)
Theorem circuit_config_state : forall program s baseDepth shape memory boundary c,
  CircuitConfig program s baseDepth shape memory boundary c ->
  exists cursor z, fun_store z = s /\ ControlState program baseDepth cursor z memory boundary c.
Proof.
  intros program s baseDepth shape memory boundary c [Hfuncs [Hwf [obligations [Hdecode Hobligations]]]].
  unfold decodeControl in Hdecode.
  destruct (_ && _) eqn:Hchecks; [ | discriminate]. bools.
  destruct (decodeActive program s memory boundary (activeSite shape)) as [[inner active] | ] eqn:Hactive;
    [ | discriminate]. cbn [fst snd] in Hdecode.
  destruct (decode_active_function _ _ _ _ _ _ Hactive) as [f Hf].
  destruct (decoded_calls _ _ _ _ _ _ _ _ _ _ _ _ _ Hfuncs Hwf Hf Hdecode Hobligations)
    as [Hinner [stack [Hsites [Hsusp ->]]]].
  destruct (decoded_active _ _ _ _ _ _ _ Hfuncs Hwf Hactive Hinner) as [position [z [Hsite [Hstore Hstate]]]].
  exists (controlCursor stack position), z. split; [exact Hstore | ].
  exists f, inner. cbn [activePosition suspendedCalls].
  split; [rewrite active_function_site, Hsite; exact Hf | ]. split; [exact Hstate | ].
  split; [rewrite Hstore; exact Hsusp | ]. split; [assumption | ].
  split; [rewrite <- Hsites, length_map in *; assumption | reflexivity].
Qed.

Theorem control_state_circuit : forall program s baseDepth cursor z memory boundary c,
  store_FUNCS s = program -> wf_store s -> fun_store z = s ->
  ControlState program baseDepth cursor z memory boundary c ->
  CircuitConfig program s baseDepth (shapeOf cursor) memory boundary c.
Proof.
  intros program s baseDepth [stack position] z memory boundary c Hfuncs Hwf Hstore
    [f [inner [Hf [Hstate [Hsusp [Hbase [Hdepth ->]]]]]]].
  cbn [activePosition suspendedCalls] in *.
  split; [exact Hfuncs | split; [exact Hwf | ]].
  unfold decodeControl, shapeOf. cbn [callSites activeSite activePosition suspendedCalls]. rewrite length_map. pass_check.
  destruct (active_decodes _ _ _ _ _ _ _ Hstore Hstate) as [active [Hactive Hobligations]].
  rewrite Hactive. cbn [fst snd]. rewrite active_function_site in Hf. rewrite Hstore in Hsusp.
  exact (calls_decode _ _ _ _ _ _ _ _ _ _ _ Hf Hsusp Hobligations).
Qed.
