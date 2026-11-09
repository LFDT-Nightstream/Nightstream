(* The rows of an indirect call. Its resolve row checks the table entry and
   steps SpecTec to CALL_ADDR, leaving the reference on top; its invocation row
   enters the callee like a direct call. *)
From Stdlib Require Import List ZArith NArith Bool Lia.
Require Import Wasm2 WasmIntegers R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement StraightLineSpec StackSimulation SpecTecCalls SpecTecIndirect CallerSpec CallerRefinement CallSpec
  CallBindings CallContext CallMemory CallInitialization CallScope LocalBindings FrameInitialization
  ProgramSpec TableState ProgramEncoding CodeLayout BlockSpec BlockEncoding BlockTrace Fragment CodeFacts
  FunctionValidation ControlDecode ControlInvariant FrameFacts ControlBridge FrameRows CallRows ProgramBinding
  Selectors CircuitFacts IndirectBindings SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Section Indirect.
Variables (program : Program) (s : store).
Hypothesis Hstore : store_FUNCS s = program.
Hypothesis Hwf : wf_store s.
Hypothesis Hvalid : programValidated program s.

(* The function a validated program's function belongs to is validated with its module. *)
Lemma function_module_ok : forall index f,
  functionAt program index = Some f -> exists C, Moduleinst_ok s (guestModule f) C.
Proof.
  intros index f Hf. destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) (proj1 Hvalid) _ (nth_error_In _ _ Hnth)) as Hinst.
  destruct (funcinst_body_validated _ _ _ _ _ _ _ Hinst) as [C [HC _]]. exists C. exact HC.
Qed.

Lemma function_type_member : forall index f,
  functionAt program index = Some f -> In (guestType f) (TYPES (guestModule f)).
Proof.
  intros index f Hf. destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  pose proof (proj1 (Forall_forall _ _) (proj1 Hvalid) _ (nth_error_In _ _ Hnth)) as Hinst.
  exact (function_type_in_module _ _ _ _ _ _ _ Hinst).
Qed.

(* The resolve row of an indirect call that does not trap: the table entry at the
   index is the reference to a function of the expected type. *)
Lemma resolve_row : forall view f x y initial final row inner,
  fieldPrime -> programLoaded program initial -> tablesLoaded s initial -> FullRelation.satFull row ->
  row COL_IS_PROGRAM_ROW = 1 -> row COL_ACTIVATION_FINISHED_AFTER = 0 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (CALL_INDIRECT x y) ->
  exists view' callee inner',
    viewPosition view' = viewPosition view /\
    Steps inner inner' /\ ActiveState program s final (afterBoundary row) (resolved view' callee) inner' /\
    frameRegistersPreserved row /\ FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final.
Proof.
  intros view f x y initial final row inner Hprime Hloaded Htables Hfull Hprogram Hlive Hreplay Hstate Hf Hi.
  destruct (FrameRows.program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen0 [Hlb0 Hlc0]]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hi, Hpath, Hheights, Hbounds, Hcells, Hllen0, Hlb0, Hlc0, Hsp, Hpc |- *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup.
  destruct Hfacts as [[Hwi [Htable [params [results [Htype Hheight]]]]] _]. rewrite Nat.add_0_r in Hheight.
  inversion Hsup as [ | ? ? Hhead _].
  inversion Hhead as [ | | | | | | | | | | | | x' y' p r Htype' Hp Hr Hready_ | | ].
  rewrite Htype in Htype'. injection Htype' as Hpp Hrr.
  apply (f_equal (@length _)) in Hpp, Hrr. rewrite !repeat_length in Hpp, Hrr. subst p r.
  cbn [programCallees envTypes envTables indirectReady] in Htype, Htable, Hready_.
  destruct (split_top_values values 1 ltac:(lia)) as [prefix [top [-> Htop]]].
  destruct top as [ | tableIndex [ | ? ?]]; try discriminate. clear Htop.
  rewrite length_app in Hheight. cbn [length] in Hheight.
  (* The table and type the instruction names. *)
  destruct (nth_error (TABLES (guestModule f)) (N.to_nat (x :> N))) as [ta | ] eqn:Hta;
    [ | apply nth_error_None in Hta; lia].
  destruct (encoded_path_resolve _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as [Himage [HtableRom HtypeRom]].
  rewrite Hta in HtableRom. rewrite Htype in HtypeRom. cbn [option_map romValue] in HtableRom, HtypeRom.
  unfold pathPC in Himage, HtableRom, HtypeRom.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage, HtableRom, HtypeRom.
  destruct (BlockTrace.control_row _ _ row _ CODE_CALL_INDIRECT_RESOLVE COL_SEL_CALL_INDIRECT_RESOLVE _ Hprime Hfull
    Hprogram Hreplay Logic.eq_refl ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage)
    as [Hresolve Hnext].
  (* The index is the top operand. *)
  assert (HspPositive : 1 <= row COL_SP_BEFORE).
  { change (row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE +
      Z.of_nat (length (concat segments ++ prefix ++ [tableIndex]))) in Hsp.
    rewrite !length_app in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_resolve_stack_shape row Hprime Hfull Hresolve HspPositive) as Hshape.
  destruct (replace_stack_replay row initial final Hshape Hreplay) as [Hread HstackFinal].
  assert (Htop : initial stackId [2 * (row COL_SP_BEFORE - 1)] = Some tableIndex).
  { change (row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE +
      Z.of_nat (length (concat segments ++ prefix ++ [tableIndex]))) in Hsp.
    replace (row COL_SP_BEFORE - 1) with
      (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat segments ++ prefix)))
      by (rewrite Hsp, !length_app; cbn [length]; lia).
    apply Hcells. unfold viewOperands. cbn [viewSegments viewValues].
    rewrite app_assoc, nth_error_app2, Nat.sub_diag by lia. reflexivity. }
  rewrite (replace_read_address _ Hshape), Htop in Hread. injection Hread as HindexValue.
  (* The row continues, so it does not trap: every check passed. *)
  destruct (full_resolve_checks row Hprime Hfull Hresolve) as [HidActive [HimmActive [HsizeActive [HindexCol
    [Hoob [HreadActive [Hnull [Heq [Hgate [HrefIs [Htrap HwriteRef]]]]]]]]]]].
  assert (Hnotrap : row COL_CALL_INDIRECT_IS_TRAP = 0).
  { destruct (Z.eq_dec (row COL_CALL_INDIRECT_IS_TRAP) 1) as [Hone1 | Hone1];
      [pose proof (full_resolve_trap_finishes row Hprime Hfull Hresolve Hone1); lia | ].
    rewrite Htrap in Hone1 |- *. rewrite Hgate, Hoob, Hnull, Heq in Hone1 |- *.
    destruct (_ <? _), (_ =? 0), (_ =? _); cbn in *; lia. }
  rewrite Htrap, Hgate, Hoob, Hnull, Heq in Hnotrap.
  destruct (row COL_TABLE_INDEX <? row COL_TABLE_SIZE) eqn:Hin; [ | cbn in Hnotrap; lia].
  destruct (row COL_TABLE_VALUE =? 0) eqn:Hnonnull; [cbn in Hnotrap; lia | ].
  destruct (row COL_FUNCTION_TYPE_ID =? row COL_EXPECTED_TYPE_ID) eqn:Htypes; [ | cbn in Hnotrap; lia].
  apply Z.ltb_lt in Hin. apply Z.eqb_neq in Hnonnull. apply Z.eqb_eq in Htypes.
  assert (HoobZero : row COL_CI_OOB = 0) by (cbn in Hoob; exact Hoob).
  assert (HnullZero : row COL_CI_ENTRY_IS_NULL = 0) by (cbn in Hnull; exact Hnull).
  assert (HgateOne : row COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 1) by (rewrite Hgate, HoobZero, HnullZero; reflexivity).
  (* The table: its address from the ROM, its size and entry from the store. *)
  destruct (resolve_rom_reads _ _ _ Hreplay HidActive HimmActive HsizeActive) as [HidRead [HexpRead HsizeRead]].
  rewrite HtableRom in HidRead. injection HidRead as HtableId.
  rewrite HtypeRom in HexpRead. injection HexpRead as Hexpected.
  destruct (function_module_ok _ _ Hf) as [C HC].
  pose proof (module_table_in_store _ _ _ _ _ HC Hta) as HtaStore.
  destruct (nth_error (store_TABLES s) (N.to_nat ta)) as [t | ] eqn:Ht; [ | apply nth_error_None in Ht; lia].
  destruct (Htables _ _ Ht) as [Hsize Hentries].
  replace (Z.of_nat (N.to_nat ta)) with (Z.of_N ta) in Hsize, Hentries by lia.
  rewrite HtableId, HsizeRead in Hsize. injection Hsize as Hsize.
  assert (Hindex : row COL_TABLE_INDEX = tableIndex) by (rewrite HindexCol; symmetry; exact HindexValue).
  assert (HindexBound : 0 <= tableIndex) by (pose proof (full_read0_u32 row Hprime Hfull); lia).
  destruct (nth_error (REFS t) (Z.to_nat tableIndex)) as [r | ] eqn:Hslot; [ | apply nth_error_None in Hslot; lia].
  assert (HsetOff : row COL_SEL_TABLE_SET <> 1).
  { assert (Hin' : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
    rewrite (full_opcode_selector_exclusive row _ COL_SEL_TABLE_SET Hprime Hfull Hresolve Hin'
      ltac:(vm_compute; tauto) ltac:(discriminate)). discriminate. }
  pose proof (resolve_entry_read _ _ _ Hfull Hreplay HsetOff ltac:(rewrite HreadActive, HoobZero; reflexivity)) as Hentry.
  pose proof (Hentries _ _ Hslot) as HentryStore. rewrite Z2Nat.id in HentryStore by exact HindexBound.
  rewrite HtableId, <- Hindex, Hentry in HentryStore.
  destruct r as [rt | a | h]; cbn in HentryStore; [injection HentryStore as HentryStore; lia | | discriminate].
  injection HentryStore as Hvalue.
  (* The callee: the function at a, whose type id is the expected one. *)
  pose proof (HrefIs HgateOne) as Hreference. rewrite Hvalue in Hreference.
  pose proof (resolve_type_read _ _ _ Hreplay HgateOne) as HtypeRead.
  rewrite Hreference, Htypes, <- Hexpected in HtypeRead.
  destruct (function_type_read _ _ _ _ Hloaded HtypeRead (type_id_positive _ _)) as [g [_ [Hg Hid]]].
  replace (Z.to_nat (Z.of_N a + 1 - 1)) with (N.to_nat a) in Hg by lia.
  destruct (function_at_sound _ _ _ Hg) as [Hgnth _].
  destruct (Hready_ _ _ Hgnth) as [g' [Hg' Hmodule]]. rewrite Hg in Hg'. injection Hg' as <-.
  rewrite Hmodule in Hid.
  assert (HcalleeType : guestType g = i32StackType params results).
  { apply (type_id_injective (guestModule f)); [ | exact (nth_error_In _ _ Htype) | symmetry; exact Hid].
    rewrite <- Hmodule. exact (function_type_member _ _ Hg). }
  (* SpecTec resolves the same entry. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb0).
  assert (Hu32 : 0 <= tableIndex < i32Modulus).
  { pose proof (full_read0_u32 row Hprime Hfull). normalize_columns. unfold i32Modulus. lia. }
  assert (Htarget : indirectTarget z x y tableIndex a).
  { split.
    - change (nth_error (REFS (lookup_total (store_TABLES s) (lookup_total (TABLES (guestModule f)) (x :> N))))
        (Z.to_nat tableIndex) = Some (REF_FUNC_ADDR a)).
      rewrite (spec_lookup_from_nth_error _ _ _ _ _ Hta), (spec_lookup_from_nth_error _ _ _ _ _ Ht). exact Hslot.
    - exists (i32Function (guestModule g) (guestTypeIndex g) (guestParams g) (guestResults g) (guestLocals g)
        (guestBody g)).
      split; [change (nth_error (store_FUNCS s) (N.to_nat a) = Some (i32Function (guestModule g)
        (guestTypeIndex g) (guestParams g) (guestResults g) (guestLocals g) (guestBody g))); rewrite Hstore; exact Hgnth | ].
      change (guestType g = lookup_total (TYPES (guestModule f)) (y :> N)).
      rewrite (spec_lookup_from_nth_error _ _ _ _ _ Htype). exact HcalleeType. }
  apply Forall_app in Hbounds. destruct Hbounds as [HsegBounds HvalBounds].
  apply Forall_app in HvalBounds. destruct HvalBounds as [HprefixBounds HtopBounds].
  set (resolvedView := frameView (codePosition index labels offset) segments prefix locals).
  exists resolvedView, (N.to_nat a),
    (frameConfig f resolvedView (stackPrefix prefix (instructionSuffix (skipn (S offset) code)
      (mk_config (mk_state s (activeFrame f locals)) [CALL_ADDR (N.of_nat (N.to_nat a))])))).
  split; [reflexivity | ]. split.
  { unfold frameConfig, resolvedView. cbn [viewPosition viewSegments positionLabels].
    apply steps_in_labels; [exact Hready | ].
    rewrite (skipn_at_instruction _ _ _ Hi), stack_config_split. rewrite Nnat.N2Nat.id.
    apply steps_with_stack_prefix; [exact HprefixBounds | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | ].
    apply one_spec_step.
    - apply stack_config_wf; [exact Hz | exact HtopBounds | constructor; [exact Hwi | constructor]].
    - constructor; [exact Hz | repeat constructor].
    - exact (indirect_call_resolves z x y tableIndex a Hu32 Htarget). }
  (* The circuit side: the reference replaces the index; nothing else changes. *)
  destruct (full_resolve_frame row Hprime Hfull Hresolve) as [Hpush [Hpop [HfbpAfter HdepthAfter]]].
  assert (Hin' : In COL_SEL_CALL_INDIRECT_RESOLVE selectorColumns) by (vm_compute; tauto).
  destruct (full_selected_no_host_switch row _ Hprime Hfull Hin' Hresolve) as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (program_row_keeps_idle row _ Hprime Hfull Hin' Hresolve ltac:(vm_compute; intuition discriminate)
    Hpop Hidle) as HidleAfter.
  assert (HtrapZero : row COL_CALL_INDIRECT_IS_TRAP = 0).
  { rewrite Htrap, HgateOne. cbn in Heq. rewrite Heq. reflexivity. }
  pose proof (HwriteRef HtrapZero) as Hwritten. rewrite Hreference in Hwritten.
  assert (Hkeep : forall k, k <> row COL_SP_BEFORE - 1 -> final stackId [2 * k] = initial stackId [2 * k]).
  { intros k Hk. rewrite HstackFinal, (replace_write_address _ Hshape), (replace_write_hi_address _ Hshape).
    rewrite !update_other; [reflexivity | cell_differs | cell_differs]. }
  assert (HtopFinal : final stackId [2 * (row COL_SP_BEFORE - 1)] = Some (Z.of_N a + 1)).
  { rewrite HstackFinal, (replace_write_address _ Hshape), (replace_write_hi_address _ Hshape).
    rewrite update_other by cell_differs. rewrite update_same, Hwritten. reflexivity. }
  change (row COL_SP_BEFORE = row COL_STACK_FRAME_BASE_BEFORE +
    Z.of_nat (length (concat segments ++ prefix ++ [tableIndex]))) in Hsp.
  rewrite app_assoc, !length_app in Hsp. cbn [length] in Hsp.
  split.
  - exists f, g, (CALL_INDIRECT x y).
    unfold resolvedView. cbn [viewPosition viewSegments viewValues viewLocals viewOperands positionFunction
      positionLabels positionOffset afterBoundary boundaryStackBase boundaryLocalsBase boundarySP boundaryPC
      boundaryZeroRemaining boundaryMoveRemaining].
    split.
    { apply FrameRows.frame_at_unmoved_intro;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands].
      - exact Hf.
      - exact Hpath.
      - rewrite Hheights, length_app. cbn [length]. f_equal. f_equal. lia.
      - rewrite HfbpAfter. exact Hfbp0.
      - rewrite (replace_stack_base _ Hshape), HfbpAfter. exact Hbase0.
      - unfold viewOperands. cbn [viewSegments viewValues]. apply Forall_app. split; assumption.
      - intros i v Hv. unfold viewOperands in Hv. cbn [viewSegments viewValues] in Hv.
        assert (Hlt : (i < length (concat segments ++ prefix))%nat) by (apply nth_error_Some; rewrite Hv; discriminate).
        rewrite (replace_stack_base _ Hshape), Hkeep by (rewrite length_app in Hlt; lia).
        apply Hcells. unfold viewOperands. cbn [viewSegments viewValues].
        rewrite app_assoc, nth_error_app1 by exact Hlt. exact Hv.
      - exact Hllen0.
      - exact Hlb0.
      - intros i v Hv. rewrite HfbpAfter, Hkeep.
        + exact (Hlc0 _ _ Hv).
        + assert (i < length locals)%nat by (apply nth_error_Some; rewrite Hv; discriminate).
          unfold frameLocals in Hbase0, Hllen0. lia. }
    split; [exact Hi | ]. split; [reflexivity | ]. split; [exact Hg | ].
    split; [cbn [callsInto programCallees envTypes]; rewrite Htype; f_equal; symmetry; exact HcalleeType | ].
    split.
    { unfold viewOperands. cbn [viewSegments viewValues].
      rewrite (replace_stack_base _ Hshape), length_app. replace (2 * (row COL_STACK_FRAME_BASE_BEFORE +
        Z.of_nat (length (concat segments) + length prefix))) with (2 * (row COL_SP_BEFORE - 1)) by lia.
      rewrite HtopFinal. f_equal. lia. }
    split; [unfold viewOperands; cbn [viewSegments viewValues]; rewrite length_app;
      rewrite (replace_stack_pointer _ Hshape), (replace_stack_base _ Hshape); lia | ].
    split; [rewrite Hnext, Hpc; reflexivity | ].
    destruct HidleAfter as [Hzr [Hmr [Hza Hma]]].
    split; [unfold initializationFlags; rewrite Hza, Hma, Hzr, Hmr; split; reflexivity | ].
    split; [exact Hzr | ]. split; [exact Hmr | ]. rewrite Nnat.N2Nat.id. reflexivity.
  - split; [split; [exact HdepthAfter | split; [exact (replace_stack_base _ Hshape) | exact HfbpAfter]] | ].
    constructor.
    + intros k Hk. apply Hkeep. unfold frameLocals in Hbase0. lia.
    + intros id column address Hmem. eapply context_preserved_without_push; eassumption.
Qed.

(* The invocation row from a resolved indirect call: SpecTec enters the callee at
   CALL_ADDR; the circuit pops the reference and enters the callee like a direct
   call, saving the return context after the call_indirect. *)
Lemma invoke_row : forall baseDepth stack view callee f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (resolved view callee) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  SuspendedStack program initial f (row COL_LOCALS_FBP_BEFORE) (row COL_CALL_STACK_DEPTH_BEFORE) stack ->
  0 <= baseDepth -> row COL_CALL_STACK_DEPTH_BEFORE = baseDepth + Z.of_nat (length stack) ->
  exists cursor c1, cursorRoot cursor = rootFunction stack (positionFunction (viewPosition view)) /\
    Steps (plugCalls stack inner) c1 /\
    ControlState program s baseDepth cursor final (afterBoundary row) c1.
Proof.
  intros baseDepth stack view callee f initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf
    Hsusp Hbase0 Hdepth.
  destruct Hstate as [f' [g [i [Hframe [Hi [Hind [Hg [Hcalls [Href [Hsp [Hpc [Hflags [Hzr [Hmr ->]]]]]]]]]]]]]].
  cbn [beforeBoundary boundaryStackBase boundaryLocalsBase boundarySP boundaryPC boundaryZeroRemaining
    boundaryMoveRemaining] in Hframe, Href, Hsp, Hpc, Hzr, Hmr.
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [HbaseLayout [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  pose proof (FrameRows.frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  destruct i; try discriminate Hind. rename v_tableidx into x, v_typeidx into y.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in Hf, Hi, Hpath, Hheights, Hbounds, Hcells, Hllen, Hlb, Hlc, Hsp, Hpc, Href |- *.
  unfold viewOperands in Hbounds, Hcells, Hsp, Href. cbn [viewSegments viewValues] in Hbounds, Hcells, Hsp, Href.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup.
  destruct Hfacts as [[Hwi [_ [params [results [Htype Hheight]]]]] _].
  inversion Hsup as [ | ? ? Hhead _].
  inversion Hhead as [ | | | | | | | | | | | | x' y' p r Htype' Hp Hr _ | | ].
  rewrite Htype in Htype'. injection Htype' as Hpp Hrr.
  apply (f_equal (@length _)) in Hpp, Hrr. rewrite !repeat_length in Hpp, Hrr. subst p r.
  cbn [callsInto programCallees envTypes] in Hcalls, Htype. rewrite Htype in Hcalls. injection Hcalls as Hgp Hgr.
  apply (f_equal (@length _)) in Hgp, Hgr. rewrite !repeat_length in Hgp, Hgr.
  destruct (split_top_values values params ltac:(lia)) as [prefix [arguments [-> Hlen]]].
  (* The invocation opcode, its return edge, and the reference on top. *)
  destruct (encoded_path_invoke _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as [Hop [_ HretRom]].
  unfold pathPC in Hop, HretRom. rewrite <- Hpc in Hop, HretRom.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hop in Hfetch. injection Hfetch as Hcode.
  assert (Hsel : row COL_SEL_CALL_INDIRECT = 1).
  { apply (full_opcode_selects row COL_SEL_CALL_INDIRECT CODE_CALL_INDIRECT Hprime Hfull Hprogram);
      [vm_compute; reflexivity | symmetry; exact Hcode]. }
  destruct (full_indirect_reference row Hprime Hfull Hsel ltac:(lia)) as [Hfref [Haddr Hactive0]].
  pose proof (first_stack_read _ _ _ Hreplay Hactive0) as Hread.
  rewrite Haddr in Hread. replace (row COL_SP_BEFORE - 1) with
    (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat segments ++ prefix ++ arguments))) in Hread by lia.
  rewrite Href in Hread. injection Hread as Hreference.
  assert (HrefValue : row COL_FUNCTION_REF = Z.of_nat callee + 1) by (normalize_columns; lia). clear Hfref Hreference.
  (* The callee's metadata: a guest with params and results. *)
  destruct (function_at_sound _ _ _ Hg) as [Hgnth [Hgaddr Hgentry]].
  pose proof (encoded_guest_image _ _ _ _ Hloaded Hg) as [Hmeta [Hcount [Hentry _]]].
  rewrite Hgaddr in Hmeta, Hcount, Hentry. replace (Z.of_N (N.of_nat callee) + 1) with (row COL_FUNCTION_REF)
    in Hmeta, Hcount, Hentry by (rewrite HrefValue; lia).
  pose proof (indirect_metadata_read _ _ _ Hreplay Hsel) as HmetaRead. rewrite Hmeta in HmetaRead.
  injection HmetaRead as Hmetadata.
  destruct (guest_metadata_decodes row (Z.of_nat (guestParams g)) (Z.of_nat (guestResults g))
    Hprime Hfull ltac:(lia) ltac:(lia) (eq_sym Hmetadata)) as [HparamCount [_ HguestTarget]].
  destruct (full_indirect_guest_call row Hprime Hfull Hsel HguestTarget) as [Hguest [Hpush Hpop]].
  destruct (full_indirect_guest_call_stack row Hprime Hfull Hsel HguestTarget) as [_ [_ [_ [Hwrites HspAfter]]]].
  destruct (guest_entry_reads _ _ _ Hreplay Hguest) as [HtotalRead HentryRead].
  rewrite Hcount in HtotalRead. rewrite Hentry in HentryRead.
  injection HtotalRead as HtotalValue. injection HentryRead as HentryValue.
  pose proof (full_guest_initialization_seed row Hprime Hfull Hprogram Hguest) as Hseed.
  destruct (full_call_frame_entry row Hprime Hfull Hguest Hpush Hpop Hprogram) as [HfbpAfter [HbaseAfter HdepthAfter]].
  destruct (full_call_context_push row Hfull Hpush) as [HaddrPush [HsavedFbp [HsavedBase Hchoice]]].
  assert (Hin : In COL_SEL_CALL_INDIRECT selectorColumns) by (vm_compute; tauto).
  destruct (full_selected_no_host_switch row COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hsel)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (no_stack_writes_preserve _ _ _ Hreplay Hwrites) as HstackFinal.
  (* A call seeds no frame move. *)
  assert (HmoveAfter : row COL_FRAME_MOVE_REMAINING_AFTER = 0).
  { destruct Hflags as [Hma _]. cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining] in Hma.
    rewrite Hmr in Hma. cbn in Hma.
    assert (Hone : row COL_ONE = 1) by exact (proj1 Hfull).
    circuit_row ([(COL_ONE, 1); (COL_FRAME_SLIDE_ACTIVE, -1); (COL_GUEST_ENTRY_ACTIVE, -1);
      (COL_CALL_STACK_PUSH_PRESENT, 1); (COL_FRAME_MOVE_ACTIVE_BEFORE, -1)],
      [(COL_FRAME_MOVE_REMAINING_AFTER, 1); (COL_FRAME_MOVE_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    pose proof (full_selected_no_slide row COL_SEL_CALL_INDIRECT Hprime Hfull Hin Hsel
      ltac:(vm_compute; intuition discriminate) Hpop) as Hslide.
    expand_row Hrow. normalize_columns. rewrite Hone, Hslide, Hguest, Hpush, Hma, Hmr in Hrow.
    pose proof (value_bound row COL_FRAME_MOVE_REMAINING_AFTER). field_linear Hrow. }
  pose proof (context_push_records initial final row callStackReturnPcId COL_CALL_STACK_RETURN_PC_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedPcMem.
  pose proof (context_push_records initial final row callStackCallerFbpId COL_CALL_STACK_CALLER_FBP_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedFbpMem.
  pose proof (context_push_records initial final row callStackCallerSpBaseId COL_CALL_STACK_CALLER_SP_BASE_VALUE
    ltac:(cbn; auto) Hpush Hsuspend HcontextRead Hreplay) as HsavedBaseMem.
  pose proof (call_return_pc_read initial final row Hreplay Hpush) as HreturnRead.
  rewrite Hchoice, HretRom in HreturnRead. injection HreturnRead as HreturnValue.
  apply Forall_app in Hbounds. destruct Hbounds as [HsegBounds HvalBounds].
  apply Forall_app in HvalBounds. destruct HvalBounds as [HprefixBounds HargBounds].
  rewrite !length_app in Hsp.
  (* The callee's frame starts at its arguments, below the popped reference. *)
  assert (Hinit : InitializingFrame final (afterBoundary row) arguments (Z.of_nat (guestLocals g))
      (Z.of_nat (guestLocals g))).
  { apply initializing_frame_start; cbn [afterBoundary boundarySP boundaryStackBase boundaryLocalsBase].
    - lia.
    - rewrite HfbpAfter, HspAfter, HparamCount. lia.
    - rewrite HfbpAfter, HspAfter, HparamCount. lia.
    - rewrite HbaseAfter, HfbpAfter, HspAfter, HparamCount. lia.
    - intros j v Hv. rewrite HstackFinal, HfbpAfter, HspAfter, HparamCount.
      replace (row COL_SP_BEFORE - 1 - Z.of_nat (guestParams g) + Z.of_nat j) with
        (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length (concat segments ++ prefix) + j)) by
        (rewrite length_app, Nat2Z.inj_add, Nat2Z.inj_add; lia).
      apply Hcells. rewrite app_assoc, nth_error_app2 by lia.
      replace (length (concat segments ++ prefix) + j - length (concat segments ++ prefix))%nat with j by lia.
      exact Hv. }
  assert (HzeroAfter : row COL_LOCAL_ZERO_REMAINING_AFTER = Z.of_nat (guestLocals g)) by lia.
  rewrite <- HzeroAfter in Hinit at 2.
  destruct (callee_active_state program s callee g arguments final (afterBoundary row) Hg ltac:(lia)
    HargBounds Hinit (full_initialization_flags row Hprime Hfull) HmoveAfter
    ltac:(cbn [afterBoundary boundaryPC]; lia)) as [position [Hactive Hentered]].
  (* SpecTec enters the callee from CALL_ADDR. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  assert (Hfuncwf : wf_funcinst (i32Function (guestModule g) (guestTypeIndex g) (length arguments)
      (guestResults g) (guestLocals g) (guestBody g))) by (rewrite Hlen, Hgp; exact (function_inst_wf program s Hvalid _ _ Hg)).
  assert (Hlookup : nth_error (fun_funcinst z) (N.to_nat (N.of_nat callee)) = Some (i32Function (guestModule g)
      (guestTypeIndex g) (length arguments) (guestResults g) (guestLocals g) (guestBody g))).
  { change (nth_error (store_FUNCS s) (N.to_nat (N.of_nat callee)) = Some (i32Function (guestModule g)
      (guestTypeIndex g) (length arguments) (guestResults g) (guestLocals g) (guestBody g))).
    rewrite Hstore, Nnat.Nat2N.id, Hlen, Hgp. exact Hgnth. }
  pose proof (call_address_enters_i32_frame z _ _ _ _ _ _ _ Hlookup Hfuncwf HargBounds) as Henter.
  set (top := suspendedCall (frameView (codePosition index labels offset) segments prefix locals) f
    (guestResults g) (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE)).
  exists (controlCursor (top :: stack) position),
    (plugCalls (top :: stack) (stackConfig (mk_state s (enteredFrame (guestModule g) arguments (guestLocals g))) []
      (guestBody g))).
  split; [reflexivity | ].
  split.
  { cbn [plugCalls]. apply steps_in_plug_calls;
      [eapply (suspended_stack_ready program s Hstore Hwf Hvalid); exact Hsusp | ].
    unfold callerConfig, frameConfig, top.
    cbn [callView callerFunction callResults viewPosition viewSegments viewValues viewLocals positionLabels
      positionOffset].
    apply steps_in_labels; [exact Hready | ].
    replace (stackPrefix (prefix ++ arguments) (instructionSuffix (skipn (S offset) code)
        (mk_config z [CALL_ADDR (N.of_nat callee)]))) with
      (stackPrefix prefix (instructionSuffix (skipn (S offset) code)
        (mk_config z (List.map i32Constant arguments ++ [CALL_ADDR (N.of_nat callee)]))))
      by (unfold stackPrefix, instructionSuffix; rewrite map_app, <- !app_assoc; reflexivity).
    apply steps_with_stack_prefix; [exact HprefixBounds | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | ].
    apply one_spec_step; [ | | exact Henter].
    - constructor; [exact Hz | ]. apply Forall_app. split; [ | repeat constructor].
      apply Forall_map. eapply Forall_impl; [ | exact HargBounds]. intros v Hv. exact (i32_constant_wf v Hv).
    - change (wf_config (callContext (activeFrame f locals) (N.of_nat (guestResults g))
        (stackConfig (mk_state s (enteredFrame (guestModule g) arguments (guestLocals g))) [] (guestBody g)))).
      apply call_context_wf; [inversion Hz; assumption | ].
      exact (entered_body_wf s (activeFrame f locals) _ _ _ _ _ _ Hz Hfuncwf HargBounds). }
  exists g, (stackConfig (mk_state s (enteredFrame (guestModule g) arguments (guestLocals g))) [] (guestBody g)).
  split; [exact Hactive | ]. split; [exact Hentered | ].
  split; [ | split; [exact Hbase0 | split; [cbn [suspendedCalls length afterBoundary boundaryCallDepth]; lia |
    reflexivity]]].
  unfold top. cbn [SuspendedStack suspendedCalls afterBoundary boundaryLocalsBase boundaryCallDepth
    callView callerFunction callResults callerBase callerFbp viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset].
  exists (CALL_INDIRECT x y). split.
  { apply FrameRows.frame_at_unmoved_intro;
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset];
      unfold viewOperands; cbn [viewSegments viewValues].
    - exact Hf.
    - exact Hpath.
    - rewrite Hheights, length_app. cbn [callOperands]. f_equal. f_equal. lia.
    - exact Hfbp0.
    - exact HbaseLayout.
    - apply Forall_app; split; assumption.
    - intros j v Hv. rewrite HstackFinal. apply Hcells.
      rewrite app_assoc, nth_error_app1; [exact Hv | apply nth_error_Some; rewrite Hv; discriminate].
    - exact Hllen.
    - exact Hlb.
    - intros j v Hv. rewrite HstackFinal. exact (Hlc _ _ Hv). }
  split; [exact Hi | ]. split; [cbn [callsInto programCallees envTypes]; rewrite Htype; f_equal;
    unfold guestType; rewrite <- Hgp, <- Hgr; reflexivity | ]. split; [reflexivity | ].
  split.
  { unfold viewOperands. cbn [viewSegments viewValues]. rewrite length_app.
    rewrite HfbpAfter, HspAfter, HparamCount. lia. }
  replace (row COL_CALL_STACK_DEPTH_AFTER - 1) with (row COL_CALL_STACK_ADDR) by lia.
  split.
  { rewrite HsavedPcMem, <- HreturnValue, (path_offset_step _ _ _ _ _ Hpath Hi).
    change (instructionSize (CALL_INDIRECT x y)) with 2%nat. f_equal. lia. }
  split; [rewrite HsavedFbpMem, HsavedFbp; reflexivity | ].
  split; [rewrite HsavedBaseMem, HsavedBase; reflexivity | ].
  (* The callers already suspended keep their memory: the push writes only its own level. *)
  replace (row COL_CALL_STACK_ADDR) with (row COL_CALL_STACK_DEPTH_BEFORE) by lia.
  eapply suspended_stack_preserved; [exact Hsusp | intros j _; apply HstackFinal | ].
  intros id column level Hmem Hbelow. eapply context_push_elsewhere;
    [exact Hmem | exact Hsuspend | exact HcontextRead | exact Hreplay | ].
  intros Heq. injection Heq as Heq. lia.
Qed.

End Indirect.
