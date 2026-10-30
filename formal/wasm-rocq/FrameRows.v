(* Rows of the active frame's code, at any depth of enclosing blocks: an ordinary
   instruction, entering a block, a block's END, and a depth-0 branch. Each keeps
   the frame registers and touches no memory below the frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers Add R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog SlideLayout
  TraceRefinement StraightLineSpec StackSimulation LocalSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerRefinement ProgramSpec ProgramEncoding CodeLayout BlockSpec BlockBindings BlockEncoding BlockInstructions
  BlockTrace BranchBindings StackBindings CallInitialization Selectors CallScope CallContext CallMemory LocalBindings
  StraightLineInstructions ProgramBinding Fragment CodeFacts ControlDecode ControlInvariant FrameFacts ControlBridge
  CircuitFacts.
Import ListNotations.
Open Scope Z_scope.

Ltac not_call_selector :=
  let H := fresh in intro H; vm_compute in H; repeat (destruct H as [H | H]; [discriminate H | ]); exact H.

(* With no operand moved, a frame's operands sit at plain offsets from its base,
   and the pending-move distance does not matter. *)
Lemma frame_at_unmoved_cells : forall program memory base fbp pending distance f view,
  FrameAt program memory base fbp pending 0 distance f view ->
  forall index value, nth_error (viewOperands view) index = Some value ->
    memory stackId [2 * (base + Z.of_nat index)] = Some value.
Proof.
  intros program memory base fbp pending distance f view [_ [_ [_ [_ [_ [_ [Hcells _]]]]]]] index value Hv.
  rewrite <- (operand_cell_unmoved base (length (viewOperands view)) distance index).
  - exact (Hcells _ _ Hv).
  - apply nth_error_Some. rewrite Hv. discriminate.
Qed.

Lemma frame_at_unmoved_intro : forall program memory base fbp pending distance f view,
  let position := viewPosition view in
  functionAt program (positionFunction position) = Some f ->
  pathCode (guestBody f) (positionLabels position) = Some (innerCode f position) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels position)
    (positionOffset position) =
    Some (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) ->
  0 <= fbp -> base = fbp + Z.of_nat (frameLocals f) ->
  Forall (fun value => 0 <= value < i32Modulus) (viewOperands view) ->
  (forall index value, nth_error (viewOperands view) index = Some value ->
    memory stackId [2 * (base + Z.of_nat index)] = Some value) ->
  length (viewLocals view) = frameLocals f ->
  Forall (fun value => 0 <= value < i32Modulus) (viewLocals view) ->
  (forall index value, nth_error (viewLocals view) index = Some value ->
    memory stackId [2 * (fbp + Z.of_nat index)] = Some value) ->
  FrameAt program memory base fbp pending 0 distance f view.
Proof.
  intros program memory base fbp pending distance f view position Hf Hpath Hheights Hfbp Hbase Hbounds Hcells
    Hllen Hlb Hlc.
  split; [exact Hf | ]. split; [exact Hpath | ]. split; [exact Hheights | ]. split; [exact Hfbp | ].
  split; [exact Hbase | ]. split; [exact Hbounds | ]. split; [ | split; [exact Hllen | split; [exact Hlb | exact Hlc]]].
  intros index value Hv. rewrite operand_cell_unmoved; [exact (Hcells _ _ Hv) | ].
  apply nth_error_Some. rewrite Hv. discriminate.
Qed.

Lemma frame_at_distance : forall program memory base fbp pending distance distance' f view,
  FrameAt program memory base fbp pending 0 distance f view ->
  FrameAt program memory base fbp pending 0 distance' f view.
Proof.
  intros program memory base fbp pending distance distance' f view Hframe.
  pose proof Hframe as [Hf [Hpath [Hheights [Hfbp [Hbase [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  apply frame_at_unmoved_intro; try assumption.
  exact (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe).
Qed.

(* A program row runs only once every countdown is done, so its frame is idle. *)
Lemma program_row_active : forall program s memory row view inner,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  ActiveState program s memory (beforeBoundary row) (atCode view) inner ->
  exists f, let position := viewPosition view in
    FrameAt program memory (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) 0 0 0 f view /\
    (positionOffset position <= length (innerCode f position))%nat /\
    boundarySP (beforeBoundary row) = boundaryStackBase (beforeBoundary row) + Z.of_nat (length (viewOperands view)) /\
    boundaryPC (beforeBoundary row) =
      guestEntry f + Z.of_nat (pathOffset (guestBody f) (positionLabels position) (positionOffset position)) /\
    initializationIdle (beforeBoundary row) /\
    inner = frameConfig f view (stackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
      (skipn (positionOffset position) (innerCode f position))).
Proof.
  intros program s memory row view inner Hprime Hfull Hprogram
    [f [Hframe [Hoffset [Hsp [Hpc [[Hma Hza] [Hzr [_ ->]]]]]]]].
  cbn [beforeBoundary boundaryMoveActive boundaryMoveRemaining boundaryZeroActive boundaryZeroRemaining
    boundaryStackBase boundaryLocalsBase boundaryMoveDistance] in *.
  assert (Hmove : row COL_FRAME_MOVE_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive row COL_IS_PROGRAM_ROW COL_FRAME_MOVE_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  assert (Hzero : row COL_LOCAL_ZERO_ACTIVE_BEFORE = 0).
  { exact (full_row_kind_exclusive row COL_IS_PROGRAM_ROW COL_LOCAL_ZERO_ACTIVE_BEFORE Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  assert (Hremaining : row COL_FRAME_MOVE_REMAINING_BEFORE = 0).
  { rewrite Hmove in Hma. destruct (row COL_FRAME_MOVE_REMAINING_BEFORE =? 0) eqn:H; [ | discriminate].
    apply Z.eqb_eq. exact H. }
  rewrite Hremaining in Hframe. cbn in Hframe.
  exists f. split; [exact (frame_at_distance _ _ _ _ _ _ _ _ _ Hframe) | ].
  split; [exact Hoffset | ]. split; [exact Hsp | ]. split; [exact Hpc | ].
  split; [ | reflexivity]. cbn [beforeBoundary initializationIdle boundaryZeroRemaining boundaryMoveRemaining
    boundaryZeroActive boundaryMoveActive]. repeat split; assumption.
Qed.

(* An idle frame at code is an active state, whatever distance a finished move left. *)
Lemma idle_active_state : forall program s memory boundary f view,
  let position := viewPosition view in
  FrameAt program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0 0 0 f view ->
  (positionOffset position <= length (innerCode f position))%nat ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) ->
  boundaryPC boundary =
    guestEntry f + Z.of_nat (pathOffset (guestBody f) (positionLabels position) (positionOffset position)) ->
  initializationIdle boundary ->
  ActiveState program s memory boundary (atCode view)
    (frameConfig f view (stackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
      (skipn (positionOffset position) (innerCode f position)))).
Proof.
  intros program s memory boundary f view position Hframe Hoffset Hsp Hpc [Hzr [Hmr [Hza Hma]]].
  exists f. rewrite Hmr. cbn [Z.to_nat].
  split; [exact (frame_at_distance _ _ _ _ _ _ _ _ _ Hframe) | ].
  split; [exact Hoffset | ]. split; [exact Hsp | ]. split; [exact Hpc | ].
  split; [unfold initializationFlags; rewrite Hza, Hma, Hzr, Hmr; split; reflexivity | ].
  split; [exact Hzr | ]. split; [lia | reflexivity].
Qed.

(* An ordinary instruction's row selects its own, non-call opcode. *)
Lemma fetched_ordinary_keeps_idle : forall initial final (w : Witness) i,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories w) final ->
  initial programOpcodeId [w COL_PC_BEFORE] = encodedOpcode i -> (exists op, encodedOpcode i = Some op) ->
  initializationIdle (beforeBoundary w) -> initializationIdle (afterBoundary w).
Proof.
  intros initial final w i Hprime Hfull Hprogram Hreplay Hrom [op Hop] Hidle.
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch.
  rewrite Hrom, Hop in Hfetch. injection Hfetch as Hcode.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> ->] | [[-> ->] | [[x [-> ->]] | [x [-> ->]]]]];
  match goal with |- _ =>
    first
    [ assert (Hchosen : w COL_SEL_I32_ADD = 1) by (apply (full_opcode_selects w COL_SEL_I32_ADD CODE_I32_ADD Hprime Hfull
        Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode]); pose (chosen := COL_SEL_I32_ADD)
    | assert (Hchosen : w COL_SEL_I32_SUB = 1) by (apply (full_opcode_selects w COL_SEL_I32_SUB CODE_I32_SUB Hprime Hfull
        Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode]); pose (chosen := COL_SEL_I32_SUB)
    | assert (Hchosen : w COL_SEL_LOCAL_GET = 1) by (apply (full_opcode_selects w COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime
        Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode]); pose (chosen := COL_SEL_LOCAL_GET)
    | assert (Hchosen : w COL_SEL_LOCAL_SET = 1) by (apply (full_opcode_selects w COL_SEL_LOCAL_SET CODE_LOCAL_SET Hprime
        Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode]); pose (chosen := COL_SEL_LOCAL_SET) ]
  end;
  (apply (program_row_keeps_idle w chosen); [exact Hprime | exact Hfull | vm_compute; tauto | exact Hchosen |
    unfold chosen; not_call_selector | | exact Hidle]);
  exact (body_pop w (full_body_frame_sp_drop_idle w chosen Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(unfold chosen; vm_compute; intuition discriminate))).
Qed.

(* A row that writes no stack cell and no call context. *)
Lemma quiet_row_frame_effect : forall initial final row fbp,
  Replay initial (eventsForRow memories row) final ->
  filter isStackWrite (eventsForRow memories row) = [] -> row COL_CALL_STACK_PUSH_PRESENT = 0 ->
  row COL_HOST_SUSPEND = 0 -> row COL_HOST_CONTEXT_READ = 0 -> FrameEffect fbp initial final.
Proof.
  intros initial final row fbp Hreplay Hwrites Hpush Hsuspend Hread. constructor.
  - intros index _. eapply no_stack_writes_preserve; eassumption.
  - intros memory column address Hin. eapply context_preserved_without_push; eassumption.
Qed.

(* An ordinary row under a stack prefix: the prefix is just more of the frame's stack. *)
Lemma framed_row_scope : forall prefix values i z initial final row,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> encodedOpcode i = Some (row COL_OPCODE_CODE) ->
  wf_state z -> wf_instr i -> programTyped (length (stateLocals z)) (length values) [i] ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  row COL_LOCALS_FBP_BEFORE <= row COL_STACK_FRAME_BASE_BEFORE ->
  FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final /\
  frameRegistersPreserved row.
Proof.
  intros prefix values i z initial final row Hprime Hfull Hprogram Hreplay Hcode Hz Hi Htyped Hstack Hlocals Hlayout.
  eapply frame_row_scope with (z := z) (rest := []) (c0 := stackConfig z (prefix ++ values) [i]); try eassumption.
  split.
  - rewrite (stack_matches_height _ _ _ Hstack), length_app, Nat.add_comm.
    destruct (encoded_opcode_cases _ _ Hcode) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
      cbn [programTyped instructionStackHeight] in Htyped |- *;
      destruct (length values) as [ | [ | n]]; cbn in Htyped |- *; tauto.
  - exists (prefix ++ values). split; [exact Hstack | ]. split; [reflexivity | ]. split; [ | exact Hlocals].
    apply stack_config_wf; [exact Hz | exact (proj1 (proj2 (proj2 Hstack))) | constructor; [exact Hi | constructor]].
Qed.

(* The frame's SpecTec state and its memory representation. *)
Lemma function_module_wf : forall program s index f,
  store_FUNCS s = program -> wf_store s -> functionAt program index = Some f -> wf_moduleinst (guestModule f).
Proof.
  intros program s index f Hfuncs Hwf Hf.
  destruct (function_at_sound _ _ _ Hf) as [Hnth _].
  inversion Hwf as [funcs ? ? ? ? ? Hall]; subst. cbn in Hnth.
  pose proof (proj1 (Forall_forall _ _) Hall _ (nth_error_In _ _ Hnth)) as Hfi.
  inversion Hfi; assumption.
Qed.

Lemma frame_state_wf : forall program s index f locals,
  store_FUNCS s = program -> wf_store s -> functionAt program index = Some f ->
  Forall (fun value => 0 <= value < i32Modulus) locals -> wf_state (mk_state s (activeFrame f locals)).
Proof.
  intros program s index f locals Hfuncs Hwf Hf Hlocals. constructor; [exact Hwf | constructor].
  - apply Forall_map. eapply Forall_impl; [ | exact Hlocals].
    intros value Hvalue. apply val_case_0. apply spectec_i32_well_formed. exact Hvalue.
  - eapply function_module_wf; eassumption.
Qed.

Lemma frame_stack_matches : forall program memory boundary fbp pending distance f view,
  FrameAt program memory (boundaryStackBase boundary) fbp pending 0 distance f view ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) ->
  stackMatches memory boundary (viewOperands view).
Proof.
  intros program memory boundary fbp pending distance f view Hframe Hsp.
  pose proof Hframe as [_ [_ [_ [Hfbp [Hbase [Hbounds _]]]]]].
  split; [lia | ]. split; [exact Hsp | ]. split; [exact Hbounds | ].
  exact (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe).
Qed.

Lemma frame_locals_match : forall program memory boundary base pending moved distance f view s,
  FrameAt program memory base (boundaryLocalsBase boundary) pending moved distance f view ->
  localsMatch memory boundary (mk_state s (activeFrame f (viewLocals view))).
Proof.
  intros program memory boundary base pending moved distance f view s
    [_ [_ [_ [_ [_ [_ [_ [_ [Hlb Hlc]]]]]]]]] index v Hv.
  cbn [stateLocals activeFrame LOCALS] in Hv. rewrite nth_error_map in Hv.
  destruct (nth_error (viewLocals view) index) as [value | ] eqn:Hi; cbn in Hv; [injection Hv as <- | discriminate].
  exists value. split; [reflexivity | ]. split; [exact (proj1 (Forall_forall _ _) Hlb _ (nth_error_In _ _ Hi)) | ].
  exact (Hlc _ _ Hi).
Qed.

(* A state whose frame has f's module and locals in memory is a frame of f. *)
Lemma state_locals : forall s f z memory boundary,
  fun_store z = s -> fun_moduleinst z = guestModule f -> length (stateLocals z) = frameLocals f ->
  localsMatch memory boundary z ->
  exists locals, z = mk_state s (activeFrame f locals) /\ length locals = frameLocals f /\
    Forall (fun value => 0 <= value < i32Modulus) locals /\
    forall index value, nth_error locals index = Some value ->
      memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index)] = Some value.
Proof.
  intros s f z memory boundary Hstore Hmodule Hlen Hlocals.
  destruct (read_cells_total (fun index => memory stackId [2 * index])
    (length (stateLocals z)) (boundaryLocalsBase boundary)) as [locals Hread].
  { intros index Hindex. destruct (nth_error (stateLocals z) index) as [v | ] eqn:Hv;
      [ | apply nth_error_None in Hv; lia].
    destruct (Hlocals index v Hv) as [value [_ [Hrange Hcell]]]. exists value. split; [exact Hcell | exact Hrange]. }
  destruct (proj1 (read_cells_some _ _ _ _) Hread) as [Hcount [Hbounds Hcells]].
  exists locals. split; [ | split; [lia | split; [exact Hbounds | exact Hcells]]].
  destruct z as [s' [values module]]. cbn in Hstore, Hmodule, Hlen, Hcount |- *. subst s' module.
  unfold activeFrame. f_equal. f_equal.
  apply nth_error_ext. intros index. rewrite nth_error_map.
  destruct (nth_error values index) as [v | ] eqn:Hv.
  - destruct (Hlocals index v Hv) as [value [-> [_ Hcell]]].
    destruct (nth_error locals index) as [w | ] eqn:Hw.
    + rewrite (Hcells _ _ Hw) in Hcell. injection Hcell as ->. reflexivity.
    + apply nth_error_None in Hw. assert (index < length values)%nat by (apply nth_error_Some; rewrite Hv; discriminate).
      lia.
  - apply nth_error_None in Hv. replace (nth_error locals index) with (@None Z); [reflexivity | ].
    symmetry. apply nth_error_None. lia.
Qed.

(* Heights and addresses one instruction further along a path. *)
Lemma path_heights_step : forall callees labels code inner offset i heights h h',
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  pathHeights callees code labels offset = Some (heights, h) -> sourceHeight callees i h = Some h' ->
  pathHeights callees code labels (S offset) = Some (heights, h').
Proof.
  intros callees. induction labels as [ | b rest IH];
    intros code inner offset i heights h h' Hpath Hi Hheights Hstep; cbn [pathCode pathHeights] in *.
  - injection Hpath as <-. rewrite (firstn_at_instruction _ _ _ Hi), source_heights_app.
    destruct (sourceHeights callees (firstn offset code) 0) as [h0 | ]; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-. cbn [sourceHeights]. rewrite Hstep. reflexivity.
  - destruct (nth_error code b) as [j | ]; [ | discriminate]. destruct j; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0); [ | discriminate].
    match goal with Hp : pathCode ?nested rest = Some inner |- _ =>
      destruct (pathHeights callees nested rest offset) as [[hs t] | ] eqn:Hin; cbn in Hheights; [ | discriminate];
      injection Hheights as <- <-; rewrite (IH _ _ _ _ _ _ _ Hp Hi Hin Hstep); reflexivity end.
Qed.

Lemma path_offset_step : forall labels code inner offset i,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  pathOffset code labels (S offset) = (pathOffset code labels offset + instructionSize i)%nat.
Proof.
  intros labels code inner offset i Hpath Hi.
  rewrite (path_offset_shift _ _ _ (S offset) Hpath), (path_offset_shift _ _ _ offset Hpath).
  rewrite (code_offset_next _ _ _ Hi). lia.
Qed.

Lemma ordinary_facts : forall localCount resultCount arities callees i h,
  (exists op, encodedOpcode i = Some op) -> instrFacts localCount resultCount arities callees i h ->
  wf_instr i /\ programTyped localCount h [i] /\ sourceHeight callees i h = instructionStackHeight i h.
Proof.
  intros localCount resultCount arities callees i h [op Hop] H.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    destruct H as [Hw Ht].
  all: split; [exact Hw | split; [exact Ht | reflexivity]].
Qed.

Lemma instruction_suffix_steps : forall z z' values values' i rest,
  wf_state z -> wf_state z' -> Forall (fun v => 0 <= v < i32Modulus) values ->
  Forall (fun v => 0 <= v < i32Modulus) values' -> wf_instr i -> Forall wf_instr rest ->
  Step (stackConfig z values [i]) (stackConfig z' values' []) ->
  Steps (stackConfig z values (i :: rest)) (stackConfig z' values' rest).
Proof.
  intros z z' values values' i rest Hz Hz' Hv Hv' Hi Hrest Hstep.
  pose proof (one_spec_step _ _ (stack_config_wf _ _ _ Hz Hv ltac:(constructor; [exact Hi | constructor]))
    (stack_config_wf _ _ _ Hz' Hv' (Forall_nil _)) Hstep) as Hone.
  exact (stack_suffix_steps z z' values values' [i] rest Hone Hrest).
Qed.

(* Paths with their innermost label split off. *)
Lemma path_heights_prefix : forall callees labels code b offset heights top,
  pathHeights callees code (labels ++ [b]) offset = Some (heights, top) ->
  exists inner bt body heights' below,
    pathCode code labels = Some inner /\ nth_error inner b = Some (BLOCK bt body) /\
    pathHeights callees code labels b = Some (heights', below) /\ heights = heights' ++ [below] /\
    sourceHeights callees (firstn offset body) 0 = Some top.
Proof.
  intros callees. induction labels as [ | b' rest IH]; intros code b offset heights top H; cbn [app pathHeights] in H.
  - destruct (nth_error code b) as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    cbn [pathHeights] in H.
    match goal with Hblock : nth_error code b = Some (BLOCK ?bt ?body) |- _ =>
      destruct (sourceHeights callees (firstn offset body) 0) as [t | ] eqn:Ht; cbn in H; [ | discriminate];
      injection H as <- <-; exists code, bt, body, [], below end.
    cbn [pathCode pathHeights]. rewrite Hbelow. repeat split; assumption || reflexivity.
  - destruct (nth_error code b') as [i | ] eqn:Hb; [ | discriminate]. destruct i; try discriminate.
    destruct (sourceHeights callees (firstn b' code) 0) as [below | ] eqn:Hbelow; [ | discriminate].
    match goal with H : option_map _ (pathHeights callees ?nested (rest ++ [b]) offset) = _ |- _ =>
      destruct (pathHeights callees nested (rest ++ [b]) offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate];
      injection H as <- <-;
      destruct (IH _ _ _ _ _ Hin) as [inner [bt [body [hs' [below' [Hpath [Hblock [Hheights [-> Htop]]]]]]]]];
      exists inner, bt, body, (below :: hs'), below' end.
    cbn [pathCode pathHeights]. rewrite Hb, Hbelow, Hheights. repeat split; assumption || reflexivity.
Qed.

Lemma lists_split_last : forall (A B : Type) (xs : list A) (ys : list B),
  length xs = length ys -> xs <> [] ->
  exists xs' x ys' y, xs = xs' ++ [x] /\ ys = ys' ++ [y] /\ length xs' = length ys'.
Proof.
  intros A B xs ys Hlen Hne.
  destruct (exists_last Hne) as [xs' [x ->]].
  assert (Hys : ys <> []) by (intros ->; rewrite length_app in Hlen; cbn in Hlen; lia).
  destruct (exists_last Hys) as [ys' [y ->]].
  exists xs', x, ys', y. rewrite !length_app in Hlen. cbn in Hlen. repeat split; lia.
Qed.

Lemma stack_config_split : forall z prefix top i rest,
  stackConfig z (prefix ++ top) (i :: rest) =
    stackPrefix prefix (instructionSuffix rest (stackConfig z top [i])).
Proof. intros. unfold stackConfig, stackPrefix, instructionSuffix. rewrite map_app, <- !app_assoc. reflexivity. Qed.

Lemma block_instruction_size : forall bt body, instructionSize (BLOCK bt body) = (codeSize body + 2)%nat.
Proof. intros. unfold instructionSize. rewrite flatten_block. cbn [length]. rewrite length_app, block_code_length. cbn. lia. Qed.

Lemma exit_offset : forall code labels inner b bt body,
  pathCode code labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  (pathOffset code (labels ++ [b]) (length body) + 1 = pathOffset code labels (S b))%nat.
Proof.
  intros code labels inner b bt body Hpath Hb.
  rewrite (path_offset_snoc _ _ _ _ _ _ _ Hpath Hb), (path_offset_step _ _ _ _ _ Hpath Hb), block_instruction_size,
    code_offset_end. lia.
Qed.

Lemma list_split_at : forall (A : Type) (xs : list A) k,
  (k < length xs)%nat -> exists pre x post, xs = pre ++ x :: post /\ length pre = k.
Proof.
  intros A xs k Hk. destruct (nth_error xs k) as [x | ] eqn:Hx; [ | apply nth_error_None in Hx; lia].
  exists (firstn k xs), x, (skipn (S k) xs). split; [ | rewrite length_firstn; lia].
  rewrite <- (firstn_skipn k xs) at 1. f_equal. clear Hk. revert k Hx.
  induction xs as [ | y ys IH]; intros [ | k] Hx; try discriminate; cbn in Hx |- *;
    [injection Hx as ->; reflexivity | exact (IH k Hx)].
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

Lemma split_top_values : forall (values : list Z) n,
  (n <= length values)%nat -> exists prefix top, values = prefix ++ top /\ length top = n.
Proof.
  intros values n H. exists (firstn (length values - n) values), (skipn (length values - n) values).
  split; [symmetry; apply firstn_skipn | rewrite length_skipn; lia].
Qed.

Section Rows.
Variables (program : Program) (s : store).
Hypothesis Hstore : store_FUNCS s = program.
Hypothesis Hwf : wf_store s.
Hypothesis Hvalid : programValidated program s.

(* The frame after a row: the same function and labels, at the given position. *)
Definition RowResult (initial final : MemoryState) (row : Witness) (f : GuestFunction) (inner : config)
    (view : FrameView) : Prop :=
  exists view' inner',
    functionAt program (positionFunction (viewPosition view')) = Some f /\
    Steps inner inner' /\ ActiveState program s final (afterBoundary row) (atCode view') inner' /\
    frameRegistersPreserved row /\
    FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final.

Lemma code_ordinary_row : forall view f i initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  (exists op, encodedOpcode i = Some op) ->
  RowResult initial final row f inner view.
Proof.
  intros view f i initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi Hop.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights _]]]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals]. cbn [viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset] in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [_ [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts. cbn [codeFacts] in Hfacts.
  destruct Hfacts as [Hinstr Hrest]. rewrite Nat.add_0_r in Hinstr, Hrest.
  destruct (ordinary_facts _ _ _ _ _ _ Hop Hinstr) as [Hwi [Htyped Hsource]].
  pose proof (encoded_path_ordinary _ _ _ _ Hloaded Hf _ _ _ _ Hpath Hi Hop) as Himage.
  unfold pathPC in Himage. change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  pose proof Hframe as [_ [_ [_ [Hfbp0 [Hbase0 [Hbounds [_ [Hllen0 [Hlb0 _]]]]]]]]].
  cbn [viewLocals] in Hllen0, Hlb0.
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb0).
  pose proof (frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack. unfold viewOperands in Hstack.
  cbn [viewSegments viewValues] in Hstack.
  pose proof (frame_locals_match _ _ (beforeBoundary row) _ _ _ _ _ _ s Hframe) as Hlocals. cbn [viewLocals] in Hlocals.
  assert (Htyped' : programTyped (length (stateLocals z)) (length values) [i])
    by (cbn [z stateLocals activeFrame LOCALS]; rewrite length_map, Hllen0; exact Htyped).
  assert (Hlayout : boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
      boundaryStackBase (beforeBoundary row))
    by (cbn [z beforeBoundary boundaryLocalsBase boundaryStackBase stateLocals activeFrame LOCALS];
        rewrite length_map, Hllen0; lia).
  destruct (framed_instruction (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Himage Hz
    Hwi Htyped' Hstack Hlocals Hlayout) as [z' [values' [[Hstep [Hz' [Hstack' [Hlocals' [Hsame Hheight']]]]] [Hnext Hregs]]]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite (proj1 Himage) in Hfetch.
  destruct (framed_row_scope (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Hfetch Hz
    Hwi Htyped' Hstack Hlocals ltac:(lia)) as [Heffect _].
  destruct Hsame as [Hstore' [Hmodule' Hlen']].
  destruct (state_locals s f z' final (afterBoundary row) ltac:(rewrite Hstore'; reflexivity)
    ltac:(rewrite Hmodule'; reflexivity) ltac:(rewrite Hlen'; cbn [z stateLocals activeFrame LOCALS];
      rewrite length_map; exact Hllen0) Hlocals')
    as [locals' [-> [Hllen [Hlb Hlc]]]].
  pose proof (proj1 (proj2 (proj2 Hstack'))) as Hbounds'. apply Forall_app in Hbounds'.
  exists (frameView (codePosition index labels (S offset)) segments values' locals'),
    (frameConfig f (frameView (codePosition index labels (S offset)) segments values' locals')
      (stackConfig (mk_state s (activeFrame f locals')) values' (skipn (S offset) code))).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    apply steps_in_labels; [exact Hready | ]. rewrite (skipn_at_instruction _ _ _ Hi).
    apply instruction_suffix_steps; [exact Hz | exact Hz' | exact (proj2 (proj1 (Forall_app _ _ _) (proj1 (proj2 (proj2 Hstack)))))
      | exact (proj2 Hbounds') | exact Hwi | | exact Hstep].
    apply forall_skipn. exact Hwfcode.
  - split; [ | split; [exact Hregs | exact Heffect]].
    destruct Hregs as [Hdepth [Hbase Hfbp]].
    apply idle_active_state.
    + apply frame_at_unmoved_intro;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands].
      * exact Hf.
      * exact Hpath.
      * assert (Hstep' : sourceHeight (programCallees program (guestModule f)) i (length values + 0) =
            Some (length values' + 0)%nat) by (rewrite !Nat.add_0_r, Hsource; exact Hheight').
        exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights Hstep').
      * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
      * cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
      * exact (proj1 (proj2 (proj2 Hstack'))).
      * intros index' value Hv. exact (proj2 (proj2 (proj2 Hstack')) _ _ Hv).
      * exact Hllen.
      * exact Hlb.
      * intros index' value Hv. pose proof (Hlc _ _ Hv) as H. cbn [afterBoundary boundaryLocalsBase] in H. exact H.
    + cbn [viewPosition positionOffset]. change (innerCode f (codePosition index labels (S offset))) with code.
      assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate). lia.
    + exact (proj1 (proj2 Hstack')).
    + cbn [afterBoundary boundaryPC viewPosition positionLabels positionOffset]. rewrite Hnext, Hpc,
        (path_offset_step _ _ _ _ _ Hpath Hi), (ordinary_instruction_size _ Hop); lia.
    + exact (fetched_ordinary_keeps_idle _ _ _ _ Hprime Hfull Hprogram Hreplay (proj1 Himage) Hop Hidle).
Qed.

(* Leaving the innermost label: its operands are the frame's innermost values again,
   and execution resumes after its END. *)
Lemma exit_label_state : forall memory boundary index labels b segments segment locals f inner bt body,
  functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels (S b) =
    Some (map (@length Z) segments, length segment) ->
  0 <= boundaryLocalsBase boundary ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (frameLocals f) ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segments ++ segment) ->
  (forall index' value, nth_error (concat segments ++ segment) index' = Some value ->
    memory stackId [2 * (boundaryStackBase boundary + Z.of_nat index')] = Some value) ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (concat segments ++ segment)) ->
  length locals = frameLocals f -> Forall (fun value => 0 <= value < i32Modulus) locals ->
  (forall index' value, nth_error locals index' = Some value ->
    memory stackId [2 * (boundaryLocalsBase boundary + Z.of_nat index')] = Some value) ->
  boundaryPC boundary = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels (S b)) ->
  initializationIdle boundary ->
  ActiveState program s memory boundary (atCode (frameView (codePosition index labels (S b)) segments segment locals))
    (frameConfig f (frameView (codePosition index labels (S b)) segments segment locals)
      (stackConfig (mk_state s (activeFrame f locals)) segment (skipn (S b) inner))).
Proof.
  intros memory boundary index labels b segments segment locals f inner bt body Hf Hpath Hb Hheights Hfbp Hbase
    Hbounds Hcells Hsp Hllen Hlb Hlc Hpc Hidle.
  assert (Hinner : innerCode f (codePosition index labels (S b)) = inner)
    by (unfold innerCode; cbn [positionLabels]; rewrite Hpath; reflexivity).
  pose proof (idle_active_state program s memory boundary f
    (frameView (codePosition index labels (S b)) segments segment locals)) as Hstate.
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hstate.
  rewrite Hinner in Hstate. apply Hstate; clear Hstate.
  - apply frame_at_unmoved_intro;
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
      try rewrite Hinner; try assumption.
    rewrite Nat.add_0_r. exact Hheights.
  - assert (b < length inner)%nat by (apply nth_error_Some; rewrite Hb; discriminate). lia.
  - exact Hsp.
  - exact Hpc.
  - exact Hidle.
Qed.

Lemma code_block_enter_row : forall view f bt body initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (BLOCK bt body) ->
  RowResult initial final row f inner view.
Proof.
  intros view f bt body initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hi.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup. cbn [codeFacts] in Hfacts. destruct Hfacts as [Hblock _].
  destruct (proj1 (block_facts _ _ _ _ _ _ _) Hblock) as [Hwb [Hzero Hbody]].
  inversion Hsup as [ | ? ? Hsupported _]; subst.
  inversion Hsupported as [ | | | | | result ? _ Hresult | | ]; subst.
  assert (Hwbody : Forall wf_instr body) by (inversion Hwb; assumption).
  pose proof (encoded_path_block_entry _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (BlockTrace.control_row _ _ row _ CODE_BLOCK COL_SEL_BLOCK _ Hprime Hfull Hprogram Hreplay Logic.eq_refl
    ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage) as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row COL_SEL_BLOCK Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ Hread]]]].
  destruct (full_block_selector_quiet row COL_SEL_BLOCK Hprime Hfull (or_introl Logic.eq_refl) Hchosen)
    as [Hpush Hpop].
  destruct (full_block_context row COL_SEL_BLOCK Hprime Hfull (or_introl Logic.eq_refl) Hchosen Hpop Hhalt)
    as [Hsp' [Hbase' [Hfbp' [Hdepth' Hwrites]]]].
  pose proof (program_row_keeps_idle row COL_SEL_BLOCK Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hpop Hidle) as HidleAfter.
  pose proof (no_stack_writes_preserve _ _ _ Hreplay Hwrites) as HstackMem.
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  assert (Hpath' : pathCode (guestBody f) (labels ++ [offset]) = Some body)
    by (rewrite path_code_snoc, Hpath, Hi; reflexivity).
  assert (Hinner' : innerCode f (codePosition index (labels ++ [offset]) 0) = body)
    by (unfold innerCode; cbn [positionLabels]; rewrite Hpath'; reflexivity).
  exists (frameView (codePosition index (labels ++ [offset]) 0) (segments ++ [values]) [] locals),
    (frameConfig f (frameView (codePosition index (labels ++ [offset]) 0) (segments ++ [values]) [] locals)
      (stackConfig z [] body)).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  apply Forall_app in Hvb.
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    assert (Hlen : length segments = length labels)
      by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
    rewrite (wrap_labels_snoc _ _ _ _ _ _ segments values _ Hpath Hi Hlen).
    apply steps_in_labels; [exact Hready | ].
    rewrite (skipn_at_instruction _ _ _ Hi), <- (app_nil_r values), stack_config_split, app_nil_r.
    apply steps_with_stack_prefix; [exact (proj2 Hvb) | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | ].
    exact (enter_block z result body Hresult Hz Hwbody).
  - split; [ | split; [split; [exact Hdepth' | split; assumption] |
      eapply quiet_row_frame_effect; eassumption]].
    pose proof (idle_active_state program s final (afterBoundary row) f
      (frameView (codePosition index (labels ++ [offset]) 0) (segments ++ [values]) [] locals)) as Hafter.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
    rewrite Hinner' in Hafter. apply Hafter; clear Hafter.
    + apply frame_at_unmoved_intro;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
        unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner'.
      * exact Hf.
      * exact Hpath'.
      * rewrite (path_heights_snoc _ _ _ _ _ _ _ _ _ 0 Hpath Hi Hheights), map_app.
        cbn [firstn sourceHeights option_map map length]. rewrite Nat.add_0_r. reflexivity.
      * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
      * cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
      * rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hbounds.
      * rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
        intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
        exact (Hcells _ _ Hv).
      * exact Hllen.
      * exact Hlb.
      * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp', HstackMem. exact (Hlc _ _ Hv).
    + cbn. lia.
    + unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
      cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. rewrite Hsp', Hbase'. exact Hsp.
    + cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath Hi).
      unfold codeOffset. cbn. lia.
    + exact HidleAfter.
Qed.

(* The innermost label of a frame at a path, split off. *)
Lemma innermost_label : forall f index labels offset segments values locals pending moved distance memory base fbp,
  FrameAt program memory base fbp pending moved distance f
    (frameView (codePosition index labels offset) segments values locals) ->
  labels <> [] ->
  exists labels' b segments' segment outer bt,
    labels = labels' ++ [b] /\ segments = segments' ++ [segment] /\ length segments' = length labels' /\
    pathCode (guestBody f) labels' = Some outer /\
    nth_error outer b = Some (BLOCK bt (innerCode f (codePosition index labels offset))) /\
    pathHeights (programCallees program (guestModule f)) (guestBody f) labels' b =
      Some (map (@length Z) segments', length segment) /\
    sourceHeights (programCallees program (guestModule f)) (firstn offset (innerCode f (codePosition index labels offset))) 0 =
      Some (length values + pending)%nat.
Proof.
  intros f index labels offset segments values locals pending moved distance memory base fbp Hframe Hne.
  pose proof Hframe as [_ [Hpath [Hheights _]]].
  cbn [viewPosition viewSegments viewValues positionLabels positionOffset] in Hpath, Hheights.
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  destruct (lists_split_last _ _ labels segments ltac:(lia) Hne) as [labels' [b [segments' [segment [-> [-> Hlen']]]]]].
  destruct (path_heights_prefix _ _ _ _ _ _ _ Hheights) as [outer [bt [body [hs' [below [Hpath' [Hb [Hheights' [Hhs Htop]]]]]]]]].
  rewrite path_code_snoc, Hpath', Hb in Hpath. injection Hpath as Hbody.
  rewrite map_app in Hhs. cbn [map] in Hhs. apply app_inj_tail in Hhs. destruct Hhs as [Hhs' Hbelow].
  exists labels', b, segments', segment, outer, bt.
  split; [reflexivity | ]. split; [reflexivity | ]. split; [lia | ]. split; [exact Hpath' | ].
  rewrite <- Hbody. split; [exact Hb | ]. split; [rewrite Hheights', Hhs', Hbelow; reflexivity | exact Htop].
Qed.

Lemma code_block_end_row : forall view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) <> [] ->
  positionOffset (viewPosition view) = length (innerCode f (viewPosition view)) ->
  RowResult initial final row f inner view.
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hne Hend.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [Hblock _]]].
  cbn [viewPosition positionLabels] in Hblock. pose proof (Hblock Hne) as Hzero.
  destruct (innermost_label _ _ _ _ _ _ _ _ _ _ _ _ _ Hframe Hne)
    as [labels' [b [segments' [segment [outer [bt [-> [-> [Hlen' [Hpath' [Hb [Hheights' Htop]]]]]]]]]]]].
  set (body := innerCode f (codePosition index (labels' ++ [b]) offset)) in *.
  (* The block's values are its results. *)
  rewrite Hend, firstn_all in Htop. pose proof (Hzero _ Htop) as Hvalues.
  rewrite (path_arities_app _ [] _ _ _ _ _ Hpath' Hb), last_last, Nat.add_0_r in Hvalues.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts _].
  pose proof (code_facts_wf _ _ _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite concat_app in Hvb. cbn [concat] in Hvb. rewrite app_nil_r, <- app_assoc in Hvb.
  pose proof (proj2 (proj1 (Forall_app _ _ _) Hvb)) as Hinner. apply Forall_app in Hinner as [Hsegment Hresults].
  pose proof (proj1 (proj1 (Forall_app _ _ _) Hvb)) as Houter.
  assert (Hready' : LabelsReady (guestBody f) labels' segments')
    by exact (labels_ready _ _ _ _ Hwfbody Hpath' Hlen' Houter).
  pose proof (encoded_path_block_end _ _ _ _ Hloaded Hf (labels' ++ [b]) body Hne Hpath) as Himage.
  unfold pathPC in Himage. rewrite <- Hend in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (labels' ++ [b]) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  pose proof (control_row_static_edge _ _ row _ _ _ Hprogram Hreplay Logic.eq_refl Himage) as Hedge.
  destruct (BlockTrace.control_row _ _ row _ CODE_END COL_SEL_END _ Hprime Hfull Hprogram Hreplay Logic.eq_refl
    ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage) as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ Hread]]]].
  destruct (full_block_end_quiet row Hprime Hfull Hchosen Hedge Hhalt) as [Hpush Hpop].
  destruct (full_block_context row COL_SEL_END Hprime Hfull (or_intror Logic.eq_refl) Hchosen Hpop Hhalt)
    as [Hsp' [Hbase' [Hfbp' [Hdepth' Hwrites]]]].
  pose proof (program_row_keeps_idle row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hpop Hidle) as HidleAfter.
  pose proof (no_stack_writes_preserve _ _ _ Hreplay Hwrites) as HstackMem.
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  exists (frameView (codePosition index labels' (S b)) segments' (segment ++ values) locals),
    (frameConfig f (frameView (codePosition index labels' (S b)) segments' (segment ++ values) locals)
      (stackConfig z (segment ++ values) (skipn (S b) outer))).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    rewrite Hend, skipn_all, (wrap_labels_snoc _ _ _ _ _ _ segments' segment _ Hpath' Hb Hlen').
    apply steps_in_labels; [exact Hready' | ].
    rewrite <- stack_prefix_config, <- (app_nil_l (skipn (S b) outer)), <- instruction_suffix_config.
    apply steps_with_stack_prefix; [exact Hsegment | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
    rewrite <- Hvalues. exact (exit_block z values Hz Hresults).
  - split; [ | split; [split; [exact Hdepth' | split; assumption] | eapply quiet_row_frame_effect; eassumption]].
    apply (exit_label_state _ _ _ _ _ _ _ _ _ _ _ _ Hf Hpath' Hb).
    + rewrite length_app, Hvalues. exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath' Hb Hheights' Logic.eq_refl).
    + cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
    + cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
    + exact Hvb.
    + intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
      apply Hcells. unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app.
      cbn [concat]. rewrite app_nil_r, <- app_assoc. exact Hv.
    + unfold viewOperands in Hsp. cbn [viewSegments viewValues] in Hsp. rewrite concat_app in Hsp.
      cbn [concat] in Hsp. rewrite app_nil_r, <- app_assoc in Hsp.
      cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. normalize_columns. lia.
    + exact Hllen.
    + exact Hlb.
    + intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp', HstackMem. exact (Hlc _ _ Hv).
    + cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, Hend, <- (exit_offset _ _ _ _ _ _ Hpath' Hb), Nat2Z.inj_add.
      lia.
    + exact HidleAfter.
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
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  destruct (list_split_at _ segments (length segments - S depth) ltac:(lia)) as [segsHead [segT [segsTail [-> HsegHead]]]].
  rewrite !length_app in Hlen, Hexits, HsegHead. cbn [length] in Hlen, Hexits, HsegHead.
  destruct (list_split_at _ labels (length segsHead) ltac:(lia)) as [pre [b [post [-> Hpre]]]].
  rewrite !length_app in Hlen. cbn [length] in Hlen.
  assert (Hpost : length post = depth) by lia.
  assert (HtailLen : length segsTail = depth) by lia.
  destruct (path_heights_split _ _ _ _ _ _ _ _ Hheights)
    as [outer [bt [body [hsHead' [below [hsTail' [Hpath' [Hb [Hheights' [Hhs' HheightsTail]]]]]]]]]].
  pose proof (path_heights_length _ _ _ _ _ _ Hheights') as HheadLen.
  rewrite map_app in Hhs'. cbn [map] in Hhs'.
  destruct (app_split_eq _ (map (@length Z) segsHead) _ hsHead' _ ltac:(rewrite length_map; lia) Hhs') as [Hhead' Htail''].
  injection Htail'' as Hbelow _.
  rewrite <- Hhead', <- Hbelow in Hheights'.
  pose proof Hpath as Hinside. rewrite path_code_app, Hpath' in Hinside. cbn [pathCode] in Hinside.
  rewrite Hb in Hinside.
  (* The target's arity: validation puts its results below the condition. *)
  assert (HtargetArity : nth_error (rev (pathArities (guestBody f) (pre ++ b :: post)) ++ [guestResults f]) depth =
      Some (blockResults bt)).
  { assert (Htail : length (pathArities body post) = depth)
      by (rewrite (path_arities_length _ _ _ _ _ _ HheightsTail), (path_heights_length _ _ _ _ _ _ HheightsTail);
          exact Hpost).
    rewrite (path_arities_app _ _ _ _ _ _ _ Hpath' Hb), rev_app_distr. cbn [rev]. rewrite <- !app_assoc.
    rewrite nth_error_app2 by (rewrite length_rev; lia). rewrite length_rev, Htail, Nat.sub_diag. reflexivity. }
  rewrite HtargetArity in Harity. injection Harity as <-.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts _].
  pose proof (code_facts_wf _ _ _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  assert (Hwfblock : Forall wf_instr body).
  { pose proof (proj1 (Forall_forall _ _) Hwfouter _ (nth_error_In _ _ Hb)) as H. inversion H; assumption. }
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite concat_app in Hvb. cbn [concat] in Hvb.
  apply Forall_app in Hvb as [Hvb Hvalues]. apply Forall_app in Hvalues as [Hunder Hcond]. apply Forall_inv in Hcond.
  apply Forall_app in Hvb as [Hhead Hvb]. apply Forall_app in Hvb as [Hsegment Htail].
  assert (HreadyHead : LabelsReady (guestBody f) pre segsHead)
    by exact (labels_ready pre _ _ segsHead Hwfbody Hpath' ltac:(lia) Hhead).
  assert (HreadyTail : LabelsReady body post segsTail)
    by (apply (labels_ready _ _ _ _ Hwfblock Hinside); [lia | exact Htail]).
  (* The circuit: pop the condition, then fall through or slide past the target. *)
  pose proof (encoded_path_branch _ _ _ _ Hloaded Hf (pre ++ b :: post) code offset l pre b post Hpath Hi
    Logic.eq_refl (Logic.eq_sym Hpost)) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (pre ++ b :: post) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (circuit_br_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram Logic.eq_refl Himage) as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  pose proof (br_if_depth_preserved row Hprime Hfull Hchosen) as Hdepth.
  destruct (full_block_selector_quiet row COL_SEL_BR_IF_EQZ Hprime Hfull (or_intror Logic.eq_refl) Hchosen)
    as [Hpush Hpop].
  pose proof (frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack.
  (* Falling through slides nothing. Taken, the slide keeps the target's results
     and drops the operands below them, down to the target's base. *)
  set (distance := (length under - blockResults bt + length (concat segsTail))%nat).
  assert (HslideAfter : row COL_FRAME_MOVE_REMAINING_AFTER =
        (if Z.eqb (row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) 0 then 0
         else slideWord true (blockResults bt) distance) /\
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
      assert (Hcompute : branchSlide hs ars top target = Some (blockResults bt, distance));
      [replace target with depth by reflexivity | rewrite Hcompute in Hkept, Hrom] end.
    { rewrite length_app, Nat.add_0_r. cbn [length]. rewrite Nat.add_1_r. unfold branchSlide, labelSlide.
      rewrite HtargetArity. replace (Nat.leb (blockResults bt) (length under)) with true
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
  pose proof (full_br_if_frame_unchanged row Hprime Hfull Hchosen) as Hfbp.
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
    destruct (split_top_values under (blockResults bt) ltac:(lia)) as [dropped [kept [-> Hlength]]].
    apply Forall_app in Hunder as [Hdropped Hkept].
    destruct (wrap_labels_exit post body _ segsTail z (dropped ++ kept) 0 _ Hinside ltac:(lia) HreadyTail Hz
      (proj2 (Forall_app _ _ _) (conj Htail (proj2 (Forall_app _ _ _) (conj Hdropped Hkept)))) Hwfl Hwfrest)
      as [rest' [Hrest' Hexit]].
    set (resumed := frameView (codePosition index pre (S b)) segsHead (segT ++ kept) locals).
    assert (HinnerPre : innerCode f (codePosition index pre (S b)) = outer)
      by (unfold innerCode; cbn [positionLabels]; rewrite Hpath'; reflexivity).
    exists resumed, (frameConfig f resumed (stackConfig z (segT ++ kept) (skipn (S b) outer))).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + rewrite Hbefore, (wrap_labels_app pre (b :: post) _ _ segsHead (segT :: segsTail) _ Hpath' ltac:(lia)).
      cbn [wrapLabels]. rewrite Hb.
      unfold frameConfig, resumed. cbn [viewPosition viewSegments positionLabels].
      apply steps_in_labels; [exact HreadyHead | ].
      replace (stackConfig z (segT ++ kept) (skipn (S b) outer)) with
        (stackPrefix segT (instructionSuffix (skipn (S b) outer) (stackConfig z kept [])))
        by (rewrite instruction_suffix_config, stack_prefix_config; reflexivity).
      apply steps_with_stack_prefix; [exact Hsegment | ].
      apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
      eapply spec_steps_transitive; [apply steps_in_block; eapply spec_steps_transitive;
        [exact (steps_in_labels _ _ _ _ _ HreadyTail Hsteps) | exact Hexit] | ].
      rewrite <- Hlength, app_assoc. apply br_exit_target; [exact Hz | | exact Hrest'].
      rewrite <- app_assoc. apply Forall_app. split; [exact Htail | apply Forall_app; split; assumption].
    + split; [ | split; [exact Hregs | exact Heffect]].
      cbn [beforeBoundary boundarySP boundaryStackBase] in Hsp.
      set (L := length (concat segsHead ++ segT)).
      (* Below the kept results nothing moves; they sit `distance` cells higher. *)
      assert (Hcell : forall n i, n = (L + length kept)%nat ->
          operandCell (row COL_STACK_FRAME_BASE_BEFORE) n (Z.to_nat (row COL_FRAME_MOVE_REMAINING_AFTER))
            (row COL_FRAME_MOVE_DISTANCE_AFTER) i =
          row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat i + (if (i <? L)%nat then 0 else Z.of_nat distance)).
      { intros n' i ->. unfold operandCell. rewrite HkeptAfter, HdistAfter. unfold slideWord.
        destruct (Nat.eqb distance 0) eqn:Hd0.
        - apply Nat.eqb_eq in Hd0. rewrite Hd0. cbn [Z.to_nat Z.of_nat].
          destruct (_ <? _)%nat, (_ <? _)%nat; lia.
        - replace (Z.to_nat (Z.of_nat (blockResults bt))) with (length kept) by lia.
          replace (L + length kept - length kept)%nat with L by lia. reflexivity. }
      assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
        by (intros xs ys k; rewrite nth_error_app2 by lia; f_equal; lia).
      exists f. cbn [afterBoundary boundaryMoveRemaining boundaryStackBase boundaryLocalsBase boundaryMoveDistance
        boundarySP boundaryPC boundaryZeroRemaining].
      unfold resumed. cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels
        positionOffset viewOperands]. rewrite HinnerPre.
      rewrite (local_stack_base_after _ _ _ Hshape).
      split; [ | split; [ | split; [ | split; [ | split; [ | split; [ | split; [ | reflexivity]]]]]]].
      * unfold FrameAt, viewOperands. cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels
          positionOffset]. rewrite HinnerPre.
        split; [exact Hf | ]. split; [exact Hpath' | ].
        split; [rewrite length_app, Hlength, Nat.add_0_r;
          exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath' Hb Hheights' Logic.eq_refl) | ].
        split; [cbn [afterBoundary boundaryLocalsBase]; normalize_columns; lia | ].
        split; [cbn [afterBoundary boundaryLocalsBase]; normalize_columns; lia | ].
        split; [rewrite app_assoc; exact (proj2 (Forall_app _ _ _) (conj
          (proj2 (Forall_app _ _ _) (conj Hhead Hsegment)) Hkept)) | ].
        split; [ | split; [exact Hllen | split; [exact Hlb | ]]].
        -- intros i value Hv. rewrite Hcell by (rewrite !length_app; unfold L; rewrite length_app; lia).
           rewrite HstackMem.
           destruct (Nat.lt_ge_cases i L) as [Hlt | Hge].
           ++ replace (i <? L)%nat with true by (symmetry; apply Nat.ltb_lt; lia). rewrite Z.add_0_r.
              apply Hcells. rewrite app_assoc, nth_error_app1 in Hv by (unfold L in Hlt; exact Hlt).
              unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat]. rewrite <- !app_assoc, app_assoc, nth_error_app1 by exact Hlt.
              exact Hv.
           ++ replace (i <? L)%nat with false by (symmetry; apply Nat.ltb_ge; lia).
              rewrite app_assoc in Hv. replace i with (L + (i - L))%nat in Hv by lia.
              unfold L in Hv. rewrite Hshift in Hv. fold L in Hv.
              assert (Hj : (i - L < length kept)%nat) by (apply nth_error_Some; rewrite Hv; discriminate).
              replace (row COL_STACK_FRAME_BASE_BEFORE + Z.of_nat i + Z.of_nat distance) with
                (row COL_STACK_FRAME_BASE_BEFORE +
                  Z.of_nat (L + (length (concat segsTail ++ dropped) + (i - L)))) by
                (unfold distance; rewrite !length_app; lia).
              apply Hcells. unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat].
              replace ((concat segsHead ++ segT ++ concat segsTail) ++ (dropped ++ kept) ++ [condition]) with
                ((concat segsHead ++ segT) ++ (concat segsTail ++ dropped) ++ kept ++ [condition])
                by (rewrite <- !app_assoc; reflexivity).
              unfold L. rewrite !Hshift, nth_error_app1 by exact Hj. exact Hv.
        -- intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HstackMem.
           exact (Hlc _ _ Hv).
      * assert (b < length outer)%nat by (apply nth_error_Some; rewrite Hb; discriminate). lia.
      * unfold viewOperands in Hsp. cbn [viewSegments viewValues] in Hsp. rewrite concat_app in Hsp.
        cbn [concat] in Hsp. rewrite (local_stack_pointer_after _ _ _ Hshape), HdistAfter, Hsp.
        unfold distance, viewOperands. cbn [viewSegments viewValues]. rewrite !length_app. cbn [length]. lia.
      * rewrite Hnext. reflexivity.
      * exact (full_initialization_flags row Hprime Hfull).
      * exact (branch_no_zeroing row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen
          ltac:(vm_compute; tauto) Hidle).
      * rewrite HkeptAfter. unfold slideWord. rewrite length_app. destruct (Nat.eqb distance 0); lia.
Qed.

End Rows.
