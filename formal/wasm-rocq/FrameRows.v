(* Rows of the active frame's code, at any depth of enclosing labels: an ordinary
   instruction, entering a block, a loop or an if, an arm's END or ELSE, and the
   state after leaving a label, which branches share (BranchRows). Each keeps the
   frame registers and touches no memory below the frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues Add R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog SlideLayout
  TraceRefinement StraightLineSpec StackSimulation LocalSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerRefinement ProgramSpec ProgramEncoding CodeLayout BlockSpec BlockBindings BlockEncoding BlockInstructions
  BlockTrace BranchBindings StackBindings CallInitialization Selectors CallScope CallContext CallMemory LocalBindings
  StraightLineInstructions ProgramBinding Fragment CodeFacts ControlDecode ControlInvariant FrameFacts ControlBridge
  CircuitFacts IntegerMemory IntegerDecode.
Import ListNotations.
Open Scope Z_scope.

Ltac not_call_selector :=
  let H := fresh in intro H; vm_compute in H; repeat (destruct H as [H | H]; [discriminate H | ]); exact H.

(* With no move pending, a frame's operands sit at plain offsets from its base,
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
  valuesTyped (guestFrameTypes f) (viewLocals view) ->
  (forall index value, nth_error (viewLocals view) index = Some value ->
    CellMatches memory (fbp + Z.of_nat index) value) ->
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
  - intros index limb _ _. eapply no_stack_writes_preserve; eassumption.
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
  Forall wf_val locals -> wf_state (mk_state s (activeFrame f locals)).
Proof.
  intros program s index f locals Hfuncs Hwf Hf Hlocals. constructor; [exact Hwf | constructor].
  - exact Hlocals.
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

Lemma frame_locals_match : forall program memory boundary base pending remaining distance f view s,
  FrameAt program memory base (boundaryLocalsBase boundary) pending remaining distance f view ->
  localsMatch memory boundary (mk_state s (activeFrame f (viewLocals view))).
Proof.
  intros program memory boundary base pending remaining distance f view s Hframe index v Hv.
  destruct (frame_locals_i32 _ _ _ _ _ _ _ _ _ Hframe) as [raw [Hraw [Hbounds Hcells]]].
  cbn [stateLocals activeFrame LOCALS] in Hv. rewrite Hraw, nth_error_map in Hv.
  destruct (nth_error raw index) as [value | ] eqn:Hi; cbn in Hv; [injection Hv as <- | discriminate].
  exists value. split; [reflexivity | ]. split; [exact (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hi)) | ].
  exact (Hcells _ _ Hi).
Qed.

(* A state whose frame has f's module and locals in memory is a frame of f. *)
Lemma state_locals : forall program index s f z memory boundary,
  functionAt program index = Some f ->
  fun_store z = s -> fun_moduleinst z = guestModule f -> length (stateLocals z) = frameLocals f ->
  localsMatch memory boundary z ->
  exists locals, z = mk_state s (activeFrame f locals) /\ length locals = frameLocals f /\
    valuesTyped (guestFrameTypes f) locals /\
    forall index value, nth_error locals index = Some value ->
      CellMatches memory (boundaryLocalsBase boundary + Z.of_nat index) value.
Proof.
  intros program index s f z memory boundary Hf Hstore Hmodule Hlen Hlocals.
  exists (stateLocals z). split.
  { destruct z as [store [locals module]]. cbn in *. subst. reflexivity. }
  split; [exact Hlen | ]. split.
  - destruct (function_at_i32_types _ _ _ Hf) as [Hp [_ Hl]].
    unfold guestFrameTypes. rewrite Hp, Hl, <- repeat_app. fold (frameLocals f). rewrite <- Hlen.
    apply values_typed_repeat, Forall_forall. intros value Hin.
    apply In_nth_error in Hin as [n Hnth]. destruct (Hlocals _ _ Hnth) as [raw [-> [Hbound _]]].
    split; [reflexivity | apply i32_value_wf; exact Hbound].
  - intros n value Hnth. destruct (Hlocals _ _ Hnth) as [raw [-> [Hbound Hcell]]].
    constructor; assumption.
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
  - destruct (siteEntry code b) as [[j body] | ]; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset b) code) 0); [ | discriminate].
    destruct (pathHeights callees body rest offset) as [[hs t] | ] eqn:Hin; cbn in Hheights; [ | discriminate].
    injection Hheights as <- <-. rewrite (IH _ _ _ _ _ _ _ Hpath Hi Hin Hstep). reflexivity.
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
Lemma path_heights_prefix : forall callees labels code s offset heights top,
  pathHeights callees code (labels ++ [s]) offset = Some (heights, top) ->
  exists inner i body heights' height,
    pathCode code labels = Some inner /\ siteEntry inner s = Some (i, body) /\
    pathHeights callees code labels (siteOffset s) = Some (heights', height) /\
    heights = heights' ++ [labelBase i height] /\
    sourceHeights callees (firstn offset body) 0 = Some top.
Proof.
  intros callees. induction labels as [ | s' rest IH]; intros code s offset heights top H; cbn [app pathHeights] in H.
  - destruct (siteEntry code s) as [[i body] | ] eqn:Hs; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s) code) 0) as [height | ] eqn:Hheight; [ | discriminate].
    cbn [pathHeights] in H.
    destruct (sourceHeights callees (firstn offset body) 0) as [t | ] eqn:Ht; cbn in H; [ | discriminate].
    injection H as <- <-. exists code, i, body, [], height.
    cbn [pathCode pathHeights]. rewrite Hheight. repeat split; assumption || reflexivity.
  - destruct (siteEntry code s') as [[j nested] | ] eqn:Hs'; [ | discriminate].
    destruct (sourceHeights callees (firstn (siteOffset s') code) 0) as [height0 | ] eqn:Hheight0; [ | discriminate].
    destruct (pathHeights callees nested (rest ++ [s]) offset) as [[hs t] | ] eqn:Hin; cbn in H; [ | discriminate].
    injection H as <- <-.
    destruct (IH _ _ _ _ _ Hin) as [inner [i [body [hs' [height [Hpath [Hs [Hheights [-> Htop]]]]]]]]].
    exists inner, i, body, (labelBase j height0 :: hs'), height.
    cbn [pathCode pathHeights]. rewrite Hs', Hheight0, Hheights. repeat split; assumption || reflexivity.
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
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf (values_typed_wf _ _ Hlb0)).
  pose proof (frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack. unfold viewOperands in Hstack.
  cbn [viewSegments viewValues] in Hstack.
  pose proof (frame_locals_match _ _ (beforeBoundary row) _ _ _ _ _ _ s Hframe) as Hlocals. cbn [viewLocals] in Hlocals.
  assert (Htyped' : programTyped (length (stateLocals z)) (length values) [i])
    by (cbn [z stateLocals activeFrame LOCALS]; rewrite Hllen0; exact Htyped).
  assert (Hlayout : boundaryLocalsBase (beforeBoundary row) + Z.of_nat (length (stateLocals z)) <=
      boundaryStackBase (beforeBoundary row))
    by (cbn [z beforeBoundary boundaryLocalsBase boundaryStackBase stateLocals activeFrame LOCALS];
        rewrite Hllen0; lia).
  destruct (framed_instruction (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Himage Hz
    Hwi Htyped' Hstack Hlocals Hlayout) as [z' [values' [[Hstep [Hz' [Hstack' [Hlocals' [Hsame Hheight']]]]] [Hnext Hregs]]]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite (proj1 Himage) in Hfetch.
  destruct (framed_row_scope (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Hfetch Hz
    Hwi Htyped' Hstack Hlocals ltac:(lia)) as [Heffect _].
  destruct Hsame as [Hstore' [Hmodule' Hlen']].
  destruct (state_locals program index s f z' final (afterBoundary row) Hf ltac:(rewrite Hstore'; reflexivity)
    ltac:(rewrite Hmodule'; reflexivity) ltac:(rewrite Hlen'; cbn [z stateLocals activeFrame LOCALS];
      exact Hllen0) Hlocals')
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

(* Leaving a structured instruction's label: execution resumes after the
   instruction at b with the operands below its label, `segment`, then its
   results, `kept`. While a slide's move is pending, the results still sit
   `distance` cells higher. *)
Lemma exit_label_state : forall memory boundary index labels b segments segment kept distance locals f inner,
  functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> (b < length inner)%nat ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels (S b) =
    Some (map (@length Z) segments, length (segment ++ kept)) ->
  0 <= boundaryLocalsBase boundary ->
  boundaryStackBase boundary = boundaryLocalsBase boundary + Z.of_nat (frameLocals f) ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segments ++ segment ++ kept) ->
  (forall index' value, nth_error (concat segments ++ segment) index' = Some value ->
    memory stackId [2 * (boundaryStackBase boundary + Z.of_nat index')] = Some value) ->
  (forall j value, nth_error kept j = Some value ->
    memory stackId [2 * (boundaryStackBase boundary +
      Z.of_nat (length (concat segments ++ segment) + distance + j))] = Some value) ->
  boundaryMoveRemaining boundary = slideWord true (length kept) distance ->
  (distance <> 0%nat -> boundaryMoveDistance boundary = Z.of_nat distance) ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (concat segments ++ segment ++ kept)) ->
  length locals = frameLocals f -> valuesTyped (guestFrameTypes f) locals ->
  (forall index' value, nth_error locals index' = Some value ->
    CellMatches memory (boundaryLocalsBase boundary + Z.of_nat index') value) ->
  boundaryPC boundary = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels (S b)) ->
  initializationFlags boundary -> boundaryZeroRemaining boundary = 0 ->
  ActiveState program s memory boundary
    (atCode (frameView (codePosition index labels (S b)) segments (segment ++ kept) locals))
    (frameConfig f (frameView (codePosition index labels (S b)) segments (segment ++ kept) locals)
      (stackConfig (mk_state s (activeFrame f locals)) (segment ++ kept) (skipn (S b) inner))).
Proof.
  intros memory boundary index labels b segments segment kept distance locals f inner Hf Hpath Hb Hheights
    Hfbp Hbase Hbounds Hcells Hkept Hremaining Hdistance Hsp Hllen Hlb Hlc Hpc Hflags Hzero.
  assert (Hinner : innerCode f (codePosition index labels (S b)) = inner)
    by (unfold innerCode; cbn [positionLabels]; rewrite Hpath; reflexivity).
  set (L := length (concat segments ++ segment)) in *.
  (* Below the results nothing moves. *)
  assert (Hcell : forall i, (i < L + length kept)%nat ->
      operandCell (boundaryStackBase boundary) (L + length kept)
        (Z.to_nat (boundaryMoveRemaining boundary)) (boundaryMoveDistance boundary) i =
      boundaryStackBase boundary + Z.of_nat i + (if (i <? L)%nat then 0 else Z.of_nat distance)).
  { intros i Hi. unfold operandCell. rewrite Hremaining. unfold slideWord.
    destruct (Nat.eqb distance 0) eqn:Hd0.
    - cbv iota. apply Nat.eqb_eq in Hd0. rewrite Hd0. cbn [Z.to_nat Z.of_nat]. rewrite Nat.sub_0_r.
      replace (i <? L + length kept)%nat with true by (symmetry; apply Nat.ltb_lt; exact Hi).
      destruct (i <? L)%nat; reflexivity.
    - cbv iota. apply Nat.eqb_neq in Hd0. rewrite (Hdistance Hd0).
      replace (Z.to_nat (Z.of_nat (length kept))) with (length kept) by lia.
      replace (L + length kept - length kept)%nat with L by lia. reflexivity. }
  assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
    by (intros xs ys k; rewrite nth_error_app2 by lia; f_equal; lia).
  exists f. cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
  rewrite Hinner.
  split; [ | split; [ | split; [ | split; [ | split; [ | split; [ | split; [ | reflexivity]]]]]]].
  - unfold FrameAt, viewOperands. cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels
      positionOffset]. rewrite Hinner.
    split; [exact Hf | ]. split; [exact Hpath | ].
    split; [rewrite Nat.add_0_r; exact Hheights | ].
    split; [exact Hfbp | ]. split; [exact Hbase | ]. split; [exact Hbounds | ].
    split; [ | split; [exact Hllen | split; [exact Hlb | exact Hlc]]].
    intros i value Hv.
    replace (length (concat segments ++ segment ++ kept)) with (L + length kept)%nat
      by (unfold L; rewrite !length_app; lia).
    rewrite Hcell by (assert (i < length (concat segments ++ segment ++ kept))%nat
      by (apply nth_error_Some; rewrite Hv; discriminate); unfold L; rewrite !length_app in *; lia).
    rewrite app_assoc in Hv.
    destruct (Nat.lt_ge_cases i L) as [Hlt | Hge].
    + replace (i <? L)%nat with true by (symmetry; apply Nat.ltb_lt; lia). rewrite Z.add_0_r.
      apply Hcells. rewrite nth_error_app1 in Hv by exact Hlt. exact Hv.
    + replace (i <? L)%nat with false by (symmetry; apply Nat.ltb_ge; lia).
      replace i with (L + (i - L))%nat in Hv |- * by lia. unfold L in Hv. rewrite Hshift in Hv.
      replace (boundaryStackBase boundary + Z.of_nat (L + (i - L)) + Z.of_nat distance) with
        (boundaryStackBase boundary + Z.of_nat (L + distance + (i - L))) by lia.
      exact (Hkept _ _ Hv).
  - lia.
  - unfold viewOperands. cbn [viewSegments viewValues]. exact Hsp.
  - exact Hpc.
  - exact Hflags.
  - exact Hzero.
  - rewrite Hremaining. unfold slideWord. rewrite length_app. destruct (Nat.eqb distance 0); lia.
Qed.

(* The opcodes whose row is a static jump that leaves the stack and the frame as
   they are, with their selectors. *)
Definition quietControl (opcode : Z) (chosen : nat) : Prop :=
  opcode = CODE_END /\ chosen = COL_SEL_END \/ opcode = CODE_ELSE /\ chosen = COL_SEL_ELSE \/
  opcode = CODE_BLOCK /\ chosen = COL_SEL_BLOCK \/ opcode = CODE_LOOP /\ chosen = COL_SEL_LOOP \/
  opcode = CODE_NOP /\ chosen = COL_SEL_NOP.

Lemma quiet_control_row : forall initial final row pc opcode chosen next,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 -> row COL_ACTIVATION_FINISHED_AFTER = 0 ->
  Replay initial (eventsForRow memories row) final -> row COL_PC_BEFORE = pc -> quietControl opcode chosen ->
  controlImage initial pc opcode next -> initializationIdle (beforeBoundary row) ->
  row COL_PC_AFTER = next /\ row COL_SP_AFTER = row COL_SP_BEFORE /\
  row COL_STACK_FRAME_BASE_AFTER = row COL_STACK_FRAME_BASE_BEFORE /\ frameRegistersPreserved row /\
  FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final /\
  (forall address, final stackId address = initial stackId address) /\ initializationIdle (afterBoundary row).
Proof.
  intros initial final row pc opcode chosen next Hprime Hfull Hprogram Hhalt Hreplay Hpc Hkind Himage Hidle.
  assert (Hin : In chosen selectorColumns)
    by (destruct Hkind as [[_ ->] | [[_ ->] | [[_ ->] | [[_ ->] | [_ ->]]]]]; vm_compute; tauto).
  pose proof (control_row_static_edge _ _ row _ _ _ Hprogram Hreplay Hpc Himage) as Hedge.
  assert (Hrow : row chosen = 1 /\ row COL_PC_AFTER = next).
  { destruct Hkind as [[-> ->] | [[-> ->] | [[-> ->] | [[-> ->] | [-> ->]]]]].
    - exact (BlockTrace.control_row _ _ row _ CODE_END COL_SEL_END _ Hprime Hfull Hprogram Hreplay Hpc
        ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage).
    - exact (BlockTrace.control_row _ _ row _ CODE_ELSE COL_SEL_ELSE _ Hprime Hfull Hprogram Hreplay Hpc
        ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage).
    - exact (BlockTrace.control_row _ _ row _ CODE_BLOCK COL_SEL_BLOCK _ Hprime Hfull Hprogram Hreplay Hpc
        ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage).
    - exact (BlockTrace.control_row _ _ row _ CODE_LOOP COL_SEL_LOOP _ Hprime Hfull Hprogram Hreplay Hpc
        ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage).
    - exact (BlockTrace.control_row _ _ row _ CODE_NOP COL_SEL_NOP _ Hprime Hfull Hprogram Hreplay Hpc
        ltac:(vm_compute; reflexivity) ltac:(vm_compute; intuition discriminate) Himage). }
  destruct Hrow as [Hchosen Hnext].
  destruct (full_selected_no_host_switch row chosen Hprime Hfull Hin Hchosen) as [Hsuspend [_ [_ [_ Hread]]]].
  assert (Hquiet : row COL_CALL_STACK_PUSH_PRESENT = 0 /\ row COL_CALL_STACK_POP_PRESENT = 0).
  { destruct Hkind as [[_ ->] | [[_ ->] | [[_ ->] | [[_ ->] | [_ ->]]]]].
    - exact (full_block_end_quiet row Hprime Hfull Hchosen Hedge Hhalt).
    - exact (full_block_selector_quiet row COL_SEL_ELSE Hprime Hfull
        (or_intror (or_intror (or_intror (or_intror (or_introl Logic.eq_refl))))) Hchosen).
    - exact (full_block_selector_quiet row COL_SEL_BLOCK Hprime Hfull (or_introl Logic.eq_refl) Hchosen).
    - exact (full_block_selector_quiet row COL_SEL_LOOP Hprime Hfull
        (or_intror (or_intror (or_introl Logic.eq_refl))) Hchosen).
    - exact (full_block_selector_quiet row COL_SEL_NOP Hprime Hfull
        (or_intror (or_intror (or_intror (or_intror (or_intror Logic.eq_refl))))) Hchosen). }
  destruct Hquiet as [Hpush Hpop].
  destruct (full_block_context row chosen Hprime Hfull
    ltac:(destruct Hkind as [[_ ->] | [[_ ->] | [[_ ->] | [[_ ->] | [_ ->]]]]];
      [right; left | right; right; right; left | left | right; right; left | right; right; right; right];
      reflexivity) Hchosen Hpop Hhalt)
    as [Hsp' [Hbase' [Hfbp' [Hdepth' Hwrites]]]].
  pose proof (program_row_keeps_idle row chosen Hprime Hfull Hin Hchosen
    ltac:(destruct Hkind as [[_ ->] | [[_ ->] | [[_ ->] | [[_ ->] | [_ ->]]]]]; not_call_selector) Hpop Hidle)
    as HidleAfter.
  split; [exact Hnext | ]. split; [exact Hsp' | ]. split; [exact Hbase' | ].
  split; [split; [exact Hdepth' | split; assumption] | ].
  split; [eapply quiet_row_frame_effect; eassumption | ].
  split; [exact (no_stack_writes_preserve _ _ _ Hreplay Hwrites) | exact HidleAfter].
Qed.
(* Entering a block or a loop: its label goes on over the operands, and its body runs. *)
Lemma code_label_enter_row : forall view f i body initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some i ->
  (exists bt, i = BLOCK bt body) \/ (exists bt, i = LOOP bt body) ->
  RowResult initial final row f inner view.
Proof.
  intros view f i body initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hi Hlabel.
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
  rewrite (skipn_at_instruction _ _ _ Hi) in Hsup.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf (values_typed_wf _ _ Hlb)).
  (* The opcode and SpecTec's entry step, for a block or a loop. *)
  assert (Hentry : exists opcode chosen, quietControl opcode chosen /\
      controlImage initial (row COL_PC_BEFORE) opcode (row COL_PC_BEFORE + 1) /\
      instrArm i 0 = Some body /\ (forall h, labelBase i h = h) /\
      Steps (stackConfig z [] [i]) (blockContext (labelArity i) (labelContinuation i) (stackConfig z [] body))).
  { inversion Hsup as [ | ? ? Hsupported _]; subst.
    pose proof (proj1 (Forall_forall _ _) Hwfcode _ (nth_error_In _ _ Hi)) as Hwi.
    destruct Hlabel as [[bt ->] | [bt ->]].
    - inversion Hsupported as [ | | | | | result ? _ Hresult | | | | | | |  | | ]; subst.
      assert (Hwbody : Forall wf_instr body) by (inversion Hwi; assumption).
      pose proof (encoded_path_block_entry _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as Himage.
      unfold pathPC in Himage. rewrite <- Hpc in Himage.
      exists CODE_BLOCK, COL_SEL_BLOCK. split; [right; right; left; split; reflexivity | ].
      split; [exact Himage | ]. split; [reflexivity | ]. split; [reflexivity | exact (enter_block z result body Hresult Hz Hwbody)].
    - inversion Hsupported as [ | | | | | | | | | | | result ? _ Hresult |  | | ]; subst.
      assert (Hwbody : Forall wf_instr body) by (inversion Hwi; assumption).
      pose proof (encoded_path_loop_entry _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath Hi) as Himage.
      unfold pathPC in Himage. rewrite <- Hpc in Himage.
      exists CODE_LOOP, COL_SEL_LOOP. split; [right; right; right; left; split; reflexivity | ].
      split; [exact Himage | ]. split; [reflexivity | ]. split; [reflexivity | exact (enter_loop z result body Hresult Hz Hwbody)]. }
  destruct Hentry as [opcode [chosen [Hkind [Himage [Harm [HlabelBase Hsteps]]]]]].
  destruct (quiet_control_row _ _ row _ _ chosen _ Hprime Hfull Hprogram Hhalt Hreplay Logic.eq_refl Hkind Himage
    Hidle) as [Hnext [Hsp' [Hbase' [Hregs [Heffect [HstackMem HidleAfter]]]]]].
  pose proof Hregs as [_ [_ Hfbp']].
  assert (Hsite : siteEntry code (labelSite offset 0) = Some (i, body))
    by (unfold siteEntry; cbn [siteOffset siteArm]; rewrite Hi, Harm; reflexivity).
  assert (Hpath' : pathCode (guestBody f) (labels ++ [labelSite offset 0]) = Some body)
    by (rewrite path_code_snoc, Hpath, Hsite; reflexivity).
  assert (Hinner' : innerCode f (codePosition index (labels ++ [labelSite offset 0]) 0) = body)
    by (unfold innerCode; cbn [positionLabels]; rewrite Hpath'; reflexivity).
  exists (frameView (codePosition index (labels ++ [labelSite offset 0]) 0) (segments ++ [values]) [] locals),
    (frameConfig f (frameView (codePosition index (labels ++ [labelSite offset 0]) 0) (segments ++ [values]) [] locals)
      (stackConfig z [] body)).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  apply Forall_app in Hvb.
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    assert (Hlen : length segments = length labels)
      by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
    rewrite (wrap_labels_snoc _ _ _ _ _ _ segments values _ Hpath Hsite Hlen).
    apply steps_in_labels; [exact Hready | ].
    rewrite (skipn_at_instruction _ _ _ Hi), <- (app_nil_r values), stack_config_split, app_nil_r.
    apply steps_with_stack_prefix; [exact (proj2 Hvb) | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | ].
    exact Hsteps.
  - split; [ | split; [exact Hregs | exact Heffect]].
    pose proof (idle_active_state program s final (afterBoundary row) f
      (frameView (codePosition index (labels ++ [labelSite offset 0]) 0) (segments ++ [values]) [] locals)) as Hafter.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
    rewrite Hinner' in Hafter. apply Hafter; clear Hafter.
    + apply frame_at_unmoved_intro;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
        unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner'.
      * exact Hf.
      * exact Hpath'.
      * rewrite (path_heights_snoc _ _ _ _ _ _ _ _ _ 0 Hpath Hsite Hheights), map_app.
        cbn [firstn sourceHeights option_map map length]. rewrite HlabelBase, !Nat.add_0_r. reflexivity.
      * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
      * cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
      * rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hbounds.
      * rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
        intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
        exact (Hcells _ _ Hv).
      * exact Hllen.
      * exact Hlb.
      * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp'. eapply cell_matches_stack; [exact HstackMem | exact (Hlc _ _ Hv)].
    + cbn. lia.
    + unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
      cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. rewrite Hsp', Hbase'. exact Hsp.
    + cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath Hsite).
      replace (armStart i (siteArm (labelSite offset 0))) with 1%nat by (destruct i; reflexivity).
      unfold codeOffset. cbn [siteOffset firstn codeSize]. lia.
    + exact HidleAfter.
Qed.

(* A nop: the next instruction runs on the same operands. *)
Lemma code_nop_row : forall view f initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_ACTIVATION_FINISHED_AFTER = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some NOP ->
  RowResult initial final row f inner view.
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hi.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  pose proof (frame_at_unmoved_cells _ _ _ _ _ _ _ _ Hframe) as Hcells.
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewSegments positionLabels] in Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  pose proof (encoded_path_nop _ _ _ _ Hloaded Hf _ _ _ Hpath Hi) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (quiet_control_row _ _ row _ _ COL_SEL_NOP _ Hprime Hfull Hprogram Hhalt Hreplay Logic.eq_refl
    ltac:(right; right; right; right; split; reflexivity) Himage Hidle)
    as [Hnext [Hsp' [Hbase' [Hregs [Heffect [HstackMem HidleAfter]]]]]].
  pose proof Hregs as [_ [_ Hfbp']].
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf (values_typed_wf _ _ Hlb)).
  exists (frameView (codePosition index labels (S offset)) segments values locals),
    (frameConfig f (frameView (codePosition index labels (S offset)) segments values locals)
      (stackConfig z values (skipn (S offset) code))).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    apply steps_in_labels; [exact Hready | ].
    rewrite (skipn_at_instruction _ _ _ Hi), <- (app_nil_r values), stack_config_split, app_nil_r.
    replace (stackConfig z values (skipn (S offset) code)) with
      (stackPrefix values (instructionSuffix (skipn (S offset) code) (stackConfig z [] [])))
      by (rewrite instruction_suffix_config, stack_prefix_config, app_nil_r; reflexivity).
    apply steps_with_stack_prefix; [exact (proj2 (proj1 (Forall_app _ _ _) Hbounds)) | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfcode | ].
    exact (nop_steps z Hz).
  - split; [ | split; [exact Hregs | exact Heffect]].
    assert (Hinner : innerCode f (codePosition index labels (S offset)) = code) by reflexivity.
    pose proof (idle_active_state program s final (afterBoundary row) f
      (frameView (codePosition index labels (S offset)) segments values locals)) as Hafter.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
    rewrite Hinner in Hafter. apply Hafter; clear Hafter.
    + apply frame_at_unmoved_intro;
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
        unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner.
      * exact Hf.
      * exact Hpath.
      * refine (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights _). rewrite Nat.add_0_r. reflexivity.
      * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
      * cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
      * exact Hbounds.
      * intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
        exact (Hcells _ _ Hv).
      * exact Hllen.
      * exact Hlb.
      * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp'. eapply cell_matches_stack; [exact HstackMem | exact (Hlc _ _ Hv)].
    + assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate). cbn. lia.
    + cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. rewrite Hsp', Hbase'. exact Hsp.
    + cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi).
      change (instructionSize NOP) with 1%nat. lia.
    + exact HidleAfter.
Qed.

(* An if pops its condition. A nonzero condition enters the then-body under the
   if's label, a zero one the else-body; an empty else-body is skipped, the label
   it would enter keeping nothing. *)
Lemma code_if_row : forall view f bt t e initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (IFELSE bt t e) ->
  RowResult initial final row f inner view.
Proof.
  intros view f bt t e initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi.
  destruct (program_row_active _ _ _ _ _ _ Hprime Hfull Hprogram Hstate)
    as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hfbp0 [Hbase0 [Hbounds [_ [Hllen [Hlb Hlc]]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [[Hsup _] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hsup, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts, Hsup. cbn [codeFacts] in Hfacts. destruct Hfacts as [Hif _].
  destruct (proj1 (if_facts _ _ _ _ _ _ _ _) Hif) as [Hwif [Hone [_ [Hends _]]]].
  inversion Hsup as [ | ? ? Hsupported _]; subst.
  inversion Hsupported as [ | | | | | | | | | | result ? ? _ _ Hresult | |  | | ]; subst.
  (* The condition is the top operand; `under` are the operands below it. *)
  assert (Hnonempty : values <> []) by (intros ->; cbn in Hone; lia).
  destruct (exists_last Hnonempty) as [under [condition ->]]. clear Hnonempty.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  apply Forall_app in Hvb as [Hsegments Hvalues]. apply Forall_app in Hvalues as [Hunder Hcond].
  apply Forall_inv in Hcond.
  (* The circuit: pop the condition, then enter an arm or skip the if. *)
  destruct (encoded_path_if _ _ _ _ Hloaded Hf _ _ _ _ _ _ Hpath Hi) as [Hop [Hkind Hedges]].
  unfold pathPC in Hop, Hkind, Hedges.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Hop, Hkind, Hedges.
  destruct (circuit_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram Logic.eq_refl Hop Hkind Hedges)
    as [Hchosen Hnext].
  assert (Hin : In COL_SEL_IF selectorColumns) by (vm_compute; tauto).
  destruct (full_selected_no_host_switch row COL_SEL_IF Hprime Hfull Hin Hchosen)
    as [Hsuspend [_ [_ [_ HcontextRead]]]].
  destruct (full_block_selector_quiet row COL_SEL_IF Hprime Hfull
    (or_intror (or_intror (or_intror (or_introl Logic.eq_refl)))) Hchosen) as [Hpush Hpop].
  pose proof (full_call_depth_balance row COL_SEL_IF Hprime Hfull Hin Hchosen) as Hdepth.
  rewrite Hpush, Hpop in Hdepth.
  pose proof (frame_stack_matches _ _ (beforeBoundary row) _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack.
  destruct (full_if_stack_shape row Hprime Hfull Hchosen) as [Hshape Hdistance].
  { destruct Hstack as [Hb0 [Hsp0 _]]. cbn [beforeBoundary boundarySP boundaryStackBase] in Hb0, Hsp0.
    rewrite !length_app in Hsp0. cbn [length] in Hsp0. lia. }
  rewrite app_assoc in Hstack.
  destruct (stack_matches_pop row initial final _ condition false Hshape Hreplay (fun _ => Hdistance)
    ltac:(intro H; discriminate H) Hstack) as [Hread HstackAfter].
  destruct HstackAfter as [Hbase1 [Hsp1 [Hbounds1 Hcells1]]].
  rewrite Hread in Hnext.
  pose proof (full_if_fbp_unchanged row Hprime Hfull Hchosen) as Hfbp.
  assert (Hwrites : filter isStackWrite (eventsForRow memories row) = []).
  { rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite (local_write_active _ _ _ Hshape), (local_write_hi_active _ _ _ Hshape), (local_write_gate _ _ _ Hshape),
      (local_gather_lo_inactive _ _ _ Hshape), (local_gather_inactive _ _ _ Hshape). reflexivity. }
  pose proof (no_stack_writes_preserve _ _ _ Hreplay Hwrites) as HstackMem.
  assert (Hregs : frameRegistersPreserved row)
    by (split; [lia | split; [exact (local_stack_base_after _ _ _ Hshape) | exact Hfbp]]).
  assert (Heffect : FrameEffect (row COL_LOCALS_FBP_BEFORE) initial final)
    by (eapply quiet_row_frame_effect; [exact Hreplay | exact Hwrites | exact Hpush | exact Hsuspend |
      exact HcontextRead]).
  pose proof (program_row_keeps_idle row COL_SEL_IF Hprime Hfull Hin Hchosen ltac:(not_call_selector) Hpop Hidle)
    as HidleAfter.
  (* SpecTec: the same choice. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf (values_typed_wf _ _ Hlb)).
  pose proof (forall_skipn _ _ (S offset) _ Hwfcode) as Hwfrest.
  pose proof (if_steps z condition _ t e (skipn (S offset) code) Hz Hcond Hwif Hwfrest) as Hsteps.
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  assert (Hbefore : frameConfig f (frameView (codePosition index labels offset) segments (under ++ [condition]) locals)
      (stackConfig z (under ++ [condition]) (skipn offset code)) =
      wrapLabels (guestBody f) labels segments
        (stackPrefix under (instructionSuffix (skipn (S offset) code) (stackConfig z [condition] [IFELSE (_RESULT result) t e])))).
  { unfold frameConfig. cbn [viewPosition viewSegments positionLabels]. rewrite (skipn_at_instruction _ _ _ Hi).
    rewrite stack_config_split. reflexivity. }
  assert (Hblock : Steps (instructionSuffix (skipn (S offset) code) (stackConfig z [condition] [IFELSE (_RESULT result) t e]))
      (instructionSuffix (skipn (S offset) code)
        (stackConfig z [] [BLOCK (_RESULT result) (if Z.eqb condition 0 then e else t)]))).
  { rewrite !instruction_suffix_config. exact Hsteps. }
  (* Entering an arm: the if's label goes on, over `under`. *)
  assert (Henter : forall arm body, siteEntry code (labelSite offset arm) = Some (IFELSE (_RESULT result) t e, body) ->
      body = (if Z.eqb condition 0 then e else t) ->
      row COL_PC_AFTER = guestEntry f + Z.of_nat (pathOffset (guestBody f) (labels ++ [labelSite offset arm]) 0) ->
      RowResult initial final row f
        (frameConfig f (frameView (codePosition index labels offset) segments (under ++ [condition]) locals)
          (stackConfig z (under ++ [condition]) (skipn offset code)))
        (frameView (codePosition index labels offset) segments (under ++ [condition]) locals)).
  { intros arm body Hsite Hbody HpcAfter.
    assert (Hwbody : Forall wf_instr body).
    { inversion Hwif. rewrite Hbody. destruct (Z.eqb condition 0); assumption. }
    assert (Hpath' : pathCode (guestBody f) (labels ++ [labelSite offset arm]) = Some body)
      by (rewrite path_code_snoc, Hpath, Hsite; reflexivity).
    assert (Hinner' : innerCode f (codePosition index (labels ++ [labelSite offset arm]) 0) = body)
      by (unfold innerCode; cbn [positionLabels]; rewrite Hpath'; reflexivity).
    exists (frameView (codePosition index (labels ++ [labelSite offset arm]) 0) (segments ++ [under]) [] locals),
      (frameConfig f (frameView (codePosition index (labels ++ [labelSite offset arm]) 0) (segments ++ [under]) []
        locals) (stackConfig z [] body)).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    - rewrite Hbefore. unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      rewrite (wrap_labels_snoc _ _ _ _ _ _ segments under _ Hpath Hsite Hlen).
      apply steps_in_labels; [exact Hready | ].
      apply steps_with_stack_prefix; [exact Hunder | ].
      eapply spec_steps_transitive; [exact Hblock | ].
      apply steps_with_instruction_suffix; [exact Hwfrest | ].
      rewrite <- Hbody. exact (enter_block z result body Hresult Hz Hwbody).
    - split; [ | split; [exact Hregs | exact Heffect]].
      pose proof (idle_active_state program s final (afterBoundary row) f
        (frameView (codePosition index (labels ++ [labelSite offset arm]) 0) (segments ++ [under]) [] locals)) as Hafter.
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
      rewrite Hinner' in Hafter. apply Hafter; clear Hafter.
      + apply frame_at_unmoved_intro;
          cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands];
          unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner'.
        * exact Hf.
        * exact Hpath'.
        * rewrite (path_heights_snoc _ _ _ _ _ _ _ _ _ 0 Hpath Hsite Hheights), map_app.
          cbn [firstn sourceHeights option_map map length labelBase]. rewrite length_app. cbn [length].
          rewrite !Nat.add_0_r, Nat.add_sub. reflexivity.
        * cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
        * pose proof (local_stack_base_after _ _ _ Hshape).
          cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
        * rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hbounds1.
        * rewrite concat_app. cbn [concat]. rewrite !app_nil_r. exact Hcells1.
        * exact Hllen.
        * exact Hlb.
        * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp. eapply cell_matches_stack; [exact HstackMem | exact (Hlc _ _ Hv)].
      + cbn. lia.
      + unfold viewOperands. cbn [viewSegments viewValues]. rewrite concat_app. cbn [concat]. rewrite !app_nil_r.
        exact Hsp1.
      + cbn [afterBoundary boundaryPC]. exact HpcAfter.
      + exact HidleAfter. }
  destruct (Z.eqb condition 0) eqn:Hzero.
  - destruct e as [ | i0 e'].
    + (* An empty else-body: enter and leave its label at once. *)
      assert (Hnone : blockResults (_RESULT result) = 0%nat)
        by (symmetry; apply (Hends 0%nat); reflexivity).
      exists (frameView (codePosition index labels (S offset)) segments under locals),
        (frameConfig f (frameView (codePosition index labels (S offset)) segments under locals)
          (stackConfig z under (skipn (S offset) code))).
      cbn [viewPosition positionFunction]. split; [exact Hf | ].
      split.
      * rewrite Hbefore. unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
        apply steps_in_labels; [exact Hready | ].
        rewrite <- (app_nil_r under) at 2. rewrite <- stack_prefix_config.
        apply steps_with_stack_prefix; [exact Hunder | ].
        eapply spec_steps_transitive; [exact Hblock | ].
        rewrite <- (app_nil_l (skipn (S offset) code)) at 2. rewrite <- instruction_suffix_config.
        apply steps_with_instruction_suffix; [exact Hwfrest | ].
        eapply spec_steps_transitive; [exact (enter_block z result [] Hresult Hz (Forall_nil _)) | ].
        exact (exit_block _ _ z [] (Forall_nil _) Hz (Forall_nil _)).
      * split; [ | split; [exact Hregs | exact Heffect]].
        assert (Hinner : innerCode f (codePosition index labels (S offset)) = code) by reflexivity.
        pose proof (idle_active_state program s final (afterBoundary row) f
          (frameView (codePosition index labels (S offset)) segments under locals)) as Hafter.
        cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset] in Hafter.
        rewrite Hinner in Hafter. apply Hafter; clear Hafter.
        -- apply frame_at_unmoved_intro;
             cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset
               viewOperands];
             unfold viewOperands; cbn [viewSegments viewValues]; try rewrite Hinner.
           ++ exact Hf.
           ++ exact Hpath.
           ++ refine (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights _).
              rewrite length_app. cbn [length sourceHeight]. rewrite Nat.add_1_r, Hnone, Nat.add_0_r. reflexivity.
           ++ cbn [afterBoundary boundaryLocalsBase]. normalize_columns. lia.
           ++ pose proof (local_stack_base_after _ _ _ Hshape).
              cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. normalize_columns. lia.
           ++ exact Hbounds1.
           ++ exact Hcells1.
           ++ exact Hllen.
           ++ exact Hlb.
           ++ intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp. eapply cell_matches_stack; [exact HstackMem | exact (Hlc _ _ Hv)].
        -- assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate). cbn. lia.
        -- unfold viewOperands. cbn [viewSegments viewValues]. exact Hsp1.
        -- cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi),
             if_instruction_size. lia.
        -- exact HidleAfter.
    + apply (Henter 1%nat (i0 :: e')).
      * unfold siteEntry. cbn [siteOffset siteArm]. rewrite Hi. reflexivity.
      * reflexivity.
      * rewrite Hnext, Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath
          (ltac:(unfold siteEntry; cbn [siteOffset siteArm]; rewrite Hi; reflexivity)
            : siteEntry code (labelSite offset 1) = Some (IFELSE (_RESULT result) t (i0 :: e'), i0 :: e'))).
        unfold codeOffset. cbn [armStart firstn codeSize siteOffset siteArm]. lia.
  - apply (Henter 0%nat t).
    + unfold siteEntry. cbn [siteOffset siteArm]. rewrite Hi. reflexivity.
    + reflexivity.
    + rewrite Hnext, Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath
        (ltac:(unfold siteEntry; cbn [siteOffset siteArm]; rewrite Hi; reflexivity)
          : siteEntry code (labelSite offset 0) = Some (IFELSE (_RESULT result) t e, t))).
      unfold codeOffset. cbn [armStart firstn codeSize siteOffset siteArm]. lia.
Qed.

(* The innermost label of a frame at a path, split off: the operands below it are
   its instruction's height less what it consumed. *)
Lemma innermost_label : forall f index labels offset segments values locals pending remaining distance memory base fbp,
  FrameAt program memory base fbp pending remaining distance f
    (frameView (codePosition index labels offset) segments values locals) ->
  labels <> [] ->
  exists labels' site segments' segment outer i height,
    labels = labels' ++ [site] /\ segments = segments' ++ [segment] /\ length segments' = length labels' /\
    pathCode (guestBody f) labels' = Some outer /\
    siteEntry outer site = Some (i, innerCode f (codePosition index labels offset)) /\
    pathHeights (programCallees program (guestModule f)) (guestBody f) labels' (siteOffset site) =
      Some (map (@length Z) segments', height) /\
    labelBase i height = length segment /\
    sourceHeights (programCallees program (guestModule f)) (firstn offset (innerCode f (codePosition index labels offset))) 0 =
      Some (length values + pending)%nat.
Proof.
  intros f index labels offset segments values locals pending remaining distance memory base fbp Hframe Hne.
  pose proof Hframe as [_ [Hpath [Hheights _]]].
  cbn [viewPosition viewSegments viewValues positionLabels positionOffset] in Hpath, Hheights.
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  destruct (lists_split_last _ _ labels segments ltac:(lia) Hne) as [labels' [site [segments' [segment [-> [-> Hlen']]]]]].
  destruct (path_heights_prefix _ _ _ _ _ _ _ Hheights)
    as [outer [i [body [hs' [height [Hpath' [Hs [Hheights' [Hhs Htop]]]]]]]]].
  rewrite path_code_snoc, Hpath', Hs in Hpath. cbn [option_map snd] in Hpath. injection Hpath as Hbody.
  rewrite map_app in Hhs. cbn [map] in Hhs. apply app_inj_tail in Hhs. destruct Hhs as [Hhs' Hbelow].
  exists labels', site, segments', segment, outer, i, height.
  split; [reflexivity | ]. split; [reflexivity | ]. split; [lia | ]. split; [exact Hpath' | ].
  rewrite <- Hbody. split; [exact Hs | ]. split; [rewrite Hheights', Hhs'; reflexivity | ].
  split; [exact (eq_sym Hbelow) | exact Htop].
Qed.

(* A structured instruction'site typing fixes where its label resumes: after it, the
   operands below its label and its results. *)
Lemma structured_resume : forall localCount resultCount arities callees i h arm body,
  instrFacts localCount resultCount arities callees i h -> instrArm i arm = Some body ->
  sourceHeight callees i h = Some (labelBase i h + labelResults i)%nat.
Proof.
  intros localCount resultCount arities callees i h arm body Hfacts Harm.
  destruct i; try discriminate; try reflexivity.
  destruct (proj1 (if_facts _ _ _ _ _ _ _ _) Hfacts) as [_ [Hone _]].
  destruct h as [ | h]; [lia | ]. cbn [sourceHeight labelBase labelResults]. f_equal. lia.
Qed.


(* The last slot of an arm: its END, or an if's then-body's ELSE. The label's
   values are its results; they join the operands below the label, and execution
   resumes after the structured instruction. *)
Lemma code_arm_end_row : forall view f initial final row inner,
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
    as [labels' [s0 [segments' [segment [outer [i [height [-> [-> [Hlen' [Hpath' [Hs [Hheights' [Hbase Htop]]]]]]]]]]]]]].
  set (body := innerCode f (codePosition index (labels' ++ [s0]) offset)) in *.
  (* The label's values are its results. *)
  rewrite Hend, firstn_all in Htop. pose proof (Hzero _ Htop) as Hvalues.
  rewrite (path_results_snoc _ _ _ _ _ _ Hpath' Hs), Nat.add_0_r in Hvalues.
  destruct (site_entry_split _ _ _ _ Hs) as [Hb Harm].
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts [_ [Hsup0 Hbr0]]].
  pose proof (code_facts_wf _ _ _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  (* The structured instruction's typing: where its label resumes. *)
  destruct (path_facts _ _ _ _ _ _ _ _ _ _ _ Hfacts Hsup0 Hbr0 Hpath' Hheights') as [Hat _].
  rewrite (skipn_at_instruction _ _ _ Hb) in Hat. destruct Hat as [Hinstr _].
  pose proof (structured_resume _ _ _ _ _ _ _ _ Hinstr Harm) as Hresume.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite concat_app in Hvb. cbn [concat] in Hvb. rewrite app_nil_r, <- app_assoc in Hvb.
  pose proof (proj2 (proj1 (Forall_app _ _ _) Hvb)) as Hinner. apply Forall_app in Hinner as [Hsegment Hresults].
  pose proof (proj1 (proj1 (Forall_app _ _ _) Hvb)) as Houter.
  assert (Hready' : LabelsReady (guestBody f) labels' segments')
    by exact (labels_ready _ _ _ _ Hwfbody Hpath' Hlen' Houter).
  pose proof (encoded_path_arm_end _ _ _ _ Hloaded Hf _ _ _ _ _ Hpath' Hs) as Himage.
  unfold pathPC in Himage. rewrite <- Hend in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (labels' ++ [s0]) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  assert (Hkind : armEndOpcode i (siteArm s0) = CODE_END /\ COL_SEL_END = COL_SEL_END \/
      armEndOpcode i (siteArm s0) = CODE_ELSE /\ COL_SEL_ELSE = COL_SEL_ELSE).
  { unfold armEndOpcode. destruct i; try (left; split; reflexivity).
    lazymatch goal with |- context [match ?e with [] => _ | _ :: _ => _ end] => destruct e end;
      [left; split; reflexivity | ].
    destruct (siteArm s0); [right | left]; split; reflexivity. }
  assert (Hmarker : exists chosen, quietControl (armEndOpcode i (siteArm s0)) chosen)
    by (unfold quietControl; destruct Hkind as [H | H]; [exists COL_SEL_END | exists COL_SEL_ELSE]; tauto).
  destruct Hmarker as [chosen Hmarker].
  destruct (quiet_control_row _ _ row _ _ chosen _ Hprime Hfull Hprogram Hhalt Hreplay Logic.eq_refl Hmarker Himage Hidle)
    as [Hnext [Hsp' [Hbase' [Hregs [Heffect [HstackMem HidleAfter]]]]]].
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf (values_typed_wf _ _ Hlb)).
  exists (frameView (codePosition index labels' (S (siteOffset s0))) segments' (segment ++ values) locals),
    (frameConfig f (frameView (codePosition index labels' (S (siteOffset s0))) segments' (segment ++ values) locals)
      (stackConfig z (segment ++ values) (skipn (S (siteOffset s0)) outer))).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    rewrite Hend, skipn_all, (wrap_labels_snoc _ _ _ _ _ _ segments' segment _ Hpath' Hs Hlen').
    apply steps_in_labels; [exact Hready' | ].
    replace (stackConfig z (segment ++ values) (skipn (S (siteOffset s0)) outer)) with
      (stackPrefix segment (instructionSuffix (skipn (S (siteOffset s0)) outer) (stackConfig z values [])))
      by (rewrite instruction_suffix_config, stack_prefix_config; reflexivity).
    apply steps_with_stack_prefix; [exact Hsegment | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
    exact (exit_block _ _ z values (continuation_wf _ _ _ _ Hwfouter Hs) Hz Hresults).
  - split; [ | split; [exact Hregs | exact Heffect]].
    assert (Hshift : forall (xs ys : list Z) k, nth_error (xs ++ ys) (length xs + k) = nth_error ys k)
      by (intros xs ys k; rewrite nth_error_app2 by lia; f_equal; lia).
    destruct HidleAfter as [HzAfter [HmAfter [HzaAfter HmaAfter]]].
    unfold viewOperands in Hsp, Hcells. cbn [viewSegments viewValues] in Hsp, Hcells.
    rewrite concat_app in Hsp, Hcells. cbn [concat] in Hsp, Hcells. rewrite app_nil_r, <- app_assoc in Hsp, Hcells.
    apply (exit_label_state _ _ _ _ _ _ _ _ 0 _ _ _ Hf Hpath').
    + apply nth_error_Some. rewrite Hb. discriminate.
    + rewrite length_app, Hvalues, <- Hbase.
      exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath' Hb Hheights' Hresume).
    + cbn [afterBoundary boundaryLocalsBase]. destruct Hregs as [_ [_ Hfbp']]. normalize_columns. lia.
    + cbn [afterBoundary boundaryLocalsBase boundaryStackBase]. destruct Hregs as [_ [_ Hfbp']]. normalize_columns.
      lia.
    + exact Hvb.
    + intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
      apply Hcells. rewrite app_assoc, nth_error_app1 by (apply nth_error_Some; rewrite Hv; discriminate). exact Hv.
    + intros j value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem, Nat.add_0_r.
      apply Hcells. rewrite app_assoc, Hshift. exact Hv.
    + exact HmAfter.
    + intros H. contradiction.
    + cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. normalize_columns. lia.
    + exact Hllen.
    + exact Hlb.
    + intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. destruct Hregs as [_ [_ Hfbp']].
      rewrite Hfbp'. eapply cell_matches_stack; [exact HstackMem | exact (Hlc _ _ Hv)].
    + cbn [afterBoundary boundaryPC]. rewrite Hnext. reflexivity.
    + unfold initializationFlags. rewrite HmAfter, HzAfter, HmaAfter, HzaAfter. split; reflexivity.
    + exact HzAfter.
Qed.

End Rows.
