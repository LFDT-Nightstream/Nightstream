(* Rows of the active frame's code, at any depth of enclosing blocks: an ordinary
   instruction, entering a block, a block's END, and a depth-0 branch. Each keeps
   the frame registers and touches no memory below the frame. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers Add R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  TraceRefinement StraightLineSpec StackSimulation LocalSimulation SpecTecCalls SpecTecBranches CallerSpec
  CallerRefinement ProgramSpec ProgramEncoding CodeLayout BlockSpec BlockBindings BlockEncoding BlockInstructions
  BlockTrace BranchBindings StackBindings CallInitialization Selectors CallScope CallContext CallMemory LocalBindings
  StraightLineInstructions ProgramBinding Fragment CodeFacts ControlDecode ControlInvariant FrameFacts ControlBridge.
Import ListNotations.
Open Scope Z_scope.

Ltac not_call_selector :=
  let H := fresh in intro H; vm_compute in H; repeat (destruct H as [H | H]; [discriminate H | ]); exact H.

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
  destruct (encoded_opcode_cases _ _ Hop) as [[-> ->] | [[-> ->] | [[x [-> ->]] | [x [-> ->]]]]].
  - apply (program_row_keeps_idle w COL_SEL_I32_ADD); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_I32_ADD CODE_I32_ADD Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_I32_SUB); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_I32_SUB CODE_I32_SUB Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_LOCAL_GET); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_LOCAL_GET CODE_LOCAL_GET Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
  - apply (program_row_keeps_idle w COL_SEL_LOCAL_SET); [exact Hprime | exact Hfull | vm_compute; tauto | | not_call_selector | exact Hidle].
    apply (full_opcode_selects w COL_SEL_LOCAL_SET CODE_LOCAL_SET Hprime Hfull Hprogram); [vm_compute; reflexivity | symmetry; exact Hcode].
Qed.

(* A row that writes no stack cell, no local and no call context. *)
Lemma quiet_row_frame_effect : forall initial final row base fbp,
  Replay initial (eventsForRow memories row) final ->
  row COL_STACK_WRITE0_ACTIVE = 0 -> row COL_STACK_WRITE0_HI_ACTIVE = 0 ->
  filter isLocalWrite (eventsForRow memories row) = [] -> row COL_CALL_STACK_PUSH_PRESENT = 0 ->
  FrameEffect base fbp initial final.
Proof.
  intros initial final row base fbp Hreplay Hlo Hhi Hlocal Hpush. constructor.
  - intros index _. eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption].
  - intros other index _. eapply no_local_writes_preserve; eassumption.
  - intros memory column address Hin. eapply context_preserved_without_push; eassumption.
Qed.

(* An ordinary row under a stack prefix: the prefix is just more of the frame's stack. *)
Lemma framed_row_scope : forall prefix values i z initial final row,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> encodedOpcode i = Some (row COL_OPCODE_CODE) ->
  wf_state z -> wf_instr i -> programTyped (length (stateLocals z)) (length values) [i] ->
  stackMatches initial (beforeBoundary row) (prefix ++ values) -> localsMatch initial (beforeBoundary row) z ->
  FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final /\
  frameRegistersPreserved row.
Proof.
  intros prefix values i z initial final row Hprime Hfull Hprogram Hreplay Hcode Hz Hi Htyped Hstack Hlocals.
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

Lemma frame_stack_matches : forall program memory boundary fbp pending f view,
  FrameAt program memory (boundaryStackBase boundary) fbp pending f view ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) ->
  stackMatches memory boundary (viewOperands view).
Proof.
  intros program memory boundary fbp pending f view [_ [_ [_ [Hbase [Hbounds [Hcells _]]]]]] Hsp.
  split; [exact Hbase | ]. split; [exact Hsp | ]. split; [exact Hbounds | exact Hcells].
Qed.

Lemma frame_locals_match : forall program memory boundary base pending f view s,
  FrameAt program memory base (boundaryLocalsBase boundary) pending f view ->
  localsMatch memory boundary (mk_state s (activeFrame f (viewLocals view))).
Proof.
  intros program memory boundary base pending f view s [_ [_ [_ [_ [_ [_ [_ [Hlb Hlc]]]]]]]] index v Hv.
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
      memory localLoId [boundaryLocalsBase boundary; Z.of_nat index] = Some value.
Proof.
  intros s f z memory boundary Hstore Hmodule Hlen Hlocals.
  destruct (read_cells_total (fun index => memory localLoId [boundaryLocalsBase boundary; index])
    (length (stateLocals z)) 0) as [locals Hread].
  { intros index Hindex. destruct (nth_error (stateLocals z) index) as [v | ] eqn:Hv;
      [ | apply nth_error_None in Hv; lia].
    destruct (Hlocals index v Hv) as [value [_ [Hrange Hcell]]]. exists value. split; [exact Hcell | exact Hrange]. }
  destruct (proj1 (read_cells_some _ _ _ _) Hread) as [Hcount [Hbounds Hc]].
  assert (Hcells : forall index value, nth_error locals index = Some value ->
      memory localLoId [boundaryLocalsBase boundary; Z.of_nat index] = Some value)
    by (intros index value Hi; pose proof (Hc _ _ Hi) as H; cbn beta in H; rewrite Z.add_0_l in H; exact H).
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

Lemma ordinary_facts : forall localCount callees i h,
  (exists op, encodedOpcode i = Some op) -> instrFacts localCount callees i h ->
  wf_instr i /\ programTyped localCount h [i] /\ sourceHeight callees i h = instructionStackHeight i h.
Proof.
  intros localCount callees i h [op Hop] H.
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

Lemma concat_empties : forall (segments : list (list Z)), Forall (fun segment => segment = []) segments ->
  concat segments = [].
Proof. intros segments H. induction H as [ | seg more -> _ IH]; [reflexivity | exact IH]. Qed.

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
    FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final.

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
  destruct Hstate as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights _]]]. rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals]. cbn [viewPosition viewSegments viewValues viewLocals
    positionFunction positionLabels positionOffset] in *.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [_ [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts. cbn [codeFacts] in Hfacts.
  destruct Hfacts as [Hinstr Hrest]. rewrite Nat.add_0_r in Hinstr, Hrest.
  destruct (ordinary_facts _ _ _ _ Hop Hinstr) as [Hwi [Htyped Hsource]].
  pose proof (encoded_path_ordinary _ _ _ _ Hloaded Hf _ _ _ _ Hpath Hi Hop) as Himage.
  unfold pathPC in Himage. change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  pose proof Hframe as [_ [_ [_ [_ [Hbounds [_ [Hllen0 [Hlb0 _]]]]]]]].
  cbn [viewLocals] in Hllen0, Hlb0.
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb0).
  pose proof (frame_stack_matches _ _ _ _ _ _ _ Hframe Hsp) as Hstack. unfold viewOperands in Hstack.
  cbn [viewSegments viewValues] in Hstack.
  pose proof (frame_locals_match _ _ _ _ _ _ _ s Hframe) as Hlocals. cbn [viewLocals] in Hlocals.
  assert (Htyped' : programTyped (length (stateLocals z)) (length values) [i])
    by (cbn [z stateLocals activeFrame LOCALS]; rewrite length_map, Hllen0; exact Htyped).
  destruct (framed_instruction (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Himage Hz
    Hwi Htyped' Hstack Hlocals) as [z' [values' [[Hstep [Hz' [Hstack' [Hlocals' [Hsame Hheight']]]]] [Hnext Hregs]]]].
  pose proof (circuit_row_opcode_fetch _ _ _ Hreplay Hprogram) as Hfetch. rewrite (proj1 Himage) in Hfetch.
  destruct (framed_row_scope (concat segments) values i z initial final row Hprime Hfull Hprogram Hreplay Hfetch Hz
    Hwi Htyped' Hstack Hlocals) as [Heffect _].
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
    exists f. unfold FrameAt.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
    split.
    + split; [exact Hf | ]. split; [exact Hpath | ].
      split.
      { assert (Hstep' : sourceHeight (programCallees program (guestModule f)) i (length values + 0) =
            Some (length values' + 0)%nat) by (rewrite !Nat.add_0_r, Hsource; exact Hheight').
        exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights Hstep'). }
      split; [pose proof (proj1 Hstack) as H0; cbn [beforeBoundary afterBoundary boundaryStackBase] in H0 |- *; normalize_columns; lia | ].
      unfold viewOperands. cbn [viewSegments viewValues].
      split; [exact (proj1 (proj2 (proj2 Hstack'))) | ].
      split; [intros index' value Hv; exact (proj2 (proj2 (proj2 Hstack')) _ _ Hv) | ].
      split; [exact Hllen | ]. split; [exact Hlb | ].
      intros index' value Hv. pose proof (Hlc _ _ Hv) as H. cbn [afterBoundary boundaryLocalsBase] in H. exact H.
    + change (innerCode f (codePosition index labels (S offset))) with code.
      split; [assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate); lia | ].
      split; [exact (proj1 (proj2 Hstack')) | ].
      split; [cbn [afterBoundary boundaryPC]; rewrite Hnext, Hpc,
        (path_offset_step _ _ _ _ _ Hpath Hi), (ordinary_instruction_size _ Hop); lia | ].
      split; [exact (fetched_ordinary_keeps_idle _ _ _ _ Hprime Hfull Hprogram Hreplay (proj1 Himage) Hop Hidle) |
        reflexivity].
Qed.

(* Leaving the innermost label: its operands are the frame's innermost values again,
   and execution resumes after its END. *)
Lemma exit_label_state : forall memory boundary index labels b segments segment locals f inner bt body,
  functionAt program index = Some f ->
  pathCode (guestBody f) labels = Some inner -> nth_error inner b = Some (BLOCK bt body) ->
  pathHeights (programCallees program (guestModule f)) (guestBody f) labels b =
    Some (map (@length Z) segments, length segment) ->
  0 <= boundaryStackBase boundary ->
  Forall (fun value => 0 <= value < i32Modulus) (concat segments ++ segment) ->
  (forall index' value, nth_error (concat segments ++ segment) index' = Some value ->
    memory stackId [2 * (boundaryStackBase boundary + Z.of_nat index')] = Some value) ->
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (concat segments ++ segment)) ->
  length locals = frameLocals f -> Forall (fun value => 0 <= value < i32Modulus) locals ->
  (forall index' value, nth_error locals index' = Some value ->
    memory localLoId [boundaryLocalsBase boundary; Z.of_nat index'] = Some value) ->
  boundaryPC boundary = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels (S b)) ->
  initializationIdle boundary ->
  ActiveState program s memory boundary (atCode (frameView (codePosition index labels (S b)) segments segment locals))
    (frameConfig f (frameView (codePosition index labels (S b)) segments segment locals)
      (stackConfig (mk_state s (activeFrame f locals)) segment (skipn (S b) inner))).
Proof.
  intros memory boundary index labels b segments segment locals f inner bt body Hf Hpath Hb Hheights Hbase
    Hbounds Hcells Hsp Hllen Hlb Hlc Hpc Hidle.
  assert (Hinner : innerCode f (codePosition index labels (S b)) = inner)
    by (unfold innerCode; cbn [positionLabels]; rewrite Hpath; reflexivity).
  exists f. unfold FrameAt, viewOperands.
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
  rewrite Hinner. split.
  - split; [exact Hf | ]. split; [exact Hpath | ].
    split; [rewrite Nat.add_0_r; exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hb Hheights Logic.eq_refl) | ].
    split; [exact Hbase | ]. split; [exact Hbounds | ]. split; [exact Hcells | ].
    split; [exact Hllen | ]. split; [exact Hlb | exact Hlc].
  - split; [assert (b < length inner)%nat by (apply nth_error_Some; rewrite Hb; discriminate); lia | ].
    split; [exact Hsp | ].
    split; [exact Hpc | ].
    split; [exact Hidle | reflexivity].
Qed.

Lemma code_block_enter_row : forall view f body initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  row COL_HALTED = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) =
    Some (BLOCK (_RESULT None) body) ->
  RowResult initial final row f inner view.
Proof.
  intros view f body initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hi.
  destruct Hstate as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hbase0 [Hbounds [Hcells [Hllen [Hlb Hlc]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [Hfacts [_ [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hfacts, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hfacts. cbn [codeFacts] in Hfacts. destruct Hfacts as [Hblock _].
  destruct (proj1 (block_facts _ _ _ _ _) Hblock) as [Hwb [Hzero Hbody]].
  assert (Hwbody : Forall wf_instr body) by (inversion Hwb; assumption).
  pose proof (encoded_path_block_entry _ _ _ _ Hloaded Hf _ _ _ _ Hpath Hi) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (BlockTrace.control_row _ _ row _ CODE_BLOCK COL_SEL_BLOCK _ Hprime Hfull Hprogram Hreplay Logic.eq_refl
    ltac:(vm_compute; reflexivity) Himage) as [Hchosen Hnext].
  destruct (full_block_selector_quiet row COL_SEL_BLOCK Hprime Hfull (or_introl Logic.eq_refl) Hchosen)
    as [Hpush Hpop].
  destruct (full_block_context row COL_SEL_BLOCK Hprime Hfull (or_introl Logic.eq_refl) Hchosen Hpop Hhalt)
    as [Hsp' [Hbase' [Hfbp' [Hdepth' [Hw0 [Hw1 Hlw]]]]]].
  pose proof (program_row_keeps_idle row COL_SEL_BLOCK Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hidle) as HidleAfter.
  assert (HstackMem : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HlocalMem : forall address, final localLoId address = initial localLoId address)
    by (eapply no_local_writes_preserve; eassumption).
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
    exact (enter_empty_block z body Hz Hwbody).
  - split; [ | split; [split; [exact Hdepth' | split; assumption] |
      eapply quiet_row_frame_effect; eassumption]].
    exists f. unfold FrameAt, viewOperands.
    cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
    rewrite Hinner', concat_app. cbn [concat]. rewrite !app_nil_r. split.
    + split; [exact Hf | ]. split; [exact Hpath' | ].
      split; [rewrite (path_heights_snoc _ _ _ _ _ _ _ _ _ 0 Hpath Hi Hheights), map_app;
        cbn [firstn sourceHeights option_map map length]; rewrite Nat.add_0_r; reflexivity | ].
      split; [pose proof Hbase0 as H0; cbn [beforeBoundary afterBoundary boundaryStackBase] in H0 |- *;
        normalize_columns; lia | ].
      split; [exact Hbounds | ].
      split; [intros index' value Hv; cbn [afterBoundary boundaryStackBase]; rewrite Hbase', HstackMem;
        exact (Hcells _ _ Hv) | ].
      split; [exact Hllen | ]. split; [exact Hlb | ].
      intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp', HlocalMem. exact (Hlc _ _ Hv).
    + split; [lia | ].
      split; [cbn [afterBoundary boundarySP boundaryStackBase] in Hsp |- *; rewrite Hsp', Hbase'; exact Hsp | ].
      split; [cbn [afterBoundary boundaryPC]; rewrite Hnext, Hpc, (path_offset_snoc _ _ _ _ _ _ 0 Hpath Hi);
        unfold codeOffset; cbn; lia | ].
      split; [exact HidleAfter | reflexivity].
Qed.

(* The innermost label of a frame at a path, split off. *)
Lemma innermost_label : forall f index labels offset segments values locals pending memory base fbp,
  FrameAt program memory base fbp pending f (frameView (codePosition index labels offset) segments values locals) ->
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
  intros f index labels offset segments values locals pending memory base fbp Hframe Hne.
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
  row COL_HALTED = 0 -> Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  positionLabels (viewPosition view) <> [] ->
  positionOffset (viewPosition view) = length (innerCode f (viewPosition view)) ->
  RowResult initial final row f inner view.
Proof.
  intros view f initial final row inner Hprime Hloaded Hfull Hprogram Hhalt Hreplay Hstate Hf Hne Hend.
  destruct Hstate as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hbase0 [Hbounds [Hcells [Hllen [Hlb Hlc]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [_ [Hblock _]]].
  cbn [viewPosition positionLabels] in Hblock. pose proof (Hblock Hne) as Hzero.
  destruct (innermost_label _ _ _ _ _ _ _ _ _ _ _ Hframe Hne)
    as [labels' [b [segments' [segment [outer [bt [-> [-> [Hlen' [Hpath' [Hb [Hheights' Htop]]]]]]]]]]]].
  set (body := innerCode f (codePosition index (labels' ++ [b]) offset)) in *.
  rewrite Hend, firstn_all, Hzero in Htop. injection Htop as Hvalues.
  destruct values; [ | cbn in Hvalues; lia]. clear Hvalues.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts _].
  pose proof (code_facts_wf _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb.
  rewrite app_nil_r, concat_app in Hvb. cbn [concat] in Hvb. rewrite app_nil_r in Hvb.
  pose proof (proj2 (proj1 (Forall_app _ _ _) Hvb)) as Hsegment.
  assert (Hready' : LabelsReady (guestBody f) labels' segments')
    by exact (labels_ready _ _ _ _ Hwfbody Hpath' Hlen' (proj1 (proj1 (Forall_app _ _ _) Hvb))).
  pose proof (encoded_path_block_end _ _ _ _ Hloaded Hf (labels' ++ [b]) body Hne Hpath) as Himage.
  unfold pathPC in Himage. rewrite <- Hend in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (labels' ++ [b]) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  pose proof (control_row_static_edge _ _ row _ _ _ Hprogram Hreplay Logic.eq_refl Himage) as Hedge.
  destruct (BlockTrace.control_row _ _ row _ CODE_END COL_SEL_END _ Hprime Hfull Hprogram Hreplay Logic.eq_refl
    ltac:(vm_compute; reflexivity) Himage) as [Hchosen Hnext].
  destruct (full_block_end_quiet row Hprime Hfull Hchosen Hedge Hhalt) as [Hpush Hpop].
  destruct (full_block_context row COL_SEL_END Hprime Hfull (or_intror Logic.eq_refl) Hchosen Hpop Hhalt)
    as [Hsp' [Hbase' [Hfbp' [Hdepth' [Hw0 [Hw1 Hlw]]]]]].
  pose proof (program_row_keeps_idle row COL_SEL_END Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hidle) as HidleAfter.
  assert (HstackMem : forall address, final stackId address = initial stackId address)
    by (intros address; eapply replay_preserves; [exact Hreplay | apply stack_writes_inactive; assumption]).
  assert (HlocalMem : forall address, final localLoId address = initial localLoId address)
    by (eapply no_local_writes_preserve; eassumption).
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  exists (frameView (codePosition index labels' (S b)) segments' segment locals),
    (frameConfig f (frameView (codePosition index labels' (S b)) segments' segment locals)
      (stackConfig z segment (skipn (S b) outer))).
  cbn [viewPosition positionFunction]. split; [exact Hf | ].
  split.
  - unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
    rewrite Hend, skipn_all, (wrap_labels_snoc _ _ _ _ _ _ segments' segment _ Hpath' Hb Hlen').
    apply steps_in_labels; [exact Hready' | ].
    change (stackConfig z segment (skipn (S b) outer)) with
      (stackPrefix segment (instructionSuffix (skipn (S b) outer) (stackConfig z [] []))).
    apply steps_with_stack_prefix; [exact Hsegment | ].
    apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
    exact (exit_empty_block z Hz).
  - split; [ | split; [split; [exact Hdepth' | split; assumption] | eapply quiet_row_frame_effect; eassumption]].
    apply (exit_label_state _ _ _ _ _ _ _ _ _ _ _ _ Hf Hpath' Hb Hheights').
    + pose proof Hbase0 as H0. cbn [beforeBoundary afterBoundary boundaryStackBase] in H0 |- *. normalize_columns. lia.
    + exact Hvb.
    + intros index' value Hv. cbn [afterBoundary boundaryStackBase]. rewrite Hbase', HstackMem.
      apply Hcells. unfold viewOperands. cbn [viewSegments viewValues]. rewrite app_nil_r, concat_app.
      cbn [concat]. rewrite app_nil_r. exact Hv.
    + unfold viewOperands in Hsp. cbn [viewSegments viewValues] in Hsp. rewrite app_nil_r, concat_app in Hsp.
      cbn [concat] in Hsp. rewrite app_nil_r in Hsp.
      cbn [afterBoundary beforeBoundary boundarySP boundaryStackBase] in Hsp |- *. normalize_columns. lia.
    + exact Hllen.
    + exact Hlb.
    + intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp', HlocalMem. exact (Hlc _ _ Hv).
    + cbn [afterBoundary boundaryPC]. rewrite Hnext, Hpc, Hend, <- (exit_offset _ _ _ _ _ _ Hpath' Hb), Nat2Z.inj_add.
      lia.
    + exact HidleAfter.
Qed.

(* A br_if l: on zero it falls through; otherwise it leaves the l blocks inside
   its target and the target itself, whose operands below are all that remain,
   and resumes after the target's END. *)
Lemma code_branch_row : forall view f l initial final row inner,
  fieldPrime -> programLoaded program initial -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final ->
  ActiveState program s initial (beforeBoundary row) (atCode view) inner ->
  functionAt program (positionFunction (viewPosition view)) = Some f ->
  nth_error (innerCode f (viewPosition view)) (positionOffset (viewPosition view)) = Some (BR_IF l) ->
  RowResult initial final row f inner view.
Proof.
  intros view f l initial final row inner Hprime Hloaded Hfull Hprogram Hreplay Hstate Hf Hi.
  destruct Hstate as [f' [Hframe [Hoffset [Hsp [Hpc [Hidle ->]]]]]].
  pose proof Hframe as [Hf' [Hpath [Hheights [Hbase0 [Hbounds [Hcells [Hllen [Hlb Hlc]]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  destruct view as [[index labels offset] segments values locals].
  cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset viewOperands]
    in *.
  destruct (view_facts _ _ _ _ _ _ _ _ Hstore Hvalid Hframe) as [_ [[_ Hbranches] [_ [_ [Hready Hwfcode]]]]].
  cbn [viewPosition viewValues viewSegments positionLabels positionOffset] in Hbranches, Hready, Hwfcode.
  set (code := innerCode f (codePosition index labels offset)) in *.
  rewrite (skipn_at_instruction _ _ _ Hi) in Hbranches. destruct Hbranches as [[Hone Hexits] _].
  destruct values as [ | condition [ | extra more]]; cbn [length] in Hone; try lia. clear Hone.
  assert (Hwfl : wf_uN 32%N l).
  { pose proof (proj1 (Forall_forall _ _) Hwfcode _ (nth_error_In _ _ Hi)) as H. inversion H; assumption. }
  (* The target block, and the labels inside it, whose operands below are empty. *)
  destruct (label_exits_split _ _ Hexits) as [hsHead [h [hsTail [Hhs [Htail Hzeros]]]]].
  assert (Hlen : length segments = length labels)
    by (pose proof (path_heights_length _ _ _ _ _ _ Hheights) as H; rewrite length_map in H; exact H).
  assert (Hlengths : length segments = (length hsHead + S (length hsTail))%nat)
    by (rewrite <- (length_map (@length Z)), Hhs, length_app; reflexivity).
  destruct (list_split_at _ labels (length hsHead) ltac:(lia)) as [pre [b [post [-> Hpre]]]].
  destruct (list_split_at _ segments (length hsHead) ltac:(lia)) as [segsHead [segT [segsTail [-> HsegHead]]]].
  rewrite map_app in Hhs. cbn [map] in Hhs.
  destruct (app_split_eq _ (map (@length Z) segsHead) _ hsHead _ ltac:(rewrite length_map; lia) Hhs) as [_ Htail'].
  injection Htail' as _ HtailHs.
  repeat rewrite length_app in Hlen. repeat rewrite length_app in Hlengths. cbn [length] in Hlen, Hlengths.
  assert (Hpost : length post = N.to_nat (l :> N)) by (rewrite <- Htail, <- HtailHs, length_map; lia).
  assert (Hempty : Forall (fun segment : list Z => segment = []) segsTail).
  { rewrite <- HtailHs in Hzeros. apply Forall_map in Hzeros. eapply Forall_impl; [ | exact Hzeros].
    intros seg Hseg. apply length_zero_iff_nil. exact Hseg. }
  destruct (path_heights_split _ _ _ _ _ _ _ _ Hheights)
    as [outer [bt [body [hsHead' [below [hsTail' [Hpath' [Hb [Hheights' [Hhs' _]]]]]]]]]].
  pose proof (path_heights_length _ _ _ _ _ _ Hheights') as HheadLen.
  rewrite map_app in Hhs'. cbn [map] in Hhs'.
  destruct (app_split_eq _ (map (@length Z) segsHead) _ hsHead' _ ltac:(rewrite length_map; lia) Hhs') as [Hhead' Htail''].
  injection Htail'' as Hbelow _.
  rewrite <- Hhead', <- Hbelow in Hheights'.
  pose proof Hpath as Hinside. rewrite path_code_app, Hpath' in Hinside. cbn [pathCode] in Hinside.
  rewrite Hb in Hinside.
  destruct (function_facts _ _ _ _ Hstore Hvalid Hf) as [Hfacts _].
  pose proof (code_facts_wf _ _ _ _ Hfacts) as Hwfbody.
  pose proof (path_code_wf _ _ _ Hwfbody Hpath') as Hwfouter.
  assert (Hwfblock : Forall wf_instr body).
  { pose proof (proj1 (Forall_forall _ _) Hwfouter _ (nth_error_In _ _ Hb)) as H. inversion H; assumption. }
  pose proof Hbounds as Hvb. unfold viewOperands in Hvb. cbn [viewSegments viewValues] in Hvb. rewrite concat_app in Hvb. cbn [concat] in Hvb.
  rewrite (concat_empties _ Hempty), app_nil_r in Hvb.
  apply Forall_app in Hvb. destruct Hvb as [Hvb Hcond]. apply Forall_inv in Hcond.
  apply Forall_app in Hvb. destruct Hvb as [Hhead Hsegment].
  assert (HreadyHead : LabelsReady (guestBody f) pre segsHead)
    by exact (labels_ready pre _ _ segsHead Hwfbody Hpath' ltac:(lia) Hhead).
  assert (HreadyTail : LabelsReady body post segsTail)
    by (apply (labels_ready _ _ _ _ Hwfblock Hinside); [lia | rewrite (concat_empties _ Hempty); constructor]).
  (* The circuit: pop the condition, then fall through or jump past the target. *)
  pose proof (encoded_path_branch _ _ _ _ Hloaded Hf (pre ++ b :: post) code offset l pre b post Hpath Hi
    Logic.eq_refl (Logic.eq_sym Hpost)) as Himage.
  unfold pathPC in Himage.
  change (row COL_PC_BEFORE = guestEntry f + Z.of_nat (pathOffset (guestBody f) (pre ++ b :: post) offset)) in Hpc.
  rewrite <- Hpc in Himage.
  destruct (circuit_br_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram Logic.eq_refl Himage) as [Hchosen Hnext].
  pose proof (br_if_depth_preserved row Hprime Hfull Hchosen) as Hdepth.
  destruct (full_block_selector_quiet row COL_SEL_BR_IF_EQZ Hprime Hfull (or_intror Logic.eq_refl) Hchosen)
    as [Hpush _].
  pose proof (frame_stack_matches _ _ _ _ _ _ _ Hframe Hsp) as Hstack.
  unfold viewOperands in Hstack. cbn [viewSegments viewValues] in Hstack.
  assert (Hshape : UnaryStackShape row true).
  { apply full_br_if_stack_shape; [exact Hprime | exact Hfull | exact Hchosen | ].
    destruct Hstack as [Hb0 [Hsp0 _]]. cbn [beforeBoundary boundarySP boundaryStackBase] in Hb0, Hsp0.
    rewrite length_app in Hsp0. cbn [length] in Hsp0. lia. }
  destruct (stack_matches_pop row initial final _ condition Hshape Hreplay Hstack) as [Hread HstackAfter].
  rewrite Hread in Hnext.
  destruct (full_br_if_locals_unchanged row Hprime Hfull Hchosen) as [Hfbp Hlw].
  assert (HlocalMem : forall address, final localLoId address = initial localLoId address)
    by (eapply no_local_writes_preserve; eassumption).
  assert (Hregs : frameRegistersPreserved row)
    by (split; [exact Hdepth | split; [exact (local_stack_base_after _ _ Hshape) | exact Hfbp]]).
  assert (Heffect : FrameEffect (row COL_STACK_FRAME_BASE_BEFORE) (row COL_LOCALS_FBP_BEFORE) initial final)
    by (eapply quiet_row_frame_effect; [exact Hreplay | exact (local_write_active _ _ Hshape) |
      exact (local_write_hi_active _ _ Hshape) | exact Hlw | exact Hpush]).
  pose proof (program_row_keeps_idle row COL_SEL_BR_IF_EQZ Hprime Hfull ltac:(vm_compute; tauto) Hchosen
    ltac:(not_call_selector) Hidle) as HidleAfter.
  (* SpecTec: the same choice. *)
  set (z := mk_state s (activeFrame f locals)).
  assert (Hz : wf_state z) by exact (frame_state_wf _ _ _ _ _ Hstore Hwf Hf Hlb).
  pose proof (forall_skipn _ _ (S offset) _ Hwfcode) as Hwfrest.
  pose proof (br_if_steps z condition l (skipn (S offset) code) Hz Hcond Hwfl Hwfrest) as Hsteps.
  assert (Hbefore : frameConfig f (frameView (codePosition index (pre ++ b :: post) offset)
      (segsHead ++ segT :: segsTail) [condition] locals) (stackConfig z [condition] (skipn offset code)) =
      wrapLabels (guestBody f) (pre ++ b :: post) (segsHead ++ segT :: segsTail)
        (stackConfig z [condition] (BR_IF l :: skipn (S offset) code))).
  { unfold frameConfig. cbn [viewPosition viewSegments positionLabels]. rewrite (skipn_at_instruction _ _ _ Hi).
    reflexivity. }
  destruct (Z.eqb condition 0) eqn:Hzero.
  - (* Fall through: every label stays. *)
    exists (frameView (codePosition index (pre ++ b :: post) (S offset)) (segsHead ++ segT :: segsTail) [] locals),
      (frameConfig f (frameView (codePosition index (pre ++ b :: post) (S offset)) (segsHead ++ segT :: segsTail) []
        locals) (stackConfig z [] (skipn (S offset) code))).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + rewrite Hbefore. unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      exact (steps_in_labels _ _ _ _ _ Hready Hsteps).
    + split; [ | split; [exact Hregs | exact Heffect]].
      assert (Hinner : innerCode f (codePosition index (pre ++ b :: post) (S offset)) = code) by reflexivity.
      exists f. unfold FrameAt, viewOperands.
      cbn [viewPosition viewSegments viewValues viewLocals positionFunction positionLabels positionOffset].
      rewrite Hinner. rewrite app_nil_r. split.
      * split; [exact Hf | ]. split; [exact Hpath | ].
        split; [exact (path_heights_step _ _ _ _ _ _ _ _ _ Hpath Hi Hheights Logic.eq_refl) | ].
        destruct HstackAfter as [Hbase1 [Hsp1 [Hbounds1 Hcells1]]].
        split; [exact Hbase1 | ]. split; [exact Hbounds1 | ]. split; [exact Hcells1 | ].
        split; [exact Hllen | ]. split; [exact Hlb | ].
        intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HlocalMem. exact (Hlc _ _ Hv).
      * split; [assert (offset < length code)%nat by (apply nth_error_Some; rewrite Hi; discriminate); lia | ].
        split; [exact (proj1 (proj2 HstackAfter)) | ].
        split; [cbn [afterBoundary boundaryPC]; rewrite Hnext, Hpc, (path_offset_step _ _ _ _ _ Hpath Hi);
          change (instructionSize (BR_IF l)) with 1%nat; lia | ].
        split; [exact HidleAfter | reflexivity].
  - (* Taken: leave the labels inside the target, then the target. *)
    destruct l as [ln]. cbn in Hpost.
    assert (Hindex : mk_uN ln = mk_uN (N.of_nat (length post + 0))) by (rewrite Nat.add_0_r, Hpost; f_equal; lia).
    rewrite Hindex in Hsteps, Hwfl.
    destruct (wrap_labels_exit post body _ segsTail z 0 _ Hinside ltac:(lia) Hempty HreadyTail Hz Hwfl Hwfrest)
      as [rest' [Hrest' Hexit]].
    exists (frameView (codePosition index pre (S b)) segsHead segT locals),
      (frameConfig f (frameView (codePosition index pre (S b)) segsHead segT locals)
        (stackConfig z segT (skipn (S b) outer))).
    cbn [viewPosition positionFunction]. split; [exact Hf | ].
    split.
    + rewrite Hbefore, (wrap_labels_app pre (b :: post) _ _ segsHead (segT :: segsTail) _ Hpath' ltac:(lia)).
      cbn [wrapLabels]. rewrite Hb, Hindex.
      unfold frameConfig. cbn [viewPosition viewSegments positionLabels].
      apply steps_in_labels; [exact HreadyHead | ].
      change (stackConfig z segT (skipn (S b) outer)) with
        (stackPrefix segT (instructionSuffix (skipn (S b) outer) (stackConfig z [] []))).
      apply steps_with_stack_prefix; [exact Hsegment | ].
      apply steps_with_instruction_suffix; [apply forall_skipn; exact Hwfouter | ].
      eapply spec_steps_transitive; [apply steps_in_block; eapply spec_steps_transitive;
        [exact (steps_in_labels _ _ _ _ _ HreadyTail Hsteps) | exact Hexit] | ].
      exact (br_exit_target z rest' Hz Hrest').
    + split; [ | split; [exact Hregs | exact Heffect]].
      destruct HstackAfter as [Hbase1 [Hsp1 [Hbounds1 Hcells1]]].
      rewrite concat_app in Hbounds1, Hcells1, Hsp1. cbn [concat] in Hbounds1, Hcells1, Hsp1.
      rewrite (concat_empties _ Hempty), app_nil_r in Hbounds1, Hcells1, Hsp1.
      apply (exit_label_state _ _ _ _ _ _ _ _ _ _ _ _ Hf Hpath' Hb Hheights' Hbase1 Hbounds1 Hcells1 Hsp1 Hllen Hlb).
      * intros index' value Hv. cbn [afterBoundary boundaryLocalsBase]. rewrite Hfbp, HlocalMem. exact (Hlc _ _ Hv).
      * cbn [afterBoundary boundaryPC]. rewrite Hnext. reflexivity.
      * exact HidleAfter.
Qed.

End Rows.
