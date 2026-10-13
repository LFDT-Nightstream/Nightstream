(* Straight-line executions are singleton-segment instances of the main theorem. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2
  StraightLineSpec StackSimulation CallSpec CallerSpec CallerRefinement StraightLineInstructions.
Import ListNotations.
Open Scope Z_scope.

Lemma straight_line_caller_typed : forall program z height,
  supportedStraightLineProgram program -> Forall wf_instr program ->
  (callerTyped (fun _ => None) z height program <->
    programTyped (length (stateLocals z)) height program).
Proof.
  induction program as [ | instruction rest IH]; intros z height Hsupported Hwf; [reflexivity | ].
  destruct (Forall_inv Hsupported) as [code Hcode].
  assert (Htail : forall h, callerTyped (fun _ => None) z h rest <->
      programTyped (length (stateLocals z)) h rest).
  { intros h. apply IH; [exact (Forall_inv_tail Hsupported) | exact (Forall_inv_tail Hwf)]. }
  assert (Hsingle : supportedStraightLineProgram [instruction]) by (constructor; [eauto | constructor]).
  pose proof (Forall_inv Hwf) as Hi.
  destruct (encoded_opcode_cases instruction code Hcode)
    as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    cbn [callerTyped programTyped instructionStackHeight];
    destruct height as [ | [ | height]]; cbn [programTyped instructionStackHeight];
    try rewrite Htail; tauto.
Qed.

Lemma represented_straight_line_wf : forall program z memory boundary c,
  supportedStraightLineProgram program -> represents z memory boundary program c ->
  Forall wf_instr program.
Proof.
  intros program z memory boundary c Hsupported [values [_ [-> [Hwf _]]]].
  inversion Hwf as [z0 code Hz Hcode]; subst.
  change (Forall wf_admininstr (map i32Constant values ++ map admininstr_instr program)) in Hcode.
  apply Forall_app in Hcode. destruct Hcode as [_ Hcode].
  rewrite Forall_map in Hcode.
  apply Forall_forall. intros i Hin.
  pose proof (proj1 (Forall_forall _ _) Hsupported i Hin) as [code Hencoded].
  pose proof (proj1 (Forall_forall _ _) Hcode i Hin) as Hi.
  destruct (encoded_opcode_cases i code Hencoded)
    as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    cbn [admininstr_instr] in Hi; inversion Hi; subst; constructor; assumption.
Qed.

Lemma loaded_program_caller : forall program z memory pc,
  supportedStraightLineProgram program -> loadedProgram program memory ->
  loadedCaller (fun _ => None) z memory (Z.of_nat pc) (skipn pc program).
Proof.
  intros program z memory pc Hsupported [Hop [Hkind [Hnext Hlocal]]].
  assert (Hsuffix : forall suffix pc, suffix = skipn pc program ->
      loadedCaller (fun _ => None) z memory (Z.of_nat pc) suffix).
  { induction suffix as [ | instruction rest IH]; intros index Hsuffix; [exact I | ].
    assert (Hlookup : nth_error program index = Some instruction).
    { pose proof (f_equal (fun xs => nth_error xs 0) Hsuffix) as H.
      cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact (eq_sym H). }
    assert (Hinside : (index <? length program)%nat = true).
    { apply Nat.ltb_lt. apply nth_error_Some. rewrite Hlookup. discriminate. }
    assert (Hsupport : exists code, encodedOpcode instruction = Some code).
    { apply (proj1 (Forall_forall _ _) Hsupported). eapply nth_error_In; exact Hlookup. }
    assert (Himage : instructionImage memory (Z.of_nat index) instruction).
    { unfold instructionImage. rewrite Hop, Hkind, Hinside. split.
      - unfold programOpcodeAt. now rewrite Hlookup.
      - split; [reflexivity | ]. split.
        + intros choice. rewrite Hnext, Hinside. cbn. rewrite Nat2Z.inj_add. reflexivity.
        + destruct instruction; try exact I; rewrite Hlocal; unfold programLocalIndexAt;
            rewrite Hlookup; reflexivity. }
    assert (Htail : rest = skipn (S index) program).
    { rewrite (skipn_at_instruction _ _ _ Hlookup) in Hsuffix. now injection Hsuffix. }
    assert (Hloaded : loadedCaller (fun _ => None) z memory (Z.of_nat index + 1) rest).
    { replace (Z.of_nat index + 1) with (Z.of_nat (S index)) by lia. now apply IH. }
    destruct Hsupport as [code Hcode].
    destruct (encoded_opcode_cases instruction code Hcode)
      as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; split; assumption. }
  apply Hsuffix. reflexivity.
Qed.

(* The ROM and linked PCs bound the number of rows; no extra prefix-length premise. *)
Lemma straight_line_run_length : forall rows program initial final pc,
  fieldPrime -> loadedProgram program initial -> validCircuitRun initial rows final ->
  Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) rows ->
  (pc <= length program)%nat ->
  (forall row, hd_error rows = Some row -> row COL_PC_BEFORE = Z.of_nat pc) ->
  (pc + length rows <= length program)%nat.
Proof.
  induction rows as [ | row rows IH]; intros program initial final pc Hprime Hloaded Hrun Hprogram Hbound Hpc;
    [cbn; lia | ].
  destruct (valid_run_head _ _ _ _ Hrun) as [middle [Hfull [Hreplay Htail]]].
  specialize (Hpc row eq_refl).
  destruct (program_row_fetch program initial middle row pc Hloaded (Forall_inv Hprogram) Hpc Hreplay)
    as [i [Hlookup _]].
  assert (Hinside : (pc < length program)%nat).
  { apply nth_error_Some. rewrite Hlookup. discriminate. }
  pose proof (program_row_next_pc program initial middle row pc Hprime Hfull Hloaded
    (Forall_inv Hprogram) Hpc Hreplay) as [_ [_ Hnext]].
  assert (Hloaded' : loadedProgram program middle).
  { eapply loaded_program_preserved; [exact Hloaded | exact Hreplay | ].
    intros memory address Hreadonly. apply read_only_row. exact Hreadonly. }
  assert (Hpc' : forall next, hd_error rows = Some next -> next COL_PC_BEFORE = Z.of_nat (S pc)).
  { intros next Hhead. destruct rows as [ | r rs]; [discriminate | ]. injection Hhead as <-.
    destruct Hrun as [_ [Hchain _]]. inversion Hchain; subst.
    match goal with H : Follows _ row r |- _ =>
      rewrite <- (H COL_PC_AFTER COL_PC_BEFORE ltac:(vm_compute; tauto)) end.
    rewrite Hnext. f_equal. lia. }
  pose proof (IH program middle final (S pc) Hprime Hloaded' Htail (Forall_inv_tail Hprogram)
    ltac:(lia) Hpc'). cbn. lia.
Qed.

Lemma singleton_segments_match : forall rows program,
  length rows = length program -> supportedStraightLineProgram program ->
  Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) rows ->
  Forall2 segmentMatches program (map instructionSegment rows).
Proof.
  induction rows as [ | row rows IH]; intros [ | i program] Hlength Hsupported Hprogram;
    try discriminate; [constructor | ].
  constructor.
  - destruct (Forall_inv Hsupported) as [code Hcode].
    destruct (encoded_opcode_cases i code Hcode)
      as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; exact (Forall_inv Hprogram).
  - apply IH; [cbn in Hlength; lia | exact (Forall_inv_tail Hsupported) | exact (Forall_inv_tail Hprogram)].
Qed.

Lemma singleton_segment_rows : forall rows,
  callerRows (map instructionSegment rows) = rows.
Proof. unfold callerRows. induction rows as [ | row rows IH]; cbn; [reflexivity | now rewrite IH]. Qed.

Lemma singleton_final_boundary : forall first rest,
  callerFinalBoundary (beforeBoundary first) (map instructionSegment (first :: rest)) =
  afterBoundary (finalRow first rest).
Proof.
  intros first rest. cbn [map callerFinalBoundary segmentLast]. revert first.
  induction rest as [ | next rest IH]; intros first; cbn [map callerFinalBoundary segmentLast finalRow]; auto.
Qed.

(* The same run theorem, with one segment per row. The remaining source stays in
   the final configuration, including when the run stops before the program ends. *)
Theorem straight_line_run_refinement : forall program z initial final first rest c0,
  fieldPrime -> supportedStraightLineProgram program -> loadedProgram program initial ->
  validCircuitRun initial (first :: rest) final ->
  Forall (fun row : Witness => row COL_IS_PROGRAM_ROW = 1) (first :: rest) ->
  first COL_PC_BEFORE = 0 ->
  programTyped (length (stateLocals z)) (stackHeight (beforeBoundary first)) program ->
  represents z initial (beforeBoundary first) program c0 ->
  exists z' c1, Steps c0 c1 /\
    programTyped (length (stateLocals z')) (stackHeight (afterBoundary (finalRow first rest)))
      (skipn (length (first :: rest)) program) /\
    represents z' final (afterBoundary (finalRow first rest)) (skipn (length (first :: rest)) program) c1 /\
    (finalRow first rest) COL_PC_AFTER = Z.of_nat (length (first :: rest)).
Proof.
  intros program z initial final first rest c0 Hprime Hsupported Hloaded Hrun Hprogram Hpc Htyped Hrep.
  set (n := length (first :: rest)).
  assert (Hlength : (n <= length program)%nat).
  { eapply (straight_line_run_length (first :: rest) program initial final 0 Hprime Hloaded Hrun Hprogram);
      [lia | intros row Hhead; injection Hhead as <-; exact Hpc]. }
  assert (Hwf : Forall wf_instr program) by (eapply represented_straight_line_wf; eassumption).
  assert (Hcaller : callerTyped (fun _ => None) z (stackHeight (beforeBoundary first)) program).
  { apply straight_line_caller_typed; assumption. }
  assert (HloadedCaller : loadedCaller (fun _ => None) z initial 0 program).
  { exact (loaded_program_caller program z initial 0 Hsupported Hloaded). }
  assert (Hmatches : Forall2 segmentMatches (firstn n program) (map instructionSegment (first :: rest))).
  { apply singleton_segments_match.
    - rewrite length_firstn. unfold n in *. lia.
    - apply Forall_forall. intros i Hin. apply (proj1 (Forall_forall _ _) Hsupported).
      rewrite <- (firstn_skipn n program). apply in_or_app. left. exact Hin.
    - exact Hprogram. }
  pose proof (caller_run_refinement (firstn n program) (skipn n program)
    (map instructionSegment (first :: rest)) (fun _ => None) z initial final (beforeBoundary first) 0 c0)
    as Hmain.
  rewrite firstn_skipn, singleton_segment_rows in Hmain.
  destruct (Hmain Hprime Hrun Hmatches eq_refl ltac:(constructor; assumption))
    as [z' [c1 [Hsteps [[_ Htyped' Hrep' Hpc'] _]]]].
  rewrite singleton_final_boundary in Htyped', Hrep', Hpc'.
  exists z', c1. split; [exact Hsteps | ]. split.
  - apply straight_line_caller_typed in Htyped'; [exact Htyped' | | ].
    + apply Forall_forall. intros i Hin. apply (proj1 (Forall_forall _ _) Hsupported).
      rewrite <- (firstn_skipn n program). apply in_or_app. right. exact Hin.
    + apply Forall_forall. intros i Hin. apply (proj1 (Forall_forall _ _) Hwf).
      rewrite <- (firstn_skipn n program). apply in_or_app. right. exact Hin.
  - split; [exact Hrep' | ]. rewrite length_firstn in Hpc'.
    replace (Nat.min n (length program)) with n in Hpc' by lia. exact Hpc'.
Qed.
