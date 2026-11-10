(* A real wide call, its zero rows and result move, checked by typed row lemmas.
   MixedWitness.v also checks complete runs through run_refinement. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range FullRelation Wasm2 WasmIntegers WasmValues MemoryTrace MemoryCatalog
  TraceRefinement StraightLineSpec IntegerMemory IntegerDecode IntegerLocals LocalBindings
  SpecTecCalls CallEntry CallReturn FrameInitialization
  tests.WitnessCheck tests.IntegerCallTrace.
Import ListNotations.
Open Scope Z_scope.

Definition wide := i64Value (joinLimbs 4294967295 4294967295).
(* Start after i64.const; constants are not in the proved instruction fragment. *)
Definition rows := map rowWitness (skipn 1 integerCallRows).
Definition row (n : nat) := nth n rows (rowWitness []).
Definition initial := writeCell (writeCell (writeCell
  (readInputs (eventsForTrace memories rows)) 0 0 0) 1 0 0) 2 4294967295 4294967295.
Fixpoint before (n : nat) : MemoryState :=
  match n with O => initial | S previous => replayed (before previous) [row previous] end.
Definition after (n : nat) := replayed (before n) [row n].

Example trace_checks : validRunb initial rows = true.
Proof. vm_compute. reflexivity. Qed.

Lemma row_checks : forall n, (n < 6)%nat -> validRunb (before n) [row n] = true.
Proof.
  intros [ | [ | [ | [ | [ | [ | n]]]]]] H; try lia; vm_compute; reflexivity.
Qed.

Lemma row_valid : forall n, (n < 6)%nat ->
  FullRelation.satFull (row n) /\ Replay (before n) (eventsForRow memories (row n)) (after n).
Proof.
  intros n Hn. destruct (validRunb_sound _ _ (row_checks n Hn)) as [Hsat [_ Hreplay]].
  split; [exact (Forall_inv Hsat) | ].
  cbn [eventsForTrace flat_map] in Hreplay. now rewrite app_nil_r in Hreplay.
Qed.

Example call_keeps_wide_argument : fieldPrime ->
  InitializingFrame (after 0) (afterBoundary (row 0)) [wide] 2 2.
Proof.
  intros Hprime. destruct (row_valid 0 ltac:(lia)) as [Hfull Hreplay].
  change 2 with (row 0 COL_LOCAL_ZERO_REMAINING_AFTER) at 2.
  apply (call_starts_integer_frame_initialization (before 0) (after 0) (row 0) [] [wide] 2).
  - exact Hprime.
  - exact Hfull.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - exact Hreplay.
  - change (0 <= 2). lia.
  - reflexivity.
  - intros [ | index] value H; cbn in H; [injection H as <- | destruct index; discriminate].
    unfold wide. constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
Qed.

Example zero_rows_initialize_mixed_locals : fieldPrime ->
  InitializingFrame (after 2) (afterBoundary (row 2)) [wide] 2 0.
Proof.
  intros Hprime.
  pose proof (call_keeps_wide_argument Hprime) as Hcall.
  destruct (row_valid 1 ltac:(lia)) as [Hfull1 Hreplay1].
  destruct (row_valid 2 ltac:(lia)) as [Hfull2 Hreplay2].
  assert (Hfirst : InitializingFrame (after 1) (afterBoundary (row 1)) [wide] 2
      (row 1 COL_LOCAL_ZERO_REMAINING_AFTER)).
  { apply (zero_row_initializes_frame (before 1) (after 1) (row 1) [wide] 2 Hprime Hfull1 eq_refl Hreplay1).
    assert (Hb : beforeBoundary (row 1) = afterBoundary (row 0)) by (vm_compute; reflexivity).
    rewrite Hb.
    assert (Hm : before 1 = after 0) by (unfold after; cbn [before]; reflexivity). rewrite Hm.
    assert (Hn : row 1 COL_LOCAL_ZERO_REMAINING_BEFORE = 2) by reflexivity. rewrite Hn.
    exact Hcall. }
  change 0 with (row 2 COL_LOCAL_ZERO_REMAINING_AFTER).
  apply (zero_row_initializes_frame (before 2) (after 2) (row 2) [wide] 2 Hprime Hfull2 eq_refl Hreplay2).
  assert (Hb : beforeBoundary (row 2) = afterBoundary (row 1)) by (vm_compute; reflexivity).
  rewrite Hb.
  assert (Hm : before 2 = after 1) by (unfold after; cbn [before]; reflexivity). rewrite Hm.
  assert (Hn : row 2 COL_LOCAL_ZERO_REMAINING_BEFORE = row 1 COL_LOCAL_ZERO_REMAINING_AFTER)
    by reflexivity. rewrite Hn. exact Hfirst.
Qed.

Example mixed_frame_reads : fieldPrime ->
  readIntegerCells (after 2) (fun i => 2 + Z.of_nat i) [valtype_I64; valtype_I32; valtype_I64] =
    Some [wide; i32Value 0; i64Value 0].
Proof.
  intros Hprime.
  apply (initialized_frame_reads (after 2) (afterBoundary (row 2)) [wide] [valtype_I64]
    [valtype_I32; valtype_I64] [i32Value 0; i64Value 0]);
    [repeat constructor | reflexivity | exact (zero_rows_initialize_mixed_locals Hprime)].
Qed.

Example local_get_keeps_wide_argument : fieldPrime -> CellMatches (after 3) 5 wide.
Proof.
  intros Hprime. destruct (row_valid 3 ltac:(lia)) as [Hfull Hreplay].
  change 5 with (row 3 COL_SP_BEFORE).
  eapply local_get_cell; [exact Hprime | exact Hfull | reflexivity | exact Hreplay | | ].
  - change (2 < 5). lia.
  - change (CellMatches (before 3) 2 wide).
    assert (Hm : before 3 = after 2) by (unfold after; cbn [before]; reflexivity). rewrite Hm.
    destruct (proj1 (read_integer_cells_some _ _ _ _) (mixed_frame_reads Hprime)) as [_ Hcells].
    exact (Hcells 0%nat wide eq_refl).
Qed.

Example end_keeps_result_at_source : fieldPrime -> CellMatches (after 4) 5 wide.
Proof.
  intros Hprime. destruct (row_valid 4 ltac:(lia)) as [Hfull Hreplay].
  eapply cell_matches_memory.
  - intros limb _. exact (no_stack_writes_preserve _ _ _ Hreplay
      (full_call_end_no_stack_writes (row 4) COL_SEL_END Hprime Hfull (or_intror eq_refl) eq_refl) _).
  - assert (Hm : before 4 = after 3) by (unfold after; cbn [before]; reflexivity). rewrite Hm.
    exact (local_get_keeps_wide_argument Hprime).
Qed.

Example return_move_keeps_both_limbs : fieldPrime ->
  readIntegerOperands (after 5) 2 [valtype_I64] 0 3 = Some [wide].
Proof.
  intros Hprime. destruct (row_valid 5 ltac:(lia)) as [Hfull Hreplay].
  change 0%nat with (Z.to_nat (row 5 COL_FRAME_MOVE_REMAINING_AFTER)).
  change 3 with (row 5 COL_FRAME_MOVE_DISTANCE_AFTER).
  apply (move_row_preserves_integer_operands (before 5) (after 5) (row 5) 2 [valtype_I64]);
    try assumption.
  - reflexivity.
  - reflexivity.
  - change (1 <= 1)%nat. lia.
  - apply read_integer_cells_some. split; [repeat constructor | ].
    intros [ | index] value H; cbn in H; [injection H as <- | destruct index; discriminate].
    change (CellMatches (before 5) 5 wide).
    assert (Hm : before 5 = after 4) by (unfold after; cbn [before]; reflexivity). rewrite Hm.
    exact (end_keeps_result_at_source Hprime).
Qed.

Definition wrongMove : Witness := {| entries := fun c =>
  if Nat.eqb c COL_STACK_WRITE0_VALUE_HI then field 0 else entries (row 5) c |}.
Example changed_move_high_rejected : satFullb wrongMove = false.
Proof. vm_compute. reflexivity. Qed.

(* The same typed call and return on the SpecTec side. *)
Example spectec_wide_call : forall z module typeIndex,
  nth_error (fun_funcinst z) 1 = Some (typedFunction module typeIndex [valtype_I64] [valtype_I64]
    [valtype_I32; valtype_I64] [LOCAL_GET (mk_uN 0)]) ->
  wf_funcinst (typedFunction module typeIndex [valtype_I64] [valtype_I64]
    [valtype_I32; valtype_I64] [LOCAL_GET (mk_uN 0)]) ->
  Step (mk_config z [admininstr_val wide; CALL_ADDR 1%N])
    (mk_config z [FRAME_ 1%N (MKframe [wide; i32Value 0; i64Value 0] module)
      [LABEL_ 1%N [] [admininstr_LOCAL_GET (mk_uN 0)]]]).
Proof.
  intros z module typeIndex Hlookup Hwf.
  apply (call_address_enters_frame z 1%N module typeIndex [wide] [valtype_I64] [valtype_I64]
    [valtype_I32; valtype_I64] [LOCAL_GET (mk_uN 0)] [i32Value 0; i64Value 0]);
    try assumption; try reflexivity.
  - constructor; [unfold wide; apply i64_value_wf; apply join_limbs_bound; unfold i32Modulus; lia | constructor].
  - repeat constructor.
  - constructor; [apply i32_value_wf; unfold i32Modulus; lia | ].
    constructor; [apply i64_value_wf; unfold i64Modulus; lia | constructor].
Qed.

Example spectec_wide_return : forall s caller callee,
  wf_state (mk_state s caller) -> wf_frame callee ->
  Steps (mk_config (mk_state s caller)
    [FRAME_ 1%N callee [LABEL_ 1%N [] [admininstr_val wide]]])
    (mk_config (mk_state s caller) [admininstr_val wide]).
Proof.
  intros s caller callee Hstate Hcallee. apply (value_function_returns s caller callee [wide]); try assumption.
  constructor; [unfold wide; apply i64_value_wf; apply join_limbs_bound; unfold i32Modulus; lia | constructor].
Qed.
