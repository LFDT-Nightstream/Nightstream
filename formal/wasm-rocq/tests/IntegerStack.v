(* Exported circuit rows with a nonzero-high-limb i64 below their operands.
   ROM inputs come from the trace; these tests do not validate the encoder or
   run_refinement for the fixture's constants and mixed-width function. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers WasmValues StraightLineSpec IntegerMemory IntegerDecode IntegerStack IntegerLocals
  StackBindings tests.WitnessCheck tests.IntegerStackTrace.
Import ListNotations.
Open Scope Z_scope.

Definition wide := i64Value (joinLimbs 9 1).
Definition maxLimb : Z := 4294967295.
Definition maxI64 := i64Value (joinLimbs maxLimb maxLimb).
Definition addRow := rowWitness (nth 3 integerStackRows []).
Definition setRow := rowWitness (nth 6 integerStackRows []).
Definition getRow := rowWitness (nth 7 integerStackRows []).

Definition addInitial := writeCell (writeCell (writeCell
  (readInputs (eventsForTrace memories [addRow])) 1 9 1) 2 7 0) 3 9 0.
Definition addFinal := replayed addInitial [addRow].

Example add_row_checks : validRunb addInitial [addRow] = true.
Proof. vm_compute. reflexivity. Qed.

Lemma add_initial_stack :
  valueStackMatches addInitial (beforeBoundary addRow) [wide; i32Value 7; i32Value 9].
Proof.
  split; [vm_compute; discriminate | ]. split; [reflexivity | ].
  intros [ | [ | [ | index]]] value H; cbn [nth_error] in H;
    try (destruct index; discriminate); injection H as <-.
  - change (CellMatches addInitial 1 (i64Value (joinLimbs 9 1))).
    constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
  - change (CellMatches addInitial 2 (i32Value 7)).
    constructor; [unfold i32Modulus; lia | vm_compute; reflexivity].
  - change (CellMatches addInitial 3 (i32Value 9)).
    constructor; [unfold i32Modulus; lia | vm_compute; reflexivity].
Qed.

Example add_preserves_wide_prefix : fieldPrime ->
  valueStackMatches addFinal (afterBoundary addRow) [wide; i32Value 16].
Proof.
  intros Hprime.
  destruct (validRunb_sound _ _ add_row_checks) as [Hfull [_ Hreplay]].
  inversion Hfull as [ | ? ? Hrow Htail]; subst.
  change (Replay addInitial (eventsForRow memories addRow ++ []) addFinal) in Hreplay.
  rewrite app_nil_r in Hreplay.
  pose proof (full_binary_stack_shape addRow COL_SEL_I32_ADD Hprime Hrow
    ltac:(left; reflexivity) Logic.eq_refl ltac:(vm_compute; discriminate)) as Hshape.
  exact (proj2 (proj2 (value_stack_matches_binary addRow addInitial addFinal [wide] 7 9 Hshape Hreplay
    add_initial_stack ltac:(lia) ltac:(lia) ltac:(change (0 <= 16 < i32Modulus); unfold i32Modulus; lia)))).
Qed.

(* This reads both limbs from circuit memory, using the proved preservation. *)
Example add_reads_back_wide_prefix : fieldPrime -> readIntegerCell addFinal 1 valtype_I64 = Some wide.
Proof.
  intros Hprime. destruct (add_preserves_wide_prefix Hprime) as [_ [_ Hcells]].
  apply read_integer_cell_complete; [exact (Hcells 0%nat wide Logic.eq_refl) | reflexivity].
Qed.

Definition localRows := [setRow; getRow].
Definition localInitial := writeCell (writeCell (writeCell
  (readInputs (eventsForTrace memories localRows)) 0 0 0) 1 9 1) 2 maxLimb maxLimb.
Definition localFinal := replayed localInitial localRows.

Example local_rows_check : validRunb localInitial localRows = true.
Proof. vm_compute. reflexivity. Qed.

Lemma local_initial_stack : valueStackMatches localInitial (beforeBoundary setRow) [wide; maxI64].
Proof.
  split; [vm_compute; discriminate | ]. split; [reflexivity | ].
  intros [ | [ | index]] value H; cbn [nth_error] in H;
    try (destruct index; discriminate); injection H as <-.
  - change (CellMatches localInitial 1 (i64Value (joinLimbs 9 1))).
    constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
  - change (CellMatches localInitial 2 (i64Value (joinLimbs maxLimb maxLimb))).
    constructor; try (unfold maxLimb, i32Modulus; lia); vm_compute; reflexivity.
Qed.

Example local_pair_preserves_wide_prefix : fieldPrime ->
  valueStackMatches localFinal (afterBoundary getRow) [wide; maxI64].
Proof.
  intros Hprime.
  destruct (validRunb_sound _ _ local_rows_check) as [Hfull [_ Hreplay]].
  change (Forall FullRelation.satFull [setRow; getRow]) in Hfull.
  inversion Hfull as [ | ? ? Hset Htail]; subst.
  inversion Htail as [ | ? ? Hget Hnil]; subst.
  change (Replay localInitial (eventsForRow memories setRow ++ eventsForRow memories getRow ++ []) localFinal) in Hreplay.
  rewrite app_nil_r in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [HsetReplay HgetReplay]].
  destruct (local_set_value_stack setRow localInitial middle [wide] maxI64 Hprime Hset Logic.eq_refl
    HsetReplay ltac:(vm_compute; reflexivity) local_initial_stack) as [Hstack Hlocal].
  apply (local_get_value_stack getRow middle localFinal [wide] maxI64 Hprime Hget Logic.eq_refl
    HgetReplay ltac:(vm_compute; reflexivity)); [ | exact Hlocal].
  exact Hstack.
Qed.
