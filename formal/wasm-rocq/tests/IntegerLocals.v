(* A real i64 local.set/get pair, without migrating the i32 program decoder.
   ROM read inputs come from the exported trace; this checks the row and memory
   theorems, not the encoder or run_refinement for an i64 program. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers WasmValues IntegerMemory IntegerLocals Locals
  tests.WitnessCheck tests.IntegerLocalTrace.
Import ListNotations.
Open Scope Z_scope.

Definition setRow := rowWitness (nth 1 integerLocalRows []).
Definition getRow := rowWitness (nth 2 integerLocalRows []).
Definition localRows := [setRow; getRow].
Definition maxLimb : Z := 4294967295.
Definition maxI64 := i64Value (joinLimbs maxLimb maxLimb).

Definition initial := writeCell
  (writeCell (readInputs (eventsForTrace memories localRows)) 0 0 0) 1 maxLimb maxLimb.
Definition final := replayed initial localRows.

Example wide_value_exceeds_field : modulus < joinLimbs maxLimb maxLimb.
Proof. vm_compute. reflexivity. Qed.

Example wide_value_exact : joinLimbs maxLimb maxLimb = 18446744073709551615.
Proof. reflexivity. Qed.

Lemma initial_operand : CellMatches initial 1 maxI64.
Proof.
  constructor; [unfold maxLimb, i32Modulus; lia | unfold maxLimb, i32Modulus; lia |
    apply write_cell_low | apply write_cell_high].
Qed.

Example local_rows_check : validRunb initial localRows = true.
Proof. vm_compute. reflexivity. Qed.

(* The conclusion uses both full-relation cell lemmas, not just evaluation of
   the final memory. Each row's gates, addresses and both limbs are constrained. *)
Example local_pair_copies_i64 : fieldPrime -> CellMatches final 1 maxI64.
Proof.
  intros Hprime.
  destruct (validRunb_sound initial localRows local_rows_check) as [Hfull [_ Hreplay]].
  change (Forall FullRelation.satFull [setRow; getRow]) in Hfull.
  inversion Hfull as [ | ? ? Hset Htail]; subst.
  inversion Htail as [ | ? ? Hget Hnil]; subst.
  change (Replay initial (eventsForRow memories setRow ++ eventsForRow memories getRow ++ []) final) in Hreplay.
  rewrite app_nil_r in Hreplay.
  destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [HsetReplay HgetReplay]].
  assert (Hlocal : CellMatches middle 0 maxI64).
  { change 0 with (setRow COL_LOCALS_FBP_BEFORE + setRow COL_LOCAL_INDEX).
    eapply local_set_cell; [exact Hprime | exact Hset | reflexivity | exact HsetReplay | | ].
    - vm_compute. discriminate.
    - change (CellMatches initial 1 maxI64). exact initial_operand. }
  change 1 with (getRow COL_SP_BEFORE).
  eapply local_get_cell; [exact Hprime | exact Hget | reflexivity | exact HgetReplay | | ].
  - vm_compute. reflexivity.
  - change (CellMatches middle 0 maxI64). exact Hlocal.
Qed.

Example local_get_spectec : forall z x,
  fun_local z x = maxI64 ->
  Step (mk_config z [admininstr_LOCAL_GET x]) (mk_config z [admininstr_val maxI64]).
Proof.
  intros z x Hlocal.
  change maxI64 with (i64Value (joinLimbs (getRow COL_STACK_WRITE0_VALUE_LO)
    (getRow COL_STACK_WRITE0_VALUE_HI))).
  apply (full_relation_implies_i64_local_get getRow z x); [reflexivity | | exact Hlocal].
  apply satFullb_sound. vm_compute. reflexivity.
Qed.

Example local_set_spectec : forall z x,
  Step (mk_config z [admininstr_val maxI64; admininstr_LOCAL_SET x])
    (mk_config (with_local z x maxI64) []).
Proof.
  intros z x.
  change maxI64 with (i64Value (joinLimbs
    (setRow (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))
    (setRow (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)))) at 1.
  change maxI64 with (i64Value (joinLimbs (setRow COL_LOCAL_VALUE) (setRow COL_LOCAL_VALUE_HI))).
  apply (full_relation_implies_i64_local_set setRow z x); [reflexivity | ].
  apply satFullb_sound. vm_compute. reflexivity.
Qed.

(* Changing only the copied high limb must fail even if the low limb is intact. *)
Definition changeColumn (w : Witness) (column : Column) (z : Z) : Witness :=
  {| entries := fun c => if Nat.eqb c column then field z else entries w c |}.

Example wrong_get_high_limb_rejected :
  satFullb (changeColumn getRow COL_STACK_WRITE0_VALUE_HI 0) = false.
Proof. vm_compute. reflexivity. Qed.

Example wrong_set_high_limb_rejected :
  satFullb (changeColumn setRow COL_LOCAL_VALUE_HI 0) = false.
Proof. vm_compute. reflexivity. Qed.

(* i32 cells intentionally do not constrain the unused high limb. *)
Example i32_cell_with_nonzero_high :
  CellMatches (writeCell (fun _ _ => None) 0 7 maxLimb) 0 (i32Value 7).
Proof. constructor; [unfold i32Modulus; lia | apply write_cell_low]. Qed.
