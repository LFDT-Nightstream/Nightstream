(* Mixed-width arguments survive local zeroing; zero locals require both limbs. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues MemoryTrace MemoryCatalog IntegerMemory IntegerDecode
  StraightLineSpec FrameInitialization CallScope.
Import ListNotations.
Open Scope Z_scope.

Definition emptyMemory : MemoryState := fun _ _ => None.
Definition arguments := [i32Value 7; i64Value (joinLimbs 9 1)].
Definition params := [valtype_I32; valtype_I64].
Definition locals := [valtype_I64; valtype_I32].
Definition defaults := [i64Value 0; i32Value 0].
Definition argumentMemory := writeCell (writeCell emptyMemory 4 7 0) 5 9 1.
Definition initializedMemory := writeCell (writeCell argumentMemory 6 0 0) 7 0 0.
Definition initialBoundary := mkBoundary 0 6 8 4 1 2 0 1 0 0 0.
Definition readyBoundary := mkBoundary 0 8 8 4 1 0 0 0 0 0 0.

Example mixed_arguments_start : InitializingFrame argumentMemory initialBoundary arguments 2 2.
Proof.
  apply initializing_frame_start; try (cbn; lia).
  intros [ | [ | index]] value H; cbn [arguments nth_error] in H;
    [injection H as <- | injection H as <- | destruct index; discriminate].
  - constructor; [unfold i32Modulus; lia | vm_compute; reflexivity].
  - change (CellMatches argumentMemory 5 (i64Value (joinLimbs 9 1))).
    constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
Qed.

Example mixed_frame_initialized : InitializingFrame initializedMemory readyBoundary arguments 2 0.
Proof.
  constructor; try (cbn; lia).
  - intros [ | [ | index]] value H; cbn [arguments nth_error] in H;
    [injection H as <- | injection H as <- | destruct index; discriminate].
    + constructor; [unfold i32Modulus; lia | vm_compute; reflexivity].
    + change (CellMatches initializedMemory 5 (i64Value (joinLimbs 9 1))).
      constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
  - intros index Hindex. cbn [arguments length] in Hindex.
    assert (index = 2 \/ index = 3) by lia.
    destruct H as [-> | ->]; vm_compute; split; reflexivity.
Qed.

Example mixed_frame_reads :
  readIntegerCells initializedMemory (fun index => 4 + Z.of_nat index) (params ++ locals) =
    Some (arguments ++ defaults).
Proof.
  apply (initialized_frame_reads initializedMemory readyBoundary arguments params locals defaults).
  - repeat constructor.
  - reflexivity.
  - exact mixed_frame_initialized.
Qed.

Example stale_high_limb_is_not_initialized :
  ~ InitializingFrame (writeCell initializedMemory 6 0 1) readyBoundary arguments 2 0.
Proof.
  intros H. pose proof (proj2 (zeroed_locals _ _ _ _ _ H 2 ltac:(cbn; lia))) as Hhi.
  vm_compute in Hhi. discriminate.
Qed.

(* The strengthened effect protects an i64 caller value, including its high limb. *)
Example caller_keeps_wide_argument : forall final,
  FrameEffect 6 argumentMemory final -> CellMatches final 5 (i64Value (joinLimbs 9 1)).
Proof.
  intros final Heffect. eapply frame_effect_cell; [exact Heffect | lia | ].
  constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
Qed.
