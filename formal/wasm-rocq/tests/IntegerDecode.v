(* Cell decoding and source types, independently of the i32 run fragment. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers WasmValues MemoryTrace MemoryCatalog IntegerMemory IntegerDecode
  CallerSpec StackTypes CodeLayout ControlDecode.
Require Import tests.IntegerLocals.
Import ListNotations.
Open Scope Z_scope.

Definition emptyMemory : MemoryState := fun _ _ => None.
Definition maxLimb : Z := 4294967295.
Definition wideValue := i64Value 18446744073709551615.
Definition mixedMemory := writeCell (writeCell emptyMemory 4 7 maxLimb) 5 maxLimb maxLimb.

Example mixed_cells_decode :
  readIntegerCells mixedMemory (fun i => 4 + Z.of_nat i) [valtype_I32; valtype_I64] =
    Some [i32Value 7; wideValue].
Proof. vm_compute. reflexivity. Qed.

(* Reuse the full-relation proof of the exported Rust local.set/get pair. *)
Example real_local_pair_decodes : Range.fieldPrime ->
  readIntegerCell tests.IntegerLocals.final 1 valtype_I64 = Some tests.IntegerLocals.maxI64.
Proof.
  intros Hprime. apply read_integer_cell_complete.
  - exact (tests.IntegerLocals.local_pair_copies_i64 Hprime).
  - reflexivity.
Qed.

Example i32_does_not_read_high :
  readIntegerCell (update emptyMemory stackId [8] 7) 4 valtype_I32 = Some (i32Value 7).
Proof. vm_compute. reflexivity. Qed.

Example i64_requires_high :
  readIntegerCell (update emptyMemory stackId [8] 7) 4 valtype_I64 = None.
Proof. vm_compute. reflexivity. Qed.

Example i64_rejects_out_of_range_high :
  readIntegerCell (writeCell emptyMemory 4 7 i32Modulus) 4 valtype_I64 = None.
Proof. vm_compute. reflexivity. Qed.

Example i64_rejects_out_of_range_low :
  readIntegerCell (writeCell emptyMemory 4 (-1) 0) 4 valtype_I64 = None.
Proof. vm_compute. reflexivity. Qed.

(* The first operand has moved; the other two remain one cell above their
   destinations. Cell 11 is stale and must not supply the i64 value yet. *)
Definition movingMemory := writeCell (writeCell (writeCell (writeCell emptyMemory
  10 7 0) 11 99 0) 12 maxLimb maxLimb) 13 9 0.
Definition movedOnce := writeCell movingMemory 11 maxLimb maxLimb.
Definition operandTypes := [valtype_I32; valtype_I64; valtype_I32].

Example pending_move_decodes_both_limbs :
  readIntegerOperands movingMemory 10 operandTypes 2 1 =
    Some [i32Value 7; wideValue; i32Value 9].
Proof. vm_compute. reflexivity. Qed.

Example copied_operand_uses_new_position :
  readIntegerOperands movedOnce 10 operandTypes 1 1 =
    Some [i32Value 7; wideValue; i32Value 9].
Proof.
  change movedOnce with (writeCell movingMemory (10 + Z.of_nat (length operandTypes - 2)) maxLimb maxLimb).
  apply (read_integer_operands_copy movingMemory 10 operandTypes 2 1);
    [cbn; lia | lia | exact pending_move_decodes_both_limbs | reflexivity | reflexivity].
Qed.

Definition testModule := MKmoduleinst [] [] [] [] [] [] [] [].
Definition wideCallee := mkGuestFunction 0%N testModule (mk_uN 0)
  [valtype_I32; valtype_I64] [valtype_I64] [valtype_I32] [] 0.
Definition callees := calleeEnvironment (fun _ => Some wideCallee) [guestType wideCallee] [] True.
Definition locals := [valtype_I32; valtype_I64].

(* The decoder's local reader uses parameter types followed by declared local
   types. The second value needs both limbs; the others ignore their high limb. *)
Example frame_locals_decode_by_declaration :
  readLocals (writeCell mixedMemory 6 11 maxLimb) 4 (guestFrameTypes wideCallee) =
    Some [i32Value 7; wideValue; i32Value 11].
Proof. vm_compute. reflexivity. Qed.

Example frame_types_keep_parameter_types :
  guestFrameTypes wideCallee = [valtype_I32; valtype_I64; valtype_I32].
Proof. reflexivity. Qed.

Example local_transfer_types :
  sourceTypeStacks locals callees [LOCAL_GET (mk_uN 1); LOCAL_SET (mk_uN 1)] [valtype_I32] =
    Some [valtype_I32].
Proof. vm_compute. reflexivity. Qed.

Example wrong_local_type_rejected :
  sourceTypes locals callees (LOCAL_SET (mk_uN 1)) [valtype_I32] = None.
Proof. vm_compute. reflexivity. Qed.

Example local_out_of_bounds_rejected :
  sourceTypes locals callees (LOCAL_GET (mk_uN 2)) [] = None.
Proof. vm_compute. reflexivity. Qed.

Example mixed_prefix_survives_i32_add :
  sourceTypes locals callees (BINOP I32 (mk_binop__0 Inn_I32 ADD))
    [valtype_I64; valtype_I32; valtype_I32] = Some [valtype_I64; valtype_I32].
Proof. vm_compute. reflexivity. Qed.

Example i32_add_rejects_i64_operand :
  sourceTypes locals callees (BINOP I32 (mk_binop__0 Inn_I32 ADD))
    [valtype_I32; valtype_I64] = None.
Proof. vm_compute. reflexivity. Qed.

Example call_consumes_typed_arguments :
  sourceTypes locals callees (CALL (mk_uN 0)) [valtype_I64; valtype_I32; valtype_I64] =
    Some [valtype_I64; valtype_I64].
Proof. vm_compute. reflexivity. Qed.

Example indirect_call_requires_i32_index :
  sourceTypes locals callees (CALL_INDIRECT (mk_uN 0) (mk_uN 0))
    [valtype_I32; valtype_I64; valtype_I64] = None.
Proof. vm_compute. reflexivity. Qed.

Example call_underflow_rejected : sourceTypes locals callees (CALL (mk_uN 0)) [] = None.
Proof. vm_compute. reflexivity. Qed.

Definition nestedCode := [LOCAL_GET (mk_uN 1);
  BLOCK (_RESULT (Some valtype_I64)) [LOCAL_GET (mk_uN 1)]; LOCAL_SET (mk_uN 1)].

Example inside_block_types :
  pathTypes locals callees nestedCode [labelSite 1 0] 1 =
    Some ([[valtype_I64]], [valtype_I64]).
Proof. vm_compute. reflexivity. Qed.

Example after_block_types :
  pathTypes locals callees nestedCode [] 2 = Some ([], [valtype_I64; valtype_I64]).
Proof. vm_compute. reflexivity. Qed.

Example no_fallthrough_after_return :
  sourceTypeStacks locals callees [RETURN; NOP] [valtype_I64] = None.
Proof. vm_compute. reflexivity. Qed.
