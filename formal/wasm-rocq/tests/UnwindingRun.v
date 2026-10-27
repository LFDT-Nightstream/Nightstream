(* A taken br_if 1 that leaves two blocks and discards two operands: the one below
   its condition and the outer block's operand below the inner block. Its exported
   rows satisfy the circuit, chain, and replay from the loaded program, checked by
   computation. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals StraightLineSpec
  SpecTecCalls CallerSpec CallerValidation FunctionValidation ProgramSpec tests.WitnessCheck tests.UnwindingTrace
  tests.BlockPrograms.
Import ListNotations.
Open Scope Z_scope.

(* block (local.get 0; block (local.get 1; local.get 0; br_if 1; local.set 1); local.set 1); local.get 1 *)
Definition innerBody := [LOCAL_GET (mk_uN 1); LOCAL_GET (mk_uN 0); BR_IF (mk_uN 1); LOCAL_SET (mk_uN 1)].
Definition outerBody := [LOCAL_GET (mk_uN 0); BLOCK (_RESULT None) innerBody; LOCAL_SET (mk_uN 1)].
Definition unwindingCode := [BLOCK (_RESULT None) outerBody; LOCAL_GET (mk_uN 1)].
Definition unwindingModule := MKmoduleinst [i32StackType 1 1] [0%N] [] [] [] [] [] [].
Definition unwindingFunction := i32Function unwindingModule (mk_uN 0) 1 1 1 unwindingCode.
Definition unwindingProgram := [unwindingFunction].
Definition unwindingStore := MKstore unwindingProgram [] [] [] [] [].

(* The argument 7 and the zeroed local, as the entry and zero rows before the run left them. *)
Definition unwindingMemory := loadProgram unwindingProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some 0 | [3] => Some 0 | _ => None end
  else None).
Definition unwindingWitness := map rowWitness unwindingRows.

Lemma unwinding_run_checks : validRunb unwindingMemory unwindingWitness = true.
Proof. vm_compute. reflexivity. Qed.
