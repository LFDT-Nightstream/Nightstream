From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers Locals R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  StraightLineSpec SpecTecCalls CallSpec CallEntry TraceRefinement StraightLineValidation.
Import ListNotations.
Open Scope Z_scope.

Definition readZeroBody := [LOCAL_GET (mk_uN 0)].
Example read_zero_validates :
  Instrs_ok (i32TypingContext 1) readZeroBody (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr.
  - apply local_get; [reflexivity | reflexivity | repeat constructor | repeat constructor].
  - repeat constructor.
  - repeat constructor.
Qed.

Definition callModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition readZeroFunction := i32Function callModule (mk_uN 0) 0 1 1 readZeroBody.
Definition callState := mk_state (MKstore [readZeroFunction] [] [] [] [] [])
  (MKframe [i32Value 5] callModule).
