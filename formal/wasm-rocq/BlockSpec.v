(* The ROM image of a control instruction with one static successor. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 R1CS MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallSpec BranchBindings SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Definition conditionalBlock (x : localidx) (tail : list instr) : instr :=
  BLOCK (_RESULT None) (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail).

Definition controlImage memory pc opcode next : Prop :=
  memory programOpcodeId [pc] = Some opcode /\ memory pcEdgeKindId [pc] = Some 0 /\
  forall choice, memory pcRomId [pc; choice] = if choice =? 0 then Some next else Some 0.
