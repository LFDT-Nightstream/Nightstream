(* The ROM image of a control instruction with one static successor. *)
Require Import InstructionHeader.
From Stdlib Require Import List ZArith.
Require Import Wasm2 R1CS MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallSpec BranchBindings SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Definition conditionalBlock (x : localidx) (tail : list instr) : instr :=
  BLOCK (_RESULT None) (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail).

Definition controlImage (memory : MemoryState) (pc opcode next : Z) : Prop :=
  opcodeAt memory [pc] = Some opcode /\ edgeKindAt memory [pc] = Some 0 /\
  if existsb (Z.eqb opcode) fallthroughCodes then next = pc + 1 else memory pcRomId [pc] = Some next.
