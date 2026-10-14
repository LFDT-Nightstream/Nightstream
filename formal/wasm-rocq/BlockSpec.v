(* Complete zero-result blocks. A local supplies the condition; the fallthrough
   tail is straight-line code. There are no nested labels or calls in this body. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 R1CS MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallSpec BranchBindings SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Definition conditionalBlock (x : localidx) (tail : list instr) : instr :=
  BLOCK (_RESULT None) (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail).

Definition supportedBlock (bt : blocktype) (body : list instr) : Prop :=
  bt = _RESULT None /\ exists x tail,
    body = LOCAL_GET x :: BR_IF (mk_uN 0) :: tail /\ supportedStraightLineProgram tail.

Definition blockTypingContext (localCount : nat) : context :=
  MKcontext [] [] [] [] [] [] [] (repeat valtype_I32 localCount) [mk_list _ []] None.

Definition typedBlock (localCount : nat) (body : list instr) : Prop :=
  Instrs_ok (blockTypingContext localCount) body (i32StackType 0 0).

Definition blockExitPC (pc : Z) (tail : list instr) : Z := pc + 4 + Z.of_nat (length tail).

Definition controlImage memory pc opcode next : Prop :=
  memory programOpcodeId [pc] = Some opcode /\ memory pcEdgeKindId [pc] = Some 0 /\
  forall choice, memory pcRomId [pc; choice] = if choice =? 0 then Some next else None.

Definition loadedBlock memory pc x tail : Prop :=
  controlImage memory pc CODE_BLOCK (pc + 1) /\
  instructionImage memory (pc + 1) (LOCAL_GET x) /\
  branchImage memory (pc + 2) (pc + 3) (blockExitPC pc tail) /\
  loadedBody memory (pc + 3) tail /\
  controlImage memory (pc + 3 + Z.of_nat (length tail)) CODE_END (blockExitPC pc tail).
