(* A br_if 0 boundary inside a zero-result block. The stack below the label is
   preserved; at this boundary the block itself holds only the i32 condition.
   For a flat tail, its END is at pc + 1 + length tail.
   Result-carrying branches, deeper labels, and extra discarded values are outside
   this first fragment. No executed-prefix equation describes the branch. *)
From Stdlib Require Import List ZArith.
Require Import R1CS Wasm2 MemoryTrace StraightLineSpec SpecTecCalls SpecTecBranches BranchBindings.
Import ListNotations.
Open Scope Z_scope.

Inductive BlockBranchPosition := BeforeBranch | Fallthrough | AfterBlock.

Definition blockBranchPC (pc : Z) (tail : list instr) (position : BlockBranchPosition) : Z :=
  match position with
  | BeforeBranch => pc
  | Fallthrough => pc + 1
  | AfterBlock => pc + 2 + Z.of_nat (length tail)
  end.

Definition blockBranchImage memory pc tail : Prop :=
  supportedStraightLineProgram tail /\
  branchImage memory pc (blockBranchPC pc tail Fallthrough) (blockBranchPC pc tail AfterBlock).

Definition blockBranchRepresents (z : state) (memory : MemoryState) (boundary : Boundary)
    (pc : Z) (prefix : list Z) (tail continuation : list instr)
    (position : BlockBranchPosition) (c : config) : Prop :=
  wf_state z /\ Forall wf_instr tail /\ Forall wf_instr continuation /\
  localsMatch memory boundary z /\ boundaryPC boundary = blockBranchPC pc tail position /\
  match position with
  | BeforeBranch => exists condition,
      stackMatches memory boundary (prefix ++ [condition]) /\
      c = blockBranchConfig z prefix condition tail continuation
  | Fallthrough => stackMatches memory boundary prefix /\
      c = stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z [] tail)))
  | AfterBlock => stackMatches memory boundary prefix /\ c = stackConfig z prefix continuation
  end.
