(* An active source position is either an outer instruction or a position inside
   one zero-result block. The label retains the operand prefix and continuation.
   Calls still complete as segments; nested labels/calls are not supported here. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 R1CS MemoryTrace StraightLineSpec StackSimulation SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout BlockSpec FunctionValidation.
Import ListNotations.
Open Scope Z_scope.

(* The rest of a block body, typed as SpecTec's BLOCK rule types it: the active
   function's context with the block's empty result label prepended. *)
Definition validatedBlockRest (f : GuestFunction) (z : state) (height : nat) (code : list instr) : Prop :=
  exists C, Moduleinst_ok (fun_store z) (guestModule f) C /\
    Instrs_ok (labelContext [] (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f)))) code (i32StackType height 0).

Inductive ControlPosition :=
| atInstruction (position : SourcePosition)
| inBlock (position : SourcePosition) (offset : nat) (prefix : list Z).

Definition ControlState (program : Program) (cursor : ControlPosition) (z : state)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  match cursor with
  | atInstruction position => exists code, SourceState program position z memory boundary code c
  | inBlock position offset prefix => exists f body continuation values,
      functionAt program (sourceFunction position) = Some f /\
      nth_error (guestBody f) (sourceOffset position) = Some (BLOCK (_RESULT None) body) /\
      codeAt f (S (sourceOffset position)) continuation /\
      frameMatches program f z /\
      validatedCode program f z (length prefix) continuation /\
      supportedBranchBody body /\ (offset <= length body)%nat /\
      validatedBlockRest f z (length values) (skipn offset body) /\
      branchSitesBalanced (skipn offset body) (length values) /\ wf_state z /\
      stackMatches memory boundary (prefix ++ values) /\ localsMatch memory boundary z /\
      boundaryPC boundary = guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position))
        + 1 + Z.of_nat offset /\
      c = stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z values (skipn offset body))))
  end.
