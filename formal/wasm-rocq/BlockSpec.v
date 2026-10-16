(* Zero-result blocks with arithmetic, locals and depth-zero branches.
   Branch sites must not require stack unwinding. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 R1CS MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallSpec BranchBindings SpecTecBranches.
Import ListNotations.
Open Scope Z_scope.

Definition conditionalBlock (x : localidx) (tail : list instr) : instr :=
  BLOCK (_RESULT None) (LOCAL_GET x :: BR_IF (mk_uN 0) :: tail).

Definition branchHeight (i : instr) (height : nat) : option nat :=
  match i with BR_IF _ => match height with S h => Some h | _ => None end
  | _ => instructionStackHeight i height end.
Fixpoint branchHeights (code : list instr) (height : nat) : option nat :=
  match code with [] => Some height | i :: rest =>
    match branchHeight i height with Some h => branchHeights rest h | None => None end end.
Definition supportedBranchBody (code : list instr) : Prop :=
  Forall (fun i => i = BR_IF (mk_uN 0) \/ exists op, encodedOpcode i = Some op) code.

(* The current circuit pops the condition but cannot discard other operands.
   This is a restriction on the program, checked on both possible branch paths. *)
Fixpoint branchSitesBalanced (code : list instr) (height : nat) : Prop :=
  match code with
  | [] => True
  | BR_IF _ :: rest => height = 1%nat /\ branchSitesBalanced rest 0%nat
  | i :: rest => match instructionStackHeight i height with
      | Some next => branchSitesBalanced rest next | None => False end
  end.
Definition supportedBlock (bt : blocktype) (body : list instr) : Prop :=
  bt = _RESULT None /\ supportedBranchBody body /\ branchSitesBalanced body 0%nat.

(* Internal proof context. The specification states block validation in SpecTec's
   own block context (ControlSpec.validatedBlockRest); ControlState transports it. *)
Definition blockTypingContext (localCount : nat) : context :=
  MKcontext [] [] [] [] [] [] [] (repeat valtype_I32 localCount) [mk_list _ []] None.

Definition typedBlock (localCount : nat) (body : list instr) : Prop :=
  Instrs_ok (blockTypingContext localCount) body (i32StackType 0 0).

Definition controlImage memory pc opcode next : Prop :=
  memory programOpcodeId [pc] = Some opcode /\ memory pcEdgeKindId [pc] = Some 0 /\
  forall choice, memory pcRomId [pc; choice] = if choice =? 0 then Some next else None.
