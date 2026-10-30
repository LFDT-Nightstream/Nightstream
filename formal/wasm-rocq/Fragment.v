(* The supported Wasm fragment: one constructor per supported instruction, with its
   restrictions, each instruction's operand-stack effect, and the stack discipline
   at branch sites. Supporting a new instruction, or relaxing a restriction, is a
   change here. Validation is separate (ProgramSpec). *)
From Stdlib Require Import List ZArith NArith Arith.
Require Import Wasm2 StraightLineSpec CallerSpec.
Import ListNotations.

(* The operands a block leaves: its label's arity. *)
Definition blockResults (bt : blocktype) : nat := match bt with _RESULT (Some _) => 1 | _ => 0 end.

(* The operand height at the next instruction in program order, when control falls
   through to it. None where nothing falls through (after a return), or where the
   operands would underflow, which validation rules out. *)
Definition sourceHeight (callees : CalleeEnvironment) (i : instr) (height : nat) : option nat :=
  match i with
  | BLOCK bt _ => Some (height + blockResults bt)%nat
  | RETURN => None
  | CALL x => match callees (x :> N) with
      | Some f => if Nat.leb (guestParams f) height then Some (height - guestParams f + guestResults f)%nat else None
      | None => None
      end
  | BR_IF _ => match height with S below => Some below | O => None end
  | _ => instructionStackHeight i height
  end.

Fixpoint sourceHeights (callees : CalleeEnvironment) (code : list instr) (height : nat) : option nat :=
  match code with
  | [] => Some height
  | i :: rest => match sourceHeight callees i height with
      | Some next => sourceHeights callees rest next | None => None end
  end.

(* A br_if leaves an enclosing block, never the function: `exits` counts the
   enclosing blocks. A taken branch discards the operands above its target's base.
   Code nothing falls through to is dead and unconstrained. *)
Fixpoint instrBranches (callees : CalleeEnvironment) (i : instr) (height exits : nat) : Prop :=
  match i with
  | BR_IF l => (N.to_nat (l :> N) < exits)%nat
  | BLOCK _ body =>
      (fix bodyBranches (code : list instr) (h : nat) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrBranches callees j h (S exits) /\
             match sourceHeight callees j h with Some next => bodyBranches rest next | None => True end
         end) body 0%nat
  | _ => True
  end.

Definition codeBranches (callees : CalleeEnvironment) (exits : nat) : list instr -> nat -> Prop :=
  fix codeBranches (code : list instr) (h : nat) : Prop :=
    match code with
    | [] => True
    | j :: rest => instrBranches callees j h exits /\
        match sourceHeight callees j h with Some next => codeBranches rest next | None => True end
    end.

Inductive Supported (callees : CalleeEnvironment) : instr -> Prop :=
| supported_i32_add : Supported callees (BINOP I32 (mk_binop__0 Inn_I32 ADD))
| supported_i32_sub : Supported callees (BINOP I32 (mk_binop__0 Inn_I32 SUB))
| supported_local_get : forall x, Supported callees (LOCAL_GET x)
| supported_local_set : forall x, Supported callees (LOCAL_SET x)
(* Direct calls, resolved through the function's callees. The circuit's call
   metadata holds the parameter and result counts in one byte each. *)
| supported_call : forall x f, callees (x :> N) = Some f ->
    (guestParams f < 256)%nat -> (guestResults f < 256)%nat -> Supported callees (CALL x)
(* Blocks without parameters and with at most one i32 result, nested to any depth. *)
| supported_block : forall result body, Forall (Supported callees) body ->
    (result = None \/ result = Some valtype_I32) -> Supported callees (BLOCK (_RESULT result) body)
(* Branches to an enclosing block, where codeBranches allows them. *)
| supported_br_if : forall l, Supported callees (BR_IF l)
(* Returns from any depth of blocks, discarding the operands below the results. *)
| supported_return : Supported callees RETURN.

Definition supportedCode (callees : CalleeEnvironment) (code : list instr) : Prop := Forall (Supported callees) code.
