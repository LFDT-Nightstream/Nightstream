(* The supported Wasm fragment: one constructor per supported instruction, with the
   restriction on where it may occur, and each instruction's operand-stack effect.
   Supporting a new instruction, or relaxing a restriction, is a change here.
   Validation is separate (ProgramSpec). *)
From Stdlib Require Import List ZArith Arith.
Require Import Wasm2 StraightLineSpec CallerSpec.
Import ListNotations.

(* Where an instruction sits: calls resolve through the function's callees, and a
   branch needs an enclosing block. *)
Record Site := codeSite { siteCallees : CalleeEnvironment; insideBlock : bool }.

(* The operand height after an instruction, or None where it would underflow. *)
Definition sourceHeight (callees : CalleeEnvironment) (i : instr) (height : nat) : option nat :=
  match i with
  | BLOCK _ _ => Some height
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

(* The circuit pops a branch's condition but cannot discard other operands: at each
   branch site only the condition is above the block's base. *)
Fixpoint branchSitesBalanced (callees : CalleeEnvironment) (code : list instr) (height : nat) : Prop :=
  match code with
  | [] => True
  | BR_IF _ :: rest => height = 1%nat /\ branchSitesBalanced callees rest 0
  | i :: rest => match sourceHeight callees i height with
      | Some next => branchSitesBalanced callees rest next | None => False end
  end.

Inductive Supported : Site -> instr -> Prop :=
| supported_i32_add : forall site, Supported site (BINOP I32 (mk_binop__0 Inn_I32 ADD))
| supported_i32_sub : forall site, Supported site (BINOP I32 (mk_binop__0 Inn_I32 SUB))
| supported_local_get : forall site x, Supported site (LOCAL_GET x)
| supported_local_set : forall site x, Supported site (LOCAL_SET x)
(* Direct calls. The circuit's call metadata holds the parameter and result
   counts in one byte each. *)
| supported_call : forall callees inside x f, callees (x :> N) = Some f ->
    (guestParams f < 256)%nat -> (guestResults f < 256)%nat ->
    Supported (codeSite callees inside) (CALL x)
(* Zero-result blocks, nested to any depth, without unwinding at branch sites. *)
| supported_block : forall callees inside body,
    Forall (Supported (codeSite callees true)) body -> branchSitesBalanced callees body 0 ->
    Supported (codeSite callees inside) (BLOCK (_RESULT None) body)
(* Branches to the innermost enclosing block. *)
| supported_br_if : forall callees, Supported (codeSite callees true) (BR_IF (mk_uN 0)).

Definition supportedCode (site : Site) (code : list instr) : Prop := Forall (Supported site) code.
