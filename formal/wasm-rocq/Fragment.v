(* The supported Wasm fragment: one constructor per supported instruction, with its
   restrictions, each instruction's operand-stack effect, and the stack discipline
   at branch sites. Supporting a new instruction, or relaxing a restriction, is a
   change here. Validation is separate (ProgramSpec). *)
From Stdlib Require Import List ZArith NArith Arith Lia.
Require Import Wasm2 StraightLineSpec CallerSpec.
Import ListNotations.

(* The operands a block leaves: its label's arity. *)
Definition blockResults (bt : blocktype) : nat := match bt with _RESULT (Some _) => 1 | _ => 0 end.

(* The operands below a structured instruction's label when it runs at `height`:
   an if first pops its condition. *)
Definition labelBase (i : instr) (height : nat) : nat :=
  match i with IFELSE _ _ _ => height - 1 | _ => height end.

(* The operands a branch to a structured instruction's label keeps: its results,
   or a loop's parameters, which a result type does not have. *)
Definition labelArity (i : instr) : nat :=
  match i with BLOCK bt _ | IFELSE bt _ _ => blockResults bt | _ => 0 end.

(* The operands a structured instruction leaves when control falls out of it. *)
Definition labelResults (i : instr) : nat :=
  match i with BLOCK bt _ | LOOP bt _ | IFELSE bt _ _ => blockResults bt | _ => 0 end.

(* What a branch to a structured instruction's label runs next: a loop again. *)
Definition labelContinuation (i : instr) : list instr :=
  match i with LOOP bt body => [LOOP bt body] | _ => [] end.

(* The operand height at the next instruction in program order, when control falls
   through to it. None where nothing falls through (after a return), or where the
   operands would underflow, which validation rules out. *)
Definition sourceHeight (callees : CalleeEnvironment) (i : instr) (height : nat) : option nat :=
  match i with
  | BLOCK bt _ | LOOP bt _ => Some (height + blockResults bt)%nat
  | IFELSE bt _ _ => match height with S below => Some (below + blockResults bt)%nat | O => None end
  | RETURN | BR _ | BR_TABLE _ _ | UNREACHABLE => None
  | CALL x => match callees (x :> N) with
      | Some f => if Nat.leb (guestParams f) height then Some (height - guestParams f + guestResults f)%nat else None
      | None => None
      end
  | CALL_INDIRECT _ y => match nth_error (envTypes callees) (N.to_nat (y :> N)) with
      | Some (mk_functype (mk_list params) (mk_list results)) =>
          if Nat.leb (S (length params)) height then Some (height - S (length params) + length results)%nat else None
      | None => None
      end
  | BR_IF _ => match height with S below => Some below | O => None end
  | NOP => Some height
  | _ => instructionStackHeight i height
  end.

(* A call instruction that enters `callee`: a direct call naming it, or an
   indirect call whose expected type is its type. *)
Definition callsInto (callees : CalleeEnvironment) (i : instr) (callee : GuestFunction) : Prop :=
  match i with
  | CALL x => callees (x :> N) = Some callee
  | CALL_INDIRECT _ y => nth_error (envTypes callees) (N.to_nat (y :> N)) = Some (guestType callee)
  | _ => False
  end.

(* The operands a call consumes: the callee's arguments, and an indirect call's
   table index. *)
Definition callOperands (i : instr) (callee : GuestFunction) : nat :=
  match i with CALL_INDIRECT _ _ => S (guestParams callee) | _ => guestParams callee end.

Lemma calls_into_height : forall callees i callee below, callsInto callees i callee ->
  sourceHeight callees i (below + callOperands i callee) = Some (below + guestResults callee)%nat.
Proof.
  intros callees i callee below H. destruct i; try contradiction; cbn [callsInto] in H;
    cbn [sourceHeight callOperands]; rewrite H.
  - destruct (Nat.leb_spec (guestParams callee) (below + guestParams callee)); [f_equal; lia | lia].
  - unfold guestType. cbv beta iota. rewrite !repeat_length.
    destruct (Nat.leb_spec (S (guestParams callee)) (below + S (guestParams callee))); [f_equal; lia | lia].
Qed.

(* An instruction that leaves its function from `depth` enclosing labels: a return,
   or a br to the function's own label. *)
Definition returnsFrom (depth : nat) (i : instr) : Prop :=
  i = RETURN \/ exists l, i = BR l /\ N.to_nat (l :> N) = depth.

Fixpoint sourceHeights (callees : CalleeEnvironment) (code : list instr) (height : nat) : option nat :=
  match code with
  | [] => Some height
  | i :: rest => match sourceHeight callees i height with
      | Some next => sourceHeights callees rest next | None => None end
  end.

(* A br_if or br_table targets an enclosing label, never the function's; a br may
   also target the function's: `exits` counts the enclosing labels. A taken branch discards the operands above its target's base.
   Code nothing falls through to is dead and unconstrained. *)
Fixpoint instrBranches (callees : CalleeEnvironment) (i : instr) (height exits : nat) : Prop :=
  match i with
  | BR_IF l => (N.to_nat (l :> N) < exits)%nat
  (* A br may also leave the function's own label, past every enclosing one. *)
  | BR l => (N.to_nat (l :> N) <= exits)%nat
  | BR_TABLE ls l => Forall (fun target => N.to_nat (target :> N) < exits)%nat (l :: ls)
  | BLOCK _ body | LOOP _ body =>
      (fix bodyBranches (code : list instr) (h : nat) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrBranches callees j h (S exits) /\
             match sourceHeight callees j h with Some next => bodyBranches rest next | None => True end
         end) body 0%nat
  | IFELSE _ t e =>
      (fix bodyBranches (code : list instr) (h : nat) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrBranches callees j h (S exits) /\
             match sourceHeight callees j h with Some next => bodyBranches rest next | None => True end
         end) t 0%nat /\
      (fix bodyBranches (code : list instr) (h : nat) : Prop :=
         match code with
         | [] => True
         | j :: rest => instrBranches callees j h (S exits) /\
             match sourceHeight callees j h with Some next => bodyBranches rest next | None => True end
         end) e 0%nat
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
(* Branches to an enclosing label, where codeBranches allows them. *)
| supported_br_if : forall l, Supported callees (BR_IF l)
| supported_br : forall l, Supported callees (BR l)
| supported_br_table : forall ls l, Supported callees (BR_TABLE ls l)
(* Returns from any depth of labels, discarding the operands below the results. *)
| supported_return : Supported callees RETURN
(* Ifs like blocks, each arm in the fragment. *)
| supported_if : forall result t e, Forall (Supported callees) t -> Forall (Supported callees) e ->
    (result = None \/ result = Some valtype_I32) -> Supported callees (IFELSE (_RESULT result) t e)
(* Loops like blocks: branches to a loop's label keep nothing and run it again. *)
| supported_loop : forall result body, Forall (Supported callees) body ->
    (result = None \/ result = Some valtype_I32) -> Supported callees (LOOP (_RESULT result) body)
(* Indirect calls to i32 functions, typed as the call metadata allows. Any
   function may be called, so every function must be a guest of this module. *)
| supported_call_indirect : forall x y params results,
    nth_error (envTypes callees) (N.to_nat (y :> N)) = Some (i32StackType params results) ->
    (params < 256)%nat -> (results < 256)%nat -> indirectReady callees ->
    Supported callees (CALL_INDIRECT x y)
| supported_nop : Supported callees NOP
| supported_unreachable : Supported callees UNREACHABLE.

Definition supportedCode (callees : CalleeEnvironment) (code : list instr) : Prop := Forall (Supported callees) code.
