(* Guest functions as the circuit sees them, and the callees a function's code
   can name. *)
From Stdlib Require Import List ZArith.
Require Import R1CS Columns Wasm2 MemoryTrace StraightLineSpec CallSpec CallScope.
Import ListNotations.
Open Scope Z_scope.

Record GuestFunction := mkGuestFunction {
  guestAddress : N;
  guestModule : moduleinst;
  guestTypeIndex : typeidx;
  guestParams : nat;
  guestResults : nat;
  guestLocals : nat;
  guestBody : list instr;
  guestEntry : Z
}.

(* A guest function's type: i32 parameters and results. *)
Definition guestType (f : GuestFunction) : functype :=
  mk_functype (mk_list _ (repeat valtype_I32 (guestParams f))) (mk_list _ (repeat valtype_I32 (guestResults f))).

(* What a function's code can name: callees by function index, local to its
   module, and the module's types. An indirect call may reach any function, so it
   needs every function in the program to be a guest of this module. *)
Record CalleeEnvironment := calleeEnvironment {
  calleeAt :> N -> option GuestFunction;
  envTypes : list functype;
  indirectReady : Prop
}.
