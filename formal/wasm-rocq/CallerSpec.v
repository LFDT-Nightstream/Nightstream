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

(* Function indices are local to the calling function's module. *)
Definition CalleeEnvironment := N -> option GuestFunction.
