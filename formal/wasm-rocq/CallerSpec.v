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
  guestParamTypes : list valtype;
  guestResultTypes : list valtype;
  guestLocalTypes : list valtype;
  guestBody : list instr;
  guestEntry : Z
}.

Definition guestParams (f : GuestFunction) : nat := length (guestParamTypes f).
Definition guestResults (f : GuestFunction) : nat := length (guestResultTypes f).
Definition guestLocals (f : GuestFunction) : nat := length (guestLocalTypes f).
Definition guestFrameTypes (f : GuestFunction) : list valtype := guestParamTypes f ++ guestLocalTypes f.

Definition guestType (f : GuestFunction) : functype :=
  mk_functype (mk_list _ (guestParamTypes f)) (mk_list _ (guestResultTypes f)).

(* What a function's code can name: callees by function index, local to its
   module, and the module's types. An indirect call may reach any function, so it
   needs every function in the program to be a guest of this module. *)
Record CalleeEnvironment := calleeEnvironment {
  calleeAt :> N -> option GuestFunction;
  envTypes : list functype;
  envTables : list tableaddr;
  indirectReady : Prop
}.
