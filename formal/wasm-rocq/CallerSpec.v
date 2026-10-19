(* Guest functions as the circuit sees them, and the static facts a caller's code
   must satisfy given the functions it calls. *)
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

(* Fragment restrictions are independent of validation and stack height. *)
Definition supportedCaller (callees : CalleeEnvironment) (code : list instr) : Prop :=
  Forall (fun instruction => match instruction with
    | CALL x => exists f, callees (x :> N) = Some f /\
        (guestParams f < 256)%nat /\ (guestResults f < 256)%nat
    | _ => exists opcode, encodedOpcode instruction = Some opcode
    end) code.

(* Internal numerical facts derived from SpecTec validation. *)
Fixpoint callerFacts (callees : CalleeEnvironment) (z : state) (height : nat)
    (program : list instr) : Prop :=
  match program with
  | [] => True
  | CALL x :: rest =>
      match callees (x :> N) with
      | Some f =>
          typedCall z x (guestModule f) (guestTypeIndex f) (guestParams f)
            (guestResults f) (guestLocals f) (guestBody f) /\
          (guestParams f <= height)%nat /\
          callerFacts callees z (height - guestParams f + guestResults f) rest
      | None => False
      end
  | instruction :: rest =>
      supportedStraightLineProgram [instruction] /\ wf_instr instruction /\
      programTyped (length (stateLocals z)) height [instruction] /\
      match instructionStackHeight instruction height with
      | Some remaining => callerFacts callees z remaining rest
      | None => False
      end
  end.
