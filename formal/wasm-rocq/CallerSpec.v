(* Internal instruction/call segments for sequential callers. TraceSegmentation
   derives this grouping from rows; nested calls remain outside the fragment. *)
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

(* Function indices are local to this caller's module. loadedCall checks each
   used entry against that module's addresses, SpecTec's store and the ROMs. *)
Definition CalleeEnvironment := N -> option GuestFunction.

(* Fragment restrictions are independent of validation and stack height. *)
Definition supportedCaller (callees : CalleeEnvironment) (code : list instr) : Prop :=
  Forall (fun instruction => match instruction with
    | CALL x => exists f, callees (x :> N) = Some f /\
        supportedStraightLineProgram (guestBody f) /\
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

Fixpoint loadedCaller (callees : CalleeEnvironment) (z : state) (memory : MemoryState) (pc : Z)
    (program : list instr) : Prop :=
  match program with
  | [] => True
  | instruction :: rest =>
      (match instruction with
       | CALL x =>
           match callees (x :> N) with
           | Some f => loadedCall memory pc (pc + 1) z x (guestAddress f)
               (guestModule f) (guestTypeIndex f) (guestParams f) (guestResults f)
               (guestLocals f) (guestBody f) (guestEntry f)
           | None => False
           end
       | _ => instructionImage memory pc instruction
       end) /\ loadedCaller callees z memory (pc + 1) rest
  end.

(* Internal invariant for composing segments; ProgramSpec.SourceState separates
   the program-wide ROM binding from the active source position. *)
Record CallerState (callees : CalleeEnvironment) (z : state) (memory : MemoryState)
    (boundary : Boundary) (pc : Z) (code : list instr) (configuration : config) : Prop := {
  callerStateLoaded : loadedCaller callees z memory pc code;
  callerStateTyped : callerFacts callees z (stackHeight boundary) code;
  callerStateRepresents : represents z memory boundary code configuration;
  callerStatePC : boundaryPC boundary = pc
}.

Inductive CallerSegment :=
| instructionSegment (row : Witness)
| callSegment (first : Witness) (middle : list Witness) (last : Witness).

Definition segmentRows (segment : CallerSegment) : list Witness :=
  match segment with instructionSegment row => [row] | callSegment first middle last => first :: middle ++ [last] end.
Definition segmentFirst (segment : CallerSegment) : Witness :=
  match segment with instructionSegment row | callSegment row _ _ => row end.
Definition segmentLast (segment : CallerSegment) : Witness :=
  match segment with instructionSegment row | callSegment _ _ row => row end.
Definition callerRows (segments : list CallerSegment) : list Witness := flat_map segmentRows segments.

Definition segmentMatches (instruction : instr) (segment : CallerSegment) : Prop :=
  match instruction, segment with
  | CALL _, callSegment first middle last =>
      first COL_IS_PROGRAM_ROW = 1 /\ last COL_SEL_END = 1 /\ Forall bodyRow middle
  | CALL _, instructionSegment _ | _, callSegment _ _ _ => False
  | _, instructionSegment row => row COL_IS_PROGRAM_ROW = 1
  end.

Definition callerStartsAt (boundary : Boundary) (segments : list CallerSegment) : Prop :=
  match segments with [] => True | segment :: _ => beforeBoundary (segmentFirst segment) = boundary end.

Fixpoint callerFinalBoundary (boundary : Boundary) (segments : list CallerSegment) : Boundary :=
  match segments with [] => boundary | segment :: rest => callerFinalBoundary (afterBoundary (segmentLast segment)) rest end.
