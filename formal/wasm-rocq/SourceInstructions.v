(* Source instruction support is distinct from validation. Calls still have
   straight-line callees; blocks have the conditional shape in BlockSpec. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 StraightLineSpec CallerSpec BlockSpec CodeLayout.
Import ListNotations.

Definition supportedSourceInstruction callees i : Prop :=
  match i with BLOCK bt body => supportedBlock bt body | _ => supportedCaller callees [i] end.
Definition supportedSourceCode callees code := Forall (supportedSourceInstruction callees) code.

Definition sourceHeight (callees : CalleeEnvironment) (i : instr) (height : nat) : option nat :=
  match i with
  | BLOCK _ _ => Some height
  | CALL x => match callees (x :> N) with
      | Some f => Some (height - guestParams f + guestResults f)%nat | None => None end
  | _ => instructionStackHeight i height
  end.
Fixpoint sourceHeights callees code height : option nat :=
  match code with [] => Some height | i :: rest =>
    match sourceHeight callees i height with Some next => sourceHeights callees rest next | None => None end end.

Definition sourceInstructionFacts callees z height i : Prop :=
  match i with
  | BLOCK bt body => supportedBlock bt body /\ wf_instr i /\ typedBlock (length (stateLocals z)) body
  | _ => callerFacts callees z height [i]
  end.
Fixpoint sourceFacts callees z height code : Prop :=
  match code with [] => True | i :: rest =>
    sourceInstructionFacts callees z height i /\
    match sourceHeight callees i height with Some next => sourceFacts callees z next rest | None => False end end.
