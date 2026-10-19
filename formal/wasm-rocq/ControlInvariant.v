(* The proof invariant between two rows, a relational form of ControlDecode: the
   active function's position and the suspended callers beneath it. The active
   position is an outer instruction, a position inside one zero-result block, or a
   callee whose frame is still being initialized. Each suspended caller owns the memory its return restores, and its
   SpecTec continuation wraps the active configuration in a FRAME_/LABEL_ context. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec StackSimulation
  SpecTecCalls SpecTecBranches CallerSpec ProgramSpec CodeLayout BlockSpec FunctionValidation
  FrameInitialization Locals ControlDecode.
Import ListNotations.
Open Scope Z_scope.

Inductive ControlPosition :=
| atInstruction (position : SourcePosition)
| inBlock (position : SourcePosition) (offset : nat) (prefix : list Z)
| entering (function : nat) (arguments : list Z).

(* The countdown flags follow the counters, as the circuit computes them on the row
   before this boundary; the next row's kind is fixed by them. *)
Definition initializationFlags (boundary : Boundary) : Prop :=
  boundaryZeroActive boundary = (if boundaryZeroRemaining boundary =? 0 then 0 else 1) /\
  boundaryParamActive boundary =
    (if boundaryZeroRemaining boundary =? 0 then
       if boundaryParamRemaining boundary =? 0 then 0 else 1 else 0).

(* The active function. Outside frame initialization no countdown is pending. While
   a callee's frame is initialized, SpecTec has already entered it with zeroed
   locals; the circuit catches up row by row, with no SpecTec step. *)
Definition ActiveState (program : Program) (cursor : ControlPosition) (z : state)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  match cursor with
  | atInstruction position =>
      (exists code, SourceState program position z memory boundary code c) /\ initializationIdle boundary
  | inBlock position offset prefix => exists f body continuation values,
      functionAt program (sourceFunction position) = Some f /\
      nth_error (guestBody f) (sourceOffset position) = Some (BLOCK (_RESULT None) body) /\
      codeAt f (S (sourceOffset position)) continuation /\
      frameMatches program f z /\
      validatedCode program f z (length prefix) continuation /\
      supportedBranchBody body /\ (offset <= length body)%nat /\
      validatedBlockRest f z (length values) (skipn offset body) /\
      branchSitesBalanced (skipn offset body) (length values) /\ wf_state z /\
      stackMatches memory boundary (prefix ++ values) /\ localsMatch memory boundary z /\
      boundaryPC boundary = guestEntry f + Z.of_nat (codeOffset (guestBody f) (sourceOffset position))
        + 1 + Z.of_nat offset /\
      initializationIdle boundary /\
      c = stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z values (skipn offset body))))
  | entering function arguments => exists f,
      functionAt program function = Some f /\ length arguments = guestParams f /\
      Forall (fun value => 0 <= value < i32Modulus) arguments /\
      z = mk_state (fun_store z) (enteredFrame (guestModule f) arguments (guestLocals f)) /\
      frameMatches program f z /\ wf_state z /\ validatedCode program f z 0 (guestBody f) /\
      wf_config (stackConfig z [] (guestBody f)) /\
      InitializingFrame memory boundary arguments (Z.of_nat (guestLocals f))
        (boundaryZeroRemaining boundary) (boundaryParamRemaining boundary) /\
      initializationFlags boundary /\
      0 < boundaryZeroRemaining boundary + boundaryParamRemaining boundary /\
      boundaryPC boundary = guestEntry f /\
      c = stackConfig z [] (guestBody f)
  end.

(* A caller suspended at a direct CALL, with what its return restores: the operand
   values below the arguments, its frame, and its frame registers. *)
Record SuspendedCall := suspendedCall {
  callPosition : SourcePosition;
  callerFrame : frame;
  callerValues : list Z;
  callerBase : Z;
  callerFbp : Z;
  callContinuation : list instr;
  callResults : nat
}.

Record ControlCursor := controlCursor {
  suspendedCalls : list SuspendedCall;
  activePosition : ControlPosition
}.

Definition activeFunction (program : Program) (cursor : ControlPosition) : option GuestFunction :=
  match cursor with
  | atInstruction position | inBlock position _ _ => functionAt program (sourceFunction position)
  | entering function _ => functionAt program function
  end.

(* SpecTec's configuration: the active one inside each caller's call context. *)
Fixpoint plugCalls (stack : list SuspendedCall) (c : config) : config :=
  match stack with
  | [] => c
  | suspended :: rest => plugCalls rest (stackPrefix (callerValues suspended) (instructionSuffix (callContinuation suspended)
      (callContext (callerFrame suspended) (N.of_nat (callResults suspended)) c)))
  end.

Definition localsAt (memory : MemoryState) (fbp : Z) (z : state) : Prop :=
  forall index v, nth_error (stateLocals z) index = Some v ->
    exists value, v = i32Value value /\ 0 <= value < i32Modulus /\
      memory localLoId [fbp; Z.of_nat index] = Some value.

(* Each suspended call, relative to the frame registers and depth of the frame it
   called. Its return context is the one saved at the depth below that frame. *)
Fixpoint SuspendedStack (program : Program) (callee : GuestFunction) (base fbp depth : Z) (s : store)
    (memory : MemoryState) (stack : list SuspendedCall) : Prop :=
  match stack with
  | [] => True
  | suspended :: rest => exists g x,
      functionAt program (sourceFunction (callPosition suspended)) = Some g /\
      nth_error (guestBody g) (sourceOffset (callPosition suspended)) = Some (CALL x) /\
      codeAt g (S (sourceOffset (callPosition suspended))) (callContinuation suspended) /\
      programCallees program (guestModule g) (x :> N) = Some callee /\
      callResults suspended = guestResults callee /\
      frameMatches program g (mk_state s (callerFrame suspended)) /\ wf_frame (callerFrame suspended) /\
      validatedCode program g (mk_state s (callerFrame suspended))
        (length (callerValues suspended) + callResults suspended) (callContinuation suspended) /\
      Forall (fun value => 0 <= value < i32Modulus) (callerValues suspended) /\
      0 <= callerBase suspended /\ base = callerBase suspended + Z.of_nat (length (callerValues suspended)) /\
      (forall index value, nth_error (callerValues suspended) index = Some value ->
        memory stackId [2 * (callerBase suspended + Z.of_nat index)] = Some value) /\
      fbp = callerFbp suspended + Z.of_nat (length (LOCALS (callerFrame suspended))) /\
      localsAt memory (callerFbp suspended) (mk_state s (callerFrame suspended)) /\
      memory callStackReturnPcId [depth - 1] =
        Some (guestEntry g + Z.of_nat (codeOffset (guestBody g) (S (sourceOffset (callPosition suspended))))) /\
      memory callStackCallerFbpId [depth - 1] = Some (callerFbp suspended) /\
      memory callStackCallerSpBaseId [depth - 1] = Some (callerBase suspended) /\
      SuspendedStack program g (callerBase suspended) (callerFbp suspended) (depth - 1) s memory rest
  end.

(* The base frame sits at a non-negative baseDepth; each suspended call adds one level. *)
Definition ControlState (program : Program) (baseDepth : Z) (cursor : ControlCursor) (z : state)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  exists f inner,
    activeFunction program (activePosition cursor) = Some f /\
    ActiveState program (activePosition cursor) z memory boundary inner /\
    SuspendedStack program f (boundaryStackBase boundary) (boundaryLocalsBase boundary)
      (boundaryCallDepth boundary) (fun_store z) memory (suspendedCalls cursor) /\
    0 <= baseDepth /\ boundaryCallDepth boundary = baseDepth + Z.of_nat (length (suspendedCalls cursor)) /\
    c = plugCalls (suspendedCalls cursor) inner.
