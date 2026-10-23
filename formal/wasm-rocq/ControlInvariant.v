(* The proof invariant between two rows, a relational form of ControlDecode: the
   active frame at a code path and the suspended callers beneath it, with their
   operands, locals and saved contexts in memory. Typing is not stored here; it
   follows from the program's validation at each frame's position (FrameFacts). *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout FrameInitialization Locals Fragment ControlDecode.
Import ListNotations.
Open Scope Z_scope.

(* A frame's dynamic state at a code path: the operands below each enclosing
   label, the operands in the innermost code, and the locals. *)
Record FrameView := frameView {
  viewPosition : CodePosition;
  viewSegments : list (list Z);
  viewValues : list Z;
  viewLocals : list Z
}.

Definition viewOperands (view : FrameView) : list Z := concat (viewSegments view) ++ viewValues view.

Definition innerCode (f : GuestFunction) (position : CodePosition) : list instr :=
  match pathCode (guestBody f) (positionLabels position) with Some inner => inner | None => [] end.

(* A configuration of the frame's innermost code, inside the frame's labels. *)
Definition frameConfig (f : GuestFunction) (view : FrameView) (c : config) : config :=
  wrapLabels (guestBody f) (positionLabels (viewPosition view)) (viewSegments view) c.

(* Where a frame's operand `index` sits: from base, except that the top `moved`
   of `count` operands still sit `distance` cells higher while a move is pending. *)
Definition operandCell (base : Z) (count moved : nat) (distance : Z) (index : nat) : Z :=
  base + Z.of_nat index + (if (index <? count - moved)%nat then 0 else distance).

(* A frame in memory: its locals are its first cells from fbp, and its operands
   follow from base = fbp + locals, at the code's static heights, with `pending`
   more that have moved to a callee as its arguments. *)
Definition FrameAt (program : Program) (memory : MemoryState) (base fbp : Z) (pending moved : nat)
    (distance : Z) (f : GuestFunction) (view : FrameView) : Prop :=
  let position := viewPosition view in
  functionAt program (positionFunction position) = Some f /\
  pathCode (guestBody f) (positionLabels position) = Some (innerCode f position) /\
  pathHeights (programCallees program (guestModule f)) (guestBody f) (positionLabels position)
    (positionOffset position) =
    Some (map (@length Z) (viewSegments view), (length (viewValues view) + pending)%nat) /\
  0 <= fbp /\ base = fbp + Z.of_nat (frameLocals f) /\
  Forall (fun value => 0 <= value < i32Modulus) (viewOperands view) /\
  (forall index value, nth_error (viewOperands view) index = Some value ->
    memory stackId [2 * operandCell base (length (viewOperands view)) moved distance index] = Some value) /\
  length (viewLocals view) = frameLocals f /\
  Forall (fun value => 0 <= value < i32Modulus) (viewLocals view) /\
  (forall index value, nth_error (viewLocals view) index = Some value ->
    memory stackId [2 * (fbp + Z.of_nat index)] = Some value).

Inductive ControlPosition :=
| atCode (view : FrameView)
| entering (function : nat) (arguments : list Z).

(* The countdown flags follow the counters, as the circuit computes them on the row
   before this boundary; the next row's kind is fixed by them. A pending move runs
   before zeroing. *)
Definition initializationFlags (boundary : Boundary) : Prop :=
  boundaryMoveActive boundary = (if boundaryMoveRemaining boundary =? 0 then 0 else 1) /\
  boundaryZeroActive boundary =
    (if boundaryMoveRemaining boundary =? 0 then
       if boundaryZeroRemaining boundary =? 0 then 0 else 1 else 0).

(* The active frame. At code, no zeroing is pending, but the top operands may
   still be moving down: SpecTec has already returned them from a callee, and the
   circuit catches up row by row with no SpecTec step. While a callee's frame is
   initialized, SpecTec has already entered it with zeroed locals; the circuit
   catches up the same way. *)
Definition ActiveState (program : Program) (s : store) (memory : MemoryState) (boundary : Boundary)
    (cursor : ControlPosition) (c : config) : Prop :=
  match cursor with
  | atCode view => exists f,
      let position := viewPosition view in
      let moved := Z.to_nat (boundaryMoveRemaining boundary) in
      FrameAt program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0 moved
        (boundaryMoveDistance boundary) f view /\
      (positionOffset position <= length (innerCode f position))%nat /\
      boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) /\
      boundaryPC boundary =
        guestEntry f + Z.of_nat (pathOffset (guestBody f) (positionLabels position) (positionOffset position)) /\
      initializationFlags boundary /\ boundaryZeroRemaining boundary = 0 /\
      (moved <= length (viewValues view))%nat /\
      c = frameConfig f view (stackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
        (skipn (positionOffset position) (innerCode f position)))
  | entering function arguments => exists f,
      functionAt program function = Some f /\ length arguments = guestParams f /\
      Forall (fun value => 0 <= value < i32Modulus) arguments /\
      InitializingFrame memory boundary arguments (Z.of_nat (guestLocals f)) (boundaryZeroRemaining boundary) /\
      initializationFlags boundary /\ boundaryMoveRemaining boundary = 0 /\
      0 < boundaryZeroRemaining boundary /\
      boundaryPC boundary = guestEntry f /\
      c = stackConfig (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f))) [] (guestBody f)
  end.

(* A caller suspended at a direct CALL inside its labels: its view holds the
   operands below the callee's arguments. *)
Record SuspendedCall := suspendedCall {
  callView : FrameView;
  callerFunction : GuestFunction;
  callResults : nat;
  callerBase : Z;
  callerFbp : Z
}.

(* The callee's configuration inside the caller's FRAME_/LABEL_ context and labels. *)
Definition callerConfig (suspended : SuspendedCall) (c : config) : config :=
  let g := callerFunction suspended in
  let view := callView suspended in
  let position := viewPosition view in
  frameConfig g view (stackPrefix (viewValues view)
    (instructionSuffix (skipn (S (positionOffset position)) (innerCode g position))
      (callContext (activeFrame g (viewLocals view)) (N.of_nat (callResults suspended)) c))).

Fixpoint plugCalls (stack : list SuspendedCall) (c : config) : config :=
  match stack with [] => c | suspended :: rest => plugCalls rest (callerConfig suspended c) end.

(* Each suspended call, relative to the base pointer and depth of the frame it
   called: that frame starts right above the caller's operands, its arguments
   having become its first locals. Its return context is the one saved at the
   depth below that frame. *)
Fixpoint SuspendedStack (program : Program) (memory : MemoryState) (callee : GuestFunction)
    (fbp depth : Z) (stack : list SuspendedCall) : Prop :=
  match stack with
  | [] => True
  | suspended :: rest =>
      let g := callerFunction suspended in
      let view := callView suspended in
      let position := viewPosition view in
      exists x,
      FrameAt program memory (callerBase suspended) (callerFbp suspended) (guestParams callee) 0 0 g view /\
      nth_error (innerCode g position) (positionOffset position) = Some (CALL x) /\
      programCallees program (guestModule g) (x :> N) = Some callee /\
      callResults suspended = guestResults callee /\
      fbp = callerBase suspended + Z.of_nat (length (viewOperands view)) /\
      memory callStackReturnPcId [depth - 1] =
        Some (guestEntry g + Z.of_nat (pathOffset (guestBody g) (positionLabels position) (S (positionOffset position)))) /\
      memory callStackCallerFbpId [depth - 1] = Some (callerFbp suspended) /\
      memory callStackCallerSpBaseId [depth - 1] = Some (callerBase suspended) /\
      SuspendedStack program memory g (callerFbp suspended) (depth - 1) rest
  end.

Record ControlCursor := controlCursor {
  suspendedCalls : list SuspendedCall;
  activePosition : ControlPosition
}.

Definition activeFunction (program : Program) (cursor : ControlPosition) : option GuestFunction :=
  match cursor with
  | atCode view => functionAt program (positionFunction (viewPosition view))
  | entering function _ => functionAt program function
  end.

(* The base frame sits at a non-negative baseDepth; each suspended call adds one level. *)
Definition ControlState (program : Program) (s : store) (baseDepth : Z) (cursor : ControlCursor)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  exists f inner,
    activeFunction program (activePosition cursor) = Some f /\
    ActiveState program s memory boundary (activePosition cursor) inner /\
    SuspendedStack program memory f (boundaryLocalsBase boundary)
      (boundaryCallDepth boundary) (suspendedCalls cursor) /\
    0 <= baseDepth /\ boundaryCallDepth boundary = baseDepth + Z.of_nat (length (suspendedCalls cursor)) /\
    c = plugCalls (suspendedCalls cursor) inner.
