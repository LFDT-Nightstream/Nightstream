(* The proof invariant between two rows, a relational form of ControlDecode: the
   active frame at a code path and the suspended callers beneath it, with their
   operands, locals and saved contexts in memory. At code, typing follows from
   validation at the frame's position (FrameFacts). Entry carries typed arguments
   and the defaults of the function's declared locals. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout FrameInitialization Locals Fragment ControlDecode.
Import ListNotations.
Open Scope Z_scope.

(* The i32 frame representation used by the current row proofs. The decoder
   reads typed values directly; ControlBridge connects the two under functionAt's
   current i32 restriction. *)
Definition activeFrame (f : GuestFunction) (locals : list Z) : frame :=
  MKframe (List.map i32Value locals) (guestModule f).

(* A frame's operands from its base; while a move is pending, the top `remaining`
   operands still sit `distance` cells higher. *)
Definition readOperands (memory : MemoryState) (base : Z) (count remaining : nat) (distance : Z) : option (list Z) :=
  match readStack memory base (count - remaining),
    readStack memory (base + Z.of_nat (count - remaining) + distance) remaining with
  | Some low, Some high => Some (low ++ high)
  | _, _ => None
  end.


(* A configuration inside its enclosing labels: each label's operands and the
   code after its END stay outside its LABEL_. *)
Fixpoint wrapLabels (code : list instr) (labels : list LabelSite) (segments : list (list Z)) (c : config) : config :=
  match labels, segments with
  | s :: rest, segment :: more =>
      match siteEntry code s with
      | Some (i, body) =>
          stackPrefix segment (instructionSuffix (skipn (S (siteOffset s)) code)
            (blockContext (labelArity i) (labelContinuation i) (wrapLabels body rest more c)))
      | None => c
      end
  | _, _ => c
  end.


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

(* A frame in memory: its locals are its first cells from fbp, and its operands
   follow from base = fbp + locals, at the code's static heights, with `pending`
   more that have moved to a callee as its arguments. *)
Definition FrameAt (program : Program) (memory : MemoryState) (base fbp : Z) (pending remaining : nat)
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
    memory stackId [2 * operandCell base (length (viewOperands view)) remaining distance index] = Some value) /\
  length (viewLocals view) = frameLocals f /\
  Forall (fun value => 0 <= value < i32Modulus) (viewLocals view) /\
  (forall index value, nth_error (viewLocals view) index = Some value ->
    memory stackId [2 * (fbp + Z.of_nat index)] = Some value).

Inductive ControlPosition :=
| atCode (view : FrameView)
| resolved (view : FrameView) (callee : nat)
| entering (function : nat) (arguments : list val).

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
      let remaining := Z.to_nat (boundaryMoveRemaining boundary) in
      FrameAt program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 0 remaining
        (boundaryMoveDistance boundary) f view /\
      (positionOffset position <= length (innerCode f position))%nat /\
      boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) /\
      boundaryPC boundary =
        guestEntry f + Z.of_nat (pathOffset (guestBody f) (positionLabels position) (positionOffset position)) /\
      initializationFlags boundary /\ boundaryZeroRemaining boundary = 0 /\
      (remaining <= length (viewValues view))%nat /\
      c = frameConfig f view (stackConfig (mk_state s (activeFrame f (viewLocals view))) (viewValues view)
        (skipn (positionOffset position) (innerCode f position)))
  (* The indirect call at the view's position has resolved to `callee`: its
     reference, callee + 1, sits above the view's operands, where the table index
     was. SpecTec has stepped to CALL_ADDR. *)
  | resolved view callee => exists f g i,
      let position := viewPosition view in
      FrameAt program memory (boundaryStackBase boundary) (boundaryLocalsBase boundary) 1 0 0 f view /\
      nth_error (innerCode f position) (positionOffset position) = Some i /\ isIndirect i = true /\
      functionAt program callee = Some g /\ callsInto (programCallees program (guestModule f)) i g /\
      memory stackId [2 * (boundaryStackBase boundary + Z.of_nat (length (viewOperands view)))] =
        Some (Z.of_nat callee + 1) /\
      boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length (viewOperands view)) + 1 /\
      boundaryPC boundary =
        guestEntry f + Z.of_nat (pathOffset (guestBody f) (positionLabels position) (positionOffset position)) + 1 /\
      initializationFlags boundary /\ boundaryZeroRemaining boundary = 0 /\ boundaryMoveRemaining boundary = 0 /\
      c = frameConfig f view (stackPrefix (viewValues view)
        (instructionSuffix (skipn (S (positionOffset position)) (innerCode f position))
          (mk_config (mk_state s (activeFrame f (viewLocals view))) [CALL_ADDR (N.of_nat callee)])))
  | entering function arguments => exists f defaults,
      functionAt program function = Some f /\
      Forall2 (fun t value => integerValueType value = Some t) (guestParamTypes f) arguments /\
      integerZeros (guestLocalTypes f) = Some defaults /\
      InitializingFrame memory boundary arguments (Z.of_nat (guestLocals f)) (boundaryZeroRemaining boundary) /\
      initializationFlags boundary /\ boundaryMoveRemaining boundary = 0 /\
      0 < boundaryZeroRemaining boundary /\
      boundaryPC boundary = guestEntry f /\
      c = valueStackConfig (mk_state s (MKframe (arguments ++ defaults) (guestModule f))) [] (guestBody f)
  end.

(* A caller suspended at a call inside its labels: its view holds the operands
   below the ones the call consumed. *)
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
      exists i,
      FrameAt program memory (callerBase suspended) (callerFbp suspended) (callOperands i callee) 0 0 g view /\
      nth_error (innerCode g position) (positionOffset position) = Some i /\
      callsInto (programCallees program (guestModule g)) i callee /\
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
  | resolved view _ => functionAt program (positionFunction (viewPosition view))
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

(* The activation's original function survives calls and returns inside it. *)
Definition positionFunctionIndex (position : ControlPosition) : nat :=
  match position with
  | atCode view => positionFunction (viewPosition view)
  | resolved view _ => positionFunction (viewPosition view)
  | entering function _ => function
  end.

Fixpoint rootFunction (stack : list SuspendedCall) (active : nat) : nat :=
  match stack with
  | [] => active
  | top :: rest => rootFunction rest (positionFunction (viewPosition (callView top)))
  end.

Definition cursorRoot (cursor : ControlCursor) : nat :=
  rootFunction (suspendedCalls cursor) (positionFunctionIndex (activePosition cursor)).
