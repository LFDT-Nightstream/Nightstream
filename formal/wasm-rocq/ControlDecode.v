(* The SpecTec configuration a circuit state denotes. A static control shape says
   where each frame is: a code path through nested blocks. The decoder computes the
   operand height below each enclosing label from the code, reads operand values
   and locals from memory and callers from the saved call contexts, and fails when
   a register or cell disagrees. The store is not in circuit memory, so it is an
   input; the program it holds must validate. *)
From Stdlib Require Import List ZArith Bool.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout FunctionValidation FrameInitialization Locals Fragment.
Import ListNotations.
Open Scope Z_scope.

Local Notation "'let?' x ':=' e 'in' k" := (match e with Some x => k | None => None end)
  (at level 200, x ident, e at level 100, k at level 200).
Local Notation "'check' b 'in' k" := (if b then k else None) (at level 200, b at level 100, k at level 200).

(* A function, the offsets of the BLOCKs enclosing a position (outermost first),
   and the offset in the innermost body. *)
Record CodePosition := codePosition {
  positionFunction : nat;
  positionLabels : list nat;
  positionOffset : nat
}.

(* The active frame runs code, or is a callee whose frame is being initialized. *)
Inductive ActiveSite :=
| siteCode (position : CodePosition)
| siteEntering (function : nat).

(* The active site and, top first, the CALL each suspended caller is stopped at. *)
Record ControlShape := controlShape { callSites : list CodePosition; activeSite : ActiveSite }.

Definition siteFunction (site : ActiveSite) : nat :=
  match site with siteCode position => positionFunction position | siteEntering function => function end.

(* An i32 occupies the low limb of its stack cell. *)
Definition i32Cell (cell : option Z) : option Z :=
  match cell with
  | Some value => if (0 <=? value) && (value <? i32Modulus) then Some value else None
  | None => None
  end.

Fixpoint readCells (cell : Z -> option Z) (first : Z) (count : nat) : option (list Z) :=
  match count with
  | O => Some []
  | S count =>
      let? value := i32Cell (cell first) in
      let? values := readCells cell (first + 1) count in
      Some (value :: values)
  end.

(* Operand values bottom-to-top, and a frame's locals. *)
Definition readStack (memory : MemoryState) (first : Z) (count : nat) : option (list Z) :=
  readCells (fun index => memory stackId [2 * index]) first count.

(* A frame's locals are its first stack cells, from fbp. *)
Definition readLocals (memory : MemoryState) (fbp : Z) (count : nat) : option (list Z) :=
  readStack memory fbp count.

(* A frame's operands from its base; while a move is pending, the top `moved`
   operands still sit `distance` cells higher. *)
Definition readOperands (memory : MemoryState) (base : Z) (count moved : nat) (distance : Z) : option (list Z) :=
  let? low := readStack memory base (count - moved) in
  let? high := readStack memory (base + Z.of_nat (count - moved) + distance) moved in
  Some (low ++ high).

Definition frameLocals (f : GuestFunction) : nat := (guestParams f + guestLocals f)%nat.

Definition activeFrame (f : GuestFunction) (locals : list Z) : frame :=
  MKframe (List.map i32Value locals) (guestModule f).

(* The operand height below each enclosing label, and in the innermost code at
   the offset, from the code's stack effects. *)
Fixpoint pathHeights (callees : CalleeEnvironment) (code : list instr) (labels : list nat) (offset : nat)
    : option (list nat * nat) :=
  match labels with
  | [] => option_map (fun top => ([], top)) (sourceHeights callees (firstn offset code) 0)
  | b :: rest =>
      match nth_error code b, sourceHeights callees (firstn b code) 0 with
      | Some (BLOCK _ body), Some below =>
          option_map (fun heights => (below :: fst heights, snd heights)) (pathHeights callees body rest offset)
      | _, _ => None
      end
  end.

Definition totalHeight (heights : list nat * nat) : nat := fold_right Nat.add (snd heights) (fst heights).

(* The arity of each enclosing label, outermost first. *)
Fixpoint pathArities (code : list instr) (labels : list nat) : list nat :=
  match labels with
  | [] => []
  | b :: rest => match nth_error code b with
      | Some (BLOCK bt body) => blockResults bt :: pathArities body rest | _ => [] end
  end.

(* The operands below each label, then the innermost ones. *)
Fixpoint splitSegments (heights : list nat) (values : list Z) : list (list Z) * list Z :=
  match heights with
  | [] => ([], values)
  | h :: rest => let split := splitSegments rest (skipn h values) in (firstn h values :: fst split, snd split)
  end.

(* A configuration inside its enclosing labels: each label's operands and the
   code after its END stay outside its LABEL_. *)
Fixpoint wrapLabels (code : list instr) (labels : list nat) (segments : list (list Z)) (c : config) : config :=
  match labels, segments with
  | b :: rest, segment :: more =>
      match nth_error code b with
      | Some (BLOCK bt body) =>
          stackPrefix segment (instructionSuffix (skipn (S b) code)
            (blockContext (blockResults bt) (wrapLabels body rest more c)))
      | _ => c
      end
  | _, _ => c
  end.

Definition callAt (code : list instr) (offset : nat) : option funcidx :=
  match nth_error code offset with Some (CALL x) => Some x | _ => None end.

(* The countdown flags follow the counters, as the circuit computes them: a
   pending move runs before zeroing. *)
Definition initializationFlagsb (boundary : Boundary) : bool :=
  (boundaryMoveActive boundary =? (if boundaryMoveRemaining boundary =? 0 then 0 else 1)) &&
  (boundaryZeroActive boundary =?
    (if boundaryMoveRemaining boundary =? 0 then
       if boundaryZeroRemaining boundary =? 0 then 0 else 1 else 0)).

(* The active frame. A frame at code may still be moving its top operands (a
   callee's results) down to their cells; SpecTec has already returned. A callee
   being initialized has, for SpecTec, already been entered with its arguments and
   zeroed locals; the circuit catches up row by row. *)
Definition decodeActive (program : Program) (s : store) (memory : MemoryState) (boundary : Boundary)
    (site : ActiveSite) : option (config * Prop) :=
  let base := boundaryStackBase boundary in
  let fbp := boundaryLocalsBase boundary in
  match site with
  | siteCode position =>
      let labels := positionLabels position in
      let offset := positionOffset position in
      let moved := Z.to_nat (boundaryMoveRemaining boundary) in
      let distance := boundaryMoveDistance boundary in
      let? f := functionAt program (positionFunction position) in
      let? inner := pathCode (guestBody f) labels in
      let? heights := pathHeights (programCallees program (guestModule f)) (guestBody f) labels offset in
      check initializationFlagsb boundary && (boundaryZeroRemaining boundary =? 0) &&
        (moved <=? snd heights)%nat &&
        (0 <=? fbp) && (base =? fbp + Z.of_nat (frameLocals f)) &&
        (boundarySP boundary =? base + Z.of_nat (totalHeight heights)) && (offset <=? length inner)%nat &&
        (boundaryPC boundary =? guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) in
      let? stack := readOperands memory base (totalHeight heights) moved distance in
      let? locals := readLocals memory fbp (frameLocals f) in
      let split := splitSegments (fst heights) stack in
      Some (wrapLabels (guestBody f) labels (fst split)
        (stackConfig (mk_state s (activeFrame f locals)) (snd split) (skipn offset inner)), True)
  | siteEntering function =>
      let? f := functionAt program function in
      check initializationFlagsb boundary && (boundaryMoveRemaining boundary =? 0) &&
        (0 <? boundaryZeroRemaining boundary) && (boundaryPC boundary =? guestEntry f) in
      let? arguments := readStack memory fbp (guestParams f) in
      Some (stackConfig (mk_state s (enteredFrame (guestModule f) arguments (guestLocals f))) [] (guestBody f),
        InitializingFrame memory boundary arguments (Z.of_nat (guestLocals f)) (boundaryZeroRemaining boundary))
  end.

(* Each suspended caller, relative to the frame base pointer and depth of the
   frame it called: its return context is the one saved one level down. The
   callee's frame starts right above the caller's operands below its arguments,
   which became the callee's first locals. The callee's configuration goes inside
   the caller's FRAME_/LABEL_ context, inside the caller's enclosing labels. *)
Fixpoint decodeCalls (program : Program) (memory : MemoryState) (callee : nat) (fbp depth : Z)
    (sites : list CodePosition) (c : config) (obligations : Prop) : option (config * Prop) :=
  match sites with
  | [] => Some (c, obligations)
  | site :: rest =>
      let labels := positionLabels site in
      let offset := positionOffset site in
      let? g := functionAt program (positionFunction site) in
      let? f := functionAt program callee in
      let? inner := pathCode (guestBody g) labels in
      let? x := callAt inner offset in
      let? address := nth_error (FUNCS (guestModule g)) (N.to_nat (x :> N)) in
      let? heights := pathHeights (programCallees program (guestModule g)) (guestBody g) labels offset in
      let? returnPC := memory callStackReturnPcId [depth - 1] in
      let? callerFbp := memory callStackCallerFbpId [depth - 1] in
      let? callerBase := memory callStackCallerSpBaseId [depth - 1] in
      let below := (totalHeight heights - guestParams f)%nat in
      check (N.to_nat address =? callee)%nat && (guestParams f <=? snd heights)%nat &&
        (returnPC =? guestEntry g + Z.of_nat (pathOffset (guestBody g) labels (S offset))) &&
        (0 <=? callerFbp) && (callerBase =? callerFbp + Z.of_nat (frameLocals g)) &&
        (fbp =? callerBase + Z.of_nat below) in
      let? stack := readStack memory callerBase below in
      let? locals := readLocals memory callerFbp (frameLocals g) in
      let split := splitSegments (fst heights) stack in
      decodeCalls program memory (positionFunction site) callerFbp (depth - 1) rest
        (wrapLabels (guestBody g) labels (fst split)
          (stackPrefix (snd split) (instructionSuffix (skipn (S offset) inner)
            (callContext (activeFrame g locals) (N.of_nat (guestResults f)) c))))
        obligations
  end.

(* The base frame sits at a non-negative baseDepth; each suspended caller adds one level. *)
Definition decodeControl (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) : option (config * Prop) :=
  check (0 <=? baseDepth) && (boundaryCallDepth boundary =? baseDepth + Z.of_nat (length (callSites shape))) in
  let? active := decodeActive program s memory boundary (activeSite shape) in
  decodeCalls program memory (siteFunction (activeSite shape))
    (boundaryLocalsBase boundary) (boundaryCallDepth boundary) (callSites shape) (fst active) (snd active).

(* The circuit state denotes c: the store holds the program, which validates, and
   the state decodes to c. *)
Definition CircuitConfig (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  store_FUNCS s = program /\ wf_store s /\ programValidated program s /\
  exists obligations, decodeControl program s baseDepth shape memory boundary = Some (c, obligations) /\ obligations.

Definition runStartsAt (boundary : Boundary) (rows : list Witness) : Prop :=
  match rows with [] => True | first :: _ => beforeBoundary first = boundary end.

(* The run starts at the given boundary and never leaves the base frame: at the
   base depth a row neither pops nor halts. Calls may be left unfinished. *)
Definition runFragment (boundary : Boundary) (baseDepth : Z) (rows : list Witness) : Prop :=
  runStartsAt boundary rows /\ Forall (callerRow baseDepth) rows.
