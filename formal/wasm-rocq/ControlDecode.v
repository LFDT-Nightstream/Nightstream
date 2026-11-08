(* The SpecTec configuration a circuit state denotes. A static control shape says
   where each frame is: a code path through nested labels. The decoder computes the
   operand types below each enclosing label from the code, reads operand values
   and locals from memory and callers from the saved call contexts, and fails when
   a register or cell disagrees. The store is not in circuit memory, so it is an
   input; the program it holds must validate. *)
From Stdlib Require Import List ZArith Bool.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec TableState CodeLayout FunctionValidation FrameInitialization Locals Fragment.
Require Export IntegerDecode StackTypes.
Import ListNotations.
Open Scope Z_scope.

Local Notation "'let?' x ':=' e 'in' k" := (match e with Some x => k | None => None end)
  (at level 200, x ident, e at level 100, k at level 200).
Local Notation "'check' b 'in' k" := (if b then k else None) (at level 200, b at level 100, k at level 200).

(* A function, the labels enclosing a position (outermost first), and the offset
   in the innermost body. *)
Record CodePosition := codePosition {
  positionFunction : nat;
  positionLabels : list LabelSite;
  positionOffset : nat
}.

(* The active frame runs code, has resolved the indirect call at its position, or
   is a callee whose frame is being initialized. *)
Inductive ActiveSite :=
| siteCode (position : CodePosition)
| siteResolved (position : CodePosition)
| siteEntering (function : nat).

(* The active site and, top first, the CALL each suspended caller is stopped at. *)
Record ControlShape := controlShape { callSites : list CodePosition; activeSite : ActiveSite }.

Definition siteFunction (site : ActiveSite) : nat :=
  match site with
  | siteCode position | siteResolved position => positionFunction position
  | siteEntering function => function
  end.

(* Operand values bottom-to-top, and a frame's locals. *)
Definition readStack (memory : MemoryState) (first : Z) (count : nat) : option (list Z) :=
  readCells (fun index => memory stackId [2 * index]) first count.

(* A frame's locals are its first stack cells, from fbp. *)
Definition readLocals (memory : MemoryState) (fbp : Z) (types : list valtype) : option (list val) :=
  readIntegerCells memory (fun index => fbp + Z.of_nat index) types.


Definition frameLocals (f : GuestFunction) : nat := (guestParams f + guestLocals f)%nat.

Definition totalHeight (heights : list nat * nat) : nat := fold_right Nat.add (snd heights) (fst heights).

(* The arity of each enclosing label, outermost first. *)
Fixpoint pathArities (code : list instr) (labels : list LabelSite) : list nat :=
  match labels with
  | [] => []
  | s :: rest => match siteEntry code s with Some (i, body) => labelArity i :: pathArities body rest | None => [] end
  end.

(* The operands below each label, then the innermost ones. *)
Fixpoint splitSegments {A : Type} (heights : list nat) (values : list A) : list (list A) * list A :=
  match heights with
  | [] => ([], values)
  | h :: rest => let split := splitSegments rest (skipn h values) in (firstn h values :: fst split, snd split)
  end.

Definition valueStackConfig (z : state) (values : list val) (code : list instr) : config :=
  mk_config z (map admininstr_val values ++ map admininstr_instr code).

Definition valueStackPrefix (values : list val) (c : config) : config :=
  match c with mk_config z code => mk_config z (map admininstr_val values ++ code) end.

(* A configuration inside its enclosing labels: each label's operands and the
   code after its END stay outside its LABEL_. *)
Fixpoint wrapValueLabels (code : list instr) (labels : list LabelSite) (segments : list (list val))
    (c : config) : config :=
  match labels, segments with
  | site :: rest, segment :: more =>
      match siteEntry code site with
      | Some (i, body) =>
          valueStackPrefix segment (instructionSuffix (skipn (S (siteOffset site)) code)
            (blockContext (labelArity i) (labelContinuation i) (wrapValueLabels body rest more c)))
      | None => c
      end
  | _, _ => c
  end.

Definition typeHeights (types : list (list valtype) * list valtype) : list nat * nat :=
  (map (@length valtype) (fst types), length (snd types)).

Definition operandTypes (types : list (list valtype) * list valtype) : list valtype :=
  concat (fst types) ++ snd types.


Definition isIndirect (i : instr) : bool := match i with CALL_INDIRECT _ _ => true | _ => false end.

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
      let remaining := Z.to_nat (boundaryMoveRemaining boundary) in
      let distance := boundaryMoveDistance boundary in
      let? f := functionAt program (positionFunction position) in
      let? inner := pathCode (guestBody f) labels in
      let? types := pathTypes (guestFrameTypes f) (programCallees program (guestModule f))
        (guestBody f) labels offset in
      let heights := typeHeights types in
      check initializationFlagsb boundary && (boundaryZeroRemaining boundary =? 0) &&
        (0 <=? fbp) && (base =? fbp + Z.of_nat (frameLocals f)) &&
        (boundarySP boundary =? base + Z.of_nat (totalHeight heights)) &&
        (boundaryPC boundary =? guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset)) &&
        (remaining <=? snd heights)%nat && (offset <=? length inner)%nat in
      let? stack := readIntegerOperands memory base (operandTypes types) remaining distance in
      let? locals := readLocals memory fbp (guestFrameTypes f) in
      let split := splitSegments (fst heights) stack in
      Some (wrapValueLabels (guestBody f) labels (fst split)
        (valueStackConfig (mk_state s (MKframe locals (guestModule f))) (snd split) (skipn offset inner)), True)
  (* An indirect call has resolved: its function reference replaced the table
     index on top, and SpecTec calls that address next. *)
  | siteResolved position =>
      let labels := positionLabels position in
      let offset := positionOffset position in
      let? f := functionAt program (positionFunction position) in
      let? inner := pathCode (guestBody f) labels in
      let? i := nth_error inner offset in
      let? types := pathTypes (guestFrameTypes f) (programCallees program (guestModule f))
        (guestBody f) labels offset in
      let heights := typeHeights types in
      let? reference := memory stackId [2 * (base + Z.of_nat (totalHeight heights - 1))] in
      let callee := Z.to_nat (reference - 1) in
      let? g := functionAt program callee in
      check initializationFlagsb boundary && (boundaryZeroRemaining boundary =? 0) &&
        (0 <=? fbp) && (base =? fbp + Z.of_nat (frameLocals f)) &&
        (boundarySP boundary =? base + Z.of_nat (totalHeight heights)) &&
        (boundaryPC boundary =? guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset) + 1) &&
        (boundaryMoveRemaining boundary =? 0) && isIndirect i && callsIntob (guestModule f) i callee g &&
        (1 <=? reference) && (1 <=? snd heights)%nat in
      let? stack := readIntegerOperands memory base (firstn (totalHeight heights - 1) (operandTypes types)) 0 0 in
      let? locals := readLocals memory fbp (guestFrameTypes f) in
      let split := splitSegments (fst heights) stack in
      Some (wrapValueLabels (guestBody f) labels (fst split) (valueStackPrefix (snd split)
        (instructionSuffix (skipn (S offset) inner)
          (mk_config (mk_state s (MKframe locals (guestModule f))) [CALL_ADDR (N.of_nat callee)]))), True)
  (* Entry initialization still uses the i32 InitializingFrame invariant. *)
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
      let? i := nth_error inner offset in
      let? types := pathTypes (guestFrameTypes g) (programCallees program (guestModule g))
        (guestBody g) labels offset in
      let heights := typeHeights types in
      let? returnPC := memory callStackReturnPcId [depth - 1] in
      let? callerFbp := memory callStackCallerFbpId [depth - 1] in
      let? callerBase := memory callStackCallerSpBaseId [depth - 1] in
      let below := (totalHeight heights - callOperands i f)%nat in
      check callsIntob (guestModule g) i callee f && (callOperands i f <=? snd heights)%nat &&
        (returnPC =? guestEntry g + Z.of_nat (pathOffset (guestBody g) labels (S offset))) &&
        (0 <=? callerFbp) && (callerBase =? callerFbp + Z.of_nat (frameLocals g)) &&
        (fbp =? callerBase + Z.of_nat below) in
      let? stack := readIntegerCells memory (fun index => callerBase + Z.of_nat index)
        (firstn below (operandTypes types)) in
      let? locals := readLocals memory callerFbp (guestFrameTypes g) in
      let split := splitSegments (fst heights) stack in
      decodeCalls program memory (positionFunction site) callerFbp (depth - 1) rest
        (wrapValueLabels (guestBody g) labels (fst split)
          (valueStackPrefix (snd split) (instructionSuffix (skipn (S offset) inner)
            (callContext (MKframe locals (guestModule g)) (N.of_nat (guestResults f)) c))))
        obligations
  end.

(* The base frame sits at a non-negative baseDepth; each suspended caller adds one level. *)
Definition decodeControl (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) : option (config * Prop) :=
  check (0 <=? baseDepth) && (boundaryCallDepth boundary =? baseDepth + Z.of_nat (length (callSites shape))) in
  let? active := decodeActive program s memory boundary (activeSite shape) in
  decodeCalls program memory (siteFunction (activeSite shape))
    (boundaryLocalsBase boundary) (boundaryCallDepth boundary) (callSites shape) (fst active) (snd active).

(* The circuit state denotes c: the store holds the program, which validates, the
   table memories hold the store's tables, and the state decodes to c. *)
Definition CircuitConfig (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  store_FUNCS s = program /\ wf_store s /\ programValidated program s /\ tablesLoaded s memory /\
  exists obligations, decodeControl program s baseDepth shape memory boundary = Some (c, obligations) /\ obligations.

Definition runStartsAt (boundary : Boundary) (rows : list Witness) : Prop :=
  match rows with [] => True | first :: _ => beforeBoundary first = boundary end.

(* The run starts at the given boundary and never leaves the base frame: at the
   base depth a row neither pops nor halts. Calls may be left unfinished. *)
Definition runFragment (boundary : Boundary) (baseDepth : Z) (rows : list Witness) : Prop :=
  runStartsAt boundary rows /\ Forall (callerRow baseDepth) rows.
