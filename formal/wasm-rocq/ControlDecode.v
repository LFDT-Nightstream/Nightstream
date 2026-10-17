(* The SpecTec configuration a circuit state denotes. A static control shape says
   where each frame is; the decoder reads operand values and locals from memory,
   callers from the saved call contexts, and code from the program. It fails when
   a register or cell disagrees with the shape. The store is not in circuit memory,
   so it is an input. Each frame also yields the validation facts SpecTec's typing
   and the supported fragment require of it. *)
From Stdlib Require Import List ZArith Bool.
Require Import Wasm2 WasmIntegers R1CS MemoryTrace MemoryCatalog StraightLineSpec SpecTecCalls SpecTecBranches
  CallerSpec ProgramSpec CodeLayout BlockSpec FunctionValidation FrameInitialization Locals Fragment.
Import ListNotations.
Open Scope Z_scope.

Local Notation "'let?' x ':=' e 'in' k" := (match e with Some x => k | None => None end)
  (at level 200, x ident, e at level 100, k at level 200).
Local Notation "'check' b 'in' k" := (if b then k else None) (at level 200, b at level 100, k at level 200).

(* The rest of a block body, typed as SpecTec's BLOCK rule types it: the active
   function's context with the block's empty result label prepended. *)
Definition validatedBlockRest (f : GuestFunction) (z : state) (height : nat) (code : list instr) : Prop :=
  exists C, Moduleinst_ok (fun_store z) (guestModule f) C /\
    Instrs_ok (labelContext [] (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f)))) code (i32StackType height 0).

(* Where the active frame is: before an outer instruction, inside a zero-result
   block (with the operand height below its label), or initializing a callee. *)
Inductive ActiveSite :=
| siteInstruction (position : SourcePosition)
| siteBlock (position : SourcePosition) (offset prefixHeight : nat)
| siteEntering (function : nat).

(* The active site and, top first, the CALL each suspended caller is stopped at. *)
Record ControlShape := controlShape { callSites : list SourcePosition; activeSite : ActiveSite }.

Definition siteFunction (site : ActiveSite) : nat :=
  match site with
  | siteInstruction position | siteBlock position _ _ => sourceFunction position
  | siteEntering function => function
  end.

(* An i32 occupies its low memory limb. *)
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

Definition readLocals (memory : MemoryState) (fbp : Z) (count : nat) : option (list Z) :=
  readCells (fun index => memory localLoId [fbp; index]) 0 count.

Definition frameLocals (f : GuestFunction) : nat := (guestParams f + guestLocals f)%nat.

Definition activeFrame (f : GuestFunction) (locals : list Z) : frame :=
  MKframe (List.map i32Value locals) (guestModule f).

(* The flattened address of a body instruction. *)
Definition sourceAddress (f : GuestFunction) (offset : nat) : Z :=
  guestEntry f + Z.of_nat (codeOffset (guestBody f) offset).

Definition blockAt (f : GuestFunction) (offset : nat) : option (list instr) :=
  match nth_error (guestBody f) offset with Some (BLOCK (_RESULT None) body) => Some body | _ => None end.

Definition callAt (f : GuestFunction) (offset : nat) : option funcidx :=
  match nth_error (guestBody f) offset with Some (CALL x) => Some x | _ => None end.

Definition initializationIdleb (boundary : Boundary) : bool :=
  (boundaryZeroRemaining boundary =? 0) && (boundaryParamRemaining boundary =? 0) &&
  (boundaryZeroActive boundary =? 0) && (boundaryParamActive boundary =? 0).

(* The countdown flags follow the counters, as the circuit computes them. *)
Definition initializationFlagsb (boundary : Boundary) : bool :=
  (boundaryZeroActive boundary =? (if boundaryZeroRemaining boundary =? 0 then 0 else 1)) &&
  (boundaryParamActive boundary =?
    (if boundaryZeroRemaining boundary =? 0 then
       if boundaryParamRemaining boundary =? 0 then 0 else 1 else 0)).

(* The active frame. A callee being initialized has, for SpecTec, already been
   entered with its arguments and zeroed locals; the circuit catches up row by row. *)
Definition decodeActive (program : Program) (s : store) (memory : MemoryState) (boundary : Boundary)
    (site : ActiveSite) : option (config * Prop) :=
  let base := boundaryStackBase boundary in
  let height := Z.to_nat (boundarySP boundary - base) in
  match site with
  | siteInstruction position =>
      let? f := functionAt program (sourceFunction position) in
      let offset := sourceOffset position in
      let code := skipn offset (guestBody f) in
      check initializationIdleb boundary && (0 <=? base) && (base <=? boundarySP boundary) &&
        (offset <=? length (guestBody f))%nat && (boundaryPC boundary =? sourceAddress f offset) in
      let? values := readStack memory base height in
      let? locals := readLocals memory (boundaryLocalsBase boundary) (frameLocals f) in
      let z := mk_state s (activeFrame f locals) in
      Some (stackConfig z values code, validatedCode program f z height code)
  | siteBlock position offset prefixHeight =>
      let? f := functionAt program (sourceFunction position) in
      let? body := blockAt f (sourceOffset position) in
      let rest := skipn offset body in
      let continuation := skipn (S (sourceOffset position)) (guestBody f) in
      check initializationIdleb boundary && (0 <=? base) && (base <=? boundarySP boundary) &&
        (prefixHeight <=? height)%nat && (offset <=? length body)%nat &&
        (boundaryPC boundary =? sourceAddress f (sourceOffset position) + 1 + Z.of_nat offset) in
      let? stack := readStack memory base height in
      let? locals := readLocals memory (boundaryLocalsBase boundary) (frameLocals f) in
      let z := mk_state s (activeFrame f locals) in
      let values := skipn prefixHeight stack in
      Some (stackPrefix (firstn prefixHeight stack)
          (instructionSuffix continuation (blockContext (stackConfig z values rest))),
        validatedCode program f z prefixHeight continuation /\ supportedCode blockSite body /\
        validatedBlockRest f z (length values) rest /\ branchSitesBalanced rest (length values))
  | siteEntering function =>
      let? f := functionAt program function in
      check initializationFlagsb boundary &&
        (0 <? boundaryZeroRemaining boundary + boundaryParamRemaining boundary) &&
        (boundaryPC boundary =? guestEntry f) in
      let? arguments := readStack memory base (guestParams f) in
      let z := mk_state s (enteredFrame (guestModule f) arguments (guestLocals f)) in
      Some (stackConfig z [] (guestBody f),
        validatedCode program f z 0 (guestBody f) /\
        InitializingFrame memory boundary arguments (Z.of_nat (guestLocals f))
          (boundaryZeroRemaining boundary) (boundaryParamRemaining boundary))
  end.

(* Each suspended caller, relative to the frame registers and depth of the frame it
   called: its return context is the one saved one level down. The callee's
   configuration goes inside the caller's FRAME_/LABEL_ context. *)
Fixpoint decodeCalls (program : Program) (s : store) (memory : MemoryState) (callee : nat) (base fbp depth : Z)
    (sites : list SourcePosition) (c : config) (obligations : Prop) : option (config * Prop) :=
  match sites with
  | [] => Some (c, obligations)
  | site :: rest =>
      let? g := functionAt program (sourceFunction site) in
      let? f := functionAt program callee in
      let? x := callAt g (sourceOffset site) in
      let? address := nth_error (FUNCS (guestModule g)) (N.to_nat (x :> N)) in
      let? returnPC := memory callStackReturnPcId [depth - 1] in
      let? callerFbp := memory callStackCallerFbpId [depth - 1] in
      let? callerBase := memory callStackCallerSpBaseId [depth - 1] in
      check (N.to_nat address =? callee)%nat && (returnPC =? sourceAddress g (S (sourceOffset site))) &&
        (0 <=? callerBase) && (callerBase <=? base) && (fbp =? callerFbp + Z.of_nat (frameLocals g)) in
      let? values := readStack memory callerBase (Z.to_nat (base - callerBase)) in
      let? locals := readLocals memory callerFbp (frameLocals g) in
      let continuation := skipn (S (sourceOffset site)) (guestBody g) in
      decodeCalls program s memory (sourceFunction site) callerBase callerFbp (depth - 1) rest
        (stackPrefix values (instructionSuffix continuation
          (callContext (activeFrame g locals) (N.of_nat (guestResults f)) c)))
        (obligations /\ supportedCode calleeSite (guestBody f) /\
          validatedCode program g (mk_state s (activeFrame g locals))
            (length values + guestResults f) continuation)
  end.

(* The base frame sits at a non-negative baseDepth; each suspended caller adds one level. *)
Definition decodeControl (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) : option (config * Prop) :=
  check (0 <=? baseDepth) && (boundaryCallDepth boundary =? baseDepth + Z.of_nat (length (callSites shape))) in
  let? active := decodeActive program s memory boundary (activeSite shape) in
  decodeCalls program s memory (siteFunction (activeSite shape)) (boundaryStackBase boundary)
    (boundaryLocalsBase boundary) (boundaryCallDepth boundary) (callSites shape) (fst active) (snd active).

(* The circuit state denotes c: it decodes to c and c's validation facts hold. *)
Definition CircuitConfig (program : Program) (s : store) (baseDepth : Z) (shape : ControlShape)
    (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  store_FUNCS s = program /\ wf_store s /\
  exists obligations,
    decodeControl program s baseDepth shape memory boundary = Some (c, obligations) /\ obligations.

Definition runStartsAt (boundary : Boundary) (rows : list Witness) : Prop :=
  match rows with [] => True | first :: _ => beforeBoundary first = boundary end.

(* The run starts at the given boundary and never leaves the base frame: at the
   base depth a row neither pops nor halts. Calls may be left unfinished. *)
Definition runFragment (boundary : Boundary) (baseDepth : Z) (rows : list Witness) : Prop :=
  runStartsAt boundary rows /\ Forall (callerRow baseDepth) rows.
