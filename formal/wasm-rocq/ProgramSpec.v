(* Source functions, their ROM encoding, and what a program and a run must satisfy.
   This encoder covers i32/i64 guest functions without imports. Blocks occupy
   flattened instruction addresses; each function also has one trailing END. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS Columns Wasm2 MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec CallerSpec FunctionValidation SpecTecCalls CodeLayout Fragment.
Import ListNotations.
Open Scope Z_scope.

Definition Program := list funcinst.

Definition functionBody (f : funcinst) : list instr :=
  match CODE f with func_FUNC _ _ body => body end.

Definition functionSize (f : funcinst) : nat := S (codeSize (functionBody f)).

Definition functionEntry (program : Program) (function : nat) : Z :=
  Z.of_nat (fold_right (fun f size => functionSize f + size)%nat 0%nat (firstn function program)).

Definition integerType (t : valtype) : bool := match t with valtype_I32 | valtype_I64 => true | _ => false end.
Definition integerLocal (l : local) : bool := match l with LOCAL t => integerType t end.

Definition decodeFunction (address : nat) (entry : Z) (f : funcinst) : option GuestFunction :=
  match f with
  | MKfuncinst (mk_functype (mk_list params) (mk_list results)) module (func_FUNC typeIndex locals body) =>
      if forallb integerType params && forallb integerType results && forallb integerLocal locals then
        Some (mkGuestFunction (N.of_nat address) module typeIndex params
          results (map (fun l => match l with LOCAL t => t end) locals) body entry)
      else None
  end.

Definition functionAt (program : Program) (function : nat) : option GuestFunction :=
  match nth_error program function with
  | Some f => decodeFunction function (functionEntry program function) f
  | None => None
  end.

Definition programCallees (program : Program) (module : moduleinst) : CalleeEnvironment :=
  calleeEnvironment
    (fun index => match nth_error (FUNCS module) (N.to_nat index) with
      | Some address => functionAt program (N.to_nat address)
      | None => None
      end)
    (TYPES module) (TABLES module)
    (forall a fi, nth_error program a = Some fi -> exists g, functionAt program a = Some g /\ guestModule g = module).

(* Wasm2's functype_eq_dec rests on an admitted list comparison and does not
   compute; this one compares the parameter and result lists directly. *)
Definition functypeEqDec (t u : functype) : {t = u} + {t <> u}.
Proof.
  destruct t as [[params] [results]], u as [[params'] [results']].
  destruct (list_eq_dec valtype_eq_dec params params') as [-> | Hp]; [ | right; congruence].
  destruct (list_eq_dec valtype_eq_dec results results') as [-> | Hr]; [left; reflexivity | right; congruence].
Defined.

Fixpoint firstSeen (seen types : list functype) : list functype :=
  match types with
  | [] => seen
  | t :: rest =>
      firstSeen (if existsb (fun u => if functypeEqDec t u then true else false) seen then seen else seen ++ [t]) rest
  end.

Fixpoint typePosition (t : functype) (types : list functype) : nat :=
  match types with [] => 0 | u :: rest => if functypeEqDec t u then 0 else S (typePosition t rest) end.

Definition typeId (module : moduleinst) (t : functype) : Z :=
  Z.of_nat (typePosition t (firstSeen [] (TYPES module))) + 1.

(* The decoder's check that call instruction i in `module` enters function
   `callee`, which decodes to f. *)
Definition callsIntob (module : moduleinst) (i : instr) (callee : nat) (f : GuestFunction) : bool :=
  match i with
  | CALL x => match nth_error (FUNCS module) (N.to_nat (x :> N)) with
      | Some address => Nat.eqb (N.to_nat address) callee | None => false end
  | CALL_INDIRECT _ y => match nth_error (TYPES module) (N.to_nat (y :> N)) with
      | Some t => if functypeEqDec t (guestType f) then true else false | None => false end
  | _ => false
  end.

(* Locate an encoded instruction, including the function-ending END. *)
Fixpoint locatePC (program : Program) (pc : nat) : option (nat * nat) :=
  match program with
  | [] => None
  | f :: rest => if (pc <? functionSize f)%nat then Some (0%nat, pc)
      else option_map (fun location => (S (fst location), snd location))
        (locatePC rest (pc - functionSize f))
  end.

Definition instructionAtPC (program : Program) (pc : Z) : option (funcinst * option FlatInstruction) :=
  if pc <? 0 then None else
  match locatePC program (Z.to_nat pc) with
  | Some (function, offset) => match nth_error program function with
      | Some f => Some (f, nth_error (flattenCode (functionBody f)) offset)
      | None => None end
  | None => None
  end.

(* The entry pc of the function an encoded pc belongs to: slots' targets are
   relative to it. *)
Definition pcEntry (program : Program) (pc : Z) : Z :=
  match locatePC program (Z.to_nat pc) with Some (function, _) => functionEntry program function | None => 0 end.

Definition callAddress (f : funcinst) (index : idx) : option N :=
  nth_error (FUNCS (funcinst_MODULE f)) (N.to_nat (index :> N)).

Definition slotOpcode (slot : option FlatInstruction) : option Z :=
  match slot with
  | Some (flatInstruction (opEdge _) _) => None
  | Some (flatInstruction (opInstr (CALL _)) _) => Some CODE_CALL
  | Some (flatInstruction (opInstr (BLOCK _ _)) _) => Some CODE_BLOCK
  | Some (flatInstruction (opInstr (LOOP _ _)) _) => Some CODE_LOOP
  | Some (flatInstruction (opInstr (IFELSE _ _ _)) _) => Some CODE_IF
  | Some (flatInstruction opElse _) => Some CODE_ELSE
  | Some (flatInstruction (opInstr (BR_IF _)) _) => Some CODE_BR_IF
  | Some (flatInstruction (opInstr (BR _)) [None]) => Some CODE_RETURN
  | Some (flatInstruction (opInstr (BR _)) _) => Some CODE_BR
  | Some (flatInstruction (opInstr (BR_TABLE _ _)) _) => Some CODE_BR_TABLE
  | Some (flatInstruction (opInstr RETURN) _) => Some CODE_RETURN
  | Some (flatInstruction (opInstr NOP) _) => Some CODE_NOP
  | Some (flatInstruction (opInstr UNREACHABLE) _) => Some CODE_UNREACHABLE
  | Some (flatInstruction (opInstr (CALL_INDIRECT _ _)) _) => Some CODE_CALL_INDIRECT_RESOLVE
  | Some (flatInstruction opInvoke _) => Some CODE_CALL_INDIRECT
  | Some (flatInstruction (opInstr i) _) => encodedOpcode i
  | _ => Some CODE_END end.
Definition slotEdgeKind (slot : option FlatInstruction) : option Z :=
  match slot with
  | Some (flatInstruction (opEdge _) _) => None
  | Some (flatInstruction (opInstr RETURN) _) | Some (flatInstruction (opInstr (BR _)) [None]) | None => Some 1
  | Some (flatInstruction (opInstr UNREACHABLE) _) => Some 3
  | Some (flatInstruction opInvoke _) => Some 2
  | Some _ => Some 0
  end.
Definition instructionOpcode (program : Program) (pc : Z) : option Z :=
  match instructionAtPC program pc with Some (_, slot) => slotOpcode slot | None => None end.
Definition instructionEdgeKind (program : Program) (pc : Z) : option Z :=
  match instructionAtPC program pc with Some (_, slot) => slotEdgeKind slot | None => None end.

(* The ROM word at an encoded address. The pc ROM holds the edge of choice 0 at
   an instruction's own address and each later edge at its edge slot, so the key
   of an edge is the instruction's address plus its choice. A call's return
   address is not here: the circuit pins it to the next address. *)
Definition instructionRom (program : Program) (memory : nat) (pc : Z) : option Z :=
  match instructionAtPC program pc with
  | None => None
  | Some (f, slot) =>
      if Nat.eqb memory pcRomId then
        match slot with
        | Some (flatInstruction (opInstr (CALL index)) _) =>
            option_map (fun address => functionEntry program (N.to_nat address)) (callAddress f index)
        | Some (flatInstruction (opInstr (BR_IF _)) _) => Some (pc + 2)
        | Some (flatInstruction (opInstr (IFELSE _ _ _ | BR _ | BR_TABLE _ _)) targets)
        | Some (flatInstruction (opElse | opEdge _) targets) =>
            match targets with
            | Some target :: _ => Some (pcEntry program pc + Z.of_nat target)
            | _ => None
            end
        | Some (flatInstruction (opInstr (RETURN | UNREACHABLE)) _) => None
        (* The callee's entry comes from the function-entry ROM, not from here. *)
        | Some (flatInstruction opInvoke _) => None
        | Some (flatInstruction (opInstr (BLOCK _ _ | LOOP _ _ | CALL_INDIRECT _ _)) _)
        | Some (flatInstruction opEnd _) => Some (pc + 1)
        | _ => None end
      else if Nat.eqb memory programHeaderId then
        match slotOpcode slot, slotEdgeKind slot with
        | Some opcode, Some kind => Some (opcode + 65536 * kind)
        | _, _ => None end
      else if Nat.eqb memory programLocalIndexId then
        match slot with
        | Some (flatInstruction (opInstr (LOCAL_GET x)) _)
        | Some (flatInstruction (opInstr (LOCAL_SET x)) _) => Some (Z.of_N (x :> N))
        | _ => Some 0 end
      else if Nat.eqb memory programBrTableLenId then
        match slot with Some (flatInstruction (opInstr (BR_TABLE ls _)) _) => Some (Z.of_nat (length ls)) | _ => None end
      (* A resolve slot's table, by store address, and its expected type's id. *)
      else if Nat.eqb memory programTableId then
        match slot with Some (flatInstruction (opInstr (CALL_INDIRECT x _)) _) =>
          option_map Z.of_N (nth_error (TABLES (funcinst_MODULE f)) (N.to_nat (x :> N))) | _ => None end
      else if Nat.eqb memory expectedTypeId then
        match slot with Some (flatInstruction (opInstr (CALL_INDIRECT _ y)) _) =>
          option_map (typeId (funcinst_MODULE f)) (nth_error (TYPES (funcinst_MODULE f)) (N.to_nat (y :> N)))
        | _ => None end
      else if Nat.eqb memory callTargetId then
        match slot with Some (flatInstruction (opInstr (CALL x)) _) =>
          option_map (fun address => Z.of_N address + 1) (callAddress f x) | _ => None end
      else None
  end.

(* Function type ids as the Rust parser assigns them: the module's distinct types,
   numbered from 1 in first-seen order. *)
Definition functionRom (program : Program) (memory : nat) (reference : Z) : option Z :=
  if reference <=? 0 then None else
  let function := Z.to_nat (reference - 1) in
  match functionAt program function with
  | None => None
  | Some f =>
      if Nat.eqb memory functionEntryId then Some (guestEntry f)
      else if Nat.eqb memory functionTypeId then Some (typeId (guestModule f) (guestType f))
      else if Nat.eqb memory functionLocalCountId then Some (Z.of_nat (guestParams f + guestLocals f))
      else if Nat.eqb memory functionCallMetadataId then
        (* Parameter count, result count, and the guest-function flag. *)
        Some (Z.of_nat (guestParams f) + 256 * Z.of_nat (guestResults f) + 65536)
      else None
  end.

(* A branch to label l, from `height` operands in the innermost block: it keeps the
   label's arity of top operands and drops the operands below them, down to the
   label's base. `heights` holds the operand height below each enclosing label and
   `arities` their arities, innermost first. *)
Definition labelSlide (heights arities : list nat) (height l : nat) : option (nat * nat) :=
  match nth_error arities l with
  | Some kept => if Nat.leb kept height then Some (kept, height - kept + fold_right Nat.add 0 (firstn l heights))%nat
      else None
  | None => None
  end.

(* A br_if at `height` operands: its condition, then the label's operands below it. *)
Definition branchSlide (heights arities : list nat) (height l : nat) : option (nat * nat) :=
  match height with S below => labelSlide heights arities below l | O => None end.

(* Each flattened slot's label context, laid out like flattenInstruction: the
   operand height below each enclosing label and the labels' arities, innermost
   first, and the slot's own operand height. Slots nothing falls through to have
   none. *)
Definition SlotContext : Type := (list nat * list nat * nat)%type.

Fixpoint instructionContexts (callees : CalleeEnvironment) (heights arities : list nat) (height : option nat)
    (i : instr) : list (option SlotContext) :=
  let bodyContexts := fix bodyContexts (heights arities : list nat) (code : list instr) (h : option nat)
      : list (option SlotContext) :=
    match code with
    | [] => []
    | j :: rest => instructionContexts callees heights arities h j ++
        bodyContexts heights arities rest (match h with Some h => sourceHeight callees j h | None => None end)
    end in
  let innerHeights := (labelBase i (match height with Some h => h | None => 0 end) :: heights)%nat in
  let entered := option_map (fun _ => 0%nat) height in
  match i with
  | BLOCK _ body | LOOP _ body =>
      None :: bodyContexts innerHeights (labelArity i :: arities) body entered ++ [None]
  | IFELSE _ t e =>
      None :: None :: bodyContexts innerHeights (labelArity i :: arities) t entered ++
        match e with [] => [] | _ => None :: bodyContexts innerHeights (labelArity i :: arities) e entered end ++
        [None]
  | CALL_INDIRECT _ _ => [option_map (fun h => (heights, arities, h)) height; None]
  (* An edge slot has the context of its branch. *)
  | _ => option_map (fun h => (heights, arities, h)) height ::
      map (fun _ => option_map (fun h => (heights, arities, h)) height) (edgeLabels i)
  end.

(* A function body runs inside its own label, which holds the results. *)
Definition codeContexts (callees : CalleeEnvironment) (results : nat)
    : list instr -> option nat -> list (option SlotContext) :=
  fix codeContexts (code : list instr) (h : option nat) : list (option SlotContext) :=
    match code with
    | [] => []
    | j :: rest => instructionContexts callees [0%nat] [results] h j ++
        codeContexts rest (match h with Some h => sourceHeight callees j h | None => None end)
    end.

(* A slide's ROM words: the operands it keeps and how far down they move. A slide
   that moves nothing keeps nothing. *)
Definition slideWord (remaining : bool) (kept distance : nat) : Z :=
  if remaining then (if Nat.eqb distance 0 then 0 else Z.of_nat kept) else Z.of_nat distance.

(* The slide ROM's entry at a slot of f's flattened code, for the edges that can
   slide: f's END and returns drop its locals too; a br, a br_if's taken edge and
   each br_table edge slide within the frame. A return is the branch to the
   function's own label, the outermost one, and a br to that label is a return.
   A br_if's own slot is its fall-through, which slides nothing; a br_table's is
   its default. Each later edge is at its edge slot. Other slots have none. *)
Definition functionSlide (program : Program) (memory : nat) (f : GuestFunction) (offset : nat)
    : option Z :=
  let frame := (guestParams f + guestLocals f)%nat in
  let remaining := Nat.eqb memory frameSlideRemainingId in
  let word (base : nat) (slide : option (nat * nat)) :=
    option_map (fun kept_dropped => slideWord remaining (fst kept_dropped) (base + snd kept_dropped)) slide in
  let contexts := codeContexts (programCallees program (guestModule f)) (guestResults f) (guestBody f) (Some 0%nat) in
  match nth_error (flattenCode (guestBody f)) offset, nth_error contexts offset with
  | None, _ => Some (slideWord remaining (guestResults f) frame)
  | Some (flatInstruction (opInstr i) _), Some (Some (heights, arities, h)) =>
      match i with
      | RETURN => word frame (labelSlide heights arities h (length arities - 1))
      | BR l =>
          let frameDrop := if Nat.eqb (N.to_nat (l :> N)) (length arities - 1) then frame else 0%nat in
          word frameDrop (labelSlide heights arities h (N.to_nat (l :> N)))
      | BR_IF _ => Some 0
      | BR_TABLE _ l => word 0%nat (branchSlide heights arities h (N.to_nat (l :> N)))
      | _ => None
      end
  | Some (flatInstruction (opEdge (Some l)) _), Some (Some (heights, arities, h)) =>
      word 0%nat (branchSlide heights arities h (N.to_nat (l :> N)))
  | _, _ => None
  end.

Definition slideRom (program : Program) (memory : nat) (pc : Z) : option Z :=
  if pc <? 0 then None else
  match locatePC program (Z.to_nat pc) with
  | None => None
  | Some (function, offset) =>
      match functionAt program function with
      | None => None
      | Some f => functionSlide program memory f offset
      end
  end.

Definition programRomIds := [programHeaderId; programLocalIndexId; programTableId; expectedTypeId; pcRomId;
  callTargetId; functionEntryId; functionTypeId; functionLocalCountId; functionCallMetadataId;
  frameSlideRemainingId; frameSlideDistanceId; programBrTableLenId].

Definition encodeProgram (program : Program) : MemoryState := fun memory address =>
  match address with
  | [key] =>
      if Nat.eqb memory frameSlideRemainingId || Nat.eqb memory frameSlideDistanceId then
        slideRom program memory key
      else if Nat.eqb memory functionEntryId || Nat.eqb memory functionTypeId || Nat.eqb memory functionLocalCountId ||
        Nat.eqb memory functionCallMetadataId
        then functionRom program memory key
      else instructionRom program memory key
  | _ => None
  end.

(* The ROMs are dense: every key holds a word, and keys the encoder does not list
   hold 0, so a read alone never shows that a key belongs to the program. *)
Definition romValue (entry : option Z) : Z := match entry with Some v => v | None => 0 end.

(* Only the program's ROMs are fixed; RAM and unrelated tables remain independent. *)
Definition programLoaded (program : Program) (memory : MemoryState) : Prop :=
  forall id, In id programRomIds -> forall address, memory id address = Some (romValue (encodeProgram program id address)).

Definition loadProgram (program : Program) (memory : MemoryState) : MemoryState :=
  fun id address =>
    if existsb (Nat.eqb id) programRomIds then Some (romValue (encodeProgram program id address)) else memory id address.

(* Every function body is in the supported fragment, calling within its module. *)
Definition programSupported (program : Program) : Prop :=
  forall index f, functionAt program index = Some f ->
    let callees := programCallees program (guestModule f) in
    supportedCode callees (guestBody f) /\ codeBranches callees 0 (guestBody f) 0.

(* SpecTec validates every function in the store, and the circuit supports each. *)
Definition programValidated (program : Program) (s : store) : Prop :=
  Forall (fun fi => Funcinst_ok s fi (funcinst_TYPE fi)) program /\ programSupported program.

(* Row kinds a run may contain: program rows, a frame's zero and move rows, and
   padding. The run's activation never finishes: it neither returns from its export
   nor traps. At the given depth a row may not pop, so the run stays in that frame. *)
Definition callerRow (depth : Z) (row : Witness) : Prop :=
  (row COL_IS_PROGRAM_ROW = 1 \/ row COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/
    row COL_FRAME_MOVE_ACTIVE_BEFORE = 1 \/ row COL_PADDING_ACTIVE = 1) /\
  row COL_ACTIVATION_FINISHED_AFTER = 0 /\
  (row COL_CALL_STACK_DEPTH_BEFORE = depth -> row COL_CALL_STACK_POP_PRESENT = 0).

Fixpoint rowsFinalBoundary (boundary : Boundary) (rows : list Witness) : Boundary :=
  match rows with [] => boundary | row :: rest => rowsFinalBoundary (afterBoundary row) rest end.
