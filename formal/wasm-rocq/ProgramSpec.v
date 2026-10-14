(* Source functions, their ROM encoding, and the active source position.
   This encoder covers i32 guest functions without imports or structured control.
   PCs are consecutive across bodies, with one trailing END per function. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS Columns Wasm2 MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec CallerSpec FunctionValidation SpecTecCalls.
Import ListNotations.
Open Scope Z_scope.

Definition Program := list funcinst.

Definition functionBody (f : funcinst) : list instr :=
  match CODE f with func_FUNC _ _ body => body end.

Definition functionSize (f : funcinst) : nat := S (length (functionBody f)).

Definition functionEntry (program : Program) (function : nat) : Z :=
  Z.of_nat (fold_right (fun f size => functionSize f + size)%nat 0%nat (firstn function program)).

Definition i32Type (t : valtype) : bool := match t with valtype_I32 => true | _ => false end.
Definition i32Local (l : local) : bool := match l with LOCAL t => i32Type t end.

Definition decodeFunction (address : nat) (entry : Z) (f : funcinst) : option GuestFunction :=
  match f with
  | MKfuncinst (mk_functype (mk_list params) (mk_list results)) module (func_FUNC typeIndex locals body) =>
      if forallb i32Type params && forallb i32Type results && forallb i32Local locals then
        Some (mkGuestFunction (N.of_nat address) module typeIndex (length params)
          (length results) (length locals) body entry)
      else None
  end.

Definition functionAt (program : Program) (function : nat) : option GuestFunction :=
  match nth_error program function with
  | Some f => decodeFunction function (functionEntry program function) f
  | None => None
  end.

Definition programCallees (program : Program) (module : moduleinst) : CalleeEnvironment :=
  fun index => match nth_error (FUNCS module) (N.to_nat index) with
  | Some address => functionAt program (N.to_nat address)
  | None => None
  end.

(* Locate an encoded instruction, including the function-ending END. *)
Fixpoint locatePC (program : Program) (pc : nat) : option (nat * nat) :=
  match program with
  | [] => None
  | f :: rest => if (pc <? functionSize f)%nat then Some (0%nat, pc)
      else option_map (fun location => (S (fst location), snd location))
        (locatePC rest (pc - functionSize f))
  end.

Definition instructionAtPC (program : Program) (pc : Z) : option (funcinst * option instr) :=
  if pc <? 0 then None else
  match locatePC program (Z.to_nat pc) with
  | Some (function, offset) => match nth_error program function with
      | Some f => Some (f, nth_error (functionBody f) offset)
      | None => None end
  | None => None
  end.

Definition callAddress (f : funcinst) (index : idx) : option N :=
  nth_error (FUNCS (funcinst_MODULE f)) (N.to_nat (index :> N)).

Definition instructionRom (program : Program) (memory : nat) (pc : Z) (choice : Z) : option Z :=
  match instructionAtPC program pc with
  | None => None
  | Some (f, instruction) =>
      if Nat.eqb memory pcRomId then
        match instruction with
        | Some (CALL index) =>
            if choice =? 1 then Some (pc + 1) else if choice =? 0 then
              option_map (fun address => functionEntry program (N.to_nat address)) (callAddress f index)
            else None
        | Some _ => if choice =? 0 then Some (pc + 1) else None
        | None => None end
      else if Nat.eqb memory programOpcodeId then
        match instruction with Some (CALL _) => Some CODE_CALL | Some i => encodedOpcode i | None => Some CODE_END end
      else if Nat.eqb memory pcEdgeKindId then
        match instruction with Some _ => Some 0 | None => Some 1 end
      else if Nat.eqb memory programLocalIndexId then
        match instruction with Some (LOCAL_GET x) | Some (LOCAL_SET x) => Some (Z.of_N (x :> N)) | _ => Some 0 end
      else if Nat.eqb memory callTargetId then
        match instruction with Some (CALL x) => option_map (fun address => Z.of_N address + 1) (callAddress f x) | _ => None end
      else None
  end.

Definition functionRom (program : Program) (memory : nat) (reference : Z) : option Z :=
  if reference <=? 0 then None else
  let function := Z.to_nat (reference - 1) in
  match functionAt program function with
  | None => None
  | Some f =>
      if Nat.eqb memory functionEntryId then Some (guestEntry f)
      else if Nat.eqb memory functionLocalCountId then Some (Z.of_nat (guestParams f + guestLocals f))
      else if Nat.eqb memory functionCallMetadataId then
        (* Parameter count, result count, and the guest-function flag. *)
        Some (Z.of_nat (guestParams f) + 256 * Z.of_nat (guestResults f) + 65536)
      else None
  end.

Definition programRomIds := [programOpcodeId; programLocalIndexId; pcEdgeKindId; pcRomId;
  callTargetId; pcFunctionRefId; functionEntryId; functionLocalCountId; functionCallMetadataId].

Definition encodeProgram (program : Program) : MemoryState := fun memory address =>
  match address with
  | [pc; choice] => if Nat.eqb memory pcRomId then instructionRom program memory pc choice else None
  | [key] =>
      if Nat.eqb memory pcRomId then None else
      if Nat.eqb memory pcFunctionRefId then
        if key <? 0 then None else
        option_map (fun location => Z.of_nat (fst location + 1)) (locatePC program (Z.to_nat key))
      else if Nat.eqb memory functionEntryId || Nat.eqb memory functionLocalCountId || Nat.eqb memory functionCallMetadataId
        then functionRom program memory key
      else instructionRom program memory key 0
  | _ => None
  end.

(* Only the program's ROMs are fixed; RAM and unrelated tables remain independent. *)
Definition programLoaded (program : Program) (memory : MemoryState) : Prop :=
  forall id, In id programRomIds -> forall address, memory id address = encodeProgram program id address.

Definition loadProgram (program : Program) (memory : MemoryState) : MemoryState :=
  fun id address => if existsb (Nat.eqb id) programRomIds then encodeProgram program id address else memory id address.

Record SourcePosition := sourcePosition { sourceFunction : nat; sourceOffset : nat }.
Definition advanceSource (position : SourcePosition) (count : nat) : SourcePosition :=
  sourcePosition (sourceFunction position) (sourceOffset position + count).

Definition codeAt (f : GuestFunction) (offset : nat) (code : list instr) : Prop :=
  (offset <= length (guestBody f))%nat /\ skipn offset (guestBody f) = code.

(* Bind the function store, module function addresses, and active frame's local count. *)
Definition frameMatches (program : Program) (f : GuestFunction) (z : state) : Prop :=
  fun_funcinst z = program /\ fun_funcaddr z = FUNCS (guestModule f) /\
  length (stateLocals z) = (guestParams f + guestLocals f)%nat.

(* Validation uses SpecTec's instantiated functions and the active function's
   module context. Support only restricts the circuit fragment we can simulate. *)
Definition validatedCode (program : Program) (f : GuestFunction) (z : state)
    (height : nat) (code : list instr) : Prop :=
  Forall (fun fi => Funcinst_ok (fun_store z) fi (funcinst_TYPE fi)) program /\
  supportedCaller (programCallees program (guestModule f)) code /\
  exists C outputs,
    Moduleinst_ok (fun_store z) (guestModule f) C /\
    Instrs_ok (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) code
      (i32StackType height outputs).

Definition SourceState (program : Program) (position : SourcePosition) (z : state)
    (memory : MemoryState) (boundary : Boundary) (code : list instr) (c : config) : Prop :=
  match functionAt program (sourceFunction position) with
  | None => False
  | Some f =>
      frameMatches program f z /\ codeAt f (sourceOffset position) code /\
      validatedCode program f z (stackHeight boundary) code /\
      represents z memory boundary code c /\
      boundaryPC boundary = guestEntry f + Z.of_nat (sourceOffset position)
  end.

(* Trace endpoints are caller instruction boundaries. Initialization is allowed
   inside calls, but not at the caller's own depth. At that depth a row may
   neither pop nor halt (the depth-zero return halts without popping). *)
Definition callerRow (depth : Z) (row : Witness) : Prop :=
  (row COL_IS_PROGRAM_ROW = 1 \/ row COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 \/
    row COL_PARAM_INIT_ACTIVE_BEFORE = 1) /\
  (row COL_CALL_STACK_DEPTH_BEFORE = depth ->
    row COL_IS_PROGRAM_ROW = 1 /\ row COL_CALL_STACK_POP_PRESENT = 0 /\ row COL_HALTED = 0).

Definition rowsStartAt (boundary : Boundary) (depth : Z) (rows : list Witness) : Prop :=
  match rows with
  | [] => True
  | first :: _ => beforeBoundary first = boundary /\ first COL_CALL_STACK_DEPTH_BEFORE = depth
  end.

Fixpoint rowsFinalBoundary (boundary : Boundary) (rows : list Witness) : Boundary :=
  match rows with [] => boundary | row :: rest => rowsFinalBoundary (afterBoundary row) rest end.

Fixpoint rowsFinalDepth (depth : Z) (rows : list Witness) : Z :=
  match rows with [] => depth | row :: rest => rowsFinalDepth (row COL_CALL_STACK_DEPTH_AFTER) rest end.

Definition callerFragment (boundary : Boundary) (depth : Z) (rows : list Witness) : Prop :=
  rowsStartAt boundary depth rows /\ Forall (callerRow depth) rows /\ rowsFinalDepth depth rows = depth.
