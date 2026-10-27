(* Source functions, their ROM encoding, and what a program and a run must satisfy.
   This encoder covers i32 guest functions without imports. Blocks occupy
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

Definition instructionAtPC (program : Program) (pc : Z) : option (funcinst * option FlatInstruction) :=
  if pc <? 0 then None else
  match locatePC program (Z.to_nat pc) with
  | Some (function, offset) => match nth_error program function with
      | Some f => Some (f, nth_error (flattenCode (functionBody f)) offset)
      | None => None end
  | None => None
  end.

Definition callAddress (f : funcinst) (index : idx) : option N :=
  nth_error (FUNCS (funcinst_MODULE f)) (N.to_nat (index :> N)).

Definition instructionRom (program : Program) (memory : nat) (pc : Z) (choice : Z) : option Z :=
  match instructionAtPC program pc with
  | None => None
  | Some (f, slot) =>
      if Nat.eqb memory pcRomId then
        match slot with
        | Some (flatInstruction (Some (CALL index)) _) =>
            if choice =? 1 then Some (pc + 1) else if choice =? 0 then
              option_map (fun address => functionEntry program (N.to_nat address)) (callAddress f index)
            else None
        | Some (flatInstruction (Some (BR_IF _)) (Some distance)) =>
            if choice =? 0 then Some (pc + 1) else if choice =? 1 then Some (pc + Z.of_nat distance) else None
        | Some _ => if choice =? 0 then Some (pc + 1) else None
        | None => None end
      else if Nat.eqb memory programOpcodeId then
        match slot with
        | Some (flatInstruction (Some (CALL _)) _) => Some CODE_CALL
        | Some (flatInstruction (Some (BLOCK _ _)) _) => Some CODE_BLOCK
        | Some (flatInstruction (Some (BR_IF _)) _) => Some CODE_BR_IF
        | Some (flatInstruction (Some i) _) => encodedOpcode i
        | _ => Some CODE_END end
      else if Nat.eqb memory pcEdgeKindId then
        match slot with Some _ => Some 0 | None => Some 1 end
      else if Nat.eqb memory programLocalIndexId then
        match slot with
        | Some (flatInstruction (Some (LOCAL_GET x)) _)
        | Some (flatInstruction (Some (LOCAL_SET x)) _) => Some (Z.of_N (x :> N))
        | _ => Some 0 end
      else if Nat.eqb memory callTargetId then
        match slot with Some (flatInstruction (Some (CALL x)) _) =>
          option_map (fun address => Z.of_N address + 1) (callAddress f x) | _ => None end
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

(* Each flattened slot's branch drop, laid out like flattenInstruction. A taken
   br_if l at operand height h discards the h - 1 operands below its condition and
   those below the l labels inside its target; `labels` holds, innermost first, the
   operand height below each enclosing block. Blocks carry no results, so a taken
   branch keeps nothing. *)
Fixpoint instructionDrops (callees : CalleeEnvironment) (labels : list nat) (height : option nat) (i : instr)
    : list (option nat) :=
  match i with
  | BLOCK _ body =>
      let inner := (match height with Some h => h | None => 0 end :: labels)%nat in
      None :: (fix bodyDrops (code : list instr) (h : option nat) : list (option nat) :=
        match code with
        | [] => []
        | j :: rest => instructionDrops callees inner h j ++
            bodyDrops rest (match h with Some h => sourceHeight callees j h | None => None end)
        end) body (option_map (fun _ => 0%nat) height) ++ [None]
  | BR_IF l =>
      [match height with
       | Some (S below) =>
           let target := N.to_nat (l :> N) in
           if (target <? length labels)%nat then Some (below + fold_right Nat.add 0 (firstn target labels))%nat
           else None
       | _ => None
       end]
  | _ => [None]
  end.

Definition codeDrops (callees : CalleeEnvironment) : list instr -> option nat -> list (option nat) :=
  fix codeDrops (code : list instr) (h : option nat) : list (option nat) :=
    match code with
    | [] => []
    | j :: rest => instructionDrops callees [] h j ++
        codeDrops rest (match h with Some h => sourceHeight callees j h | None => None end)
    end.

(* The frame slide each taken edge seeds: the operands it keeps and how far down
   they move. A function's END keeps its results and drops its locals; a slide
   that moves nothing keeps nothing. *)
Definition slideRom (program : Program) (memory : nat) (pc choice : Z) : option Z :=
  if pc <? 0 then None else
  match locatePC program (Z.to_nat pc) with
  | None => None
  | Some (function, offset) =>
      match functionAt program function with
      | None => None
      | Some f =>
          let frame := Z.of_nat (guestParams f + guestLocals f) in
          let remaining := Nat.eqb memory frameSlideRemainingId in
          match nth_error (flattenCode (guestBody f)) offset with
          | None =>
              if choice =? 0 then Some (if remaining then (if frame =? 0 then 0 else Z.of_nat (guestResults f)) else frame)
              else None
          | Some (flatInstruction (Some (BR_IF _)) _) =>
              if remaining then (if (choice =? 0) || (choice =? 1) then Some 0 else None)
              else if choice =? 0 then Some 0
              else if choice =? 1 then
                match nth_error (codeDrops (programCallees program (guestModule f)) (guestBody f) (Some 0%nat)) offset with
                | Some (Some dropped) => Some (Z.of_nat dropped)
                | _ => None
                end
              else None
          | Some _ => None
          end
      end
  end.

Definition programRomIds := [programOpcodeId; programLocalIndexId; pcEdgeKindId; pcRomId;
  callTargetId; pcFunctionRefId; functionEntryId; functionLocalCountId; functionCallMetadataId;
  frameSlideRemainingId; frameSlideDistanceId].

Definition encodeProgram (program : Program) : MemoryState := fun memory address =>
  match address with
  | [pc; choice] =>
      if Nat.eqb memory pcRomId then instructionRom program memory pc choice
      else if Nat.eqb memory frameSlideRemainingId || Nat.eqb memory frameSlideDistanceId then
        slideRom program memory pc choice
      else None
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
