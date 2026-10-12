(* Single-frame i32 arithmetic and locals specification: program loading and the projection
   of circuit memory and boundary registers into a SpecTec configuration.
   Initial locals must already match memory; parameter initialization and local
   zeroing are auxiliary rows outside this fragment. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS Columns MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals.
Import ListNotations.
Open Scope Z_scope.

Definition encodedOpcode (instruction : instr) : option Z :=
  match instruction with
  | BINOP I32 (mk_binop__0 Inn_I32 ADD) => Some CODE_I32_ADD
  | BINOP I32 (mk_binop__0 Inn_I32 SUB) => Some CODE_I32_SUB
  | LOCAL_GET _ => Some CODE_LOCAL_GET
  | LOCAL_SET _ => Some CODE_LOCAL_SET
  | _ => None
  end.

Definition supportedProgram (program : list instr) : Prop :=
  Forall (fun instruction => exists code, encodedOpcode instruction = Some code) program.

Definition programOpcodeAt (program : list instr) (pc : nat) : option Z :=
  match nth_error program pc with
  | Some instruction => encodedOpcode instruction
  | None => None
  end.

Definition programLocalIndexAt (program : list instr) (pc : nat) : option Z :=
  match nth_error program pc with
  | Some (LOCAL_GET x) | Some (LOCAL_SET x) => Some (Z.of_N (x :> N))
  | _ => None
  end.

(* This fragment uses consecutive instruction addresses starting at zero.
   Equality also fixes absent entries: no extra instructions may follow it. *)
Definition loadedProgram (program : list instr) (memory : MemoryState) : Prop :=
  (forall pc, memory MemoryCatalog.programOpcodeId [Z.of_nat pc] = programOpcodeAt program pc) /\
  (forall pc, memory MemoryCatalog.pcEdgeKindId [Z.of_nat pc] =
    if (pc <? length program)%nat then Some 0 else None) /\
  (forall pc choice, memory MemoryCatalog.pcRomId [Z.of_nat pc; choice] =
    if (pc <? length program)%nat && (choice =? 0) then Some (Z.of_nat (pc + 1)) else None) /\
  (forall pc, memory MemoryCatalog.programLocalIndexId [Z.of_nat pc] =
    programLocalIndexAt program pc).

Record Boundary := mkBoundary {
  boundaryPC : Z;
  boundarySP : Z;
  boundaryStackBase : Z;
  boundaryLocalsBase : Z
}.

Definition beforeBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_BEFORE) (w COL_SP_BEFORE) (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE).

Definition afterBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_AFTER) (w COL_SP_AFTER) (w COL_STACK_FRAME_BASE_AFTER) (w COL_LOCALS_FBP_AFTER).

Fixpoint finalRow (first : Witness) (rest : list Witness) : Witness :=
  match rest with [] => first | next :: rest => finalRow next rest end.

(* Values are ordered bottom-to-top. i32 uses the low limb at 2 * stack index. *)
Definition stackMatches (memory : MemoryState) (boundary : Boundary)
    (values : list Z) : Prop :=
  0 <= boundaryStackBase boundary /\
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length values) /\
  Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value ->
    memory MemoryCatalog.stackId
      [2 * (boundaryStackBase boundary + Z.of_nat index)] = Some value.

Definition stateLocals (z : state) : list val :=
  match z with mk_state _ f => LOCALS f end.

(* Every local in the active frame is an i32 represented by its low memory limb. *)
Definition localsMatch (memory : MemoryState) (boundary : Boundary) (z : state) : Prop :=
  forall index v, nth_error (stateLocals z) index = Some v ->
    exists value, v = i32Value value /\ 0 <= value < i32Modulus /\
      memory MemoryCatalog.localLoId [boundaryLocalsBase boundary; Z.of_nat index] = Some value.

(* Bounds on local indices and stack underflow; the final height is unrestricted. *)
Fixpoint programTyped (localCount height : nat) (program : list instr) : Prop :=
  match program with
  | [] => True
  | LOCAL_GET x :: rest =>
      (N.to_nat (x :> N) < localCount)%nat /\ programTyped localCount (S height) rest
  | LOCAL_SET x :: rest =>
      (N.to_nat (x :> N) < localCount)%nat /\
      match height with S remaining => programTyped localCount remaining rest | _ => False end
  | _ :: rest =>
      match height with
      | S (S remaining) => programTyped localCount (S remaining) rest
      | _ => False
      end
  end.

Definition i32Constant (value : Z) : admininstr :=
  admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N value))).

Definition stackConfig (z : state) (values : list Z) (code : list instr) : config :=
  mk_config z (map i32Constant values ++ map admininstr_instr code).

Definition represents (program : list instr) (z : state) (memory : MemoryState)
    (boundary : Boundary) (pc : nat) (configuration : config) : Prop :=
  boundaryPC boundary = Z.of_nat pc /\
  (pc <= length program)%nat /\
  exists values,
    stackMatches memory boundary values /\
    programTyped (length (stateLocals z)) (length values) (skipn pc program) /\
    configuration = stackConfig z values (skipn pc program) /\
    wf_config configuration /\
    localsMatch memory boundary z.
