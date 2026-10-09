(* Straight-line i32.add/sub specification: program loading and the projection
   of circuit memory and boundary registers into a SpecTec configuration. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS Columns MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers.
Import ListNotations.
Open Scope Z_scope.

Definition encodedOpcode (instruction : instr) : option Z :=
  match instruction with
  | BINOP I32 (mk_binop__0 Inn_I32 ADD) => Some CODE_I32_ADD
  | BINOP I32 (mk_binop__0 Inn_I32 SUB) => Some CODE_I32_SUB
  | _ => None
  end.

Definition supportedProgram (program : list instr) : Prop :=
  Forall (fun instruction => exists code, encodedOpcode instruction = Some code) program.

Definition programOpcodeAt (program : list instr) (pc : nat) : option Z :=
  match nth_error program pc with
  | Some instruction => encodedOpcode instruction
  | None => None
  end.

(* This fragment uses consecutive instruction addresses starting at zero.
   Equality also fixes absent entries: no extra instructions may follow it. *)
Definition loadedProgram (program : list instr) (memory : MemoryState) : Prop :=
  (forall pc, memory MemoryCatalog.programOpcodeId [Z.of_nat pc] = programOpcodeAt program pc) /\
  (forall pc, memory MemoryCatalog.pcEdgeKindId [Z.of_nat pc] =
    if (pc <? length program)%nat then Some 0 else None) /\
  (forall pc choice, memory MemoryCatalog.pcRomId [Z.of_nat pc; choice] =
    if (pc <? length program)%nat && (choice =? 0) then Some (Z.of_nat (pc + 1)) else None).

Record Boundary := mkBoundary {
  boundaryPC : Z;
  boundarySP : Z;
  boundaryStackBase : Z
}.

Definition beforeBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_BEFORE) (w COL_SP_BEFORE) (w COL_STACK_FRAME_BASE_BEFORE).

Definition afterBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_AFTER) (w COL_SP_AFTER) (w COL_STACK_FRAME_BASE_AFTER).

(* Values are ordered bottom-to-top. i32 uses the low limb at 2 * stack index. *)
Definition stackMatches (memory : MemoryState) (boundary : Boundary)
    (values : list Z) : Prop :=
  0 <= boundaryStackBase boundary /\
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length values) /\
  Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value ->
    memory MemoryCatalog.stackId
      [2 * (boundaryStackBase boundary + Z.of_nat index)] = Some value.

(* Check for underflow: each supported instruction requires two values
   and decreases the stack height by one. Final height is unrestricted. *)
Fixpoint arithmeticStackTyped (height : nat) (program : list instr) : Prop :=
  match program with
  | [] => True
  | _ :: rest =>
      match height with
      | S (S remaining) => arithmeticStackTyped (S remaining) rest
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
    arithmeticStackTyped (length values) (skipn pc program) /\
    configuration = stackConfig z values (skipn pc program) /\
    wf_config configuration.
