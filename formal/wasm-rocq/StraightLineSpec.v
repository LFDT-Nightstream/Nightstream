(* Shared i32 stack/local representation and call-free program predicates. *)
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

Definition supportedStraightLineProgram (program : list instr) : Prop :=
  Forall (fun instruction => exists code, encodedOpcode instruction = Some code) program.

(* The continuity-linked registers a state between two rows depends on: the
   active frame, the call depth, and the frame-initialization countdowns. *)
Record Boundary := mkBoundary {
  boundaryPC : Z;
  boundarySP : Z;
  boundaryStackBase : Z;
  boundaryLocalsBase : Z;
  boundaryCallDepth : Z;
  boundaryZeroRemaining : Z;
  boundaryParamRemaining : Z;
  boundaryZeroActive : Z;
  boundaryParamActive : Z
}.

Definition beforeBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_BEFORE) (w COL_SP_BEFORE) (w COL_STACK_FRAME_BASE_BEFORE) (w COL_LOCALS_FBP_BEFORE)
    (w COL_CALL_STACK_DEPTH_BEFORE) (w COL_LOCAL_ZERO_REMAINING_BEFORE) (w COL_PARAM_INIT_REMAINING_BEFORE)
    (w COL_LOCAL_ZERO_ACTIVE_BEFORE) (w COL_PARAM_INIT_ACTIVE_BEFORE).

Definition afterBoundary (w : Witness) : Boundary :=
  mkBoundary (w COL_PC_AFTER) (w COL_SP_AFTER) (w COL_STACK_FRAME_BASE_AFTER) (w COL_LOCALS_FBP_AFTER)
    (w COL_CALL_STACK_DEPTH_AFTER) (w COL_LOCAL_ZERO_REMAINING_AFTER) (w COL_PARAM_INIT_REMAINING_AFTER)
    (w COL_LOCAL_ZERO_ACTIVE_AFTER) (w COL_PARAM_INIT_ACTIVE_AFTER).

(* No frame initialization is pending at this boundary. *)
Definition initializationIdle (boundary : Boundary) : Prop :=
  boundaryZeroRemaining boundary = 0 /\ boundaryParamRemaining boundary = 0 /\
  boundaryZeroActive boundary = 0 /\ boundaryParamActive boundary = 0.

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

(* Unsupported instructions and insufficient operands have no successor height. *)
Definition instructionStackHeight (instruction : instr) (height : nat) : option nat :=
  match instruction with
  | LOCAL_GET _ => Some (S height)
  | LOCAL_SET _ => match height with S remaining => Some remaining | _ => None end
  | BINOP I32 (mk_binop__0 Inn_I32 ADD)
  | BINOP I32 (mk_binop__0 Inn_I32 SUB) =>
      match height with S (S remaining) => Some (S remaining) | _ => None end
  | _ => None
  end.

Definition instructionStackDelta (instruction : instr) : option Z :=
  match instruction with
  | LOCAL_GET _ => Some 1
  | LOCAL_SET _
  | BINOP I32 (mk_binop__0 Inn_I32 ADD)
  | BINOP I32 (mk_binop__0 Inn_I32 SUB) => Some (-1)
  | _ => None
  end.

Fixpoint bodyStackDelta (body : list instr) : option Z :=
  match body with
  | [] => Some 0
  | instruction :: rest =>
      match instructionStackDelta instruction, bodyStackDelta rest with
      | Some delta, Some remaining => Some (delta + remaining)
      | _, _ => None
      end
  end.

(* Validation context for this i32-only, call-free fragment. Other context
   components are unused by its supported instructions. *)
Definition i32TypingContext (localCount : nat) : context :=
  MKcontext [] [] [] [] [] [] [] (repeat valtype_I32 localCount) [] None.

Definition i32StackType (inputs outputs : nat) : functype :=
  mk_functype (mk_list _ (repeat valtype_I32 inputs)) (mk_list _ (repeat valtype_I32 outputs)).

(* Internal bounds on local indices and stack underflow, derived from SpecTec
   validation by StraightLineValidation; the final height is unrestricted. *)
Fixpoint programTyped (localCount height : nat) (program : list instr) : Prop :=
  match program with
  | [] => True
  | instruction :: rest =>
      let continuation :=
        match instructionStackHeight instruction height with
        | Some remaining => programTyped localCount remaining rest
        | None => False
        end in
      match instruction with
      | LOCAL_GET x | LOCAL_SET x => (N.to_nat (x :> N) < localCount)%nat /\ continuation
      | _ => continuation
      end
  end.

Definition i32Constant (value : Z) : admininstr :=
  admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N value))).

Definition stackConfig (z : state) (values : list Z) (code : list instr) : config :=
  mk_config z (map i32Constant values ++ map admininstr_instr code).

Definition stackHeight (boundary : Boundary) : nat :=
  Z.to_nat (boundarySP boundary - boundaryStackBase boundary).

(* Memory and register representation only; typing and program loading are separate. *)
Definition represents (z : state) (memory : MemoryState) (boundary : Boundary)
    (code : list instr) (configuration : config) : Prop :=
  exists values,
    stackMatches memory boundary values /\
    configuration = stackConfig z values code /\
    wf_config configuration /\ localsMatch memory boundary z.
