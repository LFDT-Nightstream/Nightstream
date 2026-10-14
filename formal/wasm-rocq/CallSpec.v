(* Direct guest calls with i32 parameters, results and locals. Bodies contain
   add/sub/local.get/local.set and finish with END; nested calls, early return,
   blocks and call_indirect are outside this fragment. *)
From Stdlib Require Import List ZArith NArith.
Require Import MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 SpecTecCalls StraightLineSpec.
Import ListNotations.
Open Scope Z_scope.

Definition guestFunctionImage (memory : MemoryState) (address : N) (entry : Z)
    (params results locals : nat) : Prop :=
  let reference := Z.of_N address + 1 in
  (* Bits 0–7: parameter count; bits 8–15: result count; bit 16: guest flag (65536). *)
  memory functionCallMetadataId [reference] =
    Some (Z.of_nat params + 256 * Z.of_nat results + 65536) /\
  memory functionLocalCountId [reference] = Some (Z.of_nat (params + locals)) /\
  memory functionEntryId [reference] = Some entry /\
  memory pcFunctionRefId [entry] = Some reference.

Definition directCallImage (memory : MemoryState) (pc returnPC : Z) (caller : state)
    (index : idx) (address : N) (module : moduleinst) (typeIndex : typeidx)
    (params results locals : nat) (body : list instr) (entry : Z) : Prop :=
  nth_error (fun_funcaddr caller) (N.to_nat (index :> N)) = Some address /\
  nth_error (fun_funcinst caller) (N.to_nat address) =
    Some (i32Function module typeIndex params results locals body) /\
  memory programOpcodeId [pc] = Some CODE_CALL /\
  memory callTargetId [pc] = Some (Z.of_N address + 1) /\
  memory pcRomId [pc; 1] = Some returnPC /\
  guestFunctionImage memory address entry params results locals.

Definition instructionImage (memory : MemoryState) (pc : Z) (instruction : instr) : Prop :=
  memory programOpcodeId [pc] = encodedOpcode instruction /\
  memory pcEdgeKindId [pc] = Some 0 /\
  (forall choice, memory pcRomId [pc; choice] = if choice =? 0 then Some (pc + 1) else None) /\
  match instruction with
  | LOCAL_GET x | LOCAL_SET x => memory programLocalIndexId [pc] = Some (Z.of_N (x :> N))
  | _ => True
  end.

Fixpoint loadedBody (memory : MemoryState) (entry : Z) (body : list instr) : Prop :=
  match body with
  | [] => True
  | instruction :: rest => instructionImage memory entry instruction /\ loadedBody memory (entry + 1) rest
  end.


Definition callerFrameImage (memory : MemoryState) (pc : Z) (caller : state) : Prop :=
  exists reference,
    memory pcFunctionRefId [pc] = Some reference /\
    memory functionLocalCountId [reference] = Some (Z.of_nat (length (stateLocals caller))).

Definition loadedCall (memory : MemoryState) (pc returnPC : Z) (caller : state)
    (index : idx) (address : N) (module : moduleinst) (typeIndex : typeidx)
    (params results locals : nat) (body : list instr) (entry : Z) : Prop :=
  directCallImage memory pc returnPC caller index address module typeIndex params results locals body entry /\
  callerFrameImage memory pc caller /\ loadedBody memory entry body /\
  memory programOpcodeId [entry + Z.of_nat (length body)] = Some CODE_END /\
  memory pcEdgeKindId [entry + Z.of_nat (length body)] = Some 1.

(* A direct i32 guest call with a straight-line body and a function-ending END.
   Validation/ROM construction and the permitted middle-row kinds are assumptions.
   Initialization order and length are derived within those row kinds. *)
Definition typedCall (caller : state) (index : idx) (module : moduleinst) (typeIndex : typeidx)
    (params results locals : nat) (body : list instr) : Prop :=
  wf_state caller /\ wf_uN 32%N index /\
  wf_funcinst (i32Function module typeIndex params results locals body) /\
  supportedStraightLineProgram body /\
  Instrs_ok (i32TypingContext (params + locals)) body (i32StackType 0 results) /\
  (params < 256)%nat /\ (results < 256)%nat.
