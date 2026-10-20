(* Direct guest calls with i32 parameters, results and locals: the ROM images of
   a call site, an instruction and a callee's entry. Early return and
   call_indirect are outside this fragment. *)
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


Definition callerFrameImage (memory : MemoryState) (pc : Z) (caller : state) : Prop :=
  exists reference,
    memory pcFunctionRefId [pc] = Some reference /\
    memory functionLocalCountId [reference] = Some (Z.of_nat (length (stateLocals caller))).
