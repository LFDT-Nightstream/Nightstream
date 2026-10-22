(* Arithmetic instruction simulation and decoding of the supported opcode set. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Wasm2 WasmIntegers
  StackBindings StraightLineSpec StackSimulation LocalSimulation LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma encoded_opcode_cases : forall instruction code,
  encodedOpcode instruction = Some code ->
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 ADD) /\ code = CODE_I32_ADD) \/
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 SUB) /\ code = CODE_I32_SUB) \/
  (exists x, instruction = LOCAL_GET x /\ code = CODE_LOCAL_GET) \/
  (exists x, instruction = LOCAL_SET x /\ code = CODE_LOCAL_SET).
Proof.
  intros instruction code Hcode.
  destruct instruction; try discriminate;
    repeat match goal with
    | x : numtype |- _ => destruct x; try discriminate
    | x : binop_ |- _ => destruct x; try discriminate
    | x : Inn |- _ => destruct x; try discriminate
    | x : binop_Inn |- _ => destruct x; try discriminate
    end; inversion Hcode; subst; eauto 8.
Qed.
