(* SPEC.md section 7.6: decoded views of the packed instruction ROM. *)
From Stdlib Require Import List ZArith Lia.
Require Import MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

Definition opcodeAt (memory : MemoryState) (address : list Z) : option Z :=
  option_map (fun header => header mod 65536) (memory programHeaderId address).
Definition edgeKindAt (memory : MemoryState) (address : list Z) : option Z :=
  option_map (fun header => header / 65536) (memory programHeaderId address).

Lemma header_opcode : forall opcode kind,
  0 <= opcode < 65536 -> (opcode + 65536 * kind) mod 65536 = opcode.
Proof. intros. rewrite Z.mul_comm, Z.mod_add by lia. now apply Z.mod_small. Qed.
Lemma header_edge_kind : forall opcode kind,
  0 <= opcode < 65536 -> (opcode + 65536 * kind) / 65536 = kind.
Proof. intros. rewrite Z.mul_comm, Z.div_add by lia. rewrite Z.div_small by lia. lia. Qed.
