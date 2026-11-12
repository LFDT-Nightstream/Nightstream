(* SPEC.md section 7.6: decoded views of the packed instruction ROM. *)
From Stdlib Require Import List ZArith Lia.
Require Import MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

Definition opcodeAt (memory : MemoryState) (address : list Z) : option Z :=
  option_map (fun header => header mod 65536) (memory programHeaderId address).
Definition edgeKindAt (memory : MemoryState) (address : list Z) : option Z :=
  option_map (fun header => header / 65536) (memory programHeaderId address).

Lemma packed_low16 : forall low high,
  0 <= low < 65536 -> (low + 65536 * high) mod 65536 = low.
Proof. intros. rewrite Z.mul_comm, Z.mod_add by lia. now apply Z.mod_small. Qed.
Lemma packed_high16 : forall low high,
  0 <= low < 65536 -> (low + 65536 * high) / 65536 = high.
Proof. intros. rewrite Z.mul_comm, Z.div_add by lia. rewrite Z.div_small by lia. lia. Qed.
