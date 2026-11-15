From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Definition i32Modulus : Z := 4294967296.

Definition i64Modulus : Z := 18446744073709551616.

(* Reconstruct the Wasm integer in Z, not in the circuit field. *)
Definition joinLimbs (lo hi : Z) : Z := lo + i32Modulus * hi.

Lemma join_limbs_bound : forall lo hi,
  0 <= lo < i32Modulus -> 0 <= hi < i32Modulus ->
  0 <= joinLimbs lo hi < i64Modulus.
Proof. unfold joinLimbs, i32Modulus, i64Modulus; intros; lia. Qed.

Lemma join_limbs_injective : forall lo hi lo' hi',
  0 <= lo < i32Modulus -> 0 <= lo' < i32Modulus ->
  joinLimbs lo hi = joinLimbs lo' hi' -> lo = lo' /\ hi = hi'.
Proof. unfold joinLimbs, i32Modulus; intros; lia. Qed.
