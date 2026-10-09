(* Reusable scalar zero-test rows and their value theorem. *)
From Stdlib Require Import List ZArith Znumtheory Lia.
Require Import R1CS Columns.
Import ListNotations.
Open Scope Z_scope.

Definition inverseRow (input inverse flag : nat) : Row :=
  ([(input, 1)], [(inverse, 1)], [(COL_ONE, 1); (flag, -1)]).

Definition zeroRow (input flag : nat) : Row :=
  ([(input, 1)], [(flag, 1)], []).

Theorem zero_test_value : forall (w : Witness) input inverse flag,
  prime modulus ->
  w COL_ONE = 1 ->
  sat w (inverseRow input inverse flag) ->
  sat w (zeroRow input flag) ->
  w flag = (if Z.eqb (w input) 0 then 1 else 0).
Proof.
  intros w input inverse flag Hprime Hone Hinv Hzero.
  unfold sat, inverseRow, zeroRow in *.
  cbn [eval fold_right] in Hinv, Hzero.
  rewrite Hone in Hinv.
  destruct (Z.eqb (w input) 0) eqn:Heq.
  - apply Z.eqb_eq in Heq.
    rewrite Heq in Hinv.
    pose proof (value_bound w flag) as [Hlo Hhi].
    replace ((1 * 0 + 0) * (1 * w inverse + 0) -
      (1 * 1 + (-1 * w flag + 0))) with (w flag - 1)
      in Hinv by ring.
    apply Z.mod_divide in Hinv; [|unfold modulus; lia].
    destruct Hinv as [k Hk].
    unfold modulus in *; nia.
  - apply Z.eqb_neq in Heq.
    pose proof (value_bound w input) as [Hi0 Hi1].
    pose proof (value_bound w flag) as [Hf0 Hf1].
    replace ((1 * w input + 0) * (1 * w flag + 0) - 0)
      with (w input * w flag) in Hzero by ring.
    apply Z.mod_divide in Hzero; [|unfold modulus; lia].
    apply prime_mult in Hzero; [|exact Hprime].
    destruct Hzero as [Hdiv|Hdiv]; destruct Hdiv as [k Hk];
      unfold modulus in *; nia.
Qed.
