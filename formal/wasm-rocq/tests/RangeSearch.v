(* Range discovery survives reordered constraints and misleading partial matches. *)
From Stdlib Require Import List ZArith Bool.
Require Import R1CS Columns Range.
Import ListNotations.
Open Scope Z_scope.

Definition target : nat := 10%nat.
Definition bits := u32Bits 100%nat.
Definition recomposition := recomposeRow target bits.
Definition booleans := map booleanRow bits.
Definition redundant : Row := ([(target, 1)], [(COL_ONE, 1)], [(target, 1)]).

Example ignores_earlier_partial_match :
  findRow (rangeRowChecked (redundant :: recomposition :: booleans) target 32)
    (redundant :: recomposition :: booleans) = Some 1%nat.
Proof. vm_compute. reflexivity. Qed.

(* The recomposition precedes its Boolean rows, which are also reversed. *)
Example boolean_row_order_is_irrelevant :
  rangeRowChecked (recomposition :: rev booleans) target 32 recomposition = true.
Proof. vm_compute. reflexivity. Qed.

(* Even a complete recomposition is skipped if its bits lack Boolean checks. *)
Definition unchecked := recomposeRow target (u32Bits 200%nat).
Example skips_unchecked_decomposition :
  findRow (rangeRowChecked (unchecked :: recomposition :: booleans) target 32)
    (unchecked :: recomposition :: booleans) = Some 1%nat.
Proof. vm_compute. reflexivity. Qed.

Example missing_boolean_rejected :
  rangeRowChecked (recomposition :: tl booleans) target 32 recomposition = false.
Proof. vm_compute. reflexivity. Qed.

Example wrong_coefficient_rejected :
  rangeRowChecked booleans target 32
    ((100%nat, 2) :: tl (bitTerms bits), [(COL_ONE, 1)], [(target, 1)]) = false.
Proof. vm_compute. reflexivity. Qed.
