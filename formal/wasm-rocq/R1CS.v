(* Integer representatives of the Goldilocks R1CS relation. *)
From Stdlib Require Import List ZArith.
Import ListNotations.
Open Scope Z_scope.

Definition modulus : Z := 18446744069414584321.
Definition Column := nat.
Definition SparseRow := list (Column * Z).
Definition Row := (SparseRow * SparseRow * SparseRow)%type.
Definition FieldElement := { z : Z | 0 <= z < modulus }.
Record Witness := { entries : Column -> FieldElement }.

Definition value (w : Witness) (column : Column) : Z :=
  proj1_sig (entries w column).
Coercion value : Witness >-> Funclass.

Lemma value_bound : forall (w : Witness) (column : Column),
  0 <= w column < modulus.
Proof.
  intros [entries] column.
  exact (proj2_sig (entries column)).
Qed.

Definition eval (w : Column -> Z) (terms : SparseRow) : Z :=
  fold_right (fun '(column, coefficient) acc => coefficient * w column + acc) 0 terms.

Definition sat (w : Witness) (row : Row) : Prop :=
  let '(a, b, c) := row in
  (eval w a * eval w b - eval w c) mod modulus = 0.

Definition satAll (w : Witness) (rows : list Row) : Prop :=
  Forall (sat w) rows.
