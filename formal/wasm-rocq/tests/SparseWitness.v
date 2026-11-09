(* Sparse test fixtures are expanded once; proof checks still use dense lists. *)
From Stdlib Require Import List ZArith Bool.
Import ListNotations.
Open Scope Z_scope.

(* Entries must be strictly increasing and within the declared width. *)
Fixpoint expandFrom (width next : nat) (entries : list (nat * Z)) : option (list Z) :=
  match entries with
  | [] => Some (repeat 0 (width - next))
  | (column, value) :: rest =>
      if (next <=? column)%nat && (column <? width)%nat then
        match expandFrom width (S column) rest with
        | Some values => Some (repeat 0 (column - next) ++ value :: values)
        | None => None
        end
      else None
  end.

Definition expandWitness (width : nat) (entries : list (nat * Z)) : option (list Z) :=
  expandFrom width 0 entries.

(* Invalid rows become empty, and the fixture's expansion check rejects them. *)
Definition expandRows (width : nat) (rows : list (list (nat * Z))) : list (list Z) :=
  map (fun row => match expandWitness width row with Some values => values | None => [] end) rows.

Example fills_gaps : expandWitness 5 [(1%nat, 7); (3%nat, 9)] = Some [0; 7; 0; 9; 0].
Proof. reflexivity. Qed.
Example rejects_duplicate : expandWitness 5 [(1%nat, 7); (1%nat, 9)] = None.
Proof. reflexivity. Qed.
Example rejects_unordered : expandWitness 5 [(3%nat, 7); (1%nat, 9)] = None.
Proof. reflexivity. Qed.
Example rejects_out_of_bounds : expandWitness 5 [(5%nat, 7)] = None.
Proof. reflexivity. Qed.
