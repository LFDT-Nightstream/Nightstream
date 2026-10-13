(* Locate concrete rows by shape; the kernel checks each resulting lookup.
   Search is proof automation, not an additional assumption about the circuit. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import R1CS Columns FullRelation Range Selectors ZeroTest.
Import ListNotations.
Open Scope Z_scope.

Fixpoint termsEqual (a b : SparseRow) : bool :=
  match a, b with
  | [], [] => true
  | (c, k) :: a, (d, l) :: b => Nat.eqb c d && Z.eqb k l && termsEqual a b
  | _, _ => false
  end.

Definition rowsEqual (a b : Row) : bool :=
  let '(a1, a2, a3) := a in let '(b1, b2, b3) := b in
  termsEqual a1 b1 && termsEqual a2 b2 && termsEqual a3 b3.

Fixpoint findRow (test : Row -> bool) (rows : list Row) : option nat :=
  match rows with
  | [] => None
  | row :: rest => if test row then Some 0%nat else option_map S (findRow test rest)
  end.

Ltac circuit_row shape name :=
  match goal with Hfull : FullRelation.satFull ?w |- _ =>
    let found := eval vm_compute in (findRow (rowsEqual shape) FullRelation.rows) in
    lazymatch found with
    | Some ?index =>
      let Hlookup := fresh "Hlookup" in
      assert (Hlookup : nth_error FullRelation.rows index = Some shape)
        by (vm_compute; reflexivity);
      pose proof (Range.sat_row w index shape (proj2 Hfull) Hlookup) as name;
      clear Hlookup
    | None => fail "Missing circuit row" shape
    end
  end.

Ltac circuit_boolean column name :=
  let Hrow := fresh "Hrow" in
  circuit_row (Range.booleanRow column) Hrow;
  match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    pose proof (Range.boolean_row_bound w column Hprime (proj1 Hfull) Hrow) as name;
    clear Hrow
  end.

Ltac circuit_range column name :=
  let found := eval vm_compute in (findRow
    (fun row => let '(_, b, c) := row in
      termsEqual b [(COL_ONE, 1)] && termsEqual c [(column, 1)]) FullRelation.rows) in
  lazymatch found with
  | Some ?index =>
    let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
    lazymatch raw with ((?bit, _) :: ?tail, _, _) =>
      let count := eval compute in (S (length tail)) in
      let start := eval compute in (index - count)%nat in
      let bits := constr:(seq bit count) in
      let Hblock := fresh "Hblock" in let Hslice := fresh "Hslice" in
      let Hbits := fresh "Hbits" in let Hrow := fresh "Hrow" in
      assert (Hblock : firstn count (skipn start FullRelation.rows) = map booleanRow bits)
        by (vm_compute; reflexivity);
      match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
        pose proof (Range.sat_slice w start count (proj2 Hfull)) as Hslice;
        rewrite Hblock in Hslice;
        pose proof (proj1 (Forall_map booleanRow (sat w) bits) Hslice) as Hbits;
        circuit_row (recomposeRow column bits) Hrow;
        pose proof (range_from_rows w column bits Hprime (proj1 Hfull) Hbits Hrow
          ltac:(vm_compute; reflexivity)) as name;
        let maximum := eval vm_compute in (eval (fun _ => 1) (bitTerms bits)) in
        change (0 <= w column <= maximum) in name;
        clear Hblock Hslice Hbits Hrow
      end
    end
  | None => fail "Missing range recomposition" column
  end.

Ltac circuit_zero_test input inverse flag name :=
  let Hinv := fresh "Hinv" in let Hzero := fresh "Hzero" in
  circuit_row (inverseRow input inverse flag) Hinv;
  circuit_row (zeroRow input flag) Hzero;
  match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    pose proof (zero_test_value w input inverse flag Hprime (proj1 Hfull) Hinv Hzero) as name;
    clear Hinv Hzero
  end.

Ltac expand_row H := unfold sat in H; cbn [eval fold_right] in H; normalize_columns.

Lemma full_aux_selectors_zero : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 ->
  forall column, In column selectorColumns -> w column = 0.
Proof.
  intros w Hprime Hfull Hprogram column Hin.
  destruct (full_opcode_selectors_one_hot w Hprime Hfull) as [_ [Hsum _]].
  pose proof (selector_sum_one w selectorColumns column Hin).
  pose proof (value_bound w column). lia.
Qed.

Ltac zero_opcode_columns Hzero H :=
  repeat match type of H with
  | context [value ?w ?column] =>
      let member := eval vm_compute in (existsb (Nat.eqb column) selectorColumns) in
      lazymatch member with
      | true => rewrite (Hzero column ltac:(vm_compute; tauto)) in H
      end
  end.

Definition auxiliaryTerms (terms : SparseRow) : SparseRow :=
  filter (fun entry => negb (existsb (Nat.eqb (fst entry)) selectorColumns)) terms.

Definition auxiliaryRow (row : Row) : Row :=
  let '(a, b, c) := row in (auxiliaryTerms a, auxiliaryTerms b, auxiliaryTerms c).

Lemma auxiliary_terms_eval : forall (w : Witness) terms,
  (forall column, In column selectorColumns -> w column = 0) ->
  eval w (auxiliaryTerms terms) = eval w terms.
Proof.
  intros w terms Hzero. induction terms as [ | [column coefficient] rest IH]; [reflexivity | ].
  cbn [auxiliaryTerms filter fst].
  destruct (existsb (Nat.eqb column) selectorColumns) eqn:Hmember.
  - apply existsb_exists in Hmember. destruct Hmember as [c [Hin Heq]].
    apply Nat.eqb_eq in Heq. subst c. cbn [negb eval fold_right].
    rewrite Hzero by exact Hin. change (eval w (auxiliaryTerms rest) = coefficient * 0 + eval w rest).
    rewrite IH. ring.
  - cbn [negb eval fold_right].
    change (coefficient * w column + eval w (auxiliaryTerms rest) = coefficient * w column + eval w rest).
    rewrite IH. reflexivity.
Qed.

Lemma auxiliary_row_sat : forall w row,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 ->
  sat w row -> sat w (auxiliaryRow row).
Proof.
  intros w [[a b] c] Hprime Hfull Hprogram Hsat.
  unfold sat, auxiliaryRow. rewrite !auxiliary_terms_eval
    by exact (full_aux_selectors_zero w Hprime Hfull Hprogram).
  exact Hsat.
Qed.

Ltac auxiliary_row index name :=
  match goal with
  | Hprime : fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
    let reduced := eval vm_compute in (auxiliaryRow raw) in
    let Hsat := fresh "Hsat" in
    pose proof (Range.sat_row w index raw (proj2 Hfull) eq_refl) as Hsat;
    pose proof (auxiliary_row_sat w raw Hprime Hfull ltac:(assumption) Hsat) as name;
    clear Hsat; change (sat w reduced) in name; expand_row name
  end.

Ltac selected_circuit_row chosen shape name :=
  let found := eval vm_compute in (findRow (rowsEqual shape) FullRelation.rows) in
  lazymatch found with
  | Some ?index => selected_row chosen index name
  | None => fail "Missing circuit row" shape
  end.
