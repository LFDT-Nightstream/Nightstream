(* Range checks obtained from the exported Boolean and recomposition rows. *)
From Stdlib Require Import List ZArith Znumtheory Bool Lia.
Require Import R1CS Columns FullRelation.
Import ListNotations.
Open Scope Z_scope.

Definition fieldPrime : Prop := prime modulus.

Definition booleanRow (column : nat) : Row :=
  ([(column, 1)], [(column, 1); (COL_ONE, -1)], []).

Definition bitTerms (bits : list nat) : SparseRow :=
  map (fun '(index, column) => (column, 2 ^ Z.of_nat index))
    (combine (seq 0%nat (length bits)) bits).

Definition recomposeRow (column : nat) (bits : list nat) : Row :=
  (bitTerms bits, [(COL_ONE, 1)], [(column, 1)]).

Lemma sat_row : forall w n row,
  satAll w FullRelation.rows ->
  nth_error FullRelation.rows n = Some row -> sat w row.
Proof.
  intros w n row Hall Hrow.
  unfold satAll in Hall.
  apply (proj1 (Forall_forall (sat w) FullRelation.rows) Hall).
  eapply nth_error_In; exact Hrow.
Qed.

Lemma boolean_row_bound : forall (w : Witness) column,
  fieldPrime -> w COL_ONE = 1 -> sat w (booleanRow column) ->
  w column = 0 \/ w column = 1.
Proof.
  intros w column Hprime Hone Hsat.
  unfold sat, booleanRow, eval in Hsat; cbn in Hsat.
  rewrite Hone in Hsat.
  pose proof (value_bound w column) as [Hlo Hhi].
  assert (Hdiv : (modulus | w column * (w column - 1))).
  { apply Z.mod_divide; [unfold modulus; lia|].
    replace (w column * (w column - 1)) with
      ((1 * w column + 0) * (1 * w column + (-1 * 1 + 0)) - 0)
      by ring.
    exact Hsat. }
  apply prime_mult in Hdiv; [|exact Hprime].
  destruct Hdiv as [Hdiv|Hdiv]; destruct Hdiv as [k Hk];
    unfold modulus in *; nia.
Qed.

Lemma bits_bound : forall w terms,
  Forall (fun '(column, coeff) =>
    0 <= coeff /\ (w column = 0 \/ w column = 1)) terms ->
  0 <= eval w terms <= eval (fun _ => 1) terms.
Proof.
  intros w terms Hterms.
  induction Hterms as [|[column coeff] terms [Hcoeff Hbit] Hrest IH]; cbn.
  - lia.
  - change (0 <= coeff * w column + eval w terms <=
      coeff * 1 + eval (fun _ => 1) terms).
    destruct Hbit as [Hbit|Hbit]; rewrite Hbit; nia.
Qed.

Lemma range_from_rows : forall (w : Witness) column bits,
  fieldPrime -> w COL_ONE = 1 ->
  Forall (fun bit => sat w (booleanRow bit)) bits ->
  sat w (recomposeRow column bits) ->
  eval (fun _ => 1) (bitTerms bits) < modulus ->
  0 <= w column <= eval (fun _ => 1) (bitTerms bits).
Proof.
  intros w column bits Hprime Hone Hbits Hrecompose Hmax.
  assert (Hterms : Forall (fun '(column, coeff) =>
    0 <= coeff /\ (w column = 0 \/ w column = 1)) (bitTerms bits)).
  { unfold bitTerms.
    apply Forall_forall.
    intros [bit coeff] Hin.
    apply in_map_iff in Hin.
    destruct Hin as [[index bit'] [Heq Hin]].
    inversion Heq; subst; clear Heq.
    split; [apply Z.pow_nonneg; lia|].
    apply boolean_row_bound with (w := w); try assumption.
    apply (proj1 (Forall_forall _ _) Hbits).
    apply in_combine_r in Hin; exact Hin. }
  pose proof (bits_bound w (bitTerms bits) Hterms) as [Hlo Hhi].
  unfold sat, recomposeRow in Hrecompose.
  cbn [eval fold_right] in Hrecompose.
  rewrite Hone in Hrecompose.
  pose proof (value_bound w column) as [Hcol0 Hcol1].
  assert (Hsum : eval w (bitTerms bits) = w column).
  { apply Z.mod_divide in Hrecompose; [|unfold modulus; lia].
    destruct Hrecompose as [k Hk].
    unfold modulus in *.
    nia. }
  lia.
Qed.

Definition u32Bits (start : nat) : list nat := seq start 32%nat.

Lemma u32_max : forall start,
  eval (fun _ => 1) (bitTerms (u32Bits start)) = 4294967295.
Proof. intro start. unfold bitTerms, u32Bits, eval. cbn. reflexivity. Qed.

(* Locate rows by shape. The search is computation, not an assumption: a row it
   finds is checked equal to the requested one, so the circuit's row order and
   auxiliary columns never appear in a proof. *)
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

Lemma terms_equal_sound : forall a b, termsEqual a b = true -> a = b.
Proof.
  induction a as [ | [c k] a IH]; intros [ | [d l] b] H; try discriminate; [reflexivity | ].
  cbn in H. apply andb_prop in H as [H Hrest]. apply andb_prop in H as [Hc Hk].
  apply Nat.eqb_eq in Hc. apply Z.eqb_eq in Hk. subst. f_equal. exact (IH _ Hrest).
Qed.

Lemma rows_equal_sound : forall a b, rowsEqual a b = true -> a = b.
Proof.
  intros [[a1 a2] a3] [[b1 b2] b3] H. cbn in H.
  apply andb_prop in H as [H H3]. apply andb_prop in H as [H1 H2].
  rewrite (terms_equal_sound _ _ H1), (terms_equal_sound _ _ H2), (terms_equal_sound _ _ H3). reflexivity.
Qed.

Lemma find_row_sound : forall test rows index, findRow test rows = Some index ->
  exists row, nth_error rows index = Some row /\ test row = true.
Proof.
  intros test. induction rows as [ | row rest IH]; intros index H; cbn in H; [discriminate | ].
  destruct (test row) eqn:Htest.
  - injection H as <-. exists row. split; [reflexivity | exact Htest].
  - destruct (findRow test rest) as [n | ]; cbn in H; [ | discriminate]. injection H as <-.
    exact (IH n Logic.eq_refl).
Qed.

Definition inRows (rows : list Row) (row : Row) : bool :=
  match findRow (rowsEqual row) rows with Some _ => true | None => false end.

Lemma in_rows_sat : forall w rows row, satAll w rows -> inRows rows row = true -> sat w row.
Proof.
  intros w rows row Hall Hin. unfold inRows in Hin.
  destruct (findRow (rowsEqual row) rows) as [index | ] eqn:Hfind; [ | discriminate].
  destruct (find_row_sound _ _ _ Hfind) as [found [Hnth Hequal]].
  rewrite <- (rows_equal_sound _ _ Hequal) in Hnth.
  apply (proj1 (Forall_forall _ _) Hall). eapply nth_error_In; exact Hnth.
Qed.

(* The exported relation contains this row. *)
Definition inFull : Row -> bool := inRows FullRelation.rows.

Lemma in_full_sat : forall w row, satAll w FullRelation.rows -> inFull row = true -> sat w row.
Proof. intros. eapply in_rows_sat; eassumption. Qed.

Lemma full_boolean : forall (w : Witness) column,
  fieldPrime -> FullRelation.satFull w -> inFull (booleanRow column) = true -> w column = 0 \/ w column = 1.
Proof.
  intros w column Hprime [Hone Hall] Hin. exact (boolean_row_bound w column Hprime Hone (in_full_sat w _ Hall Hin)).
Qed.

(* Check the complete candidate before selecting it. Boolean rows may occur
   anywhere in the relation; only the bit columns themselves are consecutive. *)
Definition rangeRowChecked (rows : list Row) (column count : nat) (row : Row) : bool :=
  match row with
  | ((bit, _) :: _, _, _) =>
      let bits := seq bit count in
      if rowsEqual row (recomposeRow column bits) then
        forallb (fun b => inRows rows (booleanRow b)) bits
      else false
  | _ => false
  end.

Definition u32Checked (column : nat) : bool :=
  match findRow (rangeRowChecked FullRelation.rows column 32%nat) FullRelation.rows with
  | Some _ => true | None => false end.

Lemma full_u32 : forall (w : Witness) column,
  fieldPrime -> FullRelation.satFull w -> u32Checked column = true -> 0 <= w column < 4294967296.
Proof.
  intros w column Hprime [Hone Hall] Hchecked. unfold u32Checked in Hchecked.
  destruct (findRow _ FullRelation.rows) as [index | ] eqn:Hfind; [ | discriminate].
  destruct (find_row_sound _ _ _ Hfind) as [row [Hnth Hrow]].
  destruct row as [[[ | [bit coeff] rest] middle] last]; [discriminate | ].
  change ((if rowsEqual ((bit, coeff) :: rest, middle, last) (recomposeRow column (u32Bits bit))
    then forallb (fun b => inRows FullRelation.rows (booleanRow b)) (u32Bits bit)
    else false) = true) in Hrow.
  destruct (rowsEqual ((bit, coeff) :: rest, middle, last)
    (recomposeRow column (u32Bits bit))) eqn:Hrecompose; [ | discriminate].
  rename Hrow into Hbits.
  apply rows_equal_sound in Hrecompose.
  assert (Hbooleans : Forall (fun bit => sat w (booleanRow bit)) (u32Bits bit)).
  { apply Forall_forall. intros other Hother. apply (in_rows_sat w _ _ Hall).
    exact (proj1 (forallb_forall _ _) Hbits other Hother). }
  pose proof (sat_row w index _ Hall (eq_trans Hnth (f_equal (@Some Row) Hrecompose))) as Hsat.
  pose proof (range_from_rows w column (u32Bits bit) Hprime Hone Hbooleans Hsat)
    as Hbound.
  rewrite u32_max in Hbound. assert (4294967295 < modulus) by (unfold modulus; lia).
  specialize (Hbound H). lia.
Qed.

Lemma full_read0_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) < 4294967296.
Proof. intros w Hprime Hfull. apply (full_u32 w _ Hprime Hfull). vm_compute. reflexivity. Qed.

Lemma full_read1_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat) < 4294967296.
Proof. intros w Hprime Hfull. apply (full_u32 w _ Hprime Hfull). vm_compute. reflexivity. Qed.

Lemma full_write0_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < 4294967296.
Proof. intros w Hprime Hfull. apply (full_u32 w _ Hprime Hfull). vm_compute. reflexivity. Qed.

Lemma full_local_value_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w COL_LOCAL_VALUE < 4294967296.
Proof. intros w Hprime Hfull. apply (full_u32 w _ Hprime Hfull). vm_compute. reflexivity. Qed.

Lemma full_eq_selector_boolean : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_I32_EQ = 0 \/ w COL_SEL_I32_EQ = 1.
Proof. intros w Hprime Hfull. apply (full_boolean w _ Hprime Hfull). vm_compute. reflexivity. Qed.

Lemma full_wide_aux0_boolean : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_WIDE_AUX0 = 0 \/ w COL_WIDE_AUX0 = 1.
Proof. intros w Hprime Hfull. apply (full_boolean w _ Hprime Hfull). vm_compute. reflexivity. Qed.
