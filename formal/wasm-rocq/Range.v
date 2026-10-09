(* Range checks obtained from the exported Boolean and recomposition rows. *)
From Stdlib Require Import List ZArith Znumtheory Lia.
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

Lemma sat_slice : forall w start count,
  satAll w FullRelation.rows ->
  Forall (sat w) (firstn count (skipn start FullRelation.rows)).
Proof.
  intros w start count Hall.
  unfold satAll in Hall.
  apply Forall_forall.
  intros row Hin.
  apply (proj1 (Forall_forall (sat w) FullRelation.rows) Hall).
  rewrite <- (firstn_skipn start FullRelation.rows).
  apply in_or_app.
  right.
  rewrite <- (firstn_skipn count (skipn start FullRelation.rows)).
  apply in_or_app.
  left; exact Hin.
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

Definition rangeBlock (rowStart bitStart : nat) : Prop :=
  firstn 32%nat (skipn rowStart FullRelation.rows) =
    map booleanRow (u32Bits bitStart).

Lemma u32_from_full : forall (w : Witness) column rowStart bitStart,
  fieldPrime -> FullRelation.satFull w ->
  rangeBlock rowStart bitStart ->
  nth_error FullRelation.rows (rowStart + 32)%nat =
    Some (recomposeRow column (u32Bits bitStart)) ->
  0 <= w column < 4294967296.
Proof.
  intros w column rowStart bitStart Hprime [Hone Hall] Hblock Hrow.
  assert (Hbits : Forall (fun bit => sat w (booleanRow bit)) (u32Bits bitStart)).
  { pose proof (sat_slice w rowStart 32%nat Hall) as Hslice.
    rewrite Hblock in Hslice.
    exact (proj1 (Forall_map booleanRow (sat w) (u32Bits bitStart)) Hslice). }
  pose proof (sat_row w _ _ Hall Hrow) as Hrecompose.
  pose proof (range_from_rows w column (u32Bits bitStart)
    Hprime Hone Hbits Hrecompose) as Hbound.
  rewrite u32_max in Hbound.
  assert (4294967295 < modulus) by (unfold modulus; lia).
  specialize (Hbound H).
  lia.
Qed.

Lemma read0_range_block : rangeBlock 2557%nat 2243%nat.
Proof. vm_compute. reflexivity. Qed.
Lemma read0_recompose :
  nth_error FullRelation.rows 2589%nat =
  Some (recomposeRow (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)
    (u32Bits 2243%nat)).
Proof. vm_compute. reflexivity. Qed.
Lemma read1_range_block : rangeBlock 2590%nat 2275%nat.
Proof. vm_compute. reflexivity. Qed.
Lemma read1_recompose :
  nth_error FullRelation.rows 2622%nat =
  Some (recomposeRow (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)
    (u32Bits 2275%nat)).
Proof. vm_compute. reflexivity. Qed.
Lemma write0_range_block : rangeBlock 2821%nat 2499%nat.
Proof. vm_compute. reflexivity. Qed.
Lemma write0_recompose :
  nth_error FullRelation.rows 2853%nat =
  Some (recomposeRow COL_STACK_WRITE0_VALUE_LO (u32Bits 2499%nat)).
Proof. vm_compute. reflexivity. Qed.

Lemma full_read0_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) < 4294967296.
Proof.
  intros w Hprime Hsat.
  exact (u32_from_full w _ 2557%nat 2243%nat Hprime Hsat
    read0_range_block read0_recompose).
Qed.

Lemma full_read1_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat) < 4294967296.
Proof.
  intros w Hprime Hsat.
  exact (u32_from_full w _ 2590%nat 2275%nat Hprime Hsat
    read1_range_block read1_recompose).
Qed.

Lemma full_write0_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w COL_STACK_WRITE0_VALUE_LO < 4294967296.
Proof.
  intros w Hprime Hsat.
  exact (u32_from_full w _ 2821%nat 2499%nat Hprime Hsat
    write0_range_block write0_recompose).
Qed.

Lemma local_value_range_block : rangeBlock 1779%nat 1491%nat.
Proof. vm_compute. reflexivity. Qed.
Lemma local_value_recompose :
  nth_error FullRelation.rows 1811%nat =
  Some (recomposeRow COL_LOCAL_VALUE (u32Bits 1491%nat)).
Proof. vm_compute. reflexivity. Qed.

Lemma full_local_value_u32 : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  0 <= w COL_LOCAL_VALUE < 4294967296.
Proof.
  intros w Hprime Hsat.
  exact (u32_from_full w _ 1779%nat 1491%nat Hprime Hsat
    local_value_range_block local_value_recompose).
Qed.

Lemma wide_aux0_boolean_row :
  nth_error FullRelation.rows 2887%nat = Some (booleanRow COL_WIDE_AUX0).
Proof. vm_compute. reflexivity. Qed.

Lemma eq_selector_boolean_row :
  nth_error FullRelation.rows 1625%nat = Some (booleanRow COL_SEL_I32_EQ).
Proof. vm_compute. reflexivity. Qed.

Lemma full_eq_selector_boolean : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_SEL_I32_EQ = 0 \/ w COL_SEL_I32_EQ = 1.
Proof.
  intros w Hprime [Hone Hall].
  exact (boolean_row_bound w COL_SEL_I32_EQ Hprime Hone
    (sat_row w 1625%nat _ Hall eq_selector_boolean_row)).
Qed.

Lemma full_wide_aux0_boolean : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_WIDE_AUX0 = 0 \/ w COL_WIDE_AUX0 = 1.
Proof.
  intros w Hprime [Hone Hall].
  exact (boolean_row_bound w COL_WIDE_AUX0 Hprime Hone
    (sat_row w 2887%nat _ Hall wide_aux0_boolean_row)).
Qed.
