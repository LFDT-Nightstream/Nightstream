(* The exported i32.eqz rows refine SpecTec's test operation. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import eqtype ssrbool seq.
Require Import R1CS Columns FullRelation Range ZeroTest Wasm2 WasmIntegers Add.
Import ListNotations.
Open Scope Z_scope.

Lemma spectec_eqz_value : forall a : Z,
  0 <= a ->
  ieqz_ (sizenn (numtype_Inn Inn_I32)) (mk_uN (Z.to_N a)) =
    mk_uN (Z.to_N (if Z.eqb a 0 then 1 else 0)).
Proof.
  intros a Ha.
  unfold ieqz_, sizenn, numtype_Inn, res_bool.
  cbn.
  destruct (Z.to_N a == 0%N) eqn:Hn.
  - apply (elimT eqP) in Hn.
    apply (f_equal Z.of_N) in Hn.
    rewrite Z2N.id in Hn by lia.
    subst a. reflexivity.
  - apply (elimF eqP) in Hn.
    assert (a <> 0) by (intro Heq; subst a; apply Hn; reflexivity).
    apply Z.eqb_neq in H.
    rewrite H.
    unfold N_eqb, ssrbool.is_left.
    destruct (N_eq_dec (Z.to_N a) 0%N); [contradiction|reflexivity].
Qed.

Definition cmpInput : nat := COL_CMP_LO_DIFF.
Definition cmpInverse : nat := COL_CMP_LO_INV.
Definition cmpFlag : nat := COL_CMP_LO_IS_ZERO.
Definition eqzSelector : nat := COL_SEL_I32_EQZ.
Definition eqzRead : nat := List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat.
Definition eqzWrite : nat := COL_STACK_WRITE0_VALUE_LO.

Definition pinRow : Row :=
  ([(eqzSelector, 1)], [(cmpInput, 1); (eqzRead, -1)], []).

Definition writeRow : Row :=
  ([(eqzSelector, 1); (COL_SEL_I32_EQ, 1)],
   [(eqzWrite, 1); (cmpFlag, -1)], []).

Lemma inverse_row_in_full :
  nth_error FullRelation.rows 515%nat =
    Some (inverseRow cmpInput cmpInverse cmpFlag).
Proof. vm_compute. reflexivity. Qed.

Lemma zero_row_in_full :
  nth_error FullRelation.rows 516%nat = Some (zeroRow cmpInput cmpFlag).
Proof. vm_compute. reflexivity. Qed.

Lemma pin_row_in_full :
  nth_error FullRelation.rows 512%nat = Some pinRow.
Proof. vm_compute. reflexivity. Qed.

Lemma write_row_in_full :
  nth_error FullRelation.rows 522%nat = Some writeRow.
Proof. vm_compute. reflexivity. Qed.

Theorem full_relation_eqz_value : forall (w : Witness),
  Range.fieldPrime ->
  w eqzSelector = 1 ->
  FullRelation.satFull w ->
  w eqzWrite = (if Z.eqb (w eqzRead) 0 then 1 else 0).
Proof.
  intros w Hprime Hsel [Hone Hall].
  pose proof (sat_row w 515%nat _ Hall inverse_row_in_full) as Hinv.
  pose proof (sat_row w 516%nat _ Hall zero_row_in_full) as Hzero.
  pose proof (zero_test_value w cmpInput cmpInverse cmpFlag
    Hprime Hone Hinv Hzero) as Hvalue.
  pose proof (sat_row w 512%nat _ Hall pin_row_in_full) as Hpin.
  pose proof (sat_row w 522%nat _ Hall write_row_in_full) as Hwrite.
  pose proof (Range.full_eq_selector_boolean w Hprime (conj Hone Hall)) as Heq.
  unfold sat, pinRow in Hpin; cbn [eval fold_right] in Hpin.
  rewrite Hsel in Hpin.
  assert (Hinput : w cmpInput = w eqzRead).
  { pose proof (value_bound w cmpInput) as [Hlo Hhi].
    pose proof (full_read0_u32 w Hprime (conj Hone Hall)) as [Hread0 Hread1].
    change (0 <= w eqzRead) in Hread0.
    change (w eqzRead < 4294967296) in Hread1.
    replace ((1 * 1 + 0) * (1 * w cmpInput + (-1 * w eqzRead + 0)) - 0)
      with (w cmpInput - w eqzRead) in Hpin by ring.
    apply Z.mod_divide in Hpin; try (unfold modulus; lia).
    destruct Hpin as [k Hk].
    unfold modulus in *; lia. }
  unfold sat, writeRow in Hwrite; cbn [eval fold_right] in Hwrite.
  rewrite Hsel in Hwrite.
  replace (((1 * 1 + (1 * w COL_SEL_I32_EQ + 0)) *
    (1 * w eqzWrite + (-1 * w cmpFlag + 0)) - 0))
    with ((1 + w COL_SEL_I32_EQ) * (w eqzWrite - w cmpFlag))
    in Hwrite by ring.
  assert (Houtput : w eqzWrite = w cmpFlag).
  { pose proof (full_write0_u32 w Hprime (conj Hone Hall)) as [Hlo Hhi].
    change (0 <= w eqzWrite) in Hlo.
    change (w eqzWrite < 4294967296) in Hhi.
    destruct Heq as [Heq|Heq]; rewrite Heq in Hwrite;
    destruct (Z.eqb (w cmpInput) 0) eqn:Hz; rewrite Hvalue in Hwrite;
      apply Z.mod_divide in Hwrite; try (unfold modulus; lia);
      destruct Hwrite as [k Hk]; unfold modulus in *; nia. }
  rewrite Houtput, <- Hinput.
  exact Hvalue.
Qed.

Theorem full_relation_implies_spectec_eqz : forall (w : Witness),
  Range.fieldPrime ->
  w eqzSelector = 1 ->
  FullRelation.satFull w ->
  Step_pure
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w eqzRead)))) ::
      admininstr_TESTOP I32 (mk_testop__0 Inn_I32 EQZ) :: nil)%list)
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w eqzWrite)))) :: nil)%list).
Proof.
  intros w Hprime Hselector Hfull.
  pose proof (full_relation_eqz_value w Hprime Hselector Hfull) as Hvalue.
  pose proof (full_read0_u32 w Hprime Hfull) as [Hread0 Hread1].
  change (0 <= w eqzRead) in Hread0.
  pose proof (full_write0_u32 w Hprime Hfull) as Hwrite.
  change (0 <= w eqzWrite < i32Modulus) in Hwrite.
  eapply Step_pure__testop.
  - reflexivity.
  - rewrite Hvalue.
    unfold fun_testop_.
    cbn.
    rewrite spectec_eqz_value by exact Hread0.
    unfold num__eqb, ssrbool.is_left.
    destruct (num__eq_dec _ _); [reflexivity|contradiction].
  - apply Add.spectec_i32_well_formed.
    exact Hwrite.
Qed.
