(* The Rust-emitted i32.add row, interpreted over Goldilocks, computes Wasm addition. *)
From Stdlib Require Import List ZArith Lia Bool.
From mathcomp Require Import eqtype ssrbool seq.
Require Import R1CS Columns FullRelation Range Wasm2 WasmIntegers.
Import ListNotations.
Open Scope Z_scope.

Definition addSelector : nat := COL_SEL_I32_ADD.
Definition addRead0 : nat := List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat.
Definition addRead1 : nat := List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat.
Definition addWrite : nat := COL_STACK_WRITE0_VALUE_LO.
Definition addCarry : nat := COL_WIDE_AUX0.

Definition addRow : Row :=
  ([(addSelector, 1)],
   [(addRead0, 1); (addRead1, 1); (addWrite, -1); (addCarry, -4294967296)],
   []).

Definition add_bounds (w : Witness) : Prop :=
  0 <= w addRead0 < i32Modulus /\
  0 <= w addRead1 < i32Modulus /\
  0 <= w addWrite < i32Modulus /\
  (w addCarry = 0 \/ w addCarry = 1).

Theorem add_row_implies_modulo : forall (w : Witness),
  w addSelector = 1 -> add_bounds w -> sat w addRow ->
  w addWrite = (w addRead0 + w addRead1) mod i32Modulus.
Proof.
  intros w Hselector [[Ha0 Ha1] [[Hb0 Hb1] [[Hz0 Hz1] Hcarry]]] Hsat.
  unfold sat, addRow, modulus in Hsat.
  cbn [eval fold_right] in Hsat.
  rewrite Hselector in Hsat.
  replace ((1 * 1 + 0) *
    (1 * w addRead0 +
    (1 * w addRead1 + (-1 * w addWrite + (-4294967296 * w addCarry + 0)))) - 0)
    with (w addRead0 + w addRead1 - w addWrite - 4294967296 * w addCarry)
    in Hsat by ring.
  unfold i32Modulus in *.
  pose proof (Z.mod_eq
    (w addRead0 + w addRead1 - w addWrite - 4294967296 * w addCarry)
    18446744069414584321) as Hdiv.
  specialize (Hdiv ltac:(lia)).
  rewrite Hsat in Hdiv.
  assert (Hcarry_bound : 0 <= w addCarry <= 1) by lia.
  assert (Heq : w addRead0 + w addRead1 - w addWrite - 4294967296 * w addCarry = 0) by nia.
  replace (w addRead0 + w addRead1)
    with (4294967296 * w addCarry + w addWrite) by lia.
  rewrite Z.add_comm.
  replace (4294967296 * w addCarry) with (w addCarry * 4294967296) by ring.
  rewrite Z.mod_add by lia.
  rewrite Z.mod_small by lia.
  reflexivity.
Qed.

Lemma spectec_i32_well_formed : forall z : Z,
  0 <= z < i32Modulus -> wf_num_ I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N z))).
Proof.
  intros z [Hz0 Hz1].
  apply num__case_0.
  - reflexivity.
  - apply uN_case_0.
    cbn.
    change (((0 <=? Z.to_N z)%N && (Z.to_N z <=? 4294967295)%N) = true).
    apply andb_true_iff.
    split.
    + apply N.leb_le. lia.
    + apply N.leb_le.
      apply N2Z.inj_le.
      rewrite Z2N.id by lia.
      unfold i32Modulus in Hz1.
      lia.
  - reflexivity.
Qed.

Lemma spectec_i32_add_value : forall a b z : Z,
  0 <= a -> 0 <= b -> 0 <= z ->
  z = (a + b) mod i32Modulus ->
  iadd_ (sizenn (numtype_Inn Inn_I32))
    (mk_uN (Z.to_N a)) (mk_uN (Z.to_N b)) = mk_uN (Z.to_N z).
Proof.
  intros a b z Ha Hb Hz Hresult.
  unfold iadd_, sizenn, numtype_Inn.
  cbn.
  f_equal.
  apply N2Z.inj.
  rewrite N2Z.inj_mod by discriminate.
  rewrite N2Z.inj_add.
  rewrite !Z2N.id by lia.
  change (Z.of_N (2 ^ 32)) with i32Modulus.
  symmetry.
  exact Hresult.
Qed.

Theorem add_row_implies_spectec_step : forall (w : Witness),
  w addSelector = 1 -> add_bounds w -> sat w addRow ->
  Step_pure
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addRead0)))) ::
      admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addRead1)))) ::
      admininstr_BINOP I32 (mk_binop__0 Inn_I32 ADD) :: nil)%list)
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addWrite)))) :: nil)%list).
Proof.
  intros w Hselector Hbounds Hsat.
  pose proof (add_row_implies_modulo w Hselector Hbounds Hsat) as Hmod.
  destruct Hbounds as [[Ha0 Ha1] [[Hb0 Hb1] [[Hz0 Hz1] Hcarry]]].
  pose proof (spectec_i32_add_value _ _ _ Ha0 Hb0 Hz0 Hmod) as Hvalue.
  eapply binop_val with
    (var_0 := Some ((mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addWrite))) :: nil)%list)).
  - rewrite <- Hvalue.
    apply fun_binop___fun_binop__case_0.
    rewrite Hvalue.
    apply spectec_i32_well_formed.
    split; assumption.
  - reflexivity.
  - reflexivity.
  - apply mem_head.
Qed.

Lemma add_row_in_full_relation :
  nth_error FullRelation.rows FullRelation.addRowIndex = Some addRow.
Proof. vm_compute. reflexivity. Qed.

Theorem full_relation_implies_spectec_add : forall (w : Witness),
  Range.fieldPrime ->
  w addSelector = 1 -> FullRelation.satFull w ->
  Step_pure
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addRead0)))) ::
      admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addRead1)))) ::
      admininstr_BINOP I32 (mk_binop__0 Inn_I32 ADD) :: nil)%list)
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w addWrite)))) :: nil)%list).
Proof.
  intros w Hprime Hselector Hfull.
  assert (Hbounds : add_bounds w).
  { unfold add_bounds.
    split.
    - exact (Range.full_read0_u32 w Hprime Hfull).
    - split.
      + exact (Range.full_read1_u32 w Hprime Hfull).
      + split.
        * exact (Range.full_write0_u32 w Hprime Hfull).
        * exact (Range.full_wide_aux0_boolean w Hprime Hfull). }
  destruct Hfull as [_ Hall].
  apply add_row_implies_spectec_step; try assumption.
  unfold satAll in Hall.
  assert (Hin : In addRow FullRelation.rows) by
    (eapply nth_error_In; exact add_row_in_full_relation).
  exact ((proj1 (Forall_forall (sat w) FullRelation.rows) Hall) addRow Hin).
Qed.
