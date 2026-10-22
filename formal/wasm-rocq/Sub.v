(* The Rust-emitted i32.sub row computes Wasm subtraction in stack order. *)
From Stdlib Require Import List ZArith Lia Bool.
From mathcomp Require Import eqtype ssrbool seq.
Require Import R1CS Columns FullRelation Range Wasm2 WasmIntegers Add.
Import ListNotations.
Open Scope Z_scope.

Definition subSelector : nat := COL_SEL_I32_SUB.
Definition subRead0 : nat := List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat.
Definition subRead1 : nat := List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat.
Definition subWrite : nat := COL_STACK_WRITE0_VALUE_LO.
Definition subBorrow : nat := COL_WIDE_AUX0.

Definition subRow : Row :=
  ([(subSelector, 1)],
   [(subRead0, 1); (subRead1, -1); (subWrite, -1);
    (subBorrow, 4294967296)],
   []).

Definition sub_bounds (w : Witness) : Prop :=
  0 <= w subRead0 < i32Modulus /\
  0 <= w subRead1 < i32Modulus /\
  0 <= w subWrite < i32Modulus /\
  (w subBorrow = 0 \/ w subBorrow = 1).

Theorem sub_row_implies_modulo : forall (w : Witness),
  w subSelector = 1 -> sub_bounds w -> sat w subRow ->
  w subWrite = (w subRead0 - w subRead1) mod i32Modulus.
Proof.
  intros w Hselector [[Ha0 Ha1] [[Hb0 Hb1] [[Hz0 Hz1] Hborrow]]] Hsat.
  unfold sat, subRow, modulus in Hsat.
  cbn [eval fold_right] in Hsat.
  rewrite Hselector in Hsat.
  replace ((1 * 1 + 0) *
    (1 * w subRead0 +
    (-1 * w subRead1 + (-1 * w subWrite + (4294967296 * w subBorrow + 0)))) - 0)
    with (w subRead0 - w subRead1 - w subWrite + 4294967296 * w subBorrow)
    in Hsat by ring.
  unfold i32Modulus in *.
  pose proof (Z.mod_eq
    (w subRead0 - w subRead1 - w subWrite + 4294967296 * w subBorrow)
    18446744069414584321) as Hdiv.
  specialize (Hdiv ltac:(lia)).
  rewrite Hsat in Hdiv.
  assert (Hborrow_bound : 0 <= w subBorrow <= 1) by lia.
  assert (Heq : w subRead0 - w subRead1 - w subWrite + 4294967296 * w subBorrow = 0) by nia.
  replace (w subRead0 - w subRead1)
    with (w subWrite + (- w subBorrow) * 4294967296) by lia.
  rewrite Z.mod_add by lia.
  rewrite Z.mod_small by lia.
  reflexivity.
Qed.

Lemma spectec_i32_sub_value : forall a b z : Z,
  0 <= a < i32Modulus -> 0 <= b < i32Modulus -> 0 <= z ->
  z = (a - b) mod i32Modulus ->
  isub_ (sizenn (numtype_Inn Inn_I32))
    (mk_uN (Z.to_N a)) (mk_uN (Z.to_N b)) = mk_uN (Z.to_N z).
Proof.
  intros a b z [Ha0 Ha1] [Hb0 Hb1] Hz Hresult.
  change (sizenn (numtype_Inn Inn_I32)) with 32%N.
  unfold isub_.
  lazy beta iota zeta.
  f_equal.
  apply N2Z.inj.
  rewrite Z2N.id by (apply Z.mod_pos_bound; lia).
  rewrite Z2N.id by lia.
  cbn [coerce proj_uN_0].
  rewrite N2Z.inj_add.
  cbn [proj_uN_0_coercion].
  unfold coerce, proj_uN_0_coercion.
  cbn [proj_uN_0].
  rewrite !Z2N.id by lia.
  change ((i32Modulus + a - b) mod i32Modulus = z).
  replace (i32Modulus + a - b) with ((a - b) + 1 * i32Modulus) by ring.
  rewrite Z.mod_add by (unfold i32Modulus; lia).
  symmetry; exact Hresult.
Qed.

Theorem sub_row_implies_spectec_step : forall (w : Witness),
  w subSelector = 1 -> sub_bounds w -> sat w subRow ->
  Step_pure
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subRead0)))) ::
      admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subRead1)))) ::
      admininstr_BINOP I32 (mk_binop__0 Inn_I32 SUB) :: nil)%list)
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subWrite)))) :: nil)%list).
Proof.
  intros w Hselector Hbounds Hsat.
  pose proof (sub_row_implies_modulo w Hselector Hbounds Hsat) as Hmod.
  destruct Hbounds as [[Ha0 Ha1] [[Hb0 Hb1] [[Hz0 Hz1] Hborrow]]].
  pose proof (spectec_i32_sub_value _ _ _ (conj Ha0 Ha1) (conj Hb0 Hb1) Hz0 Hmod) as Hvalue.
  eapply binop_val with
    (var_0 := Some ((mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subWrite))) :: nil)%list)).
  - rewrite <- Hvalue.
    apply fun_binop___fun_binop__case_2.
    rewrite Hvalue.
    apply Add.spectec_i32_well_formed.
    split; assumption.
  - reflexivity.
  - reflexivity.
  - apply mem_head.
Qed.

Lemma sub_row_in_full_relation :
  nth_error FullRelation.rows FullRelation.subRowIndex = Some subRow.
Proof. vm_compute. reflexivity. Qed.

Theorem full_relation_implies_spectec_sub : forall (w : Witness),
  Range.fieldPrime ->
  w subSelector = 1 -> FullRelation.satFull w ->
  Step_pure
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subRead0)))) ::
      admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subRead1)))) ::
      admininstr_BINOP I32 (mk_binop__0 Inn_I32 SUB) :: nil)%list)
    ((admininstr_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N (w subWrite)))) :: nil)%list).
Proof.
  intros w Hprime Hselector Hfull.
  assert (Hbounds : sub_bounds w).
  { unfold sub_bounds, i32Modulus.
    split.
    - exact (Range.full_read0_u32 w Hprime Hfull).
    - split.
      + exact (Range.full_read1_u32 w Hprime Hfull).
      + split.
        * exact (Range.full_write0_u32 w Hprime Hfull).
        * exact (Range.full_wide_aux0_boolean w Hprime Hfull). }
  destruct Hfull as [_ Hall].
  apply sub_row_implies_spectec_step; try assumption.
  unfold satAll in Hall.
  assert (Hin : In subRow FullRelation.rows) by
    (eapply nth_error_In; exact sub_row_in_full_relation).
  exact ((proj1 (Forall_forall (sat w) FullRelation.rows) Hall) subRow Hin).
Qed.
