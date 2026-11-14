(* Local stack-port bindings and their SpecTec state steps. *)
From Stdlib Require Import List ZArith NArith Lia.
Require Import R1CS Columns FullRelation Range Wasm2 WasmIntegers CircuitFacts.
Require Export WasmValues.
Import ListNotations.
Open Scope Z_scope.

Definition localGetRow : Row :=
  ([(COL_SEL_LOCAL_GET, 1)],
   [(COL_LOCAL_VALUE, 1); (COL_STACK_WRITE0_VALUE_LO, -1)], []).

Definition localSetRow : Row :=
  ([(COL_SEL_LOCAL_SET, 1)],
   [(COL_LOCAL_VALUE, 1); (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], []).

Lemma local_get_row_in_full :
  nth_error FullRelation.rows localGetCopyRowIndex = Some localGetRow.
Proof. vm_compute. reflexivity. Qed.

Lemma local_set_row_in_full :
  nth_error FullRelation.rows localSetCopyRowIndex = Some localSetRow.
Proof. vm_compute. reflexivity. Qed.

Lemma active_copy_row : forall (w : Witness) selector source dest,
  w selector = 1 ->
  sat w ([(selector, 1)], [(source, 1); (dest, -1)], []) ->
  w source = w dest.
Proof.
  intros w selector source dest Hsel Hsat.
  unfold sat in Hsat; cbn [eval fold_right] in Hsat.
  rewrite Hsel in Hsat.
  replace ((1 * 1 + 0) * (1 * w source + (-1 * w dest + 0)) - 0)
    with (w source - w dest) in Hsat by ring.
  apply Z.mod_divide in Hsat; try (unfold modulus; lia).
  destruct Hsat as [k Hk].
  pose proof (value_bound w source) as [Hs0 Hs1].
  pose proof (value_bound w dest) as [Hd0 Hd1].
  unfold modulus in *; lia.
Qed.

Lemma full_local_get_value : forall (w : Witness),
  w COL_SEL_LOCAL_GET = 1 -> FullRelation.satFull w ->
  w COL_LOCAL_VALUE = w COL_STACK_WRITE0_VALUE_LO.
Proof.
  intros w Hsel [_ Hall].
  eapply active_copy_row.
  - exact Hsel.
  - exact (sat_row w localGetCopyRowIndex _ Hall local_get_row_in_full).
Qed.

Lemma full_local_set_value : forall (w : Witness),
  w COL_SEL_LOCAL_SET = 1 -> FullRelation.satFull w ->
  w COL_LOCAL_VALUE = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat).
Proof.
  intros w Hsel [_ Hall].
  eapply active_copy_row.
  - exact Hsel.
  - exact (sat_row w localSetCopyRowIndex _ Hall local_set_row_in_full).
Qed.

Lemma full_local_value_wf : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  wf_val (i32Value (w COL_LOCAL_VALUE)).
Proof.
  intros w Hprime Hfull.
  apply val_case_0.
  apply WasmValues.spectec_i32_well_formed.
  exact (Range.full_local_value_u32 w Hprime Hfull).
Qed.

Theorem full_relation_implies_local_get : forall (w : Witness) (z : state) (x : localidx),
  w COL_SEL_LOCAL_GET = 1 -> FullRelation.satFull w ->
  fun_local z x = i32Value (w COL_LOCAL_VALUE) ->
  Step (mk_config z ((admininstr_LOCAL_GET x :: nil)%list))
    (mk_config z ((admininstr_val (i32Value (w COL_STACK_WRITE0_VALUE_LO)) :: nil)%list)).
Proof.
  intros w z x Hsel Hfull Hlocal.
  pose proof (full_local_get_value w Hsel Hfull) as Hport.
  rewrite <- Hport.
  rewrite <- Hlocal.
  apply read.
  apply Step_read__local_get.
Qed.

Theorem full_relation_implies_local_set : forall (w : Witness) (z : state) (x : localidx),
  w COL_SEL_LOCAL_SET = 1 -> FullRelation.satFull w ->
  Step (mk_config z
      ((admininstr_val (i32Value (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))) ::
        admininstr_LOCAL_SET x :: nil)%list))
    (mk_config (with_local z x (i32Value (w COL_LOCAL_VALUE))) nil).
Proof.
  intros w z x Hsel Hfull.
  pose proof (full_local_set_value w Hsel Hfull) as Hport.
  rewrite <- Hport.
  apply Step__local_set.
Qed.

Lemma full_local_get_high_value : forall (w : Witness),
  w COL_SEL_LOCAL_GET = 1 -> FullRelation.satFull w ->
  w COL_LOCAL_VALUE_HI = w COL_STACK_WRITE0_VALUE_HI.
Proof.
  intros w Hsel Hfull.
  circuit_row ([(COL_SEL_LOCAL_GET, 1)],
    [(COL_LOCAL_VALUE_HI, 1); (COL_STACK_WRITE0_VALUE_HI, -1)], ([] : SparseRow)) Hrow.
  eapply active_copy_row; eassumption.
Qed.

Lemma full_local_set_high_value : forall (w : Witness),
  w COL_SEL_LOCAL_SET = 1 -> FullRelation.satFull w ->
  w COL_LOCAL_VALUE_HI = w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat).
Proof.
  intros w Hsel Hfull.
  circuit_row ([(COL_SEL_LOCAL_SET, 1)],
    [(COL_LOCAL_VALUE_HI, 1); (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat, -1)], ([] : SparseRow)) Hrow.
  eapply active_copy_row; eassumption.
Qed.

Lemma full_local_i64_value_wf : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  wf_val (i64Value (joinLimbs (w COL_LOCAL_VALUE) (w COL_LOCAL_VALUE_HI))).
Proof.
  intros w Hprime Hfull. apply i64_value_wf. apply join_limbs_bound.
  - exact (Range.full_local_value_u32 w Hprime Hfull).
  - circuit_range COL_LOCAL_VALUE_HI Hhi. unfold i32Modulus; lia.
Qed.

Theorem full_relation_implies_i64_local_get : forall (w : Witness) (z : state) (x : localidx),
  w COL_SEL_LOCAL_GET = 1 -> FullRelation.satFull w ->
  fun_local z x = i64Value (joinLimbs (w COL_LOCAL_VALUE) (w COL_LOCAL_VALUE_HI)) ->
  Step (mk_config z [admininstr_LOCAL_GET x])
    (mk_config z [admininstr_val
      (i64Value (joinLimbs (w COL_STACK_WRITE0_VALUE_LO) (w COL_STACK_WRITE0_VALUE_HI)))]).
Proof.
  intros w z x Hsel Hfull Hlocal.
  rewrite <- (full_local_get_value w Hsel Hfull), <- (full_local_get_high_value w Hsel Hfull), <- Hlocal.
  apply read. apply Step_read__local_get.
Qed.

Theorem full_relation_implies_i64_local_set : forall (w : Witness) (z : state) (x : localidx),
  w COL_SEL_LOCAL_SET = 1 -> FullRelation.satFull w ->
  Step (mk_config z
      [admininstr_val (i64Value (joinLimbs
        (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat))
        (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)))); admininstr_LOCAL_SET x])
    (mk_config (with_local z x (i64Value (joinLimbs (w COL_LOCAL_VALUE) (w COL_LOCAL_VALUE_HI)))) []).
Proof.
  intros w z x Hsel Hfull.
  rewrite <- (full_local_set_value w Hsel Hfull), <- (full_local_set_high_value w Hsel Hfull).
  apply Step__local_set.
Qed.
