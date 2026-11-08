(* SpecTec integer values built from canonical unsigned integers. *)
From Stdlib Require Import ZArith NArith Lia Bool.
Require Import Wasm2 WasmIntegers.
Open Scope Z_scope.

Definition i32Value (z : Z) : val :=
  val_CONST I32 (mk_num__0 Inn_I32 (mk_uN (Z.to_N z))).

Definition i64Value (z : Z) : val :=
  val_CONST I64 (mk_num__0 Inn_I64 (mk_uN (Z.to_N z))).

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

Lemma spectec_i64_well_formed : forall z : Z,
  0 <= z < i64Modulus -> wf_num_ I64 (mk_num__0 Inn_I64 (mk_uN (Z.to_N z))).
Proof.
  intros z [Hz0 Hz1].
  apply num__case_0.
  - reflexivity.
  - apply uN_case_0.
    cbn.
    change (((0 <=? Z.to_N z)%N && (Z.to_N z <=? 18446744073709551615)%N) = true).
    apply andb_true_iff.
    split.
    + apply N.leb_le. lia.
    + apply N.leb_le.
      apply N2Z.inj_le.
      rewrite Z2N.id by lia.
      unfold i64Modulus in Hz1.
      lia.
  - reflexivity.
Qed.

Lemma i32_value_wf : forall z, 0 <= z < i32Modulus -> wf_val (i32Value z).
Proof. intros. apply val_case_0. now apply spectec_i32_well_formed. Qed.

Lemma i64_value_wf : forall z, 0 <= z < i64Modulus -> wf_val (i64Value z).
Proof. intros. apply val_case_0. now apply spectec_i64_well_formed. Qed.
