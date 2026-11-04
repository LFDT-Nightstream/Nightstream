(* SpecTec's call_indirect on an i32 index: it reaches CALL_ADDR when the table
   entry is a function address of the expected type, and traps otherwise. *)
From Stdlib Require Import List ZArith NArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 WasmIntegers StraightLineSpec SpecTecCalls.
Import ListNotations.
Open Scope Z_scope.

(* The function an indirect call at `index` reaches. *)
Definition indirectTarget (z : state) (x y : idx) (index : Z) (a : N) : Prop :=
  nth_error (REFS (fun_table z x)) (Z.to_nat index) = Some (REF_FUNC_ADDR a) /\
  exists fi, nth_error (fun_funcinst z) (N.to_nat a) = Some fi /\ funcinst_TYPE fi = fun_type z y.

Lemma seq_size_length : forall (A : Type) (xs : list A), seq.size xs = length xs.
Proof. intros A xs; induction xs as [ | x xs IH]; [reflexivity | cbn; rewrite IH; reflexivity]. Qed.

Lemma lookup_in_bounds : forall (A : Type) (inhabited : Inhabited A) values index,
  (N.to_nat index < length values)%nat ->
  nth_error values (N.to_nat index) = Some (@lookup_total A inhabited values index).
Proof.
  intros A inhabited values index Hbound.
  destruct (nth_error values (N.to_nat index)) as [v | ] eqn:Hv.
  - erewrite spec_lookup_from_nth_error by exact Hv. reflexivity.
  - apply nth_error_None in Hv. lia.
Qed.

Lemma indirect_call_resolves : forall z x y index a,
  0 <= index < i32Modulus -> indirectTarget z x y index a ->
  Step (mk_config z [i32Constant index; admininstr_CALL_INDIRECT x y]) (mk_config z [CALL_ADDR a]).
Proof.
  intros z x y index a Hindex [Hentry [fi [Hfi Htype]]].
  assert (Hbound : (Z.to_nat index < length (REFS (fun_table z x)))%nat)
    by (apply nth_error_Some; rewrite Hentry; discriminate).
  assert (Hfunc : (N.to_nat a < length (fun_funcinst z))%nat)
    by (apply nth_error_Some; rewrite Hfi; discriminate).
  apply read. apply call_indirect_call.
  - change (is_true (N.ltb (Z.to_N index) (N.of_nat (seq.size (REFS (fun_table z x)))))).
    apply N.ltb_lt. rewrite seq_size_length. lia.
  - reflexivity.
  - apply (introT eqP). change (lookup_total (REFS (fun_table z x)) (Z.to_N index) = REF_FUNC_ADDR a).
    apply spec_lookup_from_nth_error. rewrite Z_N_nat. exact Hentry.
  - change (is_true (N.ltb a (N.of_nat (seq.size (fun_funcinst z))))).
    apply N.ltb_lt. rewrite seq_size_length. lia.
  - apply (introT eqP). erewrite spec_lookup_from_nth_error by exact Hfi. symmetry. exact Htype.
Qed.

Lemma indirect_call_traps : forall z x y index,
  0 <= index < i32Modulus -> (forall a, ~ indirectTarget z x y index a) ->
  Step (mk_config z [i32Constant index; admininstr_CALL_INDIRECT x y]) (mk_config z [admininstr_TRAP]).
Proof.
  intros z x y index Hindex Hnone.
  apply read. apply call_indirect_trap. intros H. inversion H as [z' i x' y' a Hbound _ Hentry Hfunc Htype]; subst.
  apply (Hnone a).
  change (is_true (eq_op (lookup_total (REFS (fun_table z x)) (Z.to_N index)) (REF_FUNC_ADDR a))) in Hentry.
  change (is_true (N.ltb (Z.to_N index) (N.of_nat (seq.size (REFS (fun_table z x)))))) in Hbound.
  apply N.ltb_lt in Hbound. rewrite seq_size_length in Hbound.
  change (is_true (N.ltb a (N.of_nat (seq.size (fun_funcinst z))))) in Hfunc.
  apply N.ltb_lt in Hfunc. rewrite seq_size_length in Hfunc.
  apply (elimT eqP) in Hentry. apply (elimT eqP) in Htype.
  split.
  - rewrite <- Hentry, <- Z_N_nat. apply lookup_in_bounds. lia.
  - eexists. split; [apply lookup_in_bounds; lia | symmetry; exact Htype].
Qed.
