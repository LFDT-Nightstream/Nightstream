(* The table/type immediate is one 32-bit ROM word. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts.
Import ListNotations.
Open Scope Z_scope.

(* An out-of-range source descriptor has no valid memory word. Using -1
   makes the dense ROM reject it instead of truncating either source id. *)
Definition tableDescriptor (table expected : Z) : Z :=
  if (0 <=? table) && (table <? 256) && (0 <=? expected) && (expected <? 16777216)
  then table + 256 * expected else -1.

Lemma full_table_descriptor : forall w,
  fieldPrime -> FullRelation.satFull w ->
  w COL_TABLE_DESCRIPTOR = w COL_TABLE_ID + 256 * w COL_EXPECTED_TYPE_ID /\
  0 <= w COL_TABLE_ID < 256 /\ 0 <= w COL_EXPECTED_TYPE_ID < 16777216 /\
  0 <= w COL_TABLE_DESCRIPTOR < 4294967296.
Proof.
  intros w Hprime Hfull. pose proof (proj1 Hfull) as Hone.
  pose proof (value_bound w COL_TABLE_DESCRIPTOR) as Hpacked.
  circuit_range COL_TABLE_ID Htable. circuit_range COL_EXPECTED_TYPE_ID Htype.
  assert (Hpack : w COL_TABLE_DESCRIPTOR = w COL_TABLE_ID + 256 * w COL_EXPECTED_TYPE_ID).
  { circuit_row ([(COL_TABLE_DESCRIPTOR, 1); (COL_TABLE_ID, -1); (COL_EXPECTED_TYPE_ID, -256)],
      [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow. }
  repeat split; lia.
Qed.

Lemma table_descriptor_fields : forall w table expected,
  fieldPrime -> FullRelation.satFull w ->
  w COL_TABLE_DESCRIPTOR = tableDescriptor table expected ->
  w COL_TABLE_ID = table /\ w COL_EXPECTED_TYPE_ID = expected.
Proof.
  intros w table expected Hprime Hfull Hword.
  destruct (full_table_descriptor w Hprime Hfull) as [Hpack [Htable [Htype Hpacked]]].
  unfold tableDescriptor in Hword.
  destruct ((0 <=? table) && (table <? 256) && (0 <=? expected) && (expected <? 16777216)) eqn:Hvalid;
    [ | lia].
  repeat rewrite Bool.andb_true_iff in Hvalid.
  destruct Hvalid as [[[Ht0 Ht1] He0] He1].
  apply Z.leb_le in Ht0, He0. apply Z.ltb_lt in Ht1, He1.
  lia.
Qed.
