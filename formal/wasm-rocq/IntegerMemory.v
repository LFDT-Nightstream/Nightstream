(* Integer cells in the unified frame memory. Cell addresses are in units of
   Wasm values; the catalog addresses their low/high limbs at 2*cell and 2*cell+1. *)
From Stdlib Require Import List ZArith Lia.
Require Import MemoryTrace MemoryCatalog Wasm2 WasmIntegers WasmValues.
Import ListNotations.
Open Scope Z_scope.

(* The high limb is not part of an i32 value. The type carried by the SpecTec
   value determines whether both limbs are required. *)
Inductive CellMatches (memory : MemoryState) (cell : Z) : val -> Prop :=
| cell_i32 : forall lo,
    0 <= lo < i32Modulus ->
    memory stackId [2 * cell] = Some lo ->
    CellMatches memory cell (i32Value lo)
| cell_i64 : forall lo hi,
    0 <= lo < i32Modulus -> 0 <= hi < i32Modulus ->
    memory stackId [2 * cell] = Some lo ->
    memory stackId [2 * cell + 1] = Some hi ->
    CellMatches memory cell (i64Value (joinLimbs lo hi)).

Definition writeCell (memory : MemoryState) (cell lo hi : Z) : MemoryState :=
  update (update memory stackId [2 * cell] lo) stackId [2 * cell + 1] hi.

Lemma write_cell_low : forall memory cell lo hi,
  writeCell memory cell lo hi stackId [2 * cell] = Some lo.
Proof.
  intros. unfold writeCell. rewrite update_other.
  - apply update_same.
  - right. intro H. injection H as H. lia.
Qed.

Lemma write_cell_high : forall memory cell lo hi,
  writeCell memory cell lo hi stackId [2 * cell + 1] = Some hi.
Proof. intros. apply update_same. Qed.

Lemma write_cell_other : forall memory cell lo hi target limb,
  cell <> target -> (limb = 0 \/ limb = 1) ->
  writeCell memory cell lo hi stackId [2 * target + limb] =
    memory stackId [2 * target + limb].
Proof.
  intros memory cell lo hi target limb Hcell Hlimb. unfold writeCell.
  assert (Hlo : [2 * target + limb] <> [2 * cell]).
  { intro H. injection H as H. change (2 * target + limb = 2 * cell) in H.
    destruct Hlimb; lia. }
  assert (Hhi : [2 * target + limb] <> [2 * cell + 1]).
  { intro H. injection H as H. change (2 * target + limb = 2 * cell + 1) in H.
    destruct Hlimb; lia. }
  rewrite update_other by (right; congruence).
  rewrite update_other by (right; congruence). reflexivity.
Qed.

Lemma cell_matches_wf : forall memory cell value,
  CellMatches memory cell value -> wf_val value.
Proof.
  intros memory cell value Hcell. destruct Hcell.
  - apply i32_value_wf. assumption.
  - apply i64_value_wf. apply join_limbs_bound; assumption.
Qed.

Lemma cell_matches_memory : forall first second cell value,
  (forall limb, limb = 0 \/ limb = 1 ->
    second stackId [2 * cell + limb] = first stackId [2 * cell + limb]) ->
  CellMatches first cell value -> CellMatches second cell value.
Proof.
  intros first second cell value Hmemory Hcell.
  assert (Hlo : second stackId [2 * cell] = first stackId [2 * cell]).
  { specialize (Hmemory 0 (or_introl eq_refl)). now rewrite Z.add_0_r in Hmemory. }
  destruct Hcell.
  - constructor; [assumption | now rewrite Hlo].
  - constructor; try assumption; [now rewrite Hlo | rewrite Hmemory; auto].
Qed.

Lemma cell_matches_stack : forall initial final cell value,
  (forall address, final stackId address = initial stackId address) ->
  CellMatches initial cell value -> CellMatches final cell value.
Proof. intros. eapply cell_matches_memory; [intros; apply H | exact H0]. Qed.

Lemma write_cell_preserves : forall memory target lo hi cell value,
  target <> cell -> CellMatches memory cell value ->
  CellMatches (writeCell memory target lo hi) cell value.
Proof.
  intros memory target lo hi cell value Hother Hcell.
  eapply cell_matches_memory; [ | exact Hcell]. intros. now apply write_cell_other.
Qed.

Lemma copy_cell : forall initial destination source target lo hi value,
  CellMatches initial source value ->
  initial stackId [2 * source] = Some lo ->
  initial stackId [2 * source + 1] = Some hi ->
  CellMatches (writeCell destination target lo hi) target value.
Proof.
  intros initial destination source target lo hi value Hcell Hlo Hhi.
  destruct Hcell as [v Hv Hread | v h Hv Hh Hread Hreadhi].
  - assert (v = lo) by congruence. subst v.
    constructor; [assumption | apply write_cell_low].
  - assert (v = lo) by congruence. assert (h = hi) by congruence. subst v h.
    constructor; try assumption; [apply write_cell_low | apply write_cell_high].
Qed.

Lemma cell_matches_i32_read : forall memory cell value,
  0 <= value -> CellMatches memory cell (i32Value value) ->
  memory stackId [2 * cell] = Some value.
Proof.
  intros memory cell value Hnonnegative Hcell.
  inversion Hcell; subst.
  match goal with H : Z.to_N _ = Z.to_N _ |- _ =>
    apply Z2N.inj in H; try lia; subst
  end. assumption.
Qed.
