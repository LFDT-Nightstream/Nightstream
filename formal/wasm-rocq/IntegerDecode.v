(* Decoding integer cells, with the type supplied by the source program. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import Wasm2 WasmIntegers WasmValues MemoryTrace MemoryCatalog IntegerMemory.
Import ListNotations.
Open Scope Z_scope.

Definition i32Cell (cell : option Z) : option Z :=
  match cell with
  | Some value => if (0 <=? value) && (value <? i32Modulus) then Some value else None
  | None => None
  end.

Fixpoint readCells (cell : Z -> option Z) (first : Z) (count : nat) : option (list Z) :=
  match count with
  | O => Some []
  | S count =>
      match i32Cell (cell first), readCells cell (first + 1) count with
      | Some value, Some values => Some (value :: values)
      | _, _ => None
      end
  end.

Definition integerValueType (value : val) : option valtype :=
  match value with
  | val_CONST I32 _ => Some valtype_I32
  | val_CONST I64 _ => Some valtype_I64
  | _ => None
  end.

(* Declared integer types, together with SpecTec's value well-formedness. *)
Definition valuesTyped (types : list valtype) (values : list val) : Prop :=
  Forall2 (fun t value => integerValueType value = Some t /\ wf_val value) types values.

Lemma values_typed_wf : forall types values, valuesTyped types values -> Forall wf_val values.
Proof. intros types values H; induction H; constructor; tauto. Qed.

Lemma values_typed_types : forall types values, valuesTyped types values ->
  Forall2 (fun t value => integerValueType value = Some t) types values.
Proof. intros types values H; induction H; constructor; tauto. Qed.

Lemma values_typed_intro : forall types values,
  Forall2 (fun t value => integerValueType value = Some t) types values ->
  Forall wf_val values -> valuesTyped types values.
Proof.
  intros types values H; induction H; intros Hwf; [constructor | ].
  inversion Hwf; subst. constructor; [split; assumption | apply IHForall2; assumption].
Qed.

Lemma values_typed_repeat : forall t values,
  Forall (fun value => integerValueType value = Some t /\ wf_val value) values ->
  valuesTyped (repeat t (length values)) values.
Proof. intros t values H; induction H; cbn; constructor; assumption. Qed.

Lemma values_typed_i32 : forall values,
  Forall (fun value => 0 <= value < i32Modulus) values ->
  valuesTyped (repeat valtype_I32 (length values)) (map i32Value values).
Proof.
  intros values H; induction H; cbn; constructor; [split; [reflexivity | apply i32_value_wf; assumption] | assumption].
Qed.

(* Compatibility predicate for the current i32-only program fragment. The
   frame representation keeps values typed even while this restriction holds. *)
Definition i32Operand (value : val) : Prop :=
  exists raw, 0 <= raw < i32Modulus /\ value = i32Value raw.

Lemma i32_operand_wf : forall value, i32Operand value -> wf_val value.
Proof. intros value [raw [Hr ->]]. now apply i32_value_wf. Qed.

Lemma i32_operands_wf : forall values, Forall i32Operand values -> Forall wf_val values.
Proof. intros values H. eapply Forall_impl; [ | exact H]. exact i32_operand_wf. Qed.

Lemma i32_operands_map : forall raw,
  Forall (fun value => 0 <= value < i32Modulus) raw -> Forall i32Operand (map i32Value raw).
Proof.
  intros raw H. apply Forall_map. eapply Forall_impl; [ | exact H].
  intros value Hv. exists value. auto.
Qed.

Lemma i32_operands_inv : forall values,
  Forall i32Operand values -> exists raw, values = map i32Value raw /\
    Forall (fun value => 0 <= value < i32Modulus) raw.
Proof.
  intros values H. induction H as [ | value values [raw [Hr ->]] Hvalues [rest [-> Hrest]]].
  - exists []. split; constructor.
  - exists (raw :: rest). split; [reflexivity | constructor; assumption].
Qed.

Definition readIntegerCell (memory : MemoryState) (cell : Z) (t : valtype) : option val :=
  match t, i32Cell (memory stackId [2 * cell]) with
  | valtype_I32, Some lo => Some (i32Value lo)
  | valtype_I64, Some lo =>
      match i32Cell (memory stackId [2 * cell + 1]) with
      | Some hi => Some (i64Value (joinLimbs lo hi))
      | None => None
      end
  | _, _ => None
  end.

(* Default values for the integer local types supported by the cell reader. *)
Definition integerZero (t : valtype) : option val :=
  match t with
  | valtype_I32 => Some (i32Value 0)
  | valtype_I64 => Some (i64Value 0)
  | _ => None
  end.

Fixpoint integerZeros (types : list valtype) : option (list val) :=
  match types with
  | [] => Some []
  | t :: rest =>
      match integerZero t, integerZeros rest with
      | Some value, Some values => Some (value :: values)
      | _, _ => None
      end
  end.

Lemma integer_zeros_i32 : forall count,
  integerZeros (repeat valtype_I32 count) = Some (repeat (i32Value 0) count).
Proof. induction count; cbn; [reflexivity | now rewrite IHcount]. Qed.

Lemma integer_zero_cell : forall memory cell t value,
  integerZero t = Some value ->
  memory stackId [2 * cell] = Some 0 -> memory stackId [2 * cell + 1] = Some 0 ->
  CellMatches memory cell value /\ integerValueType value = Some t.
Proof.
  intros memory cell t value Hzero Hlo Hhi.
  destruct t; try discriminate; injection Hzero as <-; split;
    try reflexivity.
  - constructor; [unfold i32Modulus; lia | exact Hlo].
  - change (CellMatches memory cell (i64Value (joinLimbs 0 0))).
    constructor; try assumption; unfold i32Modulus; lia.
Qed.

Lemma integer_zeros_types : forall types values,
  integerZeros types = Some values ->
  Forall2 (fun t value => integerValueType value = Some t) types values /\
  Forall2 (fun t value => integerZero t = Some value) types values.
Proof.
  induction types as [ | t rest IH]; intros values H; cbn in H.
  - injection H as <-. split; constructor.
  - destruct (integerZero t) as [value | ] eqn:Ht; [ | discriminate].
    destruct (integerZeros rest) as [vs | ] eqn:Hs; [ | discriminate]. injection H as <-.
    destruct (IH _ eq_refl) as [Htypes Hzeros].
    assert (Htype : integerValueType value = Some t).
    { destruct t; try discriminate; injection Ht as <-; reflexivity. }
    split; constructor; assumption.
Qed.

(* The address function also covers operands whose frame move is still pending. *)
Fixpoint readIntegerCells (memory : MemoryState) (address : nat -> Z) (types : list valtype)
    : option (list val) :=
  match types with
  | [] => Some []
  | t :: rest =>
      match readIntegerCell memory (address 0%nat) t,
        readIntegerCells memory (fun i => address (S i)) rest with
      | Some value, Some values => Some (value :: values)
      | _, _ => None
      end
  end.

(* The top `remaining` operands still sit `distance` cells above their destinations. *)
Definition operandCell (base : Z) (count remaining : nat) (distance : Z) (index : nat) : Z :=
  base + Z.of_nat index + (if (index <? count - remaining)%nat then 0 else distance).

Definition readIntegerOperands (memory : MemoryState) (base : Z) (types : list valtype)
    (remaining : nat) (distance : Z) : option (list val) :=
  readIntegerCells memory (operandCell base (length types) remaining distance) types.

Lemma i32_cell_some : forall cell value,
  i32Cell cell = Some value <-> cell = Some value /\ 0 <= value < i32Modulus.
Proof.
  intros [v | ] value; cbn; [ | split; [discriminate | intros [H _]; discriminate]].
  destruct ((0 <=? v) && (v <? i32Modulus)) eqn:Hb.
  - apply andb_true_iff in Hb as [H0 H1]. apply Z.leb_le in H0. apply Z.ltb_lt in H1.
    split; [intros H; injection H as <-; auto | intros [H _]; injection H as <-; reflexivity].
  - split; [discriminate | ]. intros [Hv Hr]. injection Hv as <-.
    apply andb_false_iff in Hb as [Hb | Hb]; [apply Z.leb_gt in Hb | apply Z.ltb_ge in Hb]; lia.
Qed.

Lemma read_integer_cell_sound : forall memory cell t value,
  readIntegerCell memory cell t = Some value ->
  CellMatches memory cell value /\ integerValueType value = Some t.
Proof.
  intros memory cell t value H. unfold readIntegerCell in H.
  destruct t; try discriminate; try destruct n; try discriminate;
    destruct (i32Cell (memory stackId [2 * cell])) as [lo | ] eqn:Hlo; try discriminate.
  - injection H as <-. apply i32_cell_some in Hlo as [Hlo Hrange].
    split; [constructor; assumption | reflexivity].
  - destruct (i32Cell (memory stackId [2 * cell + 1])) as [hi | ] eqn:Hhi; try discriminate.
    injection H as <-. apply i32_cell_some in Hlo as [Hlo Hrlo].
    apply i32_cell_some in Hhi as [Hhi Hrhi].
    split; [constructor; assumption | reflexivity].
Qed.

Lemma read_integer_cell_complete : forall memory cell value,
  CellMatches memory cell value -> forall t, integerValueType value = Some t ->
  readIntegerCell memory cell t = Some value.
Proof.
  intros memory cell value Hcell t Htype.
  destruct Hcell as [lo Hlo Hrlo | lo hi Hlo Hhi Hrlo Hrhi];
    cbn in Htype; injection Htype as <-; cbn [readIntegerCell];
    rewrite (proj2 (i32_cell_some _ _) (conj Hrlo Hlo)); try reflexivity.
  rewrite (proj2 (i32_cell_some _ _) (conj Hrhi Hhi)). reflexivity.
Qed.

Lemma read_integer_cell_some : forall memory cell t value,
  readIntegerCell memory cell t = Some value <->
  CellMatches memory cell value /\ integerValueType value = Some t.
Proof.
  split; [apply read_integer_cell_sound | intros [Hcell Htype]; eapply read_integer_cell_complete; eassumption].
Qed.

Lemma read_integer_cells_some : forall memory types address values,
  readIntegerCells memory address types = Some values <->
  Forall2 (fun t value => integerValueType value = Some t) types values /\
  forall index value, nth_error values index = Some value -> CellMatches memory (address index) value.
Proof.
  intros memory types. induction types as [ | t types IH]; intros address values; cbn [readIntegerCells].
  - split.
    + intros H. injection H as <-. split; [constructor | intros [ | ] ? H; discriminate].
    + intros [H _]. inversion H. reflexivity.
  - split.
    + destruct (readIntegerCell memory (address 0%nat) t) as [v | ] eqn:Hcell; [ | discriminate].
      destruct (readIntegerCells memory (fun i => address (S i)) types) as [vs | ] eqn:Hrest; [ | discriminate].
      intros H. injection H as <-.
      apply read_integer_cell_some in Hcell as [Hcell Htype].
      apply IH in Hrest as [Htypes Hcells].
      split; [constructor; assumption | ]. intros [ | index] value Hlookup; cbn in Hlookup.
      * injection Hlookup as <-. exact Hcell.
      * exact (Hcells _ _ Hlookup).
    + intros [Htypes Hcells]. inversion Htypes as [ | ? v ? vs Htype Hrest]; subst.
      rewrite (proj2 (read_integer_cell_some _ _ _ _) (conj (Hcells 0%nat _ eq_refl) Htype)).
      rewrite (proj2 (IH _ vs)); [reflexivity | ]. split; [assumption | ].
      intros index value Hlookup. exact (Hcells (S index) value Hlookup).
Qed.

Lemma integer_zeros_cells : forall types values memory address,
  integerZeros types = Some values ->
  (forall index, (index < length types)%nat ->
    memory stackId [2 * address index] = Some 0 /\
    memory stackId [2 * address index + 1] = Some 0) ->
  readIntegerCells memory address types = Some values.
Proof.
  induction types as [ | t rest IH]; intros values memory address Hzeros Hmemory; cbn in Hzeros.
  - injection Hzeros as <-. reflexivity.
  - destruct (integerZero t) as [value | ] eqn:Ht; [ | discriminate].
    destruct (integerZeros rest) as [vs | ] eqn:Hs; [ | discriminate]. injection Hzeros as <-.
    destruct (Hmemory 0%nat ltac:(cbn; lia)) as [Hlo Hhi].
    cbn [readIntegerCells].
    rewrite (proj2 (read_integer_cell_some _ _ _ _) (integer_zero_cell _ _ _ _ Ht Hlo Hhi)).
    rewrite (IH vs memory (fun i => address (S i)) eq_refl); [reflexivity | ].
    intros index Hindex. apply Hmemory. cbn. lia.
Qed.

Lemma read_integer_cells_app : forall memory first second address,
  readIntegerCells memory address (first ++ second) =
    match readIntegerCells memory address first,
      readIntegerCells memory (fun i => address (length first + i)%nat) second with
    | Some prefix, Some suffix => Some (prefix ++ suffix)
    | _, _ => None
    end.
Proof.
  intros memory first. induction first as [ | t rest IH]; intros second address.
  - cbn. destruct (readIntegerCells memory address second); reflexivity.
  - cbn [app readIntegerCells length]. rewrite IH. cbn [Nat.add].
    destruct (readIntegerCell memory (address 0%nat) t);
      destruct (readIntegerCells memory (fun i => address (S i)) rest);
      destruct (readIntegerCells memory (fun i => address (S (length rest + i))) second); reflexivity.
Qed.

Lemma read_integer_cells_wf : forall memory address types values,
  readIntegerCells memory address types = Some values -> Forall wf_val values.
Proof.
  intros memory address types values H. apply read_integer_cells_some in H as [_ Hcells].
  apply Forall_forall. intros value Hin. apply In_nth_error in Hin as [index Hindex].
  eapply cell_matches_wf. exact (Hcells _ _ Hindex).
Qed.

Lemma read_integer_cells_addresses : forall memory types first second,
  (forall index, (index < length types)%nat -> first index = second index) ->
  readIntegerCells memory first types = readIntegerCells memory second types.
Proof.
  intros memory types. induction types as [ | t types IH]; intros first second H; [reflexivity | ].
  cbn [readIntegerCells]. rewrite (H 0%nat ltac:(cbn; lia)).
  rewrite (IH (fun i => first (S i)) (fun i => second (S i))); [reflexivity | ].
  intros index Hindex. apply H. cbn. lia.
Qed.

(* On i32 cells the typed reader agrees with the existing scalar reader. *)
Lemma read_integer_cells_i32 : forall memory count first,
  readIntegerCells memory (fun i => first + Z.of_nat i) (repeat valtype_I32 count) =
    option_map (map i32Value) (readCells (fun cell => memory stackId [2 * cell]) first count).
Proof.
  intros memory count. induction count as [ | count IH]; intros first; [reflexivity | ].
  cbn [repeat readIntegerCells readIntegerCell readCells]. rewrite Z.add_0_r.
  rewrite (read_integer_cells_addresses memory _ (fun i => first + Z.of_nat (S i))
    (fun i => first + 1 + Z.of_nat i)) by (intros; lia).
  rewrite IH. destruct (i32Cell (memory stackId [2 * first]));
    destruct (readCells (fun cell => memory stackId [2 * cell]) (first + 1) count); reflexivity.
Qed.

Lemma read_integer_cells_i32_from : forall memory values address,
  Forall (fun value => 0 <= value < i32Modulus) values ->
  (forall index value, nth_error values index = Some value ->
    memory stackId [2 * address index] = Some value) ->
  readIntegerCells memory address (repeat valtype_I32 (length values)) = Some (map i32Value values).
Proof.
  intros memory values. induction values as [ | value rest IH]; intros address Hbounds Hcells; [reflexivity | ].
  inversion Hbounds as [ | ? ? Hvalue Hrest]; subst.
  cbn [length repeat readIntegerCells readIntegerCell map].
  rewrite (proj2 (i32_cell_some _ _) (conj (Hcells 0%nat _ eq_refl) Hvalue)).
  rewrite IH; [reflexivity | exact Hrest | ]. intros index v Hv. exact (Hcells (S index) v Hv).
Qed.

Lemma read_integer_cells_i32_inv : forall memory count address values,
  readIntegerCells memory address (repeat valtype_I32 count) = Some values ->
  exists raw, values = map i32Value raw /\ length raw = count /\
    Forall (fun value => 0 <= value < i32Modulus) raw /\
    forall index value, nth_error raw index = Some value -> memory stackId [2 * address index] = Some value.
Proof.
  intros memory count. induction count as [ | count IH]; intros address values H.
  - cbn in H. injection H as <-. exists []. repeat split; try constructor. intros [ | ] ? H; discriminate.
  - cbn [repeat readIntegerCells readIntegerCell] in H.
    destruct (i32Cell (memory stackId [2 * address 0%nat])) as [lo | ] eqn:Hlo; [ | discriminate].
    destruct (readIntegerCells memory (fun i => address (S i)) (repeat valtype_I32 count)) as [rest | ] eqn:Hrest;
      [ | discriminate]. injection H as <-.
    destruct (IH _ _ Hrest) as [raw [-> [Hlen [Hbounds Hcells]]]].
    apply i32_cell_some in Hlo as [Hlo Hbound].
    exists (lo :: raw). split; [reflexivity | ]. split; [cbn; lia | ]. split; [constructor; assumption | ].
    intros [ | index] value Hv; cbn in Hv.
    + injection Hv as <-. exact Hlo.
    + exact (Hcells _ _ Hv).
Qed.

(* A bottom-up move preserves the decoded values, including when source and
   destination regions overlap. Both limbs of the next cell are copied.
   For a circuit move, both source-read premises must follow from its memory
   events and Replay, even for i32. CellMatches for i32 alone does not require
   a defined high limb; it must not be used to supply that premise. *)
Lemma read_integer_operands_copy : forall memory base types remaining distance values lo hi,
  (0 < remaining <= length types)%nat -> 0 <= distance ->
  readIntegerOperands memory base types remaining distance = Some values ->
  memory stackId [2 * (base + Z.of_nat (length types - remaining) + distance)] = Some lo ->
  memory stackId [2 * (base + Z.of_nat (length types - remaining) + distance) + 1] = Some hi ->
  readIntegerOperands (writeCell memory (base + Z.of_nat (length types - remaining)) lo hi)
    base types (Nat.pred remaining) distance = Some values.
Proof.
  intros memory base types remaining distance values lo hi Hremaining Hdistance Hdecode Hlo Hhi.
  unfold readIntegerOperands in *.
  apply read_integer_cells_some in Hdecode as [Htypes Hcells].
  apply read_integer_cells_some. split; [exact Htypes | ].
  intros index value Hindex.
  pose proof (Hcells _ _ Hindex) as Hcell.
  assert (Hnext : (length types - Nat.pred remaining = S (length types - remaining))%nat) by lia.
  destruct (Nat.eq_dec index (length types - remaining)) as [-> | Hother].
  - unfold operandCell in Hcell |- *. rewrite Nat.ltb_irrefl in Hcell.
    rewrite Hnext, (proj2 (Nat.ltb_lt _ _) (Nat.lt_succ_diag_r _)), Z.add_0_r.
    eapply copy_cell; eassumption.
  - assert (Hsame : operandCell base (length types) (Nat.pred remaining) distance index =
        operandCell base (length types) remaining distance index).
    { unfold operandCell. rewrite Hnext.
      destruct (index <? length types - remaining)%nat eqn:Hbefore;
        destruct (index <? S (length types - remaining))%nat eqn:Hafter; try reflexivity;
        apply Nat.ltb_lt in Hbefore || apply Nat.ltb_ge in Hbefore;
        apply Nat.ltb_lt in Hafter || apply Nat.ltb_ge in Hafter; lia. }
    rewrite Hsame. apply write_cell_preserves; [ | exact Hcell].
    unfold operandCell. destruct (index <? length types - remaining)%nat eqn:Hbefore;
      [apply Nat.ltb_lt in Hbefore | apply Nat.ltb_ge in Hbefore]; lia.
Qed.

Lemma read_cells_some : forall cell count first values,
  readCells cell first count = Some values <->
  length values = count /\ Forall (fun value => 0 <= value < i32Modulus) values /\
  forall index value, nth_error values index = Some value -> cell (first + Z.of_nat index) = Some value.
Proof.
  intros cell count. induction count as [ | count IH]; intros first values; cbn [readCells].
  - split.
    + intros H. injection H as <-. split; [reflexivity | split; [constructor | intros [ | ] ? H; discriminate]].
    + intros [H _]. destruct values; [reflexivity | discriminate].
  - split.
    + destruct (i32Cell (cell first)) as [value | ] eqn:Hcell; [ | discriminate].
      destruct (readCells cell (first + 1) count) as [rest | ] eqn:Hrest; [ | discriminate].
      intros H. injection H as <-. apply i32_cell_some in Hcell as [Hcell Hrange].
      destruct (proj1 (IH _ _) Hrest) as [Hlen [Hbounds Hcells]].
      split; [cbn; lia | split; [constructor; assumption | ]].
      intros [ | index] v Hv; cbn in Hv.
      * injection Hv as <-. rewrite Z.add_0_r. exact Hcell.
      * replace (first + Z.of_nat (S index)) with (first + 1 + Z.of_nat index) by lia. apply Hcells. exact Hv.
    + intros [Hlen [Hbounds Hcells]]. destruct values as [ | value rest]; [discriminate | ].
      inversion Hbounds as [ | ? ? Hrange Hrest]; subst.
      assert (Hfirst : cell first = Some value)
        by (rewrite <- (Z.add_0_r first); exact (Hcells 0%nat value eq_refl)).
      rewrite (proj2 (i32_cell_some _ _) (conj Hfirst Hrange)).
      rewrite (proj2 (IH (first + 1) rest)); [reflexivity | ].
      split; [cbn in Hlen; lia | split; [exact Hrest | ]].
      intros index v Hv. replace (first + 1 + Z.of_nat index) with (first + Z.of_nat (S index)) by lia.
      apply Hcells. exact Hv.
Qed.

Lemma read_cells_total : forall cell count first,
  (forall index, (index < count)%nat -> exists value,
    cell (first + Z.of_nat index) = Some value /\ 0 <= value < i32Modulus) ->
  exists values, readCells cell first count = Some values.
Proof.
  intros cell count. induction count as [ | count IH]; intros first Hcells; [exists []; reflexivity | ].
  destruct (Hcells 0%nat ltac:(lia)) as [value [Hvalue Hrange]]. rewrite Z.add_0_r in Hvalue.
  destruct (IH (first + 1)) as [rest Hrest].
  { intros index Hindex. destruct (Hcells (S index) ltac:(lia)) as [v Hv].
    exists v. replace (first + 1 + Z.of_nat index) with (first + Z.of_nat (S index)) by lia. exact Hv. }
  exists (value :: rest). cbn [readCells].
  rewrite (proj2 (i32_cell_some _ _) (conj Hvalue Hrange)), Hrest. reflexivity.
Qed.

(* The reader is unchanged if both limbs of every cell it reads are unchanged. *)
Lemma read_integer_cells_preserved : forall initial final types address values,
  readIntegerCells initial address types = Some values ->
  (forall index limb, (index < length types)%nat -> (limb = 0 \/ limb = 1) ->
    final stackId [2 * address index + limb] = initial stackId [2 * address index + limb]) ->
  readIntegerCells final address types = Some values.
Proof.
  intros initial final types address values Hread Hmemory.
  apply read_integer_cells_some in Hread as [Htypes Hcells].
  apply read_integer_cells_some. split; [exact Htypes | ].
  intros index value Hv. eapply cell_matches_memory; [ | exact (Hcells _ _ Hv)].
  intros limb Hlimb. apply Hmemory; [ | exact Hlimb].
  rewrite (Forall2_length Htypes). apply nth_error_Some. rewrite Hv. discriminate.
Qed.

(* Function-entry defaults agree with SpecTec, including mixed integer locals. *)
Lemma integer_zeros_spectec : forall types values,
  integerZeros types = Some values ->
  values = map (fun t => the (default_ t)) types /\
  Forall (fun t => default_ t <> None) types /\ Forall wf_val values.
Proof.
  induction types as [ | t types IH]; intros values H; cbn in H.
  - injection H as <-. repeat split; constructor.
  - destruct (integerZero t) as [value | ] eqn:Ht; [ | discriminate].
    destruct (integerZeros types) as [rest | ] eqn:Hs; [ | discriminate]. injection H as <-.
    destruct (IH _ eq_refl) as [Hvalues [Hdefault Hwf]].
    destruct t; try discriminate; injection Ht as <-; cbn [map].
    all: split; [now rewrite Hvalues | split; constructor; try assumption; try discriminate].
    + apply i32_value_wf. unfold i32Modulus; lia.
    + apply i64_value_wf. unfold i64Modulus; lia.
Qed.
