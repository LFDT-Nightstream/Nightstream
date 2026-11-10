(* Operand stacks represented by typed cells. Row shapes and replay determine
   addresses and preserve both limbs of every value below the operands. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Wasm2 WasmIntegers WasmValues MemoryTrace MemoryCatalog
  IntegerMemory StraightLineSpec StackBindings.
Import ListNotations.
Open Scope Z_scope.

Definition valueStackMatches (memory : MemoryState) (boundary : Boundary) (values : list val) : Prop :=
  0 <= boundaryStackBase boundary /\
  boundarySP boundary = boundaryStackBase boundary + Z.of_nat (length values) /\
  forall index value, nth_error values index = Some value ->
    CellMatches memory (boundaryStackBase boundary + Z.of_nat index) value.

Lemma value_stack_matches_wf : forall memory boundary values,
  valueStackMatches memory boundary values -> Forall wf_val values.
Proof.
  intros memory boundary values [_ [_ Hcells]]. apply Forall_forall.
  intros value Hin. apply In_nth_error in Hin as [index Hindex].
  exact (cell_matches_wf _ _ _ (Hcells _ _ Hindex)).
Qed.

(* The scalar stack interface is a specialization for canonical i32 operands. *)
Lemma value_stack_matches_i32 : forall memory boundary values,
  Forall (fun value => 0 <= value < i32Modulus) values ->
  (valueStackMatches memory boundary (map i32Value values) <-> stackMatches memory boundary values).
Proof.
  intros memory boundary values Hbounds. unfold valueStackMatches, stackMatches. rewrite length_map.
  split; intros [Hbase [Hsp Hcells]]; split; [exact Hbase | | exact Hbase | ].
  - split; [exact Hsp | ]. split; [exact Hbounds | ]. intros index value Hindex.
    apply cell_matches_i32_read; [exact (proj1 (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hindex))) | ].
    apply Hcells. rewrite nth_error_map, Hindex. reflexivity.
  - destruct Hcells as [_ Hcells]. split; [exact Hsp | ]. intros index value Hindex.
    rewrite nth_error_map in Hindex. destruct (nth_error values index) as [raw | ] eqn:Hraw; [ | discriminate].
    injection Hindex as <-. constructor; [ | exact (Hcells _ _ Hraw)].
    exact (proj1 (Forall_forall _ _) Hbounds _ (nth_error_In _ _ Hraw)).
Qed.

Lemma value_stack_matches_binary : forall w initial final prefix a b,
  BinaryStackShape w -> Replay initial (eventsForRow memories w) final ->
  valueStackMatches initial (beforeBoundary w) (prefix ++ [i32Value a; i32Value b]) ->
  0 <= a -> 0 <= b -> 0 <= w COL_STACK_WRITE0_VALUE_LO < i32Modulus ->
  w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = a /\
  w (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat) = b /\
  valueStackMatches final (afterBoundary w) (prefix ++ [i32Value (w COL_STACK_WRITE0_VALUE_LO)]).
Proof.
  intros w initial final prefix a b Hshape Hreplay [Hbase [Hsp Hcells]] Ha Hb Hr.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  destruct (binary_stack_replay _ _ _ Hshape Hreplay) as [Hread0 [Hread1 Hfinal]].
  assert (HcellA : CellMatches initial (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix)) (i32Value a)).
  { apply Hcells. rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity. }
  assert (HcellB : CellMatches initial (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (S (length prefix))) (i32Value b)).
  { apply Hcells. rewrite nth_error_app2 by lia.
    replace (S (length prefix) - length prefix)%nat with 1%nat by lia. reflexivity. }
  apply (cell_matches_i32_read _ _ _ Ha) in HcellA.
  apply (cell_matches_i32_read _ _ _ Hb) in HcellB.
  rewrite (stack_read0_address _ Hshape) in Hread0. rewrite (stack_read1_address _ Hshape) in Hread1.
  replace (2 * (w COL_SP_BEFORE - 2)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix))) in Hread0 by lia.
  replace (2 * (w COL_SP_BEFORE - 1)) with
    (2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (S (length prefix)))) in Hread1 by (rewrite Nat2Z.inj_succ; lia).
  split; [congruence | ]. split; [congruence | ].
  assert (Hmemory : forall address, final stackId address =
      writeCell initial (w COL_SP_BEFORE - 2) (w COL_STACK_WRITE0_VALUE_LO) (w COL_STACK_WRITE0_VALUE_HI)
        stackId address).
  { intros address. rewrite Hfinal, (stack_write_address _ Hshape), (stack_write_hi_address _ Hshape). reflexivity. }
  unfold valueStackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (stack_base_after _ Hshape), (stack_pointer_after _ Hshape).
  split; [exact Hbase | ]. split; [rewrite length_app, Nat2Z.inj_add; cbn; lia | ].
  intros index value Hindex. eapply cell_matches_stack; [exact Hmemory | ].
  destruct (Nat.lt_ge_cases index (length prefix)) as [Hin | Hout].
  - apply write_cell_preserves; [lia | ]. apply Hcells.
    rewrite nth_error_app1 in Hindex |- * by exact Hin. exact Hindex.
  - rewrite nth_error_app2 in Hindex by exact Hout.
    destruct (index - length prefix)%nat as [ | n] eqn:Hix; [ | destruct n; discriminate].
    injection Hindex as <-. replace (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) with (w COL_SP_BEFORE - 2) by lia.
    constructor; [exact Hr | apply write_cell_low].
Qed.

(* A push writes at sp; the preceding cells retain both limbs. The pushed cell
   can come from a local read or any other row proof with this stack shape. *)
Lemma value_stack_matches_push : forall w initial final values value,
  UnaryStackShape w false true -> Replay initial (eventsForRow memories w) final ->
  valueStackMatches initial (beforeBoundary w) values ->
  CellMatches final (w COL_SP_BEFORE) value ->
  valueStackMatches final (afterBoundary w) (values ++ [value]).
Proof.
  intros w initial final values value Hshape Hreplay [Hbase [Hsp Hcells]] Hvalue.
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  destruct (local_get_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hfinal]].
  assert (Hmemory : forall address, final stackId address =
      writeCell initial (w COL_SP_BEFORE) (w COL_STACK_WRITE0_VALUE_LO) (w COL_STACK_WRITE0_VALUE_HI) stackId address).
  { intros address. rewrite Hfinal, (local_write_address _ _ _ Hshape Logic.eq_refl),
      (local_write_hi_address _ _ _ Hshape Logic.eq_refl). reflexivity. }
  unfold valueStackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after _ _ _ Hshape), (local_stack_pointer_after _ _ _ Hshape).
  split; [exact Hbase | ]. split; [rewrite length_app, Nat2Z.inj_add; cbn; lia | ].
  intros index v Hindex. destruct (Nat.lt_ge_cases index (length values)) as [Hin | Hout].
  - eapply cell_matches_stack; [exact Hmemory | ]. apply write_cell_preserves; [lia | ].
    apply Hcells. rewrite nth_error_app1 in Hindex by exact Hin. exact Hindex.
  - rewrite nth_error_app2 in Hindex by exact Hout.
    destruct (index - length values)%nat as [ | n] eqn:Hix; [ | destruct n; discriminate].
    injection Hindex as <-. replace (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) with (w COL_SP_BEFORE) by lia.
    exact Hvalue.
Qed.

(* Popping a value preserves the prefix. A local.set may also write below the
   operand base; the address separation protects both limbs of the prefix. *)
Lemma value_stack_matches_pop : forall w initial final prefix value local,
  UnaryStackShape w true local -> Replay initial (eventsForRow memories w) final ->
  (local = false -> w COL_FRAME_MOVE_DISTANCE_AFTER = 0) ->
  (local = true -> w COL_LOCAL_ADDR_LO < 2 * w COL_STACK_FRAME_BASE_BEFORE /\
    w COL_LOCAL_ADDR_HI < 2 * w COL_STACK_FRAME_BASE_BEFORE) ->
  valueStackMatches initial (beforeBoundary w) (prefix ++ [value]) ->
  valueStackMatches final (afterBoundary w) prefix.
Proof.
  intros w initial final prefix value local Hshape Hreplay Hslide Hbelow [Hbase [Hsp Hcells]].
  cbn [beforeBoundary boundarySP boundaryStackBase] in Hbase, Hsp, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  assert (Hmemory : forall index limb, (index < length prefix)%nat -> limb = 0 \/ limb = 1 ->
      final stackId [2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) + limb] =
      initial stackId [2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) + limb]).
  { destruct local.
    - destruct (local_set_stack_replay _ _ _ Hshape Hreplay) as [_ [_ Hfinal]].
      intros index limb Hindex Hlimb. rewrite Hfinal. destruct (Hbelow Logic.eq_refl) as [Hlo Hhi].
      rewrite !update_other; [reflexivity | | ]; right; intro Heq; injection Heq as Heq;
        match type of Heq with ?left = _ =>
          change (left = 2 * (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat index) + limb) in Heq end;
        destruct Hlimb; lia.
    - destruct (pop_stack_replay _ _ _ Hshape Hreplay) as [_ Hfinal]. intros. apply Hfinal. }
  unfold valueStackMatches. cbn [afterBoundary boundarySP boundaryStackBase].
  rewrite (local_stack_base_after _ _ _ Hshape), (local_stack_pointer_after _ _ _ Hshape).
  split; [exact Hbase | ]. split; [destruct local; [ | rewrite (Hslide Logic.eq_refl)]; cbn; lia | ].
  intros index v Hindex.
  assert (Hin : (index < length prefix)%nat) by (apply nth_error_Some; rewrite Hindex; discriminate).
  eapply cell_matches_memory; [intros limb Hl; apply Hmemory; assumption | ].
  apply Hcells. rewrite nth_error_app1 by exact Hin. exact Hindex.
Qed.
