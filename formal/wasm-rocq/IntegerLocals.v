(* Typed local transfers through the exported stack-memory ports. The cell
   relation handles i32 and i64; no single field element stores an i64 value. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  LocalBindings StackBindings IntegerMemory IntegerStack Locals Wasm2.
Import ListNotations.
Open Scope Z_scope.

Lemma full_stack_read0_high_address : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat) =
    w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) + 1.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) Hlo.
  circuit_range (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat) Hhi.
  circuit_row ([(List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat, 1);
    (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, -1); (COL_ONE, -1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. rewrite Hone in Hrow. normalize_columns. field_linear Hrow.
Qed.

(* The local must be below the operand stack. In the current catalog the push
   precedes the local read, so this separation is needed to recover the old cell. *)
Theorem local_get_cell : forall w initial final value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_GET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX < w COL_SP_BEFORE ->
  CellMatches initial (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) value ->
  CellMatches final (w COL_SP_BEFORE) value.
Proof.
  intros w initial final value Hprime Hfull Hget Hreplay Hbelow Hcell.
  pose proof (value_bound w COL_SP_BEFORE) as Hsp.
  pose proof (full_local_get_stack_shape w Hprime Hfull Hget ltac:(lia)) as Hshape.
  destruct (full_local_addresses w Hprime Hfull) as [Hlocal HlocalHi].
  pose proof (local_write_address _ _ _ Hshape eq_refl) as Hwrite.
  pose proof (local_write_hi_address _ _ _ Hshape eq_refl) as HwriteHi.
  destruct (local_get_stack_replay w initial final Hshape Hreplay) as [Hlo [Hhi Hfinal]].
  cbn zeta in Hlo, Hhi, Hfinal.
  rewrite HlocalHi in Hhi.
  rewrite Hlocal, Hwrite, HwriteHi in Hlo, Hhi.
  rewrite Hwrite, HwriteHi in Hfinal.
  change (writeCell initial (w COL_SP_BEFORE) (w COL_STACK_WRITE0_VALUE_LO)
    (w COL_STACK_WRITE0_VALUE_HI) stackId
    [2 * (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX)] = Some (w COL_LOCAL_VALUE)) in Hlo.
  change (writeCell initial (w COL_SP_BEFORE) (w COL_STACK_WRITE0_VALUE_LO)
    (w COL_STACK_WRITE0_VALUE_HI) stackId
    [2 * (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) + 1] = Some (w COL_LOCAL_VALUE_HI)) in Hhi.
  replace (2 * (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX)) with
    (2 * (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) + 0) in Hlo by ring.
  rewrite write_cell_other in Hlo, Hhi by (auto; lia).
  pose proof (Locals.full_local_get_value w Hget Hfull) as Hcopy.
  pose proof (Locals.full_local_get_high_value w Hget Hfull) as HcopyHi.
  rewrite Hcopy in Hlo. rewrite HcopyHi in Hhi. rewrite Z.add_0_r in Hlo.
  eapply cell_matches_memory.
  - intros limb _. apply Hfinal.
  - eapply copy_cell; eassumption.
Qed.

Theorem local_set_cell : forall w initial final value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_SET = 1 ->
  Replay initial (eventsForRow memories w) final -> 1 <= w COL_SP_BEFORE ->
  CellMatches initial (w COL_SP_BEFORE - 1) value ->
  CellMatches final (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) value.
Proof.
  intros w initial final value Hprime Hfull Hset Hreplay Hsp Hcell.
  pose proof (full_local_set_stack_shape w Hprime Hfull Hset Hsp) as Hshape.
  destruct (full_local_addresses w Hprime Hfull) as [Hlocal HlocalHi].
  pose proof (local_read_address _ _ _ Hshape eq_refl) as Hread.
  pose proof (full_stack_read0_high_address w Hprime Hfull) as HreadHi.
  destruct (local_set_stack_replay w initial final Hshape Hreplay) as [Hlo [Hhi Hfinal]].
  rewrite HreadHi, Hread in Hhi. rewrite Hread in Hlo.
  setoid_rewrite HlocalHi in Hfinal. setoid_rewrite Hlocal in Hfinal.
  pose proof (Locals.full_local_set_value w Hset Hfull) as Hcopy.
  pose proof (Locals.full_local_set_high_value w Hset Hfull) as HcopyHi.
  rewrite <- Hcopy in Hlo. rewrite <- HcopyHi in Hhi.
  eapply cell_matches_memory.
  - intros limb _. apply Hfinal.
  - eapply copy_cell; eassumption.
Qed.

(* Local transfers on a complete typed operand stack, including an untouched
   mixed-width prefix. The local's cell lies below the operand base. *)
Theorem local_get_value_stack : forall w initial final values value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_GET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX < w COL_STACK_FRAME_BASE_BEFORE ->
  valueStackMatches initial (StraightLineSpec.beforeBoundary w) values ->
  CellMatches initial (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) value ->
  valueStackMatches final (StraightLineSpec.afterBoundary w) (values ++ [value]).
Proof.
  intros w initial final values value Hprime Hfull Hget Hreplay Hbelow Hstack Hcell.
  pose proof Hstack as [Hbase [Hsp _]].
  cbn [StraightLineSpec.beforeBoundary StraightLineSpec.boundaryStackBase StraightLineSpec.boundarySP] in Hbase, Hsp.
  eapply value_stack_matches_push; [ | exact Hreplay | exact Hstack | ].
  - exact (full_local_get_stack_shape w Hprime Hfull Hget ltac:(lia)).
  - eapply local_get_cell; eauto; lia.
Qed.

Theorem local_set_value_stack : forall w initial final prefix value,
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_LOCAL_SET = 1 ->
  Replay initial (eventsForRow memories w) final ->
  w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX < w COL_STACK_FRAME_BASE_BEFORE ->
  valueStackMatches initial (StraightLineSpec.beforeBoundary w) (prefix ++ [value]) ->
  valueStackMatches final (StraightLineSpec.afterBoundary w) prefix /\
  CellMatches final (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) value.
Proof.
  intros w initial final prefix value Hprime Hfull Hset Hreplay Hbelow Hstack.
  pose proof Hstack as [Hbase [Hsp Hcells]].
  cbn [StraightLineSpec.beforeBoundary StraightLineSpec.boundaryStackBase StraightLineSpec.boundarySP] in Hbase, Hsp, Hcells.
  rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp.
  pose proof (full_local_set_stack_shape w Hprime Hfull Hset ltac:(lia)) as Hshape.
  split.
  - eapply value_stack_matches_pop; [exact Hshape | exact Hreplay | discriminate | | exact Hstack].
    intros _. destruct (full_local_addresses w Hprime Hfull) as [Hlo Hhi].
    rewrite Hhi, Hlo. split; lia.
  - eapply local_set_cell; [exact Hprime | exact Hfull | exact Hset | exact Hreplay | lia | ].
    replace (w COL_SP_BEFORE - 1) with (w COL_STACK_FRAME_BASE_BEFORE + Z.of_nat (length prefix)) by lia.
    apply Hcells. rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity.
Qed.
