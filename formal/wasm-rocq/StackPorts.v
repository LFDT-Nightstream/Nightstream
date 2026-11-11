(* SPEC.md section 7.6: split gates and the prior values of combined stack ports. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace.
Import ListNotations.
Open Scope Z_scope.

Record StackPortGates (w : Witness) : Prop := {
  separate_read : w (List.nth 0%nat COL_STACK_SEPARATE 0%nat) =
    w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) - w COL_STACK_RMW;
  separate_write_lo : w (List.nth 1%nat COL_STACK_SEPARATE 0%nat) =
    w COL_STACK_WRITE0_ACTIVE - w COL_STACK_RMW;
  separate_write_hi : w (List.nth 2%nat COL_STACK_SEPARATE 0%nat) =
    w COL_STACK_WRITE0_HI_ACTIVE - w COL_STACK_RMW;
  rmw_nonnegative : 0 <= w COL_STACK_RMW;
  separate_nonnegative : forall i, (i < 3)%nat ->
    0 <= w (List.nth i COL_STACK_SEPARATE 0%nat)
}.

Lemma full_stack_port_gates : forall w,
  fieldPrime -> FullRelation.satFull w -> StackPortGates w.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  constructor; try (intros; apply (proj1 (value_bound _ _)));
    match goal with |- value w ?s = value w ?o - value w COL_STACK_RMW =>
      circuit_row ([(s, 1); (o, -1); (COL_STACK_RMW, 1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow;
      circuit_boolean s Hs; circuit_boolean o Ho; circuit_boolean COL_STACK_RMW Hr;
      expand_row Hrow; rewrite Hone in Hrow; field_linear Hrow
    end.
Qed.

Lemma stack_no_rmw : forall w, StackPortGates w ->
  w COL_STACK_WRITE0_ACTIVE = 0 -> w COL_STACK_RMW = 0.
Proof.
  intros w Hports Hzero.
  pose proof (separate_write_lo _ Hports).
  pose proof (rmw_nonnegative _ Hports).
  pose proof (separate_nonnegative _ Hports 1%nat ltac:(lia)). lia.
Qed.

Ltac rewrite_stack_ports Hports :=
  let Hr := fresh "HseparateRead" in let Hl := fresh "HseparateLo" in let Hh := fresh "HseparateHi" in
  pose proof (separate_read _ Hports) as Hr;
  pose proof (separate_write_lo _ Hports) as Hl;
  pose proof (separate_write_hi _ Hports) as Hh;
  normalize_columns; rewrite ?Hr, ?Hl, ?Hh; clear Hr Hl Hh.

(* Locate the selector-sum equation by shape; selected_row checks the lookup. *)
Definition stackRmwGateRow : nat := Eval vm_compute in
  match findRow (fun row => match row with
    | ((column, _) :: _, [(one, 1)], []) => Nat.eqb column COL_STACK_RMW && Nat.eqb one COL_ONE
    | _ => false end) FullRelation.rows with Some i => i | None => 0%nat end.

Ltac selected_stack_rmw chosen :=
  let Hrow := fresh "HrmwRow" in
  selected_row chosen stackRmwGateRow Hrow;
  match goal with Hfull : FullRelation.satFull ?w |- _ =>
    let Hone := fresh "Hunit" in
    pose proof (proj1 Hfull) as Hone;
    pose proof (value_bound w COL_STACK_RMW);
    normalize_columns; rewrite Hone in Hrow; field_linear Hrow
  end.

Lemma full_aux_stack_rmw : forall w,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 -> w COL_STACK_RMW = 0.
Proof.
  intros w Hprime Hfull Hprogram.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  auxiliary_row stackRmwGateRow Hrow. normalize_columns. rewrite Hone in Hrow.
  pose proof (value_bound w COL_STACK_RMW). field_linear Hrow.
Qed.

Ltac zero_stack_ports :=
  match goal with Hprime : fieldPrime, Hfull : FullRelation.satFull ?w,
    Hwrite : value ?w COL_STACK_WRITE0_ACTIVE = 0 |- _ =>
    let Hp := fresh "Hports" in let Hr := fresh "Hrmw" in
    pose proof (full_stack_port_gates w Hprime Hfull) as Hp;
    pose proof (stack_no_rmw w Hp Hwrite) as Hr;
    rewrite_stack_ports Hp; rewrite Hr, !Z.sub_0_r; clear Hp Hr
  end.
