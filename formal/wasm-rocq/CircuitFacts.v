(* Locate concrete rows by shape; the kernel checks each resulting lookup.
   Search is proof automation, not an additional assumption about the circuit. *)
From Stdlib Require Import List ZArith Bool Lia.
Require Import R1CS Columns FullRelation Range Selectors ZeroTest.
Import ListNotations.
Open Scope Z_scope.

Ltac circuit_row shape name :=
  match goal with Hfull : FullRelation.satFull ?w |- _ =>
    let found := eval vm_compute in (findRow (rowsEqual shape) FullRelation.rows) in
    lazymatch found with
    | Some ?index =>
      let Hlookup := fresh "Hlookup" in
      assert (Hlookup : nth_error FullRelation.rows index = Some shape)
        by (vm_compute; reflexivity);
      pose proof (Range.sat_row w index shape (proj2 Hfull) Hlookup) as name;
      clear Hlookup
    | None => fail "Missing circuit row" shape
    end
  end.

Ltac circuit_boolean column name :=
  let Hrow := fresh "Hrow" in
  circuit_row (Range.booleanRow column) Hrow;
  match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    pose proof (Range.boolean_row_bound w column Hprime (proj1 Hfull) Hrow) as name;
    clear Hrow
  end.

Ltac circuit_range column name :=
  let found := eval vm_compute in (findRow
    (fun row => let '(a, _, _) := row in
      rangeRowChecked FullRelation.rows column (length a) row) FullRelation.rows) in
  lazymatch found with
  | Some ?index =>
    let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
    lazymatch raw with ((?bit, _) :: ?tail, _, _) =>
      let count := eval compute in (S (length tail)) in
      let bits := constr:(seq bit count) in
      let Hfound := fresh "Hfound" in
      let Hbits := fresh "Hbits" in let Hrow := fresh "Hrow" in
      assert (Hfound : forallb (fun b => inFull (booleanRow b)) bits = true)
        by (vm_compute; reflexivity);
      match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
        let other := fresh "bitColumn" in let Hin := fresh "Hin" in
        assert (Hbits : Forall (fun b => sat w (booleanRow b)) bits) by
          (apply Forall_forall; intros other Hin; apply (in_full_sat w _ (proj2 Hfull));
           exact (proj1 (forallb_forall _ _) Hfound other Hin));
        circuit_row (recomposeRow column bits) Hrow;
        pose proof (range_from_rows w column bits Hprime (proj1 Hfull) Hbits Hrow
          ltac:(vm_compute; reflexivity)) as name;
        let maximum := eval vm_compute in (eval (fun _ => 1) (bitTerms bits)) in
        change (0 <= w column <= maximum) in name;
        clear Hfound Hbits Hrow
      end
    end
  | None => fail "Missing range recomposition" column
  end.

Ltac circuit_zero_test input inverse flag name :=
  let Hinv := fresh "Hinv" in let Hzero := fresh "Hzero" in
  circuit_row (inverseRow input inverse flag) Hinv;
  circuit_row (zeroRow input flag) Hzero;
  match goal with Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    pose proof (zero_test_value w input inverse flag Hprime (proj1 Hfull) Hinv Hzero) as name;
    clear Hinv Hzero
  end.

Ltac expand_row H := unfold sat in H; cbn [eval fold_right] in H; normalize_columns.

Lemma full_aux_selectors_zero : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 ->
  forall column, In column selectorColumns -> w column = 0.
Proof.
  intros w Hprime Hfull Hprogram column Hin.
  destruct (full_opcode_selectors_one_hot w Hprime Hfull) as [_ [Hsum _]].
  pose proof (selector_sum_one w selectorColumns column Hin).
  pose proof (value_bound w column). lia.
Qed.

Ltac zero_opcode_columns Hzero H :=
  repeat match type of H with
  | context [value ?w ?column] =>
      let member := eval vm_compute in (existsb (Nat.eqb column) selectorColumns) in
      lazymatch member with
      | true => rewrite (Hzero column ltac:(vm_compute; tauto)) in H
      end
  end.

Definition auxiliaryTerms (terms : SparseRow) : SparseRow :=
  filter (fun entry => negb (existsb (Nat.eqb (fst entry)) selectorColumns)) terms.

Definition auxiliaryRow (row : Row) : Row :=
  let '(a, b, c) := row in (auxiliaryTerms a, auxiliaryTerms b, auxiliaryTerms c).

Lemma auxiliary_terms_eval : forall (w : Witness) terms,
  (forall column, In column selectorColumns -> w column = 0) ->
  eval w (auxiliaryTerms terms) = eval w terms.
Proof.
  intros w terms Hzero. induction terms as [ | [column coefficient] rest IH]; [reflexivity | ].
  cbn [auxiliaryTerms filter fst].
  destruct (existsb (Nat.eqb column) selectorColumns) eqn:Hmember.
  - apply existsb_exists in Hmember. destruct Hmember as [c [Hin Heq]].
    apply Nat.eqb_eq in Heq. subst c. cbn [negb eval fold_right].
    rewrite Hzero by exact Hin. change (eval w (auxiliaryTerms rest) = coefficient * 0 + eval w rest).
    rewrite IH. ring.
  - cbn [negb eval fold_right].
    change (coefficient * w column + eval w (auxiliaryTerms rest) = coefficient * w column + eval w rest).
    rewrite IH. reflexivity.
Qed.

Lemma auxiliary_row_sat : forall w row,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 ->
  sat w row -> sat w (auxiliaryRow row).
Proof.
  intros w [[a b] c] Hprime Hfull Hprogram Hsat.
  unfold sat, auxiliaryRow. rewrite !auxiliary_terms_eval
    by exact (full_aux_selectors_zero w Hprime Hfull Hprogram).
  exact Hsat.
Qed.

Ltac auxiliary_row index name :=
  match goal with
  | Hprime : fieldPrime, Hfull : FullRelation.satFull ?w |- _ =>
    let raw := eval vm_compute in (List.nth index FullRelation.rows ([], [], [])) in
    let reduced := eval vm_compute in (auxiliaryRow raw) in
    let Hsat := fresh "Hsat" in
    pose proof (Range.sat_row w index raw (proj2 Hfull) eq_refl) as Hsat;
    pose proof (auxiliary_row_sat w raw Hprime Hfull ltac:(assumption) Hsat) as name;
    clear Hsat; change (sat w reduced) in name; expand_row name
  end.

Ltac selected_circuit_row chosen shape name :=
  let found := eval vm_compute in (findRow (rowsEqual shape) FullRelation.rows) in
  lazymatch found with
  | Some ?index => selected_row chosen index name
  | None => fail "Missing circuit row" shape
  end.

(* A split host call is a host call. *)
Lemma full_no_host_split : forall (w : Witness),
  FullRelation.satFull w -> w COL_HOST_CALL_ACTIVE = 0 -> w COL_HOST_SPLIT_CALL = 0.
Proof.
  intros w Hfull Hhost.
  circuit_row ([(COL_HOST_CALL_ACTIVE, 1)], [(COL_ONE, 1); (COL_HOST_RETURN_COUNT_IS_ZERO, -1)],
    [(COL_HOST_SPLIT_CALL, 1)]) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hhost in Hrow.
  pose proof (value_bound w COL_HOST_SPLIT_CALL). field_linear Hrow.
Qed.

Lemma full_no_resume_pop : forall (w : Witness),
  FullRelation.satFull w -> w COL_HOST_RESUME = 0 -> w COL_HOST_RESUME_POP = 0.
Proof.
  intros w Hfull Hresume. assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_ONE, 1); (COL_HOST_RESUME, -1)], [(COL_HOST_RESUME_POP, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone, Hresume in Hrow.
  pose proof (value_bound w COL_HOST_RESUME_POP). field_linear Hrow.
Qed.

(* Host contexts are read only on a resume or a nested entry. *)
Lemma full_no_host_context_read : forall (w : Witness),
  FullRelation.satFull w -> w COL_HOST_RESUME = 0 -> w COL_HOST_NESTED_ENTRY = 0 -> w COL_HOST_CONTEXT_READ = 0.
Proof.
  intros w Hfull Hresume Hnested. assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_HOST_CONTEXT_READ, 1); (COL_HOST_NESTED_ENTRY, -1); (COL_HOST_RESUME, -1)], [(COL_ONE, 1)],
    ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone, Hresume, Hnested in Hrow.
  pose proof (value_bound w COL_HOST_CONTEXT_READ). field_linear Hrow.
Qed.

(* Only a turn boundary or a resume clears a finished activation. *)
Lemma full_no_activation_clear : forall (w : Witness),
  FullRelation.satFull w -> w COL_TURN_BOUNDARY = 0 -> w COL_HOST_RESUME = 0 ->
  w COL_HOST_CLEAR_ACTIVATION_FINISHED = 0.
Proof.
  intros w Hfull Hturn Hresume.
  circuit_row ([(COL_TURN_BOUNDARY, 1); (COL_HOST_RESUME, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)],
    [(COL_HOST_CLEAR_ACTIVATION_FINISHED, 1)]) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hturn, Hresume in Hrow.
  pose proof (value_bound w COL_HOST_CLEAR_ACTIVATION_FINISHED). field_linear Hrow.
Qed.

(* An opcode row is no host switch: suspend, resume, nested entry and resume pop
   are other row kinds, or only happen with them. *)
Lemma full_selected_no_host_switch : forall (w : Witness) selected,
  Range.fieldPrime -> FullRelation.satFull w -> In selected selectorColumns -> w selected = 1 ->
  w COL_HOST_SUSPEND = 0 /\ w COL_HOST_RESUME = 0 /\ w COL_HOST_NESTED_ENTRY = 0 /\ w COL_HOST_RESUME_POP = 0 /\
  w COL_HOST_CONTEXT_READ = 0.
Proof.
  intros w selected Hprime Hfull Hin Hselected.
  pose proof (full_selected_is_program_row w selected Hprime Hfull Hin Hselected) as Hprogram.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_boolean COL_FRAME_MOVE_ACTIVE_BEFORE Hb1. circuit_boolean COL_LOCAL_ZERO_ACTIVE_BEFORE Hb2.
  circuit_boolean COL_PADDING_ACTIVE Hb4.
  circuit_boolean COL_GATHER_ACTIVE Hb5. circuit_boolean COL_TURN_BOUNDARY Hb6.
  circuit_boolean COL_HOST_RESUME Hb7. circuit_boolean COL_HOST_SUSPEND Hb8.
  circuit_boolean COL_PERM_PENDING_BEFORE Hb9. circuit_boolean COL_PERM_ROUND_BEFORE_IS_ZERO Hb10.
  pose proof (Range.in_full_sat w _ (proj2 Hfull) row_kind_row_in_full) as Hkinds.
  unfold sat, rowKindRow in Hkinds. cbn [eval fold_right] in Hkinds. change (w COL_ONE = 1) in Hone.
  rewrite Hone, Hprogram in Hkinds. apply Z.mod_divide in Hkinds; [ | unfold modulus; lia].
  destruct Hkinds as [k Hk]. unfold modulus in *.
  assert (Hturn : w COL_TURN_BOUNDARY = 0) by lia.
  assert (Hresume : w COL_HOST_RESUME = 0) by lia.
  assert (Hsuspend : w COL_HOST_SUSPEND = 0) by lia.
  assert (Hnested : w COL_HOST_NESTED_ENTRY = 0).
  { circuit_row ([(COL_TURN_BOUNDARY, 1)], [(COL_HOST_AWAITING_BEFORE, 1)], [(COL_HOST_NESTED_ENTRY, 1)]) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hturn in Hrow.
    pose proof (value_bound w COL_HOST_NESTED_ENTRY). field_linear Hrow. }
  split; [exact Hsuspend | ]. split; [exact Hresume | ]. split; [exact Hnested | ].
  split; [exact (full_no_resume_pop w Hfull Hresume) | exact (full_no_host_context_read w Hfull Hresume Hnested)].
Qed.

(* Zero the host-switch terms of a row taken on an opcode row. *)
Ltac zero_host_switch H :=
  match goal with
  | Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w, Hin : In ?s selectorColumns,
    Hs : value ?w ?s = 1 |- _ =>
      let Ha := fresh "Hsuspend" in let Hb := fresh "Hresume" in
      let Hc := fresh "Hnested" in let Hd := fresh "HresumePop" in
      destruct (full_selected_no_host_switch w s Hprime Hfull Hin Hs) as [Ha [Hb [Hc [Hd _]]]];
      normalize_columns; try rewrite Ha in H; try rewrite Hb in H; try rewrite Hc in H;
      try rewrite Hd in H; clear Ha Hb Hc Hd
  | Hprime : Range.fieldPrime, Hfull : FullRelation.satFull ?w, Hs : value ?w ?s = 1 |- _ =>
      let member := eval vm_compute in (existsb (Nat.eqb s) selectorColumns) in
      lazymatch member with
      | true =>
          let Hin := fresh "Hin" in let Ha := fresh "Hsuspend" in let Hb := fresh "Hresume" in
          let Hc := fresh "Hnested" in let Hd := fresh "HresumePop" in
          assert (Hin : In s selectorColumns) by (vm_compute; tauto);
          destruct (full_selected_no_host_switch w s Hprime Hfull Hin Hs) as [Ha [Hb [Hc [Hd _]]]];
          normalize_columns; try rewrite Ha in H; try rewrite Hb in H; try rewrite Hc in H;
          try rewrite Hd in H; clear Hin Ha Hb Hc Hd
      end
  end.

Lemma gated_value_zero : forall gate value,
  (gate * (1 * value + 0) - 0) mod modulus = 0 -> gate = 1 -> 0 <= value < modulus -> value = 0.
Proof.
  intros gate value Hrow -> Hbound.
  rewrite Z.add_0_r, Z.sub_0_r, !Z.mul_1_l, Z.mod_small in Hrow by exact Hbound. exact Hrow.
Qed.

(* Branches and popping returns slide. *)
Lemma full_frame_slide_gate : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_FRAME_SLIDE_ACTIVE =
    w COL_SEL_BR + w COL_SEL_BR_IF_EQZ + w COL_SEL_BR_TABLE + w COL_CALL_STACK_POP_PRESENT.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_FRAME_SLIDE_ACTIVE, 1); (COL_SEL_BR, -1); (COL_SEL_BR_IF_EQZ, -1); (COL_SEL_BR_TABLE, -1);
    (COL_CALL_STACK_POP_PRESENT, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  circuit_boolean COL_FRAME_SLIDE_ACTIVE Hs. circuit_boolean COL_SEL_BR Hb. circuit_boolean COL_SEL_BR_IF_EQZ Hi.
  circuit_boolean COL_SEL_BR_TABLE Ht. circuit_boolean COL_CALL_STACK_POP_PRESENT Hp.
  unfold sat in Hrow. cbn [eval fold_right] in Hrow. rewrite Hone in Hrow. field_linear Hrow.
Qed.

(* A slide drops sp by its distance. *)
Lemma full_frame_slide_drop : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_FRAME_SLIDE_ACTIVE = 1 ->
  w COL_FRAME_SP_DROP = w COL_FRAME_MOVE_DISTANCE_AFTER.
Proof.
  intros w Hprime Hfull Hslide.
  circuit_row ([(COL_FRAME_SLIDE_ACTIVE, 1)], [(COL_FRAME_SP_DROP, 1); (COL_FRAME_MOVE_DISTANCE_AFTER, -1)],
    ([] : SparseRow)) Hrow.
  pose proof (value_bound w COL_FRAME_SP_DROP). pose proof (value_bound w COL_FRAME_MOVE_DISTANCE_AFTER).
  unfold sat in Hrow. cbn [eval fold_right] in Hrow. rewrite Hslide in Hrow. field_linear Hrow.
Qed.

Lemma full_no_slide : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_CALL_STACK_POP_PRESENT = 0 ->
  w COL_SEL_BR = 0 -> w COL_SEL_BR_IF_EQZ = 0 -> w COL_SEL_BR_TABLE = 0 -> w COL_FRAME_SLIDE_ACTIVE = 0.
Proof.
  intros w Hprime Hfull Hpop Hbr Hif Htable. rewrite (full_frame_slide_gate w Hprime Hfull). lia.
Qed.

Lemma full_aux_no_slide : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 0 -> w COL_CALL_STACK_POP_PRESENT = 0 ->
  w COL_FRAME_SLIDE_ACTIVE = 0.
Proof.
  intros w Hprime Hfull Hprogram Hpop.
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hzero.
  apply full_no_slide; try assumption; apply Hzero; vm_compute; tauto.
Qed.

Lemma full_selected_no_slide : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w -> In chosen selectorColumns -> w chosen = 1 ->
  ~ In chosen [COL_SEL_BR; COL_SEL_BR_IF_EQZ; COL_SEL_BR_TABLE] -> w COL_CALL_STACK_POP_PRESENT = 0 ->
  w COL_FRAME_SLIDE_ACTIVE = 0.
Proof.
  intros w chosen Hprime Hfull Hin Hchosen Hbranch Hpop.
  apply full_no_slide; try assumption;
    (eapply full_opcode_selector_exclusive; [exact Hprime | exact Hfull | exact Hchosen | exact Hin |
      vm_compute; tauto | intro Heq; subst chosen; apply Hbranch; cbn; tauto]).
Qed.

(* Only frame transitions drop stack cells outside the stack balance: slides,
   tail calls, entries, and clean activation returns, which set the exit latch. *)
Lemma full_frame_sp_drop_idle : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_FRAME_SLIDE_ACTIVE = 0 -> w COL_GUEST_ENTRY_ACTIVE = w COL_CALL_STACK_PUSH_PRESENT ->
  w COL_TURN_BOUNDARY = 0 -> w COL_HOST_EVENT_EXIT_LATCH = 0 ->
  w COL_FRAME_SP_DROP = 0.
Proof.
  intros w Hprime Hfull Hpop Hentry Hturn Hlatch.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hstate.
  circuit_row ([(COL_ONE, 1); (COL_FRAME_SLIDE_ACTIVE, -1); (COL_GUEST_ENTRY_ACTIVE, -1);
    (COL_CALL_STACK_PUSH_PRESENT, 1); (COL_TURN_BOUNDARY, -1); (COL_ACTIVATION_FINISHED_AFTER, -1);
    (COL_ACTIVATION_FINISHED_BEFORE, 1); (COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1);
    (COL_HOST_CLEAR_ACTIVATION_FINISHED, -1)], [(COL_FRAME_SP_DROP, 1)], ([] : SparseRow)) Hrow.
  unfold sat in Hstate, Hrow. cbn [eval fold_right] in Hstate, Hrow.
  assert (Hclean : w COL_ACTIVATION_FINISHED_AFTER - w COL_ACTIVATION_FINISHED_BEFORE - w COL_TRAPPED_AFTER +
    w COL_TRAPPED_BEFORE + w COL_HOST_CLEAR_ACTIVATION_FINISHED = 0).
  { circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hfa. circuit_boolean COL_ACTIVATION_FINISHED_BEFORE Hfb.
    circuit_boolean COL_TRAPPED_AFTER Hta. circuit_boolean COL_TRAPPED_BEFORE Htb.
    circuit_boolean COL_HOST_CLEAR_ACTIVATION_FINISHED Hclear.
    rewrite Hone, Hlatch in Hstate.
    apply Z.mod_divide in Hstate; [ | unfold modulus; lia]. destruct Hstate as [k Hstate].
    unfold modulus in Hstate. lia. }
  apply (gated_value_zero _ _ Hrow); [ | exact (value_bound w COL_FRAME_SP_DROP)].
  rewrite Hone, Hpop, Hentry, Hturn. lia.
Qed.
