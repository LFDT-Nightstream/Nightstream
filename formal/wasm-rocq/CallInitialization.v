(* The circuit's initialization countdowns and the row kinds they select. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts
  MemoryTrace MemoryCatalog TraceRefinement StraightLineSpec.
Import ListNotations.
Open Scope Z_scope.

Definition rowKindFlags : list Column :=
  [COL_IS_PROGRAM_ROW; COL_PARAM_INIT_ACTIVE_BEFORE; COL_LOCAL_ZERO_ACTIVE_BEFORE;
   COL_TAIL_ENTER_ACTIVE; COL_PADDING_ACTIVE; COL_GATHER_ACTIVE; COL_TURN_BOUNDARY;
   COL_HOST_RESUME; COL_HOST_SUSPEND; COL_PERM_PENDING_BEFORE].

Lemma full_row_kind_sum : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  selectorSum w rowKindFlags = w COL_PERM_ROUND_BEFORE_IS_ZERO /\
  w COL_PERM_ROUND_BEFORE_IS_ZERO <= 1.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hbits : Forall (fun c => w c = 0 \/ w c = 1) rowKindFlags).
  { unfold rowKindFlags. repeat apply Forall_cons; try apply Forall_nil;
      match goal with |- value _ ?column = 0 \/ _ => circuit_boolean column H; exact H end. }
  circuit_boolean COL_PERM_ROUND_BEFORE_IS_ZERO Hperm.
  circuit_row (map (fun c => (c, 1)) rowKindFlags ++
    [(COL_PERM_ROUND_BEFORE_IS_ZERO, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  unfold sat in Hrow. change
    ((eval w (map (fun c => (c, 1)) rowKindFlags ++ [(COL_PERM_ROUND_BEFORE_IS_ZERO, -1)]) *
      (1 * w COL_ONE + 0) - 0) mod modulus = 0) in Hrow.
  rewrite eval_app, selector_sum_eval, Hone in Hrow.
  cbn [eval fold_right] in Hrow.
  pose proof (selector_sum_boolean_bound w rowKindFlags Hbits) as Hbound.
  change (0 <= selectorSum w rowKindFlags <= 10) in Hbound.
  apply Z.mod_divide in Hrow; [ | unfold modulus; lia].
  destruct Hrow as [k Hrow]. unfold modulus in Hrow.
  destruct Hperm; split; lia.
Qed.

Lemma full_row_kind_exclusive : forall (w : Witness) active other,
  fieldPrime -> FullRelation.satFull w ->
  In active rowKindFlags -> In other rowKindFlags -> active <> other ->
  w active = 1 -> w other = 0.
Proof.
  intros w active other Hprime Hfull Ha Ho Hneq Hone.
  destruct (full_row_kind_sum w Hprime Hfull) as [Hsum Hbound].
  pose proof (selector_sum_two w rowKindFlags active other Hneq Ha Ho).
  pose proof (value_bound w other). lia.
Qed.

(* Host suspend and resume are row kinds of their own, and a nested host entry
   happens only at a turn boundary. *)
Lemma full_host_switch_absent : forall (w : Witness) kind,
  fieldPrime -> FullRelation.satFull w -> In kind rowKindFlags ->
  kind <> COL_HOST_SUSPEND -> kind <> COL_HOST_RESUME -> kind <> COL_TURN_BOUNDARY -> w kind = 1 ->
  w COL_HOST_SUSPEND = 0 /\ w COL_HOST_RESUME = 0 /\ w COL_HOST_NESTED_ENTRY = 0.
Proof.
  intros w kind Hprime Hfull Hin Hs Hr Ht Hkind. split; [ | split].
  - exact (full_row_kind_exclusive w kind COL_HOST_SUSPEND Hprime Hfull Hin ltac:(cbn; auto 20) Hs Hkind).
  - exact (full_row_kind_exclusive w kind COL_HOST_RESUME Hprime Hfull Hin ltac:(cbn; auto 20) Hr Hkind).
  - pose proof (full_row_kind_exclusive w kind COL_TURN_BOUNDARY Hprime Hfull Hin ltac:(cbn; auto 20) Ht Hkind)
      as Hturn.
    circuit_row ([(COL_TURN_BOUNDARY, 1)], [(COL_HOST_AWAITING_BEFORE, 1)], [(COL_HOST_NESTED_ENTRY, 1)]) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hturn in Hrow.
    pose proof (value_bound w COL_HOST_NESTED_ENTRY). field_linear Hrow.
Qed.

Lemma full_initialization_flags : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_LOCAL_ZERO_ACTIVE_AFTER =
    (if w COL_LOCAL_ZERO_REMAINING_AFTER =? 0 then 0 else 1) /\
  w COL_PARAM_INIT_ACTIVE_AFTER =
    (if w COL_LOCAL_ZERO_REMAINING_AFTER =? 0 then
      if w COL_PARAM_INIT_REMAINING_AFTER =? 0 then 0 else 1 else 0).
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_zero_test COL_LOCAL_ZERO_REMAINING_AFTER COL_LOCAL_ZERO_REMAINING_AFTER_INV
    COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO Hz.
  circuit_zero_test COL_PARAM_INIT_REMAINING_AFTER COL_PARAM_INIT_REMAINING_AFTER_INV
    COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO Hp.
  circuit_row ([(COL_LOCAL_ZERO_ACTIVE_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO, 1);
    (COL_ONE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hzero.
  circuit_row ([(COL_ONE, 1); (COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, -1)],
    [(COL_ONE, 1); (COL_LOCAL_ZERO_ACTIVE_AFTER, -1)], [(COL_PARAM_INIT_ACTIVE_AFTER, 1)]) Hparam.
  expand_row Hzero. expand_row Hparam.
  normalize_columns. rewrite Hone in Hzero, Hparam.
  rewrite Hz in Hzero. rewrite Hp in Hparam.
  pose proof (value_bound w COL_LOCAL_ZERO_ACTIVE_AFTER) as Hzb.
  pose proof (value_bound w COL_PARAM_INIT_ACTIVE_AFTER) as Hpb.
  normalize_columns.
  assert (Hflag : w COL_LOCAL_ZERO_ACTIVE_AFTER =
      (if w COL_LOCAL_ZERO_REMAINING_AFTER =? 0 then 0 else 1)).
  { normalize_columns.
    let c := eval compute in COL_LOCAL_ZERO_REMAINING_AFTER in
    destruct (w c =? 0); field_linear Hzero. }
  split; [exact Hflag | ].
  normalize_columns. rewrite Hflag in Hparam.
  let z := eval compute in COL_LOCAL_ZERO_REMAINING_AFTER in
  let p := eval compute in COL_PARAM_INIT_REMAINING_AFTER in
  destruct (w z =? 0), (w p =? 0); field_linear Hparam.
Qed.

Lemma full_guest_initialization_seed : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_GUEST_ENTRY_ACTIVE = 1 ->
  w COL_LOCAL_ZERO_REMAINING_AFTER + w COL_CALL_PARAM_COUNT =
    w COL_ENTERED_FUNCTION_NUM_LOCALS /\
  w COL_PARAM_INIT_REMAINING_AFTER = w COL_CALL_PARAM_COUNT.
Proof.
  intros w Hprime Hfull Hprogram Hguest.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  pose proof (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY
    Hprime Hfull ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram) as Hturn.
  circuit_range COL_LOCAL_ZERO_REMAINING_AFTER Hz.
  circuit_range COL_CALL_PARAM_COUNT Hp.
  circuit_range COL_ENTERED_FUNCTION_NUM_LOCALS Hl.
  circuit_row ([(COL_GUEST_ENTRY_ACTIVE, 1); (COL_TURN_BOUNDARY, 1)],
    [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_CALL_PARAM_COUNT, 1);
     (COL_ENTERED_FUNCTION_NUM_LOCALS, -1)], ([] : SparseRow)) Hzero.
  circuit_row ([(COL_GUEST_ENTRY_ACTIVE, 1)],
    [(COL_PARAM_INIT_REMAINING_AFTER, 1); (COL_CALL_PARAM_COUNT, -1)], ([] : SparseRow)) Hparam.
  expand_row Hzero. expand_row Hparam.
  normalize_columns. rewrite Hguest, Hturn in Hzero. rewrite Hguest in Hparam.
  pose proof (value_bound w COL_PARAM_INIT_REMAINING_AFTER).
  split; [field_linear Hzero | field_linear Hparam].
Qed.

Lemma full_zero_initialization_row : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  w COL_LOCAL_ZERO_REMAINING_BEFORE = w COL_LOCAL_ZERO_REMAINING_AFTER + 1 /\
  w COL_LOCAL_INDEX + w COL_LOCAL_ZERO_REMAINING_BEFORE = w COL_CURRENT_FUNCTION_NUM_LOCALS /\
  w COL_LOCAL_VALUE = 0 /\ w COL_LOCAL_VALUE_HI = 0.
Proof.
  intros w Hprime Hfull Hactive.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_LOCAL_ZERO_REMAINING_BEFORE Hb.
  circuit_range COL_LOCAL_ZERO_REMAINING_AFTER Ha.
  circuit_range COL_LOCAL_INDEX Hi.
  circuit_range COL_CURRENT_FUNCTION_NUM_LOCALS Hl.
  circuit_row ([(COL_LOCAL_ZERO_ACTIVE_BEFORE, 1)],
    [(COL_LOCAL_ZERO_REMAINING_BEFORE, 1); (COL_LOCAL_ZERO_REMAINING_AFTER, -1); (COL_ONE, -1)],
    ([] : SparseRow)) Hcount.
  circuit_row ([(COL_LOCAL_ZERO_ACTIVE_BEFORE, 1)],
    [(COL_LOCAL_INDEX, 1); (COL_CURRENT_FUNCTION_NUM_LOCALS, -1); (COL_LOCAL_ZERO_REMAINING_BEFORE, 1)],
    ([] : SparseRow)) Hindex.
  circuit_row ([(COL_LOCAL_ZERO_ACTIVE_BEFORE, 1)], [(COL_LOCAL_VALUE, 1)], ([] : SparseRow)) Hlo.
  circuit_row ([(COL_LOCAL_ZERO_ACTIVE_BEFORE, 1)], [(COL_LOCAL_VALUE_HI, 1)], ([] : SparseRow)) Hhi.
  expand_row Hcount. expand_row Hindex. expand_row Hlo. expand_row Hhi.
  normalize_columns. rewrite Hactive, Hone in Hcount.
  rewrite Hactive in Hindex, Hlo, Hhi.
  pose proof (value_bound w COL_LOCAL_VALUE). pose proof (value_bound w COL_LOCAL_VALUE_HI).
  repeat split; [field_linear Hcount | field_linear Hindex | field_linear Hlo | field_linear Hhi].
Qed.

Lemma full_parameter_initialization_row : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_PARAM_INIT_ACTIVE_BEFORE = 1 ->
  w COL_PARAM_INIT_REMAINING_BEFORE = w COL_PARAM_INIT_REMAINING_AFTER + 1 /\
  w COL_LOCAL_INDEX = w COL_PARAM_INIT_REMAINING_AFTER /\
  w COL_LOCAL_VALUE = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) /\
  w COL_LOCAL_VALUE_HI = w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat).
Proof.
  intros w Hprime Hfull Hactive.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_PARAM_INIT_REMAINING_BEFORE Hb.
  circuit_range COL_PARAM_INIT_REMAINING_AFTER Ha.
  circuit_range COL_LOCAL_INDEX Hi.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(COL_PARAM_INIT_REMAINING_BEFORE, 1); (COL_PARAM_INIT_REMAINING_AFTER, -1); (COL_ONE, -1)],
    ([] : SparseRow)) Hcount.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(COL_LOCAL_INDEX, 1); (COL_PARAM_INIT_REMAINING_BEFORE, -1); (COL_ONE, 1)],
    ([] : SparseRow)) Hindex.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, 1); (COL_LOCAL_VALUE, -1)], ([] : SparseRow)) Hlo.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat, 1); (COL_LOCAL_VALUE_HI, -1)], ([] : SparseRow)) Hhi.
  expand_row Hcount. expand_row Hindex. expand_row Hlo. expand_row Hhi.
  normalize_columns. rewrite Hactive, Hone in Hcount, Hindex.
  rewrite Hactive in Hlo, Hhi.
  pose proof (value_bound w COL_LOCAL_VALUE). pose proof (value_bound w COL_LOCAL_VALUE_HI).
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)).
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)).
  assert (Hdecrement : w COL_PARAM_INIT_REMAINING_BEFORE = w COL_PARAM_INIT_REMAINING_AFTER + 1)
    by (field_linear Hcount).
  repeat split; [exact Hdecrement | field_linear Hindex | field_linear Hlo | field_linear Hhi].
Qed.

Definition initializationKind (chosen : Column) : Prop :=
  chosen = COL_LOCAL_ZERO_ACTIVE_BEFORE \/ chosen = COL_PARAM_INIT_ACTIVE_BEFORE.

Lemma initialization_kind_in : forall chosen,
  initializationKind chosen -> In chosen rowKindFlags.
Proof. intros chosen [-> | ->]; cbn; tauto. Qed.

Lemma full_initialization_auxiliary : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  w COL_IS_PROGRAM_ROW = 0 /\ w COL_GUEST_ENTRY_ACTIVE = 0 /\
  w COL_CALL_STACK_PUSH_PRESENT = 0 /\ w COL_CALL_STACK_POP_PRESENT = 0 /\
  w COL_TURN_BOUNDARY = 0 /\ ProgramRowQuiet w.
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  assert (Hin : In chosen rowKindFlags) by (apply initialization_kind_in; exact Hkind).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hprogram : w COL_IS_PROGRAM_ROW = 0).
  { eapply full_row_kind_exclusive; try eassumption; [cbn; tauto | destruct Hkind as [-> | ->]; discriminate]. }
  pose proof (full_aux_selectors_zero w Hprime Hfull Hprogram) as Hselectors.
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { full_row guestEntryFlagRow Hrow. zero_opcode_columns Hselectors Hrow.
    pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE). field_linear Hrow. }
  assert (Hpush : w COL_CALL_STACK_PUSH_PRESENT = 0).
  { full_row callStackPushRow Hrow. zero_opcode_columns Hselectors Hrow.
    pose proof (value_bound w COL_CALL_STACK_PUSH_PRESENT). field_linear Hrow. }
  assert (Hpop : w COL_CALL_STACK_POP_PRESENT = 0).
  { full_row callStackPopSupportRow Hrow. zero_opcode_columns Hselectors Hrow.
    normalize_columns. rewrite Hone in Hrow.
    pose proof (value_bound w COL_CALL_STACK_POP_PRESENT). field_linear Hrow. }
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { eapply full_row_kind_exclusive; try eassumption; [cbn; tauto | destruct Hkind as [-> | ->]; discriminate]. }
  assert (Hgather : w COL_GATHER_ACTIVE = 0).
  { eapply full_row_kind_exclusive; try eassumption; [cbn; tauto | destruct Hkind as [-> | ->]; discriminate]. }
  assert (Htail : w COL_TAIL_ENTER_ACTIVE = 0).
  { eapply full_row_kind_exclusive; try eassumption; [cbn; tauto | destruct Hkind as [-> | ->]; discriminate]. }
  repeat (apply conj; [assumption | ]).
  apply full_quiet_from_gates; assumption.
Qed.

Lemma full_initialization_preserves_frame : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_STACK_FRAME_BASE_AFTER = w COL_STACK_FRAME_BASE_BEFORE /\
  w COL_CALL_STACK_DEPTH_AFTER = w COL_CALL_STACK_DEPTH_BEFORE.
Proof.
  intros w chosen Hprime Hfull Hkind Hchosen.
  destruct (full_initialization_auxiliary w chosen Hprime Hfull Hkind Hchosen)
    as [Hprogram [Hguest [Hpush [Hpop [Hturn Hquiet]]]]].
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct (full_host_switch_absent w chosen Hprime Hfull (initialization_kind_in chosen Hkind)
    ltac:(destruct Hkind as [-> | ->]; discriminate) ltac:(destruct Hkind as [-> | ->]; discriminate)
    ltac:(destruct Hkind as [-> | ->]; discriminate) Hchosen) as [Hsuspend [Hresume Hnested]].
  full_row localsBasePreservedRow Hfbp.
  full_row stackFrameBasePreservedRow Hbase.
  circuit_row ([(COL_CALL_STACK_DEPTH_AFTER, 1); (COL_CALL_STACK_DEPTH_BEFORE, -1);
    (COL_CALL_STACK_PUSH_PRESENT, -1); (COL_CALL_STACK_POP_PRESENT, 1);
    (COL_HOST_SUSPEND, -1); (COL_HOST_RESUME, 1)], [(COL_ONE, 1)], ([] : SparseRow)) Hdepth.
  expand_row Hdepth. normalize_columns.
  rewrite Hone, Hguest, Hpop, Hresume, Hnested in Hfbp. rewrite Hone, Hpush, Hpop, Hresume, Hnested in Hbase.
  rewrite Hone, Hpush, Hpop, Hsuspend, Hresume in Hdepth.
  pose proof (value_bound w COL_LOCALS_FBP_BEFORE). pose proof (value_bound w COL_LOCALS_FBP_AFTER).
  pose proof (value_bound w COL_STACK_FRAME_BASE_BEFORE). pose proof (value_bound w COL_STACK_FRAME_BASE_AFTER).
  pose proof (value_bound w COL_CALL_STACK_DEPTH_BEFORE). pose proof (value_bound w COL_CALL_STACK_DEPTH_AFTER).
  repeat split; [field_linear Hfbp | field_linear Hbase | field_linear Hdepth].
Qed.

Lemma full_initialization_counter_carry : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> initializationKind chosen -> w chosen = 1 ->
  (w COL_LOCAL_ZERO_ACTIVE_BEFORE = 0 ->
    w COL_LOCAL_ZERO_REMAINING_AFTER = w COL_LOCAL_ZERO_REMAINING_BEFORE) /\
  (w COL_PARAM_INIT_ACTIVE_BEFORE = 0 ->
    w COL_PARAM_INIT_REMAINING_AFTER = w COL_PARAM_INIT_REMAINING_BEFORE).
Proof.
  intros w Hchosen Hprime Hfull Hkind Hactive.
  destruct (full_initialization_auxiliary w Hchosen Hprime Hfull Hkind Hactive)
    as [Hprogram [Hguest [Hpush [Hpop [Hturn Hquiet]]]]].
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_TURN_BOUNDARY, -1);
    (COL_LOCAL_ZERO_ACTIVE_BEFORE, -1)],
    [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_BEFORE, -1)],
    ([] : SparseRow)) Hzero.
  circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_PARAM_INIT_ACTIVE_BEFORE, -1);
    (COL_TURN_BOUNDARY, -1)],
    [(COL_PARAM_INIT_REMAINING_AFTER, 1); (COL_PARAM_INIT_REMAINING_BEFORE, -1)],
    ([] : SparseRow)) Hparam.
  expand_row Hzero. expand_row Hparam.
  normalize_columns. rewrite Hone, Hguest, Hturn in Hzero, Hparam.
  pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_BEFORE).
  pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_AFTER).
  pose proof (value_bound w COL_PARAM_INIT_REMAINING_BEFORE).
  pose proof (value_bound w COL_PARAM_INIT_REMAINING_AFTER).
  split; intros Hinactive; normalize_columns;
    [rewrite Hinactive in Hzero; field_linear Hzero | rewrite Hinactive in Hparam; field_linear Hparam].
Qed.

Lemma full_zero_pc_preserved : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_LOCAL_ZERO_ACTIVE_BEFORE = 1 ->
  w COL_PC_AFTER = w COL_PC_BEFORE.
Proof.
  intros w Hprime Hfull Hzero.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hother : forall c, In c rowKindFlags -> c <> COL_LOCAL_ZERO_ACTIVE_BEFORE -> w c = 0).
  { intros c Hin Hneq. exact (full_row_kind_exclusive w COL_LOCAL_ZERO_ACTIVE_BEFORE c
      Hprime Hfull ltac:(cbn; tauto) Hin ltac:(congruence) Hzero). }
  assert (Hperm : w COL_PERM_ROUND_BEFORE_IS_ZERO = 1).
  { destruct (full_row_kind_sum w Hprime Hfull) as [Hsum Hbound].
    pose proof (selector_sum_one w rowKindFlags COL_LOCAL_ZERO_ACTIVE_BEFORE ltac:(cbn; tauto)). lia. }
  circuit_row ([(COL_HOST_SUSPEND, 1); (COL_PARAM_INIT_ACTIVE_BEFORE, 1); (COL_LOCAL_ZERO_ACTIVE_BEFORE, 1);
    (COL_TAIL_ENTER_ACTIVE, 1); (COL_PADDING_ACTIVE, 1); (COL_GATHER_ACTIVE, 1);
    (COL_PERM_PENDING_BEFORE, 1); (COL_ONE, 1); (COL_PERM_ROUND_BEFORE_IS_ZERO, -1)],
    [(COL_PC_AFTER, 1); (COL_PC_BEFORE, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow.
  repeat match type of Hrow with
  | context [value w ?c] =>
      rewrite (Hother c ltac:(cbn; tauto) ltac:(discriminate)) in Hrow
  end.
  normalize_columns. rewrite Hone, Hzero, Hperm in Hrow.
  pose proof (value_bound w COL_PC_BEFORE). pose proof (value_bound w COL_PC_AFTER).
  field_linear Hrow.
Qed.

Lemma full_parameter_pc_preserved : forall (w : Witness),
  FullRelation.satFull w -> w COL_PARAM_INIT_ACTIVE_BEFORE = 1 ->
  w COL_PC_AFTER = w COL_PC_BEFORE.
Proof.
  intros w Hfull Hparam.
  circuit_row ([(COL_PARAM_INIT_ACTIVE_BEFORE, 1)],
    [(COL_PC_AFTER, 1); (COL_PC_BEFORE, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hparam in Hrow.
  pose proof (value_bound w COL_PC_BEFORE). pose proof (value_bound w COL_PC_AFTER).
  field_linear Hrow.
Qed.

Lemma full_frame_lookup_active : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w ->
  In chosen [COL_IS_PROGRAM_ROW; COL_LOCAL_ZERO_ACTIVE_BEFORE;
    COL_PARAM_INIT_ACTIVE_BEFORE; COL_TAIL_ENTER_ACTIVE] -> w chosen = 1 ->
  w COL_PC_FREF_ACTIVE = 1.
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen.
  assert (Hin : In chosen rowKindFlags) by
    (destruct Hchoice as [<- | [<- | [<- | [<- | []]]]]; cbn; tauto).
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hother : forall c, In c [COL_GATHER_ACTIVE; COL_PERM_PENDING_BEFORE;
      COL_TURN_BOUNDARY; COL_PADDING_ACTIVE; COL_HOST_RESUME] -> w c = 0).
  { intros c Hc. eapply full_row_kind_exclusive; try eassumption.
    - destruct Hc as [<- | [<- | [<- | [<- | [<- | []]]]]]; cbn; tauto.
    - destruct Hchoice as [<- | [<- | [<- | [<- | []]]]];
        destruct Hc as [<- | [<- | [<- | [<- | [<- | []]]]]]; discriminate. }
  assert (Hperm : w COL_PERM_ROUND_BEFORE_IS_ZERO = 1).
  { destruct (full_row_kind_sum w Hprime Hfull) as [Hsum Hbound].
    pose proof (selector_sum_one w rowKindFlags chosen Hin). lia. }
  circuit_row ([(COL_PC_FREF_ACTIVE, 1); (COL_PERM_ROUND_BEFORE_IS_ZERO, -1);
    (COL_GATHER_ACTIVE, 1); (COL_PERM_PENDING_BEFORE, 1); (COL_TURN_BOUNDARY, 1); (COL_PADDING_ACTIVE, 1);
    (COL_HOST_RESUME, 1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow.
  repeat match type of Hrow with
  | context [value w ?c] => rewrite (Hother c ltac:(cbn; tauto)) in Hrow
  end.
  normalize_columns. rewrite Hone, Hperm in Hrow.
  pose proof (value_bound w COL_PC_FREF_ACTIVE). field_linear Hrow.
Qed.

(* A program row that is not a call starts no frame, so an idle countdown stays idle. *)
Lemma program_row_keeps_idle : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> In chosen selectorColumns -> w chosen = 1 ->
  ~ In chosen [COL_SEL_CALL; COL_SEL_CALL_INDIRECT; COL_SEL_RETURN_CALL; COL_SEL_RETURN_CALL_INDIRECT] ->
  initializationIdle (beforeBoundary w) -> initializationIdle (afterBoundary w).
Proof.
  intros w chosen Hprime Hfull Hin Hchosen Hnotcall [Hz [Hp [Hza Hpa]]].
  cbn [beforeBoundary boundaryZeroRemaining boundaryParamRemaining boundaryZeroActive boundaryParamActive] in *.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  assert (Hcalls : forall column, In column [COL_SEL_CALL; COL_SEL_CALL_INDIRECT; COL_SEL_RETURN_CALL;
      COL_SEL_RETURN_CALL_INDIRECT] -> w column = 0).
  { intros column Hcolumn. eapply full_opcode_selector_exclusive; [exact Hprime | exact Hfull | exact Hchosen |
      exact Hin | vm_compute in Hcolumn |- *; tauto | intro Heq; subst column; contradiction]. }
  pose proof (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  assert (Hguest : w COL_GUEST_ENTRY_ACTIVE = 0).
  { circuit_row ([(COL_SEL_CALL, 1); (COL_SEL_CALL_INDIRECT, 1); (COL_SEL_RETURN_CALL, 1);
      (COL_SEL_RETURN_CALL_INDIRECT, 1)], [(COL_TARGET_FUNCTION_IS_GUEST, 1)],
      [(COL_GUEST_ENTRY_ACTIVE, 1)]) Hrow.
    pose proof (Hcalls COL_SEL_CALL ltac:(cbn; tauto)) as Hc1.
    pose proof (Hcalls COL_SEL_CALL_INDIRECT ltac:(cbn; tauto)) as Hc2.
    pose proof (Hcalls COL_SEL_RETURN_CALL ltac:(cbn; tauto)) as Hc3.
    pose proof (Hcalls COL_SEL_RETURN_CALL_INDIRECT ltac:(cbn; tauto)) as Hc4.
    expand_row Hrow. rewrite Hc1, Hc2, Hc3, Hc4 in Hrow.
    pose proof (value_bound w COL_GUEST_ENTRY_ACTIVE). field_linear Hrow. }
  assert (HzAfter : w COL_LOCAL_ZERO_REMAINING_AFTER = 0).
  { circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_TURN_BOUNDARY, -1);
      (COL_LOCAL_ZERO_ACTIVE_BEFORE, -1)],
      [(COL_LOCAL_ZERO_REMAINING_AFTER, 1); (COL_LOCAL_ZERO_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hguest, Hturn, Hza, Hz in Hrow.
    pose proof (value_bound w COL_LOCAL_ZERO_REMAINING_AFTER). field_linear Hrow. }
  assert (HpAfter : w COL_PARAM_INIT_REMAINING_AFTER = 0).
  { circuit_row ([(COL_ONE, 1); (COL_GUEST_ENTRY_ACTIVE, -1); (COL_PARAM_INIT_ACTIVE_BEFORE, -1);
      (COL_TURN_BOUNDARY, -1)],
      [(COL_PARAM_INIT_REMAINING_AFTER, 1); (COL_PARAM_INIT_REMAINING_BEFORE, -1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone, Hguest, Hturn, Hpa, Hp in Hrow.
    pose proof (value_bound w COL_PARAM_INIT_REMAINING_AFTER). field_linear Hrow. }
  destruct (full_initialization_flags w Hprime Hfull) as [HzFlag HpFlag].
  rewrite HzAfter in HzFlag. rewrite HzAfter, HpAfter in HpFlag.
  cbn [afterBoundary boundaryZeroRemaining boundaryParamRemaining boundaryZeroActive boundaryParamActive].
  repeat split; assumption.
Qed.
