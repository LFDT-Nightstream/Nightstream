(* Terminal program rows: activation flags, output capture, and the return's ROM
   distance. These are consequences of the exported constraints and ports. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts CallContext CallInitialization TrapFacts
  MemoryTrace MemoryCatalog ProgramBinding OpcodeCatalog.
Import ListNotations.
Open Scope Z_scope.

Lemma program_starts_active : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  w COL_ACTIVATION_FINISHED_BEFORE = 0 /\ w COL_TRAPPED_BEFORE = 0 /\ w COL_HOST_CLEAR_ACTIVATION_FINISHED = 0.
Proof.
  intros w Hprime Hfull Hprogram.
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  split; [ | split; [ | exact (full_no_activation_clear w Hfull Hturn Hresume)]].
  - circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow.
  - circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow.
Qed.

Lemma full_clean_finish : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  w COL_ACTIVATION_FINISHED_AFTER = 1 -> w COL_TRAPPED_AFTER = 0 ->
  w COL_HOST_EVENT_EXIT_LATCH = 1 /\
  w COL_CALL_STACK_DEPTH_BEFORE = w COL_HOST_FRAME_DEPTH_BEFORE /\
  w COL_OUTPUT_CAPTURED = w COL_SP_BEFORE - w COL_LOCALS_FBP_BEFORE - w COL_EXIT_SLIDE_DISTANCE /\
  w COL_SP_AFTER = w COL_LOCALS_FBP_BEFORE.
Proof.
  intros w Hprime Hfull Hprogram Hfinish Htrap.
  pose proof (proj1 Hfull) as Hone.
  destruct (program_starts_active w Hprime Hfull Hprogram) as [Hfb [Htb Hclear]].
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hstate.
  expand_row Hstate. normalize_columns. rewrite Hone, Hfinish, Hfb, Htrap, Htb, Hclear in Hstate.
  split.
  { pose proof (value_bound w COL_HOST_EVENT_EXIT_LATCH). field_linear Hstate. }
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_TRAPPED_AFTER, -1)],
    [(COL_CALL_STACK_DEPTH_BEFORE, 1); (COL_HOST_FRAME_DEPTH_BEFORE, -1)], ([] : SparseRow)) Hdepth.
  expand_row Hdepth. normalize_columns. rewrite Hfinish, Htrap in Hdepth.
  split.
  { pose proof (value_bound w COL_CALL_STACK_DEPTH_BEFORE). pose proof (value_bound w COL_HOST_FRAME_DEPTH_BEFORE).
    field_linear Hdepth. }
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1)],
    [(COL_SP_BEFORE, 1); (COL_LOCALS_FBP_BEFORE, -1); (COL_EXIT_SLIDE_DISTANCE, -1); (COL_OUTPUT_CAPTURED, -1)],
    ([] : SparseRow)) Hcapture.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1)],
    [(COL_SP_AFTER, 1); (COL_LOCALS_FBP_BEFORE, -1)], ([] : SparseRow)) Hsp.
  expand_row Hcapture. expand_row Hsp. normalize_columns.
  rewrite Hfinish, Hfb, Htrap, Htb, Hclear in Hcapture, Hsp.
  assert (Hspb : 0 <= w COL_SP_BEFORE < 4294967296)
    by (apply (full_u32 w COL_SP_BEFORE Hprime Hfull); vm_compute; reflexivity).
  assert (Hfbp : 0 <= w COL_LOCALS_FBP_BEFORE < 4294967296)
    by (apply (full_u32 w COL_LOCALS_FBP_BEFORE Hprime Hfull); vm_compute; reflexivity).
  assert (Hdist : 0 <= w COL_EXIT_SLIDE_DISTANCE < 4294967296)
    by (apply (full_u32 w COL_EXIT_SLIDE_DISTANCE Hprime Hfull); vm_compute; reflexivity).
  circuit_boolean COL_OUTPUT_CAPTURED Hcap.
  split; [field_linear Hcapture | ]. pose proof (value_bound w COL_SP_AFTER). field_linear Hsp.
Qed.

Definition exitDistancePort := mkPort [COL_PC_BEFORE; COL_CONTROL_CHOICE] COL_EXIT_SLIDE_DISTANCE
  Read (When COL_HOST_EVENT_EXIT_LATCH).

Lemma circuit_row_exit_distance_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow memories w) final -> w COL_HOST_EVENT_EXIT_LATCH = 1 ->
  initial frameSlideDistanceId [w COL_PC_BEFORE; w COL_CONTROL_CHOICE] = Some (w COL_EXIT_SLIDE_DISTANCE).
Proof.
  intros initial final w Hreplay Hactive.
  eapply replay_read_from_initial; [exact Hreplay | apply read_only_row; vm_compute; reflexivity | ].
  eapply port_event_in_row with
    (specification := nth frameSlideDistanceId memories (mkMemory 0%nat Rom [])) (port := exitDistancePort).
  - apply nth_In. vm_compute. lia.
  - change (In exitDistancePort (ports (nth frameSlideDistanceId memories (mkMemory 0%nat Rom [])))).
    right; left; reflexivity.
  - unfold portEvent, active, exitDistancePort. simpl. rewrite Hactive. reflexivity.
Qed.

Lemma captured_stack_reads : forall initial final (w : Witness),
  Replay initial (eventsForRow memories w) final -> w COL_OUTPUT_CAPTURED = 1 ->
  initial stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) /\
  initial stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)] =
    Some (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)).
Proof.
  intros initial final w Hreplay Hcapture.
  assert (Hprefix : exists suffix, eventsForRow memories w =
    ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
      (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) ::
      ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
        (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)) :: suffix).
  { change (exists suffix,
      flat_map (fun p => match portEvent stackId w p with Some e => [e] | None => [] end)
        (ports (hd (mkMemory 0%nat Ram []) memories)) ++ eventsForRow (tl memories) w =
      ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)]
        (w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)) ::
      ReadEvent stackId [w (List.nth 0%nat COL_STACK_READ_ADDR_HI 0%nat)]
        (w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)) :: suffix).
    cbn [hd memories ports flat_map portEvent active activation addressColumns valueColumn portKind map option_map].
    unfold portEvent, active. cbn [activation addressColumns valueColumn portKind map option_map].
    normalize_columns. rewrite Hcapture.
    destruct (w (List.nth 0%nat COL_STACK_READ_ACTIVE 0%nat) =? 1) eqn:Hr;
      cbn [COL_STACK_READ_ACTIVE List.nth] in Hr; rewrite Hr; eexists; reflexivity. }

  destruct Hprefix as [suffix Hprefix]. rewrite Hprefix in Hreplay.
  inversion Hreplay; subst.
  match goal with H : EventStep _ _ _ |- _ => inversion H; subst end.
  split; [assumption | ].
  match goal with H : Replay _ (ReadEvent _ _ _ :: _) _ |- _ => inversion H; subst end.
  match goal with H : EventStep _ _ _ |- _ => inversion H; subst; assumption end.
Qed.

Lemma full_capture_value : forall (w : Witness),
  FullRelation.satFull w -> w COL_OUTPUT_CAPTURED = 1 ->
  w COL_OUTPUT_VALUE_LO_AFTER = w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat).
Proof.
  intros w Hfull Hcapture.
  circuit_row ([(COL_OUTPUT_CAPTURED, 1)],
    [(COL_OUTPUT_VALUE_LO_AFTER, 1); (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hcapture in Hrow.
  pose proof (value_bound w COL_OUTPUT_VALUE_LO_AFTER).
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat)). field_linear Hrow.
Qed.

Lemma full_capture_value_hi : forall (w : Witness),
  FullRelation.satFull w -> w COL_OUTPUT_CAPTURED = 1 ->
  w COL_OUTPUT_VALUE_HI_AFTER = w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat).
Proof.
  intros w Hfull Hcapture.
  circuit_row ([(COL_OUTPUT_CAPTURED, 1)],
    [(COL_OUTPUT_VALUE_HI_AFTER, 1); (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hcapture in Hrow.
  pose proof (value_bound w COL_OUTPUT_VALUE_HI_AFTER).
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_VALUE_HI 0%nat)). field_linear Hrow.
Qed.

Lemma full_capture_address : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_OUTPUT_CAPTURED = 1 -> 1 <= w COL_SP_BEFORE ->
  w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat) = 2 * w COL_SP_BEFORE - 2.
Proof.
  intros w Hprime Hfull Hcapture Hsp0.
  pose proof (proj1 Hfull) as Hone.
  circuit_row ([(COL_OUTPUT_CAPTURED, 1)],
    [(List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat, 1); (COL_SP_BEFORE, -2); (COL_ONE, 2)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hcapture, Hone in Hrow.
  assert (Hsp : 0 <= w COL_SP_BEFORE < 4294967296)
    by (apply (full_u32 w COL_SP_BEFORE Hprime Hfull); vm_compute; reflexivity).
  pose proof (value_bound w (List.nth 0%nat COL_STACK_READ_ADDR_LO 0%nat)). field_linear Hrow.
Qed.


Lemma opcode_code_unique : NoDup (map snd selectorCodes).
Proof.
  replace (map snd selectorCodes) with (nodup Z.eq_dec (map snd selectorCodes)) by (vm_compute; reflexivity).
  apply NoDup_nodup.
Qed.

Lemma opcode_member_unique : forall entries chosen actual code,
  NoDup (map (@snd Column Z) entries) -> In (chosen, code) entries -> In (actual, code) entries -> chosen = actual.
Proof.
  induction entries as [ | [column opcode] entries IH]; intros chosen actual code Hunique Hin Hactual;
    [contradiction | ].
  inversion Hunique as [ | ? ? Hnot Htail]; subst.
  destruct Hin as [Heq | Hin], Hactual as [Heq' | Hactual].
  - congruence.
  - injection Heq as <- <-. exfalso. apply Hnot. exact (in_map (@snd Column Z) _ _ Hactual).
  - injection Heq' as <- <-. exfalso. apply Hnot. exact (in_map (@snd Column Z) _ _ Hin).
  - eapply IH; eassumption.
Qed.

Lemma full_opcode_selects_member : forall (w : Witness) chosen code,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 ->
  In (chosen, code) selectorCodes -> w COL_OPCODE_CODE = code -> w chosen = 1.
Proof.
  intros w chosen code Hprime Hfull Hprogram Hin Hcode.
  destruct (full_opcode_decode w Hprime Hfull Hprogram) as [actual [v [Hactual [Hset Hv]]]].
  assert (HeqCode : v = code) by congruence. rewrite HeqCode in Hactual.
  pose proof (opcode_member_unique _ _ _ _ opcode_code_unique Hin Hactual) as Heq.
  rewrite Heq. exact Hset.
Qed.

Lemma full_nontrapping_finish_edge : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> In chosen nonTrappingSelectors -> w chosen = 1 ->
  w COL_ACTIVATION_FINISHED_AFTER = 1 ->
  w COL_PC_EDGE_KIND = 1 + w COL_CALL_STACK_POP_PRESENT.
Proof.
  intros w chosen Hprime Hfull HchosenIn Hchosen Hfinish.
  assert (Hin : In chosen selectorColumns) by (vm_compute in HchosenIn; vm_compute; intuition).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen) as Hprogram.
  destruct (program_starts_active w Hprime Hfull Hprogram) as [Hfb [Htb Hclear]].
  pose proof (full_nontrapping_untrapped w chosen Hprime Hfull HchosenIn Hchosen) as HtrapAfter.
  assert (Hzero : w COL_SEL_UNREACHABLE = 0 /\ w COL_SEL_CALL_INDIRECT = 0 /\ w COL_SEL_RETURN_CALL_INDIRECT = 0).
  { repeat split; eapply full_opcode_selector_exclusive; try exact Hchosen; try exact Hin; try eassumption;
      try (vm_compute; tauto); vm_compute in HchosenIn |- *; intuition congruence. }
  destruct Hzero as [Hunreachable [Hindirect Htail]].
  circuit_row ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1);
    (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Htrap.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1);
    (COL_CALL_STACK_POP_PRESENT, 1); (COL_PC_EDGE_KIND, -1); (COL_SEL_CALL_INDIRECT, 2);
    (COL_SEL_RETURN_CALL_INDIRECT, 2); (COL_SEL_UNREACHABLE, 2); (COL_DIV_TRAP, -1);
    (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hedge.
  expand_row Htrap. expand_row Hedge.
  circuit_boolean COL_DIV_TRAP Hdiv. circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci.
  circuit_boolean COL_MEM_OOB Hmem. circuit_boolean COL_TABLE_OOB Htable. circuit_boolean COL_CALL_STACK_POP_PRESENT Hpop.
  assert (Hkind : 0 <= w COL_PC_EDGE_KIND < 4294967296)
    by (apply (full_u32 w COL_PC_EDGE_KIND Hprime Hfull); vm_compute; reflexivity).
  normalize_columns. rewrite Hone, HtrapAfter, Htb, Hunreachable in Htrap.
  rewrite Hone, Hfinish, Hfb, Hclear, Hindirect, Htail, Hunreachable in Hedge.
  apply Z.mod_divide in Htrap; [ | unfold modulus; lia]. destruct Htrap as [k Htrap].
  apply Z.mod_divide in Hedge; [ | unfold modulus; lia]. destruct Hedge as [l Hedge].
  unfold modulus in *; lia.
Qed.
