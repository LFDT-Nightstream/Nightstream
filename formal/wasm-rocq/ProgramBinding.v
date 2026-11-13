From Stdlib Require Import List ZArith Lia Bool.
Require Import InstructionHeader OpcodeCatalog R1CS Columns FullRelation Range Selectors ZeroTest CircuitFacts MemoryTrace MemoryCatalog.
Import ListNotations.
Open Scope Z_scope.

Lemma local_index_memory_read_only :
  readOnlyMemory MemoryCatalog.programLocalIndexId MemoryCatalog.memories = true.
Proof. vm_compute. reflexivity. Qed.

Definition localIndexPort : PortSpec :=
  mkPort [COL_PC_BEFORE] COL_LOCAL_INDEX Read (When COL_PROGRAM_LOCAL_INDEX_ACTIVE).

Lemma circuit_row_local_index_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PROGRAM_LOCAL_INDEX_ACTIVE = 1 ->
  initial MemoryCatalog.programLocalIndexId [w COL_PC_BEFORE] = Some (w COL_LOCAL_INDEX).
Proof.
  intros initial final w Hreplay Hactive.
  eapply replay_read_from_initial; [exact Hreplay | | ].
  - apply read_only_row. exact local_index_memory_read_only.
  - eapply port_event_in_row with
      (specification := mkMemory MemoryCatalog.programLocalIndexId Rom [localIndexPort])
      (port := localIndexPort).
    + apply nth_error_In with (n := MemoryCatalog.programLocalIndexId). vm_compute. reflexivity.
    + left. reflexivity.
    + unfold portEvent, active, localIndexPort. simpl. rewrite Hactive. reflexivity.
Qed.

Definition headerPort : PortSpec :=
  mkPort [COL_PC_BEFORE] COL_PROGRAM_HEADER Read (When COL_IS_PROGRAM_ROW).
Definition headerMemory : MemorySpec :=
  mkMemory programHeaderId Rom [headerPort].

Lemma header_memory_in_catalog : nth_error memories programHeaderId = Some headerMemory.
Proof. vm_compute. reflexivity. Qed.
Lemma header_memory_read_only : readOnlyMemory programHeaderId memories = true.
Proof. vm_compute. reflexivity. Qed.
Lemma pc_memory_read_only : readOnlyMemory pcRomId memories = true.
Proof. vm_compute. reflexivity. Qed.

Lemma header_event_in_row : forall (w : Witness), w COL_IS_PROGRAM_ROW = 1 ->
  In (ReadEvent programHeaderId [w COL_PC_BEFORE] (w COL_PROGRAM_HEADER)) (eventsForRow memories w).
Proof.
  intros w Hprogram.
  eapply port_event_in_row with (specification := headerMemory) (port := headerPort).
  - eapply nth_error_In. exact header_memory_in_catalog.
  - left. reflexivity.
  - unfold portEvent, active, headerMemory, headerPort. simpl. rewrite Hprogram. reflexivity.
Qed.

Lemma full_header_packing : forall w,
  fieldPrime -> FullRelation.satFull w ->
  w COL_PROGRAM_HEADER = w COL_OPCODE_CODE + 65536 * w COL_PC_EDGE_KIND.
Proof.
  intros w Hprime Hfull. pose proof (proj1 Hfull) as Hone.
  pose proof (value_bound w COL_PROGRAM_HEADER) as Hheader. circuit_range COL_OPCODE_CODE Hcode.
  circuit_range COL_PC_EDGE_KIND Hkind.
  circuit_row ([(COL_PROGRAM_HEADER, 1); (COL_OPCODE_CODE, -1); (COL_PC_EDGE_KIND, -65536)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow.
Qed.

Lemma full_host_slot_descriptor : forall w,
  fieldPrime -> FullRelation.satFull w ->
  w COL_HOST_EVENT_SLOT_DESCRIPTOR = w COL_HOST_EVENT_SLOT_KIND +
    32 * w COL_HOST_EVENT_SLOT_ARG + 8192 * w COL_HOST_EVENT_SLOT_VARIANT /\
  0 <= w COL_HOST_EVENT_SLOT_KIND < 32 /\
  0 <= w COL_HOST_EVENT_SLOT_ARG < 256 /\ 0 <= w COL_HOST_EVENT_SLOT_VARIANT < 16.
Proof.
  intros w Hprime Hfull. pose proof (proj1 Hfull) as Hone.
  pose proof (value_bound w COL_HOST_EVENT_SLOT_DESCRIPTOR) as Hpacked.
  circuit_range COL_HOST_EVENT_SLOT_KIND Hkind.
  circuit_range COL_HOST_EVENT_SLOT_ARG Harg.
  circuit_range COL_HOST_EVENT_SLOT_VARIANT Hvariant.
  split; [ | lia].
  circuit_row ([(COL_HOST_EVENT_SLOT_DESCRIPTOR, 1); (COL_HOST_EVENT_SLOT_KIND, -1);
    (COL_HOST_EVENT_SLOT_ARG, -32); (COL_HOST_EVENT_SLOT_VARIANT, -8192)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow.
Qed.

Lemma full_host_context_refs : forall w,
  fieldPrime -> FullRelation.satFull w ->
  w COL_HOST_CONTEXT_REFS = w COL_HOST_CONTEXT_EXPORT + 65536 * w COL_HOST_CONTEXT_IMPORT /\
  0 <= w COL_HOST_CONTEXT_EXPORT < 65536 /\
  0 <= w COL_HOST_CONTEXT_IMPORT < 65536 /\
  0 <= w COL_HOST_CONTEXT_REFS < 4294967296 /\
  w COL_HOST_CONTEXT_REFS mod 65536 = w COL_HOST_CONTEXT_EXPORT /\
  w COL_HOST_CONTEXT_REFS / 65536 = w COL_HOST_CONTEXT_IMPORT.
Proof.
  intros w Hprime Hfull. pose proof (proj1 Hfull) as Hone.
  pose proof (value_bound w COL_HOST_CONTEXT_REFS) as Hpacked.
  circuit_range COL_HOST_CONTEXT_EXPORT Hexport.
  circuit_range COL_HOST_CONTEXT_IMPORT Himport.
  assert (Hpacking : w COL_HOST_CONTEXT_REFS =
    w COL_HOST_CONTEXT_EXPORT + 65536 * w COL_HOST_CONTEXT_IMPORT).
  { circuit_row ([(COL_HOST_CONTEXT_REFS, 1); (COL_HOST_CONTEXT_EXPORT, -1);
      (COL_HOST_CONTEXT_IMPORT, -65536)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow. }
  split; [exact Hpacking | ].
  split; [lia | ]. split; [lia | ]. split; [lia | ].
  rewrite Hpacking. split; [apply packed_low16 | apply packed_high16]; lia.
Qed.

Lemma full_host_context_args_depth : forall w,
  fieldPrime -> FullRelation.satFull w ->
  w COL_HOST_CONTEXT_ARGS_DEPTH = w COL_HOST_CONTEXT_ARGS_BASE + 65536 * w COL_HOST_CONTEXT_PARENT_DEPTH /\
  0 <= w COL_HOST_CONTEXT_ARGS_BASE < 65536 /\
  0 <= w COL_HOST_CONTEXT_PARENT_DEPTH < 65536 /\
  0 <= w COL_HOST_CONTEXT_ARGS_DEPTH < 4294967296 /\
  w COL_HOST_CONTEXT_ARGS_DEPTH mod 65536 = w COL_HOST_CONTEXT_ARGS_BASE /\
  w COL_HOST_CONTEXT_ARGS_DEPTH / 65536 = w COL_HOST_CONTEXT_PARENT_DEPTH.
Proof.
  intros w Hprime Hfull. pose proof (proj1 Hfull) as Hone.
  pose proof (value_bound w COL_HOST_CONTEXT_ARGS_DEPTH) as Hpacked.
  circuit_range COL_HOST_CONTEXT_ARGS_BASE Hargs.
  circuit_range COL_HOST_CONTEXT_PARENT_DEPTH Hdepth.
  assert (Hpacking : w COL_HOST_CONTEXT_ARGS_DEPTH =
    w COL_HOST_CONTEXT_ARGS_BASE + 65536 * w COL_HOST_CONTEXT_PARENT_DEPTH).
  { circuit_row ([(COL_HOST_CONTEXT_ARGS_DEPTH, 1); (COL_HOST_CONTEXT_ARGS_BASE, -1);
      (COL_HOST_CONTEXT_PARENT_DEPTH, -65536)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow. }
  split; [exact Hpacking | ].
  split; [lia | ]. split; [lia | ]. split; [lia | ].
  rewrite Hpacking. split; [apply packed_low16 | apply packed_high16]; lia.
Qed.

Lemma full_opcode_u16 : forall w,
  fieldPrime -> FullRelation.satFull w -> w COL_IS_PROGRAM_ROW = 1 -> 0 <= w COL_OPCODE_CODE < 65536.
Proof.
  intros w Hprime Hfull Hprogram.
  destruct (full_opcode_decode w Hprime Hfull Hprogram) as [chosen [code [Hin [_ ->]]]].
  assert (Hcheck : forallb (fun entry => (0 <=? snd entry) && (snd entry <? 65536)) selectorCodes = true)
    by (vm_compute; reflexivity).
  apply forallb_forall with (x := (chosen, code)) in Hcheck; [ | exact Hin].
  apply Bool.andb_true_iff in Hcheck as [Hlo Hhi]. cbn in Hlo, Hhi.
  apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi. lia.
Qed.

Lemma circuit_row_header_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow memories w) final -> w COL_IS_PROGRAM_ROW = 1 ->
  initial programHeaderId [w COL_PC_BEFORE] = Some (w COL_PROGRAM_HEADER).
Proof.
  intros initial final w Hreplay Hprogram.
  eapply replay_read_from_initial; [exact Hreplay | apply read_only_row, header_memory_read_only | ].
  now apply header_event_in_row.
Qed.

Lemma circuit_row_opcode_fetch : forall initial final (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  Replay initial (eventsForRow memories w) final -> w COL_IS_PROGRAM_ROW = 1 ->
  opcodeAt initial [w COL_PC_BEFORE] = Some (w COL_OPCODE_CODE).
Proof.
  intros initial final w Hprime Hfull Hreplay Hprogram. unfold opcodeAt.
  rewrite (circuit_row_header_fetch _ _ _ Hreplay Hprogram), (full_header_packing w Hprime Hfull).
  cbn [option_map]. rewrite packed_low16; [reflexivity | now apply full_opcode_u16].
Qed.

Lemma circuit_row_edge_kind_fetch : forall initial final (w : Witness),
  fieldPrime -> FullRelation.satFull w ->
  Replay initial (eventsForRow memories w) final -> w COL_IS_PROGRAM_ROW = 1 ->
  edgeKindAt initial [w COL_PC_BEFORE] = Some (w COL_PC_EDGE_KIND).
Proof.
  intros initial final w Hprime Hfull Hreplay Hprogram. unfold edgeKindAt.
  rewrite (circuit_row_header_fetch _ _ _ Hreplay Hprogram), (full_header_packing w Hprime Hfull).
  cbn [option_map]. rewrite packed_high16; [reflexivity | now apply full_opcode_u16].
Qed.

Definition fallthroughFlagRow : nat := Eval vm_compute in
  match findRow (fun row => match row with
    | ((column, _) :: _, [(one, 1)], []) => Nat.eqb column COL_PC_FALLTHROUGH && Nat.eqb one COL_ONE
    | _ => false end) FullRelation.rows with Some i => i | None => 0%nat end.
Definition fallthroughFlagShape : Row := Eval vm_compute in
  List.nth fallthroughFlagRow FullRelation.rows ([], [], []).
Definition selectedFallthroughRow (code : Z) : Row :=
  (if existsb (Z.eqb code) fallthroughCodes then [(COL_PC_FALLTHROUGH, 1); (COL_ONE, -1)]
   else [(COL_PC_FALLTHROUGH, 1)], [(COL_ONE, 1)], []).

Lemma full_fallthrough_flag : forall w chosen code,
  fieldPrime -> FullRelation.satFull w -> In (chosen, code) selectorCodes -> w chosen = 1 ->
  w COL_PC_FALLTHROUGH = if existsb (Z.eqb code) fallthroughCodes then 1 else 0.
Proof.
  intros w chosen code Hprime Hfull Hin Hchosen.
  assert (Hcol : In chosen selectorColumns).
  { rewrite <- opcode_selector_columns. exact (in_map fst _ _ Hin). }
  pose proof (full_selected_row w chosen fallthroughFlagRow fallthroughFlagShape Hprime Hfull Hcol Hchosen
    ltac:(vm_compute; reflexivity)) as Hrow.
  assert (Hcheck : forallb (fun entry => rowsEqual (selectedRow (fst entry) fallthroughFlagShape)
    (selectedFallthroughRow (snd entry))) selectorCodes = true) by (vm_compute; reflexivity).
  apply forallb_forall with (x := (chosen, code)) in Hcheck; [ | exact Hin].
  apply rows_equal_sound in Hcheck. cbn [fst snd] in Hcheck. rewrite Hcheck in Hrow.
  pose proof (proj1 Hfull) as Hone. circuit_boolean COL_PC_FALLTHROUGH Hbit.
  unfold selectedFallthroughRow in Hrow. destruct (existsb (Z.eqb code) fallthroughCodes);
    expand_row Hrow; normalize_columns; rewrite Hone in Hrow; field_linear Hrow.
Qed.

Lemma full_fallthrough_pc : forall w,
  fieldPrime -> FullRelation.satFull w -> w COL_PC_FALLTHROUGH = 1 ->
  w COL_PC_AFTER = w COL_PC_BEFORE + 1.
Proof.
  intros w Hprime Hfull Hfall. pose proof (proj1 Hfull) as Hone.
  circuit_range COL_PC_BEFORE Hb. circuit_range COL_PC_AFTER Ha.
  circuit_row ([(COL_PC_FALLTHROUGH, 1)], [(COL_PC_AFTER, 1); (COL_PC_BEFORE, -1); (COL_ONE, -1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hfall, Hone in Hrow. field_linear Hrow.
Qed.

(* The key of an edge: an instruction's choice c is at the c-th address after
   its own, so the pc and slide ROMs have one address column. *)
Lemma full_edge_addr : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_EDGE_ADDR = w COL_PC_BEFORE + w COL_CONTROL_CHOICE.
Proof.
  intros w Hprime Hfull.
  pose proof (proj1 Hfull) as Hone.
  circuit_range COL_EDGE_ADDR Hedge.
  circuit_range COL_PC_BEFORE Hpc.
  circuit_range COL_CONTROL_CHOICE Hchoice.
  circuit_row ([(COL_EDGE_ADDR, 1); (COL_PC_BEFORE, -1); (COL_CONTROL_CHOICE, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Hrow.
  expand_row Hrow. normalize_columns. rewrite Hone in Hrow. field_linear Hrow.
Qed.

Definition nextPcPort : PortSpec :=
  mkPort [COL_EDGE_ADDR] COL_PC_AFTER Read (When COL_PC_ROM_ACTIVE).

Lemma circuit_row_next_pc_fetch : forall initial final (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_PC_ROM_ACTIVE = 1 ->
  initial MemoryCatalog.pcRomId [w COL_PC_BEFORE + w COL_CONTROL_CHOICE] = Some (w COL_PC_AFTER).
Proof.
  intros initial final w Hprime Hfull Hreplay Hactive.
  rewrite <- (full_edge_addr w Hprime Hfull).
  eapply replay_read_from_initial; [exact Hreplay | | ].
  - apply read_only_row. exact pc_memory_read_only.
  - eapply port_event_in_row with
      (specification := nth MemoryCatalog.pcRomId MemoryCatalog.memories (mkMemory 0%nat Rom []))
      (port := nextPcPort).
    + apply nth_In. vm_compute. lia.
    + change (In nextPcPort (ports (nth MemoryCatalog.pcRomId MemoryCatalog.memories (mkMemory 0%nat Rom [])))).
      left. reflexivity.
    + unfold portEvent, active, nextPcPort. simpl. rewrite Hactive. reflexivity.
Qed.

(* A slide reads its countdown and distance from the taken edge's static slide. *)
Definition slideRemainingPort : PortSpec :=
  mkPort [COL_EDGE_ADDR] COL_FRAME_MOVE_REMAINING_AFTER Read (When COL_FRAME_SLIDE_ACTIVE).
Definition slideDistancePort : PortSpec :=
  mkPort [COL_EDGE_ADDR] COL_FRAME_MOVE_DISTANCE_AFTER Read (When COL_FRAME_SLIDE_ACTIVE).

Lemma circuit_row_slide_fetch : forall initial final (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_FRAME_SLIDE_ACTIVE = 1 ->
  initial MemoryCatalog.frameSlideRemainingId [w COL_PC_BEFORE + w COL_CONTROL_CHOICE] =
    Some (w COL_FRAME_MOVE_REMAINING_AFTER) /\
  initial MemoryCatalog.frameSlideDistanceId [w COL_PC_BEFORE + w COL_CONTROL_CHOICE] =
    Some (w COL_FRAME_MOVE_DISTANCE_AFTER).
Proof.
  intros initial final w Hprime Hfull Hreplay Hactive. rewrite <- (full_edge_addr w Hprime Hfull). split;
  (eapply replay_read_from_initial; [exact Hreplay | apply read_only_row; vm_compute; reflexivity | ]).
  - eapply port_event_in_row with
      (specification := nth MemoryCatalog.frameSlideRemainingId MemoryCatalog.memories (mkMemory 0%nat Rom []))
      (port := slideRemainingPort).
    + apply nth_In. vm_compute. lia.
    + change (In slideRemainingPort
        (ports (nth MemoryCatalog.frameSlideRemainingId MemoryCatalog.memories (mkMemory 0%nat Rom [])))).
      left. reflexivity.
    + unfold portEvent, active, slideRemainingPort. simpl. rewrite Hactive. reflexivity.
  - eapply port_event_in_row with
      (specification := nth MemoryCatalog.frameSlideDistanceId MemoryCatalog.memories (mkMemory 0%nat Rom []))
      (port := slideDistancePort).
    + apply nth_In. vm_compute. lia.
    + change (In slideDistancePort
        (ports (nth MemoryCatalog.frameSlideDistanceId MemoryCatalog.memories (mkMemory 0%nat Rom [])))).
      left. reflexivity.
    + unfold portEvent, active, slideDistancePort. simpl. rewrite Hactive. reflexivity.
Qed.

Lemma static_pc_gate_rows :
  Range.inFull (ZeroTest.inverseRow COL_PC_EDGE_KIND COL_PC_EDGE_KIND_INV COL_PC_EDGE_KIND_IS_STATIC) = true /\
  Range.inFull (ZeroTest.zeroRow COL_PC_EDGE_KIND COL_PC_EDGE_KIND_IS_STATIC) = true /\
  Range.inFull ([(COL_IS_PROGRAM_ROW, 1); (COL_SEL_CALL, -1); (COL_SEL_RETURN_CALL, -1)], [(COL_PC_EDGE_KIND_IS_STATIC, 1)], [(COL_PC_ROM_ACTIVE, 1); (COL_PC_FALLTHROUGH, 1)]) = true.
Proof. vm_compute. repeat split; reflexivity. Qed.

Lemma full_static_pc_rom_active : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  In chosen selectorColumns -> ~ In chosen [COL_SEL_CALL; COL_SEL_RETURN_CALL] -> w chosen = 1 ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_EDGE_KIND = 0 -> w COL_PC_FALLTHROUGH = 0 -> w COL_PC_ROM_ACTIVE = 1.
Proof.
  intros w chosen Hprime Hfull Hin HnotCall Hchosen Hprogram Hkind Hfall.
  assert (Hcall : w COL_SEL_CALL = 0).
  { eapply (full_opcode_selector_exclusive w chosen COL_SEL_CALL); try eassumption;
      [vm_compute; tauto | cbn in HnotCall; intuition congruence]. }
  assert (Htail : w COL_SEL_RETURN_CALL = 0).
  { eapply (full_opcode_selector_exclusive w chosen COL_SEL_RETURN_CALL); try eassumption;
      [vm_compute; tauto | cbn in HnotCall; intuition congruence]. }
  destruct Hfull as [Hone Hall].
  destruct static_pc_gate_rows as [Hinv [Hzero Hgate]].
  pose proof (ZeroTest.zero_test_value w COL_PC_EDGE_KIND COL_PC_EDGE_KIND_INV
    COL_PC_EDGE_KIND_IS_STATIC Hprime Hone
    (Range.in_full_sat w _ Hall Hinv) (Range.in_full_sat w _ Hall Hzero)) as Hstatic.
  rewrite Hkind in Hstatic. cbn in Hstatic.
  pose proof (Range.in_full_sat w _ Hall Hgate) as Hsat.
  unfold sat in Hsat. cbn [eval fold_right] in Hsat.
  rewrite Hprogram, Hstatic, Hfall, Hcall, Htail in Hsat.
  pose proof (value_bound w COL_PC_ROM_ACTIVE) as Hbound.
  apply Z.mod_divide in Hsat; [ | unfold modulus; lia].
  destruct Hsat as [k Hk]. unfold modulus in *; lia.
Qed.

(* A br_table reads its arm count from the program's table-length ROM. *)
Definition brTableLenPort : PortSpec :=
  mkPort [COL_PC_BEFORE] COL_BR_TABLE_LEN Read (When COL_SEL_BR_TABLE).

Lemma circuit_row_br_table_len_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_SEL_BR_TABLE = 1 ->
  initial MemoryCatalog.programBrTableLenId [w COL_PC_BEFORE] = Some (w COL_BR_TABLE_LEN).
Proof.
  intros initial final w Hreplay Hactive.
  eapply replay_read_from_initial; [exact Hreplay | apply read_only_row; vm_compute; reflexivity | ].
  eapply port_event_in_row with
    (specification := nth MemoryCatalog.programBrTableLenId MemoryCatalog.memories (mkMemory 0%nat Rom []))
    (port := brTableLenPort).
  - apply nth_In. vm_compute. lia.
  - change (In brTableLenPort
      (ports (nth MemoryCatalog.programBrTableLenId MemoryCatalog.memories (mkMemory 0%nat Rom [])))).
    left. reflexivity.
  - unfold portEvent, active, brTableLenPort. simpl. rewrite Hactive. reflexivity.
Qed.
