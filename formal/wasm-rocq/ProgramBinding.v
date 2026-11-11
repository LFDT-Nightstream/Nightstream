(* Active program rows fetch from the initial opcode ROM through exported ports. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors ZeroTest CircuitFacts MemoryTrace MemoryCatalog.
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

Definition opcodePort : PortSpec :=
  mkPort [COL_PC_BEFORE] COL_OPCODE_CODE Read (When COL_IS_PROGRAM_ROW).

Definition opcodeMemory : MemorySpec :=
  mkMemory MemoryCatalog.programOpcodeId Rom [opcodePort].

Lemma opcode_memory_in_catalog :
  nth_error MemoryCatalog.memories MemoryCatalog.programOpcodeId = Some opcodeMemory.
Proof. vm_compute. reflexivity. Qed.

Lemma opcode_memory_read_only :
  readOnlyMemory MemoryCatalog.programOpcodeId MemoryCatalog.memories = true.
Proof. vm_compute. reflexivity. Qed.

Lemma edge_kind_memory_read_only :
  readOnlyMemory MemoryCatalog.pcEdgeKindId MemoryCatalog.memories = true.
Proof. vm_compute. reflexivity. Qed.

Lemma pc_memory_read_only :
  readOnlyMemory MemoryCatalog.pcRomId MemoryCatalog.memories = true.
Proof. vm_compute. reflexivity. Qed.

Lemma opcode_event_in_row : forall (w : Witness),
  w COL_IS_PROGRAM_ROW = 1 ->
  In (ReadEvent MemoryCatalog.programOpcodeId [w COL_PC_BEFORE] (w COL_OPCODE_CODE))
    (eventsForRow MemoryCatalog.memories w).
Proof.
  intros w Hprogram.
  eapply port_event_in_row with (specification := opcodeMemory) (port := opcodePort).
  - eapply nth_error_In. exact opcode_memory_in_catalog.
  - left. reflexivity.
  - unfold portEvent, active, opcodeMemory, opcodePort. simpl.
    rewrite Hprogram. reflexivity.
Qed.

Lemma circuit_row_opcode_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_IS_PROGRAM_ROW = 1 ->
  initial MemoryCatalog.programOpcodeId [w COL_PC_BEFORE] = Some (w COL_OPCODE_CODE).
Proof.
  intros initial final w Hreplay Hprogram.
  eapply replay_read_from_initial; [exact Hreplay | | ].
  - apply read_only_row. exact opcode_memory_read_only.
  - apply opcode_event_in_row. exact Hprogram.
Qed.

Definition edgeKindPort : PortSpec :=
  mkPort [COL_PC_BEFORE] COL_PC_EDGE_KIND Read (When COL_IS_PROGRAM_ROW).

Lemma circuit_row_edge_kind_fetch : forall initial final (w : Witness),
  Replay initial (eventsForRow MemoryCatalog.memories w) final ->
  w COL_IS_PROGRAM_ROW = 1 ->
  initial MemoryCatalog.pcEdgeKindId [w COL_PC_BEFORE] = Some (w COL_PC_EDGE_KIND).
Proof.
  intros initial final w Hreplay Hprogram.
  eapply replay_read_from_initial; [exact Hreplay | | ].
  - apply read_only_row. exact edge_kind_memory_read_only.
  - eapply port_event_in_row with
      (specification := mkMemory MemoryCatalog.pcEdgeKindId Rom [edgeKindPort])
      (port := edgeKindPort).
    + apply nth_error_In with (n := MemoryCatalog.pcEdgeKindId). vm_compute. reflexivity.
    + left. reflexivity.
    + unfold portEvent, active, edgeKindPort. simpl. rewrite Hprogram. reflexivity.
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
  Range.inFull ([(COL_IS_PROGRAM_ROW, 1)], [(COL_PC_EDGE_KIND_IS_STATIC, 1)], [(COL_PC_ROM_ACTIVE, 1)]) = true.
Proof. vm_compute. repeat split; reflexivity. Qed.

Lemma full_static_pc_rom_active : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_IS_PROGRAM_ROW = 1 -> w COL_PC_EDGE_KIND = 0 -> w COL_PC_ROM_ACTIVE = 1.
Proof.
  intros w Hprime [Hone Hall] Hprogram Hkind.
  destruct static_pc_gate_rows as [Hinv [Hzero Hgate]].
  pose proof (ZeroTest.zero_test_value w COL_PC_EDGE_KIND COL_PC_EDGE_KIND_INV
    COL_PC_EDGE_KIND_IS_STATIC Hprime Hone
    (Range.in_full_sat w _ Hall Hinv) (Range.in_full_sat w _ Hall Hzero)) as Hstatic.
  rewrite Hkind in Hstatic. cbn in Hstatic.
  pose proof (Range.in_full_sat w _ Hall Hgate) as Hsat.
  unfold sat in Hsat. cbn [eval fold_right] in Hsat.
  rewrite Hprogram, Hstatic in Hsat.
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
