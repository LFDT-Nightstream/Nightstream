(* The store's tables in the circuit's table memories. The fragment has no table
   writes, so a run keeps them: the encoder never places table.set's opcode, and
   only table.set's selector gates a table write. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog Wasm2
  Selectors ProgramBinding TraceRefinement StraightLineSpec CodeLayout ProgramSpec.
Import ListNotations.
Open Scope Z_scope.

(* A reference cell: 0 for null, a + 1 for the function at store address a. Host
   references have no cell; the circuit only holds function tables. *)
Definition encodeRef (r : ref) : option Z :=
  match r with
  | ref_REF_NULL _ => Some 0
  | REF_FUNC_ADDR a => Some (Z.of_N a + 1)
  | REF_HOST_ADDR _ => None
  end.

(* The table at store address a: its size at [a], its references at [a; i]. *)
Definition tablesLoaded (s : store) (memory : MemoryState) : Prop :=
  forall a t, nth_error (store_TABLES s) a = Some t ->
    memory tableSizeId [Z.of_nat a] = Some (Z.of_nat (length (REFS t))) /\
    forall i r, nth_error (REFS t) i = Some r -> memory tableElementId [Z.of_nat a; Z.of_nat i] = encodeRef r.

Lemma encoded_opcode_not_table_set : forall i, encodedOpcode i <> Some CODE_TABLE_SET.
Proof.
  intros i. unfold encodedOpcode.
  repeat match goal with |- context [match ?x with _ => _ end] => destruct x end; discriminate.
Qed.

Lemma opcode_rom_not_table_set : forall program pc,
  romValue (encodeProgram program programOpcodeId [pc]) <> CODE_TABLE_SET.
Proof.
  intros program pc.
  change (encodeProgram program programOpcodeId [pc]) with (instructionRom program programOpcodeId pc 0).
  unfold instructionRom.
  destruct (instructionAtPC program pc) as [[f slot] | ]; [ | discriminate].
  cbv [programOpcodeId pcRomId Nat.eqb].
  destruct slot as [[[i | | | ] targets] | ]; try discriminate.
  destruct i; try discriminate; try (destruct targets as [ | [t | ] [ | ]]; discriminate);
    match goal with |- romValue (encodedOpcode ?j) <> _ =>
      destruct (encodedOpcode j) eqn:Hj; cbn [romValue]; intros Hz; [subst | discriminate];
      exact (encoded_opcode_not_table_set j Hj)
    end.
Qed.

Lemma table_element_writes_gated :
  writesOnlyWhen tableElementId COL_SEL_TABLE_SET memories = true.
Proof. vm_compute. reflexivity. Qed.

Lemma table_size_read_only : readOnlyMemory tableSizeId memories = true.
Proof. vm_compute. reflexivity. Qed.

(* A row of a run over a loaded program never selects table.set. *)
Lemma run_table_set_unselected : forall program initial rows final w,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final -> In w rows ->
  w COL_SEL_TABLE_SET <> 1.
Proof.
  intros program initial rows final w Hprime Hloaded [Hfull [_ Hreplay]] Hin Hset.
  pose proof (proj1 (Forall_forall _ _) Hfull w Hin) as Hw. clear Hfull.
  assert (Hselector : In COL_SEL_TABLE_SET selectorColumns) by (vm_compute; tauto).
  pose proof (full_selected_is_program_row w _ Hprime Hw Hselector Hset) as Hprogram.
  destruct (full_opcode_decode w Hprime Hw Hprogram) as [selector [code [Hentry [Hone Hcode]]]].
  assert (Hsame : selector = COL_SEL_TABLE_SET).
  { destruct (Nat.eq_dec selector COL_SEL_TABLE_SET) as [ | Hneq]; [assumption | exfalso].
    assert (Hcolumn : In selector selectorColumns).
    { rewrite <- opcode_selector_columns. apply (in_map fst _ _ Hentry). }
    pose proof (full_opcode_selector_exclusive w _ _ Hprime Hw Hset Hselector Hcolumn ltac:(congruence)). lia. }
  subst selector.
  assert (HcodeSet : code = CODE_TABLE_SET).
  { unfold selectorCodes in Hentry. vm_compute in Hentry.
    repeat destruct Hentry as [Hentry | Hentry]; try discriminate; try contradiction.
    injection Hentry as Hc. rewrite <- Hc. reflexivity. }
  assert (Hrom : initial programOpcodeId [w COL_PC_BEFORE] = Some (w COL_OPCODE_CODE)).
  { eapply replay_read_from_initial; [exact Hreplay | apply read_only_trace, opcode_memory_read_only | ].
    unfold eventsForTrace. apply in_flat_map. exists w. split; [exact Hin | exact (opcode_event_in_row w Hprogram)]. }
  rewrite (Hloaded programOpcodeId ltac:(cbn; tauto)), Hcode, HcodeSet in Hrom. injection Hrom as Hrom.
  exact (opcode_rom_not_table_set program _ Hrom).
Qed.

Lemma run_preserves_tables : forall program s initial rows final,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  tablesLoaded s initial -> tablesLoaded s final.
Proof.
  intros program s initial rows final Hprime Hloaded Hrun Htables a t Ht.
  pose proof Hrun as [_ [_ Hreplay]].
  destruct (Htables a t Ht) as [Hsize Helements]. split.
  - rewrite <- Hsize. eapply replay_preserves; [exact Hreplay | apply read_only_trace, table_size_read_only].
  - intros i r Hr. rewrite <- (Helements i r Hr). eapply replay_preserves; [exact Hreplay | ].
    apply Forall_forall. intros event Hevent.
    unfold eventsForTrace in Hevent. apply in_flat_map in Hevent. destruct Hevent as [w [Hw Hevent]].
    exact (proj1 (Forall_forall _ _) (gated_write_row _ _ _ _ w table_element_writes_gated
      (run_table_set_unselected program initial rows final w Hprime Hloaded Hrun Hw)) event Hevent).
Qed.

Lemma no_tables_loaded : forall s memory, store_TABLES s = [] -> tablesLoaded s memory.
Proof. intros s memory Hnone a t Ht. rewrite Hnone in Ht. destruct a; discriminate. Qed.
