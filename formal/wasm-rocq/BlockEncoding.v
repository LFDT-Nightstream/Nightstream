(* The flattened program ROM supplies the block's entry, branch, tail and END. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 R1CS MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec CallSpec CallerSpec
  CodeLayout ProgramSpec ProgramEncoding BlockSpec BranchBindings StraightLineInstructions StackSimulation.
Import ListNotations.
Open Scope Z_scope.

Lemma encoded_slot : forall program index f offset slot,
  functionAt program index = Some f -> nth_error (flattenCode (guestBody f)) offset = Some slot ->
  instructionAtPC program (guestEntry f + Z.of_nat offset) =
    Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
      (guestResults f) (guestLocals f) (guestBody f), Some slot).
Proof.
  intros program index f offset slot Hf Hslot.
  rewrite (instruction_at_function_pc _ _ _ _ Hf).
  - now rewrite Hslot.
  - unfold codeSize. assert (offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate). lia.
Qed.

Lemma encoded_ordinary_slot : forall program memory index f offset i,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (flattenCode (guestBody f)) offset = Some (flatInstruction (Some i) None) ->
  (exists opcode, encodedOpcode i = Some opcode) ->
  instructionImage memory (guestEntry f + Z.of_nat offset) i.
Proof.
  intros program memory index f offset i Hloaded Hf Hslot [op Hop].
  pose proof (encoded_slot _ _ _ _ _ Hf Hslot) as Hpc.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    unfold instructionImage; repeat split; try intros choice;
    try (program_cell Hloaded; unfold encodeProgram;
      cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
      unfold instructionRom; rewrite Hpc; reflexivity).
Qed.

Lemma loaded_body_from_images : forall body memory entry,
  (forall index i, nth_error body index = Some i -> instructionImage memory (entry + Z.of_nat index) i) ->
  loadedBody memory entry body.
Proof.
  induction body as [ | i rest IH]; intros memory entry H; [exact I | ]. split.
  - replace entry with (entry + Z.of_nat 0) by lia. apply H. reflexivity.
  - apply IH. intros index j Hj.
    replace (entry + 1 + Z.of_nat index) with (entry + Z.of_nat (S index)) by lia.
    apply H. exact Hj.
Qed.

Lemma encoded_block_image : forall program memory index f offset x tail,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some (conditionalBlock x tail) ->
  supportedStraightLineProgram tail ->
  loadedBlock memory (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset)) x tail.
Proof.
  intros program memory index f offset x tail Hloaded Hf Hsource Hsupport.
  set (base := codeOffset (guestBody f) offset).
  assert (Hslot : forall inner slot,
    nth_error (flattenInstruction (conditionalBlock x tail)) inner = Some slot ->
    instructionAtPC program (guestEntry f + Z.of_nat base + Z.of_nat inner) =
      Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
        (guestResults f) (guestLocals f) (guestBody f), Some slot)).
  { intros inner slot H. rewrite <- Z.add_assoc, <- Nat2Z.inj_add.
    eapply encoded_slot; [exact Hf | ]. eapply flatten_within_instruction; eassumption. }
  pose proof (Hslot 0%nat (flatInstruction (Some (conditionalBlock x tail)) None) eq_refl) as Hentry.
  pose proof (Hslot 2%nat (flatInstruction (Some (BR_IF (mk_uN 0))) (Some (S (S (length tail))))) eq_refl) as Hbranch.
  assert (Hend : instructionAtPC program (guestEntry f + Z.of_nat base + 3 + Z.of_nat (length tail)) =
    Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
      (guestResults f) (guestLocals f) (guestBody f), Some (flatInstruction None None))).
  { replace (guestEntry f + Z.of_nat base + 3 + Z.of_nat (length tail)) with
      (guestEntry f + Z.of_nat base + Z.of_nat (3 + length tail)) by lia.
    apply Hslot. change (nth_error (blockCode tail ++ [flatInstruction None None]) (length tail) = Some (flatInstruction None None)).
    rewrite nth_error_app2, block_code_length by (rewrite block_code_length; lia).
    rewrite Nat.sub_diag. reflexivity. }
  rewrite Z.add_0_r in Hentry.
  assert (HbranchPC : instructionAtPC program (guestEntry f + Z.of_nat base + 2) =
      Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
        (guestResults f) (guestLocals f) (guestBody f),
        Some (flatInstruction (Some (BR_IF (mk_uN 0))) (Some (S (S (length tail))))))) by exact Hbranch.
  unfold loadedBlock. fold base. split.
  - unfold controlImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
      cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
      unfold instructionRom; rewrite Hentry; reflexivity.
  - split.
    + replace (guestEntry f + Z.of_nat base + 1) with (guestEntry f + Z.of_nat (base + 1)) by lia.
      eapply encoded_ordinary_slot; [exact Hloaded | exact Hf | | eexists; reflexivity].
      eapply flatten_within_instruction; [exact Hsource | reflexivity].
    + split.
      * unfold branchImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
          cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
          unfold instructionRom; rewrite HbranchPC; try reflexivity.
        cbn [Nat.eqb]. unfold blockExitPC. rewrite !Nat2Z.inj_succ.
        destruct (choice =? 0); [ | destruct (choice =? 1)]; cbn; try reflexivity.
        all: try (f_equal; lia).
      * split.
        -- apply loaded_body_from_images. intros inner i Hi.
           replace (guestEntry f + Z.of_nat base + 3 + Z.of_nat inner) with
             (guestEntry f + Z.of_nat (base + (3 + inner))) by lia.
           eapply encoded_ordinary_slot; [exact Hloaded | exact Hf | | ].
           ++ eapply flatten_within_instruction; [exact Hsource | ].
              change (nth_error (blockCode tail ++ [flatInstruction None None]) inner = Some (flatInstruction (Some i) None)).
              rewrite nth_error_app1.
              ** eapply block_code_ordinary; eassumption.
              ** rewrite block_code_length. apply nth_error_Some; rewrite Hi; discriminate.
           ++ apply (proj1 (Forall_forall _ _) Hsupport). eapply nth_error_In; exact Hi.
        -- unfold controlImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
             cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
             unfold instructionRom; rewrite Hend; try reflexivity.
           cbn [Nat.eqb]. unfold blockExitPC. destruct (choice =? 0); cbn; [f_equal; lia | reflexivity].
Qed.

Lemma block_endpoint_outside : forall program position f offset x tail boundary,
  functionAt program (sourceFunction position) = Some f ->
  nth_error (guestBody f) offset = Some (conditionalBlock x tail) ->
  sourceEndpoint program position boundary ->
  let pc := guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) in
  ~ (pc < boundaryPC boundary < blockExitPC pc tail).
Proof.
  intros program position f offset x tail boundary Hf Hi Hendpoint pc Hinside.
  unfold sourceEndpoint in Hendpoint. rewrite Hf in Hendpoint. destruct Hendpoint as [last [Hbound Hpc]].
  pose proof (no_boundary_inside_instruction _ _ _ last Hi) as Hno.
  unfold instructionSize, flattenInstruction, conditionalBlock in Hno.
  cbn [length] in Hno. rewrite length_app, block_code_length in Hno. cbn [length] in Hno.
  unfold pc, blockExitPC in Hinside. rewrite Hpc in Hinside. apply Hno. lia.
Qed.
