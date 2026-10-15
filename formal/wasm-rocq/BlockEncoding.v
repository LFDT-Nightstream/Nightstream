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

Lemma block_code_slot : forall body offset i,
  nth_error body offset = Some i ->
  nth_error (blockCode body) offset = Some (flatInstruction (Some i)
    (match i with BR_IF _ => Some (2 + length (skipn (S offset) body))%nat | _ => None end)).
Proof.
  induction body as [ | j body IH]; intros [ | offset] i H; try discriminate.
  - injection H as <-. reflexivity.
  - apply IH. exact H.
Qed.

Lemma encoded_block_slot : forall program index f offset body inner slot,
  functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some (BLOCK (_RESULT None) body) ->
  nth_error (flattenInstruction (BLOCK (_RESULT None) body)) inner = Some slot ->
  instructionAtPC program (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) + Z.of_nat inner) =
    Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
      (guestResults f) (guestLocals f) (guestBody f), Some slot).
Proof.
  intros. rewrite <- Z.add_assoc, <- Nat2Z.inj_add.
  eapply encoded_slot; [eassumption | ]. eapply flatten_within_instruction; eassumption.
Qed.

Lemma encoded_block_entry : forall program memory index f offset body,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some (BLOCK (_RESULT None) body) ->
  let pc := guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) in
  controlImage memory pc CODE_BLOCK (pc + 1).
Proof.
  intros program memory index f offset body Hloaded Hf Hi pc.
  pose proof (encoded_block_slot _ _ _ _ _ 0 _ Hf Hi eq_refl) as Hslot.
  rewrite Z.add_0_r in Hslot. unfold pc, controlImage.
  repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hslot; reflexivity.
Qed.

Lemma encoded_block_end : forall program memory index f offset body,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some (BLOCK (_RESULT None) body) ->
  let pc := guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) in
  controlImage memory (pc + 1 + Z.of_nat (length body)) CODE_END (pc + 2 + Z.of_nat (length body)).
Proof.
  intros program memory index f offset body Hloaded Hf Hi pc.
  assert (Hslot : nth_error (flattenInstruction (BLOCK (_RESULT None) body)) (S (length body)) =
    Some (flatInstruction None None)).
  { change (nth_error (blockCode body ++ [flatInstruction None None]) (length body) = Some (flatInstruction None None)).
    rewrite nth_error_app2, block_code_length by (rewrite block_code_length; lia).
    now rewrite Nat.sub_diag. }
  pose proof (encoded_block_slot _ _ _ _ _ _ _ Hf Hi Hslot) as Hpc.
  replace (guestEntry f + Z.of_nat (codeOffset (guestBody f) offset) + Z.of_nat (S (length body))) with
    (pc + 1 + Z.of_nat (length body)) in Hpc by (unfold pc; lia).
  unfold controlImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; try reflexivity.
  cbn [Nat.eqb]. destruct (choice =? 0); cbn; [f_equal; lia | reflexivity].
Qed.

Lemma block_inner_slot : forall body offset i,
  nth_error body offset = Some i ->
  nth_error (flattenInstruction (BLOCK (_RESULT None) body)) (S offset) =
    Some (flatInstruction (Some i)
      (match i with BR_IF _ => Some (2 + length (skipn (S offset) body))%nat | _ => None end)).
Proof.
  intros body offset i Hi. change (nth_error (blockCode body ++ [flatInstruction None None]) offset = Some (flatInstruction (Some i)
    (match i with BR_IF _ => Some (2 + length (skipn (S offset) body))%nat | _ => None end))).
  rewrite nth_error_app1; [apply block_code_slot; exact Hi | ].
  rewrite block_code_length. apply nth_error_Some; rewrite Hi; discriminate.
Qed.

Lemma encoded_block_ordinary : forall program memory index f outer body offset i,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) outer = Some (BLOCK (_RESULT None) body) -> nth_error body offset = Some i ->
  (exists opcode, encodedOpcode i = Some opcode) ->
  instructionImage memory (guestEntry f + Z.of_nat (codeOffset (guestBody f) outer) + 1 + Z.of_nat offset) i.
Proof.
  intros program memory index f outer body offset i Hloaded Hf Houter Hi [op Hop].
  replace (guestEntry f + Z.of_nat (codeOffset (guestBody f) outer) + 1 + Z.of_nat offset) with
    (guestEntry f + Z.of_nat (codeOffset (guestBody f) outer + S offset)) by lia.
  eapply encoded_ordinary_slot; [exact Hloaded | exact Hf | | eexists; exact Hop].
  eapply flatten_within_instruction; [exact Houter | ].
  pose proof (block_inner_slot _ _ _ Hi) as Hslot.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; exact Hslot.
Qed.

Lemma encoded_block_branch : forall program memory index f outer body offset,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) outer = Some (BLOCK (_RESULT None) body) ->
  nth_error body offset = Some (BR_IF (mk_uN 0)) ->
  let pc := guestEntry f + Z.of_nat (codeOffset (guestBody f) outer) in
  branchImage memory (pc + 1 + Z.of_nat offset) (pc + 2 + Z.of_nat offset) (pc + 2 + Z.of_nat (length body)).
Proof.
  intros program memory index f outer body offset Hloaded Hf Houter Hi pc.
  pose proof (encoded_block_slot _ _ _ _ _ _ _ Hf Houter (block_inner_slot _ _ _ Hi)) as Hslot.
  replace (guestEntry f + Z.of_nat (codeOffset (guestBody f) outer) + Z.of_nat (S offset)) with
    (pc + 1 + Z.of_nat offset) in Hslot by (unfold pc; lia).
  assert (Hbound : (offset < length body)%nat) by (apply nth_error_Some; rewrite Hi; discriminate).
  unfold branchImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hslot; try reflexivity.
  rewrite Nat.eqb_refl. destruct (choice =? 0); [ | destruct (choice =? 1)]; cbn beta iota; try reflexivity;
    f_equal; rewrite ?length_skipn; lia.
Qed.
