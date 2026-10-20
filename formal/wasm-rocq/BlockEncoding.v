(* The flattened program ROM at a code path: a block's entry, its body's
   instructions and branches, and its END, at any depth of nesting. *)
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

(* An instruction's slots at a code path: inside a block, a branch carries its
   distance to the block's END. *)
Definition pathSlot (labels : list nat) (i : instr) (rest : list instr) : list FlatInstruction :=
  match labels with [] => flattenInstruction i | _ :: _ => blockSlot i rest end.

Lemma path_slot_head : forall labels i rest, exists more,
  pathSlot labels i rest = flatInstruction (Some i)
    (match labels, i with _ :: _, BR_IF _ => Some (S (S (codeSize rest))) | _, _ => None end) :: more.
Proof.
  intros [ | b labels] i rest; [destruct i; eexists; reflexivity | ].
  destruct (block_slot_head i rest) as [more H]. exists more. destruct i; exact H.
Qed.

Lemma flatten_at_path : forall code labels inner offset i k s,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (pathSlot labels i (skipn (S offset) inner)) k = Some s ->
  nth_error (flattenCode code) (pathOffset code labels offset + k) = Some s.
Proof.
  intros code [ | b labels] inner offset i k s Hpath Hi Hs.
  - injection Hpath as <-. exact (flatten_within_instruction _ _ _ _ _ Hi Hs).
  - cbn [pathCode] in Hpath. destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate].
    destruct j; try discriminate. eapply flatten_in_block; eassumption.
Qed.

Lemma flatten_end_at_path : forall code labels inner,
  labels <> [] -> pathCode code labels = Some inner ->
  nth_error (flattenCode code) (pathOffset code labels (length inner)) = Some (flatInstruction None None).
Proof.
  intros code [ | b labels] inner Hne Hpath; [contradiction | ].
  cbn [pathCode] in Hpath. destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate].
  destruct j; try discriminate. eapply flatten_block_end; eassumption.
Qed.

Lemma code_size_split : forall code offset,
  (codeSize code = codeOffset code offset + codeSize (skipn offset code))%nat.
Proof. intros. unfold codeOffset. rewrite <- code_size_app, firstn_skipn. reflexivity. Qed.

Section Path.
Variables (program : Program) (memory : MemoryState) (index : nat) (f : GuestFunction).
Hypothesis Hloaded : programLoaded program memory.
Hypothesis Hf : functionAt program index = Some f.

Definition pathPC (labels : list nat) (offset : nat) : Z :=
  guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset).

Lemma encoded_path_head : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  instructionAtPC program (pathPC labels offset) =
    Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
      (guestResults f) (guestLocals f) (guestBody f),
      Some (flatInstruction (Some i)
        (match labels, i with _ :: _, BR_IF _ => Some (S (S (codeSize (skipn (S offset) inner)))) | _, _ => None end))).
Proof.
  intros labels inner offset i Hpath Hi.
  destruct (path_slot_head labels i (skipn (S offset) inner)) as [more Hhead].
  pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
  rewrite Nat.add_0_r in Hslot. exact (encoded_slot _ _ _ _ _ Hf Hslot).
Qed.

Lemma encoded_path_ordinary : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  (exists opcode, encodedOpcode i = Some opcode) -> instructionImage memory (pathPC labels offset) i.
Proof.
  intros labels inner offset i Hpath Hi [op Hop].
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  assert (Hplain : (match labels, i with _ :: _, BR_IF _ => Some (S (S (codeSize (skipn (S offset) inner))))
      | _, _ => None end) = None).
  { destruct labels; [reflexivity | ].
    destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity. }
  rewrite Hplain in Hpc.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    unfold instructionImage; repeat split; try intros choice;
    try (program_cell Hloaded; unfold encodeProgram;
      cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
      unfold instructionRom; rewrite Hpc; reflexivity).
Qed.

Lemma encoded_path_block_entry : forall labels inner offset body,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BLOCK (_RESULT None) body) ->
  controlImage memory (pathPC labels offset) CODE_BLOCK (pathPC labels offset + 1).
Proof.
  intros labels inner offset body Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct labels; cbv iota in Hpc;
  unfold controlImage; repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; reflexivity.
Qed.

(* The END of the innermost enclosing block. *)
Lemma encoded_path_block_end : forall labels inner,
  labels <> [] -> pathCode (guestBody f) labels = Some inner ->
  controlImage memory (pathPC labels (length inner)) CODE_END (pathPC labels (length inner) + 1).
Proof.
  intros labels inner Hne Hpath.
  pose proof (encoded_slot _ _ _ _ _ Hf (flatten_end_at_path _ _ _ Hne Hpath)) as Hpc.
  unfold controlImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; unfold pathPC; rewrite Hpc; try reflexivity.
  all: cbn [Nat.eqb]; destruct (choice =? 0); cbn; [f_equal; lia | reflexivity].
Qed.

(* A depth-0 branch targets the instruction after the innermost block's END. *)
Lemma encoded_path_branch : forall labels inner offset,
  labels <> [] -> pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_IF (mk_uN 0)) ->
  branchImage memory (pathPC labels offset) (pathPC labels offset + 1) (pathPC labels (length inner) + 1).
Proof.
  intros labels inner offset Hne Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct labels as [ | b labels]; [contradiction | ].
  assert (Htarget : pathPC (b :: labels) (length inner) + 1 =
      pathPC (b :: labels) offset + Z.of_nat (S (S (codeSize (skipn (S offset) inner))))).
  { unfold pathPC. rewrite (path_offset_shift _ _ _ (length inner) Hpath), (path_offset_shift _ _ _ offset Hpath).
    rewrite code_offset_end, (code_size_split inner offset), (skipn_at_instruction _ _ _ Hi), code_size_cons.
    change (instructionSize (BR_IF (mk_uN 0))) with 1%nat. lia. }
  rewrite Htarget.
  unfold branchImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; try reflexivity.
  all: rewrite ?Nat.eqb_refl; destruct (choice =? 0); [ | destruct (choice =? 1)]; cbn beta iota; try reflexivity;
    f_equal; lia.
Qed.

(* A direct call's ROM image at a code path: target, return pc and callee metadata. *)
Lemma encoded_path_call : forall labels inner offset z x callee,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (CALL x) ->
  frameMatches program f z -> programCallees program (guestModule f) (x :> N) = Some callee ->
  directCallImage memory (pathPC labels offset) (pathPC labels offset + 1) z x
    (guestAddress callee) (guestModule callee) (guestTypeIndex callee)
    (guestParams callee) (guestResults callee) (guestLocals callee) (guestBody callee) (guestEntry callee) /\
  callerFrameImage memory (pathPC labels offset) z.
Proof.
  intros labels inner offset z x callee Hpath Hi Hframe Hcallee.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc0.
  assert (Hpc : instructionAtPC program (pathPC labels offset) =
      Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
        (guestResults f) (guestLocals f) (guestBody f), Some (flatInstruction (Some (CALL x)) None)))
    by (destruct labels; exact Hpc0).
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { destruct (path_slot_head labels (CALL x) (skipn (S offset) inner)) as [more Hhead].
    pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
    rewrite Nat.add_0_r in Hslot. unfold codeSize.
    assert (pathOffset (guestBody f) labels offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate). lia. }
  unfold programCallees in Hcallee.
  destruct (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) as [address | ] eqn:Haddress; [ | discriminate].
  destruct (function_at_sound _ _ _ Hcallee) as [Hinstance [HcalleeAddress Hentry]].
  replace (N.of_nat (N.to_nat address)) with address in HcalleeAddress by lia.
  split.
  - unfold directCallImage. split.
    + rewrite (frame_matches_addresses _ _ _ Hframe), HcalleeAddress. exact Haddress.
    + split.
      * rewrite (proj1 Hframe), HcalleeAddress. exact Hinstance.
      * split.
        -- program_cell Hloaded. unfold encodeProgram. cbn.
           unfold instructionRom. rewrite Hpc. reflexivity.
        -- split.
           ++ program_cell Hloaded. unfold encodeProgram. cbn.
              unfold instructionRom. rewrite Hpc. cbn [Nat.eqb pcRomId programOpcodeId pcEdgeKindId programLocalIndexId callTargetId].
              change (option_map (fun address => Z.of_N address + 1)
                (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) = Some (Z.of_N (guestAddress callee) + 1)).
              rewrite Haddress, HcalleeAddress. reflexivity.
           ++ split.
              ** program_cell Hloaded. unfold encodeProgram. cbn.
                 unfold instructionRom. rewrite Hpc. reflexivity.
              ** eapply encoded_guest_image; eassumption.
  - unfold pathPC. eapply encoded_frame_image; [exact Hloaded | exact Hf | exact Hbound | exact Hframe].
Qed.

End Path.
