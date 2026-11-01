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
   distance to its target. *)
Definition pathSlot (code : list instr) (labels : list nat) (i : instr) (rest : list instr) : list FlatInstruction :=
  match labels with [] => flattenInstruction [] i | _ :: _ => blockSlot (pathOuter code labels) i rest end.

Definition pathBranches (code : list instr) (labels : list nat) (i : instr) (rest : list instr) : list (option nat) :=
  match labels with [] => [] | _ :: _ => branchDistances (pathOuter code labels) i (codeSize rest) end.

Lemma path_slot_head : forall code labels i rest, exists more,
  pathSlot code labels i rest = flatInstruction (Some i) (pathBranches code labels i rest) :: more.
Proof.
  intros code [ | b labels] i rest; [destruct i; eexists; reflexivity | ].
  destruct (block_slot_head (pathOuter code (b :: labels)) i rest) as [more H]. exists more. exact H.
Qed.

Lemma flatten_at_path : forall code labels inner offset i k s,
  pathCode code labels = Some inner -> nth_error inner offset = Some i ->
  nth_error (pathSlot code labels i (skipn (S offset) inner)) k = Some s ->
  nth_error (flattenCode code) (pathOffset code labels offset + k) = Some s.
Proof.
  intros code [ | b labels] inner offset i k s Hpath Hi Hs.
  - injection Hpath as <-. exact (flatten_within_instruction _ _ _ _ _ Hi Hs).
  - cbn [pathCode] in Hpath. cbn [pathSlot pathOuter] in Hs. destruct (nth_error code b) as [j | ] eqn:Hb;
      [ | discriminate].
    destruct j; try discriminate. eapply flatten_in_block; eassumption.
Qed.

Lemma flatten_end_at_path : forall code labels inner,
  labels <> [] -> pathCode code labels = Some inner ->
  nth_error (flattenCode code) (pathOffset code labels (length inner)) = Some (flatInstruction None []).
Proof.
  intros code [ | b labels] inner Hne Hpath; [contradiction | ].
  cbn [pathCode] in Hpath. destruct (nth_error code b) as [j | ] eqn:Hb; [ | discriminate].
  destruct j; try discriminate. eapply flatten_block_end; eassumption.
Qed.

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
      Some (flatInstruction (Some i) (pathBranches (guestBody f) labels i (skipn (S offset) inner)))).
Proof.
  intros labels inner offset i Hpath Hi.
  destruct (path_slot_head (guestBody f) labels i (skipn (S offset) inner)) as [more Hhead].
  pose proof (flatten_at_path _ _ _ _ _ 0 _ Hpath Hi ltac:(rewrite Hhead; reflexivity)) as Hslot.
  rewrite Nat.add_0_r in Hslot. exact (encoded_slot _ _ _ _ _ Hf Hslot).
Qed.

Lemma encoded_path_ordinary : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  (exists opcode, encodedOpcode i = Some opcode) -> instructionImage memory (pathPC labels offset) i.
Proof.
  intros labels inner offset i Hpath Hi [op Hop].
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  assert (Hplain : pathBranches (guestBody f) labels i (skipn (S offset) inner) = []).
  { destruct labels; [reflexivity | ].
    destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity. }
  rewrite Hplain in Hpc.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    unfold instructionImage; repeat split; try intros choice;
    try (program_cell Hloaded; unfold encodeProgram;
      cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
      unfold instructionRom; rewrite Hpc; try reflexivity; destruct (choice =? 0); reflexivity).
Qed.

Lemma encoded_path_block_entry : forall labels inner offset bt body,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BLOCK bt body) ->
  controlImage memory (pathPC labels offset) CODE_BLOCK (pathPC labels offset + 1).
Proof.
  intros labels inner offset bt body Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct labels; cbv iota in Hpc;
  unfold controlImage; repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; try reflexivity; destruct (choice =? 0); reflexivity.
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

(* A branch to the block at b after pre targets the instruction after that block. *)
Lemma encoded_path_branch : forall labels inner offset l pre b post,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  labels = pre ++ b :: post -> N.to_nat (l :> N) = length post ->
  branchImage memory (pathPC labels offset) (pathPC labels offset + 1) (pathPC pre (S b)).
Proof.
  intros labels inner offset l pre b post Hpath Hi Hsplit Hl.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct (flatten_branch_target _ _ _ _ _ _ _ _ _ Hpath Hi (or_introl Logic.eq_refl) Hsplit Hl) as [d [Hd Htarget]].
  assert (Hbranch : pathBranches (guestBody f) labels (BR_IF l) (skipn (S offset) inner) = [Some d])
    by (subst labels; destruct pre; unfold pathBranches, branchDistances; cbn [branchTargets map]; rewrite Hd;
        reflexivity).
  rewrite Hbranch in Hpc.
  assert (Hpcs : pathPC pre (S b) = pathPC labels offset + Z.of_nat d) by (unfold pathPC; rewrite <- Htarget; lia).
  rewrite Hpcs.
  unfold branchImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; try reflexivity.
  all: rewrite ?Nat.eqb_refl; destruct (choice =? 0); [ | destruct (choice =? 1)]; cbn beta iota; try reflexivity;
    f_equal; lia.
Qed.

(* A br to the block at b after pre: its one edge targets the instruction after
   that block. *)
Lemma encoded_path_br : forall labels inner offset l pre b post,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR l) ->
  labels = pre ++ b :: post -> N.to_nat (l :> N) = length post ->
  controlImage memory (pathPC labels offset) CODE_BR (pathPC pre (S b)).
Proof.
  intros labels inner offset l pre b post Hpath Hi Hsplit Hl.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct (flatten_branch_target _ _ _ _ _ _ _ _ _ Hpath Hi (or_introl Logic.eq_refl) Hsplit Hl) as [d [Hd Htarget]].
  assert (Hbranch : pathBranches (guestBody f) labels (BR l) (skipn (S offset) inner) = [Some d])
    by (subst labels; destruct pre; unfold pathBranches, branchDistances; cbn [branchTargets map]; rewrite Hd;
        reflexivity).
  rewrite Hbranch in Hpc.
  assert (Hpcs : pathPC pre (S b) = pathPC labels offset + Z.of_nat d) by (unfold pathPC; rewrite <- Htarget; lia).
  rewrite Hpcs.
  unfold controlImage. repeat split; try intro choice; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; try reflexivity.
  destruct (Z.ltb_spec choice 0) as [Hneg | Hnonneg].
  - replace (choice =? 0) with false by (symmetry; apply Z.eqb_neq; lia). reflexivity.
  - destruct (Z.eqb_spec choice 0) as [-> | Hne]; [reflexivity | ].
    replace (Z.to_nat choice) with (S (Z.to_nat choice - 1)) by lia. destruct (Z.to_nat choice - 1)%nat; reflexivity.
Qed.

(* A br_table's ROM image: its opcode, a static edge, its arm count, and each
   edge's target, the instruction after the target block: the default at choice
   0, arm k at choice k + 1. *)
Lemma encoded_path_br_table : forall labels inner offset ls l,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_TABLE ls l) ->
  memory programOpcodeId [pathPC labels offset] = Some CODE_BR_TABLE /\
  memory pcEdgeKindId [pathPC labels offset] = Some 0 /\
  memory programBrTableLenId [pathPC labels offset] = Some (Z.of_nat (length ls)) /\
  forall k target pre b post, nth_error (l :: ls) k = Some target ->
    labels = pre ++ b :: post -> N.to_nat (target :> N) = length post ->
    memory pcRomId [pathPC labels offset; Z.of_nat k] = Some (pathPC pre (S b)).
Proof.
  intros labels inner offset ls l Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  split; [ | split; [ | split]].
  1-3: program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; reflexivity.
  intros k target pre b post Hk Hsplit Htarget.
  destruct (flatten_branch_target _ _ _ _ _ _ _ _ _ Hpath Hi (nth_error_In _ _ Hk) Hsplit Htarget)
    as [d [Hd Hend]].
  assert (Hdistance : nth_error (pathBranches (guestBody f) labels (BR_TABLE ls l) (skipn (S offset) inner)) k =
      Some (Some d)).
  { rewrite Hsplit in Hd |- *. destruct pre; cbn [app pathBranches] in Hd |- *; unfold branchDistances; rewrite nth_error_map; cbn [branchTargets]; rewrite Hk;
      cbn [Datatypes.option_map]; rewrite Hd; reflexivity. }
  program_cell Hloaded. unfold encodeProgram.
  cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb].
  unfold instructionRom. rewrite Hpc.
  replace (Z.of_nat k <? 0) with false by (symmetry; apply Z.ltb_ge; lia). rewrite Nat2Z.id, Hdistance.
  unfold pathPC. rewrite <- Hend, Nat.eqb_refl. cbn iota. f_equal. lia.
Qed.

(* A direct call's ROM image at a code path: target, return pc and callee metadata. *)
Lemma encoded_path_call : forall labels inner offset s values x callee,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (CALL x) ->
  store_FUNCS s = program -> length values = (guestParams f + guestLocals f)%nat ->
  programCallees program (guestModule f) (x :> N) = Some callee ->
  let z := mk_state s (MKframe values (guestModule f)) in
  directCallImage memory (pathPC labels offset) (pathPC labels offset + 1) z x
    (guestAddress callee) (guestModule callee) (guestTypeIndex callee)
    (guestParams callee) (guestResults callee) (guestLocals callee) (guestBody callee) (guestEntry callee) /\
  callerFrameImage memory (pathPC labels offset) z.
Proof.
  intros labels inner offset s values x callee Hpath Hi Hstore Hcount Hcallee z.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc0.
  assert (Hpc : instructionAtPC program (pathPC labels offset) =
      Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
        (guestResults f) (guestLocals f) (guestBody f), Some (flatInstruction (Some (CALL x)) [])))
    by (destruct labels; exact Hpc0).
  assert (Hbound : (pathOffset (guestBody f) labels offset <= codeSize (guestBody f))%nat).
  { destruct (path_slot_head (guestBody f) labels (CALL x) (skipn (S offset) inner)) as [more Hhead].
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
    + change (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N)) = Some (guestAddress callee)).
      rewrite HcalleeAddress. exact Haddress.
    + split.
      * change (nth_error (store_FUNCS s) (N.to_nat (guestAddress callee)) = Some (SpecTecCalls.i32Function
          (guestModule callee) (guestTypeIndex callee) (guestParams callee) (guestResults callee) (guestLocals callee)
          (guestBody callee))).
        rewrite Hstore, HcalleeAddress. exact Hinstance.
      * split.
        -- program_cell Hloaded. unfold encodeProgram. cbn.
           unfold instructionRom. rewrite Hpc. reflexivity.
        -- split.
           ++ program_cell Hloaded. unfold encodeProgram. cbn.
              unfold instructionRom. rewrite Hpc. cbn [Nat.eqb pcRomId programOpcodeId pcEdgeKindId programLocalIndexId callTargetId].
              change (Some (match option_map (fun address => Z.of_N address + 1)
                (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) with Some v => v | None => 0 end) =
                Some (Z.of_N (guestAddress callee) + 1)).
              rewrite Haddress, HcalleeAddress. reflexivity.
           ++ split.
              ** program_cell Hloaded. unfold encodeProgram. cbn.
                 unfold instructionRom. rewrite Hpc. reflexivity.
              ** eapply encoded_guest_image; eassumption.
  - unfold pathPC. eapply encoded_frame_image; [exact Hloaded | exact Hf | exact Hbound | exact Hcount].
Qed.

(* A return's slot: the RETURN opcode, with a return-like edge. *)
Lemma encoded_path_return : forall labels inner offset,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some RETURN ->
  memory programOpcodeId [pathPC labels offset] = Some CODE_RETURN /\
  memory pcEdgeKindId [pathPC labels offset] = Some 1.
Proof.
  intros labels inner offset Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  split; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; reflexivity.
Qed.

End Path.
