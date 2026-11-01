(* The flattened program ROM at a code path: structured instructions' heads, the
   last slot of each arm, and the instructions and branches of any body, at any
   depth of nesting. *)
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
  - rewrite <- flatten_code_length. assert (offset < length (flattenCode (guestBody f)))%nat
      by (apply nth_error_Some; rewrite Hslot; discriminate). lia.
Qed.

(* Slots' targets are relative to their function's entry. *)
Lemma slot_pc_entry : forall program index f offset slot,
  functionAt program index = Some f -> nth_error (flattenCode (guestBody f)) offset = Some slot ->
  pcEntry program (guestEntry f + Z.of_nat offset) = guestEntry f.
Proof.
  intros program index f offset slot Hf Hslot. apply (pc_entry_function _ _ _ _ Hf).
  rewrite <- flatten_code_length. assert (offset < length (flattenCode (guestBody f)))%nat
    by (apply nth_error_Some; rewrite Hslot; discriminate). lia.
Qed.

(* A one-edge pc-ROM entry: `next` at choice 0, padding elsewhere. *)
Lemma single_edge : forall pc next choice,
  Some (match (if choice <? 0 then None else
    match nth_error [Some next] (Z.to_nat choice) with Some (Some target) => Some (pc + Z.of_nat target)
    | _ => None end) with Some v => v | None => 0 end) =
  if choice =? 0 then Some (pc + Z.of_nat next) else Some 0.
Proof.
  intros pc next choice. destruct (Z.ltb_spec choice 0) as [Hneg | Hnonneg].
  - replace (choice =? 0) with false by (symmetry; apply Z.eqb_neq; lia). reflexivity.
  - destruct (Z.eqb_spec choice 0) as [-> | Hne]; [reflexivity | ].
    replace (Z.to_nat choice) with (S (Z.to_nat choice - 1)) by lia. destruct (Z.to_nat choice - 1)%nat; reflexivity.
Qed.

Section Path.
Variables (program : Program) (memory : MemoryState) (index : nat) (f : GuestFunction).
Hypothesis Hloaded : programLoaded program memory.
Hypothesis Hf : functionAt program index = Some f.

Definition pathPC (labels : list LabelSite) (offset : nat) : Z :=
  guestEntry f + Z.of_nat (pathOffset (guestBody f) labels offset).

(* Where branches to the label of the instruction i at s, after pre, land. *)
Definition sitePC (pre : list LabelSite) (s : LabelSite) (i : instr) : Z :=
  guestEntry f + Z.of_nat (pathOffset (guestBody f) pre (siteOffset s) + labelTarget i).

Lemma encoded_path_head : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  instructionAtPC program (pathPC labels offset) =
    Some (SpecTecCalls.i32Function (guestModule f) (guestTypeIndex f) (guestParams f)
      (guestResults f) (guestLocals f) (guestBody f),
      Some (flatInstruction (opInstr i)
        (slotTargets (pathTargets (guestBody f) labels) (pathOffset (guestBody f) labels offset) i))).
Proof.
  intros labels inner offset i Hpath Hi.
  exact (encoded_slot _ _ _ _ _ Hf (flatten_at_path _ _ _ _ _ Hpath Hi)).
Qed.

Lemma path_pc_entry : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  pcEntry program (pathPC labels offset) = guestEntry f.
Proof.
  intros labels inner offset i Hpath Hi. exact (slot_pc_entry _ _ _ _ _ Hf (flatten_at_path _ _ _ _ _ Hpath Hi)).
Qed.

Lemma path_offset_inside : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  (pathOffset (guestBody f) labels offset < codeSize (guestBody f))%nat.
Proof.
  intros labels inner offset i Hpath Hi. rewrite <- flatten_code_length.
  apply nth_error_Some. rewrite (flatten_at_path _ _ _ _ _ Hpath Hi). discriminate.
Qed.

Ltac pc_cell Hpc :=
  program_cell Hloaded; unfold encodeProgram;
  cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
  unfold instructionRom; rewrite Hpc;
  cbn [Nat.eqb pcRomId programOpcodeId pcEdgeKindId programLocalIndexId programBrTableLenId callTargetId].

Lemma encoded_path_ordinary : forall labels inner offset i,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some i ->
  (exists opcode, encodedOpcode i = Some opcode) -> instructionImage memory (pathPC labels offset) i.
Proof.
  intros labels inner offset i Hpath Hi [op Hop].
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    cbn [slotTargets] in Hpc;
    unfold instructionImage; repeat split; try intros choice;
    try (pc_cell Hpc; try reflexivity; destruct (choice =? 0); reflexivity).
Qed.

Lemma encoded_path_block_entry : forall labels inner offset bt body,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BLOCK bt body) ->
  controlImage memory (pathPC labels offset) CODE_BLOCK (pathPC labels offset + 1).
Proof.
  intros labels inner offset bt body Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets] in Hpc.
  unfold controlImage; repeat split; try intro choice; pc_cell Hpc; try reflexivity; destruct (choice =? 0); reflexivity.
Qed.

(* An if's head: its true edge enters the then-body; its false edge lands past
   the then-body and ELSE, at the else-body or after the if's END. *)
Lemma encoded_path_if : forall labels inner offset bt t e,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (IFELSE bt t e) ->
  memory programOpcodeId [pathPC labels offset] = Some CODE_IF /\
  memory pcEdgeKindId [pathPC labels offset] = Some 0 /\
  forall choice, memory pcRomId [pathPC labels offset; choice] =
    if choice =? 1 then Some (pathPC labels offset + 1)
    else if choice =? 0 then Some (pathPC labels offset + Z.of_nat (S (S (codeSize t)))) else Some 0.
Proof.
  intros labels inner offset bt t e Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets] in Hpc.
  pose proof (path_pc_entry _ _ _ _ Hpath Hi) as Hentry.
  split; [ | split; [ | intros choice]]; pc_cell Hpc; try reflexivity.
  rewrite Hentry. destruct (choice =? 1); [reflexivity | ]. destruct (choice =? 0); [ | reflexivity].
  unfold pathPC. f_equal. lia.
Qed.

(* The last slot of an arm: an if's then-body ends at its ELSE when it has an
   else-body, every other arm at its instruction's END. Either continues after
   the instruction. *)
Definition armEndOpcode (i : instr) (arm : nat) : Z :=
  match i, arm with IFELSE _ _ (_ :: _), O => CODE_ELSE | _, _ => CODE_END end.

Lemma encoded_path_arm_end : forall pre s inner outer i,
  pathCode (guestBody f) pre = Some outer -> siteEntry outer s = Some (i, inner) ->
  controlImage memory (pathPC (pre ++ [s]) (length inner)) (armEndOpcode i (siteArm s))
    (pathPC pre (S (siteOffset s))).
Proof.
  intros pre s inner outer i Hpre Hentry.
  destruct (path_snoc _ _ _ _ _ _ Hpre Hentry) as [Hpath Hoffset].
  destruct (flatten_arm_end _ _ _ _ Hpath) as [i' [Hentry' Hslot]].
  rewrite Hpre, Hentry in Hentry'. injection Hentry' as <-.
  unfold siteEntry in Hentry. destruct (nth_error outer (siteOffset s)) as [j | ] eqn:Hj; [ | discriminate].
  destruct (instrArm j (siteArm s)) as [body | ] eqn:Harm; [ | discriminate].
  cbn [option_map] in Hentry. injection Hentry as <- <-.
  pose proof (encoded_slot _ _ _ _ _ Hf Hslot) as Hpc. pose proof (slot_pc_entry _ _ _ _ _ Hf Hslot) as Hentry.
  change (guestEntry f + Z.of_nat (pathOffset (guestBody f) (pre ++ [s]) (length body)))
    with (pathPC (pre ++ [s]) (length body)) in Hpc, Hentry.
  assert (Hnext : pathPC pre (S (siteOffset s)) = guestEntry f +
      Z.of_nat (pathOffset (guestBody f) pre (siteOffset s) + instructionSize j))
    by (unfold pathPC; rewrite (path_offset_next _ _ _ _ _ Hpre Hj); reflexivity).
  rewrite Hnext.
  (* An END is its instruction's last slot; a then-body's ELSE jumps past the else-body. *)
  assert (Hshape : (armEndOpcode j (siteArm s) = CODE_END /\
      (forall after, armEnd j (siteArm s) after = flatInstruction opEnd []) /\
      (armStart j (siteArm s) + codeSize body + 1 = instructionSize j)%nat) \/
    (armEndOpcode j (siteArm s) = CODE_ELSE /\
      forall after, armEnd j (siteArm s) after = flatInstruction opElse [Some after])).
  { destruct j; try discriminate; destruct (siteArm s) as [ | [ | arm]]; try discriminate; cbn [instrArm] in Harm.
    - injection Harm as <-. left. repeat split. rewrite block_instruction_size. cbn [armStart]. lia.
    - injection Harm as <-. left. repeat split. rewrite loop_instruction_size. cbn [armStart]. lia.
    - injection Harm as <-. cbn [armEndOpcode armEnd]. rewrite if_instruction_size. cbn [armStart].
      match goal with |- context [match ?e with [] => _ | _ :: _ => _ end] => destruct e end;
        [left | right]; repeat split; lia.
    - match type of Harm with match ?e with [] => _ | _ :: _ => _ end = _ => destruct e end; [discriminate | ].
      injection Harm as <-. left. cbn [armEndOpcode armEnd armStart]. repeat split.
      rewrite if_instruction_size. cbn [codeSize]. lia. }
  unfold controlImage. pose proof (Hoffset (length body)) as Hoff. rewrite code_offset_end in Hoff.
  destruct Hshape as [[Hop [Hend Hsize]] | [Hop Helse]]; rewrite ?Hend, ?Helse in Hpc; rewrite Hop.
  - repeat split; try intro choice; pc_cell Hpc; try reflexivity.
    destruct (choice =? 0); [ | reflexivity]. unfold pathPC. rewrite Hoff. f_equal. lia.
  - repeat split; try intro choice; pc_cell Hpc; try reflexivity.
    rewrite Hentry, single_edge. destruct (choice =? 0); reflexivity.
Qed.

(* A branch at a path to its label at s after pre lands where that label's
   branches land. *)
Lemma path_branch_target : forall labels inner (l : labelidx) pre s post outer j body,
  pathCode (guestBody f) labels = Some inner -> labels = pre ++ s :: post -> N.to_nat (l :> N) = length post ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer s = Some (j, body) ->
  nth_error (pathTargets (guestBody f) labels) (N.to_nat (l :> N)) =
    Some (pathOffset (guestBody f) pre (siteOffset s) + labelTarget j)%nat.
Proof.
  intros labels inner l pre s post outer j body Hpath -> Hl Hpre Hentry.
  rewrite Hl. exact (path_target _ _ _ _ _ _ _ _ Hpath Hpre Hentry).
Qed.

Lemma encoded_path_branch : forall labels inner offset l pre s post outer j body,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_IF l) ->
  labels = pre ++ s :: post -> N.to_nat (l :> N) = length post ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer s = Some (j, body) ->
  branchImage memory (pathPC labels offset) (pathPC labels offset + 1) (sitePC pre s j).
Proof.
  intros labels inner offset l pre s post outer j body Hpath Hi Hsplit Hl Hpre Hentry.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets branchTargets map] in Hpc.
  rewrite (path_branch_target _ _ _ _ _ _ _ _ _ Hpath Hsplit Hl Hpre Hentry) in Hpc.
  pose proof (path_pc_entry _ _ _ _ Hpath Hi) as Hentry_pc.
  unfold branchImage. repeat split; try intro choice; pc_cell Hpc; try reflexivity.
  rewrite Hentry_pc. destruct (choice =? 0); [reflexivity | ]. destruct (choice =? 1); reflexivity.
Qed.

(* A br: its one edge lands where its target label's branches land. *)
Lemma encoded_path_br : forall labels inner offset l pre s post outer j body,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR l) ->
  labels = pre ++ s :: post -> N.to_nat (l :> N) = length post ->
  pathCode (guestBody f) pre = Some outer -> siteEntry outer s = Some (j, body) ->
  controlImage memory (pathPC labels offset) CODE_BR (sitePC pre s j).
Proof.
  intros labels inner offset l pre s post outer j body Hpath Hi Hsplit Hl Hpre Hentry.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets branchTargets map] in Hpc.
  rewrite (path_branch_target _ _ _ _ _ _ _ _ _ Hpath Hsplit Hl Hpre Hentry) in Hpc.
  pose proof (path_pc_entry _ _ _ _ Hpath Hi) as Hentry_pc.
  unfold controlImage. repeat split; try intro choice; pc_cell Hpc; try reflexivity.
  rewrite Hentry_pc, single_edge. destruct (choice =? 0); reflexivity.
Qed.

(* A br_table's ROM image: its opcode, a static edge, its arm count, and each
   edge's target: the default at choice 0, arm k at choice k + 1. *)
Lemma encoded_path_br_table : forall labels inner offset ls l,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some (BR_TABLE ls l) ->
  memory programOpcodeId [pathPC labels offset] = Some CODE_BR_TABLE /\
  memory pcEdgeKindId [pathPC labels offset] = Some 0 /\
  memory programBrTableLenId [pathPC labels offset] = Some (Z.of_nat (length ls)) /\
  forall k target pre s post outer j body, nth_error (l :: ls) k = Some target ->
    labels = pre ++ s :: post -> N.to_nat (target :> N) = length post ->
    pathCode (guestBody f) pre = Some outer -> siteEntry outer s = Some (j, body) ->
    memory pcRomId [pathPC labels offset; Z.of_nat k] = Some (sitePC pre s j).
Proof.
  intros labels inner offset ls l Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc.
  split; [ | split; [ | split]].
  1-3: pc_cell Hpc; reflexivity.
  intros k target pre s post outer j body Hk Hsplit Htarget Hpre Hentry.
  assert (Hedge : nth_error (slotTargets (pathTargets (guestBody f) labels) (pathOffset (guestBody f) labels offset)
      (BR_TABLE ls l)) k = Some (Some (pathOffset (guestBody f) pre (siteOffset s) + labelTarget j)%nat)).
  { cbn [slotTargets branchTargets]. rewrite nth_error_map, Hk. cbn [Datatypes.option_map].
    rewrite (path_branch_target _ _ _ _ _ _ _ _ _ Hpath Hsplit Htarget Hpre Hentry).
    reflexivity. }
  pose proof (path_pc_entry _ _ _ _ Hpath Hi) as Hentry_pc.
  pc_cell Hpc.
  replace (Z.of_nat k <? 0) with false by (symmetry; apply Z.ltb_ge; lia). rewrite Nat2Z.id, Hedge, Hentry_pc.
  reflexivity.
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
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets] in Hpc.
  pose proof (path_offset_inside _ _ _ _ Hpath Hi) as Hbound.
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
  - unfold pathPC. eapply encoded_frame_image; [exact Hloaded | exact Hf | lia | exact Hcount].
Qed.

(* A return's slot: the RETURN opcode, with a return-like edge. *)
Lemma encoded_path_return : forall labels inner offset,
  pathCode (guestBody f) labels = Some inner -> nth_error inner offset = Some RETURN ->
  memory programOpcodeId [pathPC labels offset] = Some CODE_RETURN /\
  memory pcEdgeKindId [pathPC labels offset] = Some 1.
Proof.
  intros labels inner offset Hpath Hi.
  pose proof (encoded_path_head _ _ _ _ Hpath Hi) as Hpc. cbn [slotTargets] in Hpc.
  split; pc_cell Hpc; reflexivity.
Qed.

End Path.
