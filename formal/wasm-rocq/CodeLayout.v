(* Flatten one level of blocks into instruction addresses. A branch stores its
   relative distance to the instruction after the block's END. *)
From Stdlib Require Import List Arith Lia.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StackSimulation.
Import ListNotations.

Record FlatInstruction := flatInstruction {
  flatOpcode : option instr;
  flatBranchDistance : option nat
}.

Fixpoint blockCode (body : list instr) : list FlatInstruction :=
  match body with [] => [] | i :: rest =>
    flatInstruction (Some i)
      (match i with BR_IF _ => Some (S (S (length rest))) | _ => None end) :: blockCode rest
  end.

Definition flattenInstruction (i : instr) : list FlatInstruction :=
  match i with
  | BLOCK _ body => flatInstruction (Some i) None :: blockCode body ++ [flatInstruction None None]
  | _ => [flatInstruction (Some i) None]
  end.
Definition flattenCode (code : list instr) := flat_map flattenInstruction code.
Definition instructionSize (i : instr) := length (flattenInstruction i).
Definition codeSize (code : list instr) := length (flattenCode code).
Definition codeOffset (code : list instr) (offset : nat) := codeSize (firstn offset code).

Lemma block_code_length : forall code, length (blockCode code) = length code.
Proof. induction code; cbn; congruence. Qed.
Lemma code_size_app : forall a b, codeSize (a ++ b) = codeSize a + codeSize b.
Proof. intros; unfold codeSize, flattenCode; rewrite flat_map_app, length_app; reflexivity. Qed.
Lemma code_size_cons : forall i rest, codeSize (i :: rest) = instructionSize i + codeSize rest.
Proof. intros; unfold codeSize, flattenCode; cbn [flat_map]; apply length_app. Qed.
Lemma ordinary_instruction_size : forall i,
  (exists op, encodedOpcode i = Some op) -> instructionSize i = 1.
Proof.
  intros i [op H]. destruct (encoded_opcode_cases _ _ H) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; reflexivity.
Qed.
Lemma straight_code_size : forall code,
  supportedStraightLineProgram code -> codeSize code = length code.
Proof.
  intros code H. induction H; [reflexivity | ].
  rewrite code_size_cons, ordinary_instruction_size by assumption. cbn. congruence.
Qed.
Lemma firstn_at_instruction : forall (code : list instr) offset i,
  nth_error code offset = Some i -> firstn (S offset) code = firstn offset code ++ [i].
Proof.
  induction code as [ | a code IH]; intros [ | offset] i H; try discriminate.
  - injection H as <-. reflexivity.
  - change (a :: firstn (S offset) code = a :: (firstn offset code ++ [i])).
    f_equal. apply IH. exact H.
Qed.
Lemma code_offset_next : forall code offset i,
  nth_error code offset = Some i -> codeOffset code (S offset) = codeOffset code offset + instructionSize i.
Proof.
  intros code offset i H. unfold codeOffset.
  rewrite (firstn_at_instruction code offset i H), code_size_app, code_size_cons.
  change (codeSize (firstn offset code) + (instructionSize i + 0) = codeSize (firstn offset code) + instructionSize i).
  lia.
Qed.
Lemma code_offset_bound : forall code offset, codeOffset code offset <= codeSize code.
Proof.
  intros code offset. pose proof (firstn_skipn offset code) as H.
  rewrite <- H at 2. rewrite code_size_app. unfold codeOffset; lia.
Qed.
Lemma code_offset_end : forall code, codeOffset code (length code) = codeSize code.
Proof. intros; unfold codeOffset; now rewrite firstn_all. Qed.
Lemma flatten_at_source : forall code offset i,
  nth_error code offset = Some i ->
  nth_error (flattenCode code) (codeOffset code offset) = Some (flatInstruction (Some i) None).
Proof.
  intros code offset i H.
  rewrite <- (firstn_skipn offset code) at 1.
  unfold flattenCode at 1. rewrite flat_map_app.
  rewrite nth_error_app2 by (unfold codeOffset, codeSize, flattenCode; lia).
  change (nth_error (flat_map flattenInstruction (skipn offset code))
    (codeOffset code offset - codeOffset code offset) = Some (flatInstruction (Some i) None)).
  rewrite Nat.sub_diag, (skipn_at_instruction code offset i H).
  destruct i; reflexivity.
Qed.

Lemma flatten_within_instruction : forall code offset i inner slot,
  nth_error code offset = Some i -> nth_error (flattenInstruction i) inner = Some slot ->
  nth_error (flattenCode code) (codeOffset code offset + inner) = Some slot.
Proof.
  intros code offset i inner slot Hsource Hslot.
  rewrite <- (firstn_skipn offset code) at 1. unfold flattenCode at 1. rewrite flat_map_app.
  rewrite nth_error_app2 by (unfold codeOffset, codeSize, flattenCode; lia).
  replace (codeOffset code offset + inner - length (flat_map flattenInstruction (firstn offset code)))
    with inner by (unfold codeOffset, codeSize, flattenCode; lia).
  rewrite (skipn_at_instruction code offset i Hsource). cbn [flat_map].
  rewrite nth_error_app1; [exact Hslot | apply nth_error_Some; rewrite Hslot; discriminate].
Qed.
