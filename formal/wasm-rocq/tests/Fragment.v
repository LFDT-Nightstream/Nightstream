(* The fragment's per-instruction restrictions and its branch-site discipline. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 CallerSpec Fragment.
Import ListNotations.

Definition noCallees : CalleeEnvironment := fun _ => None.

Example nop_rejected : forall callees, ~ Supported callees NOP.
Proof. intros callees H. inversion H. Qed.

Example nested_block_supported :
  Supported noCallees (BLOCK (_RESULT None) [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 1)]]).
Proof. repeat constructor. Qed.

(* The call metadata holds the parameter count in one byte. *)
Example wide_call_rejected : forall callees x f, callees (x :> N) = Some f -> guestParams f = 256%nat ->
  ~ Supported callees (CALL x).
Proof.
  intros callees x f Hf Hparams H. inversion H as [ | | | | ? g Hg Hp _ | | | ]; subst.
  rewrite Hf in Hg. injection Hg as <-. lia.
Qed.

Example result_block_rejected : forall callees body, ~ Supported callees (BLOCK (_RESULT (Some valtype_I32)) body).
Proof. intros callees body H. inversion H. Qed.

(* Branches leave blocks, never the function. *)
Example br_if_outside_block_rejected : ~ codeBranches noCallees 0 [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0)] 0.
Proof. intros H. vm_compute in H. intuition lia. Qed.

Example br_if_to_function_rejected :
  ~ codeBranches noCallees 0 [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 1)]] 0.
Proof. intros H. vm_compute in H. intuition lia. Qed.

Example br_if_out_of_two_blocks :
  codeBranches noCallees 0 [BLOCK (_RESULT None) [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 1)]]] 0.
Proof. vm_compute. repeat split; lia. Qed.

(* A taken branch may discard operands: here the outer block's value below the
   inner block, or a value below the condition. *)
Definition aboveValue (depth : N) :=
  [BLOCK (_RESULT None)
    [LOCAL_GET (mk_uN 0); BLOCK (_RESULT None) [LOCAL_GET (mk_uN 1); BR_IF (mk_uN depth)]; LOCAL_SET (mk_uN 0)]].

Example inner_branch_above_value : codeBranches noCallees 0 (aboveValue 0) 0.
Proof. vm_compute. repeat split; lia. Qed.

Example outer_branch_above_value : codeBranches noCallees 0 (aboveValue 1) 0.
Proof. vm_compute. repeat split; lia. Qed.

Definition unwindingBody :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 0); LOCAL_SET (mk_uN 0)].

Example unwinding_branch : codeBranches noCallees 0 [BLOCK (_RESULT None) unwindingBody] 0.
Proof. vm_compute. repeat split; lia. Qed.
