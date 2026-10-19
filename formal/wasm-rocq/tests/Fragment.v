(* The fragment's per-site restrictions. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 CallerSpec Fragment.
Import ListNotations.

Example nop_rejected : forall site, ~ Supported site NOP.
Proof. intros site H. inversion H. Qed.

Example nested_block_rejected : forall callees body,
  ~ Supported (functionSite callees) (BLOCK (_RESULT None) [BLOCK (_RESULT None) body]).
Proof. intros callees body H. inversion H as [ | | | | | ? ? Hbody _ | ]; subst. inversion Hbody as [ | ? ? Hinner _]. inversion Hinner. Qed.

(* The call metadata holds the parameter count in one byte. *)
Example wide_call_rejected : forall callees x f, callees (x :> N) = Some f -> guestParams f = 256%nat ->
  ~ Supported (functionSite callees) (CALL x).
Proof.
  intros callees x f Hf Hparams H. inversion H as [ | | | | ? ? g Hg Hp _ | | ]; subst.
  rewrite Hf in Hg. injection Hg as <-. lia.
Qed.

Example call_in_block_rejected : forall x, ~ Supported blockSite (CALL x).
Proof. intros x H. inversion H. Qed.

Example br_if_outside_block_rejected : forall callees, ~ Supported (functionSite callees) (BR_IF (mk_uN 0)).
Proof. intros callees H. inversion H. Qed.

Example deeper_branch_rejected : ~ Supported blockSite (BR_IF (mk_uN 1)).
Proof. intros H. inversion H. Qed.

Example result_block_rejected : forall callees body,
  ~ Supported (functionSite callees) (BLOCK (_RESULT (Some valtype_I32)) body).
Proof. intros callees body H. inversion H. Qed.
