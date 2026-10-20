(* The fragment's per-site restrictions. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 CallerSpec Fragment.
Import ListNotations.

Definition noCallees : CalleeEnvironment := fun _ => None.

Example nop_rejected : forall site, ~ Supported site NOP.
Proof. intros site H. inversion H. Qed.

(* Blocks nest, and branch to the innermost one from any depth. *)
Example nested_block_supported :
  Supported (codeSite noCallees false)
    (BLOCK (_RESULT None) [BLOCK (_RESULT None) [LOCAL_GET (mk_uN 0); BR_IF (mk_uN 0)]]).
Proof.
  constructor; [ | reflexivity]. repeat constructor.
Qed.

(* The call metadata holds the parameter count in one byte. *)
Example wide_call_rejected : forall callees inside x f, callees (x :> N) = Some f -> guestParams f = 256%nat ->
  ~ Supported (codeSite callees inside) (CALL x).
Proof.
  intros callees inside x f Hf Hparams H. inversion H as [ | | | | ? ? ? g Hg Hp _ | | ]; subst.
  rewrite Hf in Hg. injection Hg as <-. lia.
Qed.

Example br_if_outside_block_rejected : forall callees, ~ Supported (codeSite callees false) (BR_IF (mk_uN 0)).
Proof. intros callees H. inversion H. Qed.

Example deeper_branch_rejected : forall site, ~ Supported site (BR_IF (mk_uN 1)).
Proof. intros site H. inversion H. Qed.

Example result_block_rejected : forall site body, ~ Supported site (BLOCK (_RESULT (Some valtype_I32)) body).
Proof. intros site body H. inversion H. Qed.

(* SpecTec permits a taken branch to discard the first local.get's value. The
   circuit only removes the condition, so a branch site must hold only it. *)
Definition unwindingBody :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 0); LOCAL_SET (mk_uN 0)].

Example unwinding_block_rejected : forall site, ~ Supported site (BLOCK (_RESULT None) unwindingBody).
Proof.
  intros site H. inversion H as [ | | | | | ? ? ? _ Hbalanced | ]; subst.
  destruct Hbalanced as [Hheight _]. discriminate.
Qed.
