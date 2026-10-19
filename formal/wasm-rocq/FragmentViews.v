(* The internal support predicates, derived from the circuit's opcode encoding,
   describe exactly the instructions the Fragment table lists. *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 StraightLineSpec StraightLineInstructions CallerSpec BlockSpec SourceInstructions Fragment.
Import ListNotations.

Lemma forall_iff : forall (A : Type) (P Q : A -> Prop) xs,
  (forall x, P x <-> Q x) -> Forall P xs <-> Forall Q xs.
Proof.
  intros A P Q xs H. split; intros Hall; eapply Forall_impl; try exact Hall; intros x Hx; apply H; exact Hx.
Qed.

Lemma opcode_supported : forall site i op, encodedOpcode i = Some op -> Supported site i.
Proof.
  intros site i op Hop.
  destruct (encoded_opcode_cases _ _ Hop) as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]]; constructor.
Qed.

Lemma block_supported_iff : forall i,
  Supported blockSite i <-> i = BR_IF (mk_uN 0) \/ exists op, encodedOpcode i = Some op.
Proof.
  intros i. split.
  - intros H. inversion H; subst; first [left; reflexivity | right; eexists; reflexivity].
  - intros [-> | [op Hop]]; [constructor | exact (opcode_supported _ _ _ Hop)].
Qed.

Lemma branch_body_supported_iff : forall code, supportedCode blockSite code <-> supportedBranchBody code.
Proof. intros code. apply forall_iff. exact block_supported_iff. Qed.

Lemma function_supported_iff : forall callees i,
  Supported (functionSite callees) i <-> supportedSourceInstruction callees i.
Proof.
  intros callees i. split.
  - intros H. inversion H as [ | | | | ? x f Hf Hp Hr | ? body Hbody Hbalanced | ]; subst;
      cbn [supportedSourceInstruction].
    1-4: constructor; [eexists; reflexivity | constructor].
    + constructor; [ | constructor]. exists f. split; [exact Hf | split; assumption].
    + split; [reflexivity | split; [apply branch_body_supported_iff; exact Hbody | exact Hbalanced]].
  - intros H.
    assert (Hcases : (exists bt body, i = BLOCK bt body) \/ (exists x, i = CALL x) \/
        (exists op, encodedOpcode i = Some op))
      by (destruct i; try (right; right; exact (Forall_inv H)); eauto).
    destruct Hcases as [[bt [body ->]] | [[x ->] | [op Hop]]].
    + destruct H as [-> [Hbody Hbalanced]].
      constructor; [apply branch_body_supported_iff; exact Hbody | exact Hbalanced].
    + destruct (Forall_inv H) as [f [Hf [Hp Hr]]]. econstructor; [exact Hf | exact Hp | exact Hr].
    + exact (opcode_supported _ _ _ Hop).
Qed.

Lemma source_code_supported_iff : forall callees code,
  supportedCode (functionSite callees) code <-> supportedSourceCode callees code.
Proof. intros callees code. apply forall_iff. apply function_supported_iff. Qed.
