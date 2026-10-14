From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 StraightLineSpec BlockSpec BlockValidation.
Import ListNotations.

Definition computedBranches :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BINOP I32 (mk_binop__0 Inn_I32 SUB);
   BR_IF (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 0);
   LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1)].

Example computed_branches_supported : supportedBranchBody computedBranches.
Proof.
  unfold supportedBranchBody, computedBranches.
  repeat (constructor; [first [left; reflexivity | right; eexists; reflexivity] | ]).
  constructor.
Qed.

Example computed_branches_balanced : branchSitesBalanced computedBranches 0.
Proof. repeat split; reflexivity. Qed.

Example computed_branches_validate : typedBlock 2 computedBranches.
Proof.
  apply branch_facts_validate.
  - exact computed_branches_supported.
  - repeat constructor.
  - repeat constructor; cbn; lia.
  - reflexivity.
Qed.

Example remainder_after_sub_validates :
  Instrs_ok (blockTypingContext 2) (skipn 3 computedBranches) (i32StackType 1 0).
Proof.
  pose proof (branch_validation_tail 2 0 0 _ _ 1 computed_branches_validate
    computed_branches_supported eq_refl) as H1.
  pose proof (branch_validation_tail 2 1 0 _ _ 2 H1
    (Forall_inv_tail computed_branches_supported) eq_refl) as H2.
  exact (branch_validation_tail 2 2 0 _ _ 1 H2
    (Forall_inv_tail (Forall_inv_tail computed_branches_supported)) eq_refl).
Qed.

Example remainder_after_sub_balanced : branchSitesBalanced (skipn 3 computedBranches) 1.
Proof.
  pose proof (balanced_tail _ _ _ _ computed_branches_balanced eq_refl) as H1.
  pose proof (balanced_tail _ _ _ _ H1 eq_refl) as H2.
  exact (balanced_tail _ _ _ _ H2 eq_refl).
Qed.

(* SpecTec permits the taken branch to discard the first local.get's value.
   The circuit currently only removes the condition, so support must reject it. *)
Definition unwindingBody :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); BR_IF (mk_uN 0); LOCAL_SET (mk_uN 0)].

Example unwinding_body_validates : typedBlock 2 unwindingBody.
Proof.
  apply branch_facts_validate.
  - unfold supportedBranchBody, unwindingBody.
    repeat (constructor; [first [left; reflexivity | right; eexists; reflexivity] | ]).
    constructor.
  - repeat constructor.
  - repeat constructor; cbn; lia.
  - reflexivity.
Qed.

Example unwinding_body_is_not_balanced : ~ branchSitesBalanced unwindingBody 0.
Proof. intros [H _]; discriminate. Qed.
