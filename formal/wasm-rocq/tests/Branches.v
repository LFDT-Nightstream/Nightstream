From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range Wasm2 WasmIntegers Locals StraightLineSpec SpecTecCalls SpecTecBranches
  BranchBindings.
Import ListNotations.
Open Scope Z_scope.

Definition branchState := mk_state (MKstore [] [] [] [] [] [])
  (MKframe [i32Value 9] (MKmoduleinst [] [] [] [] [] [] [] [])).
(* A br_if under a block label, with 42 below the label. *)
Definition branchTail := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 0)].
Definition branchContinuation := [LOCAL_GET (mk_uN 0)].

Example zero_condition_keeps_label :
  Steps (blockBranchConfig branchState [42] 0 branchTail branchContinuation)
    (stackPrefix [42] (instructionSuffix branchContinuation (blockContext (stackConfig branchState [] branchTail)))).
Proof.
  exact (br_if_under_block_context branchState [42] 0 branchTail branchContinuation
    ltac:(repeat constructor) ltac:(constructor; [unfold i32Modulus; lia | constructor])
    ltac:(unfold i32Modulus; lia) ltac:(repeat constructor) ltac:(repeat constructor)).
Qed.

Example non_boolean_condition_exits_label :
  Steps (blockBranchConfig branchState [42] 7 branchTail branchContinuation)
    (stackConfig branchState [42] branchContinuation).
Proof.
  exact (br_if_under_block_context branchState [42] 7 branchTail branchContinuation
    ltac:(repeat constructor) ltac:(constructor; [unfold i32Modulus; lia | constructor])
    ltac:(unfold i32Modulus; lia) ltac:(repeat constructor) ltac:(repeat constructor)).
Qed.

Example wrong_branch_choice_rejected : forall row,
  fieldPrime -> FullRelation.satFull row -> row COL_SEL_BR_IF_EQZ = 1 ->
  row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = 7 -> row COL_CONTROL_CHOICE <> 0.
Proof.
  intros row Hprime Hfull Hselector Hcondition.
  pose proof (full_br_if_choice row Hprime Hfull Hselector) as Hchoice.
  rewrite Hcondition in Hchoice. cbn in Hchoice. congruence.
Qed.
