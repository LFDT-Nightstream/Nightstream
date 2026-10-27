From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range Wasm2 WasmIntegers Locals StraightLineSpec SpecTecCalls SpecTecBranches
  BranchBindings.
Import ListNotations.
Open Scope Z_scope.

Definition branchState := mk_state (MKstore [] [] [] [] [] [])
  (MKframe [i32Value 9] (MKmoduleinst [] [] [] [] [] [] [] [])).
Definition branchTail := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 0)].
Definition branchContinuation := [LOCAL_GET (mk_uN 0)].

Example zero_condition_falls_through :
  Steps (stackConfig branchState [0] (BR_IF (mk_uN 0) :: branchTail)) (stackConfig branchState [] branchTail).
Proof.
  exact (br_if_steps branchState 0 (mk_uN 0) branchTail ltac:(repeat constructor)
    ltac:(unfold i32Modulus; lia) ltac:(repeat constructor) ltac:(repeat constructor)).
Qed.

(* A non-boolean condition takes the branch: br_if 1 inside two blocks leaves both,
   dropping the inner block's continuation. *)
Example non_boolean_condition_exits_two_labels :
  Steps (blockContext (instructionSuffix branchContinuation
      (blockContext (stackConfig branchState [7] (BR_IF (mk_uN 1) :: branchTail)))))
    (stackConfig branchState [] []).
Proof.
  assert (Hz : wf_state branchState) by repeat constructor.
  assert (Hl : wf_uN 32%N (mk_uN 1)) by repeat constructor.
  assert (Htail : Forall wf_instr branchTail) by repeat constructor.
  assert (Hcont : Forall wf_instr branchContinuation) by repeat constructor.
  pose proof (br_if_steps branchState 7 (mk_uN 1) branchTail Hz ltac:(unfold i32Modulus; lia) Hl Htail) as Hbr.
  cbn [Z.eqb] in Hbr.
  pose proof (spec_steps_transitive _ _ _ (steps_in_block _ _ Hbr) (br_exit_label_carrying branchState [] 0 branchTail Hz (Forall_nil _) Hl Htail))
    as Hinner.
  pose proof (steps_in_block _ _ (steps_with_instruction_suffix branchContinuation _ _ Hcont Hinner)) as Houter.
  eapply spec_steps_transitive; [exact Houter | exact (br_exit_target_dropping branchState [] branchContinuation Hz (Forall_nil _) Hcont)].
Qed.

Example wrong_branch_choice_rejected : forall row,
  fieldPrime -> FullRelation.satFull row -> row COL_SEL_BR_IF_EQZ = 1 ->
  row (nth 0%nat COL_STACK_READ_VALUE_LO 0%nat) = 7 -> row COL_CONTROL_CHOICE <> 0.
Proof.
  intros row Hprime Hfull Hselector Hcondition.
  pose proof (full_br_if_choice row Hprime Hfull Hselector) as Hchoice.
  rewrite Hcondition in Hchoice. cbn in Hchoice. congruence.
Qed.
