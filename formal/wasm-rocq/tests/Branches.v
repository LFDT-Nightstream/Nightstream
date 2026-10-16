From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog Wasm2 WasmIntegers Locals
  StraightLineSpec StraightLineValidation SpecTecCalls SpecTecBranches BranchBindings BranchSpec BranchRefinement.
Import ListNotations.
Open Scope Z_scope.

Definition branchState := mk_state (MKstore [] [] [] [] [] [])
  (MKframe [i32Value 9] (MKmoduleinst [] [] [] [] [] [] [] [])).
Definition branchTail := [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 0)].
Definition branchContinuation := [LOCAL_GET (mk_uN 0)].

(* PC 10 is br_if, 11..12 its tail, 13 the block END, and 14 the continuation.
   The 42 below the block label must survive either outcome. *)
Definition branchMemory condition : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then
    match address with [8] => Some 42 | [10] => Some condition | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some 9 | _ => None end
  else if Nat.eqb memory programOpcodeId then
    match address with [10] => Some CODE_BR_IF | _ => None end
  else if Nat.eqb memory pcEdgeKindId then
    match address with [10] => Some 0 | _ => None end
  else if Nat.eqb memory pcRomId then
    match address with
    | [10; choice] => if choice =? 0 then Some 11 else if choice =? 1 then Some 14 else None
    | _ => None
    end
  else None.

Example branch_initial_representation : forall condition,
  0 <= condition < i32Modulus ->
  blockBranchImage (branchMemory condition) 10 branchTail /\
  blockBranchRepresents branchState (branchMemory condition) (mkBoundary 10 6 4 7 0 0 0 0 0)
    10 [42] branchTail branchContinuation BeforeBranch
    (blockBranchConfig branchState [42] condition branchTail branchContinuation).
Proof.
  intros condition Hbound. split.
  - split.
    + repeat constructor; eexists; reflexivity.
    + unfold branchImage. repeat split; reflexivity.
  - split; [repeat constructor | ]. split; [repeat constructor | ].
    split; [repeat constructor | ]. split.
    + intros [ | index] value Hvalue.
      * cbn in Hvalue. injection Hvalue as <-. exists 9. repeat split; try reflexivity; unfold i32Modulus; lia.
      * destruct index; discriminate.
    + split; [reflexivity | ]. exists condition. split; [ | reflexivity].
      unfold stackMatches. cbn. split; [lia | ]. split; [reflexivity | ]. split.
      * constructor; [unfold i32Modulus; lia | constructor; [exact Hbound | constructor]].
      * intros [ | [ | index]] value Hvalue; cbn in Hvalue.
        -- injection Hvalue as <-. reflexivity.
        -- injection Hvalue as <-. reflexivity.
        -- destruct index; discriminate.
Qed.

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
