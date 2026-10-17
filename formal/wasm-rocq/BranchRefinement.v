(* Structured branch simulation from an actual circuit row and memory replay. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Wasm2 WasmIntegers MemoryTrace MemoryCatalog
  TraceRefinement StraightLineSpec LocalBindings LocalSimulation StackBindings StackSimulation
  SpecTecCalls SpecTecBranches BranchBindings BranchSpec.
Import ListNotations.
Open Scope Z_scope.

Theorem block_branch_row_step : forall z initial final row pc prefix tail continuation c0,
  fieldPrime -> FullRelation.satFull row -> row COL_IS_PROGRAM_ROW = 1 ->
  Replay initial (eventsForRow memories row) final -> branchImage initial pc (pc + 1) (pc + 2 + Z.of_nat (length tail)) ->
  blockBranchRepresents z initial (beforeBoundary row) pc prefix tail continuation BeforeBranch c0 ->
  exists position c1, position <> BeforeBranch /\ Steps c0 c1 /\
    blockBranchRepresents z final (afterBoundary row) pc prefix tail continuation position c1.
Proof.
  intros z initial final row pc prefix tail continuation c0 Hprime Hfull Hprogram Hreplay
    Himage [Hz [Htail [Hcontinuation [Hlocals [Hpc [condition [Hstack ->]]]]]]].
  destruct (circuit_br_if_control _ _ _ _ _ _ Hprime Hfull Hreplay Hprogram Hpc Himage)
    as [Hchosen Hnext].
  pose proof Hstack as [Hbase [Hsp [Hbounds Hcells]]].
  apply Forall_app in Hbounds. destruct Hbounds as [Hprefix Hcondition].
  apply Forall_inv in Hcondition.
  assert (Hdepth : 1 <= row COL_SP_BEFORE).
  { cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_br_if_stack_shape row Hprime Hfull Hchosen Hdepth) as Hshape.
  destruct (stack_matches_pop row initial final prefix condition Hshape Hreplay Hstack) as [Hread HstackAfter].
  rewrite Hread in Hnext.
  destruct (full_br_if_locals_unchanged row Hprime Hfull Hchosen) as [Hfbp Hwrites].
  assert (HlocalsAfter : localsMatch final (afterBoundary row) z).
  { eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z); [exact Hfbp | | exact Hlocals].
    eapply no_local_writes_preserve; eassumption. }
  pose proof (br_if_under_block_context z prefix condition tail continuation
    Hz Hprefix Hcondition Htail Hcontinuation) as Hsteps.
  unfold blockBranchResult, branchBlockResult in Hsteps.
  destruct (Z.eqb condition 0) eqn:HconditionZero.
  - exists Fallthrough, (stackPrefix prefix (instructionSuffix continuation (blockContext (stackConfig z [] tail)))).
    split; [discriminate | ]. split; [exact Hsteps | ].
    unfold blockBranchRepresents. repeat first [assumption | reflexivity | split].
  - exists AfterBlock, (stackConfig z prefix continuation).
    split; [discriminate | ]. split; [exact Hsteps | ].
    unfold blockBranchRepresents. repeat first [assumption | reflexivity | split].
Qed.
