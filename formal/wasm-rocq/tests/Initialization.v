(* Circuit-only regression for frame initialization: after a guest call into a
   function with locals, the next row cannot already be a program row. The main
   refinement depends on this too; here it fails close to the cause. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog TraceRefinement
  CallInitialization CallBindings.
Import ListNotations.
Open Scope Z_scope.

Theorem program_resumption_requires_initialized_frame : forall previous next,
  fieldPrime -> FullRelation.satFull previous -> FullRelation.satFull next ->
  Follows MemoryCatalog.continuityLinks previous next -> next COL_IS_PROGRAM_ROW = 1 ->
  previous COL_LOCAL_ZERO_REMAINING_AFTER = 0 /\
  previous COL_PARAM_INIT_REMAINING_AFTER = 0.
Proof.
  intros previous next Hprime Hprevious Hnext Hlink Hprogram.
  pose proof (full_initialization_flags previous Hprime Hprevious) as [Hz Hp].
  assert (Hza : previous COL_LOCAL_ZERO_ACTIVE_AFTER = 0).
  { rewrite (Hlink COL_LOCAL_ZERO_ACTIVE_AFTER COL_LOCAL_ZERO_ACTIVE_BEFORE ltac:(vm_compute; tauto)).
    eapply full_row_kind_exclusive; try eassumption; cbn; tauto || discriminate. }
  assert (Hpa : previous COL_PARAM_INIT_ACTIVE_AFTER = 0).
  { rewrite (Hlink COL_PARAM_INIT_ACTIVE_AFTER COL_PARAM_INIT_ACTIVE_BEFORE ltac:(vm_compute; tauto)).
    eapply full_row_kind_exclusive; try eassumption; cbn; tauto || discriminate. }
  rewrite Hza in Hz. rewrite Hpa in Hp.
  destruct (previous COL_LOCAL_ZERO_REMAINING_AFTER =? 0) eqn:Hzero; [ | discriminate].
  destruct (previous COL_PARAM_INIT_REMAINING_AFTER =? 0) eqn:Hparam; [ | discriminate].
  split; apply Z.eqb_eq; assumption.
Qed.

Theorem guest_call_cannot_skip_initialization : forall callRow next,
  fieldPrime -> FullRelation.satFull callRow -> FullRelation.satFull next ->
  callRow COL_SEL_CALL = 1 -> callRow COL_TARGET_FUNCTION_IS_GUEST = 1 ->
  callRow COL_ENTERED_FUNCTION_NUM_LOCALS > 0 ->
  next COL_IS_PROGRAM_ROW = 1 ->
  ~ Follows MemoryCatalog.continuityLinks callRow next.
Proof.
  intros callRow next Hprime Hcall Hnext Hselected Htarget Hlocals Hprogram Hlink.
  destruct (full_direct_guest_call callRow Hprime Hcall Hselected Htarget) as [Hguest _].
  pose proof (full_selected_is_program_row callRow COL_SEL_CALL Hprime Hcall
    ltac:(vm_compute; tauto) Hselected) as HcallProgram.
  destruct (full_guest_initialization_seed callRow Hprime Hcall HcallProgram Hguest) as [Hz Hp].
  destruct (program_resumption_requires_initialized_frame callRow next Hprime Hcall Hnext Hlink Hprogram).
  lia.
Qed.
