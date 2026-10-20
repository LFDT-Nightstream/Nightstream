(* Typing contexts that agree with the resolved callees' function types, and
   i32 instruction typing above extra operands. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import Wasm2 StraightLineSpec StraightLineInstructions StraightLineValidation
  SpecTecCalls CallSpec CallerSpec CallerRefinement FunctionValidation.
Import ListNotations.
Open Scope Z_scope.

Definition calleeTypes (C : context) (callees : CalleeEnvironment) : Prop :=
  forall x f, callees x = Some f ->
    nth_error (context_FUNCS C) (N.to_nat x) = Some (i32StackType (guestParams f) (guestResults f)).

Lemma i32_validation_frame : forall C i inputs outputs extra,
  Instr_ok C i (i32StackType inputs outputs) -> wf_context C -> wf_instr i ->
  Instrs_ok C [i] (i32StackType (extra + inputs) (extra + outputs)).
Proof.
  intros. unfold i32StackType. rewrite !repeat_app.
  apply Instrs_ok__frame; [apply Instrs_ok__instr; assumption | assumption | constructor; [assumption | constructor]].
Qed.
