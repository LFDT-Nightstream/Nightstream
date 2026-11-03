(* Complete real runs exercise the audited theorem, including the terminal row. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec StackSimulation SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout Fragment
  ProgramSpec ProgramEncoding ControlDecode ExecutionSpec ExecutionRefinement
  tests.BlockPrograms tests.WitnessCheck tests.NopWitness tests.NopTerminalTrace
  tests.TrapTrace.
Import ListNotations.
Open Scope Z_scope.

Definition outerFrame module := MKframe [] module.
Definition entryShape := controlShape [] (siteCode (codePosition 0 [] 0)).

(* Returning outcomes are determined by the row, not chosen by the existential. *)
Lemma terminal_return_decodes : forall program s caller depth root memory boundary rows outcome c row value,
  ExecutionConfig program s caller depth root memory boundary rows outcome c ->
  lastRow rows = Some row -> row COL_ACTIVATION_FINISHED_AFTER = 1 -> row COL_TRAPPED_AFTER = 0 ->
  row COL_OUTPUT_CAPTURED = 1 -> row COL_OUTPUT_VALUE_LO_AFTER = value ->
  c = mk_config (mk_state s caller) [i32Constant value].
Proof.
  intros program s caller depth root memory boundary rows outcome c row value H Hlast Hfinish Htrap Hcapture Hvalue.
  destruct outcome as [shape | | [v | ]]; cbn [ExecutionConfig] in H.
  - destruct H as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct H as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
  - destruct H as [last [Hlast' [_ [_ [[Hcap Hv] ->]]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-.
    congruence.
  - destruct H as [last [Hlast' [_ [_ [Hcap _]]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
Qed.

(* The terminal row also checks the host schedule of export reference 1. *)
Definition exportSchedules (entryCount exitCount : Z) (ram : MemoryState) : MemoryState :=
  fun memory address =>
    if Nat.eqb memory exportEntryScheduleCountId then
      match address with [1] => Some entryCount | _ => None end
    else if Nat.eqb memory exportExitScheduleCountId then
      match address with [1] => Some exitCount | _ => None end
    else ram memory address.
Definition nopCompleteMemory := loadProgram nopProgram (exportSchedules 5 1 nopMemory).
Lemma nop_complete_start : CircuitConfig nopProgram nopStore 0 entryShape nopCompleteMemory nopStart
    (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode).
Proof.
  apply (function_start_config nopProgram nopStore 0 nopGuest [7] _ nopStart); try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact nop_program_validated.
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | i] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | destruct i; discriminate].
  - repeat split.
Qed.

Definition nopComplete := map rowWitness nopTerminalRows.
Lemma nop_complete_valid : validRunb nopCompleteMemory nopComplete = true.
Proof. vm_compute. reflexivity. Qed.
Lemma nop_complete_fragment : executionFragment nopStart 0 nopComplete.
Proof. vm_compute. repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I]. Qed.

Example nop_complete_refines : fieldPrime ->
  Steps (callContext (outerFrame nopModule) 1
    (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode))
    (mk_config (mk_state nopStore (outerFrame nopModule)) [i32Constant 7]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig nopProgram nopStore (outerFrame nopModule) 0 0 entryShape nopCompleteMemory nopStart
      (callContext (outerFrame nopModule) 1 (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ]. exists nopGuest, (stackConfig (mk_state nopStore (activeFrame nopGuest [7])) [] nopCode).
    split; [reflexivity | ]. split; [exact nop_complete_start | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ nop_complete_valid) nop_complete_fragment Hstart)
    as [outcome [c [Hsteps [_ Hconfig]]]].
  replace c with (mk_config (mk_state nopStore (outerFrame nopModule)) [i32Constant 7]) in Hsteps; [exact Hsteps | ].
  symmetry. eapply terminal_return_decodes with (row := rowWitness (List.last nopTerminalRows []));
    [exact Hconfig | reflexivity | | | | ]; vm_compute; reflexivity.
Qed.

Definition trapModule := MKmoduleinst [i32StackType 0 0] [0%N; 1%N] [] [] [] [] [] [].
Definition trapMainCode := [BLOCK (_RESULT None) [CALL (mk_uN 1)]].
Definition trapCalleeCode := [BLOCK (_RESULT None) [UNREACHABLE]].
Definition trapMain := i32Function trapModule (mk_uN 0) 0 0 0 trapMainCode.
Definition trapCallee := i32Function trapModule (mk_uN 0) 0 0 0 trapCalleeCode.
Definition trapProgram := [trapMain; trapCallee].
Definition trapStore := MKstore trapProgram [] [] [] [] [].
Definition trapContext := moduleContext [i32StackType 0 0] [i32StackType 0 0; i32StackType 0 0].
Definition trapGuest := mkGuestFunction 0%N trapModule (mk_uN 0) 0 0 0 trapMainCode 0.

Lemma trap_module_validates : Moduleinst_ok trapStore trapModule trapContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := trapMain);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := trapCallee);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma trap_functions_validate :
  Forall (fun fi => Funcinst_ok trapStore fi (funcinst_TYPE fi)) trapProgram.
Proof.
  constructor; [ | constructor; [ | constructor]].
  all: apply mk_Funcinst_ok with (C := trapContext); try exact trap_module_validates; try solve [repeat constructor].
  all: apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity); try solve [repeat constructor].
  all: apply mk_Expr_ok; try solve [repeat constructor].
  all: apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  all: apply (block _ _ _ [] []); try (apply Blocktype_ok__valtype; repeat constructor); try solve [repeat constructor].
  all: apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
  - apply call; try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma trap_program_validated : programValidated trapProgram trapStore.
Proof.
  split; [exact trap_functions_validate | ].
  intros [ | [ | index]] f Hf; cbn in Hf; [injection Hf as <- | injection Hf as <- | destruct index; discriminate].
  - split.
    + constructor; [apply supported_block; [ | left; reflexivity] | constructor].
      constructor; [ | constructor].
      apply supported_call with (f := mkGuestFunction 1%N trapModule (mk_uN 0) 0 0 0 trapCalleeCode 4);
        [reflexivity | cbn; lia | cbn; lia].
    + vm_compute; tauto.
  - split; [repeat constructor | vm_compute; tauto].
Qed.

Definition trapMemory := loadProgram trapProgram (fun _ _ => Some 0).
Definition trapWitness := map rowWitness trapRows.
Definition trapStart := mkBoundary 0 0 0 0 0 0 0 0 0 0 0.
Lemma trap_run_valid : validRunb trapMemory trapWitness = true.
Proof. vm_compute. reflexivity. Qed.
Lemma trap_run_fragment : executionFragment trapStart 0 trapWitness.
Proof. vm_compute. repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I]. Qed.
Lemma trap_start_config : CircuitConfig trapProgram trapStore 0 entryShape trapMemory trapStart
  (stackConfig (mk_state trapStore (activeFrame trapGuest [])) [] trapMainCode).
Proof.
  apply (function_start_config trapProgram trapStore 0 trapGuest [] _ trapStart); try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact trap_program_validated.
  - constructor.
  - intros index value H; destruct index; discriminate.
  - repeat split.
Qed.

Example nested_trap_refines : fieldPrime ->
  Steps (callContext (outerFrame trapModule) 0
    (stackConfig (mk_state trapStore (activeFrame trapGuest [])) [] trapMainCode))
    (mk_config (mk_state trapStore (outerFrame trapModule)) [admininstr_TRAP]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig trapProgram trapStore (outerFrame trapModule) 0 0 entryShape trapMemory trapStart
      (callContext (outerFrame trapModule) 0 (stackConfig (mk_state trapStore (activeFrame trapGuest [])) [] trapMainCode))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ]. exists trapGuest, (stackConfig (mk_state trapStore (activeFrame trapGuest [])) [] trapMainCode).
    split; [reflexivity | ]. split; [exact trap_start_config | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ trap_run_valid) trap_run_fragment Hstart)
    as [outcome [c [Hsteps [_ Hconfig]]]].
  assert (Hlast : exists row, lastRow trapWitness = Some row /\ row COL_ACTIVATION_FINISHED_AFTER = 1 /\ row COL_TRAPPED_AFTER = 1).
  { exists (rowWitness (List.last trapRows [])). split; [reflexivity | ]. vm_compute; split; reflexivity. }
  destruct Hlast as [row [Hlast [Hfinish Htrap]]].
  destruct outcome as [shape | | result].
  - destruct Hconfig as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct Hconfig as [last [Hlast' [_ [_ Hc]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-.
    rewrite <- Hc; exact Hsteps.
  - destruct Hconfig as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
Qed.
