(* A real run with an indirect call: main(7, 0) calls table entry 0, double, which
   returns 14. The exported rows meet every premise of active_run_refinement. With
   index 5 the resolve row traps, and so does SpecTec. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode Fragment TableState ExecutionSpec ExecutionRefinement tests.BlockPrograms
  tests.WitnessCheck tests.IndirectTrace tests.IndirectTrapTrace.
Import ListNotations.
Open Scope Z_scope.

(* main: local.get 0; local.get 1; call_indirect (type 0). double: local.get 0; local.get 0; i32.add *)
Definition mainBody := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1); CALL_INDIRECT (mk_uN 0) (mk_uN 0)].
Definition doubleBody := [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Definition indirectTypes := [i32StackType 1 1; i32StackType 2 1].
Definition indirectModule := MKmoduleinst indirectTypes [0%N; 1%N] [] [0%N] [] [] [] [].
Definition mainFunction := i32Function indirectModule (mk_uN 1) 2 1 0 mainBody.
Definition doubleFunction := i32Function indirectModule (mk_uN 0) 1 1 0 doubleBody.
Definition indirectProgram := [mainFunction; doubleFunction].
Definition tableType := mk_tabletype (mk_limits (mk_uN 1) None) FUNCREF.
Definition indirectTable := MKtableinst tableType [REF_FUNC_ADDR 1%N].
Definition indirectStore := MKstore indirectProgram [] [indirectTable] [] [] [].
Definition mainGuest := mkGuestFunction 0%N indirectModule (mk_uN 1) 2 1 0 mainBody 0.

Definition indirectContext := MKcontext indirectTypes [i32StackType 2 1; i32StackType 1 1] [] [tableType]
  [] [] [] [] [] None.

Example indirect_module_validates : Moduleinst_ok indirectStore indirectModule indirectContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  all: first
    [ solve [eapply Externaddr_ok__table with (v_tableinst := indirectTable);
        try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__func with (v_funcinst := mainFunction);
        try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor]
    | solve [eapply Externaddr_ok__func with (v_funcinst := doubleFunction);
        try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor] ].
Qed.

Lemma indirect_function_validates : forall typeIndex params body,
  is_true (N.ltb (typeIndex :> N) 2) -> lookup_total indirectTypes (typeIndex :> N) = i32StackType params 1 ->
  wf_uN 32%N typeIndex -> Forall wf_instr body ->
  Instrs_ok (functionContext indirectContext (i32Function indirectModule typeIndex params 1 0 body)) body
    (i32StackType 0 1) ->
  Funcinst_ok indirectStore (i32Function indirectModule typeIndex params 1 0 body) (i32StackType params 1).
Proof.
  intros typeIndex params body Hbound Hlookup Hwu Hw Hv.
  apply mk_Funcinst_ok with (C := indirectContext).
  all: try exact indirect_module_validates.
  all: try (apply mk_Func_ok).
  all: first [exact Hbound | apply (introT eqP); exact Hlookup |
    apply mk_Expr_ok; [exact Hv | repeat constructor | exact Hw] |
    solve [repeat constructor; first [exact Hw | exact Hwu]] | solve [repeat constructor]].
Qed.

Section Typing.
Local Open Scope nat_scope.
Example main_body_validates : Instrs_ok (functionContext indirectContext mainFunction) mainBody (i32StackType 0 1).
Proof.
  typed_step 0 1 0. typed_step 0 1 1.
  apply (i32_validation_frame _ _ 2 1 0); [ | repeat constructor | repeat constructor].
  change (Instr_ok (functionContext indirectContext mainFunction) (CALL_INDIRECT (mk_uN 0) (mk_uN 0))
    (mk_functype (mk_list _ ([valtype_I32] ++ [valtype_I32])) (mk_list _ [valtype_I32]))).
  apply (call_indirect _ _ _ [valtype_I32] [valtype_I32] (mk_limits (mk_uN 1) None));
    try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Example double_body_validates :
  Instrs_ok (functionContext indirectContext doubleFunction) doubleBody (i32StackType 0 1).
Proof. typed_step 0 1 0. typed_step 0 1 1. typed_instr 2 1 0. Qed.
End Typing.

Example indirect_program_validated : programValidated indirectProgram indirectStore.
Proof.
  split.
  - constructor; [ | constructor; [ | constructor]].
    + apply indirect_function_validates; [reflexivity | reflexivity | repeat constructor | repeat constructor |
        exact main_body_validates].
    + apply indirect_function_validates; [reflexivity | reflexivity | repeat constructor | repeat constructor |
        exact double_body_validates].
  - intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- .. | destruct index; discriminate].
    + split; [ | vm_compute; tauto].
      repeat apply Forall_cons; try apply Forall_nil; try apply supported_local_get.
      apply (supported_call_indirect _ _ _ 1 1); [reflexivity | lia | lia | ].
      intros [ | [ | a]] fi Hfi; cbn in Hfi; [ | | destruct a; discriminate].
      1-2: eexists; split; reflexivity.
    + split; [ | vm_compute; tauto]. repeat constructor.
Qed.

(* The arguments 7 and 0, as the entry rows before the run left them, and the table. *)
Definition indirectRam (index : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some index | [3] => Some 0 | _ => None end
  else if Nat.eqb memory tableSizeId then match address with [0] => Some 1 | _ => None end
  else if Nat.eqb memory tableElementId then match address with [0; 0] => Some 2 | _ => None end
  else None.
Definition indirectMemory := loadProgram indirectProgram (indirectRam 0).
Definition indirectWitness := map rowWitness indirectRows.
Definition indirectStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma indirect_tables_loaded : forall index, tablesLoaded indirectStore (loadProgram indirectProgram (indirectRam index)).
Proof.
  intros index [ | a] t Ht; cbn in Ht; [injection Ht as <- | destruct a; discriminate].
  split; [reflexivity | ]. intros [ | i] r Hr; cbn in Hr; [injection Hr as <-; reflexivity | destruct i; discriminate].
Qed.

Lemma indirect_run_checks : validRunb indirectMemory indirectWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma indirect_fragment : runFragment indirectStart 0 indirectWitness.
Proof. split; [vm_compute; reflexivity | ]. apply callerRowb_sound. vm_compute. reflexivity. Qed.

Lemma indirect_start_config :
  CircuitConfig indirectProgram indirectStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    indirectMemory indirectStart (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 0])) [] mainBody).
Proof.
  apply (function_start_config indirectProgram indirectStore 0 mainGuest [7; 0] _ indirectStart);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact indirect_program_validated.
  - exact (indirect_tables_loaded 0).
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

(* The refinement's conclusion for this run. *)
Example indirect_witness_refines : fieldPrime ->
  exists shape c,
    Steps (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 0])) [] mainBody) c /\
    CircuitConfig indirectProgram indirectStore 0 shape (replayed indirectMemory indirectWitness)
      (rowsFinalBoundary indirectStart indirectWitness) c.
Proof.
  intros Hprime.
  destruct (active_run_refinement _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ indirect_run_checks) indirect_fragment indirect_start_config)
    as [shape [c [Hsteps [_ Hconfig]]]].
  exists shape, c. split; assumption.
Qed.

(* The resolve row at pc 2 leaves the reference, 2, on top for the invocation at pc 3. *)
Example indirect_rows_split :
  let rows := [nth 2 indirectWitness (rowWitness []); nth 3 indirectWitness (rowWitness [])] in
  map (fun w : Witness => (w COL_PC_BEFORE, w COL_SEL_CALL_INDIRECT_RESOLVE, w COL_SEL_CALL_INDIRECT,
    w COL_FUNCTION_REF)) rows = [(2, 1, 0, 2); (3, 0, 1, 2)].
Proof. vm_compute. reflexivity. Qed.

(* main(7, 5): the index is past the table's one entry, so the call traps. *)
Definition trapFrame := MKframe [] indirectModule.
Definition trapMemory := loadProgram indirectProgram (indirectRam 5).
Definition trapWitness := map rowWitness indirectTrapRows.

Lemma trap_run_valid : validRunb trapMemory trapWitness = true.
Proof. vm_compute. reflexivity. Qed.

Lemma trap_run_fragment : executionFragment indirectStart 0 trapWitness.
Proof. vm_compute. repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I]. Qed.

Lemma trap_start_config :
  CircuitConfig indirectProgram indirectStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
    trapMemory indirectStart (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 5])) [] mainBody).
Proof.
  apply (function_start_config indirectProgram indirectStore 0 mainGuest [7; 5] _ indirectStart);
    try reflexivity; try (cbn; lia).
  - repeat constructor.
  - exact indirect_program_validated.
  - exact (indirect_tables_loaded 5).
  - repeat constructor; unfold i32Modulus; lia.
  - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
      destruct i; discriminate].
  - repeat split.
Qed.

Example indirect_trap_refines : fieldPrime ->
  Steps (callContext trapFrame 1 (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 5])) [] mainBody))
    (mk_config (mk_state indirectStore trapFrame) [admininstr_TRAP]).
Proof.
  intros Hprime.
  assert (Hstart : ActivationConfig indirectProgram indirectStore trapFrame 0 0
      (controlShape [] (siteCode (codePosition 0 [] 0))) trapMemory indirectStart
      (callContext trapFrame 1 (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 5])) [] mainBody))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ].
    exists mainGuest, (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 5])) [] mainBody).
    split; [reflexivity | ]. split; [exact trap_start_config | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ trap_run_valid) trap_run_fragment Hstart)
    as [outcome [c [Hsteps [_ Hconfig]]]].
  assert (Hlast : exists row, lastRow trapWitness = Some row /\ row COL_ACTIVATION_FINISHED_AFTER = 1 /\
      row COL_TRAPPED_AFTER = 1).
  { exists (rowWitness (List.last indirectTrapRows [])). split; [reflexivity | ]. vm_compute; split; reflexivity. }
  destruct Hlast as [row [Hlast [Hfinish Htrap]]].
  destruct outcome as [shape | | result].
  - destruct Hconfig as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct Hconfig as [last [Hlast' [_ [_ Hc]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-.
    rewrite <- Hc; exact Hsteps.
  - destruct Hconfig as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
Qed.

(* The trapping row is the resolve row, at pc 2: its index is out of bounds. *)
Example indirect_trap_row :
  let w := rowWitness (List.last indirectTrapRows []) in
  (w COL_PC_BEFORE, w COL_SEL_CALL_INDIRECT_RESOLVE, w COL_CI_OOB, w COL_CALL_INDIRECT_IS_TRAP) = (2, 1, 1, 1).
Proof. vm_compute. reflexivity. Qed.
