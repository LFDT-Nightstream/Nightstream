(* Real runs with an indirect call through a table holding double, main and null.
   main(7, 0) calls double, which returns 14; the exported rows meet every premise
   of active_run_refinement. Index 1 names a function of another type, index 2 is
   null and index 5 is out of bounds: each resolve row traps, and so does SpecTec. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals
  StraightLineSpec SpecTecCalls CallerSpec CallerValidation FunctionValidation CodeLayout ProgramSpec ProgramEncoding
  ProgramRefinement ControlDecode ControlInvariant ControlBridge Fragment TableState ExecutionSpec ExecutionRefinement
  tests.BlockPrograms
  tests.WitnessCheck tests.IndirectTrace tests.IndirectTrapTrace tests.IndirectMismatchTrace tests.IndirectNullTrace.
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
Definition tableType := mk_tabletype (mk_limits (mk_uN 3) None) FUNCREF.
Definition indirectTable := MKtableinst tableType [REF_FUNC_ADDR 1%N; REF_FUNC_ADDR 0%N; ref_REF_NULL FUNCREF].
Definition indirectStore := MKstore indirectProgram [] [indirectTable] [] [] [].
Definition mainGuest :=
  mkGuestFunction 0%N indirectModule (mk_uN 1) [valtype_I32; valtype_I32] [valtype_I32] [] mainBody 0.

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
  apply (call_indirect _ _ _ [valtype_I32] [valtype_I32] (mk_limits (mk_uN 3) None));
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

(* The arguments 7 and the index, as the entry rows before the run left them, and
   the table: references are function addresses plus one, null is 0. *)
Definition indirectRam (index : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory stackId then
    match address with [0] => Some 7 | [1] => Some 0 | [2] => Some index | [3] => Some 0 | _ => None end
  else if Nat.eqb memory tableSizeId then match address with [0] => Some 3 | _ => None end
  else if Nat.eqb memory tableElementId then
    match address with [0; 0] => Some 2 | [0; 1] => Some 1 | [0; 2] => Some 0 | _ => None end
  else None.
Definition indirectMemory := loadProgram indirectProgram (indirectRam 0).
Definition indirectWitness := map rowWitness indirectRows.
Definition indirectStart := mkBoundary 0 2 2 0 0 0 0 0 0 0 0.

Lemma indirect_tables_loaded : forall index, tablesLoaded indirectStore (loadProgram indirectProgram (indirectRam index)).
Proof.
  intros index [ | a] t Ht; cbn in Ht; [injection Ht as <- | destruct a; discriminate].
  split; [reflexivity | ]. intros [ | [ | [ | i]]] r Hr; cbn in Hr; try (injection Hr as <-; reflexivity).
  destruct i; discriminate.
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

(* A trapping run from main(7, index): the resolve row is its last row. *)
Definition trapFrame := MKframe [] indirectModule.
Definition trapStart index := stackConfig (mk_state indirectStore (activeFrame mainGuest [7; index])) [] mainBody.

Lemma indirect_trap_refines : forall index rows, fieldPrime -> 0 <= index < i32Modulus ->
  validRunb (loadProgram indirectProgram (indirectRam index)) (map rowWitness rows) = true ->
  executionFragment indirectStart 0 (map rowWitness rows) ->
  (exists row, lastRow (map rowWitness rows) = Some row /\ row COL_ACTIVATION_FINISHED_AFTER = 1 /\
    row COL_TRAPPED_AFTER = 1) ->
  Steps (callContext trapFrame 1 (trapStart index)) (mk_config (mk_state indirectStore trapFrame) [admininstr_TRAP]).
Proof.
  intros index rows Hprime Hindex Hvalid Hfragment [row [Hlast [Hfinish Htrap]]].
  assert (Hconfig : CircuitConfig indirectProgram indirectStore 0 (controlShape [] (siteCode (codePosition 0 [] 0)))
      (loadProgram indirectProgram (indirectRam index)) indirectStart (trapStart index)).
  { apply (function_start_config indirectProgram indirectStore 0 mainGuest [7; index] _ indirectStart);
      try reflexivity; try (cbn; lia).
    - repeat constructor.
    - exact indirect_program_validated.
    - exact (indirect_tables_loaded index).
    - repeat constructor; unfold i32Modulus in *; lia.
    - intros [ | [ | i]] v Hv; cbn in Hv; [injection Hv as <-; reflexivity | injection Hv as <-; reflexivity |
        destruct i; discriminate].
    - repeat split. }
  assert (Hstart : ActivationConfig indirectProgram indirectStore trapFrame 0 0
      (controlShape [] (siteCode (codePosition 0 [] 0))) (loadProgram indirectProgram (indirectRam index))
      indirectStart (callContext trapFrame 1 (trapStart index))).
  { split; [reflexivity | ]. split; [repeat constructor | ]. split; [reflexivity | ].
    exists mainGuest, (trapStart index). split; [reflexivity | ]. split; [exact Hconfig | reflexivity]. }
  destruct (run_refinement _ _ _ _ _ _ _ _ _ _ _ Hprime (load_program_loaded _ _)
    (validRunb_sound _ _ Hvalid) Hfragment Hstart) as [outcome [c [Hsteps [_ Hresult]]]].
  destruct outcome as [shape | | result].
  - destruct Hresult as [Hflag _]. rewrite Hlast in Hflag; congruence.
  - destruct Hresult as [last [Hlast' [_ [_ Hc]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-.
    rewrite <- Hc; exact Hsteps.
  - destruct Hresult as [last [Hlast' [_ [Ht _]]]]. rewrite Hlast in Hlast'; injection Hlast' as <-; congruence.
Qed.

Ltac trap_witness rows :=
  intros Hprime; apply (indirect_trap_refines _ rows Hprime);
  [unfold i32Modulus; lia | vm_compute; reflexivity |
   vm_compute; repeat first [split | left; reflexivity | right; repeat split; reflexivity | reflexivity | exact I] |
   exists (rowWitness (List.last rows [])); split; [reflexivity | vm_compute; split; reflexivity]].

(* Index 5 is past the table's three entries. *)
Example out_of_bounds_traps : fieldPrime ->
  Steps (callContext trapFrame 1 (trapStart 5)) (mk_config (mk_state indirectStore trapFrame) [admininstr_TRAP]).
Proof. trap_witness indirectTrapRows. Qed.

(* Index 1 holds main, whose type is not the expected one. *)
Example type_mismatch_traps : fieldPrime ->
  Steps (callContext trapFrame 1 (trapStart 1)) (mk_config (mk_state indirectStore trapFrame) [admininstr_TRAP]).
Proof. trap_witness indirectMismatchRows. Qed.

(* Index 2 is null. *)
Example null_entry_traps : fieldPrime ->
  Steps (callContext trapFrame 1 (trapStart 2)) (mk_config (mk_state indirectStore trapFrame) [admininstr_TRAP]).
Proof. trap_witness indirectNullRows. Qed.

(* Each run's last row is its resolve row, at pc 2, trapping for its own reason:
   out of bounds (the entry is not read, so it also reads as null), a type mismatch
   after the lookup, and a null entry. *)
Example indirect_trap_rows :
  map (fun rows => let w := rowWitness (List.last rows []) in
    (w COL_PC_BEFORE, w COL_SEL_CALL_INDIRECT_RESOLVE, w COL_CI_OOB, w COL_CI_ENTRY_IS_NULL,
     w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, w COL_CALL_INDIRECT_IS_TRAP))
    [indirectTrapRows; indirectMismatchRows; indirectNullRows] =
  [(2, 1, 1, 1, 0, 1); (2, 1, 0, 0, 1, 1); (2, 1, 0, 1, 0, 1)].
Proof. vm_compute. reflexivity. Qed.

(* At pc 4, main's END, the only control state is main's own frame after the call. *)
Lemma indirect_final_shape : forall cursor memory boundary c,
  ControlState indirectProgram indirectStore 0 cursor memory boundary c -> boundaryPC boundary = 4 ->
  shapeOf cursor = controlShape [] (siteCode (codePosition 0 [] 3)).
Proof.
  intros [stack active] memory boundary c [f [inner [Hf [Hactive [Hsusp [_ [_ ->]]]]]]] Hpc.
  cbn [activePosition suspendedCalls] in *.
  destruct active as [[[index labels offset] segments values locals] | view callee | function arguments].
  - destruct Hactive as [f' [[Hf' [Hpath _]] [Hoffset [_ [Hpos _]]]]].
    cbn [activeFunction viewPosition positionFunction positionLabels positionOffset] in *.
    rewrite Hf in Hf'. injection Hf' as <-.
    destruct index as [ | [ | index]]; [ | | destruct index; discriminate]; injection Hf as <-;
      (destruct labels as [ | b more]; [ | destruct b as [[ | [ | [ | [ | b]]]] arm]; cbn in Hpath; discriminate]);
      unfold innerCode in Hoffset; cbn in Hoffset, Hpos.
    + destruct offset as [ | [ | [ | [ | offset]]]]; cbn in Hpos; try lia.
      destruct stack as [ | top more]; [reflexivity | exfalso].
      destruct Hsusp as [x [[Hg _] [Hcall [Hcallee _]]]].
      destruct (callView top) as [[[ | [ | caller]] glabels goffset] gsegments gvalues glocals];
        cbn [viewPosition positionFunction positionLabels positionOffset] in Hg, Hcall, Hcallee;
        [ | | destruct caller; discriminate]; cbn in Hg; injection Hg as Hg; rewrite <- Hg in Hcall, Hcallee;
        unfold innerCode in Hcall; (destruct glabels as [ | b more'];
          [ | destruct b as [[ | [ | [ | [ | b]]]] arm]; cbn in Hcall; destruct goffset; discriminate]);
        cbn in Hcall;
        destruct goffset as [ | [ | [ | goffset]]]; try discriminate; try (destruct goffset; discriminate);
        injection Hcall as <-; cbn in Hcallee; first [contradiction | discriminate Hcallee].
    + destruct offset as [ | [ | [ | [ | offset]]]]; cbn in Hpos; lia.
  - exfalso. destruct Hactive as [f' [g [i [[Hf' [Hpath _]] [Hi [Hind [_ [_ [_ [_ [Hpos _]]]]]]]]]]].
    destruct view as [[index labels offset] segments values locals].
    cbn [activeFunction viewPosition positionFunction positionLabels positionOffset] in *.
    rewrite Hf in Hf'. injection Hf' as <-.
    destruct index as [ | [ | index]]; [ | | destruct index; discriminate]; injection Hf as <-;
      unfold innerCode in Hi; (destruct labels as [ | b more];
        [ | destruct b as [[ | [ | [ | [ | b]]]] arm]; cbn in Hi; destruct offset; discriminate]);
      cbn in Hi, Hpos; destruct offset as [ | [ | [ | offset]]]; try discriminate; try (destruct offset; discriminate);
      injection Hi as <-; try discriminate Hind; unfold codeOffset, mainBody in Hpos; cbn in Hpos; lia.
  - exfalso. destruct Hactive as [f' [Hf' [_ [_ [_ [_ [_ [_ [Hpos _]]]]]]]]].
    destruct function as [ | [ | function]]; try (destruct function; discriminate);
      injection Hf' as <-; cbn in Hpos; lia.
Qed.

(* The run ends at main's END, where the state decodes to main's frame holding 14.
   Both are computed on the goal: reducing a hypothesis instead leaves the kernel to
   recheck it by lazy conversion, which does not finish on a replayed trace. *)
Lemma indirect_final_pc : boundaryPC (rowsFinalBoundary indirectStart indirectWitness) = 4.
Proof. vm_compute. reflexivity. Qed.

Lemma indirect_final_decode :
  option_map fst (decodeControl indirectProgram indirectStore 0 (controlShape [] (siteCode (codePosition 0 [] 3)))
    (replayed indirectMemory indirectWitness) (rowsFinalBoundary indirectStart indirectWitness)) =
  Some (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 0])) [14] []).
Proof. vm_compute. reflexivity. Qed.

(* main(7, 0) runs to 14: the run's final state, whatever its shape, is main's frame
   holding only double's result. *)
Example indirect_call_returns_double : fieldPrime ->
  Steps (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 0])) [] mainBody)
    (stackConfig (mk_state indirectStore (activeFrame mainGuest [7; 0])) [14] []).
Proof.
  intros Hprime.
  destruct (indirect_witness_refines Hprime) as [shape [c [Hsteps Hconfig]]].
  destruct (circuit_config_state _ _ _ _ _ _ _ Hconfig) as [cursor [Hshape Hstate]].
  pose proof (indirect_final_shape _ _ _ _ Hstate indirect_final_pc) as Hfinal.
  destruct Hconfig as [_ [_ [_ [_ [obligations [Hdecode _]]]]]].
  rewrite <- Hshape, Hfinal in Hdecode.
  pose proof indirect_final_decode as Hresult. rewrite Hdecode in Hresult.
  injection Hresult as <-. exact Hsteps.
Qed.
