From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog TraceRefinement Wasm2 WasmIntegers Locals SpecTecCalls
  StraightLineSpec CallerSpec ProgramSpec ProgramEncoding ProgramRefinement FunctionValidation
  ControlDecode Fragment.
Import ListNotations.
Open Scope Z_scope.

(* top calls relay, which calls readZero: two callers suspended at once.
   readZero at PCs 0..1, relay at 2..3, top at 4..5. *)
Definition nestedModule := MKmoduleinst [i32StackType 0 1] [0%N; 1%N] [] [] [] [] [] [].
Definition nestedFunction (body : list instr) := i32Function nestedModule (mk_uN 0) 0 1 1 body.
Definition nestedProgram :=
  [nestedFunction [LOCAL_GET (mk_uN 0)]; nestedFunction [CALL (mk_uN 0)]; nestedFunction [CALL (mk_uN 1)]].
Definition nestedStore := MKstore nestedProgram [] [] [] [] [].
Definition nestedGuest (address : N) (body : list instr) (entry : Z) :=
  mkGuestFunction address nestedModule (mk_uN 0) 0 1 1 body entry.

Example nested_layout :
  functionEntry nestedProgram 1 = 2 /\ functionEntry nestedProgram 2 = 4 /\
  functionAt nestedProgram 2 = Some (nestedGuest 2 [CALL (mk_uN 1)] 4).
Proof. repeat split; reflexivity. Qed.

Definition nestedContext :=
  moduleContext [i32StackType 0 1] [i32StackType 0 1; i32StackType 0 1].
Definition nestedBodyContext := functionContext nestedContext (nestedFunction []).

Example nested_module_validates : Moduleinst_ok nestedStore nestedModule nestedContext.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := nestedFunction [LOCAL_GET (mk_uN 0)]);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := nestedFunction [CALL (mk_uN 0)]);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma nested_function_validates : forall body,
  Forall wf_instr body -> Instrs_ok nestedBodyContext body (i32StackType 0 1) ->
  Funcinst_ok nestedStore (nestedFunction body) (i32StackType 0 1).
Proof.
  intros body Hw Hv. apply mk_Funcinst_ok with (C := nestedContext).
  - repeat constructor.
  - exact nested_module_validates.
  - apply mk_Func_ok; try reflexivity; try (apply (introT eqP); reflexivity).
    + repeat constructor.
    + apply mk_Expr_ok; [exact Hv | repeat constructor | exact Hw].
    + repeat constructor.
    + repeat constructor; exact Hw.
    + repeat constructor.
  - repeat constructor.
  - repeat constructor.
  - repeat constructor; exact Hw.
Qed.

Example nested_call_validates : forall index,
  (index < 2)%nat -> Instrs_ok nestedBodyContext [CALL (mk_uN (N.of_nat index))] (i32StackType 0 1).
Proof.
  intros [ | [ | index]] Hindex; [ | | lia];
    (apply Instrs_ok__instr; [apply call | | ]; try reflexivity; try (apply (introT eqP); reflexivity);
     repeat constructor).
Qed.

Example nested_program_validated : programValidated nestedProgram nestedStore.
Proof.
  split.
  - constructor; [ | constructor; [ | constructor; [ | constructor]]]; apply nested_function_validates;
      try solve [repeat constructor].
    + exact (nested_call_validates 0 ltac:(lia)).
    + exact (nested_call_validates 1 ltac:(lia)).
  - intros [ | [ | [ | index]]] g Hg; cbn in Hg; [injection Hg as <- .. | destruct index; discriminate].
    + split; [repeat constructor | vm_compute; tauto].
    + split; [ | vm_compute; tauto]. constructor; [ | constructor].
      apply supported_call with (f := nestedGuest 0 [LOCAL_GET (mk_uN 0)] 0); [reflexivity | cbn; lia | cbn; lia].
    + split; [ | vm_compute; tauto]. constructor; [ | constructor].
      apply supported_call with (f := nestedGuest 1 [CALL (mk_uN 0)] 2); [reflexivity | cbn; lia | cbn; lia].
Qed.

(* Every local is zero; each frame is its one local: top's at cell 7, relay's at
   8, readZero's at 9. The saved contexts return to top at pc 5 (level 0) and to
   relay at pc 3 (level 1), with each caller's frame and operand bases. *)
Definition nestedMemory := loadProgram nestedProgram (fun memory address =>
  if Nat.eqb memory stackId then
    match address with [14] | [16] | [18] => Some 0 | _ => None end
  else if Nat.eqb memory callStackReturnPcId then match address with [0] => Some 5 | [1] => Some 3 | _ => None end
  else if Nat.eqb memory callStackCallerFbpId then match address with [0] => Some 7 | [1] => Some 8 | _ => None end
  else if Nat.eqb memory callStackCallerSpBaseId then match address with [0] => Some 8 | [1] => Some 9 | _ => None end
  else None).

Definition zeroFrame := MKframe [i32Value 0] nestedModule.

(* readZero runs inside relay's FRAME_/LABEL_, itself inside top's. *)
Example nested_suspended_calls_decode :
  option_map fst (decodeControl nestedProgram nestedStore 0
    (controlShape [codePosition 1 [] 0; codePosition 2 [] 0] (siteCode (codePosition 0 [] 0)))
    nestedMemory (mkBoundary 0 10 10 9 2 0 0 0 0 0)) =
  Some (mk_config (mk_state nestedStore zeroFrame)
    [FRAME_ 1%N zeroFrame [LABEL_ 1%N [] [FRAME_ 1%N zeroFrame [LABEL_ 1%N []
      [admininstr_instr (LOCAL_GET (mk_uN 0))]]]]]).
Proof. vm_compute. reflexivity. Qed.

(* The start of top: the theorem's premise holds for a nested program. *)
Example nested_initial_config :
  CircuitConfig nestedProgram nestedStore 0 (controlShape [] (siteCode (codePosition 2 [] 0)))
    nestedMemory (mkBoundary 4 8 8 7 0 0 0 0 0 0)
    (mk_config (mk_state nestedStore zeroFrame) [admininstr_instr (CALL (mk_uN 1))]).
Proof.
  split; [reflexivity | ]. split; [repeat constructor | ]. split; [exact nested_program_validated | ].
  exists True. split; [vm_compute; reflexivity | exact I].
Qed.

(* Any run of top's rows refines SpecTec, through both nested calls and back. *)
Example nested_program_refines : forall rows final,
  fieldPrime -> validCircuitRun nestedMemory rows final ->
  runFragment (mkBoundary 4 8 8 7 0 0 0 0 0 0) 0 rows ->
  exists shape c, Steps (mk_config (mk_state nestedStore zeroFrame) [admininstr_instr (CALL (mk_uN 1))]) c /\
    programLoaded nestedProgram final /\
    CircuitConfig nestedProgram nestedStore 0 shape final (rowsFinalBoundary (mkBoundary 4 8 8 7 0 0 0 0 0 0) rows) c.
Proof.
  intros rows final Hprime Hrun Hfragment.
  exact (run_refinement nestedProgram nestedStore 0 _ rows nestedMemory final _ _ Hprime
    (load_program_loaded _ _) Hrun Hfragment nested_initial_config).
Qed.
