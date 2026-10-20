(* A call from inside two nested blocks: the suspended caller's operands, labels
   and continuation wrap the callee's configuration. *)
From Stdlib Require Import List ZArith Lia.
From mathcomp Require Import ssrbool eqtype.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement Wasm2 WasmIntegers Locals
  SpecTecCalls StraightLineSpec CallerSpec CallerValidation ProgramSpec ProgramEncoding ProgramRefinement
  FunctionValidation CodeLayout ControlDecode Fragment tests.BlockPrograms.
Import ListNotations.
Open Scope Z_scope.

(* readZero at PCs 0..1. The caller: 2 and 3 enter the blocks, 4 calls, 5 stores the
   result, 6 and 7 end the blocks, 8 reads the local, 9 returns. *)
Definition innerBody := [CALL (mk_uN 0); LOCAL_SET (mk_uN 0)].
Definition callerBody := [BLOCK (_RESULT None) [BLOCK (_RESULT None) innerBody]; LOCAL_GET (mk_uN 0)].
Definition blockModule := MKmoduleinst [i32StackType 0 1] [0%N; 1%N] [] [] [] [] [] [].
Definition blockFunction (body : list instr) := i32Function blockModule (mk_uN 0) 0 1 1 body.
Definition blockProgram := [blockFunction [LOCAL_GET (mk_uN 0)]; blockFunction callerBody].
Definition blockStore := MKstore blockProgram [] [] [] [] [].
Definition blockGuest (address : N) (body : list instr) (entry : Z) :=
  mkGuestFunction address blockModule (mk_uN 0) 0 1 1 body entry.

Example block_call_layout :
  functionEntry blockProgram 1 = 2 /\
  encodeProgram blockProgram programOpcodeId [4] = Some CODE_CALL /\
  encodeProgram blockProgram pcRomId [4; 1] = Some 5 /\
  encodeProgram blockProgram programOpcodeId [6] = Some CODE_END /\
  encodeProgram blockProgram pcRomId [6; 0] = Some 7 /\
  encodeProgram blockProgram pcRomId [7; 0] = Some 8 /\
  encodeProgram blockProgram pcEdgeKindId [9] = Some 1.
Proof. repeat split; reflexivity. Qed.

Definition blockContext' :=
  MKcontext [i32StackType 0 1] [i32StackType 0 1; i32StackType 0 1] [] [] [] [] [] [] [] None.
Definition blockBodyContext := functionContext blockContext' (blockFunction []).

Example block_module_validates : Moduleinst_ok blockStore blockModule blockContext'.
Proof.
  apply mk_Moduleinst_ok; try reflexivity; try solve [constructor].
  all: repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := blockFunction [LOCAL_GET (mk_uN 0)]);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
  - eapply Externaddr_ok__func with (v_funcinst := blockFunction callerBody);
      try reflexivity; try (apply (introT eqP); reflexivity); repeat constructor.
Qed.

Lemma block_function_validates : forall body,
  Forall wf_instr body -> Instrs_ok blockBodyContext body (i32StackType 0 1) ->
  Funcinst_ok blockStore (blockFunction body) (i32StackType 0 1).
Proof.
  intros body Hw Hv. apply mk_Funcinst_ok with (C := blockContext').
  - repeat constructor.
  - exact block_module_validates.
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

Section Typing.
Local Open Scope nat_scope.
Example caller_body_validates : Instrs_ok blockBodyContext callerBody (i32StackType 0 1).
Proof.
  apply instrs_cons_ok with (mid := 0); [repeat constructor | repeat constructor | repeat constructor | | ].
  - apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    apply Instrs_ok__instr; [ | repeat constructor | repeat constructor].
    apply (block _ _ _ [] []); [apply Blocktype_ok__valtype; repeat constructor | | repeat constructor |
      repeat constructor | repeat constructor].
    apply (instrs_cons_ok _ _ _ 0 1 0); [repeat constructor | repeat constructor | repeat constructor | | ].
    + apply Instrs_ok__instr; [apply call | | ]; try reflexivity; try (apply (introT eqP); reflexivity);
        repeat constructor.
    + typed_instr 1 0 0.
  - typed_instr 0 1 0.
Qed.
End Typing.

Example block_program_validated : programValidated blockProgram blockStore.
Proof.
  split.
  - constructor; [ | constructor; [ | constructor]]; apply block_function_validates; try solve [repeat constructor];
      try exact caller_body_validates.
  - intros [ | [ | index]] g Hg; cbn in Hg; [injection Hg as <- .. | destruct index; discriminate].
    + repeat constructor.
    + constructor; [ | constructor; [constructor | constructor]].
      apply supported_block; [ | repeat split; reflexivity]. constructor; [ | constructor].
      apply supported_block; [ | repeat split; reflexivity]. constructor; [ | constructor; [constructor | constructor]].
      apply supported_call with (f := blockGuest 0 [LOCAL_GET (mk_uN 0)] 0); [reflexivity | cbn; lia | cbn; lia].
Qed.

(* The caller's local is 5 at frame base 7; readZero's is 0 at 8. The saved
   context returns to pc 5 with the caller's frame and stack bases. *)
Definition blockMemory := loadProgram blockProgram (fun memory address =>
  if Nat.eqb memory localLoId then match address with [7; 0] => Some 5 | [8; 0] => Some 0 | _ => None end
  else if Nat.eqb memory callStackReturnPcId then match address with [0] => Some 5 | _ => None end
  else if Nat.eqb memory callStackCallerFbpId then match address with [0] => Some 7 | _ => None end
  else if Nat.eqb memory callStackCallerSpBaseId then match address with [0] => Some 4 | _ => None end
  else None).

Definition callerFrame := MKframe [i32Value 5] blockModule.
Definition zeroFrame := MKframe [i32Value 0] blockModule.

(* readZero runs inside the caller's FRAME_/LABEL_, inside both block labels; the
   caller's local.set waits inside the inner label and its local.get after both. *)
Example block_suspended_call_decodes :
  option_map fst (decodeControl blockProgram blockStore 0
    (controlShape [codePosition 1 [0; 0]%nat 0] (siteCode (codePosition 0 [] 0)))
    blockMemory (mkBoundary 0 4 4 8 1 0 0 0 0)) =
  Some (mk_config (mk_state blockStore callerFrame)
    [LABEL_ 0%N [] [LABEL_ 0%N []
      [FRAME_ 1%N zeroFrame [LABEL_ 1%N [] [admininstr_instr (LOCAL_GET (mk_uN 0))]];
       admininstr_instr (LOCAL_SET (mk_uN 0))]];
     admininstr_instr (LOCAL_GET (mk_uN 0))]).
Proof. vm_compute. reflexivity. Qed.

(* A return pc off the CALL's successor does not decode. *)
Example block_wrong_return_rejected :
  decodeControl blockProgram blockStore 0
    (controlShape [codePosition 1 [0; 0]%nat 0] (siteCode (codePosition 0 [] 0)))
    (update blockMemory callStackReturnPcId [0] 8) (mkBoundary 0 4 4 8 1 0 0 0 0) = None.
Proof. vm_compute. reflexivity. Qed.

Example block_initial_config :
  CircuitConfig blockProgram blockStore 0 (controlShape [] (siteCode (codePosition 1 [] 0)))
    blockMemory (mkBoundary 2 4 4 7 0 0 0 0 0) (stackConfig (mk_state blockStore callerFrame) [] callerBody).
Proof.
  split; [reflexivity | ]. split; [repeat constructor | ]. split; [exact block_program_validated | ].
  exists True. split; [vm_compute; reflexivity | exact I].
Qed.

(* Any run of the caller's rows refines SpecTec, into both blocks, through the call and out. *)
Example block_program_refines : forall rows final,
  fieldPrime -> validCircuitRun blockMemory rows final ->
  runFragment (mkBoundary 2 4 4 7 0 0 0 0 0) 0 rows ->
  exists shape c, Steps (stackConfig (mk_state blockStore callerFrame) [] callerBody) c /\
    programLoaded blockProgram final /\
    CircuitConfig blockProgram blockStore 0 shape final (rowsFinalBoundary (mkBoundary 2 4 4 7 0 0 0 0 0) rows) c.
Proof.
  intros rows final Hprime Hrun Hfragment.
  exact (run_refinement blockProgram blockStore 0 _ rows blockMemory final _ _ Hprime
    (load_program_loaded _ _) Hrun Hfragment block_initial_config).
Qed.
