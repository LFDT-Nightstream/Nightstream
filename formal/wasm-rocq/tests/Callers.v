From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns Range MemoryTrace MemoryCatalog OpcodeCatalog TraceRefinement
  Wasm2 WasmIntegers Locals StraightLineSpec StackSimulation CallSpec CallScope CallerSpec CallerRefinement SpecTecCalls tests.Calls.
Import ListNotations.
Open Scope Z_scope.

Definition readZeroCallee := mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 10.
Definition callerCallees : CalleeEnvironment := fun index =>
  if N.eqb index 0 then Some readZeroCallee else None.
Definition getCallAdd : list instr :=
  [LOCAL_GET (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].

Example get_call_add_typed : callerFacts callerCallees callState 0 getCallAdd.
Proof.
  pose proof (proj1 (proj2 (dirty_callee_is_a_valid_call_input 99))) as Hcall.
  cbn [getCallAdd callerFacts callerCallees N.eqb instructionStackHeight
    readZeroCallee guestParams guestResults guestModule guestTypeIndex guestLocals guestBody].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ].
  split; [vm_compute; repeat constructor | ].
  split; [exact Hcall | ]. split; [vm_compute; lia | ].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; exact I.
Qed.

Example add_after_call_needs_two_values :
  ~ callerFacts callerCallees callState 0 [CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Proof. intros [_ [_ [_ [_ [Hheight _]]]]]. exact Hheight. Qed.

Example unsupported_caller_instruction_rejected :
  ~ callerFacts callerCallees callState 0 [NOP].
Proof. intros [Hsupported _]. destruct (Forall_inv Hsupported) as [code Hcode]. discriminate. Qed.

Example missing_callee_rejected :
  ~ callerFacts callerCallees callState 0 [CALL (mk_uN 1)].
Proof. exact (fun H => H). Qed.

Example calls_share_one_environment_entry :
  callerFacts callerCallees callState 0
    [CALL (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].
Proof.
  pose proof (proj1 (proj2 (dirty_callee_is_a_valid_call_input 99))) as Hcall.
  split; [exact Hcall | ]. split; [vm_compute; lia | ].
  split; [exact Hcall | ]. split; [cbn; lia | ].
  split; [repeat constructor; eexists; reflexivity | ].
  split; [repeat constructor | ]. split; exact I.
Qed.
