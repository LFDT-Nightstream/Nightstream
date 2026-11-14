(* The mixed ordinary-instruction dispatcher on exported circuit rows. The ROM
   inputs come from the fixture; whole-program encoding is tested separately. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers WasmValues StraightLineSpec SpecTecCalls CallerSpec CallSpec
  IntegerMemory IntegerDecode IntegerStack IntegerInstructions StackTypes
  tests.WitnessCheck tests.IntegerStack.
Import ListNotations.
Open Scope Z_scope.

Definition emptyState := mk_state (MKstore [] [] [] [] [] []) (MKframe [] (MKmoduleinst [] [] [] [] [] [] [] [])).
Definition emptyCallees := calleeEnvironment (fun _ => None) [] [] True.

Definition addMemory : MemoryState := addInitial.
Definition addMemoryFinal := replayed addMemory [addRow].

Lemma add_memory_checks : validRunb addMemory [addRow] = true.
Proof. vm_compute. reflexivity. Qed.

Lemma add_memory_stack : valueStackMatches addMemory (beforeBoundary addRow) [wide; i32Value 7; i32Value 9].
Proof.
  destruct add_initial_stack as [Hbase [Hsp Hcells]]. split; [exact Hbase | ]. split; [exact Hsp | ].
  intros index value Hnth. eapply cell_matches_stack; [ | exact (Hcells _ _ Hnth)]. intros address. reflexivity.
Qed.

Example add_with_wide_prefix_simulates : fieldPrime ->
  exists z' values', IntegerResult [] addMemoryFinal addRow emptyState [wide; i32Value 7; i32Value 9]
    (BINOP I32 (mk_binop__0 Inn_I32 ADD)) [valtype_I64; valtype_I32] z' values'.
Proof.
  intros Hprime. destruct (validRunb_sound _ _ add_memory_checks) as [Hfull [_ Hreplay]].
  inversion Hfull as [ | ? ? Hrow Htail]; subst.
  change (Replay addMemory (eventsForRow memories addRow ++ []) addMemoryFinal) in Hreplay.
  rewrite app_nil_r in Hreplay.
  assert (Himage : instructionImage addMemory (addRow COL_PC_BEFORE) (BINOP I32 (mk_binop__0 Inn_I32 ADD))).
  { repeat split; reflexivity. }
  destruct (integer_instruction [] [wide; i32Value 7; i32Value 9] _ emptyState addMemory addMemoryFinal addRow
    emptyCallees [valtype_I64; valtype_I32] Hprime Hrow Logic.eq_refl Hreplay Himage
    ltac:(repeat constructor) ltac:(repeat constructor) Logic.eq_refl add_memory_stack
    ltac:(intros index value H; destruct index; discriminate) ltac:(vm_compute; discriminate))
    as [z' [values' [Hresult _]]]. eauto.
Qed.

Definition localState := mk_state (MKstore [] [] [] [] [] [])
  (MKframe [i64Value 0] (MKmoduleinst [] [] [] [] [] [] [] [])).

Lemma initial_wide_local : valueLocalsMatch localInitial 0 (stateLocals localState).
Proof.
  intros [ | index] value H; cbn [stateLocals localState LOCALS nth_error] in H;
    [injection H as <- | destruct index; discriminate].
  change (CellMatches localInitial 0 (i64Value (joinLimbs 0 0))).
  constructor; try (unfold i32Modulus; lia); vm_compute; reflexivity.
Qed.

Example set_wide_local_simulates : fieldPrime ->
  exists middle,
    Step (valueStackConfig localState [wide; maxI64] [LOCAL_SET (mk_uN 0)])
      (valueStackConfig (with_local localState (mk_uN 0) maxI64) [wide] []) /\
    valueStackMatches middle (afterBoundary setRow) [wide] /\
    valueLocalsMatch middle 0 (stateLocals (with_local localState (mk_uN 0) maxI64)).
Proof.
  intros Hprime. destruct (validRunb_sound _ _ local_rows_check) as [Hfull [_ Hreplay]].
  inversion Hfull as [ | ? ? Hrow Htail]; subst.
  change (Replay localInitial (eventsForRow memories setRow ++ eventsForRow memories getRow ++ []) localFinal) in Hreplay.
  rewrite app_nil_r in Hreplay. destruct (replay_app_inv _ _ _ _ Hreplay) as [middle [Hset Hget]].
  assert (Hz : wf_state localState).
  { repeat constructor. }
  destruct (integer_local_set setRow localInitial middle [] [wide] localState (mk_uN 0) maxI64 Hprime Hrow
    Logic.eq_refl Hset Logic.eq_refl Hz ltac:(repeat constructor) ltac:(cbn; lia)
    local_initial_stack initial_wide_local ltac:(vm_compute; discriminate)) as [Hstep [_ [Hstack Hlocals]]].
  exists middle. auto.
Qed.

Example set_cannot_change_a_local_type :
  sourceTypes [valtype_I32] emptyCallees (LOCAL_SET (mk_uN 0)) [valtype_I64] = None.
Proof. reflexivity. Qed.
