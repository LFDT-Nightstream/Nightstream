(* Arithmetic instruction simulation and decoding of the supported opcode set. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog OpcodeCatalog
  Wasm2 WasmIntegers
  StackBindings StraightLineSpec StackSimulation LocalSimulation LocalBindings.
Import ListNotations.
Open Scope Z_scope.

Lemma binary_frame_refinement : forall z initial final row c0 op rest chosen,
  Range.fieldPrime -> FullRelation.satFull row ->
  Replay initial (eventsForRow MemoryCatalog.memories row) final ->
  binaryArithmeticSelector chosen -> row chosen = 1 ->
  Step_pure
    [i32Constant (row (List.nth 0%nat COL_STACK_READ_VALUE_LO 0%nat));
     i32Constant (row (List.nth 1%nat COL_STACK_READ_VALUE_LO 0%nat)); admininstr_instr (BINOP I32 op)]
    [i32Constant (row COL_STACK_WRITE0_VALUE_LO)] ->
  (programTyped (length (stateLocals z)) (stackHeight (beforeBoundary row)) (BINOP I32 op :: rest) /\
    represents z initial (beforeBoundary row) (BINOP I32 op :: rest) c0) ->
  exists c1, Step c0 c1 /\ (programTyped (length (stateLocals z)) (stackHeight (afterBoundary row)) rest /\
    represents z final (afterBoundary row) rest c1).
Proof.
  intros z initial final row c0 op rest chosen Hprime Hfull Hreplay Hchoice Hchosen Hpure
    Hrep. apply represented_values in Hrep.
  destruct Hrep as [values [Hstack [Htyped [Hconfig [Hwf Hlocals]]]]].
  apply program_typed_binary in Htyped.
  assert (HlocalChoice : straightLineSelector chosen).
  { destruct Hchoice as [-> | ->]; cbn; auto. }
  assert (Hheight : (2 <= length values)%nat).
  { destruct (length values) as [ | [ | n]]; cbn [programTyped instructionStackHeight] in Htyped; tauto || lia. }
  destruct (stack_has_two_values values Hheight) as [prefix [a [b Hvalues]]].
  subst values c0.
  assert (Hdepth : 2 <= row COL_SP_BEFORE).
  { destruct Hstack as [Hbase [Hsp _]].
    cbn [beforeBoundary boundaryStackBase boundarySP] in Hbase, Hsp.
    rewrite length_app, Nat2Z.inj_add in Hsp. cbn [length] in Hsp. lia. }
  pose proof (full_binary_stack_shape row chosen Hprime Hfull Hchoice Hchosen Hdepth) as Hshape.
  pose proof (Range.full_write0_u32 row Hprime Hfull) as Hresult.
  change (0 <= row COL_STACK_WRITE0_VALUE_LO < i32Modulus) in Hresult.
  destruct (stack_matches_binary_step row initial final prefix a b Hshape Hreplay Hstack Hresult)
    as [Hread0 [Hread1 HstackAfter]].
  rewrite Hread0, Hread1 in Hpure.
  destruct (stack_config_binary_step z prefix a b (row COL_STACK_WRITE0_VALUE_LO)
    (BINOP I32 op) rest Hpure Hresult Hwf) as [Hstep HwfAfter].
  exists (stackConfig z (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]) rest).
  split; [exact Hstep | ].
  apply represented_values. exists (prefix ++ [row COL_STACK_WRITE0_VALUE_LO]). split; [exact HstackAfter | ].
  split.
  - rewrite length_app in Htyped |- *. cbn [length] in Htyped |- *.
    replace (length prefix + 2)%nat with (S (S (length prefix))) in Htyped by lia.
    cbn [programTyped instructionStackHeight] in Htyped.
    replace (length prefix + 1)%nat with (S (length prefix)) by lia. exact Htyped.
  - split; [reflexivity | ]. split; [exact HwfAfter | ].
    eapply (locals_match_preserved initial final (beforeBoundary row) (afterBoundary row) z); [ | | exact Hlocals].
    + exact (proj1 (full_straight_line_local_gates row chosen Hprime Hfull HlocalChoice Hchosen)).
    + apply no_local_writes_preserve with (events := eventsForRow MemoryCatalog.memories row);
         [exact Hreplay | ].
       rewrite (full_straight_line_local_writes row chosen Hprime Hfull HlocalChoice Hchosen).
       destruct Hchoice as [-> | ->]; reflexivity.
Qed.

Lemma encoded_opcode_cases : forall instruction code,
  encodedOpcode instruction = Some code ->
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 ADD) /\ code = CODE_I32_ADD) \/
  (instruction = BINOP I32 (mk_binop__0 Inn_I32 SUB) /\ code = CODE_I32_SUB) \/
  (exists x, instruction = LOCAL_GET x /\ code = CODE_LOCAL_GET) \/
  (exists x, instruction = LOCAL_SET x /\ code = CODE_LOCAL_SET).
Proof.
  intros instruction code Hcode.
  destruct instruction; try discriminate;
    repeat match goal with
    | x : numtype |- _ => destruct x; try discriminate
    | x : binop_ |- _ => destruct x; try discriminate
    | x : Inn |- _ => destruct x; try discriminate
    | x : binop_Inn |- _ => destruct x; try discriminate
    end; inversion Hcode; subst; eauto 8.
Qed.
