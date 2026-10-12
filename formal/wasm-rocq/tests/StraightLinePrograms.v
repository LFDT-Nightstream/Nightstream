From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 StraightLineSpec OpcodeCatalog MemoryTrace MemoryCatalog Locals.
Import ListNotations.
Open Scope Z_scope.

(* Start with an empty operand stack and two initialized i32 locals. *)
Definition localArithmetic :=
  [LOCAL_GET (mk_uN 0); LOCAL_GET (mk_uN 1);
   BINOP I32 (mk_binop__0 Inn_I32 SUB);
   LOCAL_SET (mk_uN 0); LOCAL_GET (mk_uN 0)].

Example local_arithmetic_supported : supportedProgram localArithmetic.
Proof.
  unfold supportedProgram, localArithmetic. repeat constructor;
    eexists; reflexivity.
Qed.

Example local_arithmetic_no_initial_stack : programTyped 2 0 localArithmetic.
Proof. vm_compute. repeat constructor. Qed.

(* Concrete ROM contents, locals [20; 3] at frame base 7, and an empty stack.
   This checks that loading and representation can hold together; it does not
   construct a circuit witness or prove a validCircuitRun. *)
Definition localArithmeticMemory : MemoryState := fun memory address =>
  match address with
  | [pc] =>
      if pc <? 0 then None else
      if Nat.eqb memory programOpcodeId then
        nth_error [CODE_LOCAL_GET; CODE_LOCAL_GET; CODE_I32_SUB; CODE_LOCAL_SET; CODE_LOCAL_GET]
          (Z.to_nat pc)
      else if Nat.eqb memory pcEdgeKindId then
        nth_error [0; 0; 0; 0; 0] (Z.to_nat pc)
      else if Nat.eqb memory programLocalIndexId then
        List.nth (Z.to_nat pc) [Some 0; Some 1; None; Some 0; Some 0] None
      else None
  | [base; index] =>
      if Nat.eqb memory localLoId then
        match base, index with 7, 0 => Some 20 | 7, 1 => Some 3 | _, _ => None end
      else if Nat.eqb memory pcRomId then
        if (base <? 0) then None else
        if index =? 0 then nth_error [1; 2; 3; 4; 5] (Z.to_nat base) else None
      else None
  | _ => None
  end.

Definition localArithmeticState : state :=
  mk_state (MKstore [] [] [] [] [] [])
    (MKframe [i32Value 20; i32Value 3] (MKmoduleinst [] [] [] [] [] [] [] [])).

Example local_arithmetic_initial_state :
  loadedProgram localArithmetic localArithmeticMemory /\
  represents localArithmetic localArithmeticState localArithmeticMemory
    (mkBoundary 0 4 4 7) 0 (stackConfig localArithmeticState [] localArithmetic).
Proof.
  split.
  - unfold loadedProgram. repeat split; intros pc;
      assert (Hnonnegative : (Z.of_nat pc <? 0) = false) by
        (apply Z.ltb_ge; lia).
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
    + intros choice. unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; cbn;
        destruct (choice =? 0); reflexivity.
    + unfold localArithmeticMemory. rewrite Hnonnegative, Nat2Z.id.
      destruct pc as [ | [ | [ | [ | [ | [ | pc]]]]]]; reflexivity.
  - unfold represents. split; [reflexivity | ]. split; [cbn; lia | ].
    exists []. split.
    + unfold stackMatches. cbn. repeat split; try lia; try constructor.
      intros [ | index] value H; discriminate.
    + split; [exact local_arithmetic_no_initial_stack | ].
      split; [reflexivity | ]. split.
      * unfold stackConfig, localArithmeticState, localArithmetic, i32Value.
        cbn. repeat constructor.
      * intros [ | [ | index]] v H; cbn [stateLocals localArithmeticState nth_error] in H.
        -- inversion H; subst. exists 20. vm_compute. repeat split; congruence.
        -- inversion H; subst. exists 3. vm_compute. repeat split; congruence.
        -- destruct index; discriminate.
Qed.

(* Repeated writes to the same local are allowed; no readback-pair restriction. *)
Example overwrite_local_supported :
  programTyped 2 0
    [LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1);
     LOCAL_GET (mk_uN 0); LOCAL_SET (mk_uN 1); LOCAL_GET (mk_uN 1)].
Proof. vm_compute. repeat constructor. Qed.

Example local_set_underflow_rejected : ~ programTyped 1 0 [LOCAL_SET (mk_uN 0)].
Proof. vm_compute. tauto. Qed.

Example nonexistent_local_rejected : ~ programTyped 1 0 [LOCAL_GET (mk_uN 1)].
Proof. vm_compute. lia. Qed.

Example local_immediates_distinguished :
  programOpcodeAt localArithmetic 0 = programOpcodeAt localArithmetic 1 /\
  programLocalIndexAt localArithmetic 0 = Some 0 /\
  programLocalIndexAt localArithmetic 1 = Some 1 /\
  programLocalIndexAt localArithmetic 2 = None.
Proof. vm_compute. repeat constructor. Qed.

Example unsupported_instruction_rejected : ~ supportedProgram [NOP].
Proof.
  intros H. inversion H; subst. match goal with Hcode : exists _, _ |- _ =>
    destruct Hcode as [code Hcode]; discriminate Hcode end.
Qed.
