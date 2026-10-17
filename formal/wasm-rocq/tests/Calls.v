From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 WasmIntegers Locals R1CS Columns Range FullRelation MemoryTrace MemoryCatalog OpcodeCatalog
  StraightLineSpec SpecTecCalls CallSpec CallEntry TraceRefinement StraightLineValidation.
Import ListNotations.
Open Scope Z_scope.

Definition readZeroBody := [LOCAL_GET (mk_uN 0)].
Example read_zero_validates :
  Instrs_ok (i32TypingContext 1) readZeroBody (i32StackType 0 1).
Proof.
  apply Instrs_ok__instr.
  - apply local_get; [reflexivity | reflexivity | repeat constructor | repeat constructor].
  - repeat constructor.
  - repeat constructor.
Qed.

Definition callModule := MKmoduleinst [i32StackType 0 1] [0%N] [] [] [] [] [] [].
Definition readZeroFunction := i32Function callModule (mk_uN 0) 0 1 1 readZeroBody.
Definition callState := mk_state (MKstore [readZeroFunction] [] [] [] [] [])
  (MKframe [i32Value 5] callModule).

(* A stale callee cell is deliberately present. Loading does not require it to
   be zero: that must follow from the circuit's call and initialization rows. *)
Definition reusedFrameMemory (stale : Z) : MemoryState := fun memory address =>
  if Nat.eqb memory programOpcodeId then
    match address with [0] => Some CODE_CALL | [10] => Some CODE_LOCAL_GET | [11] => Some CODE_END | _ => None end
  else if Nat.eqb memory pcEdgeKindId then
    match address with [0] | [10] => Some 0 | [11] => Some 1 | _ => None end
  else if Nat.eqb memory pcRomId then
    match address with [0; 0] => Some 10 | [0; 1] => Some 1 | [10; choice] =>
      if choice =? 0 then Some 11 else None | _ => None end
  else if Nat.eqb memory callTargetId then
    match address with [0] => Some 1 | _ => None end
  else if Nat.eqb memory functionCallMetadataId then
    match address with [1] => Some 65792 | _ => None end
  else if Nat.eqb memory functionLocalCountId then
    match address with [1] | [2] => Some 1 | _ => None end
  else if Nat.eqb memory functionEntryId then
    match address with [1] => Some 10 | _ => None end
  else if Nat.eqb memory pcFunctionRefId then
    match address with [0] => Some 2 | [10] | [11] => Some 1 | _ => None end
  else if Nat.eqb memory programLocalIndexId then
    match address with [10] => Some 0 | _ => None end
  else if Nat.eqb memory localLoId then
    match address with [7; 0] => Some 5 | [8; 0] => Some stale | _ => None end
  else if Nat.eqb memory stackId then
    match address with [8] => Some 11 | _ => None end
  else None.

Example dirty_callee_is_a_valid_call_input : forall stale,
  loadedCall (reusedFrameMemory stale) 0 1 callState (mk_uN 0) 0%N callModule (mk_uN 0)
    0 1 1 readZeroBody 10 /\
  typedCall callState (mk_uN 0) callModule (mk_uN 0) 0 1 1 readZeroBody /\
  stackMatches (reusedFrameMemory stale) (mkBoundary 0 5 4 7 0 0 0 0 0) [11] /\
  localsMatch (reusedFrameMemory stale) (mkBoundary 0 5 4 7 0 0 0 0 0) callState /\
  reusedFrameMemory stale localLoId [8; 0] = Some stale.
Proof.
  intros stale. split.
  - unfold loadedCall, directCallImage, guestFunctionImage. split.
    + repeat split; reflexivity.
    + split.
      * exists 2. split; reflexivity.
      * split.
        -- cbn [loadedBody readZeroBody]. split; [ | exact I].
           unfold instructionImage. repeat split; try reflexivity.
        -- split; reflexivity.
  - split.
    + unfold typedCall, callState, readZeroFunction, i32Function, callModule, readZeroBody, i32Value.
      split; [repeat constructor | ]. split; [repeat constructor | ]. split; [repeat constructor | ].
      split; [repeat constructor; eexists; reflexivity | ].
      split; [exact read_zero_validates | split; lia].
    + split.
      * unfold stackMatches. cbn [boundaryStackBase boundarySP]. split; [lia | ].
        split; [reflexivity | ]. split; [constructor; [unfold i32Modulus; lia | constructor] | ].
        intros [ | n] value Hlookup; [inversion Hlookup; subst; reflexivity | destruct n; discriminate].
      * split.
        -- intros [ | n] value Hlookup; cbn [callState stateLocals LOCALS nth_error] in Hlookup.
           ++ inversion Hlookup; subst. exists 5. split; [reflexivity | ]. split; [unfold i32Modulus; lia | reflexivity].
           ++ destruct n; discriminate.
        -- reflexivity.
Qed.

Example wrong_result_arity_rejected :
  ~ typedCall callState (mk_uN 0) callModule (mk_uN 0) 0 0 1 readZeroBody.
Proof.
  intros [_ [_ [_ [Hsupported [Hvalidated _]]]]].
  pose proof (proj2 (i32_validation_facts _ _ _ _ Hvalidated Hsupported)) as Harity.
  discriminate Harity.
Qed.
