(* Connect the executable ROM encoding to the instruction/call loading lemmas. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import Wasm2 SpecTecCalls MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallerSpec CallerRefinement CallSpec ProgramSpec StraightLineInstructions.
Import ListNotations.
Open Scope Z_scope.

Lemma load_program_loaded : forall program memory, programLoaded program (loadProgram program memory).
Proof.
  intros program memory id Hin address. unfold loadProgram.
  assert (H : existsb (Nat.eqb id) programRomIds = true).
  { apply existsb_exists. exists id. split; [exact Hin | apply Nat.eqb_refl]. }
  now rewrite H.
Qed.

Lemma program_loaded_preserved : forall program initial final,
  sameRoms initial final -> programLoaded program initial -> programLoaded program final.
Proof.
  intros program initial final Hrom Hloaded id Hin address.
  rewrite (Hrom id); [apply Hloaded; exact Hin | ].
  unfold programRomIds in Hin. repeat destruct Hin as [<- | Hin]; try contradiction; reflexivity.
Qed.

Lemma i32_types_repeat : forall types,
  forallb i32Type types = true -> types = repeat valtype_I32 (length types).
Proof.
  induction types as [ | t types IH]; [reflexivity | ].
  cbn. rewrite andb_true_iff. intros [Ht Hts]. destruct t; try discriminate.
  cbn. f_equal. exact (IH Hts).
Qed.

Lemma i32_locals_repeat : forall locals,
  forallb i32Local locals = true -> locals = map LOCAL (repeat valtype_I32 (length locals)).
Proof.
  induction locals as [ | [t] locals IH]; [reflexivity | ].
  cbn. rewrite andb_true_iff. intros [Ht Hts]. destruct t; try discriminate.
  cbn. f_equal. exact (IH Hts).
Qed.

Lemma function_at_sound : forall program index f,
  functionAt program index = Some f ->
  nth_error program index = Some (i32Function (guestModule f) (guestTypeIndex f)
    (guestParams f) (guestResults f) (guestLocals f) (guestBody f)) /\
  guestAddress f = N.of_nat index /\ guestEntry f = functionEntry program index.
Proof.
  intros program index f. unfold functionAt.
  destruct (nth_error program index) as [[[ [params] [results]] module [typeIndex locals body]] | ] eqn:Hlookup;
    [ | discriminate].
  unfold decodeFunction. destruct (forallb i32Type params && forallb i32Type results && forallb i32Local locals)
    eqn:Hshape; [ | discriminate].
  intros H. injection H as <-. cbn [guestModule guestTypeIndex guestParams guestResults guestLocals guestBody guestAddress guestEntry].
  apply andb_true_iff in Hshape. destruct Hshape as [Htypes Hlocals].
  apply andb_true_iff in Htypes. destruct Htypes as [Hparams Hresults].
  split; [ | auto]. unfold i32Function.
  rewrite <- (i32_types_repeat _ Hparams), <- (i32_types_repeat _ Hresults), <- (i32_locals_repeat _ Hlocals).
  reflexivity.
Qed.

Lemma locate_function_pc : forall program index raw offset,
  nth_error program index = Some raw -> (offset < functionSize raw)%nat ->
  locatePC program (Z.to_nat (functionEntry program index + Z.of_nat offset)) = Some (index, offset).
Proof.
  induction program as [ | first program IH]; intros [ | index] raw offset Hlookup Hoffset;
    try discriminate.
  - injection Hlookup as <-. unfold functionEntry. cbn [firstn fold_right].
    rewrite Z.add_0_l, Nat2Z.id. cbn [locatePC].
    assert (Hlt : (offset <? functionSize first)%nat = true) by (apply Nat.ltb_lt; exact Hoffset).
    now rewrite Hlt.
  - cbn [nth_error] in Hlookup. unfold functionEntry. cbn [firstn fold_right].
    rewrite Nat2Z.inj_add, <- Z.add_assoc, Z2Nat.inj_add by lia. rewrite Nat2Z.id.
    cbn [locatePC].
    assert (Hlt : (functionSize first + Z.to_nat
      (Z.of_nat (fold_right (fun f size => functionSize f + size)%nat 0%nat (firstn index program)) + Z.of_nat offset)
      <? functionSize first)%nat = false) by (apply Nat.ltb_ge; lia).
    rewrite Hlt. rewrite Nat.add_sub_swap by lia. rewrite Nat.sub_diag, Nat.add_0_l.
    change (option_map (fun location => (S (fst location), snd location))
      (locatePC program (Z.to_nat (functionEntry program index + Z.of_nat offset))) = Some (S index, offset)).
    rewrite (IH index raw offset Hlookup Hoffset). reflexivity.
Qed.

Lemma instruction_at_function_pc : forall program index f offset,
  functionAt program index = Some f -> (offset <= length (guestBody f))%nat ->
  instructionAtPC program (guestEntry f + Z.of_nat offset) =
    Some (i32Function (guestModule f) (guestTypeIndex f) (guestParams f) (guestResults f) (guestLocals f) (guestBody f),
      nth_error (guestBody f) offset).
Proof.
  intros program index f offset Hf Hoffset.
  destruct (function_at_sound _ _ _ Hf) as [Hlookup [_ Hentry]].
  unfold instructionAtPC. rewrite Hentry.
  assert (Hnonneg : (functionEntry program index + Z.of_nat offset <? 0) = false).
  { apply Z.ltb_ge. unfold functionEntry. lia. }
  rewrite Hnonneg, (locate_function_pc _ _ _ offset Hlookup ltac:(unfold functionSize; cbn [functionBody i32Function CODE]; lia)), Hlookup.
  reflexivity.
Qed.

Ltac program_cell H :=
  match goal with |- ?memory ?id ?address = _ =>
    rewrite (H id ltac:(unfold programRomIds; simpl; tauto))
  end.

Lemma encoded_frame_image : forall program memory index f offset z,
  programLoaded program memory -> functionAt program index = Some f ->
  (offset <= length (guestBody f))%nat -> frameMatches program f z ->
  callerFrameImage memory (guestEntry f + Z.of_nat offset) z.
Proof.
  intros program memory index f offset z Hloaded Hf Hoffset [_ [_ Hcount]].
  destruct (function_at_sound _ _ _ Hf) as [Hlookup [Haddress Hentry]].
  exists (Z.of_nat index + 1). split.
  - program_cell Hloaded. unfold encodeProgram. cbn [Nat.eqb pcRomId pcFunctionRefId].
    rewrite Hentry. assert (Hnonneg : (functionEntry program index + Z.of_nat offset <? 0) = false)
      by (apply Z.ltb_ge; unfold functionEntry; lia).
    rewrite Hnonneg, (locate_function_pc _ _ _ offset Hlookup ltac:(unfold functionSize; cbn [functionBody i32Function CODE]; lia)).
    cbn. f_equal. lia.
  - program_cell Hloaded. unfold encodeProgram. cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb].
    unfold functionRom.
    assert (Hpositive : (Z.of_nat index + 1 <=? 0) = false) by (apply Z.leb_gt; lia).
    rewrite Hpositive. replace (Z.of_nat index + 1 - 1) with (Z.of_nat index) by lia.
    rewrite Nat2Z.id, Hf. cbn [Nat.eqb functionEntryId functionLocalCountId]. now rewrite Hcount.
Qed.

Lemma encoded_guest_image : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  guestFunctionImage memory (guestAddress f) (guestEntry f) (guestParams f) (guestResults f) (guestLocals f).
Proof.
  intros program memory index f Hloaded Hf.
  destruct (function_at_sound _ _ _ Hf) as [Hlookup [Haddress Hentry]].
  unfold guestFunctionImage. rewrite Haddress.
  replace (Z.of_N (N.of_nat index)) with (Z.of_nat index) by lia.
  assert (Hpositive : (Z.of_nat index + 1 <=? 0) = false) by (apply Z.leb_gt; lia).
  repeat split; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb].
  all: try (unfold functionRom; rewrite Hpositive;
    replace (Z.of_nat index + 1 - 1) with (Z.of_nat index) by lia;
    rewrite Nat2Z.id, Hf; reflexivity).
  rewrite Hentry.
  assert (Hnonneg : (functionEntry program index <? 0) = false)
    by (apply Z.ltb_ge; unfold functionEntry; lia).
  rewrite Hnonneg.
  replace (functionEntry program index) with (functionEntry program index + Z.of_nat 0) by lia.
  rewrite (locate_function_pc _ _ _ 0 Hlookup ltac:(unfold functionSize; lia)).
  cbn. f_equal. lia.
Qed.

Lemma encoded_instruction_image : forall program memory index f offset instruction,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some instruction ->
  (exists code, encodedOpcode instruction = Some code) ->
  instructionImage memory (guestEntry f + Z.of_nat offset) instruction.
Proof.
  intros program memory index f offset instruction Hloaded Hf Hlookup [code Hcode].
  assert (Hoffset : (offset <= length (guestBody f))%nat).
  { assert (Hlt : (offset < length (guestBody f))%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. } lia. }
  pose proof (instruction_at_function_pc _ _ _ _ Hf Hoffset) as Hpc.
  rewrite Hlookup in Hpc.
  destruct (encoded_opcode_cases instruction code Hcode)
    as [[-> _] | [[-> _] | [[x [-> _]] | [x [-> _]]]]];
    unfold instructionImage; repeat split; try intros choice;
    try (program_cell Hloaded; unfold encodeProgram;
      cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
      unfold instructionRom; rewrite Hpc; reflexivity).
Qed.

Lemma encoded_body_loaded : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  supportedStraightLineProgram (guestBody f) -> loadedBody memory (guestEntry f) (guestBody f).
Proof.
  intros program memory index f Hloaded Hf Hsupported.
  assert (Hsuffix : forall code offset, skipn offset (guestBody f) = code ->
    loadedBody memory (guestEntry f + Z.of_nat offset) code).
  { induction code as [ | instruction rest IH]; intros offset Hcode; [exact I | ].
    assert (Hlookup : nth_error (guestBody f) offset = Some instruction).
    { pose proof (f_equal (fun xs => nth_error xs 0) Hcode) as H.
      cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
    split.
    - eapply encoded_instruction_image; [exact Hloaded | exact Hf | exact Hlookup | ].
      apply (proj1 (Forall_forall _ _) Hsupported). eapply nth_error_In; exact Hlookup.
    - replace (guestEntry f + Z.of_nat offset + 1) with (guestEntry f + Z.of_nat (S offset)) by lia.
      apply IH. rewrite (StackSimulation.skipn_at_instruction _ _ _ Hlookup) in Hcode. now injection Hcode. }
  replace (guestEntry f) with (guestEntry f + Z.of_nat 0) by lia. apply Hsuffix. reflexivity.
Qed.

Lemma encoded_function_end : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  memory programOpcodeId [guestEntry f + Z.of_nat (length (guestBody f))] = Some CODE_END /\
  memory pcEdgeKindId [guestEntry f + Z.of_nat (length (guestBody f))] = Some 1.
Proof.
  intros program memory index f Hloaded Hf.
  pose proof (instruction_at_function_pc _ _ _ (length (guestBody f)) Hf ltac:(lia)) as Hpc.
  assert (Hnone : nth_error (guestBody f) (length (guestBody f)) = None) by (apply nth_error_None; lia).
  rewrite Hnone in Hpc. split; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId pcFunctionRefId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; reflexivity.
Qed.

Lemma encoded_call_loaded : forall program memory index f offset z x callee,
  programLoaded program memory -> functionAt program index = Some f ->
  nth_error (guestBody f) offset = Some (CALL x) -> frameMatches program f z ->
  programCallees program (guestModule f) (x :> N) = Some callee ->
  supportedStraightLineProgram (guestBody callee) ->
  loadedCall memory (guestEntry f + Z.of_nat offset) (guestEntry f + Z.of_nat offset + 1) z x
    (guestAddress callee) (guestModule callee) (guestTypeIndex callee)
    (guestParams callee) (guestResults callee) (guestLocals callee) (guestBody callee) (guestEntry callee).
Proof.
  intros program memory index f offset z x callee Hloaded Hf Hlookup Hframe Hcallee Hsupported.
  assert (Hoffset : (offset <= length (guestBody f))%nat).
  { assert (Hlt : (offset < length (guestBody f))%nat).
    { apply nth_error_Some. rewrite Hlookup. discriminate. } lia. }
  pose proof (instruction_at_function_pc _ _ _ offset Hf Hoffset) as Hpc. rewrite Hlookup in Hpc.
  unfold programCallees in Hcallee.
  destruct (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) as [address | ] eqn:Haddress; [ | discriminate].
  destruct (function_at_sound _ _ _ Hcallee) as [Hinstance [HcalleeAddress Hentry]].
  replace (N.of_nat (N.to_nat address)) with address in HcalleeAddress by lia.
  split.
  - unfold directCallImage. split.
    + rewrite (proj1 (proj2 Hframe)), HcalleeAddress. exact Haddress.
    + split.
      * rewrite (proj1 Hframe), HcalleeAddress. exact Hinstance.
      * split.
        -- program_cell Hloaded. unfold encodeProgram. cbn.
           unfold instructionRom. rewrite Hpc. reflexivity.
        -- split.
           ++ program_cell Hloaded. unfold encodeProgram. cbn.
              unfold instructionRom. rewrite Hpc. cbn [Nat.eqb pcRomId programOpcodeId pcEdgeKindId programLocalIndexId callTargetId].
              change (option_map (fun address => Z.of_N address + 1)
                (nth_error (FUNCS (guestModule f)) (N.to_nat (x :> N))) = Some (Z.of_N (guestAddress callee) + 1)).
              rewrite Haddress, HcalleeAddress. reflexivity.
           ++ split.
              ** program_cell Hloaded. unfold encodeProgram. cbn.
                 unfold instructionRom. rewrite Hpc. reflexivity.
              ** eapply encoded_guest_image; eassumption.
  - split; [eapply encoded_frame_image; eassumption | ]. split.
    + eapply encoded_body_loaded; eassumption.
    + eapply encoded_function_end; eassumption.
Qed.

Lemma program_loaded_caller : forall program memory index f offset code z height,
  programLoaded program memory -> functionAt program index = Some f ->
  codeAt f offset code -> frameMatches program f z ->
  callerFacts (programCallees program (guestModule f)) z height code ->
  loadedCaller (programCallees program (guestModule f)) z memory (guestEntry f + Z.of_nat offset) code.
Proof.
  intros program memory index f offset code. revert offset.
  induction code as [ | instruction rest IH]; intros offset z height Hloaded Hf [Hbound Hcode] Hframe Htyped;
    [exact I | ].
  assert (Hlookup : nth_error (guestBody f) offset = Some instruction).
  { pose proof (f_equal (fun xs => nth_error xs 0) Hcode) as H.
    cbn beta in H. rewrite nth_error_skipn, Nat.add_0_r in H. exact H. }
  assert (HtailCode : codeAt f (S offset) rest).
  { split.
    - assert (Hlt : (offset < length (guestBody f))%nat).
      { apply nth_error_Some. rewrite Hlookup. discriminate. } lia.
    - rewrite (StackSimulation.skipn_at_instruction _ _ _ Hlookup) in Hcode. now injection Hcode. }
  assert (Hcontinue : forall next,
    callerFacts (programCallees program (guestModule f)) z next rest ->
    loadedCaller (programCallees program (guestModule f)) z memory (guestEntry f + Z.of_nat offset + 1) rest).
  { intros next Hnext. replace (guestEntry f + Z.of_nat offset + 1)
      with (guestEntry f + Z.of_nat (S offset)) by lia.
    eapply IH; eassumption. }
  assert (Hordinary : forall i,
    supportedStraightLineProgram [i] /\ wf_instr i /\
      programTyped (length (stateLocals z)) height [i] /\
      (match instructionStackHeight i height with
       | Some next => callerFacts (programCallees program (guestModule f)) z next rest | None => False end) ->
    nth_error (guestBody f) offset = Some i ->
    instructionImage memory (guestEntry f + Z.of_nat offset) i /\
    loadedCaller (programCallees program (guestModule f)) z memory (guestEntry f + Z.of_nat offset + 1) rest).
  { intros i [Hs [_ [_ Hnext]]] Hi. split.
    - eapply encoded_instruction_image; [exact Hloaded | exact Hf | exact Hi | exact (Forall_inv Hs)].
    - destruct (instructionStackHeight i height) as [next | ]; [eapply Hcontinue; exact Hnext | contradiction]. }
  destruct instruction; cbn [callerFacts loadedCaller] in *;
    try (eapply Hordinary; eassumption).
  match type of Htyped with context [programCallees _ _ ?x] =>
    destruct (programCallees program (guestModule f) x) as [callee | ] eqn:Hcallee end; [ | contradiction].
  destruct Htyped as [[_ [_ [_ [Hsupported _]]]] [_ Hnext]]. split.
  - eapply encoded_call_loaded; eassumption.
  - eapply Hcontinue; exact Hnext.
Qed.
