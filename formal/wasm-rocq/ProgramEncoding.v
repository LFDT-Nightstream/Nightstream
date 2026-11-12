(* Connect the executable ROM encoding to the instruction/call loading lemmas. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import Wasm2 SpecTecCalls MemoryTrace MemoryCatalog OpcodeCatalog StraightLineSpec
  CallerSpec CallerRefinement CallSpec ProgramSpec StraightLineInstructions CodeLayout.
Import ListNotations.
Open Scope Z_scope.

(* A SpecTec state for a frame of f: the program's functions, f's module, and its locals. *)
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

Lemma function_at_typed : forall program index f,
  functionAt program index = Some f ->
  nth_error program index = Some (typedFunction (guestModule f) (guestTypeIndex f)
    (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f)) /\
  guestAddress f = N.of_nat index /\ guestEntry f = functionEntry program index.
Proof.
  intros program index f. unfold functionAt.
  destruct (nth_error program index) as [[[[params] [results]] module [ti locals body]] | ] eqn:Hlookup;
    [ | discriminate].
  unfold decodeFunction. destruct (forallb integerType params && forallb integerType results && forallb integerLocal locals);
    [ | discriminate].
  intros H. injection H as <-. cbn [guestModule guestTypeIndex guestParamTypes guestResultTypes
    guestLocalTypes guestBody guestAddress guestEntry].
  split; [ | auto]. unfold typedFunction. rewrite map_map.
  assert (Hlocals : map (fun l => LOCAL (match l with LOCAL t => t end)) locals = locals).
  { clear Hlookup. induction locals as [ | [t] rest IH]; [reflexivity | cbn; f_equal; exact IH]. }
  now rewrite Hlocals.
Qed.

Lemma function_integer_types : forall program index f,
  functionAt program index = Some f ->
  Forall (fun t => t = valtype_I32 \/ t = valtype_I64) (guestParamTypes f) /\
  Forall (fun t => t = valtype_I32 \/ t = valtype_I64) (guestResultTypes f) /\
  Forall (fun t => t = valtype_I32 \/ t = valtype_I64) (guestLocalTypes f).
Proof.
  intros program index f. unfold functionAt.
  destruct (nth_error program index) as [[[[params] [results]] module [ti locals body]] | ]; [ | discriminate].
  unfold decodeFunction. destruct (forallb integerType params && forallb integerType results && forallb integerLocal locals)
    eqn:Hshape; [ | discriminate].
  intros H. injection H as <-. cbn [guestParamTypes guestResultTypes guestLocalTypes].
  apply andb_true_iff in Hshape as [Htypes Hl]. apply andb_true_iff in Htypes as [Hp Hr].
  assert (Htypes : forall ts, forallb integerType ts = true -> Forall (fun t => t = valtype_I32 \/ t = valtype_I64) ts).
  { intros ts Hts. apply Forall_forall. intros t Ht. apply (proj1 (forallb_forall _ _) Hts) in Ht.
    destruct t; cbn in Ht; intuition discriminate. }
  split; [apply Htypes; exact Hp | ]. split; [apply Htypes; exact Hr | ].
  apply Forall_map, Forall_forall. intros [t] Ht.
  apply (proj1 (forallb_forall _ _) Hl) in Ht. destruct t; cbn in Ht; intuition discriminate.
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
  functionAt program index = Some f -> (offset <= codeSize (guestBody f))%nat ->
  instructionAtPC program (guestEntry f + Z.of_nat offset) =
    Some (typedFunction (guestModule f) (guestTypeIndex f) (guestParamTypes f) (guestResultTypes f) (guestLocalTypes f) (guestBody f),
      nth_error (flattenCode (guestBody f)) offset).
Proof.
  intros program index f offset Hf Hoffset.
  destruct (function_at_typed _ _ _ Hf) as [Hlookup [_ Hentry]].
  unfold instructionAtPC. rewrite Hentry.
  assert (Hnonneg : (functionEntry program index + Z.of_nat offset <? 0) = false).
  { apply Z.ltb_ge. unfold functionEntry. lia. }
  rewrite Hnonneg, (locate_function_pc _ _ _ offset Hlookup ltac:(unfold functionSize; cbn [functionBody typedFunction CODE]; lia)), Hlookup.
  reflexivity.
Qed.

Ltac program_cell H :=
  match goal with |- ?memory ?id ?address = _ =>
    rewrite (H id ltac:(unfold programRomIds; simpl; tauto)); unfold romValue
  end.

Lemma encoded_guest_image : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  guestFunctionImage memory (guestAddress f) (guestEntry f) (guestParams f) (guestResults f) (guestLocals f).
Proof.
  intros program memory index f Hloaded Hf.
  destruct (function_at_typed _ _ _ Hf) as [Hlookup [Haddress Hentry]].
  unfold guestFunctionImage. rewrite Haddress.
  replace (Z.of_N (N.of_nat index)) with (Z.of_nat index) by lia.
  assert (Hpositive : (Z.of_nat index + 1 <=? 0) = false) by (apply Z.leb_gt; lia).
  repeat split; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold functionRom; rewrite Hpositive;
    replace (Z.of_nat index + 1 - 1) with (Z.of_nat index) by lia;
    rewrite Nat2Z.id, Hf; reflexivity.
Qed.

Lemma encoded_function_end : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  memory programOpcodeId [guestEntry f + Z.of_nat (codeSize (guestBody f))] = Some CODE_END /\
  memory pcEdgeKindId [guestEntry f + Z.of_nat (codeSize (guestBody f))] = Some 1.
Proof.
  intros program memory index f Hloaded Hf.
  pose proof (instruction_at_function_pc _ _ _ (codeSize (guestBody f)) Hf ltac:(lia)) as Hpc.
  assert (Hnone : nth_error (flattenCode (guestBody f)) (codeSize (guestBody f)) = None) by (apply nth_error_None; rewrite flatten_code_length; lia).
  rewrite Hnone in Hpc. split; program_cell Hloaded; unfold encodeProgram;
    cbn [Nat.eqb pcRomId functionEntryId functionLocalCountId functionCallMetadataId orb];
    unfold instructionRom; rewrite Hpc; reflexivity.
Qed.
Lemma slide_at_function_pc : forall program memory index f offset,
  functionAt program index = Some f -> (offset <= codeSize (guestBody f))%nat ->
  slideRom program memory (guestEntry f + Z.of_nat offset) = functionSlide program memory f offset.
Proof.
  intros program memory index f offset Hf Hoffset.
  destruct (function_at_typed _ _ _ Hf) as [Hlookup [_ Hentry]].
  unfold slideRom. rewrite Hentry.
  assert (Hnonneg : (functionEntry program index + Z.of_nat offset <? 0) = false).
  { apply Z.ltb_ge. unfold functionEntry. lia. }
  rewrite Hnonneg, (locate_function_pc _ _ _ offset Hlookup ltac:(unfold functionSize; cbn [functionBody typedFunction CODE]; lia)), Hf.
  reflexivity.
Qed.

(* A function's END slides its results down over its locals. *)
Lemma encoded_end_slide : forall program memory index f,
  programLoaded program memory -> functionAt program index = Some f ->
  let pc := guestEntry f + Z.of_nat (codeSize (guestBody f)) in
  let frame := Z.of_nat (guestParams f + guestLocals f) in
  memory frameSlideRemainingId [pc] = Some (if frame =? 0 then 0 else Z.of_nat (guestResults f)) /\
  memory frameSlideDistanceId [pc] = Some frame.
Proof.
  intros program memory index f Hloaded Hf pc frame.
  assert (Hnone : nth_error (flattenCode (guestBody f)) (codeSize (guestBody f)) = None)
    by (apply nth_error_None; rewrite flatten_code_length; lia).
  split; program_cell Hloaded; unfold encodeProgram; cbn [Nat.eqb pcRomId frameSlideRemainingId frameSlideDistanceId orb];
    unfold pc; rewrite (slide_at_function_pc _ _ _ _ (codeSize (guestBody f)) Hf ltac:(lia));
    unfold functionSlide; rewrite Hnone;
    unfold slideWord, frame; cbn [Nat.eqb frameSlideRemainingId frameSlideDistanceId];
    destruct (guestParams f + guestLocals f)%nat; reflexivity.
Qed.

Lemma pc_entry_function : forall program index f offset,
  functionAt program index = Some f -> (offset <= codeSize (guestBody f))%nat ->
  pcEntry program (guestEntry f + Z.of_nat offset) = guestEntry f.
Proof.
  intros program index f offset Hf Hoffset.
  destruct (function_at_typed _ _ _ Hf) as [Hlookup [_ Hentry]].
  unfold pcEntry. rewrite Hentry, (locate_function_pc _ _ _ offset Hlookup
    ltac:(unfold functionSize; cbn [functionBody typedFunction CODE]; lia)).
  reflexivity.
Qed.

Lemma type_position_found : forall t types, In t types -> nth_error types (typePosition t types) = Some t.
Proof.
  intros t types. induction types as [ | u rest IH]; [contradiction | ].
  intros Hin. cbn. destruct (functypeEqDec t u) as [-> | Hne]; [reflexivity | ].
  destruct Hin as [Heq | Hin]; [symmetry in Heq; contradiction | exact (IH Hin)].
Qed.

Lemma first_seen_keeps : forall types seen t, In t seen \/ In t types -> In t (firstSeen seen types).
Proof.
  induction types as [ | u rest IH]; intros seen t Hin; cbn.
  - destruct Hin as [Hin | []]; exact Hin.
  - apply IH. destruct Hin as [Hin | [<- | Hin]]; [ | | right; exact Hin].
    + left. destruct (existsb _ seen); [exact Hin | apply in_or_app; left; exact Hin].
    + left. destruct (existsb _ seen) eqn:Hseen.
      * apply existsb_exists in Hseen as [v [Hv Heq]].
        destruct (functypeEqDec u v) as [-> | ]; [exact Hv | discriminate].
      * apply in_or_app. right. left. reflexivity.
Qed.

(* Within a module, equal type ids are equal types. *)
Lemma type_id_injective : forall module t u,
  In t (TYPES module) -> In u (TYPES module) -> typeId module t = typeId module u -> t = u.
Proof.
  intros module t u Ht Hu Hid. unfold typeId in Hid.
  assert (Hpos : typePosition t (firstSeen [] (TYPES module)) = typePosition u (firstSeen [] (TYPES module))) by lia.
  pose proof (type_position_found t _ (first_seen_keeps _ [] t (or_intror Ht))) as Hfound.
  rewrite Hpos, (type_position_found u _ (first_seen_keeps _ [] u (or_intror Hu))) in Hfound.
  injection Hfound as ->. reflexivity.
Qed.

(* A function-type read of a nonzero id names a guest function of that type. *)
Lemma function_type_read : forall program memory reference id,
  programLoaded program memory -> memory functionTypeId [reference] = Some id -> 1 <= id ->
  exists g, 1 <= reference /\ functionAt program (Z.to_nat (reference - 1)) = Some g /\
    id = typeId (guestModule g) (guestType g).
Proof.
  intros program memory reference id Hloaded Hread Hid.
  rewrite (Hloaded functionTypeId ltac:(unfold programRomIds; simpl; tauto)) in Hread.
  injection Hread as Hread. unfold romValue, encodeProgram in Hread. cbn in Hread.
  unfold functionRom in Hread.
  destruct (reference <=? 0) eqn:Hpositive; [subst id; lia | ].
  destruct (functionAt program (Z.to_nat (reference - 1))) as [g | ] eqn:Hg; [ | subst id; lia].
  cbn in Hread. exists g. split; [apply Z.leb_gt in Hpositive; lia | ]. split; [reflexivity | ]. lia.
Qed.

Lemma type_id_positive : forall module t, 1 <= typeId module t.
Proof. intros. unfold typeId. lia. Qed.

Lemma function_type_rom : forall program memory index g,
  programLoaded program memory -> functionAt program index = Some g ->
  memory functionTypeId [Z.of_nat index + 1] = Some (typeId (guestModule g) (guestType g)).
Proof.
  intros program memory index g Hloaded Hg.
  rewrite (Hloaded functionTypeId ltac:(unfold programRomIds; simpl; tauto)).
  unfold romValue, encodeProgram. cbn. unfold functionRom.
  replace (Z.of_nat index + 1 <=? 0) with false by (symmetry; apply Z.leb_gt; lia).
  replace (Z.to_nat (Z.of_nat index + 1 - 1)) with index by lia. rewrite Hg. reflexivity.
Qed.
